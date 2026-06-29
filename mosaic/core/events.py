# mosaic/core/events.py
"""Mosaic 事件总线。

本模块实现了一个进程内的事件总线 :class:`EventBus`（单例），用于在管道
运行过程中发布与订阅各类事件，方便监控、调试与可观测性集成。

设计要点
--------
* :class:`EventBus` 是单例，全局唯一，通过 :func:`get_event_bus` 获取。
* 支持 ``on(event_type, callback)`` / ``off(event_type, callback)`` 订阅管理。
* 回调既可以是普通同步函数，也可以是 ``async`` 协程函数；同步回调立即执行，
  异步回调会被调度到事件循环（若当前没有运行中的事件循环则创建后台线程
  专属循环执行），保证事件发布本身始终非阻塞。
* **回调异常一律捕获并记录**，绝不向上抛出，从而不影响管道运行。
* 内置 :class:`LoggingListener`，用标准库 ``logging`` 输出事件日志。
* 事件类型为字符串常量，集中定义于 :class:`EventType`，便于静态检查。

事件类型
--------
见 :class:`EventType`，覆盖管道级、节点级、模型加载/卸载与进度事件。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "EventType",
    "MosaicEvent",
    "EventCallback",
    "EventBus",
    "get_event_bus",
    "LoggingListener",
]


# ---------------------------------------------------------------------------
# 事件类型常量
# ---------------------------------------------------------------------------
class EventType:
    """所有内置事件类型常量。

    每个常量是事件 ``type`` 字段的取值，用于 ``on``/``emit`` 时匹配。
    """

    #: 管道开始执行。payload: ``{"pipeline_name": str, "input_data": MosaicData}``
    PIPELINE_START = "pipeline_start"
    #: 管道正常完成。payload: ``{"pipeline_name": str, "output_data": MosaicData, "duration": float}``
    PIPELINE_COMPLETE = "pipeline_complete"
    #: 管道执行出错。payload: ``{"pipeline_name": str, "error": BaseException}``
    PIPELINE_ERROR = "pipeline_error"
    #: 节点开始执行。payload: ``{"node_name": str, "node_domain": str}``
    NODE_START = "node_start"
    #: 节点正常完成。payload: ``{"node_name": str, "duration": float, "output_summary": Any}``
    NODE_COMPLETE = "node_complete"
    #: 节点执行出错。payload: ``{"node_name": str, "error": BaseException}``
    NODE_ERROR = "node_error"
    #: 模型加载到设备。payload: ``{"node_name": str, "device": str, "memory_used": float}``
    MODEL_LOAD = "model_load"
    #: 模型卸载。payload: ``{"node_name": str, "memory_freed": float}``
    MODEL_UNLOAD = "model_unload"
    #: 进度更新。payload: ``{"pipeline_name": str, "current_step": int, "total_steps": int}``
    PROGRESS = "progress"

    #: 通配符，订阅所有事件类型。
    ALL = "*"

    @classmethod
    def all_types(cls) -> list[str]:
        """返回所有具体事件类型常量（不含通配符）。"""
        return [
            cls.PIPELINE_START,
            cls.PIPELINE_COMPLETE,
            cls.PIPELINE_ERROR,
            cls.NODE_START,
            cls.NODE_COMPLETE,
            cls.NODE_ERROR,
            cls.MODEL_LOAD,
            cls.MODEL_UNLOAD,
            cls.PROGRESS,
        ]


# ---------------------------------------------------------------------------
# 事件对象
# ---------------------------------------------------------------------------
@dataclass
class MosaicEvent:
    """事件总线中传递的事件对象。

    Attributes
    ----------
    event_type:
        事件类型，取值见 :class:`EventType`。
    timestamp:
        事件创建时间戳（``time.time()``，秒）。
    payload:
        事件附带的任意数据字典。
    """

    event_type: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    payload: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MosaicEvent(event_type={self.event_type!r}, "
            f"timestamp={self.timestamp:.3f}, "
            f"payload_keys={list(self.payload.keys())})"
        )


#: 事件回调函数签名：同步 ``Callable[[MosaicEvent], None]`` 或异步
#: ``Callable[[MosaicEvent], Awaitable[None]]``。
EventCallback = Callable[[MosaicEvent], None] | Callable[[MosaicEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# EventBus — 事件总线（单例）
# ---------------------------------------------------------------------------
class EventBus:
    """进程内事件总线（单例）。

    使用 :meth:`on` 订阅事件，:meth:`emit` 发布事件。回调异常被捕获并记录，
    不会影响事件发布方或管道运行。

    通过 :func:`get_event_bus` 获取全局单例实例。本类也支持直接实例化以
    构造独立的事件总线（例如为测试隔离）。

    线程安全：订阅管理与事件派发均受内部锁保护。
    """

    _instance: "EventBus" | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        """单例构造：首次实例化后，后续 ``EventBus()`` 返回同一对象。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        # 单例避免重复初始化
        if getattr(self, "_initialized", False):
            return
        self._initialized: bool = True
        # event_type -> 回调列表
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._lock = threading.RLock()
        # 异步回调专用事件循环（后台线程）
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._async_lock = threading.Lock()
        # 内部日志器
        self._logger = logging.getLogger("mosaic.events")

    # -- 订阅管理 ----------------------------------------------------------
    def on(self, event_type: str, callback: EventCallback) -> EventCallback:
        """订阅事件。

        Parameters
        ----------
        event_type:
            事件类型常量（见 :class:`EventType`），或 ``EventType.ALL``
            （``"*"``）订阅所有事件。
        callback:
            同步或异步回调函数。

        Returns
        -------
        EventCallback
            返回传入的回调，便于装饰器用法与后续 :meth:`off` 取消。
        """
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)!r}.")
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
        return callback

    def off(self, event_type: str, callback: EventCallback) -> bool:
        """取消订阅。

        Parameters
        ----------
        event_type:
            订阅时使用的事件类型。
        callback:
            要移除的回调函数（须与 :meth:`on` 注册时为同一对象）。

        Returns
        -------
        bool
            是否成功移除一个回调。
        """
        with self._lock:
            callbacks = self._subscribers.get(event_type)
            if not callbacks:
                return False
            try:
                callbacks.remove(callback)
            except ValueError:
                return False
            if not callbacks:
                self._subscribers.pop(event_type, None)
            return True

    def clear(self, event_type: str | None = None) -> int:
        """清除订阅。

        Parameters
        ----------
        event_type:
            指定类型则只清除该类型；``None`` 清除全部订阅。

        Returns
        -------
        int
            被清除的回调总数。
        """
        with self._lock:
            if event_type is None:
                count = sum(len(cbs) for cbs in self._subscribers.values())
                self._subscribers.clear()
                return count
            cbs = self._subscribers.pop(event_type, [])
            return len(cbs)

    def subscriber_count(self, event_type: str | None = None) -> int:
        """返回订阅者数量。``None`` 表示全部类型合计。"""
        with self._lock:
            if event_type is None:
                return sum(len(cbs) for cbs in self._subscribers.values())
            return len(self._subscribers.get(event_type, []))

    # -- 事件发布 ----------------------------------------------------------
    def emit(self, event_type: str, **payload: Any) -> MosaicEvent:
        """发布一个事件。

        将事件派发给所有匹配的订阅者（含通配符订阅者）。同步回调在当前
        线程立即执行；异步回调被调度到后台事件循环执行。**所有回调异常
        均被捕获并记录**，不会向上抛出。

        Parameters
        ----------
        event_type:
            事件类型常量。
        **payload:
            事件附带数据，作为 ``MosaicEvent.payload`` 传入。

        Returns
        -------
        MosaicEvent
            已发布的事件对象。
        """
        event = MosaicEvent(event_type=event_type, payload=payload)
        # 收集匹配回调（持锁快照）
        with self._lock:
            callbacks: list[EventCallback] = []
            callbacks.extend(self._subscribers.get(event_type, []))
            callbacks.extend(self._subscribers.get(EventType.ALL, []))

        for cb in callbacks:
            self._safe_invoke(cb, event)
        return event

    def _safe_invoke(self, callback: EventCallback, event: MosaicEvent) -> None:
        """安全调用回调：捕获异常、区分同步/异步。"""
        try:
            if inspect.iscoroutinefunction(callback):
                self._schedule_async(callback, event)
            else:
                callback(event)
        except Exception as exc:  # noqa: BLE001 - 故意宽泛
            self._logger.exception(
                "Event callback %r raised on event %r: %s",
                getattr(callback, "__name__", callback),
                event.event_type,
                exc,
            )

    # -- 异步回调支持 ------------------------------------------------------
    def _schedule_async(
        self,
        callback: Callable[[MosaicEvent], Awaitable[None]],
        event: MosaicEvent,
    ) -> None:
        """将异步回调调度到后台事件循环执行。"""
        loop = self._get_async_loop()
        coro = callback(event)
        # run_coroutine_threadsafe 线程安全地向其他线程的事件循环提交协程
        asyncio.run_coroutine_threadsafe(coro, loop)

    def _get_async_loop(self) -> asyncio.AbstractEventLoop:
        """获取（惰性创建）后台线程专属的事件循环。"""
        with self._async_lock:
            if self._async_loop is not None and not self._async_loop.is_closed():
                return self._async_loop
            # 创建新事件循环并运行在后台守护线程
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_async_loop,
                args=(loop,),
                name="mosaic-eventbus-async",
                daemon=True,
            )
            self._async_loop = loop
            self._async_thread = thread
            thread.start()
            return loop

    @staticmethod
    def _run_async_loop(loop: asyncio.AbstractEventLoop) -> None:
        """后台线程入口：持续运行事件循环。"""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        """关闭后台异步事件循环，释放资源。

        通常在程序退出时调用。已注册的同步订阅不受影响。
        """
        with self._async_lock:
            loop = self._async_loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            self._async_loop = None
            self._async_thread = None

    # -- 单例重置（主要供测试使用）----------------------------------------
    @classmethod
    def _reset_singleton(cls) -> None:
        """重置单例实例（仅供测试隔离使用）。"""
        with cls._instance_lock:
            inst = cls._instance
            if inst is not None:
                try:
                    inst.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            cls._instance = None

    def __repr__(self) -> str:
        with self._lock:
            total = sum(len(cbs) for cbs in self._subscribers.values())
        return f"<EventBus subscribers={total}>"


def get_event_bus() -> EventBus:
    """返回全局 :class:`EventBus` 单例。"""
    return EventBus()


# ---------------------------------------------------------------------------
# LoggingListener — 内置日志监听器
# ---------------------------------------------------------------------------
class LoggingListener:
    """内置事件日志监听器。

    将事件总线上的事件通过标准库 ``logging`` 输出，便于开发期调试。
    可通过 :meth:`attach` / :meth:`detach` 挂载/卸载到事件总线。

    Parameters
    ----------
    logger:
        使用的日志器，默认 ``logging.getLogger("mosaic.events")``。
    level:
        事件日志级别，默认 :data:`logging.INFO`。
    event_types:
        仅订阅这些事件类型；``None`` 表示订阅全部（``EventType.ALL``）。
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        event_types: list[str] | None = None,
    ) -> None:
        self._logger: logging.Logger = logger or logging.getLogger("mosaic.events")
        self._level: int = level
        self._event_types: list[str] = event_types or [EventType.ALL]
        self._bus: EventBus | None = None
        # 记录已注册的回调，便于 detach
        self._callbacks: dict[str, EventCallback] = {}

    def attach(self, bus: EventBus | None = None) -> "LoggingListener":
        """挂载到事件总线，开始监听并输出日志。

        Parameters
        ----------
        bus:
            目标事件总线，``None`` 表示使用全局单例。

        Returns
        -------
        LoggingListener
            ``self``，便于链式调用。
        """
        self._bus = bus or get_event_bus()
        for et in self._event_types:
            cb = self._make_callback(et)
            self._callbacks[et] = cb
            self._bus.on(et, cb)
        return self

    def detach(self) -> None:
        """从事件总线卸载，停止输出日志。"""
        if self._bus is None:
            return
        for et, cb in self._callbacks.items():
            self._bus.off(et, cb)
        self._callbacks.clear()
        self._bus = None

    def _make_callback(self, event_type: str) -> EventCallback:
        """为指定事件类型构造日志回调。"""

        def _log(event: MosaicEvent) -> None:
            msg = self._format_event(event)
            self._logger.log(self._level, msg)

        _log.__name__ = f"logging_listener[{event_type}]"
        return _log

    @staticmethod
    def _format_event(event: MosaicEvent) -> str:
        """格式化事件为日志字符串。"""
        payload_str = ", ".join(f"{k}={_truncate(v)!r}" for k, v in event.payload.items())
        return f"[{event.event_type}] {payload_str}"

    def __repr__(self) -> str:
        return (
            f"<LoggingListener level={logging.getLevelName(self._level)} "
            f"types={self._event_types} attached={self._bus is not None}>"
        )


def _truncate(value: Any, max_len: int = 60) -> Any:
    """截断过长的值，避免日志爆炸。"""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return value
