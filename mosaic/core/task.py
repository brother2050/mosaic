# mosaic/core/task.py
"""异步任务封装。

定义 :class:`AsyncTask`，封装管道在新线程中的异步执行、状态管理、
进度追踪与回调通知。

设计要点
--------
* 使用 Python 标准库 ``threading`` 实现（不引入 ``asyncio``，保持简单）。
* 状态流转：``pending`` → ``running`` → ``completed`` / ``failed`` / ``cancelled``。
* ``progress`` 通过 :class:`~mosaic.core.events.EventBus` 的 ``PROGRESS`` 事件更新。
* ``cancel()`` 设置标志位（``threading.Event``），不强制终止线程；
  节点可通过 ``context.shared["_cancel_event"]`` 协作式检查。
* ``wait()`` 使用 ``threading.Event`` 实现阻塞等待，支持超时。
* ``AsyncTask`` 全部属性线程安全（内部 ``threading.Lock`` 保护）。
* 多个 ``AsyncTask`` 可并行运行（不同管道或同一管道的不同输入）。

回调机制
--------
通过 ``on_complete`` / ``on_error`` / ``on_progress`` 注册回调函数：
- 完成回调在任务成功完成后被调用，参数为 ``PipelineResult``。
- 错误回调在任务失败或被取消时被调用，参数为 ``BaseException``。
- 进度回调在进度更新时被调用，参数为 ``(progress: float, node_name: str)``。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.result import PipelineResult

__all__ = ["AsyncTask", "TaskCancelledError", "TaskStatus"]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
class TaskStatus:
    """任务状态常量。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def all_statuses(cls) -> List[str]:
        """返回所有状态值。"""
        return [cls.PENDING, cls.RUNNING, cls.COMPLETED, cls.FAILED, cls.CANCELLED]

    @classmethod
    def terminal_statuses(cls) -> List[str]:
        """返回终态状态值（不会再变化）。"""
        return [cls.COMPLETED, cls.FAILED, cls.CANCELLED]


class TaskCancelledError(Exception):
    """任务被取消时抛出的异常。

    Attributes
    ----------
    task_id:
        被取消的任务 ID。
    """

    def __init__(self, task_id: str) -> None:
        self.task_id: str = task_id
        super().__init__(f"Task {task_id!r} was cancelled.")


# 回调函数类型
ProgressCallback = Callable[[float, str], None]
CompleteCallback = Callable[[PipelineResult], None]
ErrorCallback = Callable[[BaseException], None]


class AsyncTask:
    """异步管道执行任务。

    封装管道在新线程中的异步执行，提供状态查询、进度追踪、
    回调通知和协作式取消功能。

    Parameters
    ----------
    task_id:
        任务唯一标识。``None`` 时自动生成 UUID。
    pipeline_name:
        管道名称（用于日志和状态展示）。
    pipeline:
        要执行的 :class:`~mosaic.core.pipeline.Pipeline` 实例。
    input_data:
        管道输入数据。
    bus:
        事件总线实例，``None`` 使用全局单例。
    **kwargs:
        透传给 ``pipeline.execute_result()`` 的额外参数
        （如 ``config``、``fail_fast``、``max_workers``）。

    Examples
    --------
    >>> task = pipe.run_async(input_data)
    >>> task.status      # "pending" / "running" / "completed" / "failed"
    >>> task.progress    # 0.0 - 1.0
    >>> result = task.wait(timeout=300)

    使用回调：
    >>> task = pipe.run_async(input_data)
    >>> task.on_complete(lambda r: print(f"Done: {r}"))
    >>> task.on_error(lambda e: print(f"Error: {e}"))
    >>> task.on_progress(lambda p, n: print(f"Progress: {p:.0%} ({n})"))

    取消任务：
    >>> task.cancel()
    >>> task.status  # "cancelled"
    """

    def __init__(
        self,
        pipeline_name: str,
        pipeline: Any,
        input_data: Any,
        task_id: Optional[str] = None,
        bus: Optional[EventBus] = None,
        **kwargs: Any,
    ) -> None:
        self._task_id: str = task_id or str(uuid.uuid4())
        self._pipeline_name: str = pipeline_name
        self._pipeline: Any = pipeline
        self._input_data: Any = input_data
        self._kwargs: Dict[str, Any] = kwargs
        self._bus: EventBus = bus or get_event_bus()

        # 状态（受锁保护）
        self._status: str = TaskStatus.PENDING
        self._progress: float = 0.0
        self._current_node: Optional[str] = None
        self._result: Optional[PipelineResult] = None
        self._error: Optional[BaseException] = None

        # 时间戳
        self._created_at: float = time.time()
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None

        # 线程同步
        self._lock: threading.Lock = threading.Lock()
        self._done_event: threading.Event = threading.Event()
        self._cancel_event: threading.Event = threading.Event()

        # 回调列表
        self._complete_callbacks: List[CompleteCallback] = []
        self._error_callbacks: List[ErrorCallback] = []
        self._progress_callbacks: List[ProgressCallback] = []

        # 工作线程
        self._thread: Optional[threading.Thread] = None

        # EventBus 订阅记录（用于完成后取消订阅）
        self._bus_subscriptions: List[tuple] = []

        # 管道中包含的节点名集合（用于过滤 EventBus 事件）
        self._node_names: Set[str] = set()

        # 日志器
        self._logger: logging.Logger = logging.getLogger(
            f"mosaic.task.{self._task_id[:8]}"
        )

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def task_id(self) -> str:
        """任务唯一标识。"""
        return self._task_id

    @property
    def status(self) -> str:
        """当前状态。"""
        with self._lock:
            return self._status

    @property
    def progress(self) -> float:
        """进度值（0.0 - 1.0）。"""
        with self._lock:
            return self._progress

    @property
    def current_node(self) -> Optional[str]:
        """当前正在执行的节点名。"""
        with self._lock:
            return self._current_node

    @property
    def pipeline_name(self) -> str:
        """管道名称。"""
        return self._pipeline_name

    @property
    def created_at(self) -> float:
        """任务创建时间戳。"""
        return self._created_at

    @property
    def started_at(self) -> Optional[float]:
        """任务开始执行时间戳。"""
        with self._lock:
            return self._started_at

    @property
    def completed_at(self) -> Optional[float]:
        """任务完成时间戳。"""
        with self._lock:
            return self._completed_at

    @property
    def error(self) -> Optional[BaseException]:
        """错误信息（失败时），``None`` 表示无错误。"""
        with self._lock:
            return self._error

    @property
    def is_cancelled(self) -> bool:
        """任务是否已被取消。"""
        return self._cancel_event.is_set()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        """任务是否已完成（无论成功/失败/取消）。

        Returns
        -------
        bool
            ``True`` 表示任务已到达终态，可以调用 :meth:`result` 获取结果。
        """
        with self._lock:
            return self._status in TaskStatus.terminal_statuses()

    def result(self) -> Optional[PipelineResult]:
        """获取执行结果（非阻塞）。

        Returns
        -------
        Optional[PipelineResult]
            成功完成时返回 ``PipelineResult``；未完成或失败时返回 ``None``。
        """
        with self._lock:
            return self._result

    # ------------------------------------------------------------------
    # 阻塞等待
    # ------------------------------------------------------------------
    def wait(self, timeout: Optional[float] = None) -> PipelineResult:
        """阻塞等待任务完成，返回执行结果。

        Parameters
        ----------
        timeout:
            最大等待秒数。``None`` 表示无限等待。

        Returns
        -------
        PipelineResult
            管道执行结果。

        Raises
        ------
        TimeoutError
            超时未完成。
        TaskCancelledError
            任务被取消。
        RuntimeError
            任务失败（原始异常链在 ``__cause__`` 中）。
        """
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(
                f"Task {self._task_id!r} did not complete within {timeout}s."
            )

        with self._lock:
            if self._status == TaskStatus.COMPLETED:
                return self._result  # type: ignore[return-value]
            elif self._status == TaskStatus.CANCELLED:
                raise TaskCancelledError(self._task_id)
            else:
                # FAILED
                raise RuntimeError(
                    f"Task {self._task_id!r} failed: {self._error}"
                ) from self._error

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def cancel(self) -> bool:
        """请求取消任务。

        设置取消标志位，不强制终止线程。节点可通过
        ``context.shared["_cancel_event"]`` 协作式检查并提前退出。

        Returns
        -------
        bool
            ``True`` 表示成功设置取消标志（任务尚未到达终态）；
            ``False`` 表示任务已到达终态，无法取消。
        """
        with self._lock:
            if self._status in TaskStatus.terminal_statuses():
                return False
        self._cancel_event.set()
        self._logger.info("Cancellation requested for task %s.", self._task_id)
        return True

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------
    def on_complete(self, fn: CompleteCallback) -> CompleteCallback:
        """注册完成回调。

        任务成功完成后被调用，参数为 :class:`PipelineResult`。
        如果任务已经完成，回调会被立即调用。
        """
        if not callable(fn):
            raise TypeError("Callback must be callable.")
        with self._lock:
            if self._status == TaskStatus.COMPLETED and self._result is not None:
                fn(self._result)
            else:
                self._complete_callbacks.append(fn)
        return fn

    def on_error(self, fn: ErrorCallback) -> ErrorCallback:
        """注册错误回调。

        任务失败或被取消时被调用，参数为 ``BaseException``。
        如果任务已经失败，回调会被立即调用。
        """
        if not callable(fn):
            raise TypeError("Callback must be callable.")
        with self._lock:
            if self._status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                err = self._error or TaskCancelledError(self._task_id)
                fn(err)
            else:
                self._error_callbacks.append(fn)
        return fn

    def on_progress(self, fn: ProgressCallback) -> ProgressCallback:
        """注册进度回调。

        每次进度更新时被调用，参数为 ``(progress: float, node_name: str)``。
        """
        if not callable(fn):
            raise TypeError("Callback must be callable.")
        with self._lock:
            self._progress_callbacks.append(fn)
        return fn

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """将任务状态序列化为字典。

        Returns
        -------
        Dict[str, Any]
            包含任务 ID、状态、进度、时间戳等信息的字典。
        """
        with self._lock:
            return {
                "task_id": self._task_id,
                "status": self._status,
                "progress": self._progress,
                "current_node": self._current_node,
                "pipeline_name": self._pipeline_name,
                "created_at": self._created_at,
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "duration": (
                    self._completed_at - self._started_at
                    if self._started_at and self._completed_at
                    else None
                ),
                "is_cancelled": self._cancel_event.is_set(),
                "error": str(self._error) if self._error else None,
            }

    # ------------------------------------------------------------------
    # 内部方法（由 async_pipeline 模块调用）
    # ------------------------------------------------------------------
    def _start(self) -> None:
        """启动工作线程（由创建者调用）。"""
        # 收集管道节点名（用于 EventBus 事件过滤）
        try:
            for spec in self._pipeline.node_specs:
                self._node_names.add(spec.name)
        except Exception:  # noqa: BLE001
            pass

        # 订阅 EventBus 事件
        self._subscribe_bus_events()

        # 启动工作线程
        self._thread = threading.Thread(
            target=self._run,
            name=f"mosaic-task-{self._task_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """工作线程目标函数：执行管道并更新状态。"""
        from mosaic.core.context import Context

        with self._lock:
            self._status = TaskStatus.RUNNING
            self._started_at = time.time()

        self._logger.info(
            "Task %s started (pipeline=%s).",
            self._task_id,
            self._pipeline_name,
        )

        try:
            # 构建上下文，注入取消事件
            context = Context(initial_data=self._input_data)
            context.shared["_cancel_event"] = self._cancel_event
            context.shared["_task_id"] = self._task_id

            # 注册 Context 事件回调（跟踪 current_node）
            def _on_context_event(event: Any) -> None:
                if event.type == "node_start":
                    with self._lock:
                        self._current_node = event.node_name
                elif event.type == "pipeline_end":
                    with self._lock:
                        self._progress = 1.0
                        self._current_node = None

            context.on_event(_on_context_event)

            # 检查是否在开始前就已取消
            if self._cancel_event.is_set():
                raise TaskCancelledError(self._task_id)

            # 执行管道
            result = self._pipeline.execute_result(
                self._input_data,
                context=context,
                **self._kwargs,
            )

            # 检查是否在执行过程中被取消
            if self._cancel_event.is_set():
                with self._lock:
                    self._status = TaskStatus.CANCELLED
                    self._completed_at = time.time()
                self._done_event.set()
                err: BaseException = TaskCancelledError(self._task_id)
                self._fire_error_callbacks(err)
                self._logger.info("Task %s cancelled.", self._task_id)
                return

            # 成功完成
            with self._lock:
                self._status = TaskStatus.COMPLETED
                self._progress = 1.0
                self._result = result
                self._completed_at = time.time()
            self._done_event.set()
            self._fire_complete_callbacks(result)
            self._logger.info(
                "Task %s completed in %.3fs.",
                self._task_id,
                self._completed_at - (self._started_at or 0),
            )

        except Exception as exc:
            with self._lock:
                if self._cancel_event.is_set():
                    self._status = TaskStatus.CANCELLED
                    self._error = exc
                else:
                    self._status = TaskStatus.FAILED
                    self._error = exc
                self._completed_at = time.time()
            self._done_event.set()
            self._fire_error_callbacks(exc)
            self._logger.error(
                "Task %s failed: %s",
                self._task_id,
                exc,
                exc_info=True,
            )
        finally:
            self._unsubscribe_bus_events()

    def _subscribe_bus_events(self) -> None:
        """订阅 EventBus 事件以跟踪进度。"""
        # 订阅 PROGRESS 事件
        def on_progress_event(event: Any) -> None:
            node_name = event.payload.get("node_name")
            if node_name and node_name in self._node_names:
                current = event.payload.get("current", 0)
                total = event.payload.get("total", 1)
                if total and total > 0:
                    progress = min(1.0, current / total)
                    with self._lock:
                        self._progress = progress
                    self._fire_progress_callbacks(progress, node_name)

        self._bus.on(EventType.PROGRESS, on_progress_event)
        self._bus_subscriptions.append((EventType.PROGRESS, on_progress_event))

        # 订阅 NODE_START 事件（补充 current_node 跟踪）
        def on_node_start_event(event: Any) -> None:
            node_name = event.payload.get("node_name")
            if node_name and node_name in self._node_names:
                with self._lock:
                    self._current_node = node_name

        self._bus.on(EventType.NODE_START, on_node_start_event)
        self._bus_subscriptions.append((EventType.NODE_START, on_node_start_event))

    def _unsubscribe_bus_events(self) -> None:
        """取消 EventBus 订阅。"""
        for event_type, callback in self._bus_subscriptions:
            try:
                self._bus.off(event_type, callback)
            except Exception:  # noqa: BLE001
                pass
        self._bus_subscriptions.clear()

    def _fire_complete_callbacks(self, result: PipelineResult) -> None:
        """触发完成回调。"""
        for cb in self._complete_callbacks:
            try:
                cb(result)
            except Exception:  # noqa: BLE001
                self._logger.exception("Complete callback raised.")

    def _fire_error_callbacks(self, error: BaseException) -> None:
        """触发错误回调。"""
        for cb in self._error_callbacks:
            try:
                cb(error)
            except Exception:  # noqa: BLE001
                self._logger.exception("Error callback raised.")

    def _fire_progress_callbacks(self, progress: float, node_name: str) -> None:
        """触发进度回调。"""
        for cb in self._progress_callbacks:
            try:
                cb(progress, node_name)
            except Exception:  # noqa: BLE001
                self._logger.exception("Progress callback raised.")

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<AsyncTask id={self._task_id[:8]!r} "
                f"pipeline={self._pipeline_name!r} "
                f"status={self._status!r} "
                f"progress={self._progress:.0%}>"
            )
