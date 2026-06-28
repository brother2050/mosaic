# mosaic/core/context.py
"""管道运行上下文。

本模块定义了 ``Context`` 与 ``RunConfig``，在管道运行期间于节点之间传递，
提供共享数据存储、运行配置、事件回调接口与中间产物存储能力。

设计要点
--------
* ``RunConfig`` 封装设备、精度、批大小等运行时配置，使用 dataclass 表达。
* ``Context`` 是运行期总线：
    * **共享数据存储** —— 跨节点的全局键值区（``shared``）。
    * **中间产物存储** —— 任意节点的输出都可按节点名取出（``artifacts``）。
    * **事件回调** —— 通过 ``on_event`` 注册回调，在节点执行前后触发。
* 上下文支持上下文管理器协议，进入时触发 ``pipeline_start`` 事件，
  退出时触发 ``pipeline_end`` 事件，并捕获异常触发 ``error`` 事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mosaic.core.types import MosaicData

__all__ = [
    "RunConfig",
    "Event",
    "EventHandler",
    "Context",
]


# ---------------------------------------------------------------------------
# RunConfig — 运行配置
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """管道运行时配置。

    Attributes
    ----------
    device:
        计算设备，如 ``"cuda"``、``"cuda:0"``、``"cpu"``、``"mps"``。
    precision:
        推理精度，可选 ``"fp32"``、``"fp16"``、``"bf16"``。
    batch_size:
        批大小。
    seed:
        随机种子，用于结果可复现。``None`` 表示不固定。
    dtype_size:
        混合精度时的权重位宽（仅记录用）。
    extra:
        任意附加配置，供特定节点读取。
    """

    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 1
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验配置合法性。"""
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(
                f"Invalid precision {self.precision!r}, "
                f"expected one of: fp32, fp16, bf16."
            )
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive int, got {self.batch_size!r}."
            )


# ---------------------------------------------------------------------------
# 事件系统
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """运行期事件。

    Attributes
    ----------
    type:
        事件类型，如 ``"pipeline_start"``、``"node_start"``、
        ``"node_end"``、``"error"``、``"pipeline_end"``。
    node_name:
        触发事件的节点名（非节点事件时为 ``None``）。
    payload:
        事件附带的任意数据（如耗时、输出摘要等）。
    """

    type: str
    node_name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


#: 事件回调函数签名：接收一个 :class:`Event`，无返回值。
EventHandler = Callable[[Event], None]


# ---------------------------------------------------------------------------
# Context — 管道运行上下文
# ---------------------------------------------------------------------------
class Context:
    """管道运行上下文，在节点之间传递。

    Parameters
    ----------
    config:
        运行配置。若为 ``None`` 使用默认配置。
    initial_data:
        初始输入数据，写入 ``shared`` 供首个节点读取。

    功能
    ----
    * **共享数据存储** ``shared``：跨节点的全局键值区。
    * **中间产物存储** ``artifacts``：按节点名保存其输出，可单独取出。
    * **事件回调** ``on_event``：注册回调，在节点执行前后触发。
    * **运行配置** ``config``：设备、精度、批大小等。
    """

    def __init__(
        self,
        config: Optional[RunConfig] = None,
        initial_data: Optional[MosaicData] = None,
    ) -> None:
        self.config: RunConfig = config or RunConfig()
        # 共享数据存储：跨节点的全局键值区
        self.shared: MosaicData = initial_data if initial_data is not None else MosaicData()
        # 中间产物存储：node_name -> 该节点最近一次输出
        self.artifacts: Dict[str, MosaicData] = {}
        # 事件回调列表
        self._handlers: List[EventHandler] = []
        # 运行状态
        self._active: bool = False

    # -- 上下文管理器协议 --------------------------------------------------
    def __enter__(self) -> "Context":
        """进入运行：标记为活跃并触发 ``pipeline_start`` 事件。"""
        self._active = True
        self.emit(Event(type="pipeline_start", payload={"config": self.config.__dict__}))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出运行：触发 ``pipeline_end`` 或 ``error`` 事件。"""
        if exc_type is not None:
            self.emit(
                Event(
                    type="error",
                    payload={
                        "exception_type": exc_type.__name__ if exc_type else None,
                        "exception": str(exc_val) if exc_val else None,
                    },
                )
            )
        else:
            self.emit(Event(type="pipeline_end", payload={"artifacts": list(self.artifacts.keys())}))
        self._active = False
        return None  # 不吞掉异常

    # -- 事件系统 ----------------------------------------------------------
    def on_event(self, handler: EventHandler) -> EventHandler:
        """注册一个事件回调函数，返回该函数以支持装饰器用法。"""
        if not callable(handler):
            raise TypeError("Event handler must be callable.")
        self._handlers.append(handler)
        return handler

    def emit(self, event: Event) -> None:
        """触发一个事件，依次调用所有已注册的回调。

        单个回调抛出的异常会被捕获并忽略，避免影响其他回调或管道运行。
        """
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                # 回调失败不应中断管道
                continue

    # -- 共享数据存储 ------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        """向共享数据存储写入一个键值对。"""
        self.shared[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """从共享数据存储读取一个值。"""
        return self.shared.get(key, default)

    # -- 中间产物存储 ------------------------------------------------------
    def store_artifact(self, node_name: str, data: MosaicData) -> None:
        """保存某个节点的输出作为中间产物。

        Parameters
        ----------
        node_name:
            产出该数据的节点名称。
        data:
            节点输出的数据容器。
        """
        self.artifacts[node_name] = data

    def get_artifact(self, node_name: str) -> MosaicData:
        """取出某个节点最近一次的输出。

        Raises
        ------
        KeyError
            该节点尚无产物记录。
        """
        if node_name not in self.artifacts:
            raise KeyError(
                f"No artifact for node {node_name!r}. "
                f"Available: {list(self.artifacts.keys())}"
            )
        return self.artifacts[node_name]

    def has_artifact(self, node_name: str) -> bool:
        """判断某个节点是否已有产物记录。"""
        return node_name in self.artifacts

    def list_artifacts(self) -> List[str]:
        """列出所有已记录产物的节点名。"""
        return list(self.artifacts.keys())

    # -- 节点执行辅助 ------------------------------------------------------
    def notify_node_start(self, node_name: str, input_data: MosaicData) -> None:
        """通知节点即将开始执行。"""
        self.emit(
            Event(
                type="node_start",
                node_name=node_name,
                payload={"input_keys": list(input_data.keys())},
            )
        )

    def notify_node_end(
        self,
        node_name: str,
        output_data: MosaicData,
        elapsed: Optional[float] = None,
    ) -> None:
        """通知节点执行结束，并自动存储产物。"""
        self.store_artifact(node_name, output_data)
        payload: Dict[str, Any] = {"output_keys": list(output_data.keys())}
        if elapsed is not None:
            payload["elapsed_seconds"] = elapsed
        self.emit(
            Event(
                type="node_end",
                node_name=node_name,
                payload=payload,
            )
        )

    # -- 运行状态 ----------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """管道是否处于运行中。"""
        return self._active

    def __repr__(self) -> str:
        return (
            f"Context(active={self._active}, device={self.config.device!r}, "
            f"precision={self.config.precision!r}, "
            f"shared_keys={list(self.shared.keys())}, "
            f"artifacts={list(self.artifacts.keys())})"
        )
