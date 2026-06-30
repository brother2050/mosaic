# mosaic/core/context.py
"""管道运行上下文。

本模块定义了 ``Context`` 与 ``RunConfig``，在管道运行期间于节点之间传递，
提供共享数据存储、运行配置、事件回调接口与中间产物存储能力。

设计要点
--------
* ``RunConfig`` 封装设备、精度、批大小等运行时配置，使用 dataclass 表达。
* ``Context`` 是运行期总线：
    * **共享数据存储** —— 跨节点的全局键值区（``shared``）。
    * **中间产物存储** —— 任意节点的输出都可按节点名取出（``artifacts``），
      每条记录包含数据、时间戳与耗时（``NodeOutput``）。
    * **事件回调** —— 通过 ``on_event`` 注册回调，在节点执行前后触发。
* 上下文支持上下文管理器协议，进入时触发 ``pipeline_start`` 事件，
  退出时触发 ``pipeline_end`` 事件，并捕获异常触发 ``error`` 事件。
* 中间产物支持快照导出/导入（JSON），用于调试与复现。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from mosaic.core.types import MosaicData, data_from_dict

__all__ = [
    "RunConfig",
    "Event",
    "EventHandler",
    "Context",
    "NodeOutput",
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
    extra:
        任意附加配置，供特定节点读取。
    """

    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 1
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

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
    event_type:
        事件类型，如 ``"pipeline_start"``、``"node_start"``、
        ``"node_end"``、``"error"``、``"pipeline_end"``。
    node_name:
        触发事件的节点名（非节点事件时为 ``None``）。
    payload:
        事件附带的任意数据（如耗时、输出摘要等）。
    """

    event_type: str
    node_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


#: 事件回调函数签名：接收一个 :class:`Event`，无返回值。
EventHandler = Callable[[Event], None]


# ---------------------------------------------------------------------------
# NodeOutput — 中间产物记录
# ---------------------------------------------------------------------------
@dataclass
class NodeOutput:
    """单个节点的中间产物记录。

    Attributes
    ----------
    data:
        节点输出的数据容器。
    timestamp:
        产出时间戳（``time.time()``， Unix 纪元秒）。
    duration:
        节点执行耗时（秒）。
    """

    data: MosaicData
    timestamp: float = field(default_factory=time.time)
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "data": self.data.to_dict() if isinstance(self.data, MosaicData) else self.data,
            "timestamp": self.timestamp,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeOutput":
        """从字典反序列化。"""
        raw_data = d.get("data", {})
        data = data_from_dict(raw_data) if isinstance(raw_data, dict) else MosaicData()
        return cls(
            data=data,
            timestamp=d.get("timestamp", time.time()),
            duration=d.get("duration"),
        )


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
    max_intermediate:
        中间产物最大存储数量。``None`` 表示不限制。超过时按 FIFO 淘汰
        最早的记录，防止内存溢出。

    功能
    ----
    * **共享数据存储** ``shared``：跨节点的全局键值区。
    * **中间产物存储** ``artifacts``：按节点名保存其输出（含时间戳与耗时）。
    * **事件回调** ``on_event``：注册回调，在节点执行前后触发。
    * **运行配置** ``config``：设备、精度、批大小等。
    * **快照导出/导入**：``snapshot()`` / ``save_snapshot()`` / ``load_snapshot()``。
    """

    def __init__(
        self,
        config: RunConfig | None = None,
        initial_data: MosaicData | None = None,
        max_intermediate: int | None = None,
    ) -> None:
        self.config: RunConfig = config or RunConfig()
        # 共享数据存储：跨节点的全局键值区
        self.shared: MosaicData = initial_data if initial_data is not None else MosaicData()
        # 中间产物存储：node_name -> NodeOutput（含 data/timestamp/duration）
        self._artifacts: dict[str, NodeOutput] = {}
        # 中间产物插入顺序（用于 FIFO 淘汰）
        self._artifact_order: list[str] = []
        # 最大中间产物数量
        self._max_intermediate: int | None = max_intermediate
        # 事件回调列表
        self._handlers: list[EventHandler] = []
        # 运行状态
        self._active: bool = False
        # 日志器（用于记录回调异常等）
        self._logger: logging.Logger = logging.getLogger("mosaic.context")

    # -- 上下文管理器协议 --------------------------------------------------
    def __enter__(self) -> "Context":
        """进入运行：标记为活跃并触发 ``pipeline_start`` 事件。"""
        self._active = True
        # 使用浅拷贝：避免后续 config 被修改时已发出的事件 payload 随之变化
        self.emit(
            Event(event_type="pipeline_start", payload={"config": dict(self.config.__dict__)})
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出运行：触发 ``pipeline_end`` 或 ``error`` 事件。"""
        if exc_type is not None:
            self.emit(
                Event(
                    event_type="error",
                    payload={
                        "exception_type": exc_type.__name__ if exc_type else None,
                        "exception": str(exc_val) if exc_val else None,
                    },
                )
            )
        else:
            self.emit(
                Event(
                    event_type="pipeline_end",
                    payload={"artifacts": list(self._artifacts.keys())},
                )
            )
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

        单个回调抛出的异常会被捕获并记录日志，避免影响其他回调或管道运行。
        """
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001
                # 回调失败不应中断管道，但需记录以便排查
                self._logger.warning(
                    "Event callback for %r raised: %s",
                    event.event_type,
                    exc,
                    exc_info=True,
                )
                continue

    # -- 共享数据存储 ------------------------------------------------------
    def set_value(self, key: str, value: Any) -> None:
        """向共享数据存储写入一个键值对。"""
        self.shared[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """从共享数据存储读取一个值。"""
        return self.shared.get(key, default)

    # -- 中间产物存储（增强版） ---------------------------------------------
    def store_artifact(
        self,
        node_name: str,
        data: MosaicData,
        duration: float | None = None,
    ) -> None:
        """保存某个节点的输出作为中间产物。

        Parameters
        ----------
        node_name:
            产出该数据的节点名称。
        data:
            节点输出的数据容器。
        duration:
            节点执行耗时（秒），可选。
        """
        # 如果已存在，更新但保持原顺序位置
        if node_name not in self._artifacts:
            self._artifact_order.append(node_name)
            # FIFO 淘汰
            if (
                self._max_intermediate is not None
                and len(self._artifact_order) > self._max_intermediate
            ):
                evicted = self._artifact_order.pop(0)
                self._artifacts.pop(evicted, None)
        self._artifacts[node_name] = NodeOutput(
            data=data, duration=duration
        )

    def get_artifact(self, node_name: str) -> MosaicData:
        """取出某个节点最近一次的输出数据。

        Raises
        ------
        KeyError
            该节点尚无产物记录。
        """
        if node_name not in self._artifacts:
            raise KeyError(
                f"No artifact for node {node_name!r}. "
                f"Available: {list(self._artifacts.keys())}"
            )
        return self._artifacts[node_name].data

    def get_artifact_record(self, node_name: str) -> NodeOutput:
        """取出某个节点的完整产物记录（含时间戳与耗时）。

        Raises
        ------
        KeyError
            该节点尚无产物记录。
        """
        if node_name not in self._artifacts:
            raise KeyError(
                f"No artifact for node {node_name!r}. "
                f"Available: {list(self._artifacts.keys())}"
            )
        return self._artifacts[node_name]

    def has_artifact(self, node_name: str) -> bool:
        """判断某个节点是否已有产物记录。"""
        return node_name in self._artifacts

    def list_artifacts(self) -> list[str]:
        """列出所有已记录产物的节点名。"""
        return list(self._artifacts.keys())

    @property
    def artifacts(self) -> dict[str, MosaicData]:
        """中间产物的便捷视图：``{node_name: MosaicData}``（只读快照）。"""
        return {name: rec.data for name, rec in self._artifacts.items()}

    # -- 中间产物访问 API（新增） -------------------------------------------
    def get_intermediate(self, node_name: str) -> MosaicData:
        """获取指定节点的中间产物。

        等价于 :meth:`get_artifact`，提供与 :class:`PipelineResult` 一致的 API。

        Raises
        ------
        KeyError
            找不到对应产物。
        """
        return self.get_artifact(node_name)

    def list_intermediate(self) -> list[str]:
        """列出所有已存储的中间产物节点名。"""
        return self.list_artifacts()

    def get_all_intermediate(self) -> dict[str, MosaicData]:
        """获取全部中间产物字典 ``{node_name: MosaicData}``。"""
        return dict(self.artifacts)

    def get_node_durations(self) -> dict[str, float | None]:
        """获取各节点的执行耗时字典。

        Returns
        -------
        dict[str, float | None]
            ``{node_name: duration_seconds}``，未记录耗时的为 ``None``。
        """
        return {name: rec.duration for name, rec in self._artifacts.items()}

    # -- 快照导出/导入（新增） ----------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """导出所有中间产物为可序列化字典。

        Returns
        -------
        dict[str, Any]
            包含 ``config``、``artifacts``（每个含 data/timestamp/duration）
            的字典，可 ``json.dumps`` 序列化。
        """
        return {
            "config": {
                "device": self.config.device,
                "precision": self.config.precision,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "artifacts": {
                name: rec.to_dict()
                for name, rec in self._artifacts.items()
            },
            "artifact_order": list(self._artifact_order),
        }

    def save_snapshot(self, path: str) -> None:
        """保存中间产物快照到 JSON 文件。

        图片等二进制数据会通过 ``MosaicData.to_dict()`` 转为 base64 编码。

        Parameters
        ----------
        path:
            目标文件路径。
        """
        snap = self.snapshot()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)

    def load_snapshot(self, path: str) -> None:
        """从 JSON 文件加载中间产物快照。

        清除当前所有中间产物，替换为文件中的内容，并恢复运行配置
        （与 :meth:`snapshot` 对称）。

        Parameters
        ----------
        path:
            快照文件路径。
        """
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        self._artifacts.clear()
        self._artifact_order.clear()
        # 恢复运行配置（与 snapshot() 对称：snapshot 保存了 config 字段）
        config_data = snap.get("config")
        if isinstance(config_data, dict):
            # 仅取 RunConfig 已知字段，避免多余键导致构造失败
            _known_fields = {"device", "precision", "batch_size", "seed", "extra"}
            filtered = {
                k: v for k, v in config_data.items() if k in _known_fields
            }
            self.config = RunConfig(**filtered)
        elif isinstance(config_data, RunConfig):
            self.config = config_data
        for name in snap.get("artifact_order", []):
            rec_dict = snap.get("artifacts", {}).get(name)
            if rec_dict is not None:
                self._artifacts[name] = NodeOutput.from_dict(rec_dict)
                self._artifact_order.append(name)

    # -- 节点执行辅助 ------------------------------------------------------
    def notify_node_start(self, node_name: str, input_data: MosaicData) -> None:
        """通知节点即将开始执行。"""
        self.emit(
            Event(
                event_type="node_start",
                node_name=node_name,
                payload={"input_keys": list(input_data.keys())},
            )
        )

    def notify_node_end(
        self,
        node_name: str,
        output_data: MosaicData,
        elapsed: float | None = None,
    ) -> None:
        """通知节点执行结束，并自动存储产物。"""
        self.store_artifact(node_name, output_data, duration=elapsed)
        payload: dict[str, Any] = {"output_keys": list(output_data.keys())}
        if elapsed is not None:
            payload["elapsed_seconds"] = elapsed
        self.emit(
            Event(
                event_type="node_end",
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
            f"artifacts={list(self._artifacts.keys())})"
        )
