# mosaic/core/__init__.py
"""Mosaic 核心抽象层。

本包定义了框架的基础抽象、编排引擎与运行时基础设施：

* :mod:`mosaic.core.types`        — 节点间传递的统一数据类型
* :mod:`mosaic.core.node`         — 节点抽象基类与规格说明
* :mod:`mosaic.core.registry`     — 全局节点注册表
* :mod:`mosaic.core.context`      — 管道运行上下文
* :mod:`mosaic.core.pipeline`     — 管道编排引擎（Pipeline / Branch / Merge）
* :mod:`mosaic.core.events`       — 事件总线（EventBus / LoggingListener）
* :mod:`mosaic.core.scheduler`    — 显存调度器（Scheduler，LRU + 无GPU降级）
* :mod:`mosaic.core.task`         — 异步任务封装（AsyncTask）
* :mod:`mosaic.core.async_pipeline` — 异步管道执行桥接
* :mod:`mosaic.core.task_manager`  — 任务管理器（TaskManager）
"""

from mosaic.core.context import (
    Context,
    Event,
    EventHandler,
    NodeOutput,
    RunConfig,
)
from mosaic.core.events import (
    EventBus,
    EventType,
    LoggingListener,
    MosaicEvent,
    get_event_bus,
)
from mosaic.core.node import Node, NodeSpec
from mosaic.core.branch import Branch, Merge
from mosaic.core.pipeline import (
    DryRunResult,
    Pipeline,
    PipelineError,
)
from mosaic.core.registry import NodeRegistry, get_default_registry, registry
from mosaic.core.result import NodeError, PipelineResult
from mosaic.core.scheduler import Scheduler, get_scheduler, set_scheduler
from mosaic.core.task import AsyncTask, TaskCancelledError, TaskStatus
from mosaic.core.task_manager import TaskManager
from mosaic.core.types import (
    AudioData,
    DATA_TYPE_REGISTRY,
    DocumentData,
    ImageData,
    MosaicData,
    SubtitleData,
    TextData,
    VideoData,
    data_from_dict,
)

#: :class:`NodeRegistry` 的简短别名，便于 ``from mosaic.core import Registry``。
Registry = NodeRegistry

__all__ = [
    # types
    "MosaicData",
    "TextData",
    "ImageData",
    "AudioData",
    "VideoData",
    "SubtitleData",
    "DocumentData",
    "DATA_TYPE_REGISTRY",
    "data_from_dict",
    # node
    "Node",
    "NodeSpec",
    # registry
    "NodeRegistry",
    "Registry",
    "registry",
    "get_default_registry",
    # context
    "RunConfig",
    "Event",
    "EventHandler",
    "Context",
    # pipeline
    "Pipeline",
    "Branch",
    "Merge",
    "PipelineError",
    "DryRunResult",
    "PipelineResult",
    "NodeError",
    "NodeOutput",
    # events
    "EventBus",
    "EventType",
    "MosaicEvent",
    "LoggingListener",
    "get_event_bus",
    # scheduler
    "Scheduler",
    "get_scheduler",
    "set_scheduler",
    # async task
    "AsyncTask",
    "TaskCancelledError",
    "TaskStatus",
    "TaskManager",
]
