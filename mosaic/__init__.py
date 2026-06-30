# mosaic/__init__.py
"""Mosaic — 多模态生成式 AI 编排框架。

Mosaic 提供统一的节点管道（Pipeline）抽象，将文本、图像、音频、字幕等
多模态生成能力组合为可复用、可编排的工作流。

快速开始
--------
>>> from mosaic.core import Pipeline
>>> from mosaic.nodes.text import TextGenerator
>>> pipe = Pipeline("demo", [TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")])
>>> result = pipe.run({"prompt": "你好"})

主要模块
--------
- :mod:`mosaic.core`     — 核心抽象层（Node / Pipeline / Scheduler / EventBus）
- :mod:`mosaic.nodes`    — 各域节点（text / image / audio / subtitle）
- :mod:`mosaic.cli`      — 命令行工具
- :mod:`mosaic.utils`    — 工具函数
- :mod:`mosaic.backends` — 后端适配层
"""

__version__ = "0.1.0"


def _setup_logging() -> None:
    """统一配置 Mosaic 日志。

    为 ``mosaic`` logger 添加一个 :class:`logging.StreamHandler`，使用统一的
    格式 ``%(asctime)s [%(name)s] %(levelname)s: %(message)s``。日志级别优先
    读取 ``MOSAIC_LOG_LEVEL`` 环境变量，未设置时默认 ``INFO``。

    设计要点
    --------
    * 幂等：重复调用不会重复添加 handler（通过 ``_mosaic_logging_configured``
      标志保护）。
    * 不覆盖用户既有配置：仅在没有 handler 时添加默认 StreamHandler。
    * 保留 ``propagate`` 默认值（``True``），使 pytest ``caplog`` 等基于
      root logger 的捕获机制仍可正常工作；子 logger（如 ``mosaic.scheduler``）
      继承 ``mosaic`` 的级别。
    """
    import logging

    logger = logging.getLogger("mosaic")
    if getattr(logger, "_mosaic_logging_configured", False):
        return

    # 仅在没有 handler 时添加默认 StreamHandler，避免覆盖用户配置
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
        )
        logger.addHandler(handler)

    # 日志级别：优先环境变量，其次保持现有级别，最后默认 INFO
    level_name = None
    try:
        from mosaic.core.env import MosaicEnv

        level_name = MosaicEnv.get_log_level()
    except Exception:  # noqa: BLE001
        import os

        level_name = os.environ.get("MOSAIC_LOG_LEVEL")
        if level_name:
            level_name = level_name.strip().upper()

    if level_name:
        level = logging.getLevelName(level_name)
        if isinstance(level, int):
            logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.INFO)

    logger._mosaic_logging_configured = True  # type: ignore[attr-defined]


_setup_logging()

# 便捷导出：常用核心类
from mosaic.core import (
    MosaicData,
    TextData,
    ImageData,
    AudioData,
    VideoData,
    SubtitleData,
    Node,
    Pipeline,
    Branch,
    Merge,
    PipelineResult,
    registry,
)

# 便捷导出：插件系统
from mosaic.core.plugin import PluginManager, plugin_manager, node

__all__ = [
    "__version__",
    # data types
    "MosaicData",
    "TextData",
    "ImageData",
    "AudioData",
    "VideoData",
    "SubtitleData",
    # core
    "Node",
    "Pipeline",
    "Branch",
    "Merge",
    "PipelineResult",
    "registry",
    # plugin system
    "PluginManager",
    "plugin_manager",
    "node",  # @node 装饰器
]
