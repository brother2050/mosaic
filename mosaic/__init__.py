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
