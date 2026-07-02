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
"""

__version__ = "0.1.0"


def _patch_torch_version() -> None:
    """修复 torch 版本元数据缺失导致 transformers 崩溃的问题。

    某些 conda 环境或源码编译的 torch 安装中，
    ``importlib.metadata.version("torch")`` 可能返回 ``None``，
    导致 transformers 的 ``version.parse(_torch_version)`` 抛出
    ``TypeError: expected string or bytes-like object``。

    本函数在 mosaic 导入时（早于 transformers 的延迟导入）检查并修复
    两层版本来源：
    1. ``torch.__version__`` 属性（transformers 的 fallback 路径）
    2. ``importlib.metadata.version("torch")``（transformers 的主路径）

    若任一层返回非有效字符串，则从另一层尝试获取，最终回退到 "2.2.0"。
    """
    import importlib.metadata

    # --- 第 1 步：确定一个有效的版本字符串 ---
    resolved_ver: str | None = None

    # 先试 importlib.metadata（transformers 的主路径）
    try:
        meta_ver = importlib.metadata.version("torch")
        if isinstance(meta_ver, str) and meta_ver.strip():
            resolved_ver = meta_ver
    except Exception:
        pass

    # 再试 torch.__version__（transformers 的 fallback 路径）
    if resolved_ver is None:
        try:
            import torch  # type: ignore

            attr_ver = getattr(torch, "__version__", None)
            if isinstance(attr_ver, str) and attr_ver.strip():
                resolved_ver = attr_ver
        except ImportError:
            return  # torch 未安装，无需修复

    # 最终回退
    if resolved_ver is None:
        resolved_ver = "2.2.0"

    # --- 第 2 步：修复 torch.__version__ ---
    try:
        import torch  # type: ignore

        cur = getattr(torch, "__version__", None)
        if not isinstance(cur, str) or not cur.strip():
            torch.__version__ = resolved_ver
    except ImportError:
        pass

    # --- 第 3 步：修复 importlib.metadata（transformers 的主路径）---
    # 若 importlib.metadata.version("torch") 返回 None，用 monkey-patch
    # 确保后续调用返回有效字符串。这比直接写 dist-info 更安全。
    try:
        meta_ver = importlib.metadata.version("torch")
        if not isinstance(meta_ver, str) or not meta_ver.strip():
            _orig_version = importlib.metadata.version

            def _patched_version(name, *_a, **_kw):
                if name == "torch":
                    return resolved_ver
                return _orig_version(name, *_a, **_kw)

            importlib.metadata.version = _patched_version  # type: ignore
    except Exception:
        # PackageNotFoundError 等：torch 的 dist-info 完全缺失，
        # 也需要 patch 以便 transformers 能检测到 torch
        _orig_version = importlib.metadata.version

        def _patched_version_nf(name, *_a, **_kw):
            if name == "torch":
                return resolved_ver
            return _orig_version(name, *_a, **_kw)

        importlib.metadata.version = _patched_version_nf  # type: ignore


_patch_torch_version()


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
    AudioData,
    AvatarData,
    DocumentData,
    ImageData,
    MotionData,
    MosaicData,
    Node,
    Pipeline,
    PipelineResult,
    Branch,
    Merge,
    RagQueryResult,
    SubtitleData,
    TextData,
    VideoData,
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
    "DocumentData",
    "MotionData",
    "AvatarData",
    "RagQueryResult",
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

# 库式使用时自动发现并注册所有内置节点
# （CLI 路径会自行调用 discover()，此处保证 import mosaic 即可用）
try:
    import mosaic.nodes  # noqa: F401 — 触发 @registry.register 装饰器
except ImportError as exc:
    import logging as _logging
    _logging.getLogger("mosaic").warning(
        "Failed to auto-import mosaic.nodes: %s", exc, exc_info=True,
    )
