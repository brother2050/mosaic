# tests/final/conftest.py
"""Mosaic 最终验收测试公共 fixtures。

提供共享 fixture，用于所有最终验收测试文件，包括：
- 注册表初始化和节点发现
- TTS 后端注册表
- 测试数据（mock AudioData / TextData / ImageData）
- Scheduler / EventBus / PluginManager 实例
- 预期的域和节点计数数据
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
from PIL import Image

from mosaic.core import (
    EventBus,
    ImageData,
    Scheduler,
    TextData,
    registry as _registry,
)
from mosaic.core.plugin import plugin_manager as _plugin_manager
from mosaic.core.types import AudioData


# ---------------------------------------------------------------------------
# Mock torch 模块注入（避免测试环境无 torch 时导入失败）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_torch_module() -> Any:
    """注入 mock torch 模块，确保无 GPU 环境也可运行测试。"""
    if "torch" not in sys.modules:
        _mt = types.ModuleType("torch")
        _mt.__spec__ = MagicMock()  # 避免 transformers 的 find_spec 报错
        _mt.__version__ = "2.0.0"
        _mt.LongTensor = MagicMock
        _mt.FloatTensor = MagicMock
        _mt.BoolTensor = MagicMock
        _mt.frombuffer = MagicMock(return_value=MagicMock())
        _ctx = MagicMock()
        _ctx.__enter__ = MagicMock(return_value=None)
        _ctx.__exit__ = MagicMock(return_value=None)
        _mt.inference_mode = MagicMock(return_value=_ctx)
        _mt.no_grad = MagicMock(return_value=_ctx)
        _mt.float16 = "fp16"
        _mt.float32 = "fp32"
        _mt.bfloat16 = "bf16"
        _mt.Generator = MagicMock
        _mt.Tensor = MagicMock
        _mt.ones_like = MagicMock(return_value=MagicMock())
        _mt.ones = MagicMock(return_value=MagicMock())
        _mt.tensor = MagicMock(return_value=MagicMock())
        _mt.zeros = MagicMock(return_value=MagicMock())
        _mt.randn = MagicMock(return_value=MagicMock())
        _mt.arange = MagicMock(return_value=MagicMock())
        _mt.long = "int64"
        _mt.int = "int32"
        _mt.from_numpy = MagicMock(return_value=MagicMock())
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.__spec__ = MagicMock()
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        _mcuda.empty_cache = MagicMock()
        _mt.cuda = _mcuda
        _mnn = types.ModuleType("torch.nn")
        _mnn.__spec__ = MagicMock()
        _mnn.Module = MagicMock
        _mnn.Parameter = MagicMock
        _mt.nn = _mnn
        sys.modules["torch"] = _mt
        sys.modules["torch.cuda"] = _mcuda
        sys.modules["torch.nn"] = _mnn
    yield


# ---------------------------------------------------------------------------
# 注册表相关 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reset_registry() -> Any:
    """重新发现所有节点并返回注册表。

    重置 discover 标志，然后调用 discover() 扫描 mosaic.nodes 包下所有节点，
    确保测试在一致的注册状态上运行。
    """
    _registry.reset_discovery()
    _registry.discover("mosaic.nodes")
    return _registry


@pytest.fixture
def registry(reset_registry: Any) -> Any:
    """返回全局注册表单例（已重新发现节点）。"""
    return reset_registry


@pytest.fixture
def tts_registry() -> Any:
    """返回 TTS 后端注册表单例。"""
    from mosaic.nodes.audio.tts_backends.registry import (
        tts_backend_registry,
    )

    return tts_backend_registry


@pytest.fixture
def all_nodes(registry: Any) -> list[Any]:
    """返回注册表中所有节点的 NodeSpec 列表。"""
    return registry.list_nodes()


@pytest.fixture
def all_node_names(registry: Any) -> list[str]:
    """返回注册表中所有节点的名称列表。"""
    return registry.list_names()


@pytest.fixture
def all_tts_backends(tts_registry: Any) -> list[Any]:
    """返回所有 TTS 后端规格列表。"""
    return tts_registry.list_backends()


# ---------------------------------------------------------------------------
# 预期数据 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def nine_domains() -> list[str]:
    """返回 9 个域的列表。"""
    return [
        "text",
        "image",
        "video",
        "audio",
        "subtitle",
        "consistency",
        "digital_human",
        "export",
        "rag",
    ]


@pytest.fixture
def expected_domain_counts() -> dict[str, int]:
    """返回每个域预期的节点数量。"""
    return {
        "text": 6,
        "image": 6,
        "video": 8,
        "audio": 5,
        "subtitle": 3,
        "consistency": 3,
        "digital_human": 4,
        "export": 3,
        "rag": 4,
    }


@pytest.fixture
def expected_node_names() -> dict[str, list[str]]:
    """返回每个域预期的节点名称列表。"""
    return {
        "text": [
            "TextGenerator",
            "Chat",
            "TextRewriter",
            "Translator",
            "TextSummarizer",
            "TextClassifier",
        ],
        "image": [
            "TextToImage",
            "ImageToImage",
            "Inpainting",
            "Upscaler",
            "BackgroundRemover",
            "Stylizer",
        ],
        "video": [
            "TextToVideo",
            "ImageToVideo",
            "VideoContinuation",
            "FrameInterpolator",
            "FrameExtractor",
            "HunyuanVideo",
            "LTXVideo",
            "WanVideo",
        ],
        "audio": [
            "TTS",
            "ASR",
            "MusicGenerator",
            "SoundEffectGenerator",
            "VoiceClone",
        ],
        "subtitle": [
            "SubtitleGenerator",
            "SubtitleTranslator",
            "SubtitleAligner",
        ],
        "consistency": [
            "IdentityKeeper",
            "StyleKeeper",
            "CrossFrameConsistency",
        ],
        "digital_human": [
            "AvatarDriver",
            "LipSyncer",
            "MotionGenerator",
            "RealtimeRenderer",
        ],
        "export": [
            "VideoEncoder",
            "Livestreamer",
            "MultiFormatExporter",
        ],
        "rag": [
            "DocumentParser",
            "VectorIndexer",
            "Retriever",
            "CitationGenerator",
        ],
    }


# ---------------------------------------------------------------------------
# Mock 测试数据 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_audio_data() -> AudioData:
    """返回一个简单的 AudioData，包含全零波形。"""
    waveform = np.zeros((1, 16000), dtype=np.float32)
    return AudioData(waveform=waveform, sample_rate=16000)


@pytest.fixture
def mock_text_data() -> TextData:
    """返回一个简单的 TextData，内容为 "hello world"。"""
    return TextData(content="hello world", language="en")


@pytest.fixture
def mock_image_data() -> ImageData:
    """返回一个简单的 ImageData，包含小型 PIL 图像。"""
    img = Image.new("RGB", (64, 64), color="red")
    return ImageData(image=img, size=(64, 64))


# ---------------------------------------------------------------------------
# 调度器与事件总线 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def scheduler() -> Scheduler:
    """返回一个全新的 Scheduler 实例（CPU 模式）。"""
    from mosaic.core.scheduler import set_scheduler

    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    sched = Scheduler(bus=bus, device="cpu")
    set_scheduler(sched)
    return sched


@pytest.fixture
def event_bus() -> EventBus:
    """返回一个清理后的全新 EventBus 实例。"""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


@pytest.fixture
def plugin_manager() -> Any:
    """返回全局 plugin_manager 单例。"""
    return _plugin_manager