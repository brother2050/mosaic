"""ChatTTS 与 Fish Speech 共存回归测试。

验证 ChatTTS 后端与 Fish Speech 后端在同一进程中能够独立存在、互不干扰，
且共享的基类（LlamaARModelBase、Vocoder）与注册表（TTSBackendRegistry）
行为符合预期。这些用例不依赖 transformers / 真实权重，仅验证类结构与
注册逻辑。
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
    ChatTTSBackend,
)
from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
    FishSpeechBackend,
)
from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import (
    LlamaARModel,
    LlamaARModelBase,
)
from mosaic.nodes.audio.tts_backends.acoustic_models.fish_ar import (
    FishLlamaARModel,
)
from mosaic.nodes.audio.tts_backends.vocoders.vocos import VocosVocoder
from mosaic.nodes.audio.tts_backends.vocoders.hifi_gan import HiFiGanVocoder


# ----------------------------------------------------------------------
# 后端类属性回归
# ----------------------------------------------------------------------
def test_REGR_01() -> None:
    """ChatTTS 后端类属性不变 (name, sample_rate)。"""
    assert ChatTTSBackend.name == "chattts"
    assert ChatTTSBackend.spec.sample_rate == 24000


def test_REGR_02() -> None:
    """Fish 后端类属性正确 (name, sample_rate)。"""
    assert FishSpeechBackend.name == "fish"
    assert FishSpeechBackend.spec.sample_rate == 22050


def test_REGR_03() -> None:
    """两个后端采样率不同 (24000 != 22050)。"""
    assert ChatTTSBackend.spec.sample_rate != FishSpeechBackend.spec.sample_rate
    assert ChatTTSBackend.spec.sample_rate == 24000
    assert FishSpeechBackend.spec.sample_rate == 22050


def test_REGR_04() -> None:
    """两个后端声码器类型不同 (vocos != hifi_gan)。"""
    assert ChatTTSBackend.spec.vocoder_type == "vocos"
    assert FishSpeechBackend.spec.vocoder_type == "hifi_gan"
    assert ChatTTSBackend.spec.vocoder_type != FishSpeechBackend.spec.vocoder_type


# ----------------------------------------------------------------------
# 声学模型继承回归
# ----------------------------------------------------------------------
def test_REGR_05() -> None:
    """LlamaARModel 继承 LlamaARModelBase。"""
    assert issubclass(LlamaARModel, LlamaARModelBase)


def test_REGR_06() -> None:
    """FishLlamaARModel 继承 LlamaARModelBase。"""
    assert issubclass(FishLlamaARModel, LlamaARModelBase)


def test_REGR_07() -> None:
    """两个 AR 模型都有 model_type == 'ar'。"""
    assert LlamaARModel.model_type == "ar"
    assert FishLlamaARModel.model_type == "ar"


# ----------------------------------------------------------------------
# 声码器独立性回归
# ----------------------------------------------------------------------
def test_REGR_08() -> None:
    """VocosVocoder 和 HiFiGanVocoder 互不干扰 (vocoder_type 不同)。"""
    assert VocosVocoder.vocoder_type == "vocos"
    assert HiFiGanVocoder.vocoder_type == "hifi_gan"
    assert VocosVocoder.vocoder_type != HiFiGanVocoder.vocoder_type


# ----------------------------------------------------------------------
# 后端独立实例化回归
# ----------------------------------------------------------------------
def test_REGR_09() -> None:
    """两个后端可以独立实例化。"""
    chattts = ChatTTSBackend(model_path="/tmp/test")
    fish = FishSpeechBackend(model_path="/tmp/test")
    assert chattts is not None
    assert fish is not None
    assert chattts.name == "chattts"
    assert fish.name == "fish"


# ----------------------------------------------------------------------
# 注册表回归
# ----------------------------------------------------------------------
def test_REGR_10() -> None:
    """TTSBackendRegistry 同时包含 chattts 和 fish（触发延迟注册后）。"""
    from mosaic.nodes.audio.tts_backends.registry import tts_backend_registry

    backends = tts_backend_registry.list_backends()
    names = [b.name for b in backends]
    # 两个后端模块均使用惰性导入，模块本身可在无 torch/transformers 时导入，
    # 因此注册表应同时包含 chattts 与 fish。
    assert "chattts" in names
    assert "fish" in names


def test_REGR_11() -> None:
    """TTS 节点 list_backends 包含基础后端 (edge_tts, transformers)。"""
    from mosaic.nodes.audio.tts import TTS

    backends = TTS.list_backends()
    assert "edge_tts" in backends
    assert "transformers" in backends
