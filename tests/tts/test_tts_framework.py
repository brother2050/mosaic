"""测试 TTS 框架基础架构。

覆盖 TTSBackend / TextFrontend / AcousticModel / Vocoder 抽象基类、
TTSBackendSpec 数据结构、TTSBackendRegistry 注册与自动选择，以及
StreamAdapter / StreamSession 流式拼接流程。
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec
from mosaic.nodes.audio.tts_backends.text_frontends.base import TextFrontend
from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel
from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder
from mosaic.nodes.audio.tts_backends.registry import (
    TTSBackendRegistry,
    tts_backend_registry,
)
from mosaic.nodes.audio.tts_backends.streaming.base import (
    StreamAdapter,
    StreamSession,
)
from mosaic.core.types import AudioData


# ----------------------------------------------------------------------
# 用于 registry 测试的最小 TTSBackend 具体子类（不依赖真实模型）
# ----------------------------------------------------------------------
class _DummyTTSBackend(TTSBackend):
    """最小可注册的 TTSBackend 子类（仅用于 registry 测试）。"""

    name = "ttsfw_dummy"
    spec = TTSBackendSpec(
        name="ttsfw_dummy",
        supported_languages=["zh"],
        supports_streaming=True,
        min_gpu_memory_gb=0.0,
    )

    def _build_pipeline(self) -> None:  # pragma: no cover - 仅占位
        pass


class _AutoSelectTTSBackend(TTSBackend):
    """支持独占语言 'ttsfw_lang' 的子类，用于 auto_select 测试。"""

    name = "ttsfw_autoselect"
    spec = TTSBackendSpec(
        name="ttsfw_autoselect",
        supported_languages=["ttsfw_lang"],
        supports_streaming=True,
        min_gpu_memory_gb=0.0,
    )

    def _build_pipeline(self) -> None:  # pragma: no cover - 仅占位
        pass


# ----------------------------------------------------------------------
# T_TTSFW_01~04：抽象基类不可直接实例化
# ----------------------------------------------------------------------
def test_TTSFW_01() -> None:
    """TTSBackend 抽象类不能直接实例化。"""
    with pytest.raises(TypeError):
        TTSBackend(model_path="/tmp/x")


def test_TTSFW_02() -> None:
    """TextFrontend 抽象类不能直接实例化。"""
    with pytest.raises(TypeError):
        TextFrontend()


def test_TTSFW_03() -> None:
    """AcousticModel 抽象类不能直接实例化。"""
    with pytest.raises(TypeError):
        AcousticModel()


def test_TTSFW_04() -> None:
    """Vocoder 抽象类不能直接实例化。"""
    with pytest.raises(TypeError):
        Vocoder()


# ----------------------------------------------------------------------
# T_TTSFW_05：TTSBackendSpec 创建与字段
# ----------------------------------------------------------------------
def test_TTSFW_05() -> None:
    """TTSBackendSpec 创建和字段。"""
    spec = TTSBackendSpec(
        name="test_backend",
        supported_languages=["zh", "en"],
        supports_streaming=True,
        supports_voice_clone=True,
        vocoder_type="vocos",
        acoustic_type="ar",
        min_gpu_memory_gb=2.0,
        model_license="MIT",
        sample_rate=24000,
        default_params={"temperature": 0.3, "top_p": 0.7, "top_k": 20},
    )
    # 逐字段校验
    assert spec.name == "test_backend"
    assert spec.supported_languages == ["zh", "en"]
    assert spec.supports_streaming is True
    assert spec.supports_voice_clone is True
    assert spec.vocoder_type == "vocos"
    assert spec.acoustic_type == "ar"
    assert spec.min_gpu_memory_gb == 2.0
    assert spec.model_license == "MIT"
    assert spec.sample_rate == 24000
    assert spec.default_params["temperature"] == 0.3


# ----------------------------------------------------------------------
# T_TTSFW_06：registry 注册与获取
# ----------------------------------------------------------------------
def test_TTSFW_06() -> None:
    """TTSBackendRegistry 注册和获取。"""
    registry: TTSBackendRegistry = tts_backend_registry
    registry.register("ttsfw_dummy", _DummyTTSBackend)
    # 注册后可通过 get 获取到同一个类对象
    assert registry.get("ttsfw_dummy") is _DummyTTSBackend
    # list_backends 返回 spec 列表，应包含已注册后端的名称
    names = [spec.name for spec in registry.list_backends()]
    assert "ttsfw_dummy" in names


# ----------------------------------------------------------------------
# T_TTSFW_07：registry auto_select 自动选择
# ----------------------------------------------------------------------
def test_TTSFW_07() -> None:
    """TTSBackendRegistry auto_select 自动选择。"""
    registry: TTSBackendRegistry = tts_backend_registry
    registry.register("ttsfw_autoselect", _AutoSelectTTSBackend)
    # 用独占语言约束，确保只有 ttsfw_autoselect 满足要求
    chosen = registry.auto_select({"language": "ttsfw_lang"})
    assert chosen == "ttsfw_autoselect"


# ----------------------------------------------------------------------
# T_TTSFW_08~12：流式适配器与会话
# ----------------------------------------------------------------------
def test_TTSFW_08() -> None:
    """StreamAdapter 创建 StreamSession。"""
    adapter = StreamAdapter(chunk_size=100, overlap=20, sample_rate=24000)
    session = adapter.create_stream()
    assert isinstance(session, StreamSession)


def test_TTSFW_09() -> None:
    """StreamSession push/pop 基本流程。"""
    adapter = StreamAdapter(chunk_size=100, overlap=0, sample_rate=24000)
    session = adapter.create_stream()
    # 推入恰好一个 chunk 的样本
    session.push(np.ones(100, dtype=np.float32))
    chunk = session.pop()
    assert isinstance(chunk, AudioData)
    assert chunk.waveform.shape == (100,)
    assert chunk.sample_rate == 24000


def test_TTSFW_10() -> None:
    """StreamSession overlap-add 平滑。"""
    adapter = StreamAdapter(chunk_size=100, overlap=20, sample_rate=24000)
    session = adapter.create_stream()
    # 第一块全 1，触发首个 chunk（无前段，无交叉淡入淡出）
    session.push(np.ones(100, dtype=np.float32))
    first = session.pop()
    assert first is not None and first.waveform.shape == (100,)
    # 第二块全 0，与上一块尾部（全 1）做 overlap-add 交叉淡入淡出
    session.push(np.zeros(100, dtype=np.float32))
    second = session.pop()
    assert second is not None
    out = second.waveform
    # overlap 区域应由 1 平滑过渡到 0（递减），其后再保持 0
    assert out[0] > out[10] > out[19]
    assert abs(out[20]) < 1e-6


def test_TTSFW_11() -> None:
    """StreamSession flush 输出剩余。"""
    adapter = StreamAdapter(chunk_size=100, overlap=0, sample_rate=24000)
    session = adapter.create_stream()
    # 推入不足一个 chunk 的样本，pop 应返回 None
    session.push(np.ones(30, dtype=np.float32))
    assert session.pop() is None
    # flush 输出剩余缓冲
    rest = session.flush()
    assert isinstance(rest, AudioData)
    assert rest.waveform.shape == (30,)


def test_TTSFW_12() -> None:
    """StreamSession on_chunk_ready 回调触发。"""
    adapter = StreamAdapter(chunk_size=100, overlap=0, sample_rate=24000)
    session = adapter.create_stream()
    received: list[AudioData] = []
    session.on_chunk_ready(lambda audio: received.append(audio))
    # 推入超过一个 chunk 的样本，应触发至少一次回调
    session.push(np.ones(150, dtype=np.float32))
    assert len(received) >= 1
    assert all(isinstance(a, AudioData) for a in received)
