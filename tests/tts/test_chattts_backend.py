"""测试 ChatTTSBackend。

T_CHTTS_01~08 不依赖 transformers，可在当前环境运行；
T_CHTTS_09~16 需要 transformers（真实加载），缺失时跳过。加载用例使用
极小参数的 LlamaARModel（经 monkeypatch 替换）以加速并避免 OOM，且不依赖
真实预训练权重（随机初始化即可验证管线编排逻辑）。
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 探测可选依赖
try:  # noqa: SIM105
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except Exception:  # noqa: BLE001
    _HAS_TRANSFORMERS = False

from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
    ChatTTSBackend,
)
from mosaic.nodes.audio.tts_backends.base import TTSBackendSpec
from mosaic.core.types import AudioData

# 需要 transformers 的用例统一标记
_needs_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers 未安装，跳过 ChatTTS 加载用例"
)


def _make_backend(model_path: str = "/tmp/chattts_test") -> ChatTTSBackend:
    """构造一个未加载的 ChatTTSBackend。"""
    return ChatTTSBackend(model_path=model_path)


# ----------------------------------------------------------------------
# T_CHTTS_01~08：类属性 / 规格 / 依赖检查（不需要 transformers）
# ----------------------------------------------------------------------
def test_CHTTS_01() -> None:
    """类属性正确 (name, spec)。"""
    backend = _make_backend()
    assert backend.name == "chattts"
    assert isinstance(backend.spec, TTSBackendSpec)
    assert backend.spec.name == "chattts"


def test_CHTTS_02() -> None:
    """spec 字段完整 (supported_languages, supports_streaming, etc.)。"""
    spec = _make_backend().spec
    assert spec.supported_languages == ["zh", "en"]
    assert spec.supports_streaming is True
    assert spec.supports_voice_clone is True
    assert spec.vocoder_type == "vocos"
    assert spec.acoustic_type == "ar"


def test_CHTTS_03() -> None:
    """spec.default_params 包含 temperature, top_p, top_k。"""
    params = _make_backend().spec.default_params
    assert "temperature" in params
    assert "top_p" in params
    assert "top_k" in params


def test_CHTTS_04() -> None:
    """spec.sample_rate == 24000。"""
    assert _make_backend().spec.sample_rate == 24000


def test_CHTTS_05() -> None:
    """spec.min_gpu_memory_gb == 2.0。"""
    assert _make_backend().spec.min_gpu_memory_gb == 2.0


def test_CHTTS_06() -> None:
    """spec.model_license == 'CC BY-NC 4.0'。"""
    assert _make_backend().spec.model_license == "CC BY-NC 4.0"


def test_CHTTS_07() -> None:
    """list_speakers 返回非空列表。"""
    speakers = _make_backend().list_speakers()
    assert isinstance(speakers, list)
    assert len(speakers) > 0


def test_CHTTS_08() -> None:
    """check_dependencies 返回 bool。"""
    result = ChatTTSBackend.check_dependencies()
    assert isinstance(result, bool)


# ----------------------------------------------------------------------
# T_CHTTS_09~16：加载 / 卸载 / 合成（需要 transformers）
# ----------------------------------------------------------------------
@pytest.fixture
def loaded_chattts(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """加载一个极小参数的 ChatTTS 后端（随机初始化，仅供编排逻辑测试）。

    通过 monkeypatch 将 LlamaARModel 替换为 2 层 / hidden=64 的极小版本，
    避免创建完整的 24 层模型，从而加速并降低内存占用。
    """
    if not _HAS_TRANSFORMERS:
        pytest.skip("transformers 未安装，跳过 ChatTTS 加载用例")
    pytest.importorskip("torch")

    import mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar as llama_ar_mod
    from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import LlamaARModel

    class _TinyLlamaAR(LlamaARModel):
        """极小参数 LlamaARModel，用于加速加载用例。"""

        def __init__(
            self,
            model_path: str,
            num_vq: int = 4,
            num_audio_tokens: int = 1024,
            num_text_tokens: int = 0,
            hidden_size: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            max_position_embeddings: int = 512,
            use_flash_attention: bool = True,
        ) -> None:
            super().__init__(
                model_path=model_path,
                num_vq=num_vq,
                num_audio_tokens=num_audio_tokens,
                num_text_tokens=num_text_tokens,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                max_position_embeddings=max_position_embeddings,
                use_flash_attention=use_flash_attention,
            )

    monkeypatch.setattr(llama_ar_mod, "LlamaARModel", _TinyLlamaAR)

    backend = ChatTTSBackend(model_path=str(tmp_path))
    backend.load(device="cpu", dtype="float32")
    try:
        yield backend
    finally:
        try:
            backend.unload()
        except Exception:  # noqa: BLE001
            pass


@_needs_transformers
def test_CHTTS_09(loaded_chattts: ChatTTSBackend) -> None:
    """load 成功，is_loaded 为 True。"""
    assert loaded_chattts.is_loaded is True


@_needs_transformers
def test_CHTTS_10(loaded_chattts: ChatTTSBackend) -> None:
    """synthesize 返回 AudioData。"""
    audio = loaded_chattts.synthesize("你好世界", max_new_tokens=32)
    assert isinstance(audio, AudioData)
    assert audio.sample_rate == 24000
    assert audio.waveform.shape[-1] > 0


@_needs_transformers
def test_CHTTS_11(loaded_chattts: ChatTTSBackend) -> None:
    """synthesize 空文本抛 ValueError。"""
    with pytest.raises(ValueError):
        loaded_chattts.synthesize("   ", max_new_tokens=4)


@_needs_transformers
def test_CHTTS_12(loaded_chattts: ChatTTSBackend) -> None:
    """synthesize_stream 产出至少一个 chunk。"""
    chunks = list(
        loaded_chattts.synthesize_stream("你好", max_new_tokens=32, chunk_size=2048)
    )
    assert len(chunks) >= 1
    assert all(isinstance(c, AudioData) for c in chunks)


@_needs_transformers
def test_CHTTS_13(loaded_chattts: ChatTTSBackend) -> None:
    """describe 返回 spec。"""
    spec = loaded_chattts.describe()
    assert isinstance(spec, TTSBackendSpec)
    assert spec.name == "chattts"


@_needs_transformers
def test_CHTTS_14(loaded_chattts: ChatTTSBackend) -> None:
    """list_speakers 加载后仍返回非空列表。"""
    speakers = loaded_chattts.list_speakers()
    assert len(speakers) > 0


@_needs_transformers
def test_CHTTS_15(loaded_chattts: ChatTTSBackend) -> None:
    """unload 后 is_loaded 为 False。"""
    loaded_chattts.unload()
    assert loaded_chattts.is_loaded is False


@_needs_transformers
def test_CHTTS_16(loaded_chattts: ChatTTSBackend) -> None:
    """重复 load 幂等（已加载时再次 load 不报错）。"""
    # 已加载状态下再次 load 应直接返回，不抛异常
    loaded_chattts.load(device="cpu", dtype="float32")
    assert loaded_chattts.is_loaded is True
