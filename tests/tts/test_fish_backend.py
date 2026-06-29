"""测试 FishSpeechBackend。

T_FISH_01~14 不依赖 transformers，可在当前环境运行（大部分测试只测类属性
和构造）；T_FISH_15~18 需要 transformers（真实加载），缺失时跳过。加载用例
使用极小参数的 FishLlamaARModel（经 monkeypatch 替换）以加速并避免 OOM，
且不依赖真实预训练权重（随机初始化即可验证管线编排逻辑）。
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 探测可选依赖 transformers（torch 在函数内部按需导入）
try:  # noqa: SIM105
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

from mosaic.nodes.audio.tts_backends.base import TTSBackendSpec
from mosaic.nodes.audio.tts_backends.implementations import fish_backend as _fish_mod
from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
    FishSpeechBackend,
)
from mosaic.core.types import AudioData

# 需要 transformers 的用例统一标记
_needs_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers 未安装，跳过 Fish 加载用例"
)


def _make_backend(model_path: str = "/tmp/test") -> FishSpeechBackend:
    """构造一个未加载的 FishSpeechBackend。"""
    return FishSpeechBackend(model_path=model_path)


# ----------------------------------------------------------------------
# T_FISH_01~14：类属性 / 规格 / 依赖检查 / 构造（不需要 transformers）
# ----------------------------------------------------------------------
def test_FISH_01() -> None:
    """类属性正确 (name, spec)。"""
    backend = _make_backend()
    assert backend.name == "fish"
    assert isinstance(backend.spec, TTSBackendSpec)
    assert backend.spec.name == "fish"


def test_FISH_02() -> None:
    """spec 字段完整 (supported_languages, supports_streaming, etc.)。"""
    spec = _make_backend().spec
    assert spec.supported_languages == ["zh", "en", "ja", "ko"]
    assert spec.supports_streaming is True
    assert spec.supports_voice_clone is True
    assert spec.vocoder_type == "hifi_gan"
    assert spec.acoustic_type == "ar"
    assert spec.min_gpu_memory_gb == 3.0
    assert spec.model_license == "Apache-2.0"
    assert spec.sample_rate == 22050


def test_FISH_03() -> None:
    """spec.default_params 包含 temperature, top_p, top_k, repetition_penalty, max_new_tokens。"""
    params = _make_backend().spec.default_params
    assert "temperature" in params
    assert "top_p" in params
    assert "top_k" in params
    assert "repetition_penalty" in params
    assert "max_new_tokens" in params


def test_FISH_04() -> None:
    """spec.sample_rate == 22050。"""
    assert _make_backend().spec.sample_rate == 22050


def test_FISH_05() -> None:
    """spec.min_gpu_memory_gb == 3.0。"""
    assert _make_backend().spec.min_gpu_memory_gb == 3.0


def test_FISH_06() -> None:
    """spec.model_license == 'Apache-2.0'。"""
    assert _make_backend().spec.model_license == "Apache-2.0"


def test_FISH_07() -> None:
    """list_speakers 返回非空列表。"""
    backend = _make_backend()
    speakers = backend.list_speakers()
    assert isinstance(speakers, list)
    assert len(speakers) > 0


def test_FISH_08() -> None:
    """list_speakers 返回列表类型。"""
    speakers = _make_backend().list_speakers()
    assert isinstance(speakers, list)


def test_FISH_09() -> None:
    """check_dependencies 返回 bool。"""
    result = FishSpeechBackend.check_dependencies()
    assert isinstance(result, bool)


def test_FISH_10() -> None:
    """构造成功。FishSpeechBackend(model_path='/tmp/test') 不报错。"""
    backend = FishSpeechBackend(model_path="/tmp/test")
    assert backend is not None


def test_FISH_11() -> None:
    """构造参数正确存储 (model_path, codec_type, language 等)。"""
    backend = FishSpeechBackend(
        model_path="/tmp/fish_model",
        codec_type="encodec",
        language="en",
    )
    assert backend._model_path == "/tmp/fish_model"
    assert backend._codec_type == "encodec"
    assert backend._language == "en"


def test_FISH_12() -> None:
    """is_loaded 初始为 False。"""
    backend = _make_backend()
    assert backend.is_loaded is False


def test_FISH_13() -> None:
    """_CompositeVocoder 类属性正确 (vocoder_type, input_type, sample_rate)。"""
    CompositeVocoder = _fish_mod._CompositeVocoder
    assert CompositeVocoder.vocoder_type == "hifi_gan"
    assert CompositeVocoder.input_type == "codec_tokens"
    assert CompositeVocoder.sample_rate == 22050


def test_FISH_14() -> None:
    """describe 返回正确信息 (TTSBackendSpec, name == 'fish')。"""
    spec = _make_backend().describe()
    assert isinstance(spec, TTSBackendSpec)
    assert spec.name == "fish"


# ----------------------------------------------------------------------
# T_FISH_15~18：加载 / 卸载 / 合成（需要 transformers）
# ----------------------------------------------------------------------
@pytest.fixture
def loaded_fish(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """加载一个极小参数的 Fish 后端（随机初始化，仅供编排逻辑测试）。

    通过 monkeypatch 将 FishLlamaARModel 替换为 2 层 / hidden=64 的极小版本，
    避免创建完整的 24 层模型，从而加速并降低内存占用。
    """
    if not _HAS_TRANSFORMERS:
        pytest.skip("transformers 未安装，跳过 Fish 加载用例")
    pytest.importorskip("torch")

    import mosaic.nodes.audio.tts_backends.acoustic_models.fish_ar as fish_ar_mod
    from mosaic.nodes.audio.tts_backends.acoustic_models.fish_ar import (
        FishLlamaARModel,
    )

    class _TinyFishAR(FishLlamaARModel):
        """极小参数 FishLlamaARModel，用于加速加载用例。"""

        def __init__(
            self,
            model_path: str,
            text_vocab_size: int = 0,
            audio_vocab_size: int = 0,
            hidden_size: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            max_position_embeddings: int = 512,
            use_flash_attention: bool = True,
            codec_type: str = "dac",
        ) -> None:
            super().__init__(
                model_path=model_path,
                text_vocab_size=text_vocab_size,
                audio_vocab_size=audio_vocab_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                max_position_embeddings=max_position_embeddings,
                use_flash_attention=use_flash_attention,
                codec_type=codec_type,
            )

    monkeypatch.setattr(fish_ar_mod, "FishLlamaARModel", _TinyFishAR)

    backend = FishSpeechBackend(model_path=str(tmp_path))
    backend.load(device="cpu", dtype="float32")
    try:
        yield backend
    finally:
        try:
            backend.unload()
        except Exception:
            pass


@_needs_transformers
def test_FISH_15(loaded_fish: FishSpeechBackend) -> None:
    """load 成功，is_loaded 为 True。"""
    assert loaded_fish.is_loaded is True


@_needs_transformers
def test_FISH_16(loaded_fish: FishSpeechBackend) -> None:
    """synthesize 返回 AudioData。"""
    audio = loaded_fish.synthesize("你好世界", max_new_tokens=32)
    assert isinstance(audio, AudioData)
    assert audio.sample_rate == 22050
    assert audio.waveform.shape[-1] > 0


@_needs_transformers
def test_FISH_17(loaded_fish: FishSpeechBackend) -> None:
    """unload 后 is_loaded 为 False。"""
    loaded_fish.unload()
    assert loaded_fish.is_loaded is False


@_needs_transformers
def test_FISH_18(loaded_fish: FishSpeechBackend) -> None:
    """describe 加载后仍返回正确 spec。"""
    spec = loaded_fish.describe()
    assert isinstance(spec, TTSBackendSpec)
    assert spec.name == "fish"
