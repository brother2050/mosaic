# tests/tts/test_cosyvoice_backend.py
"""CosyVoice 后端集成测试。

测试 CosyVoiceBackend 的组装、合成、流式合成、语音克隆、说话人管理、
ODE 参数调节与依赖检查等功能。

CosyVoice 与自回归后端的根本差异在于声学模型采用 Flow Matching（非自回归），
通过 ODE 求解从高斯噪声一次性生成完整 mel spectrogram，再经 HiFi-GAN 解码为
24000Hz 单声道波形。所有测试使用 mock 对象替代真实管线，避免依赖预训练权重。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _make_backend() -> Any:
    """创建一个 CosyVoiceBackend 实例（不加载权重）。"""
    from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
        CosyVoiceBackend,
    )

    backend = CosyVoiceBackend(model_path="/tmp/fake_cosyvoice")
    return backend


def _make_loaded_backend() -> Any:
    """创建一个已加载的 CosyVoiceBackend，组件使用 mock 对象。

    注入 mock 的文本前端、Flow Matching 声学模型、HiFi-GAN 声码器、
    语音 Tokenizer、说话人编码器与流式适配器，并标记为已加载。
    ``torch`` 在本函数内部局部导入，不在模块级引入。
    """
    from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
        CosyVoiceBackend,
    )

    backend = CosyVoiceBackend(model_path="/tmp/fake_cosyvoice")

    # 注入 mock 组件
    backend._text_frontend = MagicMock()
    backend._text_frontend.preprocess.side_effect = lambda t: t
    backend._text_frontend.tokenize.return_value = [100, 200, 300, 400]

    backend._acoustic_model = MagicMock()
    import torch
    backend._acoustic_model.generate.return_value = torch.randn(1, 80, 100)
    backend._acoustic_model.generate_stream.return_value = iter(
        [torch.randn(1, 80, 30) for _ in range(3)]
    )

    backend._vocoder = MagicMock()
    backend._vocoder.decode.return_value = (torch.randn(24000), 24000)
    backend._vocoder.decode_chunk.return_value = (torch.randn(4096), 24000)

    backend._speech_tokenizer = MagicMock()
    backend._speech_tokenizer.encode.return_value = torch.randint(0, 6561, (1, 30))

    backend._speaker_encoder = MagicMock()
    backend._speaker_encoder.encode.return_value = torch.randn(1, 512)

    backend._stream_adapter = MagicMock()
    backend._llm = None
    backend.is_loaded = True
    backend._device = "cpu"
    backend._dtype = "float32"
    return backend


def _setup_stream_mocks(backend: Any, num_chunks: int = 3) -> None:
    """为流式合成测试注入安全的 drain/finish mock，避免 MagicMock 无限 pop。

    ``torch`` 在本函数内部局部导入。
    """
    import torch

    from mosaic.core.types import AudioData

    backend._acoustic_model.generate_stream.return_value = iter(
        [torch.randn(1, 80, 30) for _ in range(num_chunks)]
    )
    backend._vocoder.decode.return_value = (torch.randn(4096), 24000)

    def mock_drain(
        session: Any, text: str, speaker: Any, lang: str, speed: float
    ) -> Any:
        waveform = torch.randn(4096)
        yield AudioData(
            waveform=waveform,
            sample_rate=24000,
            metadata={"streaming": True, "duration": 4096 / 24000},
        )

    backend._get_stream_session = MagicMock(return_value=MagicMock())
    backend._stream_push = MagicMock()
    backend._stream_drain = mock_drain
    backend._stream_finish = MagicMock(return_value=iter([]))


# ----------------------------------------------------------------------
# T_CVBE_01 ~ T_CVBE_06: 基本后端功能
# ----------------------------------------------------------------------
class TestCosyVoiceBackendBasic:
    """CosyVoice 后端基本功能测试。"""

    def test_T_CVBE_01(self) -> None:
        """T_CVBE_01：CosyVoiceBackend 创建成功。"""
        backend = _make_backend()
        assert backend is not None
        assert backend.name == "cosyvoice"

    def test_T_CVBE_02(self) -> None:
        """T_CVBE_02：spec 属性正确。"""
        backend = _make_backend()
        assert backend.spec.name == "cosyvoice"
        assert backend.spec.acoustic_type == "flow_matching"
        assert backend.spec.sample_rate == 24000
        assert backend.spec.model_license == "Apache-2.0"
        assert backend.spec.supports_streaming is True
        assert backend.spec.supports_voice_clone is True
        assert backend.spec.min_gpu_memory_gb == 4.0

    def test_T_CVBE_03(self) -> None:
        """T_CVBE_03：load 成功（is_loaded = True）。

        真实加载需要模型权重与 LLM；此处通过 patch ``_build_pipeline`` 验证
        ``load()`` 生命周期方法正确切换 ``is_loaded`` 状态，mock 注入模式可用。
        """
        backend = _make_backend()
        with patch.object(backend, "_build_pipeline") as mock_build:
            backend.load(device="cpu", dtype="float32")
            assert backend.is_loaded is True
            mock_build.assert_called_once()

    def test_T_CVBE_04(self) -> None:
        """T_CVBE_04：unload 成功（is_loaded = False）。"""
        backend = _make_loaded_backend()
        backend.unload()
        assert backend.is_loaded is False
        assert backend._text_frontend is None
        assert backend._acoustic_model is None
        assert backend._vocoder is None

    def test_T_CVBE_05(self) -> None:
        """T_CVBE_05：synthesize 基本合成（无参考音频），输出 AudioData。"""
        from mosaic.core.types import AudioData

        backend = _make_loaded_backend()
        result = backend.synthesize("hello", speaker=None, language="zh")
        assert isinstance(result, AudioData)

    def test_T_CVBE_06(self) -> None:
        """T_CVBE_06：synthesize 输出 sample_rate = 24000。"""
        backend = _make_loaded_backend()
        result = backend.synthesize("hello", speaker=None, language="zh")
        assert result.sample_rate == 24000


# ----------------------------------------------------------------------
# T_CVBE_07 ~ T_CVBE_11: 合成输出验证
# ----------------------------------------------------------------------
class TestCosyVoiceBackendSynth:
    """CosyVoice 后端合成输出验证。"""

    def test_T_CVBE_07(self) -> None:
        """T_CVBE_07：synthesize 输出 waveform 非空。"""
        import torch

        backend = _make_loaded_backend()
        waveform = torch.randn(24000)
        backend._vocoder.decode.return_value = (waveform, 24000)

        result = backend.synthesize("hello", language="en")
        assert result.waveform is not None
        assert len(result.waveform) > 0

    def test_T_CVBE_08(self) -> None:
        """T_CVBE_08：synthesize 中文输入。"""
        backend = _make_loaded_backend()
        result = backend.synthesize("你好世界", language="zh")
        assert result.metadata["language"] == "zh"
        assert result.metadata["text"] == "你好世界"

    def test_T_CVBE_09(self) -> None:
        """T_CVBE_09：synthesize 英文输入。"""
        backend = _make_loaded_backend()
        result = backend.synthesize("hello world", language="en")
        assert result.metadata["language"] == "en"

    def test_T_CVBE_10(self) -> None:
        """T_CVBE_10：synthesize 中英混合输入。"""
        backend = _make_loaded_backend()
        result = backend.synthesize("你好hello世界", language="zh")
        assert result is not None
        assert result.sample_rate == 24000

    def test_T_CVBE_11(self) -> None:
        """T_CVBE_11：synthesize 自定义 num_ode_steps 生效，传至 acoustic_model.generate。"""
        backend = _make_loaded_backend()
        backend.synthesize("test", speaker=None, language="zh", num_ode_steps=5)
        call_kwargs = backend._acoustic_model.generate.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("num_ode_steps") == 5


# ----------------------------------------------------------------------
# T_CVBE_12 ~ T_CVBE_13: 语音克隆
# ----------------------------------------------------------------------
class TestCosyVoiceBackendClone:
    """CosyVoice 后端语音克隆测试。"""

    def test_T_CVBE_12(self) -> None:
        """T_CVBE_12：clone_voice 语音克隆（有参考音频）。"""
        from mosaic.core.types import AudioData

        backend = _make_loaded_backend()
        ref_info = {
            "ref_speech_tokens": None,
            "speaker_embedding": None,
        }
        with patch.object(backend, "extract_speaker", return_value=ref_info):
            result = backend.clone_voice("fake_audio", "hello", language="zh")
            assert isinstance(result, AudioData)
            assert result.sample_rate == 24000

    def test_T_CVBE_13(self) -> None:
        """T_CVBE_13：clone_voice 输出 metadata 含 backend 字段。"""
        backend = _make_loaded_backend()
        ref_info = {
            "ref_speech_tokens": None,
            "speaker_embedding": None,
        }
        with patch.object(backend, "extract_speaker", return_value=ref_info):
            result = backend.clone_voice("fake_audio", "hello", language="zh")
            assert result.metadata["backend"] == "cosyvoice"


# ----------------------------------------------------------------------
# T_CVBE_14 ~ T_CVBE_16: 流式合成
# ----------------------------------------------------------------------
class TestCosyVoiceBackendStream:
    """CosyVoice 后端流式合成测试。"""

    def test_T_CVBE_14(self) -> None:
        """T_CVBE_14：synthesize_stream 流式合成，yield AudioData。"""
        from mosaic.core.types import AudioData

        backend = _make_loaded_backend()
        _setup_stream_mocks(backend, num_chunks=3)

        chunks = list(backend.synthesize_stream("测试流式", language="zh"))
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, AudioData) for c in chunks)

    def test_T_CVBE_15(self) -> None:
        """T_CVBE_15：synthesize_stream yield 多个 chunk（>= 2）。"""
        backend = _make_loaded_backend()
        _setup_stream_mocks(backend, num_chunks=3)

        chunks = list(backend.synthesize_stream("测试多个chunk", language="zh"))
        assert len(chunks) >= 2

    def test_T_CVBE_16(self) -> None:
        """T_CVBE_16：synthesize_stream 总时长合理（> 0）。"""
        backend = _make_loaded_backend()
        _setup_stream_mocks(backend, num_chunks=3)

        chunks = list(backend.synthesize_stream("test streaming", language="zh"))
        total_duration = sum(c.metadata.get("duration", 0.0) for c in chunks)
        assert total_duration > 0


# ----------------------------------------------------------------------
# T_CVBE_17 ~ T_CVBE_20: 说话人管理与 ODE 参数
# ----------------------------------------------------------------------
class TestCosyVoiceBackendSpeaker:
    """CosyVoice 后端说话人管理与 ODE 参数测试。"""

    def test_T_CVBE_17(self) -> None:
        """T_CVBE_17：extract_speaker 返回含 ref_speech_tokens 与 speaker_embedding 的字典。"""
        import torch

        from mosaic.core.types import AudioData

        backend = _make_loaded_backend()
        audio = AudioData(waveform=torch.randn(24000), sample_rate=24000)
        result = backend.extract_speaker(audio)
        assert isinstance(result, dict)
        assert "ref_speech_tokens" in result
        assert "speaker_embedding" in result

    def test_T_CVBE_18(self) -> None:
        """T_CVBE_18：save_speaker / load_speaker 保存和加载。"""
        backend = _make_loaded_backend()
        with tempfile.TemporaryDirectory() as tmpdir:
            backend._speaker_dir = os.path.join(tmpdir, "speaker")
            backend.save_speaker("test_speaker", "/tmp/fake.wav")
            loaded = backend.load_speaker("test_speaker")
            assert isinstance(loaded, dict)

    def test_T_CVBE_19(self) -> None:
        """T_CVBE_19：set_ode_params 修改 ODE 步数与求解器。"""
        backend = _make_loaded_backend()
        backend.set_ode_params(20, "midpoint")
        assert backend._num_ode_steps == 20
        assert backend._ode_solver == "midpoint"

    def test_T_CVBE_20(self) -> None:
        """T_CVBE_20：list_speakers 返回列表。"""
        backend = _make_loaded_backend()
        backend._speaker_cache = {"spk1": {}, "spk2": {}}
        speakers = backend.list_speakers()
        assert isinstance(speakers, list)
        assert "spk1" in speakers
        assert "spk2" in speakers


# ----------------------------------------------------------------------
# T_CVBE_21: 依赖检查
# ----------------------------------------------------------------------
class TestCosyVoiceBackendMisc:
    """CosyVoice 后端杂项测试。"""

    def test_T_CVBE_21(self) -> None:
        """T_CVBE_21：check_dependencies 类方法返回 bool。"""
        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )

        result = CosyVoiceBackend.check_dependencies()
        assert isinstance(result, bool)
