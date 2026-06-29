# tests/tts/test_streaming_comparison.py
"""四后端流式合成接口对比测试。

由于无法在测试环境中实际运行全部四个后端，本测试聚焦于验证流式接口
的**存在性与兼容性**：

- AR 后端（ChatTTS / Fish / GPT-SoVITS）与 Flow Matching 后端
  （CosyVoice）均提供 ``synthesize_stream`` 方法；
- CosyVoice mock 后端的流式与非流式输出采样率一致（24000Hz）；
- ``chunk_size`` 参数可被 ``synthesize_stream`` 接受而不报错。

``torch`` 仅在函数内部局部导入，不在模块级引入。
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/workspace/mosaic")

_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _make_loaded_cosyvoice() -> Any:
    """创建一个已加载的 CosyVoiceBackend，组件使用 mock 对象。

    ``torch`` 在本函数内部局部导入，不在模块级引入。
    """
    from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
        CosyVoiceBackend,
    )

    backend = CosyVoiceBackend(model_path="/tmp/fake_cosyvoice")

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
# T_STRCOMP_01 ~ T_STRCOMP_02: 流式接口存在性
# ----------------------------------------------------------------------
class TestStreamingInterface:
    """四后端流式接口存在性测试。"""

    def test_T_STRCOMP_01(self) -> None:
        """T_STRCOMP_01：AR 后端均具备 synthesize_stream 方法。"""
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        assert hasattr(ChatTTSBackend, "synthesize_stream")
        assert hasattr(FishSpeechBackend, "synthesize_stream")
        assert hasattr(GPTSoVITSBackend, "synthesize_stream")

    def test_T_STRCOMP_02(self) -> None:
        """T_STRCOMP_02：CosyVoice 后端具备 synthesize_stream 方法。"""
        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )

        assert hasattr(CosyVoiceBackend, "synthesize_stream")


# ----------------------------------------------------------------------
# T_STRCOMP_03 ~ T_STRCOMP_05: 流式输出兼容性
# ----------------------------------------------------------------------
class TestStreamingOutput:
    """CosyVoice mock 后端流式输出兼容性测试。"""

    def test_T_STRCOMP_03(self) -> None:
        """T_STRCOMP_03：流式合成 yield AudioData 对象。"""
        from mosaic.core.types import AudioData

        backend = _make_loaded_cosyvoice()
        _setup_stream_mocks(backend, num_chunks=3)

        chunks = list(backend.synthesize_stream("测试流式", language="zh"))
        assert len(chunks) > 0
        assert all(isinstance(c, AudioData) for c in chunks)

    def test_T_STRCOMP_04(self) -> None:
        """T_STRCOMP_04：流式与非流式输出采样率一致（24000Hz）。"""
        backend = _make_loaded_cosyvoice()
        _setup_stream_mocks(backend, num_chunks=3)

        non_stream = backend.synthesize("hello", language="zh")
        stream_chunks = list(backend.synthesize_stream("hello", language="zh"))

        assert non_stream.sample_rate == 24000
        assert len(stream_chunks) > 0
        for chunk in stream_chunks:
            assert chunk.sample_rate == 24000

    def test_T_STRCOMP_05(self) -> None:
        """T_STRCOMP_05：chunk_size 参数被 synthesize_stream 接受（不报错）。"""
        backend = _make_loaded_cosyvoice()
        _setup_stream_mocks(backend, num_chunks=3)

        chunks = list(
            backend.synthesize_stream("test chunk size", language="zh", chunk_size=2048)
        )
        assert isinstance(chunks, list)
        assert len(chunks) > 0
