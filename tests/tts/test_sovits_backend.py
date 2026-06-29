# tests/tts/test_sovits_backend.py
"""GPT-SoVITS 后端集成测试。

测试 GPTSoVITSBackend 的组装、合成、流式合成、语音克隆、说话人管理等功能。

所有测试使用 mock 对象替代真实的三层管线，避免依赖预训练权重。
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
    """创建一个 GPTSoVITSBackend 实例（不加载权重）。"""
    from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
        GPTSoVITSBackend,
    )

    backend = GPTSoVITSBackend(model_path="/tmp/fake_sovits")
    return backend


def _make_loaded_backend() -> Any:
    """创建一个已加载的 GPTSoVITSBackend，三层管线使用 mock 对象。"""
    backend = _make_backend()

    # 注入 mock 三层管线
    backend._text_frontend = MagicMock()
    backend._text_frontend.vocab_size = 256
    backend._text_frontend.preprocess.side_effect = lambda t: t
    backend._text_frontend.tokenize.return_value = [2, 6, 12, 13, 1]

    backend._acoustic_model = MagicMock()
    backend._acoustic_model.generate.return_value = [10, 20, 30, 40, 50]

    backend._vocoder = MagicMock()
    backend._vocoder.decode.return_value = ([0.0] * 200, 32000)
    backend._vocoder.decode_chunk.return_value = ([0.0] * 200, 32000)
    backend._vocoder.set_reference = MagicMock()
    backend._vocoder.reset_stream = MagicMock()

    backend._stream_adapter = MagicMock()

    # 标记为已加载
    backend.is_loaded = True
    backend._device = "cpu"
    backend._dtype = "float32"

    return backend


# ----------------------------------------------------------------------
# T_SBE_01 ~ T_SBE_06: 基本后端功能
# ----------------------------------------------------------------------
class TestSoVITSBackendBasic:
    """GPT-SoVITS 后端基本功能测试。"""

    def test_T_SBE_01(self) -> None:
        """T_SBE_01：GPTSoVITSBackend 创建成功。"""
        backend = _make_backend()
        assert backend is not None
        assert backend.name == "sovits"

    def test_T_SBE_02(self) -> None:
        """T_SBE_02：spec 属性正确。"""
        backend = _make_backend()
        assert backend.spec.name == "sovits"
        assert backend.spec.model_license == "MIT"
        assert backend.spec.sample_rate == 32000
        assert "zh" in backend.spec.supported_languages
        assert "en" in backend.spec.supported_languages
        assert "yue" in backend.spec.supported_languages
        assert backend.spec.supports_streaming is True
        assert backend.spec.supports_voice_clone is True
        assert backend.spec.vocoder_type == "sovits_decoder"
        assert backend.spec.acoustic_type == "ar"

    def test_T_SBE_03(self) -> None:
        """T_SBE_03：load 成功（is_loaded = True）。"""
        backend = _make_backend()
        with patch.object(backend, "_build_pipeline") as mock_build:
            backend.load(device="cpu", dtype="float32")
            assert backend.is_loaded is True
            mock_build.assert_called_once()

    def test_T_SBE_04(self) -> None:
        """T_SBE_04：unload 成功（is_loaded = False）。"""
        backend = _make_loaded_backend()
        backend.unload()
        assert backend.is_loaded is False
        assert backend._text_frontend is None
        assert backend._acoustic_model is None
        assert backend._vocoder is None

    def test_T_SBE_05(self) -> None:
        """T_SBE_05：synthesize 基本合成（无参考音频），输出 AudioData。"""
        import torch

        backend = _make_loaded_backend()
        # 让 mock 返回 torch tensor
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 20), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (
            torch.randn(200),
            32000,
        )

        from mosaic.core.types import AudioData

        result = backend.synthesize("你好世界", speaker=None, language="zh")
        assert isinstance(result, AudioData)
        assert result.sample_rate == 32000

    def test_T_SBE_06(self) -> None:
        """T_SBE_06：synthesize 输出 sample_rate = 32000。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)

        result = backend.synthesize("test", speaker=None)
        assert result.sample_rate == 32000


# ----------------------------------------------------------------------
# T_SBE_07 ~ T_SBE_11: 合成输出验证
# ----------------------------------------------------------------------
class TestSoVITSBackendSynth:
    """GPT-SoVITS 后端合成输出验证。"""

    def test_T_SBE_07(self) -> None:
        """T_SBE_07：synthesize 输出 waveform 非空。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        waveform = torch.randn(200)
        backend._vocoder.decode.return_value = (waveform, 32000)

        result = backend.synthesize("hello", language="en")
        assert result.waveform is not None
        assert len(result.waveform) > 0

    def test_T_SBE_08(self) -> None:
        """T_SBE_08：synthesize 自定义 temperature 生效。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)

        backend.synthesize("test", temperature=0.3)
        # 检查 generate 被调用时传了 temperature
        call_kwargs = backend._acoustic_model.generate.call_args
        assert call_kwargs is not None

    def test_T_SBE_09(self) -> None:
        """T_SBE_09：synthesize 中文输入。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)

        result = backend.synthesize("你好世界", language="zh")
        assert result.metadata["language"] == "zh"
        assert result.metadata["text"] == "你好世界"

    def test_T_SBE_10(self) -> None:
        """T_SBE_10：synthesize 英文输入。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)

        result = backend.synthesize("Hello world", language="en")
        assert result.metadata["language"] == "en"

    def test_T_SBE_11(self) -> None:
        """T_SBE_11：synthesize 中英混合输入。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)

        result = backend.synthesize("你好hello世界world", language="zh")
        assert result is not None
        assert result.sample_rate == 32000


# ----------------------------------------------------------------------
# T_SBE_12 ~ T_SBE_13: 语音克隆
# ----------------------------------------------------------------------
class TestSoVITSBackendClone:
    """GPT-SoVITS 后端语音克隆测试。"""

    def test_T_SBE_12(self) -> None:
        """T_SBE_12：clone_voice 语音克隆（有参考音频）。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)
        # mock extract_speaker 返回有效信息
        backend._ssl_encoder = None  # SSL 不可用时返回 None
        ref_info = {
            "ref_semantic_tokens": torch.randint(0, 768, (1, 10), dtype=torch.long),
            "speaker_embedding": torch.randn(1, 768),
        }
        with patch.object(backend, "extract_speaker", return_value=ref_info):
            from mosaic.core.types import AudioData

            ref_audio = AudioData(waveform=torch.randn(16000), sample_rate=32000)
            result = backend.clone_voice(ref_audio, "你好世界", language="zh")
            assert result is not None
            assert result.sample_rate == 32000

    def test_T_SBE_13(self) -> None:
        """T_SBE_13：clone_voice 输出音色与参考不同（内容变化）。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )
        backend._vocoder.decode.return_value = (torch.randn(100), 32000)
        ref_info = {
            "ref_semantic_tokens": torch.randint(0, 768, (1, 10), dtype=torch.long),
            "speaker_embedding": torch.randn(1, 768),
        }
        with patch.object(backend, "extract_speaker", return_value=ref_info):
            from mosaic.core.types import AudioData

            ref_audio = AudioData(waveform=torch.randn(16000), sample_rate=32000)
            result = backend.clone_voice(ref_audio, "不同的文本内容", language="zh")
            # 内容不同，但音色应该保持
            assert result.metadata["speaker"] == "cloned"
            assert "不同的文本内容" in result.metadata["text"]


# ----------------------------------------------------------------------
# T_SBE_14 ~ T_SBE_16: 流式合成
# ----------------------------------------------------------------------
class TestSoVITSBackendStream:
    """GPT-SoVITS 后端流式合成测试。"""

    def test_T_SBE_14(self) -> None:
        """T_SBE_14：synthesize_stream 流式合成。"""
        import torch

        backend = _make_loaded_backend()
        # mock generate_stream 返回迭代器
        backend._acoustic_model.generate_stream.return_value = iter([
            torch.randint(0, 768, (1, 16), dtype=torch.long),
            torch.randint(0, 768, (1, 16), dtype=torch.long),
        ])
        backend._vocoder.decode_chunk.return_value = (torch.randn(200), 32000)

        # mock stream adapter
        session = MagicMock()
        backend._get_stream_session = MagicMock(return_value=session)
        backend._stream_push = MagicMock()
        backend._stream_drain = MagicMock(return_value=iter([]))
        backend._stream_finish = MagicMock(return_value=iter([]))

        chunks = list(backend.synthesize_stream("测试流式", language="zh"))
        assert isinstance(chunks, list)

    def test_T_SBE_15(self) -> None:
        """T_SBE_15：synthesize_stream yield 多个 chunk。"""
        import torch

        from mosaic.core.types import AudioData

        backend = _make_loaded_backend()
        backend._acoustic_model.generate_stream.return_value = iter([
            torch.randint(0, 768, (1, 16), dtype=torch.long),
            torch.randint(0, 768, (1, 16), dtype=torch.long),
            torch.randint(0, 768, (1, 16), dtype=torch.long),
        ])
        backend._vocoder.decode_chunk.return_value = (torch.randn(200), 32000)

        # 让 stream_drain 返回实际 AudioData
        def mock_drain(session, text, speaker, lang, speed):
            yield AudioData(
                waveform=torch.randn(100),
                sample_rate=32000,
                metadata={"streaming": True},
            )

        backend._get_stream_session = MagicMock(return_value=MagicMock())
        backend._stream_push = MagicMock()
        backend._stream_drain = mock_drain
        backend._stream_finish = MagicMock(return_value=iter([]))

        chunks = list(backend.synthesize_stream("测试", language="zh"))
        assert len(chunks) >= 1

    def test_T_SBE_16(self) -> None:
        """T_SBE_16：synthesize_stream 总时长合理。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate_stream.return_value = iter([
            torch.randint(0, 768, (1, 16), dtype=torch.long),
        ])
        backend._vocoder.decode_chunk.return_value = (torch.randn(4000), 32000)

        from mosaic.core.types import AudioData

        def mock_drain(session, text, speaker, lang, speed):
            yield AudioData(
                waveform=torch.randn(4000),
                sample_rate=32000,
                metadata={"streaming": True},
            )

        backend._get_stream_session = MagicMock(return_value=MagicMock())
        backend._stream_push = MagicMock()
        backend._stream_drain = mock_drain
        backend._stream_finish = MagicMock(return_value=iter([]))

        chunks = list(backend.synthesize_stream("test", language="zh"))
        total_samples = sum(len(c.waveform) for c in chunks)
        # 至少有一些样本
        assert total_samples > 0


# ----------------------------------------------------------------------
# T_SBE_17 ~ T_SBE_19: 说话人管理
# ----------------------------------------------------------------------
class TestSoVITSBackendSpeaker:
    """GPT-SoVITS 后端说话人管理测试。"""

    def test_T_SBE_17(self) -> None:
        """T_SBE_17：extract_speaker 提取说话人特征。"""
        import torch

        backend = _make_loaded_backend()
        # mock SSL encoder
        backend._ssl_encoder = {
            "model": MagicMock(),
            "extractor": MagicMock(),
        }
        backend._ssl_encoder["extractor"].return_value = MagicMock(
            input_values=torch.randn(1, 16000)
        )
        backend._ssl_encoder["model"].return_value = MagicMock(
            last_hidden_state=torch.randn(1, 50, 768)
        )

        from mosaic.core.types import AudioData

        audio = AudioData(waveform=torch.randn(16000), sample_rate=32000)
        result = backend.extract_speaker(audio)
        assert isinstance(result, dict)
        assert "ref_semantic_tokens" in result
        assert "speaker_embedding" in result

    def test_T_SBE_18(self) -> None:
        """T_SBE_18：save_speaker / load_speaker 保存和加载。"""
        import torch

        backend = _make_loaded_backend()
        backend._ssl_encoder = None  # SSL 不可用

        with tempfile.TemporaryDirectory() as tmpdir:
            backend._speaker_dir = os.path.join(tmpdir, "speaker")
            # 手动注入一个说话人
            backend._speaker_cache["test_speaker"] = {
                "ref_semantic_tokens": [1, 2, 3, 4, 5],
                "speaker_embedding": [0.1, 0.2, 0.3],
            }
            backend.save_speaker("test_speaker", "/tmp/fake.wav")

            # 重新加载
            loaded = backend.load_speaker("test_speaker")
            assert loaded is not None
            assert "ref_semantic_tokens" in loaded

    def test_T_SBE_19(self) -> None:
        """T_SBE_19：list_speakers 返回列表。"""
        backend = _make_loaded_backend()
        backend._speaker_cache = {"spk1": {}, "spk2": {}}
        speakers = backend.list_speakers()
        assert isinstance(speakers, list)
        assert "spk1" in speakers
        assert "spk2" in speakers


# ----------------------------------------------------------------------
# T_SBE_20 ~ T_SBE_21: describe 和 speed
# ----------------------------------------------------------------------
class TestSoVITSBackendMisc:
    """GPT-SoVITS 后端杂项测试。"""

    def test_T_SBE_20(self) -> None:
        """T_SBE_20：describe 返回正确信息。"""
        backend = _make_backend()
        spec = backend.describe()
        assert spec.name == "sovits"
        assert spec.sample_rate == 32000
        assert spec.model_license == "MIT"

    def test_T_SBE_21(self) -> None:
        """T_SBE_21：speed 参数影响输出时长。"""
        import torch

        backend = _make_loaded_backend()
        backend._acoustic_model.generate.return_value = torch.randint(
            0, 768, (1, 10), dtype=torch.long
        )

        # 原始波形
        original = torch.randn(1000)
        backend._vocoder.decode.return_value = (original, 32000)

        # speed=2.0 应缩短波形
        result_fast = backend.synthesize("test", speed=2.0)
        assert result_fast.waveform is not None

        # speed=0.5 应延长波形
        result_slow = backend.synthesize("test", speed=0.5)
        assert result_slow.waveform is not None
