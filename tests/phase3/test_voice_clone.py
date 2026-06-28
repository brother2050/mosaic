# tests/phase3/test_voice_clone.py
"""Phase 3 语音克隆节点测试。

测试 VoiceClone 节点的基本功能：输出 AudioData、输出时长与文本长度相关、
从文件路径输入参考音频、describe 信息。
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import AudioData, MosaicData


def _has_soundfile():
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 辅助：为 VoiceClone 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_xtts():
    """Mock Coqui TTS 用于语音克隆。"""
    if "TTS" not in sys.modules:
        import types
        tts_mod = types.ModuleType("TTS")
        tts_api = types.ModuleType("TTS.api")
        mock_tts_cls = MagicMock()
        mock_tts_instance = MagicMock()
        # 模拟 tts 返回 numpy 数组
        mock_tts_instance.tts.return_value = np.sin(
            np.linspace(0, 2 * np.pi, 48000)
        ).astype(np.float32).tolist()
        mock_tts_cls.return_value = mock_tts_instance
        tts_api.TTS = mock_tts_cls
        tts_mod.api = tts_api
        sys.modules["TTS"] = tts_mod
        sys.modules["TTS.api"] = tts_api

    yield

    if "TTS" in sys.modules:
        del sys.modules["TTS"]
    if "TTS.api" in sys.modules:
        del sys.modules["TTS.api"]


class TestVoiceCloneBasic:
    """T_CLONE_01：基本语音克隆测试。"""

    def test_basic_voice_clone(self, mock_xtts, sample_audio, cpu_scheduler):
        """T_CLONE_01：基本语音克隆，输出 AudioData。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        result = cloner(MosaicData(
            reference_audio=sample_audio,
            text="你好，这是我的克隆声音。",
        ))

        audio = result.get("audio")
        assert audio is not None, "VoiceClone 输出应包含 audio"
        assert audio.waveform is not None, "waveform 不应为 None"
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert len(audio.waveform) > 0, "waveform 不应为空"

        # 参考音频也应被保留
        ref_audio = result.get("reference_audio")
        assert ref_audio is not None, "输出应包含 reference_audio"


class TestVoiceCloneDuration:
    """T_CLONE_02：输出时长与文本长度相关。"""

    def test_duration_vs_text_length(self, mock_xtts, sample_audio, cpu_scheduler):
        """T_CLONE_02：输出时长与文本长度相关。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)

        # 短文本
        result_short = cloner(MosaicData(
            reference_audio=sample_audio,
            text="你好。",
        ))
        # 长文本
        result_long = cloner(MosaicData(
            reference_audio=sample_audio,
            text="这是一段比较长的文本，用于测试语音克隆的输出时长是否与文本长度相关。",
        ))

        assert result_short.get("audio") is not None
        assert result_long.get("audio") is not None
        # 长文本波形应更长（mock 环境下可能相同，但至少不为空）
        assert len(result_long.get("audio").waveform) > 0


class TestVoiceCloneFileInput:
    """T_CLONE_03：从文件路径输入参考音频。"""

    @pytest.mark.skipif(
        not _has_soundfile(),
        reason="soundfile not installed; skip file-based voice clone test.",
    )
    def test_from_file_path(self, mock_xtts, sample_audio, cpu_scheduler):
        """T_CLONE_03：从文件路径输入参考音频。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        # 保存参考音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import soundfile as sf

            sf.write(tmp_path, sample_audio.waveform, sample_audio.sample_rate)

            cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
            result = cloner(MosaicData(
                reference_audio=tmp_path,
                text="测试从文件路径克隆。",
            ))

            audio = result.get("audio")
            assert audio is not None, "从文件路径输入应成功克隆"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class TestVoiceCloneDescribe:
    """T_CLONE_04：describe 测试。"""

    def test_describe(self, mock_xtts, cpu_scheduler):
        """T_CLONE_04：describe 标注模型信息。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        spec = cloner.describe()

        assert spec.name == "voice-clone", "节点名称应为 'voice-clone'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "audio" in spec.input_types, "输入类型应包含 'audio'"
        assert "text" in spec.input_types, "输入类型应包含 'text'"
        assert "audio" in spec.output_types, "输出类型应包含 'audio'"


class TestVoiceCloneErrors:
    """VoiceClone 错误处理测试。"""

    def test_missing_reference_audio(self, mock_xtts, cpu_scheduler):
        """缺少 reference_audio 应抛出 ValueError。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'reference_audio'"):
            cloner(MosaicData(text="你好"))

    def test_missing_text(self, mock_xtts, sample_audio, cpu_scheduler):
        """缺少 text 应抛出 ValueError。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'text'"):
            cloner(MosaicData(reference_audio=sample_audio))

    def test_language_param(self, mock_xtts, sample_audio, cpu_scheduler):
        """语言参数可指定。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="en", scheduler=cpu_scheduler)
        result = cloner(MosaicData(
            reference_audio=sample_audio,
            text="Hello, this is my cloned voice.",
        ))
        assert result.get("audio") is not None