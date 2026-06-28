# tests/phase3/test_asr.py
"""Phase 3 ASR 节点测试。

测试 ASR 节点的基本功能：语音识别输出文本、segments 时间戳、
语言检测、translate 任务、长音频处理、文件路径输入。
"""

from __future__ import annotations

import sys
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
# 辅助：为 ASR 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_asr_pipeline():
    """Mock transformers pipeline 返回伪造的 ASR 结果。"""
    import transformers as _tf

    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "text": "你好，这是一个测试音频。",
        "language": "zh",
        "chunks": [
            {"timestamp": (0.0, 2.5), "text": "你好，"},
            {"timestamp": (2.5, 5.0), "text": "这是一个测试音频。"},
        ],
    }

    # 配置 mock processor 返回包含 tokenizer/feature_extractor 的字典
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.feature_extractor = MagicMock()

    # 配置 mock 模型
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch.object(_tf, "pipeline", return_value=mock_pipe), \
         patch.object(_tf, "AutoModelForSpeechSeq2Seq") as mock_model_cls, \
         patch.object(_tf, "AutoProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = mock_processor
        yield mock_pipe


class TestASRBasic:
    """T_ASR_01-02：基本 ASR 功能测试。"""

    def test_basic_asr_output(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_01：基本语音识别，输出文本非空。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_audio))

        text = result.get("text")
        assert text is not None, "ASR 输出应包含 text"
        assert isinstance(text, str), "text 应为字符串"
        assert len(text) > 0, "text 不应为空"

    def test_segments_with_timestamps(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_02：输出包含 segments（带时间戳）。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_audio))

        segments = result.get("segments")
        assert segments is not None, "ASR 输出应包含 segments"
        assert isinstance(segments, list), "segments 应为列表"
        assert len(segments) > 0, "segments 不应为空"

        for seg in segments:
            assert "start" in seg, "每个 segment 应有 start"
            assert "end" in seg, "每个 segment 应有 end"
            assert "text" in seg, "每个 segment 应有 text"
            assert isinstance(seg["start"], (int, float)), "start 应为数字"
            assert isinstance(seg["end"], (int, float)), "end 应为数字"


class TestASRFeatures:
    """T_ASR_03-05：ASR 功能特性测试。"""

    def test_language_auto_detect(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_03：语言自动检测。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_audio))

        language = result.get("language")
        assert language is not None, "应检测到语言"
        assert isinstance(language, str), "language 应为字符串"

    def test_specified_language(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_04：指定语言参数生效。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(language="zh", scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_audio))

        language = result.get("language")
        assert language == "zh", f"指定语言应为 'zh'，实际 {language}"

    def test_translate_task(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_05：translate 任务输出英文。"""
        from mosaic.nodes.audio.asr import ASR

        # 修改 mock 返回 translate 结果
        mock_asr_pipeline.return_value = {
            "text": "Hello, this is a test audio.",
            "language": "en",
            "chunks": [
                {"timestamp": (0.0, 2.5), "text": "Hello,"},
                {"timestamp": (2.5, 5.0), "text": "this is a test audio."},
            ],
        }

        asr = ASR(task="translate", scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_audio))
        text = result.get("text")
        assert text is not None, "translate 输出不应为空"


class TestASRLongAudio:
    """T_ASR_06：长音频处理测试。"""

    def test_long_audio_handling(self, mock_asr_pipeline, sample_long_audio, cpu_scheduler):
        """T_ASR_06：长音频（>30秒）处理。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        result = asr(MosaicData(audio=sample_long_audio))

        text = result.get("text")
        assert text is not None, "长音频识别应输出文本"
        # 长音频时长应被记录
        assert result.get("duration") > 30.0, "duration 应 > 30 秒"


class TestASRFileInput:
    """T_ASR_07：从文件路径输入测试。"""

    @pytest.mark.skipif(
        not _has_soundfile(),
        reason="soundfile not installed; skip file-based ASR test.",
    )
    def test_from_file_path(self, mock_asr_pipeline, sample_audio, cpu_scheduler):
        """T_ASR_07：从文件路径输入。"""
        import os
        import tempfile

        from mosaic.nodes.audio.asr import ASR

        # 保存音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import soundfile as sf

            sf.write(tmp_path, sample_audio.waveform, sample_audio.sample_rate)

            asr = ASR(scheduler=cpu_scheduler)
            result = asr(MosaicData(audio=tmp_path))

            text = result.get("text")
            assert text is not None, "从文件路径输入应输出识别文本"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class TestASRErrorHandling:
    """ASR 错误处理测试。"""

    def test_missing_audio(self, mock_asr_pipeline, cpu_scheduler):
        """缺少 audio 应抛出 ValueError。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'audio'"):
            asr(MosaicData())

    def test_describe(self, mock_asr_pipeline, cpu_scheduler):
        """describe 返回正确信息。"""
        from mosaic.nodes.audio.asr import ASR

        asr = ASR(scheduler=cpu_scheduler)
        spec = asr.describe()

        assert spec.name == "asr", "节点名称应为 'asr'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "audio" in spec.input_types, "输入类型应包含 'audio'"
        assert "text" in spec.output_types, "输出类型应包含 'text'"