# tests/phase3/test_subtitle_aligner.py
"""Phase 3 时间轴对齐节点测试。

测试 SubtitleAligner 节点的基本功能：基本对齐输出 SubtitleData、
alignment_score 在 0-1 范围、time_shift 计算、时长不匹配处理、
不同对齐方法参数。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import AudioData, MosaicData, SubtitleData


# ---------------------------------------------------------------------------
# 辅助：为 SubtitleAligner 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_whisper_pipeline():
    """Mock Whisper pipeline 返回词级时间戳。"""
    import transformers as _tf

    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "text": "Hello welcome to Mosaic framework this is a multimodal AI system",
        "chunks": [
            {"timestamp": (0.0, 0.8), "text": "Hello"},
            {"timestamp": (0.8, 1.8), "text": "welcome"},
            {"timestamp": (1.8, 2.2), "text": "to"},
            {"timestamp": (2.2, 3.2), "text": "Mosaic"},
            {"timestamp": (3.2, 4.0), "text": "framework"},
            {"timestamp": (4.5, 5.5), "text": "this"},
            {"timestamp": (5.5, 6.5), "text": "is"},
            {"timestamp": (6.5, 7.0), "text": "a"},
            {"timestamp": (7.0, 8.0), "text": "multimodal"},
            {"timestamp": (8.0, 9.5), "text": "AI"},
            {"timestamp": (9.5, 11.0), "text": "system"},
        ],
    }

    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.feature_extractor = MagicMock()

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch.object(_tf, "pipeline", return_value=mock_pipe), \
         patch.object(_tf, "AutoModelForSpeechSeq2Seq") as mock_model_cls, \
         patch.object(_tf, "AutoProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = mock_processor
        yield mock_pipe


class TestSubtitleAlignerBasic:
    """T_ALIGN_01：基本对齐测试。"""

    def test_basic_alignment(self, mock_whisper_pipeline, sample_english_subtitle, sample_audio, cpu_scheduler):
        """T_ALIGN_01：基本对齐，输出 SubtitleData。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "对齐输出应包含 subtitle"
        assert isinstance(subtitle, SubtitleData), "subtitle 应为 SubtitleData"
        assert len(subtitle.segments) > 0, "segments 不应为空"


class TestSubtitleAlignerScore:
    """T_ALIGN_02：alignment_score 测试。"""

    def test_alignment_score_range(self, mock_whisper_pipeline, sample_english_subtitle, sample_audio, cpu_scheduler):
        """T_ALIGN_02：alignment_score 在合理范围（0-1）。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))

        score = result.get("alignment_score")
        if score is not None:
            assert 0.0 <= score <= 1.0, f"alignment_score 应在 0-1 之间，实际 {score}"


class TestSubtitleAlignerTimeShift:
    """T_ALIGN_03：time_shift 计算测试。"""

    def test_time_shift_calculated(self, mock_whisper_pipeline, sample_english_subtitle, sample_audio, cpu_scheduler):
        """T_ALIGN_03：time_shift 计算正确。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))

        time_shift = result.get("time_shift")
        assert time_shift is not None, "应输出 time_shift"
        assert isinstance(time_shift, (int, float)), "time_shift 应为数字"


class TestSubtitleAlignerMismatch:
    """T_ALIGN_04：时长不匹配处理测试。"""

    def test_duration_mismatch(self, mock_whisper_pipeline, sample_english_subtitle, sample_long_audio, cpu_scheduler):
        """T_ALIGN_04：字幕与音频时长不匹配时的处理。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        # 字幕时长 21 秒，音频时长 35 秒
        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_long_audio,
        ))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "时长不匹配时仍应返回字幕"


class TestSubtitleAlignerMethods:
    """T_ALIGN_05：不同对齐方法参数测试。"""

    def test_whisper_method(self, mock_whisper_pipeline, sample_english_subtitle, sample_audio, cpu_scheduler):
        """T_ALIGN_05：whisper 对齐方法。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))
        assert result.get("subtitle") is not None

    def test_dtw_method(self, sample_english_subtitle, sample_audio, cpu_scheduler):
        """DTW 对齐方法（不需要模型）。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="dtw", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "DTW 方法应返回字幕"
        assert len(subtitle.segments) > 0, "DTW 方法 segments 不应为空"

    def test_dtw_alignment_score(self, sample_english_subtitle, sample_audio, cpu_scheduler):
        """DTW 方法输出 alignment_score。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="dtw", scheduler=cpu_scheduler)
        result = aligner(MosaicData(
            subtitle=sample_english_subtitle,
            audio=sample_audio,
        ))

        score = result.get("alignment_score")
        assert score is not None, "DTW 应输出 alignment_score"
        assert 0.0 <= score <= 1.0, f"alignment_score 应在 0-1 之间，实际 {score}"


class TestSubtitleAlignerErrors:
    """SubtitleAligner 错误处理测试。"""

    def test_missing_subtitle(self, mock_whisper_pipeline, sample_audio, cpu_scheduler):
        """缺少 subtitle 应抛出异常。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'subtitle'"):
            aligner(MosaicData(audio=sample_audio))

    def test_missing_audio(self, mock_whisper_pipeline, sample_english_subtitle, cpu_scheduler):
        """缺少 audio 应抛出异常。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'audio'"):
            aligner(MosaicData(subtitle=sample_english_subtitle))

    def test_describe(self, mock_whisper_pipeline, cpu_scheduler):
        """describe 返回正确信息。"""
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        aligner = SubtitleAligner(method="whisper", scheduler=cpu_scheduler)
        spec = aligner.describe()

        assert spec.name == "subtitle-aligner", "节点名称应为 'subtitle-aligner'"
        assert spec.domain == "subtitle", "领域应为 'subtitle'"
        assert "subtitle" in spec.input_types, "输入类型应包含 'subtitle'"
        assert "audio" in spec.input_types, "输入类型应包含 'audio'"
        assert "subtitle" in spec.output_types, "输出类型应包含 'subtitle'"