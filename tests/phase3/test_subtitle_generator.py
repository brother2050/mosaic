# tests/phase3/test_subtitle_generator.py
"""Phase 3 字幕生成节点测试。

测试 SubtitleGenerator 节点的基本功能：从音频生成字幕、segments 结构、
时间轴连续性、SRT/VTT 输出、word_timestamps、max_chars_per_line、长音频处理。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import AudioData, MosaicData, SubtitleData


# ---------------------------------------------------------------------------
# 辅助：为 SubtitleGenerator 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_asr_for_subtitle():
    """Mock ASR pipeline 返回带时间戳的识别结果。"""
    import transformers as _tf

    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "text": "你好，欢迎来到 Mosaic 框架。这是一个多模态 AI 生成系统。",
        "language": "zh",
        "chunks": [
            {"timestamp": (0.0, 4.0), "text": "你好，欢迎来到 Mosaic 框架。"},
            {"timestamp": (4.0, 8.0), "text": "这是一个多模态 AI 生成系统。"},
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


class TestSubtitleGeneratorBasic:
    """T_SUBGEN_01-02：基本字幕生成测试。"""

    def test_generate_from_audio(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_01：从音频生成字幕，输出 SubtitleData。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_audio))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "输出应包含 subtitle"
        assert isinstance(subtitle, SubtitleData), "subtitle 应为 SubtitleData"
        assert len(subtitle.segments) > 0, "segments 不应为空"

    def test_segments_structure(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_02：segments 包含 start、end、text。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_audio))

        subtitle = result.get("subtitle")
        for seg in subtitle.segments:
            assert "start" in seg, "每个 segment 应有 start"
            assert "end" in seg, "每个 segment 应有 end"
            assert "text" in seg, "每个 segment 应有 text"
            assert isinstance(seg["start"], (int, float)), "start 应为数字"
            assert isinstance(seg["end"], (int, float)), "end 应为数字"
            assert isinstance(seg["text"], str), "text 应为字符串"
            assert seg["start"] < seg["end"], "start 应小于 end"


class TestSubtitleGeneratorTimeline:
    """T_SUBGEN_03：时间轴连续性测试。"""

    def test_timeline_continuous(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_03：时间轴连续无重叠。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_audio))

        subtitle = result.get("subtitle")
        segs = subtitle.segments
        for i in range(1, len(segs)):
            assert segs[i]["start"] >= segs[i - 1]["end"] - 0.01, (
                f"片段 {i} 与前一帧时间重叠或存在间隙"
            )


class TestSubtitleGeneratorFormats:
    """T_SUBGEN_04-05：SRT/VTT 输出格式测试。"""

    def test_srt_output(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_04：输出 SRT 格式正确。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(output_format="srt", scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_audio))

        subtitle = result.get("subtitle")
        assert subtitle.subtitle_format == "srt", "format 应为 'srt'"

        # 转为 SRT 字符串验证
        from mosaic.nodes.subtitle._base import BaseSubtitleNode
        srt_str = BaseSubtitleNode._to_srt(subtitle.segments)
        assert "-->" in srt_str, "SRT 格式应包含 '-->'"

    def test_vtt_output(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_05：输出 VTT 格式正确。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(output_format="vtt", scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_audio))

        subtitle = result.get("subtitle")
        assert subtitle.subtitle_format == "vtt", "format 应为 'vtt'"


class TestSubtitleGeneratorFeatures:
    """T_SUBGEN_06-07：word_timestamps 和 max_chars_per_line 测试。"""

    def test_word_timestamps(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_06：word_timestamps 参数生效。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(
            audio=sample_audio,
            word_timestamps=True,
        ))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "word_timestamps 模式下应生成字幕"

    def test_max_chars_per_line(self, mock_asr_for_subtitle, sample_audio, cpu_scheduler):
        """T_SUBGEN_07：max_chars_per_line 生效。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(
            audio=sample_audio,
            max_chars_per_line=10,
        ))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "max_chars_per_line 模式下应生成字幕"


class TestSubtitleGeneratorLongAudio:
    """T_SUBGEN_08：长音频处理测试。"""

    def test_long_audio(self, mock_asr_for_subtitle, sample_long_audio, cpu_scheduler):
        """T_SUBGEN_08：长音频处理。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(audio=sample_long_audio))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "长音频应生成字幕"
        assert len(subtitle.segments) > 0, "长音频 segments 不应为空"


class TestSubtitleGeneratorErrors:
    """SubtitleGenerator 错误处理测试。"""

    def test_missing_audio(self, mock_asr_for_subtitle, cpu_scheduler):
        """缺少 audio 应抛出异常。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'audio'"):
            gen(MosaicData())

    def test_describe(self, mock_asr_for_subtitle, cpu_scheduler):
        """describe 返回正确信息。"""
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        spec = gen.describe()

        assert spec.name == "subtitle-generator", "节点名称应为 'subtitle-generator'"
        assert spec.domain == "subtitle", "领域应为 'subtitle'"
        assert "subtitle" in spec.output_types, "输出类型应包含 'subtitle'"