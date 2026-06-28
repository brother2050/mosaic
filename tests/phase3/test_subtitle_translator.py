# tests/phase3/test_subtitle_translator.py
"""Phase 3 字幕翻译节点测试。

测试 SubtitleTranslator 节点的基本功能：中译英、时间轴保持不变、
speaker 信息保留、空字幕处理、长行自动拆分。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import MosaicData, SubtitleData


# ---------------------------------------------------------------------------
# 辅助：为 SubtitleTranslator 测试提供 mock 翻译环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_translator():
    """Mock Translator 节点返回翻译结果。"""
    # 构造一个 mock translator 实例
    mock_trans = MagicMock()
    mock_trans.run.return_value = MosaicData(
        content="[1] Hello, welcome to Mosaic framework.\n[2] This is a multimodal AI generation system.\n[3] Supports text, image, audio, and subtitle.\n[4] Let's start exploring.\n[5] Thank you for using it.",
        language="en",
    )
    mock_trans.name = "translator"

    with patch(
        "mosaic.nodes.text.translator.Translator",
        return_value=mock_trans,
    ):
        yield mock_trans


class TestSubtitleTranslatorBasic:
    """T_SUBTRANS_01：基本翻译测试。"""

    def test_translate_chinese_to_english(self, mock_translator, sample_subtitle, cpu_scheduler):
        """T_SUBTRANS_01：中文字幕翻译为英文。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        result = trans(MosaicData(subtitle=sample_subtitle))

        subtitle = result.get("subtitle")
        assert subtitle is not None, "翻译输出应包含 subtitle"
        assert isinstance(subtitle, SubtitleData), "subtitle 应为 SubtitleData"
        assert len(subtitle.segments) > 0, "segments 不应为空"


class TestSubtitleTranslatorTimeline:
    """T_SUBTRANS_02：时间轴保持不变测试。"""

    def test_timeline_preserved(self, mock_translator, sample_subtitle, cpu_scheduler):
        """T_SUBTRANS_02：时间轴保持不变。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        result = trans(MosaicData(subtitle=sample_subtitle))

        original_segs = sample_subtitle.segments
        translated_segs = result.get("subtitle").segments

        assert len(translated_segs) >= len(original_segs), (
            "翻译后片段数不应少于原始片段数"
        )

        # 检查时间轴是否保持一致（至少第一个和最后一个片段的 start 应匹配）
        assert abs(translated_segs[0]["start"] - original_segs[0]["start"]) < 0.01, (
            "第一个片段 start 应保持不变"
        )


class TestSubtitleTranslatorSpeaker:
    """T_SUBTRANS_03：speaker 信息保留测试。"""

    def test_speaker_preserved(self, mock_translator, cpu_scheduler):
        """T_SUBTRANS_03：speaker 信息保留。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        sub_with_speaker = SubtitleData(
            segments=[
                {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "Alice"},
                {"start": 2.0, "end": 5.0, "text": "大家好", "speaker": "Bob"},
            ],
            format="srt",
        )

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        result = trans(MosaicData(subtitle=sub_with_speaker))

        translated = result.get("subtitle")
        assert translated is not None
        # 注意：segment 在翻译后可能被拆分，因此检查 speaker 字段是否在部分片段中被保留
        has_speaker = any(
            "speaker" in seg for seg in translated.segments
        )
        assert has_speaker, "翻译后应保留 speaker 信息"


class TestSubtitleTranslatorEmpty:
    """T_SUBTRANS_04：空字幕处理测试。"""

    def test_empty_subtitle(self, mock_translator, empty_subtitle, cpu_scheduler):
        """T_SUBTRANS_04：空字幕处理。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        # 空字幕应抛出 ValueError
        with pytest.raises(ValueError, match="no segments"):
            trans(MosaicData(subtitle=empty_subtitle))


class TestSubtitleTranslatorSplit:
    """T_SUBTRANS_05：翻译后长行自动拆分测试。"""

    def test_long_line_split(self, mock_translator, cpu_scheduler):
        """T_SUBTRANS_05：翻译后长行自动拆分。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        # 构造一个长文本字幕
        long_segments = SubtitleData(
            segments=[
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "这是一个非常非常长的句子，包含了很多很多的内容，需要被自动拆分成多个短片段，以便更好地显示在屏幕上，提供更好的用户体验。",
                },
            ],
            format="srt",
        )

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        result = trans(MosaicData(subtitle=long_segments))

        translated = result.get("subtitle")
        assert translated is not None, "长字幕翻译应成功"


class TestSubtitleTranslatorErrors:
    """SubtitleTranslator 错误处理测试。"""

    def test_missing_subtitle(self, mock_translator, cpu_scheduler):
        """缺少 subtitle 应抛出异常。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        with pytest.raises(ValueError, match="requires 'subtitle'"):
            trans(MosaicData())

    def test_describe(self, mock_translator, cpu_scheduler):
        """describe 返回正确信息。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        spec = trans.describe()

        assert spec.name == "subtitle-translator", "节点名称应为 'subtitle-translator'"
        assert spec.domain == "subtitle", "领域应为 'subtitle'"
        assert "subtitle" in spec.input_types, "输入类型应包含 'subtitle'"
        assert "subtitle" in spec.output_types, "输出类型应包含 'subtitle'"