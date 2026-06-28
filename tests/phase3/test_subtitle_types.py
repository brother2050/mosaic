# tests/phase3/test_subtitle_types.py
"""Phase 3 字幕数据类型测试。

测试 SubtitleData 的创建、序列化、SRT/VTT 格式解析以及空字幕处理。
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.core.types import SubtitleData, data_from_dict
from mosaic.nodes.subtitle._base import BaseSubtitleNode


class TestSubtitleDataCreation:
    """T_SUBTYPE_01：SubtitleData 创建测试。"""

    def test_create_with_segments(self):
        """T_SUBTYPE_01：SubtitleData 创建，包含 segments。"""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 2.0, "end": 5.0, "text": "World"},
        ]
        sub = SubtitleData(segments=segments)
        assert sub.segments == segments, "segments 内容应保持一致"
        assert len(sub.segments) == 2, "应有 2 个片段"
        assert sub.format == "srt", "默认 format 应为 'srt'"
        assert sub.data_type == "subtitle", "data_type 应为 'subtitle'"

    def test_create_with_format(self):
        """指定 format 参数。"""
        sub = SubtitleData(segments=[], format="vtt")
        assert sub.format == "vtt"

    def test_create_with_metadata(self):
        """创建时可以附带 metadata。"""
        sub = SubtitleData(
            segments=[{"start": 0.0, "end": 1.0, "text": "Hi"}],
            metadata={"source": "test.mp4", "language": "zh"},
        )
        assert sub.metadata["source"] == "test.mp4"
        assert sub.metadata["language"] == "zh"

    def test_empty_segments_default(self):
        """默认 segments 应为空列表。"""
        sub = SubtitleData()
        assert sub.segments == [], "默认 segments 应为空列表"
        assert isinstance(sub.segments, list), "segments 应为列表"


class TestSubtitleDataSerialization:
    """T_SUBTYPE_02：SubtitleData 序列化/反序列化测试。"""

    def test_roundtrip(self):
        """T_SUBTYPE_02：SubtitleData 序列化/反序列化。"""
        segments = [
            {"start": 1.0, "end": 4.0, "text": "你好"},
            {"start": 4.0, "end": 8.0, "text": "世界"},
        ]
        sub = SubtitleData(segments=segments, format="srt", metadata={"lang": "zh"})

        d = sub.to_dict()
        assert "__data_type__" in d, "序列化后应包含 __data_type__"
        assert d["__data_type__"] == "subtitle", "data_type 应为 'subtitle'"

        restored = data_from_dict(d)
        assert isinstance(restored, SubtitleData), "反序列化后应为 SubtitleData"
        assert restored.format == "srt", "format 应保留"
        assert len(restored.segments) == 2, "应有 2 个片段"
        assert restored.segments[0]["start"] == 1.0
        assert restored.segments[0]["end"] == 4.0
        assert restored.segments[0]["text"] == "你好"
        assert restored.metadata["lang"] == "zh"

    def test_roundtrip_with_extra_fields(self):
        """带额外字段（如 speaker）的序列化/反序列化。"""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "Alice"},
            {"start": 2.0, "end": 5.0, "text": "Hi", "speaker": "Bob"},
        ]
        sub = SubtitleData(segments=segments)
        d = sub.to_dict()
        restored = data_from_dict(d)
        assert restored.segments[0]["speaker"] == "Alice"
        assert restored.segments[1]["speaker"] == "Bob"


class TestSRTFormatParsing:
    """T_SUBTYPE_03：SRT 格式字符串解析正确。"""

    def test_parse_srt(self, sample_srt_content):
        """T_SUBTYPE_03：SRT 格式字符串解析正确。"""
        segments = BaseSubtitleNode._parse_srt(sample_srt_content)
        assert len(segments) == 5, f"应有 5 个片段，实际 {len(segments)}"
        assert segments[0]["start"] == 1.0, "第一个片段 start 应为 1.0"
        assert segments[0]["end"] == 4.0, "第一个片段 end 应为 4.0"
        assert "欢迎来到 Mosaic" in segments[0]["text"], "文本内容应正确"
        assert segments[-1]["start"] == 17.0, "最后一个片段 start 应为 17.0"
        assert segments[-1]["end"] == 21.0, "最后一个片段 end 应为 21.0"

    def test_parse_srt_empty(self):
        """空 SRT 内容应返回空列表。"""
        segments = BaseSubtitleNode._parse_srt("")
        assert segments == [], "空字符串应返回空列表"

    def test_parse_srt_single_segment(self):
        """单段 SRT 解析。"""
        srt = "1\n00:00:00,000 --> 00:00:05,000\n测试文本\n"
        segments = BaseSubtitleNode._parse_srt(srt)
        assert len(segments) == 1
        assert segments[0]["text"] == "测试文本"


class TestVTTFormatParsing:
    """T_SUBTYPE_04：VTT 格式字符串解析正确。"""

    def test_parse_vtt(self, sample_vtt_content):
        """T_SUBTYPE_04：VTT 格式字符串解析正确。"""
        segments = BaseSubtitleNode._parse_vtt(sample_vtt_content)
        assert len(segments) == 5, f"应有 5 个片段，实际 {len(segments)}"
        assert segments[0]["start"] == 1.0, "第一个片段 start 应为 1.0"
        assert segments[0]["end"] == 4.0, "第一个片段 end 应为 4.0"
        assert "welcome" in segments[0]["text"].lower(), "文本内容应正确"

    def test_parse_vtt_empty(self):
        """空 VTT 内容应返回空列表。"""
        segments = BaseSubtitleNode._parse_vtt("")
        assert segments == [], "空字符串应返回空列表"

    def test_parse_vtt_header_only(self):
        """仅含 WEBVTT 头部的应返回空列表。"""
        segments = BaseSubtitleNode._parse_vtt("WEBVTT\n\n")
        assert segments == []


class TestEmptySubtitle:
    """T_SUBTYPE_05：空字幕的处理。"""

    def test_empty_subtitle_data(self, empty_subtitle):
        """T_SUBTYPE_05：空字幕的处理。"""
        assert empty_subtitle.segments == [], "空字幕 segments 应为空列表"
        assert empty_subtitle.format == "srt"
        assert SubtitleData.validate(empty_subtitle), "空字幕应通过校验"

    def test_empty_subtitle_serialization(self, empty_subtitle):
        """空字幕的序列化/反序列化。"""
        d = empty_subtitle.to_dict()
        restored = data_from_dict(d)
        assert restored.segments == []

    def test_empty_subtitle_dict_access(self, empty_subtitle):
        """空字幕的字典式访问。"""
        assert empty_subtitle["segments"] == []
        assert empty_subtitle["format"] == "srt"


class TestSubtitleDataValidation:
    """SubtitleData 校验测试。"""

    def test_validate_correct(self):
        """正确的 SubtitleData 应通过校验。"""
        sub = SubtitleData(segments=[{"start": 0.0, "end": 1.0, "text": "Test"}])
        assert SubtitleData.validate(sub), "正确 SubtitleData 应通过校验"

    def test_validate_missing_key(self):
        """缺少必需 key 的片段不应通过校验。"""
        sub = SubtitleData(segments=[{"start": 0.0, "end": 1.0}])  # 缺少 text
        assert not SubtitleData.validate(sub), "缺少 text 不应通过校验"

    def test_validate_wrong_type(self):
        """非 SubtitleData 类型不应通过校验。"""
        from mosaic.core.types import TextData

        text = TextData(content="test")
        assert not SubtitleData.validate(text), "TextData 不应通过 SubtitleData 校验"