# tests/phase3/test_subtitle_base.py
"""Phase 3 字幕域基类测试。

测试 BaseSubtitleNode 的静态工具方法：_parse_srt、_parse_vtt、_to_srt、
_to_vtt、_format_timestamp、_parse_timestamp、_merge_short_segments、
_split_long_segments。
"""

from __future__ import annotations

import pytest

from mosaic.nodes.subtitle._base import BaseSubtitleNode


class TestParseSRT:
    """T_SBASE_01：_parse_srt 解析测试。"""

    def test_parse_srt(self, sample_srt_content):
        """T_SBASE_01：_parse_srt 解析正确。"""
        segments = BaseSubtitleNode._parse_srt(sample_srt_content)
        assert len(segments) == 5, f"应有 5 个片段，实际 {len(segments)}"
        assert segments[0]["index"] == 1
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 4.0
        assert "欢迎来到" in segments[0]["text"]
        assert segments[-1]["start"] == 17.0
        assert segments[-1]["end"] == 21.0

    def test_parse_srt_empty(self):
        """空字符串返回空列表。"""
        assert BaseSubtitleNode._parse_srt("") == []

    def test_parse_srt_no_index(self):
        """无序号 SRT 也能解析。"""
        srt = "00:00:00,000 --> 00:00:05,000\nHello\n"
        segments = BaseSubtitleNode._parse_srt(srt)
        assert len(segments) == 1
        assert segments[0]["text"] == "Hello"

    def test_parse_srt_multiline_text(self):
        """多行文本 SRT 解析。"""
        srt = "1\n00:00:00,000 --> 00:00:03,000\nLine 1\nLine 2\n\n"
        segments = BaseSubtitleNode._parse_srt(srt)
        assert len(segments) == 1
        assert "Line 1\nLine 2" in segments[0]["text"]


class TestParseVTT:
    """T_SBASE_02：_parse_vtt 解析测试。"""

    def test_parse_vtt(self, sample_vtt_content):
        """T_SBASE_02：_parse_vtt 解析正确。"""
        segments = BaseSubtitleNode._parse_vtt(sample_vtt_content)
        assert len(segments) == 5, f"应有 5 个片段，实际 {len(segments)}"
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 4.0
        assert "welcome" in segments[0]["text"].lower()

    def test_parse_vtt_empty(self):
        """空字符串返回空列表。"""
        assert BaseSubtitleNode._parse_vtt("") == []

    def test_parse_vtt_header_only(self):
        """仅头部返回空列表。"""
        assert BaseSubtitleNode._parse_vtt("WEBVTT\n\n") == []

    def test_parse_vtt_with_cue_id(self):
        """带 cue ID 的 VTT 解析。"""
        vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nTest\n"
        segments = BaseSubtitleNode._parse_vtt(vtt)
        assert len(segments) == 1
        assert segments[0]["text"] == "Test"


class TestToSRT:
    """T_SBASE_03：_to_srt 输出格式测试。"""

    def test_to_srt(self, sample_subtitle):
        """T_SBASE_03：_to_srt 输出格式正确。"""
        srt = BaseSubtitleNode._to_srt(sample_subtitle.segments)
        assert "00:00:01,000 --> 00:00:04,000" in srt, "时间戳格式应为 SRT 格式"
        assert "Mosaic" in srt, "内容应包含原文"
        # 重新解析验证
        parsed = BaseSubtitleNode._parse_srt(srt)
        assert len(parsed) == len(sample_subtitle.segments), "序列化再解析后片段数应一致"

    def test_to_srt_empty(self):
        """空片段返回空字符串。"""
        srt = BaseSubtitleNode._to_srt([])
        assert srt.strip() == "", "空片段应返回空字符串"


class TestToVTT:
    """T_SBASE_04：_to_vtt 输出格式测试。"""

    def test_to_vtt(self, sample_english_subtitle):
        """T_SBASE_04：_to_vtt 输出格式正确。"""
        vtt = BaseSubtitleNode._to_vtt(sample_english_subtitle.segments)
        assert vtt.startswith("WEBVTT"), "应以 WEBVTT 开头"
        assert "00:00:01.000 --> 00:00:04.000" in vtt, "时间戳格式应为 VTT 格式"
        parsed = BaseSubtitleNode._parse_vtt(vtt)
        assert len(parsed) == len(sample_english_subtitle.segments), "序列化再解析后片段数应一致"


class TestFormatTimestamp:
    """T_SBASE_05：_format_timestamp 格式测试。"""

    def test_format_srt(self):
        """T_SBASE_05：SRT 格式时间戳。"""
        ts = BaseSubtitleNode._format_timestamp(3661.5, "srt")  # 1h 1m 1.5s
        assert ts == "01:01:01,500", f"应为 '01:01:01,500'，实际 '{ts}'"

    def test_format_vtt(self):
        """VTT 格式时间戳。"""
        ts = BaseSubtitleNode._format_timestamp(3661.5, "vtt")
        assert ts == "01:01:01.500", f"应为 '01:01:01.500'，实际 '{ts}'"

    def test_format_zero(self):
        """零时间。"""
        assert BaseSubtitleNode._format_timestamp(0.0, "srt") == "00:00:00,000"

    def test_format_negative(self):
        """负时间应转为 0。"""
        assert BaseSubtitleNode._format_timestamp(-5.0, "srt") == "00:00:00,000"


class TestParseTimestamp:
    """T_SBASE_06：_parse_timestamp 解析测试。"""

    def test_parse_srt_timestamp(self):
        """T_SBASE_06：SRT 时间戳解析。"""
        ts = BaseSubtitleNode._parse_timestamp("01:01:01,500")
        assert ts == 3661.5, f"应为 3661.5，实际 {ts}"

    def test_parse_vtt_timestamp(self):
        """VTT 时间戳解析。"""
        ts = BaseSubtitleNode._parse_timestamp("01:01:01.500")
        assert ts == 3661.5, f"应为 3661.5，实际 {ts}"

    def test_parse_short_format(self):
        """短格式 MM:SS。"""
        ts = BaseSubtitleNode._parse_timestamp("01:30.500")
        assert ts == 90.5, f"应为 90.5，实际 {ts}"

    def test_parse_seconds_only(self):
        """仅秒数。"""
        ts = BaseSubtitleNode._parse_timestamp("123.456")
        assert ts == 123.456

    def test_parse_invalid(self):
        """无效格式返回 0.0。"""
        assert BaseSubtitleNode._parse_timestamp("invalid") == 0.0


class TestMergeShortSegments:
    """T_SBASE_07：_merge_short_segments 合并测试。"""

    def test_merge_basic(self):
        """T_SBASE_07：_merge_short_segments 合并正确。"""
        segments = [
            {"start": 0.0, "end": 0.3, "text": "A"},  # 短于 min_duration
            {"start": 0.3, "end": 2.0, "text": "B"},
            {"start": 2.0, "end": 5.0, "text": "C"},
        ]
        merged = BaseSubtitleNode._merge_short_segments(segments, min_duration=0.5)
        # 第一个片段（0.3s < 0.5s）应与第二个合并
        assert len(merged) == 2, f"合并后应有 2 个片段，实际 {len(merged)}"
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 2.0
        assert "A" in merged[0]["text"] and "B" in merged[0]["text"]

    def test_merge_empty(self):
        """空列表返回空列表。"""
        assert BaseSubtitleNode._merge_short_segments([], 0.5) == []

    def test_merge_single(self):
        """单片段保持不变。"""
        segments = [{"start": 0.0, "end": 2.0, "text": "Only"}]
        merged = BaseSubtitleNode._merge_short_segments(segments, 0.5)
        assert len(merged) == 1

    def test_merge_preserve_speaker(self):
        """合并时保留 speaker 信息。"""
        segments = [
            {"start": 0.0, "end": 0.3, "text": "A"},
            {"start": 0.3, "end": 2.0, "text": "B", "speaker": "Alice"},
        ]
        merged = BaseSubtitleNode._merge_short_segments(segments, 0.5)
        assert "speaker" in merged[0], "应保留 speaker 信息"
        assert merged[0]["speaker"] == "Alice"


class TestSplitLongSegments:
    """T_SBASE_08：_split_long_segments 拆分测试。"""

    def test_split_basic(self):
        """T_SBASE_08：_split_long_segments 拆分正确。"""
        long_text = "第一句。第二句。第三句。第四句。" * 10
        segments = [
            {"start": 0.0, "end": 20.0, "text": long_text},
        ]
        split = BaseSubtitleNode._split_long_segments(
            segments, max_duration=10.0, max_chars=100
        )
        assert len(split) > 1, f"长段应被拆分，实际 {len(split)} 段"
        # 时间轴应连续
        for i in range(1, len(split)):
            assert abs(split[i]["start"] - split[i - 1]["end"]) < 0.01, (
                f"片段 {i} 时间轴不连续"
            )
        # 总时间应一致
        total_duration = split[-1]["end"] - split[0]["start"]
        assert abs(total_duration - 20.0) < 0.01, "总时长应保持不变"

    def test_split_no_op(self):
        """不超长的片段不拆分。"""
        segments = [{"start": 0.0, "end": 3.0, "text": "Short text."}]
        split = BaseSubtitleNode._split_long_segments(
            segments, max_duration=10.0, max_chars=100
        )
        assert len(split) == 1, "短片段不应被拆分"

    def test_split_empty(self):
        """空列表返回空列表。"""
        assert BaseSubtitleNode._split_long_segments([], 10.0, 42) == []

    def test_split_hard_cut(self):
        """无标点的长文本按字数硬切。"""
        no_punct = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 无标点
        segments = [
            {"start": 0.0, "end": 20.0, "text": no_punct},
        ]
        split = BaseSubtitleNode._split_long_segments(
            segments, max_duration=10.0, max_chars=10
        )
        assert len(split) > 1, "无标点长文本应按字数硬切"