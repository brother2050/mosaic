# tests/phase4/test_frame_extractor.py
"""Phase 4 FrameExtractor 节点测试。

测试拆帧节点的四种模式（all / interval / timestamps / keyframe），
以及 VideoData 输入和文件路径输入。
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_frame(width=64, height=64, color=(100, 150, 200)):
    """创建纯色 PIL 帧。"""
    return Image.new("RGB", (width, height), color=color)


class TestFrameExtractor:
    """T_EXTRACT 系列：FrameExtractor 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """创建 FrameExtractor 实例和测试视频。"""
        from mosaic.nodes.video.frame_extractor import FrameExtractor

        self.extractor = FrameExtractor(scheduler=cpu_scheduler)
        self.extractor.load()

        # 创建 20 帧的测试视频
        self.frames = [_make_frame(color=(100 + i * 8, 150, 200)) for i in range(20)]
        self.video = VideoData(frames=self.frames, fps=30)

    # T_EXTRACT_01
    def test_all_mode_extracts_all_frames(self):
        """T_EXTRACT_01：all 模式提取全部帧。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="all",
        ))
        frames = result.get("frames")
        assert frames is not None, "应输出 frames"
        assert len(frames) == 20, f"all 模式应提取全部 20 帧，实际 {len(frames)}"
        assert result.get("frame_count") == 20, "frame_count 应为 20"

    # T_EXTRACT_02
    def test_interval_mode_extracts_at_interval(self):
        """T_EXTRACT_02：interval 模式按间隔提取。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="interval",
            interval=5,
        ))
        frames = result.get("frames")
        assert len(frames) == 4, f"interval=5 应提取 4 帧，实际 {len(frames)}"

    # T_EXTRACT_02 (continued)
    def test_interval_mode_default_interval(self):
        """默认 interval=1 提取全部帧。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="interval",
        ))
        frames = result.get("frames")
        assert len(frames) == 20, "默认 interval=1 应提取全部帧"

    # T_EXTRACT_03
    def test_timestamps_mode_extracts_at_timestamps(self):
        """T_EXTRACT_03：timestamps 模式按时间戳提取。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="timestamps",
            timestamps=[0.0, 0.1, 0.2, 0.5],
        ))
        frames = result.get("frames")
        assert len(frames) == 4, "应提取 4 帧"

    # T_EXTRACT_04
    def test_output_timestamps_list_correct(self):
        """T_EXTRACT_04：输出 timestamps 列表正确。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="all",
        ))
        timestamps = result.get("timestamps")
        assert timestamps is not None, "应包含 timestamps"
        assert len(timestamps) == 20, "timestamps 长度应为 20"
        # 第一个时间戳应为 0.0
        assert abs(timestamps[0] - 0.0) < 0.001, "第一个时间戳应为 0.0"
        # 最后一个时间戳应为 19/30 ≈ 0.633
        expected_last = 19 / 30
        assert abs(timestamps[-1] - expected_last) < 0.001, \
            f"最后一个时间戳应为 {expected_last:.3f}"

    # T_EXTRACT_05
    def test_from_file_path_input(self):
        """T_EXTRACT_05：从文件路径输入提取帧。"""
        # 创建临时视频文件（使用 imageio 兼容格式）
        # 由于我们无法创建真实的视频文件，这里 mock _load_video
        with patch.object(self.extractor.__class__, "_load_video", return_value=self.video):
            result = self.extractor(MosaicData(
                video="/fake/path/video.mp4",
                mode="all",
            ))
            frames = result.get("frames")
            assert len(frames) == 20, "应从文件路径提取全部帧"

    # T_EXTRACT_06
    def test_keyframe_mode_basic(self):
        """T_EXTRACT_06：keyframe 模式基本行为。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="keyframe",
            keyframe_threshold=10.0,
        ))
        frames = result.get("frames")
        assert frames is not None, "应输出 frames"
        assert len(frames) > 0, "至少应提取首帧作为关键帧"
        assert len(frames) <= 20, "关键帧数不应超过总帧数"

    def test_numpy_output_format(self):
        """numpy 输出格式。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="all",
            output_format="numpy",
        ))
        frames = result.get("frames")
        assert len(frames) == 20
        assert isinstance(frames[0], np.ndarray), "应为 numpy 数组"

    def test_pil_output_format(self):
        """pil 输出格式（默认）。"""
        result = self.extractor(MosaicData(
            video=self.video,
            mode="all",
            output_format="pil",
        ))
        frames = result.get("frames")
        assert len(frames) == 20
        assert isinstance(frames[0], Image.Image), "应为 PIL.Image"