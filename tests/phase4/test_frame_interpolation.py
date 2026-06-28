# tests/phase4/test_frame_interpolation.py
"""Phase 4 FrameInterpolator 节点测试。

测试 linear 模式（无需模型），以及 target_fps、scale_factor 等参数。
RIFE/FILM 模式需要 ONNX/TensorFlow，此处仅测试 linear 模式。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.path.insert(0, "/workspace/mosaic")

import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_frame(width=64, height=64, color=(100, 150, 200)):
    """创建纯色 PIL 帧。"""
    return Image.new("RGB", (width, height), color=color)


class TestFrameInterpolation:
    """T_INTERP 系列：FrameInterpolator 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """创建 FrameInterpolator (linear) 实例。"""
        from mosaic.nodes.video.frame_interpolation import FrameInterpolator

        self.fi = FrameInterpolator(
            method="linear",
            scheduler=cpu_scheduler,
        )
        self.fi.load()

    def _make_video(self, num_frames=5, fps=30):
        """创建简单测试视频。"""
        frames = [_make_frame(color=(100 + i * 30, 150, 200)) for i in range(num_frames)]
        return VideoData(frames=frames, fps=fps)

    # T_INTERP_01
    def test_2x_interpolation_doubles_frame_count(self):
        """T_INTERP_01：2x 插帧将帧数翻倍。"""
        video = self._make_video(num_frames=4, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=2,
        ))
        out_video = result.get("video")
        assert out_video is not None, "输出应包含 video"
        # 2x 插帧：4帧 -> 7帧 (2*4 - 1)
        expected = 2 * 4 - 1
        assert len(out_video.frames) == expected, \
            f"2x 插帧后应为 {expected} 帧，实际 {len(out_video.frames)}"

    # T_INTERP_02
    def test_target_fps_parameter_works(self):
        """T_INTERP_02：target_fps 参数生效。"""
        video = self._make_video(num_frames=4, fps=30)
        result = self.fi(MosaicData(
            video=video,
            target_fps=60,
        ))
        new_fps = result.get("new_fps")
        assert new_fps is not None, "应包含 new_fps"
        assert new_fps > 30, "目标 60fps 时 new_fps 应大于 30"

    # T_INTERP_03
    def test_linear_method_runs_without_model(self):
        """T_INTERP_03：linear 模式无需模型即可运行。"""
        video = self._make_video(num_frames=5, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=2,
        ))
        assert result is not None, "linear 模式应正常完成"
        out_video = result.get("video")
        assert len(out_video.frames) > len(video.frames), \
            "插帧后帧数应增加"

    # T_INTERP_04
    def test_output_fps_is_correct(self):
        """T_INTERP_04：输出 fps 正确。"""
        video = self._make_video(num_frames=4, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=2,
        ))
        new_fps = result.get("new_fps")
        assert new_fps == 60, f"2x 插帧后 fps 应为 60，实际 {new_fps}"

    # T_INTERP_04 (continued)
    def test_scale_factor_4_output_fps(self):
        """4x 插帧后 fps 为 4 倍。"""
        video = self._make_video(num_frames=4, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=4,
        ))
        new_fps = result.get("new_fps")
        assert new_fps == 120, f"4x 插帧后 fps 应为 120，实际 {new_fps}"

    def test_scale_factor_1_no_change(self):
        """scale_factor=1 不改变帧数。"""
        video = self._make_video(num_frames=5, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=1,
        ))
        assert result.get("new_fps") == 30, "scale_factor=1 时 fps 不变"

    def test_method_and_num_passes_output(self):
        """验证 method 和 num_passes 在输出中。"""
        video = self._make_video(num_frames=4, fps=30)
        result = self.fi(MosaicData(
            video=video,
            scale_factor=2,
        ))
        assert result.get("method") == "linear", "method 应为 'linear'"
        assert result.get("num_passes") == 1, "2x 需要 1 次 pass"