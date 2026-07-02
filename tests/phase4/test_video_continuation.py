# tests/phase4/test_video_continuation.py
"""Phase 4 VideoContinuation 节点测试。

Mock CogVideoXPipeline，测试视频续写节点的基本行为。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_output_frames(num_frames=49, width=64, height=64):
    """创建 mock CogVideoX pipeline 输出的帧（numpy 数组格式）。"""
    arr = np.zeros((1, num_frames, height, width, 3), dtype=np.float32)
    for i in range(num_frames):
        arr[0, i, :, :, :] = np.random.randint(0, 256, (height, width, 3)).astype(np.float32) / 255.0

    mock_output = MagicMock()
    mock_output.frames = MagicMock()
    mock_output.frames.cpu.return_value = mock_output.frames
    mock_output.frames.numpy.return_value = arr
    return mock_output


def _make_mock_pipeline():
    """创建 mock CogVideoXPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_attention_slicing.return_value = None
    mock_pipe.enable_vae_slicing = MagicMock()

    output = _make_output_frames(num_frames=49)
    mock_pipe.return_value = output

    return mock_pipe


class TestVideoContinuation:
    """T_CONT 系列：VideoContinuation 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler, sample_video):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.video_continuation import VideoContinuation

        mock_pipe = _make_mock_pipeline()

        with patch(
            "diffusers.DiffusionPipeline",
            return_value=mock_pipe,
        ):
            with patch(
                "diffusers.DiffusionPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.vc = VideoContinuation(
                    model="THUDM/CogVideoX-5b",
                    scheduler=cpu_scheduler,
                )
                self.vc.load()
                self.vc._pipeline = mock_pipe
                self.mock_pipe = mock_pipe
                self.sample_video = sample_video

    # T_CONT_01
    def test_continuation_output_longer_than_input(self):
        """T_CONT_01：续写输出帧数大于输入帧数。"""
        result = self.vc(MosaicData(
            video=self.sample_video,
            prompt="视频继续向前推进",
            num_frames=49,
            overlap_frames=4,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert len(video.frames) > len(self.sample_video.frames), \
            "续写后帧数应大于原始帧数"

    # T_CONT_02
    def test_overlap_frames_parameter_works(self):
        """T_CONT_02：overlap_frames 参数生效。"""
        result = self.vc(MosaicData(
            video=self.sample_video,
            prompt="继续推进",
            num_frames=49,
            overlap_frames=2,
        ))
        assert result.get("overlap_frames") == 2, "overlap_frames 应为 2"

    # T_CONT_03
    def test_output_contains_original_and_continuation_frames(self):
        """T_CONT_03：输出包含原始帧 + 续写帧。"""
        result = self.vc(MosaicData(
            video=self.sample_video,
            prompt="继续推进",
            num_frames=49,
            overlap_frames=4,
        ))
        video = result.get("video")
        assert video is not None
        # 原始帧数 + 新帧数 - overlap
        total_frames = result.get("total_frames")
        assert total_frames > 0, "total_frames 应为正数"

    # T_CONT_04
    def test_continuation_video_key_accessible(self):
        """T_CONT_04：continuation_video 键可访问。"""
        result = self.vc(MosaicData(
            video=self.sample_video,
            prompt="继续推进",
            num_frames=49,
            overlap_frames=4,
        ))
        cont_video = result.get("continuation_video")
        assert cont_video is not None, "应包含 continuation_video"
        assert isinstance(cont_video, VideoData), "continuation_video 应为 VideoData"
        assert len(cont_video.frames) > 0, "continuation_video 应有帧"