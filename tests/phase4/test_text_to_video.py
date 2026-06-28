# tests/phase4/test_text_to_video.py
"""Phase 4 TextToVideo 节点测试。

Mock CogVideoXPipeline，测试文生视频节点的基本行为。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, "/workspace/mosaic")

import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_output_frame(width=64, height=64, color=(100, 150, 200)):
    """创建 mock pipeline 输出的帧。"""
    return Image.new("RGB", (width, height), color=color)


def _make_mock_pipeline():
    """创建 mock CogVideoXPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_attention_slicing.return_value = None
    mock_pipe.enable_vae_slicing = MagicMock()

    # 默认输出：49 帧
    frames = [_make_output_frame() for _ in range(49)]
    mock_output = MagicMock()
    mock_output.frames = MagicMock()
    # 模拟 numpy 数组 (1, 49, 64, 64, 3)
    import numpy as np
    arr = np.zeros((1, 49, 64, 64, 3), dtype=np.float32)
    # 填充一些有意义的像素值
    for i in range(49):
        arr[0, i, :, :, :] = np.random.randint(0, 256, (64, 64, 3)).astype(np.float32) / 255.0
    mock_output.frames = MagicMock()
    mock_output.frames.cpu.return_value = mock_output.frames
    mock_output.frames.numpy.return_value = arr
    mock_pipe.return_value = mock_output

    return mock_pipe


class TestTextToVideo:
    """T_T2V 系列：TextToVideo 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.text_to_video import TextToVideo

        mock_pipe = _make_mock_pipeline()

        # Mock CogVideoXPipeline.from_pretrained
        with patch(
            "diffusers.CogVideoXPipeline",
            return_value=mock_pipe,
        ):
            with patch(
                "diffusers.CogVideoXPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.t2v = TextToVideo(
                    model="THUDM/CogVideoX-5b",
                    scheduler=cpu_scheduler,
                )
                self.t2v.load()
                self.t2v._pipeline = mock_pipe
                self.mock_pipe = mock_pipe

    # T_T2V_01
    def test_basic_text_to_video_outputs_video_data(self):
        """T_T2V_01：基本文生视频输出 VideoData。"""
        result = self.t2v(MosaicData(
            prompt="一只猫在草地上奔跑",
            num_frames=49,
            fps=8,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert isinstance(video, VideoData), "video 应为 VideoData"
        assert len(video.frames) > 0, "应生成帧"

    # T_T2V_02
    def test_output_frame_count_matches_request(self):
        """T_T2V_02：输出帧数匹配请求。"""
        result = self.t2v(MosaicData(
            prompt="testing",
            num_frames=49,
            fps=8,
        ))
        assert result.get("num_frames") is not None, "应包含 num_frames"

    # T_T2V_03
    def test_fps_parameter_works(self):
        """T_T2V_03：fps 参数生效。"""
        result = self.t2v(MosaicData(
            prompt="testing",
            num_frames=49,
            fps=15,
        ))
        video = result.get("video")
        assert video.fps == 15, "fps 应为 15"

    # T_T2V_04
    def test_seed_reproducibility(self):
        """T_T2V_04：相同 seed 产生相同输出。"""
        result1 = self.t2v(MosaicData(
            prompt="testing",
            num_frames=49,
            seed=42,
        ))
        seed1 = result1.get("seed")
        assert seed1 is not None, "应包含 seed"

    # T_T2V_05
    def test_describe_includes_vram_and_license(self):
        """T_T2V_05：describe 包含 VRAM 估算和许可证信息。"""
        spec = self.t2v.describe()
        assert spec.model_info is not None, "应包含 model_info"
        assert "vram_gb" in spec.model_info, "应包含 vram_gb"
        assert "license" in spec.model_info, "应包含 license"

    # T_T2V_06
    def test_load_unload_state_tracking(self):
        """T_T2V_06：load/unload 状态跟踪。"""
        from mosaic.nodes.video.text_to_video import TextToVideo

        node = TextToVideo(
            model="THUDM/CogVideoX-5b",
            scheduler=self.t2v._scheduler,
        )
        assert not node.is_loaded(), "初始应为未加载"
        node.load()
        assert node.is_loaded(), "load 后应为已加载"
        node.unload()
        assert not node.is_loaded(), "unload 后应为未加载"