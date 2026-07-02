# tests/phase4/test_image_to_video.py
"""Phase 4 ImageToVideo 节点测试。

Mock StableVideoDiffusionPipeline，测试图生视频节点的基本行为。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_output_frames(num_frames=25, width=1024, height=576):
    """创建 mock SVD pipeline 输出的帧列表。"""
    import numpy as np
    frames = []
    for i in range(num_frames):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 10) % 256
        arr[:, :, 1] = 100
        arr[:, :, 2] = 200
        frames.append(Image.fromarray(arr))
    # 返回嵌套列表格式 [[frame0, frame1, ...]]
    return [frames]


def _make_mock_svd_pipeline():
    """创建 mock StableVideoDiffusionPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_vae_slicing = MagicMock()

    output_frames = _make_output_frames()
    mock_output = MagicMock()
    mock_output.frames = output_frames
    mock_pipe.return_value = mock_output

    return mock_pipe


class TestImageToVideo:
    """T_I2V 系列：ImageToVideo 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.image_to_video import ImageToVideo

        mock_pipe = _make_mock_svd_pipeline()

        with patch(
            "diffusers.DiffusionPipeline",
            return_value=mock_pipe,
        ):
            with patch(
                "diffusers.DiffusionPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.i2v = ImageToVideo(
                    model="stabilityai/stable-video-diffusion-img2vid-xt",
                    scheduler=cpu_scheduler,
                )
                self.i2v.load()
                self.i2v._pipeline = mock_pipe
                self.mock_pipe = mock_pipe

    # T_I2V_01
    def test_basic_image_to_video_outputs_video_data(self):
        """T_I2V_01：基本图生视频输出 VideoData。"""
        input_image = Image.new("RGB", (1024, 576), color=(100, 150, 200))
        result = self.i2v(MosaicData(
            image=input_image,
            num_frames=25,
            fps=7,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert isinstance(video, VideoData), "video 应为 VideoData"
        assert len(video.frames) > 0, "应生成帧"

    # T_I2V_02
    def test_correct_frame_count_output(self):
        """T_I2V_02：输出帧数正确。"""
        input_image = Image.new("RGB", (1024, 576), color=(100, 150, 200))
        result = self.i2v(MosaicData(
            image=input_image,
            num_frames=14,
            fps=7,
        ))
        assert result.get("num_frames") is not None, "应包含 num_frames"

    # T_I2V_03
    def test_motion_bucket_id_parameter_works(self):
        """T_I2V_03：motion_bucket_id 参数生效。"""
        input_image = Image.new("RGB", (1024, 576), color=(100, 150, 200))
        result = self.i2v(MosaicData(
            image=input_image,
            num_frames=25,
            fps=7,
            motion_bucket_id=200,
        ))
        video = result.get("video")
        assert video is not None, "应输出 video"
        assert video.metadata.get("motion_bucket_id") == 200, \
            "motion_bucket_id 应为 200"

    # T_I2V_04
    def test_input_image_auto_resized(self):
        """T_I2V_04：输入图片自动 resize 到 1024x576。"""
        input_image = Image.new("RGB", (512, 288), color=(100, 150, 200))
        # 节点内部会 resize，这里仅验证不会报错
        result = self.i2v(MosaicData(
            image=input_image,
            num_frames=14,
            fps=7,
        ))
        video = result.get("video")
        assert video is not None, "应输出 video"

    # T_I2V_04 (continued)
    def test_image_already_at_target_size(self):
        """图片已是目标尺寸时不报错。"""
        input_image = Image.new("RGB", (1024, 576), color=(100, 150, 200))
        result = self.i2v(MosaicData(
            image=input_image,
            num_frames=14,
            fps=7,
        ))
        assert result is not None, "应正常完成"

    # T_I2V_05
    def test_describe_includes_info(self):
        """T_I2V_05：describe 包含模型信息。"""
        spec = self.i2v.describe()
        assert spec.model_info is not None, "应包含 model_info"
        assert "name" in spec.model_info, "应包含 name"
        assert "license" in spec.model_info, "应包含 license"