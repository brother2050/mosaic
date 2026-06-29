# tests/phase4/test_ltx_video.py
"""Phase 4 LTXVideo 节点测试。

Mock LTXPipeline，测试 Lightricks LTX-Video 文生视频节点的基本行为。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
import pytest

from mosaic.core.types import MosaicData, VideoData


def _make_mock_pipeline(num_frames: int = 97, width: int = 64, height: int = 64):
    """创建 mock LTXPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload = MagicMock()
    mock_pipe.vae = MagicMock()
    mock_pipe.vae.enable_tiling = MagicMock()

    arr = np.random.randint(
        0, 256, (1, num_frames, height, width, 3)
    ).astype(np.float32) / 255.0
    mock_output = MagicMock()
    mock_output.frames = MagicMock()
    mock_output.frames.cpu.return_value = mock_output.frames
    mock_output.frames.numpy.return_value = arr
    mock_pipe.return_value = mock_output

    return mock_pipe


class TestLTXVideo:
    """LTXVideo 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.ltx_video import LTXVideo

        mock_pipe = _make_mock_pipeline()

        with patch("diffusers.LTXPipeline", return_value=mock_pipe):
            with patch(
                "diffusers.LTXPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.ltx = LTXVideo(
                    model="Lightricks/LTX-Video",
                    scheduler=cpu_scheduler,
                )
                self.ltx.load()
                self.ltx._pipeline = mock_pipe
                self.mock_pipe = mock_pipe

    # T_LTX_01
    def test_basic_text_to_video_outputs_video_data(self):
        """T_LTX_01：基本文生视频输出 VideoData。"""
        result = self.ltx(MosaicData(
            prompt="A cat walking on the beach at sunset",
            num_frames=97,
            fps=30,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert isinstance(video, VideoData), "video 应为 VideoData"
        assert len(video.frames) > 0, "应生成帧"

    # T_LTX_02
    def test_output_contains_seed_and_metadata(self):
        """T_LTX_02：输出包含 seed 和元数据。"""
        result = self.ltx(MosaicData(
            prompt="testing",
            num_frames=97,
            seed=7,
        ))
        assert result.get("seed") == 7, "seed 应为 7"
        assert result.get("num_frames") is not None, "应包含 num_frames"
        assert result.get("duration") is not None, "应包含 duration"

    # T_LTX_03
    def test_fps_parameter_works(self):
        """T_LTX_03：fps 参数生效。"""
        result = self.ltx(MosaicData(
            prompt="testing",
            num_frames=97,
            fps=25,
        ))
        video = result.get("video")
        assert video.fps == 25, "fps 应为 25"

    # T_LTX_04
    def test_default_dtype_is_bfloat16(self):
        """T_LTX_04：默认精度为 bfloat16。"""
        from mosaic.nodes.video.ltx_video import LTXVideo

        node = LTXVideo()
        assert node._dtype_str == "bfloat16", "默认应为 bfloat16"

    # T_LTX_05
    def test_describe_includes_vram_and_license(self):
        """T_LTX_05：describe 包含 VRAM 估算和许可证信息。"""
        spec = self.ltx.describe()
        assert spec.model_info is not None, "应包含 model_info"
        assert "vram_gb" in spec.model_info, "应包含 vram_gb"
        assert spec.model_info["vram_gb"] == 12.0, "LTX-Video 需约 12GB"
        assert "license" in spec.model_info, "应包含 license"
        assert "OpenRAIL" in spec.model_info["license"], "LTX 许可证为 OpenRAIL-M"

    # T_LTX_06
    def test_load_unload_state_tracking(self):
        """T_LTX_06：load/unload 状态跟踪。"""
        from mosaic.nodes.video.ltx_video import LTXVideo

        node = LTXVideo(
            model="Lightricks/LTX-Video",
            scheduler=self.ltx._scheduler,
        )
        assert not node.is_loaded(), "初始应为未加载"
        node.load()
        assert node.is_loaded(), "load 后应为已加载"
        node.unload()
        assert not node.is_loaded(), "unload 后应为未加载"

    # T_LTX_07
    def test_negative_prompt_passed_to_pipeline(self):
        """T_LTX_07：负向提示词传递给 pipeline。"""
        self.ltx(MosaicData(
            prompt="a flying bird over mountains",
            negative_prompt="blurry",
            num_frames=97,
        ))
        call_kwargs = self.mock_pipe.call_args
        assert call_kwargs is not None, "pipeline 应被调用"
        assert "negative_prompt" in call_kwargs.kwargs, "应传递 negative_prompt"

    # T_LTX_08
    def test_missing_prompt_raises_value_error(self):
        """T_LTX_08：缺少 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.ltx(MosaicData(num_frames=97))

    # T_LTX_09
    def test_empty_prompt_raises_value_error(self):
        """T_LTX_09：空 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.ltx(MosaicData(prompt="", num_frames=97))

    # T_LTX_10
    def test_custom_dimensions_passed_to_pipeline(self):
        """T_LTX_10：自定义尺寸传递给 pipeline。"""
        self.ltx(MosaicData(
            prompt="testing dimensions",
            num_frames=97,
            width=1024,
            height=576,
        ))
        call_kwargs = self.mock_pipe.call_args
        assert call_kwargs.kwargs.get("width") == 1024, "width 应为 1024"
        assert call_kwargs.kwargs.get("height") == 576, "height 应为 576"
