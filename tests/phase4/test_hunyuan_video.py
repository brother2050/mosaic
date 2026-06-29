# tests/phase4/test_hunyuan_video.py
"""Phase 4 HunyuanVideo 节点测试。

Mock HunyuanVideoPipeline，测试腾讯混元文生视频节点的基本行为，
包括 VAE chunking 专属优化。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
import pytest

from mosaic.core.types import MosaicData, VideoData


def _make_mock_pipeline(num_frames: int = 129, width: int = 64, height: int = 64):
    """创建 mock HunyuanVideoPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload = MagicMock()
    mock_pipe.vae = MagicMock()
    mock_pipe.vae.enable_tiling = MagicMock()
    mock_pipe.vae.enable_chunking = MagicMock()

    arr = np.random.randint(
        0, 256, (1, num_frames, height, width, 3)
    ).astype(np.float32) / 255.0
    mock_output = MagicMock()
    mock_output.frames = MagicMock()
    mock_output.frames.cpu.return_value = mock_output.frames
    mock_output.frames.numpy.return_value = arr
    mock_pipe.return_value = mock_output

    return mock_pipe


class TestHunyuanVideo:
    """HunyuanVideo 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.hunyuan_video import HunyuanVideo

        mock_pipe = _make_mock_pipeline()

        with patch("diffusers.HunyuanVideoPipeline", return_value=mock_pipe):
            with patch(
                "diffusers.HunyuanVideoPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.hv = HunyuanVideo(
                    model="tencent/HunyuanVideo",
                    scheduler=cpu_scheduler,
                )
                self.hv.load()
                self.hv._pipeline = mock_pipe
                self.mock_pipe = mock_pipe

    # T_HY_01
    def test_basic_text_to_video_outputs_video_data(self):
        """T_HY_01：基本文生视频输出 VideoData。"""
        result = self.hv(MosaicData(
            prompt="一只猫在草地上奔跑",
            num_frames=129,
            fps=24,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert isinstance(video, VideoData), "video 应为 VideoData"
        assert len(video.frames) > 0, "应生成帧"

    # T_HY_02
    def test_output_contains_seed_and_metadata(self):
        """T_HY_02：输出包含 seed 和元数据。"""
        result = self.hv(MosaicData(
            prompt="testing",
            num_frames=129,
            seed=99,
        ))
        assert result.get("seed") == 99, "seed 应为 99"
        assert result.get("num_frames") is not None, "应包含 num_frames"
        assert result.get("duration") is not None, "应包含 duration"

    # T_HY_03
    def test_fps_parameter_works(self):
        """T_HY_03：fps 参数生效。"""
        result = self.hv(MosaicData(
            prompt="testing",
            num_frames=129,
            fps=30,
        ))
        video = result.get("video")
        assert video.fps == 30, "fps 应为 30"

    # T_HY_04
    def test_default_dtype_is_bfloat16(self):
        """T_HY_04：默认精度为 bfloat16。"""
        from mosaic.nodes.video.hunyuan_video import HunyuanVideo

        node = HunyuanVideo()
        assert node._dtype_str == "bfloat16", "默认应为 bfloat16"

    # T_HY_05
    def test_vae_chunking_enabled_by_default(self):
        """T_HY_05：默认启用 VAE chunking。"""
        from mosaic.nodes.video.hunyuan_video import HunyuanVideo

        node = HunyuanVideo()
        assert node._enable_chunking is True, "默认应启用 chunking"

    # T_HY_06
    def test_describe_includes_vram_and_license(self):
        """T_HY_06：describe 包含 VRAM 估算和许可证信息。"""
        spec = self.hv.describe()
        assert spec.model_info is not None, "应包含 model_info"
        assert "vram_gb" in spec.model_info, "应包含 vram_gb"
        assert spec.model_info["vram_gb"] == 60.0, "HunyuanVideo 需约 60GB"
        assert "license" in spec.model_info, "应包含 license"

    # T_HY_07
    def test_load_unload_state_tracking(self):
        """T_HY_07：load/unload 状态跟踪。"""
        from mosaic.nodes.video.hunyuan_video import HunyuanVideo

        node = HunyuanVideo(
            model="tencent/HunyuanVideo",
            scheduler=self.hv._scheduler,
        )
        assert not node.is_loaded(), "初始应为未加载"
        node.load()
        assert node.is_loaded(), "load 后应为已加载"
        node.unload()
        assert not node.is_loaded(), "unload 后应为未加载"

    # T_HY_08
    def test_negative_prompt_passed_to_pipeline(self):
        """T_HY_08：负向提示词传递给 pipeline。"""
        self.hv(MosaicData(
            prompt="a dancing robot",
            negative_prompt="distorted, low quality",
            num_frames=129,
        ))
        call_kwargs = self.mock_pipe.call_args
        assert call_kwargs is not None, "pipeline 应被调用"
        assert "negative_prompt" in call_kwargs.kwargs, "应传递 negative_prompt"

    # T_HY_09
    def test_missing_prompt_raises_value_error(self):
        """T_HY_09：缺少 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.hv(MosaicData(num_frames=129))

    # T_HY_10
    def test_empty_prompt_raises_value_error(self):
        """T_HY_10：空 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.hv(MosaicData(prompt="   ", num_frames=129))
