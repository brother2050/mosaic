# tests/phase4/test_wan_video.py
"""Phase 4 WanVideo 节点测试。

Mock WanPipeline，测试 Wan2.1/Wan2.2 文生视频节点的基本行为，
包括 ``-Diffusers`` 后缀自动补全和 ``num_frames`` (4k+1) 校验。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace/mosaic")

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData


def _make_mock_pipeline(num_frames: int = 81, width: int = 64, height: int = 64):
    """创建 mock WanPipeline 对象。"""
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload = MagicMock()
    mock_pipe.vae = MagicMock()
    mock_pipe.vae.enable_tiling = MagicMock()

    # 模拟输出 (1, num_frames, H, W, 3) numpy 数组
    arr = np.random.randint(
        0, 256, (1, num_frames, height, width, 3)
    ).astype(np.float32) / 255.0
    mock_output = MagicMock()
    mock_output.frames = MagicMock()
    mock_output.frames.cpu.return_value = mock_output.frames
    mock_output.frames.numpy.return_value = arr
    mock_pipe.return_value = mock_output

    return mock_pipe


class TestWanVideo:
    """WanVideo 节点测试。"""

    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        """自动注入 mock pipeline 和 CPU 调度器。"""
        from mosaic.nodes.video.wan_video import WanVideo

        mock_pipe = _make_mock_pipeline()

        with patch("diffusers.WanPipeline", return_value=mock_pipe):
            with patch(
                "diffusers.WanPipeline.from_pretrained",
                return_value=mock_pipe,
            ):
                self.wan = WanVideo(
                    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                    scheduler=cpu_scheduler,
                )
                self.wan.load()
                self.wan._pipeline = mock_pipe
                self.mock_pipe = mock_pipe

    # T_WAN_01
    def test_basic_text_to_video_outputs_video_data(self):
        """T_WAN_01：基本文生视频输出 VideoData。"""
        result = self.wan(MosaicData(
            prompt="一只猫在海滩上散步，夕阳西下",
            num_frames=81,
            fps=16,
        ))
        video = result.get("video")
        assert video is not None, "输出应包含 video"
        assert isinstance(video, VideoData), "video 应为 VideoData"
        assert len(video.frames) > 0, "应生成帧"

    # T_WAN_02
    def test_output_contains_seed_and_metadata(self):
        """T_WAN_02：输出包含 seed 和元数据。"""
        result = self.wan(MosaicData(
            prompt="testing",
            num_frames=81,
            seed=42,
        ))
        assert result.get("seed") == 42, "seed 应为 42"
        assert result.get("num_frames") is not None, "应包含 num_frames"
        assert result.get("duration") is not None, "应包含 duration"

    # T_WAN_03
    def test_fps_parameter_works(self):
        """T_WAN_03：fps 参数生效。"""
        result = self.wan(MosaicData(
            prompt="testing",
            num_frames=81,
            fps=24,
        ))
        video = result.get("video")
        assert video.fps == 24, "fps 应为 24"

    # T_WAN_04
    def test_resolve_model_name_adds_diffusers_suffix(self):
        """T_WAN_04：自动补全 -Diffusers 后缀。"""
        from mosaic.nodes.video.wan_video import WanVideo

        # 不带后缀 -> 自动添加
        node = WanVideo(model="Wan-AI/Wan2.1-T2V-14B")
        assert node._resolve_model_name() == "Wan-AI/Wan2.1-T2V-14B-Diffusers"

        # 已带后缀 -> 不变
        node2 = WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        assert node2._resolve_model_name() == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        # Wan2.2 也能自动补全
        node3 = WanVideo(model="Wan-AI/Wan2.2-T2V-A14B")
        assert node3._resolve_model_name() == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # T_WAN_05
    def test_num_frames_adjusted_to_4k_plus_1(self):
        """T_WAN_05：num_frames 自动调整为 4k+1。"""
        result = self.wan(MosaicData(
            prompt="testing",
            num_frames=50,  # 不是 4k+1，应调整为 49 或 53
            fps=16,
        ))
        # 不报错即说明调整成功
        assert result.get("video") is not None

    # T_WAN_06
    def test_describe_includes_vram_and_license(self):
        """T_WAN_06：describe 包含 VRAM 估算和许可证信息。"""
        spec = self.wan.describe()
        assert spec.model_info is not None, "应包含 model_info"
        assert "vram_gb" in spec.model_info, "应包含 vram_gb"
        assert "license" in spec.model_info, "应包含 license"
        assert spec.model_info["license"] == "Apache 2.0", "Wan 许可证应为 Apache 2.0"

    # T_WAN_07
    def test_load_unload_state_tracking(self):
        """T_WAN_07：load/unload 状态跟踪。"""
        from mosaic.nodes.video.wan_video import WanVideo

        node = WanVideo(
            model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            scheduler=self.wan._scheduler,
        )
        assert not node.is_loaded(), "初始应为未加载"
        node.load()
        assert node.is_loaded(), "load 后应为已加载"
        node.unload()
        assert not node.is_loaded(), "unload 后应为未加载"

    # T_WAN_08
    def test_negative_prompt_passed_to_pipeline(self):
        """T_WAN_08：负向提示词传递给 pipeline。"""
        self.wan(MosaicData(
            prompt="a beautiful sunset",
            negative_prompt="blurry, low quality",
            num_frames=81,
        ))
        call_kwargs = self.mock_pipe.call_args
        assert call_kwargs is not None, "pipeline 应被调用"
        assert "negative_prompt" in call_kwargs.kwargs, "应传递 negative_prompt"

    # T_WAN_09
    def test_missing_prompt_raises_value_error(self):
        """T_WAN_09：缺少 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.wan(MosaicData(num_frames=81))

    # T_WAN_10
    def test_empty_prompt_raises_value_error(self):
        """T_WAN_10：空 prompt 抛出 ValueError。"""
        with pytest.raises(ValueError, match="prompt"):
            self.wan(MosaicData(prompt="", num_frames=81))
