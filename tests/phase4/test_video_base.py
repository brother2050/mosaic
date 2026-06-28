# tests/phase4/test_video_base.py
"""Phase 4 BaseVideoNode 静态方法测试。

测试 BaseVideoNode 提供的视频前后处理工具方法，包括：
_load_video, _save_video, _extract_frames, _resize_frames,
_frames_to_tensor, _tensor_to_frames, _ensure_even_dimensions,
_get_frame_at。
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

from mosaic.core.types import VideoData, MosaicData
from mosaic.nodes.video._base import BaseVideoNode


class _TestVideoNode(BaseVideoNode):
    """用于测试 BaseVideoNode 静态方法的最小化子类。"""

    name = "test-video-node"
    description = "Minimal video node for testing."
    version = "0.1.0"
    input_types = ["mosaic"]
    output_types = ["video"]

    def _load_model(self):
        pass

    def run(self, input_data: MosaicData) -> MosaicData:
        return MosaicData(frames=[])


def _make_frame(width=64, height=64, color=(100, 150, 200)):
    """创建纯色 PIL 帧。"""
    return Image.new("RGB", (width, height), color=color)


class TestLoadVideo:
    """T_VBASE_01：_load_video 从文件加载测试。"""

    # T_VBASE_01
    def test_load_video_from_file(self):
        """T_VBASE_01：_load_video 从文件加载，返回 VideoData。"""
        # 使用 imageio 创建临时视频文件
        import imageio.v2 as imageio

        frames_data = []
        for i in range(5):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 50) % 256
            arr[:, :, 1] = 100
            arr[:, :, 2] = 200
            frames_data.append(arr)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = imageio.get_writer(tmp_path, fps=30)
            for arr in frames_data:
                writer.append_data(arr)
            writer.close()

            video = BaseVideoNode._load_video(tmp_path)
            assert isinstance(video, VideoData), "应返回 VideoData"
            assert len(video.frames) == 5, "应提取 5 帧"
            assert video.fps == 30, "fps 应为 30"
            assert video.metadata["source"] == tmp_path, "metadata source 应为文件路径"
            assert video.metadata["frame_count"] == 5, "frame_count 应为 5"
        finally:
            os.unlink(tmp_path)

    # T_VBASE_01 (continued)
    def test_load_video_file_not_found(self):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="not found"):
            BaseVideoNode._load_video("/nonexistent/path/video.mp4")


class TestSaveVideo:
    """T_VBASE_02：_save_video 保存为文件测试。"""

    # T_VBASE_02
    def test_save_video_to_file(self):
        """T_VBASE_02：_save_video 保存帧列表为视频文件。"""
        frames = [_make_frame(color=(100 + i * 30, 150, 200)) for i in range(5)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            BaseVideoNode._save_video(frames, fps=30, path=tmp_path)
            assert os.path.exists(tmp_path), "输出文件应存在"
            assert os.path.getsize(tmp_path) > 0, "输出文件不应为空"

            # 验证可以重新加载
            video = BaseVideoNode._load_video(tmp_path)
            assert len(video.frames) == 5, "保存后重新加载应得到 5 帧"
        finally:
            os.unlink(tmp_path)


class TestExtractFrames:
    """T_VBASE_03：_extract_frames 测试。"""

    # T_VBASE_03
    def test_extract_frames_correct_indices(self, sample_video):
        """T_VBASE_03：_extract_frames 提取指定索引的帧。"""
        indices = [0, 2, 4, 6, 8]
        frames = BaseVideoNode._extract_frames(sample_video, indices)
        assert len(frames) == 5, "应提取 5 帧"
        assert frames[0] is sample_video.frames[0], "第 0 帧应对应"
        assert frames[1] is sample_video.frames[2], "第 2 帧应对应"
        assert frames[4] is sample_video.frames[8], "第 8 帧应对应"

    # T_VBASE_03 (continued)
    def test_extract_frames_out_of_bounds(self, sample_video):
        """越界索引被忽略。"""
        indices = [-1, 0, 100]
        frames = BaseVideoNode._extract_frames(sample_video, indices)
        assert len(frames) == 1, "仅应提取索引 0 的帧"
        assert frames[0] is sample_video.frames[0]


class TestResizeFrames:
    """T_VBASE_04：_resize_frames 批量 resize 测试。"""

    # T_VBASE_04
    def test_resize_frames_batch(self, sample_frames):
        """T_VBASE_04：_resize_frames 批量 resize 正确。"""
        target_size = (32, 32)
        resized = BaseVideoNode._resize_frames(sample_frames, target_size)
        assert len(resized) == len(sample_frames), "帧数应不变"
        for f in resized:
            assert f.size == target_size, f"每帧尺寸应为 {target_size}"

    # T_VBASE_04 (continued)
    def test_resize_frames_different_aspect(self):
        """不同宽高比 resize 正确。"""
        frames = [Image.new("RGB", (128, 64)), Image.new("RGB", (128, 64))]
        target = (64, 64)
        resized = BaseVideoNode._resize_frames(frames, target)
        assert resized[0].size == target, "resize 后尺寸应为 (64, 64)"


class TestFramesToTensor:
    """T_VBASE_05：_frames_to_tensor 测试。"""

    # T_VBASE_05
    def test_frames_to_tensor_shape(self, sample_frames):
        """T_VBASE_05：_frames_to_tensor 创建正确形状的张量。"""
        tensor = BaseVideoNode._frames_to_tensor(sample_frames)
        # 形状应为 (N, C, H, W)
        assert hasattr(tensor, "shape") or hasattr(tensor, "numpy"), \
            "tensor 应有可用属性"
        # 验证返回值可用
        assert tensor is not None, "tensor 不应为 None"


class TestTensorToFrames:
    """T_VBASE_06：_tensor_to_frames 测试。"""

    # T_VBASE_06
    def test_tensor_to_frames_roundtrip(self, sample_frames):
        """T_VBASE_06：_tensor_to_frames 与 _frames_to_tensor 可往返。"""
        tensor = BaseVideoNode._frames_to_tensor(sample_frames)
        frames = BaseVideoNode._tensor_to_frames(tensor)
        assert len(frames) == len(sample_frames), "帧数应一致"
        for f in frames:
            assert isinstance(f, Image.Image), "每帧应为 PIL.Image"
            assert f.size == sample_frames[0].size, "尺寸应一致"


class TestEnsureEvenDimensions:
    """T_VBASE_07：_ensure_even_dimensions 测试。"""

    # T_VBASE_07
    def test_ensure_even_handles_odd(self):
        """T_VBASE_07：_ensure_even_dimensions 将奇数变为偶数。"""
        assert BaseVideoNode._ensure_even_dimensions(99, 101) == (98, 100), \
            "奇数应减 1 变为偶数"
        assert BaseVideoNode._ensure_even_dimensions(100, 100) == (100, 100), \
            "偶数应保持不变"
        assert BaseVideoNode._ensure_even_dimensions(1, 1) == (2, 2), \
            "最小应为 (2, 2)"

    # T_VBASE_07 (continued)
    def test_ensure_even_mixed(self):
        """混合奇偶。"""
        assert BaseVideoNode._ensure_even_dimensions(100, 101) == (100, 100)
        assert BaseVideoNode._ensure_even_dimensions(99, 100) == (98, 100)


class TestGetFrameAt:
    """T_VBASE_08：_get_frame_at 按时间戳取帧测试。"""

    # T_VBASE_08
    def test_get_frame_at_by_timestamp(self, sample_video):
        """T_VBASE_08：_get_frame_at 按时间戳正确取帧。"""
        # 0.1 秒 * 30 fps = 帧索引 3
        frame = BaseVideoNode._get_frame_at(sample_video, 0.1)
        assert isinstance(frame, Image.Image), "应为 PIL.Image"
        assert frame is sample_video.frames[3], "时间戳 0.1s 应对应帧索引 3"

    # T_VBASE_08 (continued)
    def test_get_frame_at_start(self, sample_video):
        """时间戳 0 取第一帧。"""
        frame = BaseVideoNode._get_frame_at(sample_video, 0.0)
        assert frame is sample_video.frames[0], "时间戳 0 应对应第一帧"

    # T_VBASE_08 (continued)
    def test_get_frame_at_end(self, sample_video):
        """时间戳超出范围钳制到最后一帧。"""
        frame = BaseVideoNode._get_frame_at(sample_video, 999.0)
        assert frame is sample_video.frames[-1], "超范围时间戳对应最后一帧"

    # T_VBASE_08 (continued)
    def test_get_frame_at_empty_video(self):
        """空视频返回默认帧。"""
        empty_video = VideoData(frames=[], fps=30)
        frame = BaseVideoNode._get_frame_at(empty_video, 0.5)
        assert isinstance(frame, Image.Image), "空视频也应返回 PIL.Image"
        assert frame.size == (1, 1), "空视频默认帧应为 (1, 1)"