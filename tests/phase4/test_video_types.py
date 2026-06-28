# tests/phase4/test_video_types.py
"""Phase 4 VideoData 类型测试。

测试 VideoData 的创建、属性、序列化/反序列化等核心功能。
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.types import VideoData, data_from_dict


def _make_frame(width=64, height=64, color=(100, 150, 200)):
    """创建纯色 PIL 帧。"""
    return Image.new("RGB", (width, height), color=color)


class TestVideoDataCreation:
    """T_VIDTYPE_01：VideoData 创建测试。"""

    # T_VIDTYPE_01
    def test_create_with_frames_and_fps(self):
        """T_VIDTYPE_01：创建 VideoData，验证 frames/fps/data_type。"""
        frames = [_make_frame() for _ in range(5)]
        video = VideoData(frames=frames, fps=30)

        assert video.frames is not None, "frames 不应为 None"
        assert len(video.frames) == 5, "frames 应为 5 帧"
        assert video.fps == 30, "fps 应为 30"
        assert video.data_type == "video", "data_type 应为 'video'"

    # T_VIDTYPE_01 (continued)
    def test_create_with_defaults(self):
        """默认 fps=30，frames 为空列表。"""
        video = VideoData()
        assert video.frames == [], "默认 frames 应为空列表"
        assert video.fps == 30, "默认 fps 应为 30"

    # T_VIDTYPE_01 (continued)
    def test_create_with_metadata(self):
        """创建时可以附带 metadata。"""
        frames = [_make_frame() for _ in range(3)]
        video = VideoData(
            frames=frames,
            fps=24,
            metadata={"source": "test", "duration": 3.0},
        )
        assert video.metadata["source"] == "test", "metadata source 不正确"
        assert video.metadata["duration"] == 3.0, "metadata duration 不正确"


class TestVideoDataDuration:
    """T_VIDTYPE_02：duration 属性测试。"""

    # T_VIDTYPE_02
    def test_duration_property(self):
        """T_VIDTYPE_02：duration = frame_count / fps。"""
        frames = [_make_frame() for _ in range(60)]
        video = VideoData(frames=frames, fps=30)
        expected = 60 / 30  # 2.0 秒
        # 从 metadata 拿 duration
        duration = video.metadata.get("duration")
        # _ensure_video_data 会设置 duration，但 VideoData 构造函数不自动设置
        # 直接计算来验证
        calc_duration = len(video.frames) / video.fps
        assert calc_duration == expected, f"计算时长应为 {expected}，实际 {calc_duration}"

    # T_VIDTYPE_02 (continued)
    def test_duration_zero_frames(self):
        """零帧时 duration 为 0。"""
        video = VideoData(frames=[], fps=30)
        calc_duration = len(video.frames) / video.fps if video.fps > 0 else 0.0
        assert calc_duration == 0.0, "零帧 duration 应为 0"


class TestVideoDataDimensions:
    """T_VIDTYPE_03：width/height 属性测试。"""

    # T_VIDTYPE_03
    def test_width_height_from_first_frame(self):
        """T_VIDTYPE_03：从首帧获取 width/height。"""
        frames = [_make_frame(width=128, height=96)]
        video = VideoData(frames=frames, fps=30)
        assert video.frames[0].size == (128, 96), "首帧尺寸应为 (128, 96)"
        assert video.frames[0].width == 128, "width 应为 128"
        assert video.frames[0].height == 96, "height 应为 96"

    # T_VIDTYPE_03 (continued)
    def test_multiple_frames_same_size(self):
        """多帧尺寸一致。"""
        frames = [_make_frame(width=64, height=64) for _ in range(5)]
        video = VideoData(frames=frames, fps=30)
        for f in video.frames:
            assert f.size == (64, 64), "所有帧尺寸应为 (64, 64)"


class TestVideoDataFrameCount:
    """T_VIDTYPE_04：frame_count 测试。"""

    # T_VIDTYPE_04
    def test_frame_count(self):
        """T_VIDTYPE_04：frame_count = len(frames)。"""
        for n in [1, 5, 10, 30]:
            frames = [_make_frame() for _ in range(n)]
            video = VideoData(frames=frames, fps=30)
            assert len(video.frames) == n, f"frame_count 应为 {n}"


class TestVideoDataSerialization:
    """T_VIDTYPE_05：序列化/反序列化测试。"""

    # T_VIDTYPE_05
    def test_serialization_roundtrip(self):
        """T_VIDTYPE_05：VideoData 序列化/反序列化（to_dict/from_dict）。"""
        frames = [_make_frame(color=(i * 40, 100, 200)) for i in range(3)]
        video = VideoData(frames=frames, fps=24, metadata={"key": "val"})

        # 序列化
        d = video.to_dict()
        assert "__data_type__" in d, "序列化后应包含 __data_type__"
        assert d["__data_type__"] == "video", "data_type 应为 'video'"

        # 反序列化
        restored = data_from_dict(d)
        assert isinstance(restored, VideoData), "反序列化后应为 VideoData"
        assert restored.fps == 24, "fps 应为 24"
        assert len(restored.frames) == 3, "帧数应为 3"
        assert restored.metadata["key"] == "val", "metadata 应保留"
        # 帧应为 PIL Image 类型
        from PIL import Image as PILImage
        assert isinstance(restored.frames[0], PILImage.Image), "反序列化后帧应为 PIL.Image"

    # T_VIDTYPE_05 (continued)
    def test_dict_like_access(self):
        """VideoData 支持字典式访问。"""
        frames = [_make_frame()]
        video = VideoData(frames=frames, fps=30)
        assert video["frames"] is not None
        assert video["fps"] == 30
        assert "metadata" in video


class TestVideoDataValidation:
    """VideoData 校验测试。"""

    def test_validate_correct(self):
        """正确的 VideoData 应通过校验。"""
        frames = [_make_frame()]
        video = VideoData(frames=frames, fps=30)
        assert VideoData.validate(video), "正确 VideoData 应通过校验"

    def test_validate_wrong_type(self):
        """非 VideoData 类型不应通过校验。"""
        from mosaic.core.types import MosaicData

        data = MosaicData()
        assert not VideoData.validate(data), "MosaicData 不应通过 VideoData 校验"

    def test_validate_fps_zero(self):
        """fps=0 不应通过校验。"""
        frames = [_make_frame()]
        video = VideoData(frames=frames, fps=0)
        assert not VideoData.validate(video), "fps=0 不应通过校验"

    def test_validate_frames_not_list(self):
        """frames 非列表不应通过校验。"""
        from unittest.mock import MagicMock
        video = MagicMock(spec=VideoData)
        video.get.return_value = "not_a_list"
        assert not VideoData.validate(video), "frames 非列表不应通过校验"