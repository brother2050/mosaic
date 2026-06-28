# tests/phase7/test_digital_human_types.py
"""数字人数据类型测试：MotionData / AvatarData 的创建、序列化、校验。"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.core.types import (
    AudioData,
    AvatarData,
    MotionData,
    MosaicData,
    data_from_dict,
)


# ============================================================================
# T_DHTYPE_01：MotionData 创建，包含 keypoints 和 frame_count
# ============================================================================
def test_motion_data_create_basic():
    """MotionData 创建：包含 keypoints 和 frame_count。"""
    kps = np.zeros((30, 17, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=30, fps=30, skeleton_type="coco")

    assert isinstance(md, MotionData), "MotionData 应为 MotionData 实例"
    assert isinstance(md, MosaicData), "MotionData 应继承自 MosaicData"
    assert md.data_type == "motion", "data_type 应为 'motion'"

    # 属性访问
    assert md.keypoints is kps, "keypoints 属性应返回传入的 ndarray"
    assert md.frame_count == 30, "frame_count 应为 30"
    assert md.fps == 30, "fps 应为 30"
    assert md.skeleton_type == "coco", "skeleton_type 应为 'coco'"
    assert isinstance(md.metadata, dict), "metadata 应为 dict"


def test_motion_data_create_defaults():
    """MotionData 创建：使用默认参数。"""
    kps = np.zeros((10, 17, 3), dtype=np.float32)
    md = MotionData(keypoints=kps)

    assert md.frame_count == 0, "默认 frame_count 应为 0"
    assert md.fps == 30, "默认 fps 应为 30"
    assert md.skeleton_type == "coco", "默认 skeleton_type 应为 'coco'"
    assert md.metadata == {}, "默认 metadata 应为空字典"


def test_motion_data_create_with_metadata():
    """MotionData 创建：包含自定义 metadata。"""
    kps = np.ones((5, 10, 2), dtype=np.float32)
    meta = {"source": "video", "actor": "person_01"}
    md = MotionData(keypoints=kps, frame_count=5, metadata=meta)

    assert md.metadata == meta, "metadata 应等于传入的自定义字典"
    assert md.metadata["source"] == "video"
    assert md.metadata["actor"] == "person_01"


# ============================================================================
# T_DHTYPE_02：MotionData 序列化/反序列化
# ============================================================================
def test_motion_data_serialization_roundtrip():
    """MotionData 序列化/反序列化：to_dict/from_dict 往返后数据一致。"""
    rng = np.random.RandomState(42)
    kps = rng.randn(30, 17, 2).astype(np.float32)
    md = MotionData(
        keypoints=kps,
        frame_count=30,
        fps=25,
        skeleton_type="openpose",
        metadata={"source": "test"},
    )

    # 序列化
    d = md.to_dict()
    assert "__data_type__" in d, "to_dict 结果应包含 __data_type__ 键"
    assert d["__data_type__"] == "motion", "data_type 应为 'motion'"

    # 反序列化
    restored = MotionData.from_dict(d)
    assert isinstance(restored, MotionData), "from_dict 应返回 MotionData 实例"
    assert restored.data_type == "motion", "恢复后的 data_type 应为 'motion'"
    assert restored.frame_count == 30, "frame_count 往返后应一致"
    assert restored.fps == 25, "fps 往返后应一致"
    assert restored.skeleton_type == "openpose", "skeleton_type 往返后应一致"
    assert restored.metadata["source"] == "test", "metadata 往返后应一致"

    # keypoints 应为 ndarray 且数值接近
    restored_kps = restored.keypoints
    assert isinstance(restored_kps, np.ndarray), "恢复后的 keypoints 应为 ndarray"
    assert restored_kps.shape == kps.shape, (
        f"keypoints 形状应为 {kps.shape}，实际为 {restored_kps.shape}"
    )
    assert np.allclose(restored_kps, kps, atol=1e-5), "keypoints 数值往返后应一致"


def test_motion_data_serialization_data_from_dict():
    """MotionData 序列化：通过 data_from_dict 便捷函数。"""
    kps = np.zeros((10, 5, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=10, fps=30)
    d = md.to_dict()

    restored = data_from_dict(d)
    assert isinstance(restored, MotionData), "data_from_dict 应分发到 MotionData"
    assert restored.frame_count == 10


# ============================================================================
# T_DHTYPE_03：AvatarData 创建，包含 image
# ============================================================================
def test_avatar_data_create_with_image(sample_avatar_image):
    """AvatarData 创建：包含 image（使用 sample_avatar_image fixture）。"""
    av = AvatarData(image=sample_avatar_image)

    assert isinstance(av, AvatarData), "AvatarData 应为 AvatarData 实例"
    assert isinstance(av, MosaicData), "AvatarData 应继承自 MosaicData"
    assert av.data_type == "avatar", "data_type 应为 'avatar'"

    # image 属性
    from PIL import Image

    assert isinstance(av.image, Image.Image), "image 应为 PIL.Image 实例"
    assert av.image.size == (512, 512), "image 尺寸应为 (512, 512)"

    # 可选字段默认为 None
    assert av.face_embedding is None, "默认 face_embedding 应为 None"
    assert av.motion is None, "默认 motion 应为 None"
    assert av.audio is None, "默认 audio 应为 None"
    assert isinstance(av.metadata, dict), "metadata 应为 dict"


def test_avatar_data_create_defaults():
    """AvatarData 创建：使用默认参数。"""
    av = AvatarData()

    assert av.image is None, "默认 image 应为 None"
    assert av.face_embedding is None
    assert av.motion is None
    assert av.audio is None
    assert av.metadata == {}


# ============================================================================
# T_DHTYPE_04：AvatarData 包含可选的 face_embedding
# ============================================================================
def test_avatar_data_with_face_embedding():
    """AvatarData 创建：包含可选的 face_embedding。"""
    embedding = np.random.randn(512).astype(np.float32)
    av = AvatarData(face_embedding=embedding)

    assert av.face_embedding is not None, "face_embedding 应不为 None"
    assert isinstance(av.face_embedding, np.ndarray), "face_embedding 应为 ndarray"
    assert av.face_embedding.shape == (512,), "face_embedding 形状应为 (512,)"
    assert np.allclose(av.face_embedding, embedding), "face_embedding 数值应一致"


def test_avatar_data_face_embedding_default():
    """AvatarData 创建：未提供 face_embedding 时默认为 None。"""
    av = AvatarData(image=None)
    assert av.face_embedding is None, "未提供 face_embedding 时应为 None"


# ============================================================================
# T_DHTYPE_05：AvatarData 包含可选的 motion 和 audio
# ============================================================================
def test_avatar_data_with_motion_and_audio(sample_motion_data, sample_short_audio):
    """AvatarData 创建：包含可选的 motion 和 audio。"""
    av = AvatarData(motion=sample_motion_data, audio=sample_short_audio)

    assert av.motion is sample_motion_data, "motion 应等于传入的 MotionData"
    assert isinstance(av.motion, MotionData), "motion 应为 MotionData 实例"
    assert av.motion.frame_count == 30, "motion 的 frame_count 应为 30"

    assert av.audio is sample_short_audio, "audio 应等于传入的 AudioData"
    assert isinstance(av.audio, AudioData), "audio 应为 AudioData 实例"
    assert av.audio.sample_rate == 22050, "audio 的 sample_rate 应为 22050"


def test_avatar_data_with_motion_only(sample_motion_data):
    """AvatarData 创建：仅包含 motion，不含 audio。"""
    av = AvatarData(motion=sample_motion_data)

    assert av.motion is not None
    assert av.audio is None, "未提供 audio 时应为 None"


def test_avatar_data_with_audio_only(sample_short_audio):
    """AvatarData 创建：仅包含 audio，不含 motion。"""
    av = AvatarData(audio=sample_short_audio)

    assert av.audio is not None
    assert av.motion is None, "未提供 motion 时应为 None"


# ============================================================================
# T_DHTYPE_06：AvatarData 序列化/反序列化
# ============================================================================
def test_avatar_data_serialization_roundtrip(sample_avatar_image):
    """AvatarData 序列化/反序列化：to_dict/from_dict 往返后数据一致。"""
    embedding = np.random.RandomState(42).randn(512).astype(np.float32)
    av = AvatarData(
        image=sample_avatar_image,
        face_embedding=embedding,
        metadata={"name": "test_avatar"},
    )

    # 序列化
    d = av.to_dict()
    assert "__data_type__" in d, "to_dict 结果应包含 __data_type__ 键"
    assert d["__data_type__"] == "avatar", "data_type 应为 'avatar'"

    # 反序列化
    restored = AvatarData.from_dict(d)
    assert isinstance(restored, AvatarData), "from_dict 应返回 AvatarData 实例"
    assert restored.data_type == "avatar", "恢复后的 data_type 应为 'avatar'"

    # image 往返后应为 PIL.Image
    from PIL import Image

    assert isinstance(restored.image, Image.Image), "恢复后的 image 应为 PIL.Image"
    assert restored.image.size == sample_avatar_image.size, (
        f"image 尺寸应为 {sample_avatar_image.size}，实际为 {restored.image.size}"
    )

    # face_embedding 往返后应一致
    assert isinstance(restored.face_embedding, np.ndarray), (
        "恢复后的 face_embedding 应为 ndarray"
    )
    assert restored.face_embedding.shape == (512,), "face_embedding 形状应为 (512,)"
    assert np.allclose(restored.face_embedding, embedding, atol=1e-5), (
        "face_embedding 数值往返后应一致"
    )

    # metadata 往返后应一致
    assert restored.metadata["name"] == "test_avatar"


def test_avatar_data_serialization_data_from_dict(sample_avatar_image):
    """AvatarData 序列化：通过 data_from_dict 便捷函数。"""
    av = AvatarData(image=sample_avatar_image)
    d = av.to_dict()

    restored = data_from_dict(d)
    assert isinstance(restored, AvatarData), "data_from_dict 应分发到 AvatarData"


def test_avatar_data_serialization_with_motion_and_audio(
    sample_avatar_image, sample_motion_data, sample_short_audio
):
    """AvatarData 序列化：包含 motion 和 audio 的完整往返。"""
    av = AvatarData(
        image=sample_avatar_image,
        motion=sample_motion_data,
        audio=sample_short_audio,
    )

    d = av.to_dict()
    restored = AvatarData.from_dict(d)

    assert isinstance(restored, AvatarData)
    assert restored.motion is not None, "恢复后的 motion 应不为 None"
    assert isinstance(restored.motion, MotionData), "恢复后的 motion 应为 MotionData"
    assert restored.motion.frame_count == 30, "恢复后的 motion.frame_count 应为 30"

    assert restored.audio is not None, "恢复后的 audio 应不为 None"
    assert isinstance(restored.audio, AudioData), "恢复后的 audio 应为 AudioData"
    assert restored.audio.sample_rate == 22050, "恢复后的 audio.sample_rate 应为 22050"


# ============================================================================
# 校验测试
# ============================================================================
def test_motion_data_validate():
    """MotionData.validate：合法数据返回 True。"""
    kps = np.zeros((30, 17, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=30, fps=30)
    assert MotionData.validate(md) is True, "合法 MotionData 应通过校验"


def test_motion_data_validate_frame_count_negative():
    """MotionData.validate：frame_count 为负数时返回 False。"""
    kps = np.zeros((10, 5, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=-1, fps=30)
    assert MotionData.validate(md) is False, "frame_count 为负数应校验失败"


def test_motion_data_validate_fps_zero():
    """MotionData.validate：fps 为 0 时返回 False。"""
    kps = np.zeros((10, 5, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=10, fps=0)
    assert MotionData.validate(md) is False, "fps 为 0 应校验失败"


def test_motion_data_validate_not_motion_data():
    """MotionData.validate：非 MotionData 实例返回 False。"""
    av = AvatarData()
    assert MotionData.validate(av) is False, "非 MotionData 实例应校验失败"


def test_avatar_data_validate(sample_avatar_image):
    """AvatarData.validate：合法数据返回 True。"""
    av = AvatarData(image=sample_avatar_image)
    assert AvatarData.validate(av) is True, "合法 AvatarData 应通过校验"


def test_avatar_data_validate_image_none():
    """AvatarData.validate：image 为 None 时返回 True。"""
    av = AvatarData(image=None)
    assert AvatarData.validate(av) is True, "image 为 None 时应通过校验"


def test_avatar_data_validate_not_avatar_data():
    """AvatarData.validate：非 AvatarData 实例返回 False。"""
    md = MotionData()
    assert AvatarData.validate(md) is False, "非 AvatarData 实例应校验失败"


# ============================================================================
# 字典式访问测试
# ============================================================================
def test_motion_data_dict_access():
    """MotionData 支持字典式访问。"""
    kps = np.zeros((10, 5, 2), dtype=np.float32)
    md = MotionData(keypoints=kps, frame_count=10, fps=24)

    assert md["frame_count"] == 10, "字典式访问 frame_count 应为 10"
    assert md["fps"] == 24, "字典式访问 fps 应为 24"
    assert md["skeleton_type"] == "coco"
    assert "keypoints" in md, "keypoints 应在 MotionData 中"

    # 设置新键
    md["custom_key"] = "custom_value"
    assert md["custom_key"] == "custom_value"


def test_avatar_data_dict_access(sample_avatar_image):
    """AvatarData 支持字典式访问。"""
    av = AvatarData(image=sample_avatar_image)

    assert av["image"] is sample_avatar_image
    assert "face_embedding" in av
    assert "metadata" in av

    av["new_field"] = 42
    assert av["new_field"] == 42