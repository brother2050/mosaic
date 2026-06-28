# tests/phase7/test_digital_human_base.py
"""数字人域基类测试：BaseDigitalHumanNode 的静态工具方法。"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from mosaic.nodes.digital_human._base import BaseDigitalHumanNode


# ---------------------------------------------------------------------------
# 确保 insightface 可作为包被 from insightface.app import ... 导入
# conftest 将 insightface 注册为 MagicMock（session 级 autouse fixture），
# 但 MagicMock 缺少 __path__ / __spec__ 等包必需属性，导致
# "from insightface.app import FaceAnalysis" 触发 ImportError。
# 本 fixture 在 conftest session 级 fixture 之后运行，补全缺失属性。
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _fix_insightface_mock() -> None:
    """补全 insightface mock 使其可作为包被导入。"""
    if "insightface" in sys.modules:
        _insightface_mock = sys.modules["insightface"]
        if not hasattr(_insightface_mock, "__path__"):
            _insightface_mock.__path__ = []
        if not hasattr(_insightface_mock, "__spec__"):
            _insightface_mock.__spec__ = None
        if "insightface.app" not in sys.modules:
            sys.modules["insightface.app"] = _insightface_mock.app


# ============================================================================
# 辅助函数
# ============================================================================
def _image_pixels_equal(img1: Image.Image, img2: Image.Image) -> bool:
    """逐像素比较两张图片是否完全相同。"""
    if img1.size != img2.size:
        return False
    return (np.array(img1) == np.array(img2)).all()


# ============================================================================
# T_DHBASE_01：_detect_face 检测图片中的人脸
# ============================================================================
def test_detect_face_basic(sample_avatar_image):
    """_detect_face 检测图片中的人脸，返回 (face_image, bbox, landmarks)。"""
    result = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    # 验证返回 tuple 长度为 3
    assert isinstance(result, tuple), "返回值应为 tuple"
    assert len(result) == 3, f"返回值长度应为 3，实际为 {len(result)}"

    face_image, bbox, landmarks = result

    # face_image 是 PIL.Image
    assert isinstance(face_image, Image.Image), "face_image 应为 PIL.Image"

    # bbox 是 4 元组 (x1, y1, x2, y2)
    assert isinstance(bbox, tuple), "bbox 应为 tuple"
    assert len(bbox) == 4, f"bbox 长度应为 4，实际为 {len(bbox)}"
    assert all(isinstance(v, (int, np.integer)) for v in bbox), (
        "bbox 各元素应为整数"
    )
    # 使用 insightface mock 返回的 bbox: [100, 80, 400, 420]
    x1, y1, x2, y2 = bbox
    assert x1 < x2, "x1 应小于 x2"
    assert y1 < y2, "y1 应小于 y2"

    # landmarks 是 ndarray (5 个关键点）
    assert isinstance(landmarks, np.ndarray), "landmarks 应为 ndarray"
    assert landmarks.shape[0] == 5, f"landmarks 应有 5 个关键点，实际为 {landmarks.shape[0]}"
    assert landmarks.shape[1] == 2, f"landmarks 每个关键点应为 2 维，实际为 {landmarks.shape[1]}"


def test_detect_face_with_pil_image(sample_avatar_image):
    """_detect_face 接收 PIL.Image 直接工作。"""
    result = BaseDigitalHumanNode._detect_face(sample_avatar_image)
    face_image, bbox, landmarks = result

    assert isinstance(face_image, Image.Image)
    # 裁剪后人脸尺寸应小于原图尺寸
    assert face_image.size[0] <= sample_avatar_image.size[0]
    assert face_image.size[1] <= sample_avatar_image.size[1]


# ============================================================================
# T_DHBASE_02：_detect_face 无人脸图片返回友好提示
# ============================================================================
class _MockFaceAnalysisNoFace:
    """Mock FaceAnalysis 类，返回空人脸列表。"""

    def __init__(self, name: str = "buffalo_l") -> None:
        self.name = name

    def prepare(self, ctx_id: int = 0, det_size: tuple = (640, 640)) -> None:
        pass

    def get(self, img_array: np.ndarray) -> list:
        return []


def test_detect_face_no_face_raises():
    """_detect_face 检测不到人脸时抛出 ValueError。"""
    blank_img = Image.new("RGB", (100, 100), (128, 128, 128))

    # 用返回空列表的 MockFaceAnalysis 替换 insightface.app 模块中的 FaceAnalysis
    iapp = sys.modules["insightface.app"]
    with patch.object(iapp, "FaceAnalysis", _MockFaceAnalysisNoFace, create=True):
        with pytest.raises(ValueError, match="No face detected"):
            BaseDigitalHumanNode._detect_face(blank_img)


def test_detect_face_no_face_error_message():
    """_detect_face 无人脸时的错误信息应包含友好提示。"""
    blank_img = Image.new("RGB", (50, 50), (200, 200, 200))

    iapp = sys.modules["insightface.app"]
    with patch.object(iapp, "FaceAnalysis", _MockFaceAnalysisNoFace, create=True):
        with pytest.raises(ValueError) as exc_info:
            BaseDigitalHumanNode._detect_face(blank_img)

        error_msg = str(exc_info.value)
        assert "No face detected" in error_msg, (
            f"错误信息应包含 'No face detected'，实际为: {error_msg}"
        )
        assert "clear" in error_msg.lower() or "visible" in error_msg.lower(), (
            f"错误信息应包含友好提示，实际为: {error_msg}"
        )


# ============================================================================
# T_DHBASE_03：_extract_face_embedding 返回向量
# ============================================================================
def test_extract_face_embedding(sample_avatar_image):
    """_extract_face_embedding 返回向量（ndarray）。"""
    # 先用 _detect_face 获取人脸图片
    face_image, _, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    # 提取人脸特征
    embedding = BaseDigitalHumanNode._extract_face_embedding(face_image)

    assert isinstance(embedding, np.ndarray), (
        f"返回值应为 ndarray，实际为 {type(embedding)}"
    )
    # insightface 可用时 shape 为 (512,)，fallback 时为 (4096,)
    assert embedding.ndim == 1, (
        f"embedding 应为一维数组，实际维数为 {embedding.ndim}"
    )
    assert embedding.dtype == np.float32, (
        f"embedding dtype 应为 float32，实际为 {embedding.dtype}"
    )


def test_extract_face_embedding_with_pil_image():
    """_extract_face_embedding 接收 PIL.Image 直接工作。"""
    face_img = Image.new("RGB", (64, 64), (200, 180, 160))
    embedding = BaseDigitalHumanNode._extract_face_embedding(face_img)

    assert isinstance(embedding, np.ndarray)


# ============================================================================
# T_DHBASE_04：_align_face 输出尺寸正确
# ============================================================================
def test_align_face_output_size(sample_avatar_image):
    """_align_face 用 _detect_face 获取 landmarks 并对齐到 (256, 256)。"""
    _, _, landmarks = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    aligned = BaseDigitalHumanNode._align_face(
        sample_avatar_image, landmarks, target_size=(256, 256)
    )

    assert isinstance(aligned, Image.Image), "返回值应为 PIL.Image"
    assert aligned.size == (256, 256), (
        f"对齐后尺寸应为 (256, 256)，实际为 {aligned.size}"
    )


def test_align_face_custom_target_size(sample_avatar_image):
    """_align_face 支持自定义 target_size。"""
    _, _, landmarks = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    aligned = BaseDigitalHumanNode._align_face(
        sample_avatar_image, landmarks, target_size=(128, 128)
    )

    assert aligned.size == (128, 128), (
        f"对齐后尺寸应为 (128, 128)，实际为 {aligned.size}"
    )


def test_align_face_none_landmarks(sample_avatar_image):
    """_align_face landmarks 为 None 时直接 resize 到 target_size。"""
    aligned = BaseDigitalHumanNode._align_face(
        sample_avatar_image, None, target_size=(128, 128)
    )

    assert isinstance(aligned, Image.Image)
    assert aligned.size == (128, 128)


def test_align_face_insufficient_landmarks(sample_avatar_image):
    """_align_face landmarks 不足 2 个时直接 resize 到 target_size。"""
    aligned = BaseDigitalHumanNode._align_face(
        sample_avatar_image,
        np.array([[100, 100]], dtype=np.float32),  # 只有 1 个关键点
        target_size=(128, 128),
    )

    assert isinstance(aligned, Image.Image)
    assert aligned.size == (128, 128)


# ============================================================================
# T_DHBASE_05：_crop_and_resize 裁剪和缩放正确
# ============================================================================
def test_crop_and_resize_basic(sample_avatar_image):
    """_crop_and_resize 使用固定 bbox 裁剪并缩放到目标尺寸。"""
    bbox = (100, 80, 400, 420)
    result = BaseDigitalHumanNode._crop_and_resize(
        sample_avatar_image, bbox, target_size=(128, 128), padding=0
    )

    assert isinstance(result, Image.Image), "返回值应为 PIL.Image"
    assert result.size == (128, 128), (
        f"输出尺寸应为 (128, 128)，实际为 {result.size}"
    )


def test_crop_and_resize_with_padding(sample_avatar_image):
    """_crop_and_resize 带 padding 扩展裁剪区域。"""
    bbox = (100, 80, 400, 420)
    # 无 padding 时裁剪区域为 (100, 80) -> (400, 420) = 300x340
    no_pad = BaseDigitalHumanNode._crop_and_resize(
        sample_avatar_image, bbox, target_size=(128, 128), padding=0
    )
    # 有 padding 时裁剪区域扩大
    with_pad = BaseDigitalHumanNode._crop_and_resize(
        sample_avatar_image, bbox, target_size=(128, 128), padding=20
    )

    assert no_pad.size == (128, 128)
    assert with_pad.size == (128, 128)


def test_crop_and_resize_different_target_size(sample_avatar_image):
    """_crop_and_resize 支持不同的 target_size。"""
    bbox = (100, 80, 400, 420)
    result = BaseDigitalHumanNode._crop_and_resize(
        sample_avatar_image, bbox, target_size=(256, 256), padding=0
    )

    assert result.size == (256, 256)


def test_crop_and_resize_padding_clamped(sample_avatar_image):
    """_crop_and_resize padding 不会超出图片边界。"""
    bbox = (100, 80, 400, 420)
    result = BaseDigitalHumanNode._crop_and_resize(
        sample_avatar_image, bbox, target_size=(128, 128), padding=500
    )

    assert result.size == (128, 128)


# ============================================================================
# T_DHBASE_06：_blend_face 融合后图片尺寸与原图一致
# ============================================================================
def test_blend_face_output_size(sample_avatar_image):
    """_blend_face 融合后图片尺寸与原图一致。"""
    face_image, bbox, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    blended = BaseDigitalHumanNode._blend_face(
        sample_avatar_image, face_image, bbox, blend_ratio=0.5
    )

    assert isinstance(blended, Image.Image), "返回值应为 PIL.Image"
    assert blended.size == sample_avatar_image.size, (
        f"融合后尺寸应与原图一致: {sample_avatar_image.size}，实际为 {blended.size}"
    )


def test_blend_face_ratio_zero(sample_avatar_image):
    """_blend_face blend_ratio=0.0 时输出与原图完全相同（逐像素）。"""
    face_image, bbox, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    blended = BaseDigitalHumanNode._blend_face(
        sample_avatar_image, face_image, bbox, blend_ratio=0.0
    )

    assert blended.size == sample_avatar_image.size
    assert _image_pixels_equal(blended, sample_avatar_image), (
        "blend_ratio=0.0 时输出应与原图逐像素完全相同"
    )


def test_blend_face_ratio_one(sample_avatar_image):
    """_blend_face blend_ratio=1.0 时在人脸区域完全使用生成结果。"""
    face_image, bbox, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    blended = BaseDigitalHumanNode._blend_face(
        sample_avatar_image, face_image, bbox, blend_ratio=1.0
    )

    assert blended.size == sample_avatar_image.size
    # 在人脸区域内，像素值应等于 face_image resize 后的像素
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    resized_face = face_image.resize((w, h), Image.LANCZOS)
    blended_region = np.array(blended.crop((x1, y1, x2, y2)).convert("RGB"))
    face_region = np.array(resized_face.convert("RGB"))
    assert np.array_equal(blended_region, face_region), (
        "blend_ratio=1.0 时人脸区域应完全等于生成的人脸"
    )


def test_blend_face_invalid_bbox(sample_avatar_image):
    """_blend_face 无效 bbox 时返回原图。"""
    face_image, _, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    # bbox 宽度为 0
    result = BaseDigitalHumanNode._blend_face(
        sample_avatar_image, face_image, (100, 80, 100, 80), blend_ratio=0.5
    )

    assert result is sample_avatar_image, "无效 bbox 时应返回原图"


# ============================================================================
# T_DHBASE_07（附加）：_apply_expression 表情参数
# ============================================================================
def test_apply_expression_basic(sample_avatar_image):
    """_apply_expression 应用表情参数后仍返回 PIL.Image。"""
    face_image, _, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    params = {"smile": 0.5, "mouth_open": 0.3}
    result = BaseDigitalHumanNode._apply_expression(face_image, params)

    assert isinstance(result, Image.Image), "返回值应为 PIL.Image"
    assert result.size == face_image.size, (
        f"表情应用后尺寸应不变: {face_image.size}，实际为 {result.size}"
    )


def test_apply_expression_default_params(sample_avatar_image):
    """_apply_expression 空参数字典时返回原图。"""
    face_image, _, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    result = BaseDigitalHumanNode._apply_expression(face_image, {})

    assert isinstance(result, Image.Image)
    assert result.size == face_image.size


def test_apply_expression_full_params(sample_avatar_image):
    """_apply_expression 全参数设置。"""
    face_image, _, _ = BaseDigitalHumanNode._detect_face(sample_avatar_image)

    params = {
        "smile": 0.8,
        "eye_openness": 0.9,
        "mouth_open": 0.6,
        "brow_raise": 0.4,
    }
    result = BaseDigitalHumanNode._apply_expression(face_image, params)

    assert isinstance(result, Image.Image)
    assert result.size == face_image.size


# ============================================================================
# _load_image 静态方法
# ============================================================================
def test_load_image_from_pil(sample_avatar_image):
    """_load_image 接收 PIL.Image 返回 RGB 模式。"""
    result = BaseDigitalHumanNode._load_image(sample_avatar_image)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_load_image_invalid_type():
    """_load_image 不支持的类型抛出 TypeError。"""
    with pytest.raises(TypeError, match="Expected str"):
        BaseDigitalHumanNode._load_image(12345)