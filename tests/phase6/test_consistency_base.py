# tests/phase6/test_consistency_base.py
"""BaseConsistencyNode 静态方法测试。

测试 _load_image、_resize_to_model、_prepare_face_region、
_compute_image_similarity 等静态工具方法。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from mosaic.nodes.consistency._base import BaseConsistencyNode


class TestLoadImage:
    """_load_image 静态方法测试."""

    def test_load_image_from_file_path(self, sample_face_image, tmp_path):
        """# T_CBASE_01：_load_image 从文件路径加载."""
        # 保存合成图片到临时文件
        file_path = tmp_path / "test_face.png"
        sample_face_image.save(str(file_path))

        # 加载
        loaded = BaseConsistencyNode._load_image(str(file_path))
        assert isinstance(
            loaded, Image.Image
        ), "_load_image should return a PIL.Image.Image from file path"
        assert loaded.mode == "RGB", "Loaded image should be in RGB mode"
        assert loaded.size == sample_face_image.size, (
            f"Loaded image size {loaded.size} should match "
            f"original {sample_face_image.size}"
        )

    def test_load_image_from_pil_image(self, sample_face_image):
        """# T_CBASE_02：_load_image 从 PIL.Image 直接传入."""
        loaded = BaseConsistencyNode._load_image(sample_face_image)
        assert isinstance(
            loaded, Image.Image
        ), "_load_image should return a PIL.Image.Image when given PIL.Image"
        assert loaded.mode == "RGB", "Returned image should be in RGB mode"
        assert loaded.size == sample_face_image.size, (
            f"Returned image size {loaded.size} should match "
            f"original {sample_face_image.size}"
        )

    def test_load_image_from_rgba_image(self, sample_style_reference):
        """_load_image 将 RGBA 图片转换为 RGB."""
        # 创建 RGBA 图片
        rgba_img = sample_style_reference.convert("RGBA")
        loaded = BaseConsistencyNode._load_image(rgba_img)
        assert loaded.mode == "RGB", (
            f"RGBA image should be converted to RGB, got {loaded.mode}"
        )

    def test_load_image_file_not_found(self):
        """_load_image 对不存在的文件路径抛出 FileNotFoundError."""
        with pytest.raises(
            FileNotFoundError, match="Image file not found"
        ):
            BaseConsistencyNode._load_image("/nonexistent/path/image.png")

    def test_load_image_invalid_type(self):
        """_load_image 对无效类型抛出 TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            BaseConsistencyNode._load_image(12345)


class TestResizeToModel:
    """_resize_to_model 静态方法测试."""

    def test_resize_to_default_size(self, sample_face_image):
        """# T_CBASE_03：_resize_to_model 输出尺寸正确（512x512）."""
        resized = BaseConsistencyNode._resize_to_model(sample_face_image)
        assert resized.size == (512, 512), (
            f"Default target size should be (512, 512), got {resized.size}"
        )
        assert isinstance(
            resized, Image.Image
        ), "Resized output should be a PIL.Image.Image"

    def test_resize_to_custom_size(self, sample_face_image):
        """_resize_to_model 支持自定义目标尺寸."""
        resized = BaseConsistencyNode._resize_to_model(
            sample_face_image, target_size=(256, 256)
        )
        assert resized.size == (256, 256), (
            f"Custom target size should be (256, 256), got {resized.size}"
        )

    def test_resize_to_non_8_multiple(self, sample_face_image):
        """_resize_to_model 将尺寸对齐到 8 的倍数."""
        # 请求 515x515，应被对齐到 512x512
        resized = BaseConsistencyNode._resize_to_model(
            sample_face_image, target_size=(515, 515)
        )
        assert resized.size[0] % 8 == 0, (
            f"Width {resized.size[0]} should be a multiple of 8"
        )
        assert resized.size[1] % 8 == 0, (
            f"Height {resized.size[1]} should be a multiple of 8"
        )

    def test_resize_to_minimum_size(self, sample_face_image):
        """_resize_to_model 不会缩小到小于 8 像素."""
        resized = BaseConsistencyNode._resize_to_model(
            sample_face_image, target_size=(1, 1)
        )
        assert resized.size[0] >= 8, "Width should be at least 8"
        assert resized.size[1] >= 8, "Height should be at least 8"

    def test_resize_from_string_path(self, sample_face_image, tmp_path):
        """_resize_to_model 也接受文件路径字符串."""
        file_path = tmp_path / "test_face.png"
        sample_face_image.save(str(file_path))
        resized = BaseConsistencyNode._resize_to_model(
            str(file_path), target_size=(256, 256)
        )
        assert resized.size == (256, 256), "Should resize correctly from file path"


class TestPrepareFaceRegion:
    """_prepare_face_region 静态方法测试."""

    def test_prepare_face_region_returns_face_and_bbox(self, sample_face_image):
        """# T_CBASE_04：_prepare_face_region 输出 face_image 和 bbox."""
        # insightface 不可用时会回退到中心裁剪
        face_image, bbox = BaseConsistencyNode._prepare_face_region(
            sample_face_image
        )

        assert isinstance(
            face_image, Image.Image
        ), "face_image should be a PIL.Image.Image"
        assert isinstance(bbox, tuple), "bbox should be a tuple"
        assert len(bbox) == 4, "bbox should have 4 elements (x1, y1, x2, y2)"

        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 <= sample_face_image.width, (
            f"bbox x-coordinates should be within image bounds: {bbox}"
        )
        assert 0 <= y1 < y2 <= sample_face_image.height, (
            f"bbox y-coordinates should be within image bounds: {bbox}"
        )
        # 裁剪后的 face_image 尺寸应匹配 bbox
        assert face_image.size == (x2 - x1, y2 - y1), (
            f"face_image size {face_image.size} should match bbox dimensions "
            f"({x2 - x1}, {y2 - y1})"
        )

    def test_prepare_face_region_with_padding(self, sample_face_image):
        """_prepare_face_region 支持自定义 padding_ratio."""
        face_no_pad, bbox_no_pad = BaseConsistencyNode._prepare_face_region(
            sample_face_image, padding_ratio=0.0
        )
        face_pad, bbox_pad = BaseConsistencyNode._prepare_face_region(
            sample_face_image, padding_ratio=0.5
        )

        # 有 padding 时 bbox 应该更大（或至少不小于无 padding）
        w_no_pad = bbox_no_pad[2] - bbox_no_pad[0]
        h_no_pad = bbox_no_pad[3] - bbox_no_pad[1]
        w_pad = bbox_pad[2] - bbox_pad[0]
        h_pad = bbox_pad[3] - bbox_pad[1]
        assert w_pad >= w_no_pad, (
            f"Padded bbox width ({w_pad}) should be >= un-padded ({w_no_pad})"
        )
        assert h_pad >= h_no_pad, (
            f"Padded bbox height ({h_pad}) should be >= un-padded ({h_no_pad})"
        )

    def test_prepare_face_region_from_string(self, sample_face_image, tmp_path):
        """_prepare_face_region 也接受文件路径字符串."""
        file_path = tmp_path / "test_face.png"
        sample_face_image.save(str(file_path))
        face_image, bbox = BaseConsistencyNode._prepare_face_region(
            str(file_path)
        )
        assert isinstance(face_image, Image.Image), (
            "face_image should be a PIL.Image.Image even from file path"
        )


class TestComputeImageSimilarity:
    """_compute_image_similarity 静态方法测试."""

    def test_similarity_same_image(self, sample_face_image):
        """# T_CBASE_05：_compute_image_similarity 相同图片返回 1.0."""
        score = BaseConsistencyNode._compute_image_similarity(
            sample_face_image, sample_face_image
        )
        assert isinstance(score, float), "Similarity score should be a float"
        assert 0.0 <= score <= 1.0, (
            f"Similarity score should be in [0, 1], got {score}"
        )
        # 相同图片应该接近 1.0
        assert score > 0.99, (
            f"Same image similarity should be close to 1.0, got {score}"
        )

    def test_similarity_different_images(
        self, sample_face_image, sample_landscape_image
    ):
        """# T_CBASE_06：_compute_image_similarity 不同图片返回值 < 1.0."""
        score = BaseConsistencyNode._compute_image_similarity(
            sample_face_image, sample_landscape_image
        )
        assert isinstance(score, float), "Similarity score should be a float"
        assert 0.0 <= score <= 1.0, (
            f"Similarity score should be in [0, 1], got {score}"
        )
        assert score < 0.99, (
            f"Different images similarity should be < 1.0, got {score}"
        )

    def test_similarity_very_different_images(
        self, sample_face_image, sample_style_reference
    ):
        """_compute_image_similarity 对非常不同的图片应返回较低的值."""
        score = BaseConsistencyNode._compute_image_similarity(
            sample_face_image, sample_style_reference
        )
        assert score < 0.9, (
            f"Very different images should have low similarity, got {score}"
        )

    def test_similarity_symmetric(self, sample_face_image, sample_landscape_image):
        """_compute_image_similarity 应该是对称的."""
        score_ab = BaseConsistencyNode._compute_image_similarity(
            sample_face_image, sample_landscape_image
        )
        score_ba = BaseConsistencyNode._compute_image_similarity(
            sample_landscape_image, sample_face_image
        )
        assert score_ab == pytest.approx(score_ba, rel=1e-5), (
            f"Similarity should be symmetric: {score_ab} vs {score_ba}"
        )

    def test_similarity_with_string_inputs(self, sample_face_image, tmp_path):
        """_compute_image_similarity 也接受文件路径字符串."""
        file_path = tmp_path / "test_face.png"
        sample_face_image.save(str(file_path))
        score = BaseConsistencyNode._compute_image_similarity(
            str(file_path), sample_face_image
        )
        assert score > 0.99, (
            f"Similarity with string input should be close to 1.0, got {score}"
        )