# tests/phase2/test_image_types.py
"""Phase 2 图像类型测试。

覆盖 ImageData 的创建、序列化/反序列化、不同尺寸图像处理及 RGBA 模式保留。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from mosaic.core.types import ImageData, MosaicData


# ===========================================================================
# T_IMGTYPE_01: ImageData 创建并验证 image 属性与 size 自动检测
# ===========================================================================
class TestImageDataCreation:
    """ImageData 创建测试。"""

    def test_create_with_sample_image(self, sample_image):
        """T_IMGTYPE_01: 使用 sample_image fixture 创建 ImageData，验证 image 属性和 size 自动检测。"""
        idata = ImageData(image=sample_image)
        assert isinstance(idata.image, PILImage.Image)
        assert idata.image is sample_image
        assert idata.size == (512, 512)
        assert idata.data_type == "image"

    def test_create_with_explicit_metadata(self, sample_image):
        """T_IMGTYPE_01: 创建 ImageData 时携带 metadata。"""
        idata = ImageData(image=sample_image, metadata={"source": "test", "format": "png"})
        assert idata.metadata == {"source": "test", "format": "png"}

    def test_create_without_image(self):
        """T_IMGTYPE_01: 创建不带图像的 ImageData，image 为 None。"""
        idata = ImageData()
        assert idata.image is None
        assert idata.size is None

    def test_create_with_explicit_size(self):
        """T_IMGTYPE_01: 显式指定 size 创建 ImageData（无图像）。"""
        idata = ImageData(size=(800, 600))
        assert idata.size == (800, 600)
        assert idata.image is None

    def test_isinstance_of_mosaic_data(self, sample_image):
        """T_IMGTYPE_01: ImageData 是 MosaicData 的子类实例。"""
        idata = ImageData(image=sample_image)
        assert isinstance(idata, MosaicData)


# ===========================================================================
# T_IMGTYPE_02: ImageData 序列化/反序列化 roundtrip
# ===========================================================================
class TestImageDataSerialization:
    """ImageData 序列化/反序列化测试。"""

    def test_roundtrip_preserves_pil_image(self, sample_image):
        """T_IMGTYPE_02: 通过 to_dict()/from_dict() 往返，PIL 图像完整保留。"""
        idata = ImageData(image=sample_image, metadata={"desc": "test roundtrip"})
        d = idata.to_dict()
        restored = ImageData.from_dict(d)

        assert isinstance(restored, ImageData)
        assert isinstance(restored.image, PILImage.Image)
        assert restored.size == (512, 512)
        assert restored.image.size == (512, 512)
        assert restored.metadata == {"desc": "test roundtrip"}

    def test_roundtrip_preserves_image_data(self, sample_image):
        """T_IMGTYPE_02: 往返后图像像素数据一致。"""
        idata = ImageData(image=sample_image)
        d = idata.to_dict()
        restored = ImageData.from_dict(d)

        # 逐像素比对
        original_pixels = list(np.array(sample_image).flat)
        restored_pixels = list(np.array(restored.image).flat)
        assert original_pixels == restored_pixels

    def test_roundtrip_data_type_marker(self, sample_image):
        """T_IMGTYPE_02: 序列化字典中包含正确的 __data_type__ 标记。"""
        idata = ImageData(image=sample_image)
        d = idata.to_dict()
        assert d["__data_type__"] == "image"

    def test_roundtrip_without_image(self):
        """T_IMGTYPE_02: 无图像的 ImageData 往返。"""
        idata = ImageData(size=(100, 200), metadata={"key": "val"})
        d = idata.to_dict()
        restored = ImageData.from_dict(d)
        assert isinstance(restored, ImageData)
        assert restored.image is None
        assert restored.size == (100, 200)
        assert restored.metadata == {"key": "val"}


# ===========================================================================
# T_IMGTYPE_03: 不同尺寸图像处理
# ===========================================================================
class TestImageDataSizes:
    """不同尺寸图像测试。"""

    def test_size_512x512(self, sample_image):
        """T_IMGTYPE_03: 512x512 图像，size 属性正确。"""
        idata = ImageData(image=sample_image)
        assert idata.size == (512, 512)
        assert idata.image.size == (512, 512)

    def test_size_32x32(self, tiny_image):
        """T_IMGTYPE_03: 32x32 图像，size 属性正确。"""
        idata = ImageData(image=tiny_image)
        assert idata.size == (32, 32)
        assert idata.image.size == (32, 32)

    def test_size_roundtrip_512x512(self, sample_image):
        """T_IMGTYPE_03: 512x512 图像往返后尺寸不变。"""
        idata = ImageData(image=sample_image)
        d = idata.to_dict()
        restored = ImageData.from_dict(d)
        assert restored.size == (512, 512)
        assert restored.image.size == (512, 512)

    def test_size_roundtrip_32x32(self, tiny_image):
        """T_IMGTYPE_03: 32x32 图像往返后尺寸不变。"""
        idata = ImageData(image=tiny_image)
        d = idata.to_dict()
        restored = ImageData.from_dict(d)
        assert restored.size == (32, 32)
        assert restored.image.size == (32, 32)


# ===========================================================================
# T_IMGTYPE_04: RGBA 图像处理
# ===========================================================================
class TestImageDataRGBA:
    """RGBA 图像测试。"""

    def test_create_rgba_image_data(self, rgba_image):
        """T_IMGTYPE_04: 创建 RGBA ImageData，验证 RGBA 模式。"""
        assert rgba_image.mode == "RGBA"

        idata = ImageData(image=rgba_image)
        assert idata.image.mode == "RGBA"
        assert idata.size == (256, 256)
        assert idata.data_type == "image"

    def test_rgba_roundtrip_preserves_mode(self, rgba_image):
        """T_IMGTYPE_04: RGBA 图像往返后模式保持为 RGBA。"""
        idata = ImageData(image=rgba_image, metadata={"alpha": True})
        d = idata.to_dict()
        restored = ImageData.from_dict(d)

        assert isinstance(restored, ImageData)
        assert isinstance(restored.image, PILImage.Image)
        assert restored.image.mode == "RGBA"
        assert restored.size == (256, 256)
        assert restored.metadata == {"alpha": True}

    def test_rgba_roundtrip_preserves_pixel_data(self, rgba_image):
        """T_IMGTYPE_04: RGBA 图像往返后像素数据一致（含 alpha 通道）。"""
        idata = ImageData(image=rgba_image)
        d = idata.to_dict()
        restored = ImageData.from_dict(d)

        original_pixels = list(np.array(rgba_image).flat)
        restored_pixels = list(np.array(restored.image).flat)
        assert original_pixels == restored_pixels