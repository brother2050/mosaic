# /workspace/mosaic/tests/phase2/test_image_to_image.py
"""ImageToImage 节点测试。

测试 ImageToImage 节点的核心功能：基本图生图、strength 参数边界、
图像尺寸对齐以及 describe() 元信息。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.image.image_to_image import ImageToImage


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_pipeline_output(size=(512, 512)):
    """构造一个 mock pipeline 输出，包含 .images 列表。"""
    mock_output = MagicMock()
    mock_output.images = [MagicMock(size=size)]
    return mock_output


def _create_node():
    """创建已就绪的 ImageToImage 节点（跳过模型加载）。"""
    node = ImageToImage()
    node._loaded = True
    node._pipeline = MagicMock()
    return node


# ---------------------------------------------------------------------------
# T_I2I_01: 基本图生图
# ---------------------------------------------------------------------------
def test_basic_image_to_image(cpu_scheduler, sample_image):
    """T_I2I_01: 基本图生图 —— mock _run_pipeline，验证输出含 "image" 键。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        result = node.run(MosaicData(image=sample_image, prompt="a cat"))

    assert "image" in result
    assert result["image"] is mock_output.images[0]
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# T_I2I_02: strength=0
# ---------------------------------------------------------------------------
def test_strength_zero(cpu_scheduler, sample_image):
    """T_I2I_02: strength=0 —— 验证 pipeline 收到 strength=0.0。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(MosaicData(image=sample_image, prompt="test", strength=0))

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["strength"] == 0.0


# ---------------------------------------------------------------------------
# T_I2I_03: strength=1
# ---------------------------------------------------------------------------
def test_strength_one(cpu_scheduler, sample_image):
    """T_I2I_03: strength=1 —— 验证 pipeline 收到 strength=1.0。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(MosaicData(image=sample_image, prompt="test", strength=1))

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["strength"] == 1.0


# ---------------------------------------------------------------------------
# T_I2I_04: 输入图片尺寸不是 8 的倍数
# ---------------------------------------------------------------------------
def test_image_not_multiple_of_8(cpu_scheduler):
    """T_I2I_04: 100x100 输入图片 —— 验证 _resize_to_multiple_of_8 生效（96x96）。"""
    img = Image.new("RGB", (100, 100), color=(128, 64, 200))

    node = _create_node()
    mock_output = _make_pipeline_output(size=(96, 96))

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(MosaicData(image=img, prompt="test"))

    call_kwargs = mock_run.call_args.kwargs
    passed_image = call_kwargs["image"]
    assert passed_image.size == (96, 96)


# ---------------------------------------------------------------------------
# T_I2I_05: describe() 返回正确的节点元信息
# ---------------------------------------------------------------------------
def test_describe(cpu_scheduler):
    """T_I2I_05: describe() 返回正确的 name/domain/description/version 等。"""
    node = ImageToImage()
    spec = node.describe()

    assert spec.name == "image-to-image"
    assert spec.domain == "image"
    assert "Transform an input image" in spec.description
    assert spec.version == "0.1.0"
    assert "image" in spec.input_types
    assert "mosaic" in spec.input_types
    assert "image" in spec.output_types
    assert spec.model_info is not None
    assert "name" in spec.model_info
    assert spec.model_info["name"] == "stabilityai/stable-diffusion-xl-refiner-1.0"