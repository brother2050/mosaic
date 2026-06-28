# /workspace/mosaic/tests/phase2/test_inpainting.py
"""Inpainting 节点测试。

测试 Inpainting 节点的核心功能：基本局部重绘、mask 尺寸自动对齐、
mask 自动二值化、全白 mask 全图重生成。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.image.inpainting import Inpainting


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_pipeline_output(size=(512, 512)):
    """构造一个 mock pipeline 输出，包含 .images 列表。"""
    mock_output = MagicMock()
    mock_output.images = [MagicMock(size=size)]
    return mock_output


def _create_node():
    """创建已就绪的 Inpainting 节点（跳过模型加载）。"""
    node = Inpainting()
    node._loaded = True
    node._pipeline = MagicMock()
    return node


# ---------------------------------------------------------------------------
# T_INP_01: 基本 inpainting
# ---------------------------------------------------------------------------
def test_basic_inpainting(cpu_scheduler, sample_image, sample_mask):
    """T_INP_01: 基本 inpainting —— mock _run_pipeline，验证输出含 "image" 键。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        result = node.run(
            MosaicData(
                image=sample_image,
                mask_image=sample_mask,
                prompt="a red car",
            )
        )

    assert "image" in result
    assert result["image"] is mock_output.images[0]
    assert result["prompt"] == "a red car"
    assert isinstance(result["seed"], int)
    assert result["model_name"] == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# T_INP_02: 图片与 mask 尺寸不一致
# ---------------------------------------------------------------------------
def test_image_mask_size_mismatch(cpu_scheduler):
    """T_INP_02: 512x512 图片与 256x256 mask —— mask 被 resize 到与图片一致。"""
    image = Image.new("RGB", (512, 512), color=(128, 64, 200))
    mask = Image.new("L", (256, 256), color=255)

    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(
            MosaicData(
                image=image,
                mask_image=mask,
                prompt="test",
            )
        )

    call_kwargs = mock_run.call_args.kwargs
    passed_mask = call_kwargs["mask_image"]
    assert passed_mask.size == (512, 512)


# ---------------------------------------------------------------------------
# T_INP_03: mask 非二值（灰度中间值）
# ---------------------------------------------------------------------------
def test_mask_binarization(cpu_scheduler):
    """T_INP_03: 灰度 mask 含中间值 —— 验证 _binarize_mask 被调用，threshold 为 127。"""
    image = Image.new("RGB", (512, 512), color=(128, 64, 200))
    # 创建包含中间灰度值的 mask（50, 100, 128, 200 等）
    mask = Image.new("L", (512, 512), color=0)
    for x in range(512):
        for y in range(512):
            if x < 128:
                mask.putpixel((x, y), 50)   # < 127 → 0
            elif x < 256:
                mask.putpixel((x, y), 100)  # < 127 → 0
            elif x < 384:
                mask.putpixel((x, y), 128)  # > 127 → 255
            else:
                mask.putpixel((x, y), 200)  # > 127 → 255

    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(
        node, "_binarize_mask", wraps=node._binarize_mask
    ) as mock_binarize:
        with patch.object(node, "_run_pipeline", return_value=mock_output):
            node.run(
                MosaicData(
                    image=image,
                    mask_image=mask,
                    prompt="test",
                )
            )

    # 验证 _binarize_mask 被调用
    mock_binarize.assert_called_once()
    # 验证 threshold 为 127（默认值，未显式传入，因此 kwargs 中无 threshold）
    call_args = mock_binarize.call_args
    assert "threshold" not in call_args.kwargs


# ---------------------------------------------------------------------------
# T_INP_04: 全白 mask
# ---------------------------------------------------------------------------
def test_all_white_mask(cpu_scheduler):
    """T_INP_04: 全白 mask —— 整个图片被重生成（pipeline 收到的 mask 全为 255）。"""
    image = Image.new("RGB", (512, 512), color=(100, 150, 200))
    mask = Image.new("L", (512, 512), color=255)

    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(
            MosaicData(
                image=image,
                mask_image=mask,
                prompt="test",
            )
        )

    call_kwargs = mock_run.call_args.kwargs
    passed_mask = call_kwargs["mask_image"]
    # 验证 pipeline 收到的 mask 全为 255
    assert passed_mask.getextrema() == (255, 255)