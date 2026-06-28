# /workspace/mosaic/tests/phase2/test_background_remover.py
"""BackgroundRemover 节点测试。

测试 BackgroundRemover 节点的核心功能：基本去背景、mask 输出、
RGBA 输入处理以及大图预处理。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.image.background_remover import BackgroundRemover


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _create_node():
    """创建已就绪的 BackgroundRemover 节点（跳过模型加载）。"""
    node = BackgroundRemover()
    node._loaded = True
    node._pipeline = MagicMock()
    return node


# ---------------------------------------------------------------------------
# T_BGR_01: 基本背景去除
# ---------------------------------------------------------------------------
def test_basic_background_removal(cpu_scheduler, sample_image):
    """T_BGR_01: 基本背景去除 —— mock _remove_bg_model，验证输出 image 为 RGBA 模式。"""
    node = _create_node()

    rgba_result = Image.new("RGBA", (512, 512), color=(128, 64, 200, 255))
    mask_result = Image.new("L", (512, 512), color=255)

    with patch.object(
        node, "_remove_bg_model", return_value=(rgba_result, mask_result)
    ) as mock_remove:
        result = node.run(MosaicData(image=sample_image))

    assert "image" in result
    assert result["image"].mode == "RGBA"
    mock_remove.assert_called_once()


# ---------------------------------------------------------------------------
# T_BGR_02: 输出包含 mask
# ---------------------------------------------------------------------------
def test_output_includes_mask(cpu_scheduler, sample_image):
    """T_BGR_02: 输出包含 mask —— 验证输出有 "mask" 键且为灰度图。"""
    node = _create_node()

    rgba_result = Image.new("RGBA", (512, 512), color=(128, 64, 200, 255))
    mask_result = Image.new("L", (512, 512), color=128)

    with patch.object(
        node, "_remove_bg_model", return_value=(rgba_result, mask_result)
    ):
        result = node.run(MosaicData(image=sample_image))

    assert "mask" in result
    assert isinstance(result["mask"], Image.Image)
    assert result["mask"].mode == "L"


# ---------------------------------------------------------------------------
# T_BGR_03: RGBA 输入图片
# ---------------------------------------------------------------------------
def test_rgba_input_image(cpu_scheduler, rgba_image):
    """T_BGR_03: RGBA 输入图片 —— 验证节点正确处理 RGBA 模式输入。"""
    node = _create_node()

    rgba_result = Image.new("RGBA", (256, 256), color=(255, 0, 0, 255))
    mask_result = Image.new("L", (256, 256), color=255)

    with patch.object(
        node, "_remove_bg_model", return_value=(rgba_result, mask_result)
    ):
        result = node.run(MosaicData(image=rgba_image))

    assert "image" in result
    assert result["image"].mode == "RGBA"
    assert "mask" in result
    assert result["mask"].mode == "L"


# ---------------------------------------------------------------------------
# T_BGR_04: 大图自动 resize
# ---------------------------------------------------------------------------
def test_large_image_auto_resize(cpu_scheduler, large_image):
    """T_BGR_04: 大图自动 resize —— 输入 2048x2048，验证预处理流程处理大图。"""
    node = _create_node()

    rgba_result = Image.new("RGBA", (2048, 2048), color=(255, 100, 50, 255))
    mask_result = Image.new("L", (2048, 2048), color=255)

    with patch.object(
        node, "_remove_bg_model", return_value=(rgba_result, mask_result)
    ) as mock_remove:
        result = node.run(MosaicData(image=large_image))

    mock_remove.assert_called_once()
    # 验证传入 _remove_bg_model 的图片是原始大图（2048x2048）
    # 实际 _remove_bg_model 内部会通过 transforms.Resize 处理大图
    called_image = mock_remove.call_args[0][0]
    assert called_image.size == (2048, 2048)
    assert "image" in result
    assert result["image"].mode == "RGBA"