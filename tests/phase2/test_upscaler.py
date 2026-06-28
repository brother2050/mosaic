# /workspace/mosaic/tests/phase2/test_upscaler.py
"""Upscaler 节点测试。

测试 Upscaler 节点的核心功能：4x 超分辨率放大、高分辨率输入限制、
极小尺寸输入警告以及 describe() 元信息。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.image.upscaler import Upscaler


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_pipeline_output(size=(2048, 2048)):
    """构造一个 mock pipeline 输出，包含 .images 列表。"""
    mock_output = MagicMock()
    mock_output.images = [MagicMock(size=size)]
    return mock_output


def _create_node(cpu_scheduler):
    """创建已就绪的 Upscaler 节点（跳过模型加载）。"""
    node = Upscaler(scheduler=cpu_scheduler)
    node._loaded = True
    node._pipeline = MagicMock()
    return node


# ---------------------------------------------------------------------------
# T_UP_01: 4x 超分辨率放大
# ---------------------------------------------------------------------------
def test_4x_upscale(sample_image, cpu_scheduler):
    """T_UP_01: 4x 超分辨率放大。

    mock _run_pipeline 返回一张 2048x2048 的图片，验证 output_size 是
    原始尺寸（512x512）的 4 倍。
    """
    node = _create_node(cpu_scheduler)
    mock_output = _make_pipeline_output(size=(2048, 2048))

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        result = node.run(MosaicData(image=sample_image))

    assert result["image"] is mock_output.images[0]
    assert result["original_size"] == (512, 512)
    assert result["output_size"] == (2048, 2048)
    assert result["scale_factor"] == 4
    assert isinstance(result["seed"], int)
    assert result["model_name"] == "stabilityai/stable-diffusion-x4-upscaler"
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# T_UP_02: 高分辨率输入限制
# ---------------------------------------------------------------------------
def test_high_res_input_limited(large_image, cpu_scheduler):
    """T_UP_02: 已有高分辨率输入（2048x2048）—— 验证 _limit_image_size 被调用。

    输入 2048x2048 的大图，验证 _limit_image_size 被调用且传入 max_side=512，
    在进入 Pipeline 前将图片缩放到合理尺寸。
    """
    node = _create_node(cpu_scheduler)
    mock_output = _make_pipeline_output(size=(2048, 2048))

    with patch.object(node, "_limit_image_size", wraps=node._limit_image_size) as mock_limit:
        with patch.object(node, "_run_pipeline", return_value=mock_output):
            node.run(MosaicData(image=large_image))

    mock_limit.assert_called_once()
    # 提取调用参数：_limit_image_size(image, max_side=512)
    call_args, call_kwargs = mock_limit.call_args
    # 第一个位置参数是 image，max_side 作为关键字参数传入
    assert call_kwargs.get("max_side") == 512


# ---------------------------------------------------------------------------
# T_UP_03: 极小尺寸图片警告
# ---------------------------------------------------------------------------
def test_tiny_image_warning(tiny_image, cpu_scheduler, caplog):
    """T_UP_03: 极小尺寸图片（32x32）—— 验证 min 维度 < 64 时记录警告。

    输入 32x32 的 tiny_image，min(32, 32) = 32 < 64，应触发一条
    "very small" 警告日志。
    """
    node = _create_node(cpu_scheduler)
    mock_output = _make_pipeline_output(size=(128, 128))

    with patch.object(node, "_run_pipeline", return_value=mock_output):
        with caplog.at_level(logging.WARNING):
            node.run(MosaicData(image=tiny_image))

    assert "very small" in caplog.text
    assert "32x32" in caplog.text or "32" in caplog.text


# ---------------------------------------------------------------------------
# T_UP_04: describe() 返回正确的节点元信息
# ---------------------------------------------------------------------------
def test_describe(cpu_scheduler):
    """T_UP_04: describe() 返回正确的 name/domain/description/version 等。

    验证 NodeSpec 包含 upscaler 节点的正确元信息，包括模型名称、
    许可证与显存估算。
    """
    node = Upscaler(scheduler=cpu_scheduler)
    spec = node.describe()

    assert spec.name == "upscaler"
    assert spec.domain == "image"
    assert "Upscale" in spec.description
    assert spec.version == "0.1.0"
    assert "image" in spec.input_types
    assert "mosaic" in spec.input_types
    assert "image" in spec.output_types
    assert spec.model_info is not None
    assert "name" in spec.model_info
    assert spec.model_info["name"] == "stabilityai/stable-diffusion-x4-upscaler"
    assert "license" in spec.model_info
    assert "OpenRAIL" in spec.model_info["license"]
    assert "vram_gb" in spec.model_info
    assert spec.model_info["vram_gb"] == 6.0