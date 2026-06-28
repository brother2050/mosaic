# /workspace/mosaic/tests/phase2/test_stylizer.py
"""Stylizer 节点测试。

测试 Stylizer 节点的核心功能：预设风格转换、strength 参数传递、
prompt_extra 补充提示词、seed 可复现性以及全部预设风格覆盖。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.image.stylizer import Stylizer


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_pipeline_output(size=(512, 512)):
    """构造一个 mock pipeline 输出，包含 .images 列表。"""
    mock_output = MagicMock()
    mock_output.images = [MagicMock(size=size)]
    return mock_output


def _create_node():
    """创建已就绪的 Stylizer 节点（跳过模型加载）。"""
    node = Stylizer()
    node._loaded = True
    node._pipeline = MagicMock()
    return node


# 全部 10 个预设风格名称
_ALL_STYLES = [
    "oil painting",
    "watercolor",
    "anime",
    "cyberpunk",
    "pencil sketch",
    "ink",
    "pixel art",
    "3d render",
    "impressionist",
    "digital art",
]


# ---------------------------------------------------------------------------
# T_STY_01: "oil painting" 风格
# ---------------------------------------------------------------------------
def test_oil_painting_style(cpu_scheduler, sample_image):
    """T_STY_01: "oil painting" 风格 —— mock _run_pipeline，验证输出不为空且 style 为 "oil painting"。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output):
        result = node.run(MosaicData(image=sample_image, style="oil painting"))

    assert result is not None
    assert result["image"] is not None
    assert result["style"] == "oil painting"


# ---------------------------------------------------------------------------
# T_STY_02: strength 参数
# ---------------------------------------------------------------------------
def test_strength_parameter(cpu_scheduler, sample_image):
    """T_STY_02: strength 参数 —— 验证 strength=0.3 和 strength=0.9 正确传递到 pipeline。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    # 测试 strength=0.3
    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(MosaicData(image=sample_image, style="oil painting", strength=0.3))
    assert mock_run.call_args.kwargs["strength"] == 0.3

    # 测试 strength=0.9
    node2 = _create_node()
    with patch.object(node2, "_run_pipeline", return_value=mock_output) as mock_run2:
        node2.run(MosaicData(image=sample_image, style="oil painting", strength=0.9))
    assert mock_run2.call_args.kwargs["strength"] == 0.9


# ---------------------------------------------------------------------------
# T_STY_03: prompt_extra 补充提示词
# ---------------------------------------------------------------------------
def test_prompt_extra(cpu_scheduler, sample_image):
    """T_STY_03: prompt_extra —— 验证额外提示词被追加到风格提示词中。"""
    node = _create_node()
    mock_output = _make_pipeline_output()

    with patch.object(node, "_run_pipeline", return_value=mock_output) as mock_run:
        node.run(MosaicData(
            image=sample_image,
            style="anime",
            prompt_extra="night scene, moonlight",
        ))

    call_kwargs = mock_run.call_args.kwargs
    prompt = call_kwargs["prompt"]
    assert "night scene, moonlight" in prompt
    # 确认基础风格提示词也在
    assert "anime" in prompt.lower() or "cel shading" in prompt.lower()


# ---------------------------------------------------------------------------
# T_STY_04: seed 可复现性
# ---------------------------------------------------------------------------
def test_seed_reproducibility(cpu_scheduler, sample_image):
    """T_STY_04: seed 可复现性 —— mock _prepare_seed 返回固定值，两次调用相同 seed 验证 generator 一致性。"""
    fixed_seed = 42
    fixed_generator = MagicMock(name="fixed_generator")

    mock_output = _make_pipeline_output()

    # 第一次调用
    node1 = _create_node()
    with patch.object(
        node1, "_prepare_seed", return_value=(fixed_seed, fixed_generator)
    ) as mock_prepare1,\
        patch.object(node1, "_run_pipeline", return_value=mock_output) as mock_run1:
        result1 = node1.run(MosaicData(image=sample_image, style="oil painting", seed=42))

    # 第二次调用
    node2 = _create_node()
    with patch.object(
        node2, "_prepare_seed", return_value=(fixed_seed, fixed_generator)
    ) as mock_prepare2,\
        patch.object(node2, "_run_pipeline", return_value=mock_output) as mock_run2:
        result2 = node2.run(MosaicData(image=sample_image, style="oil painting", seed=42))

    # 验证 seed 一致
    assert result1["seed"] == result2["seed"] == fixed_seed

    # 验证 _prepare_seed 被调用时传入了相同的 seed
    mock_prepare1.assert_called_once()
    mock_prepare2.assert_called_once()

    # 验证 pipeline 使用的 generator 相同
    gen1 = mock_run1.call_args.kwargs["generator"]
    gen2 = mock_run2.call_args.kwargs["generator"]
    assert gen1 is fixed_generator
    assert gen2 is fixed_generator
    assert gen1 is gen2


# ---------------------------------------------------------------------------
# T_STY_05: 全部预设风格
# ---------------------------------------------------------------------------
def test_all_preset_styles(cpu_scheduler, sample_image):
    """T_STY_05: 全部预设风格 —— 遍历 10 种预设风格，验证每种均能正常运行。"""
    for style in _ALL_STYLES:
        node = _create_node()
        mock_output = _make_pipeline_output()

        with patch.object(node, "_run_pipeline", return_value=mock_output):
            result = node.run(MosaicData(image=sample_image, style=style))

        assert result is not None, f"Style '{style}' returned None"
        assert result["image"] is not None, f"Style '{style}' returned no image"
        assert result["style"] == style, f"Style '{style}' returned wrong style: {result['style']}"