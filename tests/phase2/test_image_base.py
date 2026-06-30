# /workspace/mosaic/tests/phase2/test_image_base.py
"""BaseImageNode 测试。"""

from unittest.mock import MagicMock

import pytest

from mosaic.core.types import MosaicData
from mosaic.nodes.image._base import BaseImageNode


# ---------------------------------------------------------------------------
# 测试用 BaseImageNode 具体子类
# ---------------------------------------------------------------------------
class _TestImageNode(BaseImageNode):
    """测试用 BaseImageNode 具体子类。

    实现了 ``_load_pipeline`` 与 ``run`` 两个抽象方法，
    允许在测试中直接实例化 BaseImageNode 的派生类。
    """

    name = "test-image"
    description = "Test image node"

    def _load_pipeline(self):
        self._pipeline = MagicMock()

    def run(self, input_data):
        return MosaicData()


# ---------------------------------------------------------------------------
# T_IBASE_01: 构造函数参数存储
# ---------------------------------------------------------------------------
def test_constructor_parameters_stored(cpu_scheduler, fresh_bus):
    """T_IBASE_01: 创建 BaseImageNode 具体子类，验证所有构造函数参数正确存储。

    验证 model, device, dtype, enable_attention_slicing, enable_vae_slicing,
    enable_model_cpu_offload 等参数均被正确赋值到对应的内部属性。
    """
    node = _TestImageNode(
        model="test-model-v1",
        device="cuda",
        dtype="float32",
        enable_attention_slicing=False,
        enable_vae_slicing=False,
        enable_model_cpu_offload=True,
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )

    assert node._model_name == "test-model-v1"
    # device="cuda" 在 CPU 环境下自动降级为 "cpu"
    assert node._device == "cpu"
    assert node._dtype_str == "float32"
    assert node._enable_attention_slicing is False
    assert node._enable_vae_slicing is False
    assert node._enable_model_cpu_offload is True
    assert node._scheduler is cpu_scheduler
    assert node._bus is fresh_bus


def test_constructor_default_values(cpu_scheduler, fresh_bus):
    """T_IBASE_01 补充: 验证构造函数默认值。

    CPU 环境下 device 和 dtype 会自动降级（cuda→cpu, float16→float32）。
    """
    node = _TestImageNode(scheduler=cpu_scheduler, bus=fresh_bus)

    assert node._model_name == "stabilityai/stable-diffusion-xl-base-1.0"
    # CPU 环境：cuda → cpu, float16 → float32（避免黑图）
    assert node._device == "cpu"
    assert node._dtype_str == "float32"
    assert node._enable_attention_slicing is True
    assert node._enable_vae_slicing is True
    assert node._enable_model_cpu_offload is False


# ---------------------------------------------------------------------------
# T_IBASE_02: _resolve_dtype() 解析
# ---------------------------------------------------------------------------
def test_resolve_dtype_float16(cpu_scheduler, fresh_bus):
    """T_IBASE_02: _resolve_dtype() 返回正确的 torch dtype — float16。

    CPU 环境下 float16 会被自动降级为 float32，此处验证降级行为。
    """
    import torch

    node = _TestImageNode(
        dtype="float16",
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )
    # CPU 环境下 auto_resolve_device_dtype 将 float16 降级为 float32
    assert node._dtype_str == "float32"
    result = node._resolve_dtype()
    assert result == torch.float32


def test_resolve_dtype_float32(cpu_scheduler, fresh_bus):
    """T_IBASE_02: _resolve_dtype() 返回正确的 torch dtype — float32。"""
    import torch

    node = _TestImageNode(
        dtype="float32",
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )
    result = node._resolve_dtype()
    assert result == torch.float32


def test_resolve_dtype_bfloat16(cpu_scheduler, fresh_bus):
    """T_IBASE_02: _resolve_dtype() 返回正确的 torch dtype — bfloat16。

    CPU 环境下 bfloat16 会被自动降级为 float32。
    """
    import torch

    node = _TestImageNode(
        dtype="bfloat16",
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )
    # CPU 环境下 auto_resolve_device_dtype 将 bfloat16 降级为 float32
    assert node._dtype_str == "float32"
    result = node._resolve_dtype()
    assert result == torch.float32


# ---------------------------------------------------------------------------
# T_IBASE_03: enable_attention_slicing 参数
# ---------------------------------------------------------------------------
def test_enable_attention_slicing_true(cpu_scheduler, fresh_bus):
    """T_IBASE_03: enable_attention_slicing=True 时属性正确存储。"""
    node = _TestImageNode(
        enable_attention_slicing=True,
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )
    assert node._enable_attention_slicing is True


def test_enable_attention_slicing_false(cpu_scheduler, fresh_bus):
    """T_IBASE_03: enable_attention_slicing=False 时属性正确存储。"""
    node = _TestImageNode(
        enable_attention_slicing=False,
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )
    assert node._enable_attention_slicing is False


# ---------------------------------------------------------------------------
# T_IBASE_04: device 自动降级（无 GPU 环境）
# ---------------------------------------------------------------------------
def test_device_fallback_no_gpu(cpu_scheduler, fresh_bus):
    """T_IBASE_04: 无 GPU 时（torch.cuda.is_available() 返回 False），
    _infer_device 应返回调度器的 device（cpu），而非构造函数中的 "cuda"。

    注意：auto_resolve_device_dtype 会在 __init__ 时将 "cuda" 降级为 "cpu"。
    """
    import torch

    # 确认 mock 环境中 GPU 不可用
    assert torch.cuda.is_available() is False

    node = _TestImageNode(
        device="cuda",
        scheduler=cpu_scheduler,
        bus=fresh_bus,
    )

    # auto_resolve_device_dtype 在 __init__ 时已将 "cuda" 降级为 "cpu"
    assert node._device == "cpu"

    # _infer_device 在 pipeline 未加载时应返回调度器的 device
    # 调度器 device 为 "cpu"（由 cpu_scheduler fixture 提供）
    inferred = node._infer_device()
    assert inferred == cpu_scheduler.device
    assert inferred == "cpu"