# tests/phase2/conftest.py
"""Phase 2 测试公共 fixtures。

提供测试用 PIL 图片、mock 图像节点，以及 mock torch/diffusers 注入，
确保无 GPU 环境也能运行框架级测试。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, "/workspace/mosaic")


# ---------------------------------------------------------------------------
# Mock torch 注入（模块级，session 作用域）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_torch():
    """注入 mock torch 模块。

    若 Phase 1 conftest 已注入 mock torch（不含 Generator 等属性），
    在此补齐 Phase 2 图像节点所需的属性。
    """
    if "torch" not in sys.modules:
        mt = types.ModuleType("torch")
        mt.__spec__ = MagicMock()  # 避免 find_spec 报 '__spec__ is None'
        _ctx = MagicMock()
        _ctx.__enter__ = MagicMock(return_value=None)
        _ctx.__exit__ = MagicMock(return_value=None)
        mt.inference_mode = MagicMock(return_value=_ctx)
        mt.float16 = "float16"
        mt.float32 = "float32"
        mt.bfloat16 = "bfloat16"
        mt.Generator = MagicMock
        _mcuda = MagicMock()
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        mt.cuda = _mcuda
        sys.modules["torch"] = mt
        sys.modules["torch.cuda"] = _mcuda
    else:
        # Phase 1 可能已注入 mock torch，补齐 Phase 2 所需属性
        mt = sys.modules["torch"]
        if not hasattr(mt, "Generator"):
            mt.Generator = MagicMock
        if not hasattr(mt, "Tensor"):
            mt.Tensor = MagicMock
        if not hasattr(mt, "ones_like"):
            mt.ones_like = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "ones"):
            mt.ones = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "tensor"):
            mt.tensor = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "bfloat16"):
            mt.bfloat16 = "bfloat16"
        # 确保 cuda 子模块也有完整属性
        cuda = getattr(mt, "cuda", None)
        if cuda is not None and not hasattr(cuda, "get_device_properties"):
            cuda.get_device_properties = MagicMock()
        if cuda is not None and not hasattr(cuda, "memory_allocated"):
            cuda.memory_allocated = MagicMock(return_value=0)
    yield


# ---------------------------------------------------------------------------
# Mock diffusers 注入（按需）
# ---------------------------------------------------------------------------
def _inject_mock_diffusers():
    """在 sys.modules 中注入 mock diffusers，防止 import 时报错。"""
    if "diffusers" not in sys.modules:
        dm = types.ModuleType("diffusers")
        dm.__spec__ = MagicMock()  # 避免 find_spec 报 '__spec__ is None'
        dm.StableDiffusionXLPipeline = MagicMock()
        dm.StableDiffusionXLImg2ImgPipeline = MagicMock()
        dm.StableDiffusionXLInpaintPipeline = MagicMock()
        dm.StableDiffusionUpscalePipeline = MagicMock()
        dm.AutoPipelineForText2Image = MagicMock()
        dm.AutoPipelineForImage2Image = MagicMock()
        dm.AutoPipelineForInpainting = MagicMock()
        dm.scheduler_map = {}
        sys.modules["diffusers"] = dm

# ---------------------------------------------------------------------------
# Mock transformers（按需）
# ---------------------------------------------------------------------------
def _inject_mock_transformers():
    if "transformers" not in sys.modules:
        tm = types.ModuleType("transformers")
        tm.__spec__ = MagicMock()  # 避免 find_spec 报 '__spec__ is None'
        tm.AutoModelForImageSegmentation = MagicMock()
        tm.AutoModelForImageSegmentation.from_pretrained = MagicMock()
        sys.modules["transformers"] = tm

# 在模块加载时预注入 mock 模块
_inject_mock_diffusers()
_inject_mock_transformers()


# ---------------------------------------------------------------------------
# 测试图片 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image() -> Image.Image:
    """创建一张 512x512 的纯色测试图片。"""
    return Image.new("RGB", (512, 512), color=(128, 64, 200))


@pytest.fixture
def sample_mask() -> Image.Image:
    """创建一张 512x512 的 mask 图片（白色矩形遮罩在中间）。"""
    mask = Image.new("L", (512, 512), color=0)
    # 在中间画一个白色矩形
    for x in range(128, 384):
        for y in range(128, 384):
            mask.putpixel((x, y), 255)
    return mask


@pytest.fixture
def large_image() -> Image.Image:
    """创建一张 2048x2048 的大图。"""
    return Image.new("RGB", (2048, 2048), color=(255, 100, 50))


@pytest.fixture
def tiny_image() -> Image.Image:
    """创建一张 32x32 的小图。"""
    return Image.new("RGB", (32, 32), color=(0, 255, 0))


@pytest.fixture
def rgba_image() -> Image.Image:
    """创建一张 RGBA 模式的图片。"""
    return Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))


# ---------------------------------------------------------------------------
# Mock 图像节点 fixtures（用于框架级测试）
# ---------------------------------------------------------------------------
from mosaic.core.node import Node, NodeSpec
from mosaic.core.types import MosaicData


class _MockImageNode(Node):
    """Mock 图像节点：不需要真实模型，用于测试框架核心。"""

    name = "mock-image-node"
    domain = "image"
    description = "Mock image node for testing."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-image-node", delay=0.0, tag="", **kwargs):
        super().__init__(name=name, **kwargs)
        self._delay = delay
        self._tag = tag
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        import time

        if self._delay:
            time.sleep(self._delay)
        self._run_calls += 1
        content = input_data.get("content", "")
        return MosaicData(content=f"{content}->{self._tag}", tag=self._tag)

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _FailingNode(Node):
    """会抛出异常的节点。"""

    name = "failing-node"
    domain = "image"
    description = "Failing node"
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="failing-node", error_msg="test failure", **kwargs):
        super().__init__(name=name, **kwargs)
        self._error_msg = error_msg

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        raise RuntimeError(self._error_msg)

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


@pytest.fixture
def mock_image_node():
    """返回一个 Mock 图像节点实例。"""
    return _MockImageNode()


@pytest.fixture
def MockImageNode():
    """返回 Mock 图像节点类。"""
    return _MockImageNode


@pytest.fixture
def FailingNode():
    """返回会失败的节点类。"""
    return _FailingNode


# ---------------------------------------------------------------------------
# 调度器/事件总线 fixtures
# ---------------------------------------------------------------------------
from mosaic.core.events import EventBus
from mosaic.core.scheduler import Scheduler, set_scheduler


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """每个测试前后清空 model_cache，避免 mock 对象跨测试污染。"""
    from mosaic.core.model_cache import model_cache
    model_cache.clear()
    yield
    model_cache.clear()


@pytest.fixture
def fresh_bus():
    """新鲜的事件总线。"""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


@pytest.fixture
def cpu_scheduler(fresh_bus):
    """CPU 调度器。"""
    sched = Scheduler(bus=fresh_bus, device="cpu")
    set_scheduler(sched)
    return sched


# ---------------------------------------------------------------------------
# 注册表清理 fixture
# ---------------------------------------------------------------------------
from mosaic.core.registry import NodeRegistry


@pytest.fixture
def clear_registry():
    """返回干净的注册表实例。"""
    reg = NodeRegistry()
    return reg