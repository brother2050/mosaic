# tests/phase6/conftest.py
"""Phase 6 测试公共 fixtures。

提供一致性域测试所需的 mock 环境（torch、diffusers、PIL 合成图片等）。
全部使用合成数据，不依赖外部文件或真实模型。

关键 mock 注入（session 级别）：
- torch -> mock Tensor, nn.Module, Generator, inference_mode, randn, cat 等
- diffusers -> mock StableDiffusionXLPipeline, StableDiffusionPipeline,
  StableDiffusionXLInstantIDPipeline, StableDiffusionXLControlNetPipeline,
  ControlNetModel, PhotoMakerPipeline 等
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/mosaic")


# ---------------------------------------------------------------------------
# 辅助：创建 mock Pipeline 通用行为
# ---------------------------------------------------------------------------
class _MockPipelineOutput:
    """模拟 diffusers pipeline 的 __call__ 返回值."""

    def __init__(self, images=None):
        self.images = images or [_make_dummy_pil_image()]


def _make_dummy_pil_image(size=(512, 512), color=(100, 150, 200)):
    """创建用于 mock pipeline 输出的简单 PIL 图片."""
    return Image.new("RGB", size, color=color)


class _MockVAE:
    """mock VAE 模块."""

    def __init__(self):
        self.config = MagicMock()

    def encode(self, *args, **kwargs):
        result = MagicMock()
        result.latent_dist = MagicMock()
        result.latent_dist.sample = MagicMock(return_value=MagicMock())
        return result

    def decode(self, *args, **kwargs):
        result = MagicMock()
        result.sample = MagicMock(return_value=MagicMock())
        return result

    def to(self, *args, **kwargs):
        return self

    def enable_slicing(self):
        pass

    def enable_tiling(self):
        pass


class _MockScheduler:
    """mock Scheduler 模块."""

    def __init__(self):
        self.timesteps = list(range(1000, 0, -50))
        self.config = MagicMock()

    def set_timesteps(self, *args, **kwargs):
        pass

    def add_noise(self, *args, **kwargs):
        return MagicMock()

    def step(self, *args, **kwargs):
        return MagicMock()

    def to(self, *args, **kwargs):
        return self


class _MockUNet:
    """mock UNet 模块."""

    def __init__(self):
        self.attn_processors = {}
        self.config = MagicMock()

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return MagicMock()

    def named_modules(self):
        """返回空迭代器（模拟 UNet 无特定注意模块的情况）。"""
        return iter([])

    def set_attn_processor(self, processors):
        self.attn_processors = processors


class _MockImageEncoder:
    """mock Image Encoder（IP-Adapter 等）. """

    def __init__(self):
        self.config = MagicMock()

    def to(self, *args, **kwargs):
        return self


class _MockSafetyChecker:
    """mock Safety Checker."""

    def __init__(self):
        self.config = MagicMock()

    def to(self, *args, **kwargs):
        return self


class _MockFeatureExtractor:
    """mock Feature Extractor."""

    def __init__(self):
        self.config = MagicMock()

    def to(self, *args, **kwargs):
        return self


class _MockBasePipeline:
    """所有 mock diffusers pipeline 的基类."""

    def __init__(self):
        self.unet = _MockUNet()
        self.scheduler = _MockScheduler()
        self.vae = _MockVAE()
        self.image_encoder = _MockImageEncoder()
        self.safety_checker = _MockSafetyChecker()
        self.feature_extractor = _MockFeatureExtractor()
        self.tokenizer = MagicMock()
        self.tokenizer_2 = MagicMock()
        self.text_encoder = MagicMock()
        self.text_encoder_2 = MagicMock()
        self._device = "cpu"

    def to(self, device):
        self._device = device
        return self

    def __call__(self, **kwargs):
        return _MockPipelineOutput()

    def enable_attention_slicing(self):
        pass

    def enable_vae_slicing(self):
        pass

    def enable_vae_tiling(self):
        pass

    def load_ip_adapter(self, *args, **kwargs):
        pass

    def set_ip_adapter_scale(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


# ---------------------------------------------------------------------------
# Mock torch 注入（session 作用域）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_torch():
    """注入/补齐 mock torch 模块。

    兼容 Phase 1/2/5 可能已注入的 mock torch，补齐 Phase 6 所需属性。
    """
    if "torch" not in sys.modules:
        mt = types.ModuleType("torch")
        mt.__spec__ = MagicMock()
        _ctx = MagicMock()
        _ctx.__enter__ = MagicMock(return_value=None)
        _ctx.__exit__ = MagicMock(return_value=None)
        mt.inference_mode = MagicMock(return_value=_ctx)
        mt.no_grad = MagicMock(return_value=_ctx)
        mt.float16 = "float16"
        mt.float32 = "float32"
        mt.bfloat16 = "bfloat16"
        mt.Generator = MagicMock
        mt.Tensor = MagicMock
        mt.nn = types.ModuleType("torch.nn")
        mt.nn.Module = MagicMock
        mt.ones_like = MagicMock(return_value=MagicMock())
        mt.ones = MagicMock(return_value=MagicMock())
        mt.zeros_like = MagicMock(return_value=MagicMock())
        mt.zeros = MagicMock(return_value=MagicMock())
        mt.tensor = MagicMock(return_value=MagicMock())
        mt.from_numpy = MagicMock(return_value=MagicMock())
        mt.randn = MagicMock(return_value=MagicMock())
        mt.cat = MagicMock(return_value=MagicMock())
        mt.stack = MagicMock(return_value=MagicMock())
        mt.where = MagicMock(return_value=MagicMock())
        mt.clamp = MagicMock(return_value=MagicMock())
        mt.device = MagicMock(return_value="cpu")
        # torch.cuda 子模块
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.__spec__ = MagicMock()
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        _mcuda.empty_cache = MagicMock()
        _mcuda.device_count = MagicMock(return_value=0)
        mt.cuda = _mcuda
        sys.modules["torch"] = mt
        sys.modules["torch.nn"] = mt.nn
        sys.modules["torch.cuda"] = _mcuda
    else:
        mt = sys.modules["torch"]
        if not hasattr(mt, "Generator"):
            mt.Generator = MagicMock
        if not hasattr(mt, "Tensor"):
            mt.Tensor = MagicMock
        if not hasattr(mt, "nn"):
            mt.nn = types.ModuleType("torch.nn")
            mt.nn.Module = MagicMock
            sys.modules["torch.nn"] = mt.nn
        if not hasattr(mt, "ones_like"):
            mt.ones_like = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "ones"):
            mt.ones = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "zeros_like"):
            mt.zeros_like = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "zeros"):
            mt.zeros = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "tensor"):
            mt.tensor = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "from_numpy"):
            mt.from_numpy = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "randn"):
            mt.randn = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "cat"):
            mt.cat = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "stack"):
            mt.stack = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "where"):
            mt.where = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "clamp"):
            mt.clamp = MagicMock(return_value=MagicMock())
        if not hasattr(mt, "device"):
            mt.device = MagicMock(return_value="cpu")
        if not hasattr(mt, "bfloat16"):
            mt.bfloat16 = "bfloat16"
        # 补齐 cuda 子模块
        cuda = getattr(mt, "cuda", None)
        if cuda is None:
            _mcuda = types.ModuleType("torch.cuda")
            _mcuda.__spec__ = MagicMock()
            _mcuda.is_available = MagicMock(return_value=False)
            _mcuda.get_device_properties = MagicMock()
            _mcuda.memory_allocated = MagicMock(return_value=0)
            _mcuda.empty_cache = MagicMock()
            _mcuda.device_count = MagicMock(return_value=0)
            mt.cuda = _mcuda
            sys.modules["torch.cuda"] = _mcuda
        elif not hasattr(cuda, "device_count"):
            cuda.device_count = MagicMock(return_value=0)
    yield


# ---------------------------------------------------------------------------
# Mock diffusers 注入（session 作用域）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_diffusers():
    """注入 mock diffusers 模块，提供所有一致性域使用的 Pipeline 类."""
    if "diffusers" not in sys.modules:
        dm = types.ModuleType("diffusers")
        dm.__spec__ = MagicMock()
    else:
        dm = sys.modules["diffusers"]

    # 注入所有需要的 Pipeline 类
    # 每个类都是 mock 类，支持 from_pretrained 返回自身实例
    _pipeline_classes = {
        "StableDiffusionXLPipeline": _MockBasePipeline,
        "StableDiffusionPipeline": _MockBasePipeline,
        "StableDiffusionXLInstantIDPipeline": _MockBasePipeline,
        "StableDiffusionXLControlNetPipeline": _MockBasePipeline,
        "StableDiffusionXLImg2ImgPipeline": _MockBasePipeline,
        "StableDiffusionXLInpaintPipeline": _MockBasePipeline,
        "StableDiffusionUpscalePipeline": _MockBasePipeline,
        "ControlNetModel": _MockBasePipeline,
        "PhotoMakerPipeline": _MockBasePipeline,
        "AutoencoderKL": _MockBasePipeline,
        "DPMSolverMultistepScheduler": _MockBasePipeline,
        "EulerDiscreteScheduler": _MockBasePipeline,
    }

    for name, cls in _pipeline_classes.items():
        setattr(dm, name, cls)

    if "diffusers" not in sys.modules:
        sys.modules["diffusers"] = dm
    yield


# ---------------------------------------------------------------------------
# Mock transformers 注入（session 作用域）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_transformers():
    """注入 mock transformers 模块."""
    if "transformers" not in sys.modules:
        tm = types.ModuleType("transformers")
        tm.__spec__ = MagicMock()
        tm.AutoModelForCausalLM = MagicMock()
        tm.AutoModelForCausalLM.from_pretrained = MagicMock()
        tm.AutoTokenizer = MagicMock()
        tm.AutoTokenizer.from_pretrained = MagicMock()
        tm.AutoModel = MagicMock()
        tm.AutoModel.from_pretrained = MagicMock()
        tm.CLIPImageProcessor = MagicMock()
        tm.CLIPImageProcessor.from_pretrained = MagicMock()
        tm.CLIPVisionModelWithProjection = MagicMock()
        tm.CLIPVisionModelWithProjection.from_pretrained = MagicMock()
        sys.modules["transformers"] = tm
    else:
        tm = sys.modules["transformers"]
        if not hasattr(tm, "CLIPImageProcessor"):
            tm.CLIPImageProcessor = MagicMock()
            tm.CLIPImageProcessor.from_pretrained = MagicMock()
        if not hasattr(tm, "CLIPVisionModelWithProjection"):
            tm.CLIPVisionModelWithProjection = MagicMock()
            tm.CLIPVisionModelWithProjection.from_pretrained = MagicMock()
    yield


# ---------------------------------------------------------------------------
# Mock insightface 注入（按需，session 作用域）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_insightface():
    """注入 mock insightface 模块，防止人脸检测导入失败."""
    if "insightface" not in sys.modules:
        fi = types.ModuleType("insightface")
        fi.__spec__ = MagicMock()
        fi_app = types.ModuleType("insightface.app")
        fi_app.__spec__ = MagicMock()
        fi.app = fi_app
        sys.modules["insightface"] = fi
        sys.modules["insightface.app"] = fi_app
    yield


# ---------------------------------------------------------------------------
# 合成图片 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_face_image():
    """合成人脸图片（PIL，512x512，椭圆肤色块 + 模拟五官位置的色块）。"""
    img = Image.new("RGB", (512, 512), (200, 180, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([156, 106, 356, 356], fill=(255, 220, 180))  # 脸
    draw.ellipse([200, 180, 240, 220], fill=(50, 50, 50))     # 左眼
    draw.ellipse([270, 180, 310, 220], fill=(50, 50, 50))     # 右眼
    draw.ellipse([240, 240, 270, 255], fill=(200, 150, 130))   # 鼻子
    draw.arc([220, 260, 290, 300], 0, 180, fill=(180, 80, 80), width=3)  # 嘴
    return img


@pytest.fixture
def sample_landscape_image():
    """合成风景图片（渐变天空蓝色 + 绿色地面）。"""
    img = Image.new("RGB", (512, 512), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    # 绿色地面
    draw.rectangle([0, 256, 512, 512], fill=(34, 139, 34))
    # 太阳
    draw.ellipse([400, 30, 480, 110], fill=(255, 255, 0))
    return img


@pytest.fixture
def sample_style_reference():
    """合成风格参考图片（彩色条纹纹理）。"""
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
    ]
    stripe_height = 512 // len(colors)
    for i, color in enumerate(colors):
        y0 = i * stripe_height
        y1 = (i + 1) * stripe_height
        draw.rectangle([0, y0, 512, y1], fill=color)
    return img


@pytest.fixture
def sample_prompts():
    """返回 5 个不同场景的 prompt 列表。"""
    return [
        "a portrait photo of a person wearing a suit, studio lighting",
        "a landscape painting of mountains at sunset, oil on canvas",
        "a cyberpunk city street at night, neon lights, rain",
        "a cute cartoon cat sitting on a bookshelf, Pixar style",
        "an abstract geometric pattern with vibrant colors, digital art",
    ]


# ---------------------------------------------------------------------------
# 调度器/事件总线 fixtures
# ---------------------------------------------------------------------------
from mosaic.core.events import EventBus
from mosaic.core.scheduler import Scheduler, set_scheduler


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
# 额外 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image():
    """创建一张 512x512 的通用测试图片。"""
    return Image.new("RGB", (512, 512), color=(128, 64, 200))


@pytest.fixture
def single_prompt():
    """返回单个提示词列表（用于单帧测试）。"""
    return ["a young woman with black hair standing in a garden"]