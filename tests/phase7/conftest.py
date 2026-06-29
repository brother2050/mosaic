# tests/phase7/conftest.py
"""Phase 7 数字人域测试公共 fixtures。

提供 session 级别的 mock 环境（torch / diffusers / transformers / insightface）
与合成数据 fixtures（avatar 图片、短音频、驱动视频、表情参数、动作数据等）。
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mosaic.core.types import AudioData, MotionData, MosaicData


# ============================================================================
# Session 级 mock 注入
# ============================================================================
@pytest.fixture(scope="session", autouse=True)
def _mock_torch() -> None:
    """Mock torch 模块（session 级别，避免加载真实 torch）。"""
    mock = MagicMock()
    mock.float16 = type("dtype", (), {"__repr__": lambda s: "torch.float16"})()
    mock.float32 = type("dtype", (), {"__repr__": lambda s: "torch.float32"})()
    mock.bfloat16 = type("dtype", (), {"__repr__": lambda s: "torch.bfloat16"})()
    mock.cuda.is_available.return_value = False
    mock.cuda.empty_cache = MagicMock()
    mock.Generator.return_value = MagicMock()
    mock.inference_mode = MagicMock()
    mock.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    mock.inference_mode.return_value.__exit__ = MagicMock(return_value=None)
    mock.randn.return_value = np.zeros((1, 3, 256, 256), dtype=np.float32)
    mock.cat = lambda tensors, dim=0: np.concatenate(
        [np.asarray(t) for t in tensors], axis=dim
    )
    mock.stack = lambda tensors, dim=0: np.stack(
        [np.asarray(t) for t in tensors], axis=dim
    )
    mock.where = lambda cond, x, y: np.where(cond, np.asarray(x), np.asarray(y))
    mock.clamp = lambda x, lo, hi: np.clip(np.asarray(x), lo, hi)
    mock.load = MagicMock()
    mock.Tensor = MagicMock()
    mock.nn.Module = type("Module", (), {})
    mock.no_grad = MagicMock()
    mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    sys.modules["torch"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_diffusers() -> None:
    """Mock diffusers 模块（session 级别）。"""
    mock = MagicMock()

    def _make_pipeline(name: str) -> type:
        """创建模拟 Pipeline 类。"""

        class _MockPipeline:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.unet = MagicMock()
                self.vae = MagicMock()
                self.scheduler = MagicMock()
                self.attn_processors: dict[str, Any] = {}

            def to(self, device: str) -> "_MockPipeline":
                return self

            def enable_attention_slicing(self) -> None:
                pass

            def enable_vae_slicing(self) -> None:
                pass

            def load_ip_adapter(self, *args: Any, **kwargs: Any) -> None:
                pass

            def set_ip_adapter_scale(self, scale: float) -> None:
                pass

            def __call__(self, **kwargs: Any) -> Any:
                return self

            @classmethod
            def from_pretrained(
                cls, pretrained_model_name_or_path: str, **kwargs: Any
            ) -> "_MockPipeline":
                return cls()

        _MockPipeline.__name__ = name
        return _MockPipeline

    mock.StableDiffusionXLPipeline = _make_pipeline("StableDiffusionXLPipeline")
    mock.StableDiffusionPipeline = _make_pipeline("StableDiffusionPipeline")
    mock.StableDiffusionXLInstantIDPipeline = _make_pipeline(
        "StableDiffusionXLInstantIDPipeline"
    )
    mock.StableDiffusionXLControlNetPipeline = _make_pipeline(
        "StableDiffusionXLControlNetPipeline"
    )
    mock.ControlNetModel = _make_pipeline("ControlNetModel")
    mock.PhotoMakerPipeline = _make_pipeline("PhotoMakerPipeline")
    mock.LivePortraitPipeline = _make_pipeline("LivePortraitPipeline")
    sys.modules["diffusers"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_transformers() -> None:
    """Mock transformers 模块（session 级别）。"""
    mock = MagicMock()
    mock.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    mock.AutoModel.from_pretrained.return_value = MagicMock()
    mock.pipeline.return_value = MagicMock()
    mock.Wav2Vec2Model.from_pretrained.return_value = MagicMock()
    mock.Wav2Vec2Processor.from_pretrained.return_value = MagicMock()
    mock.CLIPImageProcessor.from_pretrained.return_value = MagicMock()
    mock.CLIPVisionModelWithProjection.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_insightface() -> None:
    """Mock insightface 模块（session 级别）。"""
    mock = MagicMock()
    face_app = MagicMock()
    # 模拟检测到一张人脸
    face_obj = MagicMock()
    face_obj.bbox = np.array([100, 80, 400, 420], dtype=np.float32)
    face_obj.kps = np.array(
        [[180, 180], [320, 180], [250, 260], [190, 320], [310, 320]],
        dtype=np.float32,
    )
    face_obj.embedding = np.random.randn(512).astype(np.float32)
    face_app.get.return_value = [face_obj]
    mock.app.FaceAnalysis.return_value = face_app
    sys.modules["insightface"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_onnxruntime() -> None:
    """Mock onnxruntime 模块（session 级别）。"""
    mock = MagicMock()
    mock.get_available_providers.return_value = ["CPUExecutionProvider"]
    sys.modules["onnxruntime"] = mock


# ============================================================================
# 合成数据 fixtures
# ============================================================================
@pytest.fixture
def sample_avatar_image() -> Any:
    """创建模拟人物上半身的测试图片（头部轮廓 + 五官 + 身体轮廓）。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (200, 200, 210))
    draw = ImageDraw.Draw(img)

    # 身体
    draw.rectangle([120, 300, 392, 512], fill=(60, 60, 80))

    # 头部
    draw.ellipse([156, 60, 356, 300], fill=(255, 220, 180))

    # 头发
    draw.arc([150, 55, 362, 220], 180, 0, fill=(40, 30, 20), width=20)

    # 眼睛
    draw.ellipse([200, 150, 230, 175], fill=(50, 50, 50))
    draw.ellipse([280, 150, 310, 175], fill=(50, 50, 50))

    # 鼻子
    draw.ellipse([245, 185, 265, 200], fill=(200, 150, 130))

    # 嘴巴
    draw.arc([230, 210, 280, 245], 0, 180, fill=(180, 80, 80), width=3)

    return img


@pytest.fixture
def sample_short_audio() -> AudioData:
    """创建一段 2 秒的测试音频（正弦波，22050Hz）。"""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return AudioData(waveform=waveform, sample_rate=sr)


@pytest.fixture
def sample_driving_video() -> Any:
    """创建一个 2 秒的测试驱动视频（简单的动画帧）。"""
    from PIL import Image, ImageDraw

    frames = []
    for i in range(50):  # 2s @ 25fps
        img = Image.new("RGB", (256, 256), (100, 100, 100))
        draw = ImageDraw.Draw(img)
        x = 128 + int(20 * np.sin(2 * np.pi * i / 25))
        draw.ellipse([x - 40, 88, x + 40, 168], fill=(255, 200, 150))
        frames.append(img)
    return frames


@pytest.fixture
def sample_expression_params() -> list[dict[str, Any]]:
    """返回 10 帧的表情参数序列。"""
    params = []
    for i in range(10):
        t = i / 10.0
        params.append(
            {
                "rotation": [0.0, 0.0, np.sin(2 * np.pi * t) * 0.1],
                "scale": 1.0 + 0.02 * np.sin(2 * np.pi * t),
                "translation": [0.0, np.sin(2 * np.pi * t) * 0.02],
                "smile": 0.5 + 0.5 * np.sin(2 * np.pi * t),
                "mouth_open": 0.3 + 0.2 * np.cos(2 * np.pi * t),
            }
        )
    return params


@pytest.fixture
def sample_motion_data() -> MotionData:
    """返回 MotionData 对象（预设的挥手动作，30 帧）。"""
    t = np.linspace(0, 1.0, 30, dtype=np.float32)
    rest = np.array(
        [
            [0.50, 0.18], [0.48, 0.15], [0.52, 0.15], [0.45, 0.18], [0.55, 0.18],
            [0.42, 0.30], [0.58, 0.30], [0.38, 0.45], [0.62, 0.45], [0.36, 0.60],
            [0.64, 0.60], [0.45, 0.55], [0.55, 0.55], [0.44, 0.75], [0.56, 0.75],
            [0.44, 0.95], [0.56, 0.95],
        ],
        dtype=np.float32,
    )
    kps = np.broadcast_to(rest, (30, 17, 2)).copy()
    lift = 0.12 * (1.0 - np.cos(2 * np.pi * t / 1.2))
    swing = 0.10 * np.sin(2 * np.pi * t / 0.8)
    kps[:, 8, 1] -= lift
    kps[:, 10, 1] -= lift + 0.10
    kps[:, 10, 0] += 0.08 + swing
    kps[:, 8, 0] += 0.04
    return MotionData(
        keypoints=kps, frame_count=30, fps=30, skeleton_type="coco"
    )


@pytest.fixture
def sample_text_list() -> list[str]:
    """返回一组测试文本列表（5 句短句）。"""
    return [
        "你好，我是数字人助手。",
        "今天天气真不错。",
        "你想了解什么信息呢？",
        "这个问题很有意思。",
        "感谢你的提问，再见！",
    ]


@pytest.fixture
def fresh_bus() -> Any:
    """返回一个独立的事件总线实例。"""
    from mosaic.core.events import EventBus

    return EventBus()


@pytest.fixture
def cpu_scheduler() -> Any:
    """返回一个 CPU 调度器实例。"""
    from mosaic.core.scheduler import Scheduler

    return Scheduler(device="cpu")