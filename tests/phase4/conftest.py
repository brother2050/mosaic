# tests/phase4/conftest.py
"""Phase 4 测试公共 fixtures。

提供视频域测试所需的合成帧、VideoData、临时文件路径、mock FFmpeg 等
共用 fixture。全部使用合成 PIL 图像，不依赖外部文件或真实模型。
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.scheduler import Scheduler, get_scheduler, set_scheduler
from mosaic.core.types import VideoData, MosaicData


# ---------------------------------------------------------------------------
# Mock torch 注入（session 作用域，适配 Phase 1/2/3 已注入的情况）
# ---------------------------------------------------------------------------
def _make_mock_tensor(numpy_array):
    """创建 mock torch.Tensor，模拟真实 tensor 的 cpu/numpy 行为。"""
    mock_t = MagicMock()
    mock_t.cpu.return_value = mock_t
    mock_t.numpy.return_value = numpy_array
    # 对 numpy 数组的 mock 属性
    mock_t.shape = numpy_array.shape
    mock_t.ndim = numpy_array.ndim
    return mock_t


@pytest.fixture(scope="session", autouse=True)
def _mock_torch_phase4():
    """注入/补齐 mock torch 模块，确保无 GPU 环境也可运行 Phase 4 测试。"""
    if "torch" not in sys.modules:
        mt = types.ModuleType("torch")
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
        mt.ones_like = MagicMock(return_value=MagicMock())
        mt.ones = MagicMock(return_value=MagicMock())
        mt.tensor = MagicMock(return_value=MagicMock())
        mt.from_numpy = MagicMock(
            side_effect=lambda x: _make_mock_tensor(x)
        )
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        _mcuda.empty_cache = MagicMock()
        mt.cuda = _mcuda
        sys.modules["torch"] = mt
        sys.modules["torch.cuda"] = _mcuda
    else:
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
        if not hasattr(mt, "from_numpy"):
            mt.from_numpy = MagicMock(
                side_effect=lambda x: _make_mock_tensor(x)
            )
        if not hasattr(mt, "no_grad"):
            _ctx = MagicMock()
            _ctx.__enter__ = MagicMock(return_value=None)
            _ctx.__exit__ = MagicMock(return_value=None)
            mt.no_grad = MagicMock(return_value=_ctx)
        cuda = getattr(mt, "cuda", None)
        if cuda is not None and not hasattr(cuda, "empty_cache"):
            cuda.empty_cache = MagicMock()

    yield


# ---------------------------------------------------------------------------
# Mock diffusers 注入（按需，与 Phase 2/3 兼容）
# ---------------------------------------------------------------------------
def _inject_mock_diffusers():
    if "diffusers" not in sys.modules:
        dm = types.ModuleType("diffusers")
        dm.CogVideoXPipeline = MagicMock()
        dm.CogVideoXPipeline.from_pretrained = MagicMock()
        dm.StableVideoDiffusionPipeline = MagicMock()
        dm.StableVideoDiffusionPipeline.from_pretrained = MagicMock()
        sys.modules["diffusers"] = dm
    else:
        dm = sys.modules["diffusers"]
        if not hasattr(dm, "CogVideoXPipeline"):
            dm.CogVideoXPipeline = MagicMock()
            dm.CogVideoXPipeline.from_pretrained = MagicMock()
        if not hasattr(dm, "StableVideoDiffusionPipeline"):
            dm.StableVideoDiffusionPipeline = MagicMock()
            dm.StableVideoDiffusionPipeline.from_pretrained = MagicMock()


def _inject_mock_imageio_ffmpeg():
    """仅在真实 imageio_ffmpeg 不可用时注入 mock。"""
    try:
        import imageio_ffmpeg  # noqa: F401
        # 真实包可用，不注入 mock
        return
    except ImportError:
        pass
    if "imageio_ffmpeg" not in sys.modules:
        im = types.ModuleType("imageio_ffmpeg")
        im.get_ffmpeg_exe = MagicMock(return_value="/usr/bin/ffmpeg")
        sys.modules["imageio_ffmpeg"] = im


_inject_mock_diffusers()
_inject_mock_imageio_ffmpeg()


# ---------------------------------------------------------------------------
# 合成 PIL 帧 fixtures
# ---------------------------------------------------------------------------
def _make_gradient_frame(width: int, height: int, offset: int) -> Image.Image:
    """生成渐变颜色块 PIL 图像（64x64 RGB）。

    Parameters
    ----------
    width, height:
        图像尺寸。
    offset:
        颜色偏移量，用于区分不同帧。
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = (x + offset * 10) % 256
            g = (y + offset * 7) % 256
            b = ((x + y) // 2 + offset * 13) % 256
            arr[y, x] = [r, g, b]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_frames() -> list[Image.Image]:
    """创建 10 帧 PIL.Image（渐变颜色块，64x64 RGB）。"""
    return [_make_gradient_frame(64, 64, i) for i in range(10)]


@pytest.fixture
def sample_video(sample_frames) -> VideoData:
    """创建 VideoData，包含 sample_frames，fps=30。"""
    return VideoData(frames=list(sample_frames), fps=30)


@pytest.fixture
def sample_long_video() -> VideoData:
    """创建 VideoData，包含 30 帧，fps=30。"""
    frames = [_make_gradient_frame(64, 64, i) for i in range(30)]
    return VideoData(frames=frames, fps=30)


@pytest.fixture
def tmp_video_path() -> str:
    """临时视频文件路径（用于保存测试）。"""
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="mosaic_test_video_")
    os.close(fd)
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Mock FFmpeg fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ffmpeg():
    """Mock subprocess.Popen 用于 FFmpeg 操作（返回成功，写入输出文件）。"""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.pid = 12345

    def _side_effect(args, **kwargs):
        # 提取输出文件路径并写入占位内容
        for i, arg in enumerate(args):
            if arg == "-y" and i + 1 < len(args):
                out_path = args[i + 1]
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(b"mock-video-data")
                except (OSError, IndexError):
                    pass
                break
        mock_proc.configure_mock(**{
            "args": args,
        })
        return mock_proc

    with patch("subprocess.Popen", side_effect=_side_effect) as mock_popen:
        mock_popen.return_value = mock_proc
        yield mock_popen


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run 用于 FFmpeg 合并操作。"""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"mock-ffmpeg-output"
    mock_result.stderr = b""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        yield mock_run


# ---------------------------------------------------------------------------
# 调度器 fixtures
# ---------------------------------------------------------------------------
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