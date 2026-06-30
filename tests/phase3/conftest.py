# tests/phase3/conftest.py
"""Phase 3 测试公共 fixtures。

提供音频/字幕域测试所需的合成音频、字幕字符串、mock 节点等共用 fixture。
全部使用合成数据，不依赖外部文件或真实模型。
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core import (
    EventBus,
    MosaicData,
    Scheduler,
    set_scheduler,
)
from mosaic.core.types import AudioData, SubtitleData


# ---------------------------------------------------------------------------
# Mock torch 注入（session 作用域，适配 Phase 1/2 已注入的情况）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_torch_phase3():
    """注入/补齐 mock torch 模块，确保无 GPU 环境也可运行 Phase 3 测试。"""
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
        mt.ones_like = MagicMock(return_value=MagicMock())
        mt.ones = MagicMock(return_value=MagicMock())
        mt.tensor = MagicMock(return_value=MagicMock())
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.__spec__ = MagicMock()
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
        if not hasattr(mt, "no_grad"):
            _ctx = MagicMock()
            _ctx.__enter__ = MagicMock(return_value=None)
            _ctx.__exit__ = MagicMock(return_value=None)
            mt.no_grad = MagicMock(return_value=_ctx)
        cuda = getattr(mt, "cuda", None)
        if cuda is not None and not hasattr(cuda, "empty_cache"):
            cuda.empty_cache = MagicMock()

    # 确保 mock transformers 有 Phase 3 所需的全部属性
    # （Phase 2 conftest 可能注入了不完整的 mock transformers）
    if "transformers" in sys.modules:
        tm = sys.modules["transformers"]
        if not hasattr(tm, "AutoProcessor"):
            tm.AutoProcessor = MagicMock()
            tm.AutoProcessor.from_pretrained = MagicMock()
        if not hasattr(tm, "AutoModelForSpeechSeq2Seq"):
            tm.AutoModelForSpeechSeq2Seq = MagicMock()
            tm.AutoModelForSpeechSeq2Seq.from_pretrained = MagicMock()
        if not hasattr(tm, "MusicgenForConditionalGeneration"):
            tm.MusicgenForConditionalGeneration = MagicMock()
            tm.MusicgenForConditionalGeneration.from_pretrained = MagicMock()
        if not hasattr(tm, "pipeline"):
            tm.pipeline = MagicMock()

    # 确保 mock diffusers 有 Phase 3 所需属性
    if "diffusers" in sys.modules:
        dm = sys.modules["diffusers"]
        if not hasattr(dm, "AudioLDMPipeline"):
            dm.AudioLDMPipeline = MagicMock()
            dm.AudioLDMPipeline.from_pretrained = MagicMock()

    yield


# ---------------------------------------------------------------------------
# Mock diffusers 注入（按需，与 Phase 2 兼容）
# ---------------------------------------------------------------------------
def _inject_mock_diffusers():
    if "diffusers" not in sys.modules:
        dm = types.ModuleType("diffusers")
        dm.__spec__ = MagicMock()
        dm.AudioLDMPipeline = MagicMock()
        dm.AudioLDMPipeline.from_pretrained = MagicMock()
        dm.StableDiffusionXLPipeline = MagicMock()
        sys.modules["diffusers"] = dm
    else:
        # Phase 2 可能已注入 mock diffusers，补齐 Phase 3 所需属性
        dm = sys.modules["diffusers"]
        if not hasattr(dm, "AudioLDMPipeline"):
            dm.AudioLDMPipeline = MagicMock()
            dm.AudioLDMPipeline.from_pretrained = MagicMock()


def _inject_mock_transformers():
    if "transformers" not in sys.modules:
        tm = types.ModuleType("transformers")
        tm.__spec__ = MagicMock()
        tm.AutoProcessor = MagicMock()
        tm.AutoProcessor.from_pretrained = MagicMock()
        tm.AutoModelForSpeechSeq2Seq = MagicMock()
        tm.AutoModelForSpeechSeq2Seq.from_pretrained = MagicMock()
        tm.MusicgenForConditionalGeneration = MagicMock()
        tm.MusicgenForConditionalGeneration.from_pretrained = MagicMock()
        tm.pipeline = MagicMock()
        sys.modules["transformers"] = tm
    else:
        # Phase 1/2 可能已注入 mock transformers，补齐 Phase 3 所需属性
        tm = sys.modules["transformers"]
        if not hasattr(tm, "AutoProcessor"):
            tm.AutoProcessor = MagicMock()
            tm.AutoProcessor.from_pretrained = MagicMock()
        if not hasattr(tm, "AutoModelForSpeechSeq2Seq"):
            tm.AutoModelForSpeechSeq2Seq = MagicMock()
            tm.AutoModelForSpeechSeq2Seq.from_pretrained = MagicMock()
        if not hasattr(tm, "MusicgenForConditionalGeneration"):
            tm.MusicgenForConditionalGeneration = MagicMock()
            tm.MusicgenForConditionalGeneration.from_pretrained = MagicMock()
        if not hasattr(tm, "pipeline"):
            tm.pipeline = MagicMock()


def _inject_mock_soundfile():
    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.__spec__ = MagicMock()
        sf.read = MagicMock(return_value=(np.zeros(100, dtype=np.float32), 22050))
        sf.write = MagicMock()
        sys.modules["soundfile"] = sf


def _inject_mock_edge_tts():
    """注入 mock edge_tts 模块，使 TTS/VoiceClone 测试无需真实网络。

    edge_tts.Communicate 提供异步 ``stream()`` 生成器，产出有效 WAV 字节，
    配合真实 soundfile 即可得到 numpy 波形。
    """
    import io as _io
    import wave as _wave

    def _make_wav_bytes(duration_s: float = 0.5, sr: int = 22050) -> bytes:
        """生成一段有效的 WAV 字节数据。"""
        n_samples = int(sr * duration_s)
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x80" * n_samples)  # 静音 PCM16
        return buf.getvalue()

    if "edge_tts" not in sys.modules:
        etc = types.ModuleType("edge_tts")
        etc.__spec__ = MagicMock()

        class _Communicate:
            """模拟 edge_tts.Communicate，支持 rate/pitch 等关键字参数。"""

            def __init__(self, text, voice, rate="+0%", pitch="+0Hz",
                         volume="+0%", proxy=None, **kwargs):
                self.text = text
                self.voice = voice
                self.rate = rate
                self.pitch = pitch

            async def stream(self):
                # 产出有效 WAV 字节，配合真实 soundfile 解码
                yield {"type": "audio", "data": _make_wav_bytes()}
                yield {"type": "WordBoundary", "offset": 0, "duration": 0.1,
                       "text": self.text}

        etc.Communicate = _Communicate
        etc.SubMaker = MagicMock()
        sys.modules["edge_tts"] = etc
    else:
        etc = sys.modules["edge_tts"]
        if not hasattr(etc, "Communicate"):

            class _Communicate:
                def __init__(self, text, voice, rate="+0%", **kwargs):
                    self.text = text
                    self.voice = voice
                    self.rate = rate

                async def stream(self):
                    yield {"type": "audio", "data": _make_wav_bytes()}

            etc.Communicate = _Communicate
        if not hasattr(etc, "SubMaker"):
            etc.SubMaker = MagicMock()


_inject_mock_diffusers()
_inject_mock_transformers()
_inject_mock_soundfile()
_inject_mock_edge_tts()


# ---------------------------------------------------------------------------
# 合成音频 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_audio() -> AudioData:
    """创建一段 5 秒的测试音频（正弦波，440Hz，22050Hz 采样率）。"""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        metadata={"duration": duration, "format": "wav"},
    )


@pytest.fixture
def sample_long_audio() -> AudioData:
    """创建一段 35 秒的测试音频（多频率正弦波，22050Hz 采样率）。"""
    sr = 22050
    duration = 35.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (
        np.sin(2 * np.pi * 440.0 * t) * 0.3
        + np.sin(2 * np.pi * 880.0 * t) * 0.2
        + np.sin(2 * np.pi * 220.0 * t) * 0.1
    ).astype(np.float32)
    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        metadata={"duration": duration, "format": "wav"},
    )


@pytest.fixture
def stereo_audio() -> AudioData:
    """创建一段 3 秒的立体声测试音频（22050Hz）。"""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    right = (np.sin(2 * np.pi * 550.0 * t) * 0.5).astype(np.float32)
    waveform = np.stack([left, right])  # shape: (2, samples)
    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        metadata={"duration": duration, "channels": 2},
    )


# ---------------------------------------------------------------------------
# 字幕数据 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_srt_content() -> str:
    """返回一段标准 SRT 格式字符串。"""
    return """1
00:00:01,000 --> 00:00:04,000
你好，欢迎来到 Mosaic 框架。

2
00:00:04,000 --> 00:00:08,000
这是一个多模态 AI 生成系统。

3
00:00:08,000 --> 00:00:12,500
支持文本、图像、音频、字幕等多种模态。

4
00:00:12,500 --> 00:00:17,000
让我们开始探索吧。

5
00:00:17,000 --> 00:00:21,000
感谢你的使用。
"""


@pytest.fixture
def sample_vtt_content() -> str:
    """返回一段标准 WebVTT 格式字符串。"""
    return """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, welcome to Mosaic.

00:00:04.000 --> 00:00:08.000
This is a multimodal AI system.

00:00:08.000 --> 00:00:12.500
It supports text, image, audio, and subtitle.

00:00:12.500 --> 00:00:17.000
Let's explore together.

00:00:17.000 --> 00:00:21.000
Thank you for using it.
"""


@pytest.fixture
def sample_subtitle() -> SubtitleData:
    """返回 SubtitleData 对象（5 个片段）。"""
    segments = [
        {"start": 1.0, "end": 4.0, "text": "你好，欢迎来到 Mosaic 框架。"},
        {"start": 4.0, "end": 8.0, "text": "这是一个多模态 AI 生成系统。"},
        {"start": 8.0, "end": 12.5, "text": "支持文本、图像、音频、字幕等多种模态。"},
        {"start": 12.5, "end": 17.0, "text": "让我们开始探索吧。"},
        {"start": 17.0, "end": 21.0, "text": "感谢你的使用。"},
    ]
    return SubtitleData(segments=segments, format="srt")


@pytest.fixture
def sample_english_subtitle() -> SubtitleData:
    """返回英文 SubtitleData 对象（5 个片段）。"""
    segments = [
        {"start": 1.0, "end": 4.0, "text": "Hello, welcome to Mosaic framework."},
        {"start": 4.0, "end": 8.0, "text": "It is a multimodal AI generation system."},
        {"start": 8.0, "end": 12.5, "text": "It supports text, image, audio, and subtitle."},
        {"start": 12.5, "end": 17.0, "text": "Let's start exploring."},
        {"start": 17.0, "end": 21.0, "text": "Thank you for using it."},
    ]
    return SubtitleData(segments=segments, format="srt")


@pytest.fixture
def empty_subtitle() -> SubtitleData:
    """返回空的 SubtitleData 对象。"""
    return SubtitleData(segments=[], format="srt")


@pytest.fixture
def sample_video_path() -> str | None:
    """提供一个测试视频文件路径（如果不可用，用 None）。"""
    return None  # 不依赖外部视频文件


# ---------------------------------------------------------------------------
# Mock 音频节点 fixture
# ---------------------------------------------------------------------------
from mosaic.nodes.audio._base import BaseAudioNode


class _MockAudioNode(BaseAudioNode):
    """不需要真实模型的音频节点 mock，用于测试基类功能。"""

    name = "mock-audio-node"
    description = "Mock audio node for testing."
    version = "0.1.0"
    input_types = ["text", "audio", "mosaic"]
    output_types = ["audio"]

    def __init__(self, model="mock-model", **kwargs):
        super().__init__(model=model, **kwargs)

    def _load_model(self) -> None:
        self._model = MagicMock()

    def run(self, input_data: MosaicData) -> MosaicData:
        self._emit_start()
        waveform = np.zeros(22050, dtype=np.float32)
        audio = self._ensure_audio_data(waveform, 22050)
        return MosaicData(audio=audio)


@pytest.fixture
def mock_audio_node():
    """返回一个 MockAudioNode 实例。"""
    return _MockAudioNode()


# ---------------------------------------------------------------------------
# 调度器与事件总线 fixtures
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


# ---------------------------------------------------------------------------
# 注册表清理 fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def clear_registry():
    """返回干净的注册表实例。"""
    from mosaic.core.registry import NodeRegistry

    return NodeRegistry()