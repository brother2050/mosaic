# tests/phase3/test_tts.py
"""Phase 3 TTS 节点测试。

测试 TTS 节点的基本功能：输出 AudioData、采样率、波形形状、
长文本分句、语言参数、情感风格、describe 信息。

新版 TTS 默认使用 edge-tts（不再依赖 Coqui XTTS-v2），因此本测试
仅 mock edge-tts，不再 mock Coqui TTS 库。
"""

from __future__ import annotations

import io
import struct
import sys
import types
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# 辅助：为 TTS 测试提供 mock 环境
# ---------------------------------------------------------------------------
def _make_wav_bytes(num_samples: int = 100, sample_rate: int = 22050) -> bytes:
    """生成最小有效 WAV 字节流（单声道 16-bit PCM），供 soundfile 解码。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    return buf.getvalue()


def _ensure_edge_tts_mock() -> None:
    """确保 edge_tts 模块已被 mock（与 conftest 共用，幂等）。

    新版 TTS 默认走 edge-tts 后端，需要 ``edge_tts.Communicate`` 提供
    异步 ``stream()`` 生成器。conftest 已在会话级别注入本 mock，此处仅做
    幂等补齐，避免被其他测试清理后缺失。
    """
    etc = sys.modules.get("edge_tts")
    if etc is not None and hasattr(etc, "Communicate"):
        return

    etc = types.ModuleType("edge_tts")
    _wav_data = _make_wav_bytes()

    class _Communicate:
        """模拟 edge_tts.Communicate，支持 rate/pitch 等关键字参数。"""

        def __init__(self, text, voice, rate="+0%", pitch="+0Hz",
                     volume="+0%", proxy=None, **kwargs):
            self.text = text
            self.voice = voice
            self.rate = rate
            self.pitch = pitch

        async def stream(self):
            yield {"type": "audio", "data": _wav_data}

    etc.Communicate = _Communicate
    etc.SubMaker = MagicMock()
    sys.modules["edge_tts"] = etc


@pytest.fixture
def mock_tts_env():
    """注入 mock edge_tts，使 TTS 测试不需要真实模型/网络。

    已移除 Coqui TTS 的 mock —— 新版 TTS 默认使用 edge-tts 作为主力后端。
    """
    _ensure_edge_tts_mock()
    yield
    # 不删除 edge_tts：conftest 提供会话级 mock，保留供后续测试使用


class TestTTSBasic:
    """T_TTS_01-03：基本 TTS 功能测试。"""

    def test_basic_tts_output(self, mock_tts_env, cpu_scheduler):
        """T_TTS_01：基本 TTS，输出 AudioData 非空。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="你好，世界！"))

        audio = result.get("audio")
        assert audio is not None, "TTS 输出应包含 audio"
        assert audio.waveform is not None, "waveform 不应为 None"
        assert len(audio.waveform) > 0, "waveform 不应为空"

    def test_default_backend_is_edge_tts(self, mock_tts_env, cpu_scheduler):
        """T_TTS_01b：默认后端应为 edge_tts（不再是回退方案）。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts.load()
        assert tts._backend == "edge_tts", "默认后端应为 edge_tts"

    def test_sample_rate_correct(self, mock_tts_env, cpu_scheduler):
        """T_TTS_02：输出 sample_rate 正确。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", sample_rate=22050, scheduler=cpu_scheduler)
        result = tts(MosaicData(text="测试"))

        audio = result.get("audio")
        assert audio is not None, "TTS 输出应包含 audio"
        assert audio.sample_rate == 22050, (
            f"sample_rate 应为 22050，实际 {audio.sample_rate}"
        )

    def test_waveform_shape_correct(self, mock_tts_env, cpu_scheduler):
        """T_TTS_03：输出 waveform shape 正确。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="测试"))

        audio = result.get("audio")
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert audio.waveform.ndim == 1, "waveform 应为 1 维（单声道）"


class TestTTSFeatures:
    """T_TTS_04-06：TTS 功能特性测试。"""

    def test_long_text_sentence_split(self, mock_tts_env, cpu_scheduler):
        """T_TTS_04：长文本自动分句处理。"""
        from mosaic.nodes.audio.tts import TTS

        long_text = "第一句。第二句。第三句。第四句。第五句。第六句。"
        tts = TTS(language="zh", scheduler=cpu_scheduler)
        result = tts(MosaicData(text=long_text))

        audio = result.get("audio")
        assert audio is not None
        assert audio.waveform is not None
        # 验证原始文本被保留
        assert result.get("text") == long_text

    def test_language_param(self, mock_tts_env, cpu_scheduler):
        """T_TTS_05：指定语言参数生效。"""
        from mosaic.nodes.audio.tts import TTS

        # 输入中指定 language
        tts = TTS(language="zh", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="Hello world", language="en"))

        audio = result.get("audio")
        assert audio is not None

    def test_describe(self, mock_tts_env, cpu_scheduler):
        """T_TTS_06：describe 返回正确信息。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        spec = tts.describe()

        assert spec.name == "tts", "节点名称应为 'tts'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "text" in spec.input_types, "输入类型应包含 'text'"
        assert "audio" in spec.output_types, "输出类型应包含 'audio'"
        # model_info 应反映 edge-tts（vram=0）
        assert spec.model_info.get("vram_gb") == 0.0, "edge-tts 不需要 GPU 显存"


class TestTTSEmotion:
    """T_TTS_07-09：情感风格测试。"""

    def test_emotion_param_constructor(self, mock_tts_env, cpu_scheduler):
        """T_TTS_07：构造函数指定 emotion 生效。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", emotion="cheerful", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="今天天气真好！"))

        audio = result.get("audio")
        assert audio is not None
        assert len(audio.waveform) > 0

    def test_emotion_override_in_run(self, mock_tts_env, cpu_scheduler):
        """T_TTS_08：运行时覆盖 emotion 与 speed。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", emotion="neutral", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="慢点说", emotion="calm", speed=0.8))

        audio = result.get("audio")
        assert audio is not None
        assert len(audio.waveform) > 0

    def test_voice_override(self, mock_tts_env, cpu_scheduler):
        """T_TTS_09：显式 voice 覆盖 emotion 映射。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        result = tts(MosaicData(text="测试", voice="zh-CN-YunxiNeural"))

        audio = result.get("audio")
        assert audio is not None
        assert len(audio.waveform) > 0

    def test_emotion_voice_mapping(self, mock_tts_env, cpu_scheduler):
        """不同情感映射到不同的预设 Neural 语音。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts.load()

        assert tts._resolve_voice("zh", "neutral", None) == "zh-CN-XiaoxiaoNeural"
        assert tts._resolve_voice("zh", "cheerful", None) == "zh-CN-XiaoyiNeural"
        assert tts._resolve_voice("zh", "gentle", None) == "zh-CN-XiaomoNeural"
        assert tts._resolve_voice("zh", "calm", None) == "zh-CN-XiaoruiNeural"
        assert tts._resolve_voice("zh", "male", None) == "zh-CN-YunjianNeural"
        assert tts._resolve_voice("zh", "young_male", None) == "zh-CN-YunxiNeural"
        # voice 显式指定优先于 emotion
        assert (
            tts._resolve_voice("zh", "neutral", "zh-CN-YunxiaNeural")
            == "zh-CN-YunxiaNeural"
        )

    def test_unknown_emotion_falls_back(self, mock_tts_env, cpu_scheduler):
        """未知情感回退到 neutral。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts.load()
        voice = tts._resolve_voice("zh", "unknown_emotion", None)
        assert voice == "zh-CN-XiaoxiaoNeural"  # neutral 回退

    def test_english_emotion_mapping(self, mock_tts_env, cpu_scheduler):
        """英文情感映射。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="en", scheduler=cpu_scheduler)
        tts.load()
        assert tts._resolve_voice("en", "neutral", None) == "en-US-JennyNeural"
        assert tts._resolve_voice("en", "cheerful", None) == "en-US-AriaNeural"
        assert tts._resolve_voice("en", "male", None) == "en-US-GuyNeural"


class TestTTSEdgeTTS:
    """Edge-TTS 后端测试。"""

    def test_edge_tts_backend(self, mock_tts_env, cpu_scheduler):
        """Edge-TTS 后端模式（默认）。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts.load()
        assert tts._backend == "edge_tts", "应使用 edge_tts 后端"
        spec = tts.describe()
        assert spec.name == "tts"

    def test_explicit_edge_model(self, mock_tts_env, cpu_scheduler):
        """显式指定 model='edge-tts'。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(model="edge-tts", language="zh", scheduler=cpu_scheduler)
        tts.load()
        assert tts._backend == "edge_tts"

    def test_edge_prefix_model(self, mock_tts_env, cpu_scheduler):
        """model 以 'edge' 开头也走 edge-tts 后端。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(model="edge-tts-custom", language="zh", scheduler=cpu_scheduler)
        tts.load()
        assert tts._backend == "edge_tts"


class TestTTSErrorHandling:
    """TTS 错误处理测试。"""

    def test_missing_text(self, mock_tts_env, cpu_scheduler):
        """缺少 text 应抛出 ValueError。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'text'"):
            tts(MosaicData())

    def test_empty_text(self, mock_tts_env, cpu_scheduler):
        """空文本应抛出 ValueError。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'text'"):
            tts(MosaicData(text=""))

    def test_non_string_text(self, mock_tts_env, cpu_scheduler):
        """非字符串 text 应抛出 ValueError。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'text'"):
            tts(MosaicData(text=12345))
