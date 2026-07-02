# tests/phase3/test_integration.py
"""Phase 3 端到端集成测试。

测试音频域和字幕域节点之间的端到端工作流：
TTS→ASR、ASR→字幕生成、字幕生成→字幕翻译、TTS→字幕生成→字幕对齐、
显存释放、事件触发。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.events import EventBus, EventType
from mosaic.core.types import AudioData, MosaicData, SubtitleData


# ---------------------------------------------------------------------------
# 集成测试 mark
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def integration_mocks():
    """为集成测试提供完整的 mock 环境。"""
    import transformers as _tf

    # Mock TTS
    if "TTS" not in sys.modules:
        import types
        tts_mod = types.ModuleType("TTS")
        tts_api = types.ModuleType("TTS.api")
        mock_tts_cls = MagicMock()
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = np.sin(
            np.linspace(0, 2 * np.pi, 24000)
        ).astype(np.float32).tolist()
        mock_tts_cls.return_value = mock_tts_instance
        tts_api.TTS = mock_tts_cls
        tts_mod.api = tts_api
        sys.modules["TTS"] = tts_mod
        sys.modules["TTS.api"] = tts_api

    # Mock ASR pipeline
    mock_asr_pipe = MagicMock()
    mock_asr_pipe.return_value = {
        "text": "你好，这是一个测试音频。",
        "language": "zh",
        "chunks": [
            {"timestamp": (0.0, 2.5), "text": "你好，"},
            {"timestamp": (2.5, 5.0), "text": "这是一个测试音频。"},
        ],
    }

    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.feature_extractor = MagicMock()

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    # Mock Translator
    mock_trans = MagicMock()
    mock_trans.run.return_value = MosaicData(
        translated_text="[1] Hello, this is a test audio.",
        source_language="zh",
        target_language="en",
        translation_mode="generic",
    )
    mock_trans.name = "translator"

    with patch.object(_tf, "pipeline", return_value=mock_asr_pipe), \
         patch.object(_tf, "AutoModelForSpeechSeq2Seq") as mock_model_cls, \
         patch.object(_tf, "AutoProcessor") as mock_proc_cls, \
         patch("mosaic.nodes.text.translator.Translator", return_value=mock_trans):
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = mock_processor
        yield


# ---------------------------------------------------------------------------
# T_E2E_P3_01：TTS → ASR（文本→语音→文本，验证一致性）
# ---------------------------------------------------------------------------
class TestE2ETTSASR:
    """TTS → ASR 端到端测试。"""

    def test_tts_to_asr_roundtrip(self, integration_mocks, cpu_scheduler):
        """T_E2E_P3_01：TTS→ASR 语音生成后识别。"""
        from mosaic.nodes.audio.tts import TTS
        from mosaic.nodes.audio.asr import ASR

        input_text = "你好，这是一个测试音频。"

        # Step 1: TTS 文本转语音
        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts_result = tts(MosaicData(text=input_text))
        audio = tts_result.get("audio")
        assert audio is not None, "TTS 应生成音频"
        assert audio.waveform is not None, "TTS 波形不应为空"

        # Step 2: ASR 语音转文本
        asr = ASR(scheduler=cpu_scheduler)
        asr_result = asr(MosaicData(audio=audio))
        recognized_text = asr_result.get("text")
        assert recognized_text is not None, "ASR 应识别出文本"
        assert isinstance(recognized_text, str), "识别文本应为字符串"
        assert len(recognized_text) > 0, "识别文本不应为空"


# ---------------------------------------------------------------------------
# T_E2E_P3_02：ASR → 字幕生成（音频转字幕）
# ---------------------------------------------------------------------------
class TestE2EASRSubtitle:
    """ASR → 字幕生成端到端测试。"""

    def test_asr_to_subtitle(self, integration_mocks, sample_audio, cpu_scheduler):
        """T_E2E_P3_02：ASR → 字幕生成，音频转字幕。"""
        from mosaic.nodes.audio.asr import ASR
        from mosaic.nodes.subtitle.generator import SubtitleGenerator

        # Step 1: ASR 识别
        asr = ASR(scheduler=cpu_scheduler)
        asr_result = asr(MosaicData(audio=sample_audio))
        assert asr_result.get("text") is not None, "ASR 应识别出文本"
        assert asr_result.get("segments") is not None, "ASR 应输出 segments"

        # Step 2: 字幕生成
        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        gen_result = gen(MosaicData(audio=sample_audio))
        subtitle = gen_result.get("subtitle")
        assert subtitle is not None, "应生成字幕"
        assert isinstance(subtitle, SubtitleData), "应为 SubtitleData"
        assert len(subtitle.segments) > 0, "segments 不应为空"


# ---------------------------------------------------------------------------
# T_E2E_P3_03：字幕生成 → 字幕翻译（中文字幕→英文字幕）
# ---------------------------------------------------------------------------
class TestE2ESubtitleTranslate:
    """字幕生成 → 字幕翻译端到端测试。"""

    def test_subtitle_to_translation(self, integration_mocks, sample_subtitle, cpu_scheduler):
        """T_E2E_P3_03：字幕生成→字幕翻译，中文字幕→英文字幕。"""
        from mosaic.nodes.subtitle.translator import SubtitleTranslator

        # 验证输入字幕
        assert len(sample_subtitle.segments) == 5, "输入应有 5 个片段"
        assert "你好" in sample_subtitle.segments[0]["text"], "输入应为中文"

        # 翻译
        trans = SubtitleTranslator(
            source_language="zh",
            target_language="en",
            scheduler=cpu_scheduler,
        )
        result = trans(MosaicData(subtitle=sample_subtitle))
        translated = result.get("subtitle")
        assert translated is not None, "翻译应输出字幕"
        assert isinstance(translated, SubtitleData), "输出应为 SubtitleData"


# ---------------------------------------------------------------------------
# T_E2E_P3_04：TTS → 字幕生成 → 字幕对齐（完整语音-字幕流程）
# ---------------------------------------------------------------------------
class TestE2EFullPipeline:
    """TTS → 字幕生成 → 字幕对齐完整流程。"""

    def test_full_pipeline(self, integration_mocks, cpu_scheduler):
        """T_E2E_P3_04：TTS→字幕生成→字幕对齐完整流程。"""
        from mosaic.nodes.audio.tts import TTS
        from mosaic.nodes.subtitle.generator import SubtitleGenerator
        from mosaic.nodes.subtitle.aligner import SubtitleAligner

        input_text = "你好，这是一个完整的端到端测试流程。"

        # Step 1: TTS 生成语音
        tts = TTS(language="zh", scheduler=cpu_scheduler)
        tts_result = tts(MosaicData(text=input_text))
        audio = tts_result.get("audio")
        assert audio is not None, "TTS 应生成音频"

        # Step 2: 字幕生成
        gen = SubtitleGenerator(scheduler=cpu_scheduler)
        gen_result = gen(MosaicData(audio=audio))
        subtitle = gen_result.get("subtitle")
        assert subtitle is not None, "应生成字幕"

        # Step 3: 字幕对齐（DTW 方法，不需要额外模型）
        aligner = SubtitleAligner(method="dtw", scheduler=cpu_scheduler)
        align_result = aligner(MosaicData(
            subtitle=subtitle,
            audio=audio,
        ))
        aligned_subtitle = align_result.get("subtitle")
        assert aligned_subtitle is not None, "字幕对齐应成功"
        assert len(aligned_subtitle.segments) > 0, "对齐后 segments 不应为空"


# ---------------------------------------------------------------------------
# T_E2E_P3_05：音频节点 load/unload 后显存正确释放
# ---------------------------------------------------------------------------
class TestE2EMemoryRelease:
    """显存释放测试。"""

    def test_load_unload_memory_release(self, integration_mocks, cpu_scheduler):
        """T_E2E_P3_05：音频节点 load/unload 后显存正确释放。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", scheduler=cpu_scheduler)

        # 加载
        tts.load()
        assert tts.is_loaded(), "load 后 is_loaded 应为 True"

        tts.unload()
        assert not tts.is_loaded(), "unload 后 is_loaded 应为 False"

    def test_multiple_nodes_memory(self, integration_mocks, cpu_scheduler):
        """多个节点加载/卸载。"""
        from mosaic.nodes.audio.tts import TTS
        from mosaic.nodes.audio.asr import ASR

        tts = TTS(language="zh", scheduler=cpu_scheduler)
        asr = ASR(scheduler=cpu_scheduler)

        tts.load()
        asr.load()
        assert tts.is_loaded()
        assert asr.is_loaded()

        tts.unload()
        asr.unload()
        assert not tts.is_loaded()
        assert not asr.is_loaded()


# ---------------------------------------------------------------------------
# T_E2E_P3_06：运行过程中事件被正确触发
# ---------------------------------------------------------------------------
class TestE2EEvents:
    """事件触发测试。"""

    def test_events_triggered(self, integration_mocks, cpu_scheduler, fresh_bus):
        """T_E2E_P3_06：运行过程中事件被正确触发。"""
        from mosaic.nodes.audio.tts import TTS

        # 收集事件
        events = []
        fresh_bus.on(EventType.NODE_START, lambda e: events.append(("start", e)))
        fresh_bus.on(EventType.NODE_COMPLETE, lambda e: events.append(("complete", e)))
        fresh_bus.on(EventType.NODE_ERROR, lambda e: events.append(("error", e)))

        tts = TTS(language="zh", scheduler=cpu_scheduler, bus=fresh_bus)
        result = tts(MosaicData(text="测试事件"))
        assert result is not None, "TTS 应正常完成"

        # 验证 start 和 complete 事件被触发
        start_events = [e for e_type, e in events if e_type == "start"]
        complete_events = [e for e_type, e in events if e_type == "complete"]
        assert len(start_events) > 0, "NODE_START 事件应被触发"
        assert len(complete_events) > 0, "NODE_COMPLETE 事件应被触发"

    def test_error_event(self, integration_mocks, cpu_scheduler, fresh_bus):
        """错误事件触发测试。"""
        from mosaic.nodes.audio.tts import TTS

        events = []
        fresh_bus.on(EventType.NODE_ERROR, lambda e: events.append(e))

        tts = TTS(language="zh", scheduler=cpu_scheduler, bus=fresh_bus)
        with pytest.raises(ValueError):
            tts(MosaicData())  # 缺少 text

        # 验证错误事件被触发
        assert len(events) > 0, "NODE_ERROR 事件应被触发"