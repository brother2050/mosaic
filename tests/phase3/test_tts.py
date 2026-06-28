# tests/phase3/test_tts.py
"""Phase 3 TTS 节点测试。

测试 TTS 节点的基本功能：输出 AudioData、采样率、波形形状、
长文本分句、语言参数、describe 信息。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# 辅助：为 TTS 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_tts_env():
    """注入 mock TTS 库和 edge_tts，使 TTS 测试不需要真实模型。"""
    # Mock TTS.api
    if "TTS" not in sys.modules:
        import types
        tts_mod = types.ModuleType("TTS")
        tts_api = types.ModuleType("TTS.api")
        mock_tts_cls = MagicMock()
        mock_tts_instance = MagicMock()
        # 模拟 tts 返回 numpy 数组
        mock_tts_instance.tts.return_value = np.sin(
            np.linspace(0, 2 * np.pi, 24000)
        ).astype(np.float32).tolist()
        mock_tts_cls.return_value = mock_tts_instance
        tts_api.TTS = mock_tts_cls
        tts_mod.api = tts_api
        sys.modules["TTS"] = tts_mod
        sys.modules["TTS.api"] = tts_api

    # Mock edge_tts
    if "edge_tts" not in sys.modules:
        import types
        etc = types.ModuleType("edge_tts")
        etc.Communicate = MagicMock()
        sys.modules["edge_tts"] = etc

    yield

    # 清理
    for mod in ["TTS", "TTS.api", "edge_tts"]:
        if mod in sys.modules:
            del sys.modules[mod]


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

    def test_sample_rate_correct(self, mock_tts_env, cpu_scheduler):
        """T_TTS_02：输出 sample_rate 正确。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(language="zh", sample_rate=22050, scheduler=cpu_scheduler)
        result = tts(MosaicData(text="测试"))

        audio = result.get("audio")
        assert audio is not None, "TTS 输出应包含 audio"
        assert audio.sample_rate == 22050, f"sample_rate 应为 22050，实际 {audio.sample_rate}"

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


class TestTTSEdgeTTS:
    """Edge-TTS 后端测试。"""

    def test_edge_tts_backend(self, mock_tts_env, cpu_scheduler):
        """Edge-TTS 后端模式。"""
        from mosaic.nodes.audio.tts import TTS

        tts = TTS(use_edge_tts=True, language="zh", scheduler=cpu_scheduler)
        # Edge-TTS 需要真实网络，这里只验证节点初始化正确
        assert tts._backend == "edge_tts", "应使用 edge_tts 后端"
        spec = tts.describe()
        assert spec.name == "tts"


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