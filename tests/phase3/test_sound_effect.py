# tests/phase3/test_sound_effect.py
"""Phase 3 音效生成节点测试。

测试 SoundEffectGenerator 节点的基本功能：输出 AudioData、
不同描述生成不同音效、negative_prompt 参数、describe 信息。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# 辅助：为 SoundEffectGenerator 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_audioldm():
    """Mock auto_load_pipeline（SoundEffectGenerator 通过它加载 AudioLDM2）。"""
    mock_pipe = MagicMock()
    mock_output = MagicMock()
    mock_output.audios = [
        np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
    ]
    mock_pipe.return_value = mock_output
    mock_pipe.to.return_value = mock_pipe

    with patch(
        "mosaic.nodes._model_loader.auto_load_pipeline",
        return_value=mock_pipe,
    ):
        yield mock_pipe


class TestSoundEffectBasic:
    """T_SFX_01：基本音效生成测试。"""

    def test_basic_sound_effect(self, mock_audioldm, cpu_scheduler):
        """T_SFX_01：基本音效生成，输出 AudioData。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        result = sfx(MosaicData(prompt="下雨的声音，室内，轻柔的雨滴"))

        audio = result.get("audio")
        assert audio is not None, "SoundEffectGenerator 输出应包含 audio"
        assert audio.waveform is not None, "waveform 不应为 None"
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert len(audio.waveform) > 0, "waveform 不应为空"


class TestSoundEffectPrompts:
    """T_SFX_02：不同描述测试。"""

    def test_different_descriptions(self, mock_audioldm, cpu_scheduler):
        """T_SFX_02：不同描述生成不同音效。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        result1 = sfx(MosaicData(prompt="下雨的声音"))
        result2 = sfx(MosaicData(prompt="汽车喇叭声"))

        assert result1.get("audio") is not None
        assert result2.get("audio") is not None
        # 两个 prompt 都应成功生成
        assert result1.get("prompt") == "下雨的声音"
        assert result2.get("prompt") == "汽车喇叭声"

    def test_duration_param(self, mock_audioldm, cpu_scheduler):
        """指定 duration 参数。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        result = sfx(MosaicData(prompt="门铃声", duration=3.0))

        audio = result.get("audio")
        assert audio is not None, "指定 duration 时应生成音频"


class TestSoundEffectNegativePrompt:
    """T_SFX_03：negative_prompt 参数测试。"""

    def test_negative_prompt(self, mock_audioldm, cpu_scheduler):
        """T_SFX_03：negative_prompt 参数生效。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        result = sfx(MosaicData(
            prompt="清脆的鸟叫声",
            negative_prompt="嘈杂，噪音，刺耳",
        ))

        audio = result.get("audio")
        assert audio is not None, "带 negative_prompt 时应生成音频"

    def test_num_inference_steps(self, mock_audioldm, cpu_scheduler):
        """num_inference_steps 参数生效。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        result = sfx(MosaicData(
            prompt="海浪声",
            num_inference_steps=20,
        ))

        assert result.get("audio") is not None


class TestSoundEffectDescribe:
    """T_SFX_04：describe 测试。"""

    def test_describe(self, mock_audioldm, cpu_scheduler):
        """T_SFX_04：describe 标注模型信息。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        spec = sfx.describe()

        assert spec.name == "sound-effect-generator", "节点名称应为 'sound-effect-generator'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "text" in spec.input_types, "输入类型应包含 'text'"
        assert "audio" in spec.output_types, "输出类型应包含 'audio'"


class TestSoundEffectErrors:
    """SoundEffectGenerator 错误处理测试。"""

    def test_missing_prompt(self, mock_audioldm, cpu_scheduler):
        """缺少 prompt 应抛出 ValueError。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'prompt'"):
            sfx(MosaicData())

    def test_empty_prompt(self, mock_audioldm, cpu_scheduler):
        """空 prompt 应抛出 ValueError。"""
        from mosaic.nodes.audio.sound_effect import SoundEffectGenerator

        sfx = SoundEffectGenerator(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'prompt'"):
            sfx(MosaicData(prompt=""))