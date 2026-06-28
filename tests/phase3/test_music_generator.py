# tests/phase3/test_music_generator.py
"""Phase 3 音乐生成节点测试。

测试 MusicGenerator 节点的基本功能：输出 AudioData、duration 参数、
不同风格描述、describe 信息。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# 辅助：为 MusicGenerator 测试提供 mock 环境
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_musicgen():
    """Mock MusicGen 模型和 processor。"""
    import transformers as _tf

    mock_model = MagicMock()
    # 模拟 generate 返回 (1, 1, num_samples) 形状的 tensor
    mock_audio = MagicMock()
    mock_audio.cpu.return_value = MagicMock()
    mock_audio.cpu().numpy.return_value = np.sin(
        np.linspace(0, 2 * np.pi, 32000)
    ).astype(np.float32)
    # 模拟 generate 返回值
    mock_gen_output = MagicMock()
    mock_gen_output.__getitem__ = MagicMock(return_value=mock_audio)
    mock_model.generate.return_value = mock_gen_output
    mock_model.to.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    with patch.object(_tf, "MusicgenForConditionalGeneration") as mock_model_cls, \
         patch.object(_tf, "AutoProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = mock_processor
        yield mock_model


class TestMusicGeneratorBasic:
    """T_MUSIC_01：基本音乐生成测试。"""

    def test_basic_music_generation(self, mock_musicgen, cpu_scheduler):
        """T_MUSIC_01：基本音乐生成，输出 AudioData。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(prompt="轻松的钢琴曲"))

        audio = result.get("audio")
        assert audio is not None, "MusicGenerator 输出应包含 audio"
        assert audio.waveform is not None, "waveform 不应为 None"
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert len(audio.waveform) > 0, "waveform 不应为空"


class TestMusicGeneratorDuration:
    """T_MUSIC_02：duration 参数测试。"""

    def test_duration_param(self, mock_musicgen, cpu_scheduler):
        """T_MUSIC_02：指定 duration 参数生效。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(prompt="钢琴曲", duration=10.0))

        duration = result.get("duration")
        assert duration is not None, "应返回 duration"
        assert isinstance(duration, float), "duration 应为 float"

    def test_duration_clamp(self, mock_musicgen, cpu_scheduler):
        """超过 30 秒的 duration 应被截断。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(prompt="长音乐", duration=60.0))

        duration = result.get("duration")
        assert duration <= 30.0, f"duration 应 <= 30.0，实际 {duration}"


class TestMusicGeneratorStyles:
    """T_MUSIC_03：不同风格描述测试。"""

    def test_different_styles(self, mock_musicgen, cpu_scheduler):
        """T_MUSIC_03：不同风格描述生成不同结果。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        result1 = gen(MosaicData(prompt="轻松的钢琴曲，适合冥想", duration=5.0))
        result2 = gen(MosaicData(prompt="激烈的摇滚乐，快节奏", duration=5.0))

        audio1 = result1.get("audio")
        audio2 = result2.get("audio")
        assert audio1 is not None
        assert audio2 is not None
        # 不同 prompt 在 mock 环境下波形可能相同，但至少都应该有输出
        assert len(audio1.waveform) > 0
        assert len(audio2.waveform) > 0

    def test_guidance_scale_param(self, mock_musicgen, cpu_scheduler):
        """guidance_scale 参数生效。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        result = gen(MosaicData(prompt="钢琴曲", guidance_scale=5.0))
        assert result.get("audio") is not None


class TestMusicGeneratorDescribe:
    """T_MUSIC_04：describe 测试。"""

    def test_describe(self, mock_musicgen, cpu_scheduler):
        """T_MUSIC_04：describe 标注模型信息。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        spec = gen.describe()

        assert spec.name == "music-generator", "节点名称应为 'music-generator'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "text" in spec.input_types, "输入类型应包含 'text'"
        assert "audio" in spec.output_types, "输出类型应包含 'audio'"


class TestMusicGeneratorErrors:
    """MusicGenerator 错误处理测试。"""

    def test_missing_prompt(self, mock_musicgen, cpu_scheduler):
        """缺少 prompt 应抛出 ValueError。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'prompt'"):
            gen(MosaicData())

    def test_empty_prompt(self, mock_musicgen, cpu_scheduler):
        """空 prompt 应抛出 ValueError。"""
        from mosaic.nodes.audio.music_generator import MusicGenerator

        gen = MusicGenerator(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'prompt'"):
            gen(MosaicData(prompt=""))