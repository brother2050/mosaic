# tests/phase3/test_voice_clone.py
"""Phase 3 语音风格匹配节点测试。

测试 VoiceClone 节点的基本功能：输出 AudioData、输出时长与文本长度相关、
从文件路径输入参考音频、describe 信息、风格匹配逻辑。

新版 VoiceClone 不再依赖 Coqui XTTS-v2，改为基于 edge-tts 的"语音风格
匹配"：分析参考音频特征（时长/语速/基频代理量），从 edge-tts 预设语音
中选择最匹配的，并迁移参考音频的语速。因此本测试仅依赖 conftest 提供
的 edge-tts / soundfile mock，不再 mock Coqui TTS。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mosaic.core.types import MosaicData


def _has_soundfile():
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


class TestVoiceCloneBasic:
    """T_CLONE_01：基本语音风格匹配合成测试。"""

    def test_basic_voice_clone(self, sample_audio, cpu_scheduler):
        """T_CLONE_01：基本合成，输出 AudioData。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        result = cloner(MosaicData(
            reference_audio=sample_audio,
            text="你好，这是我的克隆声音。",
        ))

        audio = result.get("audio")
        assert audio is not None, "VoiceClone 输出应包含 audio"
        assert audio.waveform is not None, "waveform 不应为 None"
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert len(audio.waveform) > 0, "waveform 不应为空"

        # 参考音频也应被保留
        ref_audio = result.get("reference_audio")
        assert ref_audio is not None, "输出应包含 reference_audio"

    def test_default_backend_is_edge_tts(self, sample_audio, cpu_scheduler):
        """默认后端为 edge_tts（不再依赖 Coqui）。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        assert cloner._backend == "edge_tts"


class TestVoiceCloneDuration:
    """T_CLONE_02：输出时长与文本长度相关。"""

    def test_duration_vs_text_length(self, sample_audio, cpu_scheduler):
        """T_CLONE_02：输出时长与文本长度相关。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)

        # 短文本
        result_short = cloner(MosaicData(
            reference_audio=sample_audio,
            text="你好。",
        ))
        # 长文本
        result_long = cloner(MosaicData(
            reference_audio=sample_audio,
            text="这是一段比较长的文本，用于测试语音克隆的输出时长是否与文本长度相关。",
        ))

        assert result_short.get("audio") is not None
        assert result_long.get("audio") is not None
        # 长文本波形应更长（mock 环境下分句更多 -> 拼接更长）
        assert len(result_long.get("audio").waveform) > 0
        assert (
            len(result_long.get("audio").waveform)
            >= len(result_short.get("audio").waveform)
        )


class TestVoiceCloneFileInput:
    """T_CLONE_03：从文件路径输入参考音频。"""

    @pytest.mark.skipif(
        not _has_soundfile(),
        reason="soundfile not installed; skip file-based voice clone test.",
    )
    def test_from_file_path(self, sample_audio, cpu_scheduler):
        """T_CLONE_03：从文件路径输入参考音频。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        # 保存参考音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import soundfile as sf

            sf.write(tmp_path, sample_audio.waveform, sample_audio.sample_rate)

            cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
            result = cloner(MosaicData(
                reference_audio=tmp_path,
                text="测试从文件路径克隆。",
            ))

            audio = result.get("audio")
            assert audio is not None, "从文件路径输入应成功合成"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class TestVoiceCloneStyleMatching:
    """T_CLONE_05：语音风格匹配逻辑测试。"""

    def test_explicit_voice_override(self, sample_audio, cpu_scheduler):
        """显式 voice 优先于自动匹配。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        voice = cloner._match_voice(
            "zh", "neutral", "zh-CN-YunxiaNeural",
            sample_audio.waveform, sample_audio.sample_rate,
        )
        assert voice == "zh-CN-YunxiaNeural"

    def test_male_emotion_selects_male_voice(self, sample_audio, cpu_scheduler):
        """emotion='male' 显式选择男声。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        voice = cloner._match_voice(
            "zh", "male", None,
            sample_audio.waveform, sample_audio.sample_rate,
        )
        assert voice == "zh-CN-YunjianNeural"

    def test_high_pitch_selects_female_voice(self, sample_audio, cpu_scheduler):
        """高基频参考音频（440Hz 正弦）应选择女声（neutral）。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        # sample_audio 是 440Hz 正弦波，基频代理量高 -> 女声
        voice = cloner._match_voice(
            "zh", "neutral", None,
            sample_audio.waveform, sample_audio.sample_rate,
        )
        assert voice == "zh-CN-XiaoxiaoNeural"

    def test_estimate_speech_rate(self, sample_audio, cpu_scheduler):
        """语速估计返回合理区间。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        rate = cloner._estimate_speech_rate(
            sample_audio.waveform, sample_audio.sample_rate,
            "你好，这是我的克隆声音。", "zh",
        )
        assert 0.5 <= rate <= 2.0

    def test_estimate_pitch_proxy(self, sample_audio, cpu_scheduler):
        """基频代理量估计为正值。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="zh", scheduler=cpu_scheduler)
        cloner.load()
        pitch = cloner._estimate_pitch_proxy(
            sample_audio.waveform, sample_audio.sample_rate,
        )
        assert pitch > 0.0


class TestVoiceCloneDescribe:
    """T_CLONE_04：describe 测试。"""

    def test_describe(self, cpu_scheduler):
        """T_CLONE_04：describe 标注模型信息。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        spec = cloner.describe()

        assert spec.name == "voice-clone", "节点名称应为 'voice-clone'"
        assert spec.domain == "audio", "领域应为 'audio'"
        assert "audio" in spec.input_types, "输入类型应包含 'audio'"
        assert "text" in spec.input_types, "输入类型应包含 'text'"
        assert "audio" in spec.output_types, "输出类型应包含 'audio'"
        # edge-tts 不需要 GPU 显存
        assert spec.model_info.get("vram_gb") == 0.0


class TestVoiceCloneErrors:
    """VoiceClone 错误处理测试。"""

    def test_missing_reference_audio(self, cpu_scheduler):
        """缺少 reference_audio 应抛出 ValueError。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'reference_audio'"):
            cloner(MosaicData(text="你好"))

    def test_missing_text(self, sample_audio, cpu_scheduler):
        """缺少 text 应抛出 ValueError。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(scheduler=cpu_scheduler)
        with pytest.raises(ValueError, match="requires 'text'"):
            cloner(MosaicData(reference_audio=sample_audio))

    def test_language_param(self, sample_audio, cpu_scheduler):
        """语言参数可指定。"""
        from mosaic.nodes.audio.voice_clone import VoiceClone

        cloner = VoiceClone(language="en", scheduler=cpu_scheduler)
        result = cloner(MosaicData(
            reference_audio=sample_audio,
            text="Hello, this is my cloned voice.",
        ))
        assert result.get("audio") is not None
