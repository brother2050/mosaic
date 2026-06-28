# tests/phase3/test_audio_types.py
"""Phase 3 音频数据类型测试。

测试 AudioData 的创建、序列化、不同采样率/声道处理以及时长计算。
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.core.types import AudioData, data_from_dict


class TestAudioDataCreation:
    """T_AUDTYPE_01：AudioData 创建测试。"""

    def test_create_with_waveform_and_sample_rate(self):
        """T_AUDTYPE_01：AudioData 创建，包含 waveform 和 sample_rate。"""
        sr = 22050
        waveform = np.zeros(1000, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)

        assert audio.waveform is not None, "waveform 不应为 None"
        assert audio.sample_rate == sr, f"sample_rate 应为 {sr}，实际 {audio.sample_rate}"
        assert isinstance(audio.waveform, np.ndarray), "waveform 应为 numpy.ndarray"
        assert audio.data_type == "audio", "data_type 应为 'audio'"

    def test_create_with_default_sample_rate(self):
        """默认采样率应为 22050。"""
        waveform = np.zeros(500, dtype=np.float32)
        audio = AudioData(waveform=waveform)
        assert audio.sample_rate == 22050, "默认 sample_rate 应为 22050"

    def test_create_with_metadata(self):
        """创建时可以附带 metadata。"""
        waveform = np.zeros(100, dtype=np.float32)
        audio = AudioData(
            waveform=waveform,
            sample_rate=16000,
            metadata={"duration": 5.0, "format": "wav"},
        )
        assert audio.metadata["duration"] == 5.0, "metadata duration 不正确"
        assert audio.metadata["format"] == "wav", "metadata format 不正确"


class TestAudioDataSerialization:
    """T_AUDTYPE_02：AudioData 序列化/反序列化测试。"""

    def test_roundtrip(self):
        """T_AUDTYPE_02：AudioData 序列化/反序列化（waveform 保存为 numpy）。"""
        sr = 22050
        waveform = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, metadata={"key": "val"})

        # 序列化
        d = audio.to_dict()
        assert "__data_type__" in d, "序列化后应包含 __data_type__"
        assert d["__data_type__"] == "audio", "data_type 应为 'audio'"

        # 反序列化
        restored = data_from_dict(d)
        assert isinstance(restored, AudioData), "反序列化后应为 AudioData"
        assert restored.sample_rate == sr, f"sample_rate 应为 {sr}"
        assert isinstance(restored.waveform, np.ndarray), "waveform 应为 ndarray"
        np.testing.assert_array_almost_equal(
            restored.waveform, waveform, decimal=5,
            err_msg="反序列化后 waveform 与原始不一致",
        )
        assert restored.metadata["key"] == "val", "metadata 应保留"

    def test_dict_like_access(self):
        """AudioData 支持字典式访问。"""
        audio = AudioData(waveform=np.zeros(10), sample_rate=44100)
        assert audio["waveform"] is not None
        assert audio["sample_rate"] == 44100
        assert "metadata" in audio


class TestAudioDataSampleRate:
    """T_AUDTYPE_03：不同采样率的 AudioData 处理。"""

    def test_low_sample_rate(self):
        """T_AUDTYPE_03：低采样率（8000Hz）处理。"""
        sr = 8000
        waveform = np.zeros(100, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)
        assert audio.sample_rate == sr
        assert audio.validate(audio), "低采样率 AudioData 应通过校验"

    def test_high_sample_rate(self):
        """高采样率（48000Hz）处理。"""
        sr = 48000
        waveform = np.zeros(100, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)
        assert audio.sample_rate == sr
        assert audio.validate(audio), "高采样率 AudioData 应通过校验"

    def test_custom_sample_rate(self):
        """自定义采样率（44100Hz）处理。"""
        sr = 44100
        waveform = np.zeros(100, dtype=np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)
        assert audio.sample_rate == sr
        assert audio.validate(audio)


class TestAudioDataChannels:
    """T_AUDTYPE_04：单声道和立体声的 AudioData 处理。"""

    def test_mono_audio(self):
        """T_AUDTYPE_04：单声道 (samples,) 形状。"""
        sr = 22050
        waveform = np.random.randn(1000).astype(np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)
        assert audio.waveform.ndim == 1, "单声道应为 1 维"
        assert audio.waveform.shape == (1000,), "单声道形状应为 (samples,)"

    def test_stereo_audio(self):
        """立体声 (channels, samples) 形状。"""
        sr = 22050
        waveform = np.random.randn(2, 1000).astype(np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr)
        assert audio.waveform.ndim == 2, "立体声应为 2 维"
        assert audio.waveform.shape[0] == 2, "立体声第一维应为通道数"


class TestAudioDataDuration:
    """T_AUDTYPE_05：AudioData 时长计算正确。"""

    def test_duration_calculation(self):
        """T_AUDTYPE_05：AudioData 时长计算正确。"""
        sr = 22050
        duration = 3.0
        num_samples = int(sr * duration)
        waveform = np.zeros(num_samples, dtype=np.float32)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sr,
            metadata={"duration": duration},
        )
        # 时长 = 样本数 / 采样率
        calc_duration = num_samples / sr
        assert abs(calc_duration - duration) < 0.001, f"计算时长 {calc_duration} 与预期 {duration} 偏差过大"

    def test_duration_zero_samples(self):
        """零样本的时长应为 0。"""
        audio = AudioData(waveform=np.array([], dtype=np.float32), sample_rate=22050)
        assert len(audio.waveform) == 0, "零样本波形应为空"


class TestAudioDataValidation:
    """AudioData 校验测试。"""

    def test_validate_correct(self):
        """正确的 AudioData 应通过校验。"""
        audio = AudioData(waveform=np.zeros(10), sample_rate=22050)
        assert AudioData.validate(audio), "正确 AudioData 应通过校验"

    def test_validate_invalid_sample_rate(self):
        """sample_rate 非正数应不通过校验。"""
        audio = AudioData(waveform=np.zeros(10), sample_rate=0)
        assert not AudioData.validate(audio), "sample_rate=0 不应通过校验"

    def test_validate_wrong_type(self):
        """非 AudioData 类型不应通过校验。"""
        from mosaic.core.types import TextData

        text = TextData(content="test")
        assert not AudioData.validate(text), "TextData 不应通过 AudioData 校验"