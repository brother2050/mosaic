# tests/phase3/test_audio_base.py
"""Phase 3 音频域基类测试。

测试 BaseAudioNode 的静态工具方法：_load_audio、_resample、_to_mono、_normalize。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.core.types import AudioData


def _has_soundfile():
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


class TestLoadAudio:
    """T_ABASE_01-02：_load_audio 测试。"""

    @pytest.mark.skipif(
        not _has_soundfile(),
        reason="soundfile not installed; skip file-based load test.",
    )
    def test_load_from_file(self, sample_audio):
        """T_ABASE_01：_load_audio 从文件加载。"""
        # 保存测试音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import soundfile as sf

            sf.write(tmp_path, sample_audio.waveform, sample_audio.sample_rate)
            waveform, sr = BaseAudioNode._load_audio(tmp_path)
            assert sr == sample_audio.sample_rate, f"采样率应为 {sample_audio.sample_rate}"
            assert isinstance(waveform, np.ndarray), "返回应为 numpy.ndarray"
            assert len(waveform) > 0, "波形不应为空"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def test_load_from_array(self, sample_audio):
        """T_ABASE_02：_load_audio 从数组加载。"""
        waveform, sr = BaseAudioNode._load_audio(sample_audio.waveform)
        assert sr == 22050, "默认采样率 22050"
        assert isinstance(waveform, np.ndarray), "返回应为 numpy.ndarray"
        np.testing.assert_array_equal(waveform, sample_audio.waveform, "波形应一致")

    def test_load_from_audio_data(self, sample_audio):
        """从 AudioData 实例加载。"""
        waveform, sr = BaseAudioNode._load_audio(sample_audio)
        assert sr == sample_audio.sample_rate
        np.testing.assert_array_equal(waveform, sample_audio.waveform)

    def test_load_invalid_type(self):
        """无效类型应抛出 TypeError。"""
        with pytest.raises(TypeError, match="Expected file path"):
            BaseAudioNode._load_audio(123)


class TestResample:
    """T_ABASE_03：_resample 重采样测试。"""

    def test_resample_down(self):
        """T_ABASE_03：_resample 重采样正确（降采样）。"""
        sr_orig = 44100
        sr_target = 22050
        t = np.linspace(0, 1, sr_orig, endpoint=False)
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        resampled = BaseAudioNode._resample(waveform, sr_orig, sr_target)
        expected_len = int(len(waveform) * sr_target / sr_orig)
        assert len(resampled) == expected_len, (
            f"重采样后长度应为 {expected_len}，实际 {len(resampled)}"
        )
        assert isinstance(resampled, np.ndarray), "返回应为 numpy.ndarray"

    def test_resample_same_rate(self):
        """相同采样率应返回原波形。"""
        waveform = np.random.randn(1000).astype(np.float32)
        result = BaseAudioNode._resample(waveform, 22050, 22050)
        np.testing.assert_array_equal(result, waveform, "相同采样率应返回原波形")

    def test_resample_up(self):
        """升采样。"""
        sr_orig = 8000
        sr_target = 16000
        waveform = np.random.randn(800).astype(np.float32)
        resampled = BaseAudioNode._resample(waveform, sr_orig, sr_target)
        expected_len = int(len(waveform) * sr_target / sr_orig)
        assert len(resampled) == expected_len

    def test_resample_stereo(self, stereo_audio):
        """立体声重采样。"""
        waveform = stereo_audio.waveform  # (2, samples)
        resampled = BaseAudioNode._resample(waveform, 22050, 11025)
        assert resampled.ndim == 2, "立体声重采样后仍应为 2 维"
        assert resampled.shape[0] == 2, "通道数应保持"


class TestToMono:
    """T_ABASE_04：_to_mono 转换测试。"""

    def test_mono_passthrough(self):
        """T_ABASE_04：_to_mono 转换正确 —— 单声道保持。"""
        mono = np.random.randn(1000).astype(np.float32)
        result = BaseAudioNode._to_mono(mono)
        assert result.ndim == 1, "单声道应保持 1 维"
        np.testing.assert_array_equal(result, mono, "单声道应保持不变")

    def test_stereo_to_mono(self, stereo_audio):
        """立体声转单声道（取平均）。"""
        mono = BaseAudioNode._to_mono(stereo_audio.waveform)
        assert mono.ndim == 1, "立体声转单声道后应为 1 维"
        assert len(mono) == stereo_audio.waveform.shape[1], "样本数应一致"

    def test_multi_channel_to_mono(self):
        """多声道转单声道。"""
        multi = np.random.randn(4, 500).astype(np.float32)
        mono = BaseAudioNode._to_mono(multi)
        assert mono.ndim == 1
        assert len(mono) == 500


class TestNormalize:
    """T_ABASE_05：_normalize 归一化测试。"""

    def test_normalize(self):
        """T_ABASE_05：_normalize 归一化正确。"""
        waveform = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        normalized = BaseAudioNode._normalize(waveform)
        assert np.max(np.abs(normalized)) <= 1.0, "归一化后最大值不应超过 1.0"
        # 原波形最大值为 1.0，归一化后应不变
        np.testing.assert_array_almost_equal(normalized, waveform, err_msg="最大值已为 1 时归一化应不变")

    def test_normalize_large_values(self):
        """大数据幅值归一化。"""
        waveform = np.array([0.0, 2.0, -2.0, 0.5], dtype=np.float32)
        normalized = BaseAudioNode._normalize(waveform)
        assert np.max(np.abs(normalized)) <= 1.0, "归一化后最大值不应超过 1.0"
        assert normalized[1] == 1.0, "原 2.0 应归一化为 1.0"
        assert normalized[2] == -1.0, "原 -2.0 应归一化为 -1.0"

    def test_normalize_zero(self):
        """全零波形归一化。"""
        waveform = np.zeros(100, dtype=np.float32)
        normalized = BaseAudioNode._normalize(waveform)
        np.testing.assert_array_equal(normalized, waveform, "全零波形归一化应不变")


class TestGetDuration:
    """_get_duration 辅助方法测试。"""

    def test_get_duration(self):
        """时长计算正确。"""
        sr = 22050
        waveform = np.zeros(sr * 3, dtype=np.float32)  # 3 秒
        duration = BaseAudioNode._get_duration(waveform, sr)
        assert duration == 3.0, f"时长应为 3.0，实际 {duration}"

    def test_get_duration_stereo(self, stereo_audio):
        """立体声时长计算（取最后一个轴）。"""
        waveform = stereo_audio.waveform
        sr = stereo_audio.sample_rate
        duration = BaseAudioNode._get_duration(waveform, sr)
        expected = waveform.shape[-1] / sr
        assert abs(duration - expected) < 0.001


class TestSaveAudio:
    """_save_audio 辅助方法测试。"""

    @pytest.mark.skip(reason="mock soundfile.write does not actually write files; requires real soundfile")
    def test_save_and_load(self, sample_audio):
        """保存后重新加载，数据一致。"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            BaseAudioNode._save_audio(
                sample_audio.waveform, sample_audio.sample_rate, tmp_path
            )
            assert os.path.exists(tmp_path), "文件应存在"
            assert os.path.getsize(tmp_path) > 0, "文件不应为空"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass