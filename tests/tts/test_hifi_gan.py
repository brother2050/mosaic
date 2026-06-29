"""测试 HiFiGanVocoder。

依赖 torch；torch 不可用时整个模块自动跳过。HiFi-GAN 是纯卷积生成器，
不需要 transformers。权重加载用不存在的路径触发优雅降级（随机初始化），
从而可在无预训练权重环境下测试 decode / decode_chunk 行为。

torch 导入放在函数内部，避免在模块顶层污染 sys.modules。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.vocoders.hifi_gan import HiFiGanVocoder

# 默认 mel 维度与采样率
_N_MELS = 80
_SAMPLE_RATE = 22050
# 缩小初始通道数以加速测试（不影响功能正确性）
_UPSAMPLE_INITIAL_CHANNEL = 64


def _make_vocoder(
    n_mels: int = _N_MELS,
    sample_rate: int = _SAMPLE_RATE,
) -> HiFiGanVocoder:
    """构造一个 HiFiGanVocoder（不加载权重）。"""
    return HiFiGanVocoder(
        model_path="/tmp/test",
        sample_rate=sample_rate,
        n_mels=n_mels,
        upsample_initial_channel=_UPSAMPLE_INITIAL_CHANNEL,
    )


def _make_loaded_vocoder(
    n_mels: int = _N_MELS,
    sample_rate: int = _SAMPLE_RATE,
) -> HiFiGanVocoder:
    """构造并加载（随机初始化）一个 HiFiGanVocoder。

    用不存在的路径触发优雅降级：state_dict 为空，以随机初始化完成 load
    并标记为已加载，使 decode / decode_chunk 可运行。
    """
    vocoder = _make_vocoder(n_mels=n_mels, sample_rate=sample_rate)
    vocoder.load_weights(
        "/nonexistent/path", device="cpu", dtype="float32"
    )
    return vocoder


# ----------------------------------------------------------------------
# HFG_01~04：构造 / 类属性 / 未加载安全行为
# ----------------------------------------------------------------------
def test_HFG_01() -> None:
    """构造成功，类属性正确。"""
    vocoder = _make_vocoder()
    assert vocoder is not None
    assert HiFiGanVocoder.vocoder_type == "hifi_gan"
    assert HiFiGanVocoder.input_type == "mel"
    # 实例属性与类属性一致
    assert vocoder.vocoder_type == "hifi_gan"
    assert vocoder.input_type == "mel"


def test_HFG_02() -> None:
    """sample_rate 正确。"""
    vocoder = HiFiGanVocoder(model_path="/tmp/test", sample_rate=22050)
    assert vocoder.sample_rate == 22050


def test_HFG_03() -> None:
    """未加载时 decode 抛 RuntimeError。"""
    import torch

    vocoder = _make_vocoder()
    mel = torch.randn(1, _N_MELS, 50)
    with pytest.raises(RuntimeError):
        vocoder.decode(mel)


def test_HFG_04() -> None:
    """未加载时 decode_chunk 抛 RuntimeError。"""
    import torch

    vocoder = _make_vocoder()
    mel = torch.randn(1, _N_MELS, 50)
    with pytest.raises(RuntimeError):
        vocoder.decode_chunk(mel)


# ----------------------------------------------------------------------
# HFG_05~07：加载后解码
# ----------------------------------------------------------------------
def test_HFG_05() -> None:
    """load_weights 后 decode 返回 (waveform, sample_rate) 元组。"""
    import torch

    vocoder = _make_loaded_vocoder()
    mel = torch.randn(1, _N_MELS, 50)
    result = vocoder.decode(mel)
    assert isinstance(result, tuple)
    assert len(result) == 2
    # 第二个元素为采样率
    assert result[1] == _SAMPLE_RATE


def test_HFG_06() -> None:
    """decode 输出 waveform 是 1D 或 2D tensor。"""
    import torch

    vocoder = _make_loaded_vocoder()
    mel = torch.randn(1, _N_MELS, 50)
    waveform, sr = vocoder.decode(mel)
    assert torch.is_tensor(waveform)
    assert waveform.dim() in (1, 2)


def test_HFG_07() -> None:
    """decode_chunk 流式解码：返回 (waveform, sample_rate) 元组。"""
    import torch

    vocoder = _make_loaded_vocoder()
    mel = torch.randn(1, _N_MELS, 50)
    result = vocoder.decode_chunk(mel)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == _SAMPLE_RATE


# ----------------------------------------------------------------------
# HFG_08~09：mel 滤波器组 / 卸载
# ----------------------------------------------------------------------
def test_HFG_08() -> None:
    """get_mel_basis 返回正确形状 [80, 513]。"""
    vocoder = _make_vocoder()
    basis = vocoder.get_mel_basis(
        n_fft=1024, sample_rate=22050, n_mels=80
    )
    # mel 滤波器组形状：[n_mels, n_fft // 2 + 1] = [80, 513]
    assert basis.shape[0] == 80
    assert basis.shape[1] == 1024 // 2 + 1


def test_HFG_09() -> None:
    """unload_weights 安全调用（未加载时）。"""
    vocoder = _make_vocoder()
    # 未加载状态下调用不应抛异常
    vocoder.unload_weights()
    assert vocoder._is_loaded is False


# ----------------------------------------------------------------------
# HFG_10：不同 n_mels 兼容
# ----------------------------------------------------------------------
def test_HFG_10() -> None:
    """不同 n_mels 输入兼容（n_mels=128）。"""
    import torch

    vocoder = _make_loaded_vocoder(n_mels=128)
    mel = torch.randn(1, 128, 50)
    result = vocoder.decode(mel)
    assert isinstance(result, tuple)
    assert result[1] == _SAMPLE_RATE
