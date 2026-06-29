"""测试 VocosVocoder。

依赖 torch；vocos 包未安装。本模块只测试构造、类属性、未加载时的安全
行为与 mel 基矩阵生成——这些都不依赖 vocos 包或真实权重。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.vocoders.vocos import VocosVocoder


def _make_vocoder() -> VocosVocoder:
    """构造一个 VocosVocoder（不加载权重）。"""
    return VocosVocoder(model_path="/tmp/vocos_test")


def test_VOCOS_01() -> None:
    """构造成功，类属性正确。"""
    vocoder = _make_vocoder()
    assert vocoder is not None
    assert VocosVocoder.vocoder_type == "vocos"
    assert VocosVocoder.input_type == "mel"


def test_VOCOS_02() -> None:
    """vocoder_type == 'vocos'。"""
    vocoder = _make_vocoder()
    assert vocoder.vocoder_type == "vocos"


def test_VOCOS_03() -> None:
    """input_type == 'mel'。"""
    vocoder = _make_vocoder()
    assert vocoder.input_type == "mel"


def test_VOCOS_04() -> None:
    """sample_rate 正确。"""
    vocoder = _make_vocoder()
    assert vocoder.sample_rate == 24000


def test_VOCOS_05() -> None:
    """未加载时 decode 抛出 RuntimeError。"""
    import torch

    vocoder = _make_vocoder()
    mel = torch.randn(1, 80, 50)
    with pytest.raises(RuntimeError):
        vocoder.decode(mel)


def test_VOCOS_06() -> None:
    """get_mel_basis 返回正确形状（无需加载即可计算）。"""
    vocoder = _make_vocoder()
    n_fft = 1024
    n_mels = 80
    basis = vocoder.get_mel_basis(n_fft=n_fft, sample_rate=24000, n_mels=n_mels)
    # mel 滤波器组形状：[n_mels, n_fft // 2 + 1]
    assert basis.shape[0] == n_mels
    assert basis.shape[1] == n_fft // 2 + 1
