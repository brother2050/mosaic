"""测试 DVAEDecoder。

依赖 torch；torch 不可用时整个模块自动跳过。使用小模型参数以节省时间，
不加载真实预训练权重。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.vocoders.dvae import DVAEDecoder

# 小模型参数，兼顾速度与覆盖
_NUM_VQ = 4
_NUM_AUDIO_TOKENS = 64
_HIDDEN_SIZE = 32
_MEL_BINS = 80
_NUM_LAYERS = 3


def _make_decoder(num_vq: int = _NUM_VQ) -> DVAEDecoder:
    """构造一个小参数 DVAEDecoder。"""
    return DVAEDecoder(
        num_vq=num_vq,
        num_audio_tokens=_NUM_AUDIO_TOKENS,
        hidden_size=_HIDDEN_SIZE,
        mel_bins=_MEL_BINS,
        num_layers=_NUM_LAYERS,
    )


def test_DVAE_01() -> None:
    """构造成功。"""
    decoder = _make_decoder()
    assert decoder is not None


def test_DVAE_02() -> None:
    """forward 输出 mel shape 正确 [mel_bins, frames]。"""
    import torch

    decoder = _make_decoder()
    tokens = torch.randint(0, _NUM_AUDIO_TOKENS, (_NUM_VQ, 50))
    mel = decoder.forward(tokens)
    assert mel.shape[0] == _MEL_BINS
    assert mel.shape[1] == 50


def test_DVAE_03() -> None:
    """forward batch 输入 [batch, mel_bins, frames]。"""
    import torch

    decoder = _make_decoder()
    tokens = torch.randint(0, _NUM_AUDIO_TOKENS, (2, _NUM_VQ, 50))
    mel = decoder.forward(tokens)
    assert mel.shape == (2, _MEL_BINS, 50)


def test_DVAE_04() -> None:
    """forward_chunk 流式解码。"""
    import torch

    decoder = _make_decoder()
    decoder.reset_stream_buffer()
    tokens = torch.randint(0, _NUM_AUDIO_TOKENS, (_NUM_VQ, 50))
    mel = decoder.forward_chunk(tokens)
    # 当前块对应的新帧，mel 维度正确
    assert mel.shape[0] == _MEL_BINS


def test_DVAE_05() -> None:
    """不同 num_vq 值的兼容性。"""
    import torch

    for num_vq in (2, 8):
        decoder = _make_decoder(num_vq=num_vq)
        tokens = torch.randint(0, _NUM_AUDIO_TOKENS, (num_vq, 40))
        mel = decoder.forward(tokens)
        assert mel.shape[0] == _MEL_BINS


def test_DVAE_06() -> None:
    """load_weights 不存在的路径处理。

    当前实现对不存在的路径采取优雅降级：返回空 state_dict，以随机初始化
    完成 load 并标记为已加载，不抛异常。本用例校验该实际行为。
    """
    import torch

    decoder = _make_decoder()
    # 不存在的路径不应导致崩溃
    decoder.load_weights(
        "/nonexistent/path/dvae.safetensors", device="cpu", dtype="float32"
    )
    # 加载后标记为已加载且仍可前向
    assert decoder._impl._is_loaded is True
    tokens = torch.randint(0, _NUM_AUDIO_TOKENS, (_NUM_VQ, 30))
    mel = decoder.forward(tokens)
    assert mel.shape[0] == _MEL_BINS
