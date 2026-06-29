"""测试 VQDecoder。

依赖 torch；torch 不可用时整个模块自动跳过。使用小模型参数以节省时间，
不加载真实预训练权重（load_weights 用不存在的路径触发优雅降级）。

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

from mosaic.nodes.audio.tts_backends.vocoders.vq_decoder import VQDecoder

# 小模型参数，兼顾速度与覆盖
_CODEBOOK_SIZE = 100
_CODEBOOK_DIM = 8
_HIDDEN_SIZE = 64
_MEL_BINS = 80
_NUM_LAYERS = 2


def _make_decoder(
    num_codebooks: int = 1,
    mel_bins: int = _MEL_BINS,
    codebook_size: int = _CODEBOOK_SIZE,
) -> VQDecoder:
    """构造一个小参数 VQDecoder。"""
    return VQDecoder(
        codebook_size=codebook_size,
        codebook_dim=_CODEBOOK_DIM,
        num_codebooks=num_codebooks,
        hidden_size=_HIDDEN_SIZE,
        mel_bins=mel_bins,
        num_layers=_NUM_LAYERS,
    )


# ----------------------------------------------------------------------
# VQD_01~03：基本前向推理
# ----------------------------------------------------------------------
def test_VQD_01() -> None:
    """基本前向推理：输出 mel 维度为 mel_bins(80)。

    2D 输入 [num_codebooks, frames] 经 forward 返回 [mel_bins, frames]，
    故 mel 维度位于第 0 维。
    """
    import torch

    decoder = _make_decoder()
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 50))
    mel = decoder.forward(tokens)
    assert mel.shape[0] == _MEL_BINS


def test_VQD_02() -> None:
    """输入 2D token ids [num_codebooks, frames]，输出 mel 维度为 80。"""
    import torch

    decoder = _make_decoder()
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 50))
    mel = decoder.forward(tokens)
    # [mel_bins, frames]
    assert mel.shape[0] == _MEL_BINS
    assert mel.shape[1] == 50


def test_VQD_03() -> None:
    """输出 mel 值在合理范围：所有值有限。"""
    import torch

    decoder = _make_decoder()
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 50))
    mel = decoder.forward(tokens)
    assert torch.isfinite(mel).all()


# ----------------------------------------------------------------------
# VQD_04：流式解码
# ----------------------------------------------------------------------
def test_VQD_04() -> None:
    """forward_chunk 流式解码：输出不为空。"""
    import torch

    decoder = _make_decoder()
    decoder.reset_stream_buffer()
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 50))
    mel = decoder.forward_chunk(tokens)
    assert mel.numel() > 0
    # mel 维度正确
    assert mel.shape[0] == _MEL_BINS


# ----------------------------------------------------------------------
# VQD_05~06：单码本 / 多码本
# ----------------------------------------------------------------------
def test_VQD_05() -> None:
    """单码本输入（num_codebooks=1）正常工作。"""
    import torch

    decoder = _make_decoder(num_codebooks=1)
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 50))
    mel = decoder.forward(tokens)
    assert mel.shape[0] == _MEL_BINS


def test_VQD_06() -> None:
    """多码本输入（num_codebooks=2）正常工作。"""
    import torch

    decoder = _make_decoder(num_codebooks=2)
    # 输入 [num_codebooks=2, frames=50]
    tokens = torch.randint(0, _CODEBOOK_SIZE, (2, 50))
    mel = decoder.forward(tokens)
    # 拼接策略：codebook_dim * num_codebooks = 8 * 2 = 16 -> hidden
    assert mel.shape[0] == _MEL_BINS


# ----------------------------------------------------------------------
# VQD_07~08：权重加载 / 流式缓冲区
# ----------------------------------------------------------------------
def test_VQD_07() -> None:
    """load_weights 不存在的路径优雅降级，不崩溃。"""
    import torch

    decoder = _make_decoder()
    # 不存在的路径不应导致崩溃，以随机初始化完成 load 并标记为已加载
    decoder.load_weights(
        "/nonexistent/path", device="cpu", dtype="float32"
    )
    assert decoder._is_loaded is True
    # 加载后仍可前向
    tokens = torch.randint(0, _CODEBOOK_SIZE, (1, 30))
    mel = decoder.forward(tokens)
    assert mel.shape[0] == _MEL_BINS


def test_VQD_08() -> None:
    """reset_stream_buffer 安全调用。"""
    decoder = _make_decoder()
    # 未调用 forward_chunk 前缓冲区为 None，重置不应抛异常
    decoder.reset_stream_buffer()
