"""测试 FishLlamaARModel。

注意：FishLlamaARModel / LlamaARModel 的模块导入采用惰性策略，构造期与
未加载状态下的安全行为测试不需要 transformers。仅在 load_weights / 真正
推理时才需要 transformers（当前环境未安装，相关用例自动跳过）。

torch 导入放在函数内部，避免在模块顶层污染 sys.modules。
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.acoustic_models.fish_ar import (
    FishLlamaARModel,
    UnifiedEmbedding,
)
from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import (
    LlamaARModel,
    LlamaARModelBase,
)
from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel

# 小模型参数（不加载权重），兼顾速度与覆盖
_TEXT_VOCAB_SIZE = 1000
_AUDIO_VOCAB_SIZE = 1024
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2


def _make_model() -> FishLlamaARModel:
    """构造一个小参数 FishLlamaARModel（未加载权重）。"""
    return FishLlamaARModel(
        model_path="/tmp/test",
        text_vocab_size=_TEXT_VOCAB_SIZE,
        audio_vocab_size=_AUDIO_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
    )


# ----------------------------------------------------------------------
# FAR_01~04：构造 / 卸载 / 未加载生成安全行为
# ----------------------------------------------------------------------
def test_FAR_01() -> None:
    """构造成功。"""
    model = _make_model()
    assert model is not None


def test_FAR_02() -> None:
    """unload_weights 安全调用（未加载时）。"""
    model = _make_model()
    # 未加载状态下调用不应抛异常
    model.unload_weights()


def test_FAR_03() -> None:
    """generate 未加载时抛 RuntimeError。"""
    import torch

    model = _make_model()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises(RuntimeError):
        model.generate(token_ids)


def test_FAR_04() -> None:
    """generate_stream 未加载时抛 RuntimeError。"""
    import torch

    model = _make_model()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises(RuntimeError):
        list(model.generate_stream(token_ids))


# ----------------------------------------------------------------------
# FAR_05, FAR_12：UnifiedEmbedding
# ----------------------------------------------------------------------
def test_FAR_05() -> None:
    """统一 Embedding 层正确：是 nn.Module，forward 返回正确 shape。"""
    import torch
    import torch.nn as nn

    emb = UnifiedEmbedding(total_vocab_size=100, hidden_size=32)
    assert isinstance(emb, nn.Module)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = emb(input_ids)
    assert out.shape == (1, 3, 32)


def test_FAR_12() -> None:
    """UnifiedEmbedding forward 输出 shape [batch, seq, hidden]。"""
    import torch

    emb = UnifiedEmbedding(total_vocab_size=50, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    out = emb(input_ids)
    # [batch=1, seq=5, hidden=16]
    assert out.dim() == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 5
    assert out.shape[2] == 16


# ----------------------------------------------------------------------
# FAR_06~11：属性正确性
# ----------------------------------------------------------------------
def test_FAR_06() -> None:
    """vocab_size 计算：text_vocab_size + audio_vocab_size == vocab_size。"""
    model = _make_model()
    assert model.vocab_size == _TEXT_VOCAB_SIZE + _AUDIO_VOCAB_SIZE


def test_FAR_07() -> None:
    """model_type == 'ar'。"""
    model = _make_model()
    assert model.model_type == "ar"


def test_FAR_08() -> None:
    """未加载时 get_input_embeddings 返回 None。"""
    model = _make_model()
    assert model.get_input_embeddings() is None


def test_FAR_09() -> None:
    """未加载时 get_output_head 返回 None。"""
    model = _make_model()
    assert model.get_output_head() is None


def test_FAR_10() -> None:
    """codec_type 属性正确存储。"""
    model = _make_model()
    # 默认 codec_type 为 "dac"
    assert model._codec_type == "dac"


def test_FAR_11() -> None:
    """hidden_size 属性正确。"""
    model = _make_model()
    assert model.hidden_size == _HIDDEN_SIZE


# ----------------------------------------------------------------------
# FAR_13~15：类继承关系（回归测试）
# ----------------------------------------------------------------------
def test_FAR_13() -> None:
    """LlamaARModelBase 是 AcousticModel 子类。"""
    assert issubclass(LlamaARModelBase, AcousticModel)


def test_FAR_14() -> None:
    """FishLlamaARModel 是 LlamaARModelBase 子类。"""
    assert issubclass(FishLlamaARModel, LlamaARModelBase)


def test_FAR_15() -> None:
    """ChatTTS LlamaARModel 仍然是 LlamaARModelBase 子类（回归测试）。"""
    assert issubclass(LlamaARModel, LlamaARModelBase)
