"""测试 LlamaARModel。

注意：LlamaARModel 的导入不需要 transformers（惰性导入），但 load_weights
和 generate 需要 transformers。构造期属性与未加载状态下的安全行为测试
可以在无 transformers 环境下运行。
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# transformers 为可选依赖；仅在需要加载/推理的用例上跳过
try:  # noqa: SIM105 - 仅用于探测依赖
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except Exception:  # noqa: BLE001
    _HAS_TRANSFORMERS = False

_needs_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers 未安装，跳过需要加载的用例"
)

from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import (
    LlamaARModel,
    DualEmbedding,
)


# 小参数构造（不加载权重），节省时间
_NUM_VQ = 4
_NUM_AUDIO_TOKENS = 64
_NUM_TEXT_TOKENS = 120
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2


def _make_model() -> LlamaARModel:
    """构造一个小参数 LlamaARModel（未加载权重）。"""
    return LlamaARModel(
        model_path="/tmp/llama_ar_test",
        num_vq=_NUM_VQ,
        num_audio_tokens=_NUM_AUDIO_TOKENS,
        num_text_tokens=_NUM_TEXT_TOKENS,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
    )


def test_LAR_01() -> None:
    """构造成功，属性正确。"""
    model = _make_model()
    assert model is not None
    assert model.model_type == "ar"
    assert model.vocab_size > 0
    assert model.hidden_size == _HIDDEN_SIZE


def test_LAR_02() -> None:
    """vocab_size 计算正确 (num_text_tokens + num_audio_tokens * num_vq)。"""
    model = _make_model()
    expected = _NUM_TEXT_TOKENS + _NUM_AUDIO_TOKENS * _NUM_VQ
    assert model.vocab_size == expected


def test_LAR_03() -> None:
    """model_type == 'ar'。"""
    model = _make_model()
    assert model.model_type == "ar"


def test_LAR_04() -> None:
    """未加载时 get_input_embeddings 返回 None。"""
    model = _make_model()
    assert model.get_input_embeddings() is None


def test_LAR_05() -> None:
    """未加载时 get_output_head 返回 None。"""
    model = _make_model()
    assert model.get_output_head() is None


def test_LAR_06() -> None:
    """未加载时 generate 抛出 RuntimeError。"""
    import torch

    model = _make_model()
    with pytest.raises(RuntimeError):
        model.generate(torch.tensor([[1, 2, 3]]))


def test_LAR_07() -> None:
    """未加载时 generate_stream 抛出 RuntimeError。"""
    import torch

    model = _make_model()
    with pytest.raises(RuntimeError):
        list(model.generate_stream(torch.tensor([[1, 2, 3]])))


def test_LAR_08() -> None:
    """unload_weights 安全调用（未加载时）。"""
    model = _make_model()
    # 未加载状态下调用不应抛异常
    model.unload_weights()


def test_LAR_09() -> None:
    """DualEmbedding 类可导入。"""
    assert DualEmbedding is not None
    assert isinstance(DualEmbedding, type)


def test_LAR_10() -> None:
    """hidden_size 属性正确。"""
    model = _make_model()
    assert model.hidden_size == _HIDDEN_SIZE


def test_LAR_11() -> None:
    """num_vq 属性正确。"""
    model = _make_model()
    assert model._num_vq == _NUM_VQ


def test_LAR_12() -> None:
    """构造参数正确存储。"""
    model = _make_model()
    assert model._model_path == "/tmp/llama_ar_test"
    assert model._num_text_tokens == _NUM_TEXT_TOKENS
    assert model._num_audio_tokens == _NUM_AUDIO_TOKENS
    assert model._num_vq == _NUM_VQ
    assert model._hidden_size == _HIDDEN_SIZE
    assert model._num_layers == _NUM_LAYERS
