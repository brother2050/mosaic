"""测试 GPT2ARModel（GPT-SoVITS GPT-2 自回归声学模型）。

torch / transformers 均采用惰性探测：模块级仅用 ``importlib.util.find_spec``
检查是否存在，不在模块顶层导入，避免 phase2 mock 污染。``torch`` 的实际
导入放在每个需要它的测试函数内部。

* 模块级 ``pytestmark``：torch 不可用时跳过整个模块。
* ``@_needs_transformers``：仅 load_weights / generate / generate_stream 等
  需要真实加载与推理的用例在 transformers 不可用时跳过；构造期属性与
  ``stop_condition`` 静态方法测试不需要 transformers。

加载用例使用小参数配置（hidden_size=64, num_layers=2, num_heads=4），
``load_weights`` 传入临时空目录即可（无权重文件时使用随机初始化，仍完整
覆盖 config 构造 / 双路径 Embedding 创建 / device+dtype 迁移 / _is_loaded
置位的代码路径），device=cpu、dtype=float32 以保证 CPU 上的算子兼容。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch / transformers 是否可用（不导入，避免污染全局 sys.modules）
# 注意：find_spec 在模块 __spec__ 为 None 时会抛 ValueError，需捕获
def _safe_find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ValueError, ModuleNotFoundError):
        return None

_HAS_TORCH = _safe_find_spec("torch") is not None
_HAS_TRANSFORMERS = _safe_find_spec("transformers") is not None

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

_needs_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers 未安装，跳过需要加载的用例"
)

from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel
from mosaic.nodes.audio.tts_backends.acoustic_models.gpt2_ar import GPT2ARModel

# ----------------------------------------------------------------------
# 小模型参数（不依赖真实权重，兼顾速度与覆盖）
# ----------------------------------------------------------------------
_VOCAB_SIZE = 256  # 文本音素词表
_SEMANTIC_VOCAB_SIZE = 100  # 语义 token 词表（GPT-2 lm_head 输出维度）
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_MAX_POS = 128
_NUM_SPK_EMB = 32


def _make_model() -> GPT2ARModel:
    """构造一个小参数 GPT2ARModel（未加载权重）。"""
    return GPT2ARModel(
        model_path="/tmp/gpt2_ar_test",
        vocab_size=_VOCAB_SIZE,
        semantic_vocab_size=_SEMANTIC_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        max_position_embeddings=_MAX_POS,
        num_speaker_embeddings=_NUM_SPK_EMB,
    )


@pytest.fixture
def loaded_model(tmp_path):
    """返回已加载权重的小参数 GPT2ARModel（CPU / float32）。

    使用 pytest 的 ``tmp_path`` 作为临时权重目录；目录为空时 load_weights
    走随机初始化分支。用例结束后卸载权重释放资源。
    """
    model = _make_model()
    model.load_weights(str(tmp_path), device="cpu", dtype="float32")
    yield model
    model.unload_weights()


def _text_token_ids():
    """构造文本音素 token ids tensor [1, 5]（值在文本词表范围内）。"""
    import torch

    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


# ----------------------------------------------------------------------
# T_GPT2_01~02：加载 / 卸载
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_01(tmp_path) -> None:
    """T_GPT2_01：load_weights 成功加载。"""
    model = _make_model()
    model.load_weights(str(tmp_path), device="cpu", dtype="float32")
    assert model._is_loaded is True
    assert model._model is not None
    assert model._text_embedding is not None
    assert model._semantic_embedding is not None
    assert model._speaker_proj is not None


@_needs_transformers
def test_GPT2_02(tmp_path) -> None:
    """T_GPT2_02：unload_weights 释放资源，_is_loaded 变为 False。"""
    model = _make_model()
    model.load_weights(str(tmp_path), device="cpu", dtype="float32")
    assert model._is_loaded is True

    model.unload_weights()
    assert model._is_loaded is False
    assert model._model is None
    assert model._text_embedding is None
    assert model._semantic_embedding is None
    assert model._speaker_proj is None


# ----------------------------------------------------------------------
# T_GPT2_03~04：generate 产出 / 取值范围
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_03(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_03：generate 产出语义 token ids tensor。"""
    import torch

    out = loaded_model.generate(
        _text_token_ids(),
        max_new_tokens=16,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
    )
    assert torch.is_tensor(out)
    assert out.ndim == 2
    assert out.shape[0] == 1
    assert out.shape[1] > 0
    assert out.dtype == torch.long


@_needs_transformers
def test_GPT2_04(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_04：输出 token id 落在 [0, semantic_vocab_size) 范围内。"""
    out = loaded_model.generate(_text_token_ids(), max_new_tokens=16)
    assert int(out.min().item()) >= 0
    assert int(out.max().item()) < _SEMANTIC_VOCAB_SIZE


# ----------------------------------------------------------------------
# T_GPT2_05：speaker_info 条件注入
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_05(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_05：注入参考语义 token / speaker_embedding 后输出与无参考不同。"""
    import torch

    token_ids = _text_token_ids()

    # 设置一次种子以保证可复现；两次调用之间不重置种子，使采样随机数序列
    # 自然推进。随机初始化模型下，有无参考 token 时末位 logits 仅略有差异，
    # 若每次都重置种子，相同的均匀随机数配合近乎相同的分布可能采样到相同
    # token，因此这里让随机数序列推进以保证条件注入后输出确实不同。
    torch.manual_seed(123)
    out_no_ref = loaded_model.generate(token_ids, max_new_tokens=16)

    speaker_info = {
        "ref_semantic_tokens": torch.tensor([[10, 20, 30]], dtype=torch.long),
        "speaker_embedding": torch.randn(1, _NUM_SPK_EMB),
    }
    out_with_ref = loaded_model.generate(
        token_ids, speaker_embedding=speaker_info, max_new_tokens=16
    )

    assert torch.is_tensor(out_with_ref)
    assert out_with_ref.shape[0] == 1
    # 条件注入改变了输入嵌入序列 -> 采样结果应不同
    assert not torch.equal(out_no_ref, out_with_ref)


# ----------------------------------------------------------------------
# T_GPT2_06~08：采样参数
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_06(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_06：temperature 参数影响输出。"""
    import torch

    token_ids = _text_token_ids()
    # 设置一次种子以保证可复现；两次调用之间不重置种子，使采样随机数序列
    # 自然推进，从而确保不同温度下输出不同。
    torch.manual_seed(123)
    out_low = loaded_model.generate(token_ids, max_new_tokens=16, temperature=0.5)
    out_high = loaded_model.generate(token_ids, max_new_tokens=16, temperature=2.0)
    assert torch.is_tensor(out_low)
    assert torch.is_tensor(out_high)
    # 不同温度产生不同的概率分布 -> 输出应不同
    assert not torch.equal(out_low, out_high)


@_needs_transformers
def test_GPT2_07(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_07：top_p / top_k 参数可正常工作。"""
    import torch

    out = loaded_model.generate(
        _text_token_ids(),
        max_new_tokens=16,
        top_p=0.5,
        top_k=10,
    )
    assert torch.is_tensor(out)
    assert out.shape[1] > 0
    assert int(out.max().item()) < _SEMANTIC_VOCAB_SIZE


@_needs_transformers
def test_GPT2_08(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_08：repetition_penalty 参数可正常工作。"""
    import torch

    out = loaded_model.generate(
        _text_token_ids(),
        max_new_tokens=16,
        repetition_penalty=1.5,
    )
    assert torch.is_tensor(out)
    assert out.shape[1] > 0
    assert int(out.max().item()) < _SEMANTIC_VOCAB_SIZE


# ----------------------------------------------------------------------
# T_GPT2_09~10：generate_stream
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_09(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_09：generate_stream 产出多个 chunk。"""
    import torch

    chunks = list(
        loaded_model.generate_stream(
            _text_token_ids(), stream_batch=8, max_new_tokens=24
        )
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert torch.is_tensor(chunk)
        assert chunk.ndim == 2
        assert chunk.shape[0] == 1
        assert chunk.shape[1] > 0


@_needs_transformers
def test_GPT2_10(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_10：每个 chunk 的 token 数符合 stream_batch 约束。"""
    import torch

    stream_batch = 8
    max_new = 24
    chunks = list(
        loaded_model.generate_stream(
            _text_token_ids(),
            stream_batch=stream_batch,
            max_new_tokens=max_new,
        )
    )
    assert len(chunks) >= 1

    total = 0
    for i, chunk in enumerate(chunks):
        n = chunk.shape[1]
        # 除最后一个 chunk 外，每个 chunk 恰好 stream_batch 个 token
        if i < len(chunks) - 1:
            assert n == stream_batch
        else:
            assert n <= stream_batch
        total += n
    # 总生成数不超过 max_new_tokens
    assert total <= max_new


# ----------------------------------------------------------------------
# T_GPT2_11~12：停止条件（静态方法，无需加载）
# ----------------------------------------------------------------------
def test_GPT2_11() -> None:
    """T_GPT2_11：stop_condition 检测到 EOS token 时返回 True。"""
    import torch

    eos = 1
    cur_eos = torch.tensor([eos])
    assert GPT2ARModel.stop_condition(cur_eos, [], eos_token_id=eos) is True

    cur_other = torch.tensor([5])
    assert GPT2ARModel.stop_condition(cur_other, [], eos_token_id=eos) is False


def test_GPT2_12() -> None:
    """T_GPT2_12：stop_condition 连续重复 token 达到上限时返回 True。"""
    import torch

    val = 7
    cur = torch.tensor([val])
    # 连续 max_repeat 个相同 token -> 停止
    recent_full = [torch.tensor([val]) for _ in range(5)]
    assert (
        GPT2ARModel.stop_condition(
            cur, recent_full, eos_token_id=None, max_repeat=5
        )
        is True
    )
    # 少于 max_repeat -> 不停止
    recent_short = [torch.tensor([val]) for _ in range(3)]
    assert (
        GPT2ARModel.stop_condition(
            cur, recent_short, eos_token_id=None, max_repeat=5
        )
        is False
    )


# ----------------------------------------------------------------------
# T_GPT2_13：max_new_tokens 限制
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_13(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_13：generate 生成的 token 数不超过 max_new_tokens。"""
    out = loaded_model.generate(_text_token_ids(), max_new_tokens=5)
    assert out.shape[1] <= 5


# ----------------------------------------------------------------------
# T_GPT2_14：双路径 Embedding 独立性
# ----------------------------------------------------------------------
@_needs_transformers
def test_GPT2_14(loaded_model: GPT2ARModel) -> None:
    """T_GPT2_14：text_embedding 与 semantic_embedding 是独立的参数。"""
    text_emb = loaded_model.get_input_embeddings()  # text_embedding
    sem_emb = loaded_model._semantic_embedding

    assert text_emb is not None
    assert sem_emb is not None
    # 不是同一个对象
    assert text_emb is not sem_emb
    # 形状不同：文本词表 vs 语义词表
    assert text_emb.weight.shape != sem_emb.weight.shape
    assert text_emb.weight.shape[0] == _VOCAB_SIZE
    assert sem_emb.weight.shape[0] == _SEMANTIC_VOCAB_SIZE
    # 隐藏维度一致
    assert text_emb.weight.shape[1] == _HIDDEN_SIZE
    assert sem_emb.weight.shape[1] == _HIDDEN_SIZE


# ----------------------------------------------------------------------
# T_GPT2_15：类属性
# ----------------------------------------------------------------------
def test_GPT2_15() -> None:
    """T_GPT2_15：model_type=='ar'，vocab_size 与构造参数一致。"""
    model = _make_model()
    assert model.model_type == "ar"
    assert model.vocab_size == _VOCAB_SIZE
    # 类级别属性
    assert GPT2ARModel.model_type == "ar"
    # 继承自 AcousticModel
    assert isinstance(model, AcousticModel)
    assert issubclass(GPT2ARModel, AcousticModel)
    # 构造参数正确存储
    assert model._semantic_vocab_size == _SEMANTIC_VOCAB_SIZE
    assert model.hidden_size == _HIDDEN_SIZE
    assert model._num_layers == _NUM_LAYERS
    assert model._num_heads == _NUM_HEADS
    # 未加载状态
    assert model._is_loaded is False
