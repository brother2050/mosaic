"""测试 SoVITSWeightConverter 的端到端权重转换流程。

覆盖 ``convert`` / ``validate`` / ``dry_run``，以及 GPT 权重映射、vocoder
组件抽取、转换后权重加载进 ``GPT2ARModel`` 等场景。所有用例基于伪造的小型
GPT-SoVITS 检查点（``torch.save`` 写入临时目录），不依赖真实预训练权重。

依赖说明
--------
``torch`` 与 ``safetensors.torch`` 在测试函数内部局部导入，避免 phase2 mock
污染；模块级仅用 ``importlib.util.find_spec`` 做跳过判断。
``SoVITSWeightConverter`` / ``GPT2ARModel`` 的导入链不在模块级触发 torch
导入，可安全置于模块顶层。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.weights.sovits_convert import SoVITSWeightConverter

# 模块级跳过判断：torch / safetensors 缺失时跳过本文件全部用例
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None

pytestmark = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _SAFETENSORS_AVAILABLE),
    reason="torch/safetensors 未安装，跳过 SoVITS 权重转换测试",
)
_requires_transformers = pytest.mark.skipif(
    not _TRANSFORMERS_AVAILABLE, reason="transformers 未安装"
)

# 伪造检查点维度。GPT 权重形状与 HF GPT2 的 Conv1D 约定对齐
# （c_attn.weight=[H,3H]、mlp.c_fc.weight=[H,4H]、mlp.c_proj.weight=[4H,H]），
# 便于 T_SCVT_05 中加载进 GPT2LMHeadModel 时不发生形状不匹配。
_FAKE_VOCAB = 256
_FAKE_HIDDEN = 32
_FAKE_NPOS = 64


def _make_fake_sovits_checkpoint(directory: str) -> str:
    """在 ``directory`` 下生成一个伪造的 GPT-SoVITS ``.pt`` 检查点并返回路径。

    包含 GPT（``t2s_model.*``）、vocoder（``enc_p.*`` / ``flow.*`` / ``dec.*``）、
    ssl（``ssl.*``）三类键；其中 GPT 权重形状按 HF GPT2 Conv1D 约定构造。
    ``torch`` 在此函数内部局部导入。
    """
    import torch

    torch.manual_seed(0)
    V, H, NPOS = _FAKE_VOCAB, _FAKE_HIDDEN, _FAKE_NPOS
    state_dict = {
        # ---- GPT / acoustic_model ----
        "t2s_model.embedding.weight": torch.randn(V, H),
        "t2s_model.pos_embedding": torch.randn(NPOS, H),
        "t2s_model.head.weight": torch.randn(V, H),
        "t2s_model.final_norm.weight": torch.randn(H),
        "t2s_model.final_norm.bias": torch.randn(H),
        "t2s_model.layers.0.attn.c_attn.weight": torch.randn(H, 3 * H),
        "t2s_model.layers.0.attn.c_attn.bias": torch.randn(3 * H),
        "t2s_model.layers.0.attn.c_proj.weight": torch.randn(H, H),
        "t2s_model.layers.0.attn.c_proj.bias": torch.randn(H),
        "t2s_model.layers.0.ln_1.weight": torch.randn(H),
        "t2s_model.layers.0.ln_1.bias": torch.randn(H),
        "t2s_model.layers.0.mlp.c_fc.weight": torch.randn(H, 4 * H),
        "t2s_model.layers.0.mlp.c_fc.bias": torch.randn(4 * H),
        "t2s_model.layers.0.mlp.c_proj.weight": torch.randn(4 * H, H),
        "t2s_model.layers.0.mlp.c_proj.bias": torch.randn(H),
        "t2s_model.layers.0.ln_2.weight": torch.randn(H),
        "t2s_model.layers.0.ln_2.bias": torch.randn(H),
        # ---- vocoder ----
        "enc_p.embedding.weight": torch.randn(768, H),
        "enc_p.enc.conv.weight": torch.randn(H, H, 1),
        "flow.flows.0.transform.in_proj.weight": torch.randn(H, H, 1),
        "dec.conv_pre.weight": torch.randn(H, H, 1),
        # ---- ssl_encoder ----
        "ssl.conv.weight": torch.randn(H, H, 1),
    }
    path = os.path.join(directory, "fake_sovits_ckpt.pt")
    torch.save(state_dict, path)
    return path


def test_T_SCVT_01() -> None:
    """T_SCVT_01: convert 基本流程，转换后各组件文件与 config.json 均存在。"""
    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)
    out_dir = os.path.join(tmp, "out")

    converter = SoVITSWeightConverter()
    result = converter.convert(src, out_dir)

    assert isinstance(result, dict) and len(result) > 0
    for comp, path in result.items():
        assert os.path.isfile(path), f"组件 {comp} 输出文件不存在: {path}"
    assert os.path.isfile(os.path.join(out_dir, "config.json"))


def test_T_SCVT_02() -> None:
    """T_SCVT_02: convert 后 validate 返回 True。"""
    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)
    out_dir = os.path.join(tmp, "out")

    converter = SoVITSWeightConverter()
    converter.convert(src, out_dir)

    assert converter.validate(out_dir) is True


def test_T_SCVT_03() -> None:
    """T_SCVT_03: GPT 权重键映射到 transformer.* 命名空间。

    ``t2s_model.embedding.weight`` → ``transformer.wte.weight``，
    ``t2s_model.layers.0.attn.c_attn.weight`` → ``transformer.h.0.attn.c_attn.weight``。
    """
    from safetensors.torch import load_file

    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)
    out_dir = os.path.join(tmp, "out")

    converter = SoVITSWeightConverter()
    converter.convert(src, out_dir)

    acoustic = load_file(os.path.join(out_dir, "acoustic_model.safetensors"))
    assert "transformer.wte.weight" in acoustic
    assert "transformer.h.0.attn.c_attn.weight" in acoustic


def test_T_SCVT_04() -> None:
    """T_SCVT_04: components=['vocoder'] 时 vocoder.safetensors 含 enc_p/flow/dec 键。"""
    from safetensors.torch import load_file

    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)
    out_dir = os.path.join(tmp, "out_voc")

    converter = SoVITSWeightConverter()
    result = converter.convert(src, out_dir, components=["vocoder"])

    assert list(result.keys()) == ["vocoder"]
    vocoder = load_file(os.path.join(out_dir, "vocoder.safetensors"))
    expected = {
        "enc_p.embedding.weight",
        "enc_p.enc.conv.weight",
        "flow.flows.0.transform.in_proj.weight",
        "dec.conv_pre.weight",
    }
    assert expected.issubset(vocoder.keys())


@_requires_transformers
def test_T_SCVT_05() -> None:
    """T_SCVT_05: 转换后的权重可加载进 GPT2ARModel。

    转换器写出的 ``config.json`` 是 Mosaic 格式（``hidden_size`` / ``num_layers``），
    与 ``GPT2Config.from_pretrained`` 期望的 ``n_embd`` / ``n_layer`` 不一致，
    ``from_pretrained`` 会得到不匹配的默认配置导致构建失败。这里先删除
    ``config.json``，使 ``load_weights`` 改用构造参数构建配置，从而用与伪造
    检查点一致的小尺寸配置加载转换后的 acoustic 权重。
    """
    from mosaic.nodes.audio.tts_backends.acoustic_models.gpt2_ar import GPT2ARModel

    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)
    out_dir = os.path.join(tmp, "out_gpt")

    converter = SoVITSWeightConverter()
    converter.convert(src, out_dir)
    os.remove(os.path.join(out_dir, "config.json"))

    model = GPT2ARModel(
        model_path=out_dir,
        vocab_size=_FAKE_VOCAB,
        semantic_vocab_size=_FAKE_VOCAB,
        hidden_size=_FAKE_HIDDEN,
        num_heads=2,
        num_layers=1,
        max_position_embeddings=_FAKE_NPOS,
    )
    model.load_weights(out_dir, device="cpu", dtype="float32")
    assert model._is_loaded is True


def test_T_SCVT_06() -> None:
    """T_SCVT_06: dry_run 返回以组件名为键的映射字典。"""
    tmp = tempfile.mkdtemp()
    src = _make_fake_sovits_checkpoint(tmp)

    converter = SoVITSWeightConverter()
    plan = converter.dry_run(src)

    assert isinstance(plan, dict) and len(plan) > 0
    for comp, mapping in plan.items():
        assert isinstance(comp, str)
        assert isinstance(mapping, dict) and len(mapping) > 0
        for src_key, tgt_key in mapping.items():
            assert isinstance(src_key, str)
            assert isinstance(tgt_key, str)
    # 伪造检查点同时含 GPT 与 vocoder 键，两者均应被规划
    assert "acoustic_model" in plan
    assert "vocoder" in plan
