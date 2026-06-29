"""测试 FlowEstimator 架构组件的单元行为。

聚焦 Flow Matching 速度场预测网络内部各子模块的形状契约与条件注入：
FlowEstimator 前向输入输出形状、SinusoidalPosEmb 时间步嵌入、
Self-Attention / Cross-Attention 运行、AdaptiveLayerNorm (FiLM) 条件注入、
以及时间步嵌入到序列维的广播。不依赖真实预训练权重（``load_weights``
传入空目录走随机初始化分支，仍完整覆盖 ``_impl`` 与 ``estimator`` 构建）。

torch 采用惰性探测：模块级仅用 ``importlib.util.find_spec`` 检查是否存在，
不在模块顶层导入，避免 phase2 mock 污染 ``sys.modules``。``torch`` 的实际
导入放在每个需要它的测试方法内部。``FlowMatchingModel`` 的模块导入不触发
torch 加载（torch 在 ``_get_flow_matching_class()`` 与各方法内部惰性导入），
因此可在模块顶层安全导入。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.acoustic_models.flow_matching import (  # noqa: E402
    FlowMatchingModel,
)

# ----------------------------------------------------------------------
# 小模型参数（不依赖真实权重，兼顾速度与覆盖）
# ----------------------------------------------------------------------
_IN_CHANNELS = 80
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_COND_DIM = 64


def _make_loaded_model(tmp_path) -> FlowMatchingModel:
    """构造并加载一个小参数 FlowMatchingModel（CPU / float32，随机初始化）。

    ``load_weights`` 传入空临时目录时无权重文件，走随机初始化分支，
    仍会创建内部 ``_impl``（``_FlowMatchingModelImpl`` nn.Module）并完成
    device / dtype 迁移与 ``eval()`` 置位。
    """
    model = FlowMatchingModel(
        model_path=str(tmp_path),
        in_channels=_IN_CHANNELS,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        condition_dim=_COND_DIM,
        num_ode_steps=10,
        ode_solver="euler",
    )
    model.load_weights(str(tmp_path), device="cpu", dtype="float32")
    return model


class TestFlowArchitecture:
    """FlowEstimator 架构组件单元测试（T_FARCH_01~06）。"""

    # ------------------------------------------------------------------
    # T_FARCH_01：FlowEstimator 输入输出形状正确
    # ------------------------------------------------------------------
    def test_T_FARCH_01(self, tmp_path) -> None:
        """FlowEstimator forward：z_t [1,80,50] -> velocity [1,80,50]。

        输入投影 ``mel_bins -> hidden``，经 Transformer 块后输出投影
        ``hidden -> mel_bins``，序列长度保持不变。验证速度场输出形状
        与输入 ``z_t`` 完全一致。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        estimator = model._impl.estimator

        z_t = torch.randn(1, _IN_CHANNELS, 50)
        t = torch.tensor(0.5)
        condition = torch.randn(1, 10, _COND_DIM)

        velocity = estimator(z_t, t, condition)

        assert velocity.shape == (1, _IN_CHANNELS, 50)
        assert torch.isfinite(velocity).all()

    # ------------------------------------------------------------------
    # T_FARCH_02：SinusoidalPosEmb 在 t in [0,1] 处输出正确
    # ------------------------------------------------------------------
    def test_T_FARCH_02(self, tmp_path) -> None:
        """SinusoidalPosEmb 在 t=0.0/0.5/1.0 处输出有限且维度正确。

        标量 ``t`` 产生 ``[dim]`` 嵌入，``dim == hidden_size``。验证时间步
        嵌入在 ODE 积分区间内各采样点均良定义。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        time_emb = model._impl.estimator.time_emb

        for t_val in (0.0, 0.5, 1.0):
            emb = time_emb(torch.tensor(t_val))
            assert torch.isfinite(emb).all()
            # 标量 t -> [dim] 嵌入
            assert emb.shape[-1] == _HIDDEN_SIZE

    # ------------------------------------------------------------------
    # T_FARCH_03：Self-Attention 层可运行
    # ------------------------------------------------------------------
    def test_T_FARCH_03(self, tmp_path) -> None:
        """TransformerBlock.self_attn 对 [1,10,64] 输入产出同形状输出。

        Self-Attention 的 embed_dim == hidden_size，``batch_first=True``，
        验证 ``MultiheadAttention`` 前向正常且输出序列维不变。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        block = model._impl.estimator.blocks[0]

        x = torch.randn(1, 10, _HIDDEN_SIZE)
        attn_out, _ = block.self_attn(x, x, x, need_weights=False)

        assert attn_out.shape == (1, 10, _HIDDEN_SIZE)
        assert torch.isfinite(attn_out).all()

    # ------------------------------------------------------------------
    # T_FARCH_04：Cross-Attention 层可运行
    # ------------------------------------------------------------------
    def test_T_FARCH_04(self, tmp_path) -> None:
        """TransformerBlock.cross_attn：query [1,10,64], kv [1,5,64] 运行正常。

        Cross-Attention 的 key/value 维度 ``kdim=vdim=cond_dim``，query 维度
        ``embed_dim=hidden_size``（此处二者相等），验证前向产出
        ``[1, 10, hidden_size]``。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        block = model._impl.estimator.blocks[0]

        query = torch.randn(1, 10, _HIDDEN_SIZE)
        key_value = torch.randn(1, 5, _COND_DIM)
        attn_out, _ = block.cross_attn(
            query, key_value, key_value, need_weights=False
        )

        assert attn_out.shape == (1, 10, _HIDDEN_SIZE)
        assert torch.isfinite(attn_out).all()

    # ------------------------------------------------------------------
    # T_FARCH_05：AdaptiveLayerNorm (FiLM) 条件注入
    # ------------------------------------------------------------------
    def test_T_FARCH_05(self, tmp_path) -> None:
        """norm1 (AdaptiveLayerNorm)：不同 cond 产生不同输出。

        FiLM 通过 ``cond`` 生成 scale/shift 对特征做 ``x*(1+scale)+shift``，
        验证输出形状 ``[1,10,hidden]`` 且不同条件向量 ``[1, cond_dim]``
        经投影后产生不同的仿射变换。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        norm1 = model._impl.estimator.blocks[0].norm1

        x = torch.randn(1, 10, _HIDDEN_SIZE)
        cond_a = torch.randn(1, _COND_DIM)
        # 明确不同的条件，保证 scale/shift 投影结果不同
        cond_b = cond_a + torch.ones(1, _COND_DIM)

        out_a = norm1(x, cond_a)
        out_b = norm1(x, cond_b)

        assert out_a.shape == (1, 10, _HIDDEN_SIZE)
        assert out_b.shape == (1, 10, _HIDDEN_SIZE)
        assert torch.isfinite(out_a).all()
        assert torch.isfinite(out_b).all()
        # 不同 cond -> 不同 scale/shift -> 不同输出
        assert not torch.equal(out_a, out_b)

    # ------------------------------------------------------------------
    # T_FARCH_06：时间步嵌入广播到序列维
    # ------------------------------------------------------------------
    def test_T_FARCH_06(self, tmp_path) -> None:
        """time_emb 输出可作为 AdaptiveLayerNorm 的 cond 条件化序列。

        ``t=[1]``（batch 维）经 ``time_emb`` 产生 ``[1, hidden]``，再作为
        ``cond`` 传给 AdaptiveLayerNorm（要求 ``[batch, cond_dim]``），
        通过 ``scale.unsqueeze(1)`` 广播到序列维。验证形状广播链路通畅。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        estimator = model._impl.estimator

        # batch 维时间步 -> [1, hidden]
        t = torch.tensor([0.5])
        t_emb = estimator.time_emb(t)
        assert t_emb.shape == (1, _HIDDEN_SIZE)

        # 用作 AdaptiveLayerNorm 的 cond（hidden == cond_dim 时维度匹配）
        norm1 = estimator.blocks[0].norm1
        x = torch.randn(1, 10, _HIDDEN_SIZE)
        out = norm1(x, t_emb)

        assert out.shape == (1, 10, _HIDDEN_SIZE)
        assert torch.isfinite(out).all()
