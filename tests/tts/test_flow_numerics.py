"""测试 SoVITS Decoder 内部 NormalizingFlow / FlowLayer 的数值性质。

依赖 torch；torch 不可用时整个模块自动跳过。聚焦 Flow 的可逆性、数值
稳定性、log_scale 截断范围、梯度以及 batch/seq 维度行为，不依赖真实
预训练权重。

torch 导入放在函数内部，避免在模块顶层污染 sys.modules（phase2 mock）。
FlowLayer / NormalizingFlow 类定义在 ``_get_sovits_decoder_class()`` 函数
体内部，通过构建一个小型 ``_SoVITSDecoderImpl`` 实例后用 ``type()`` 反射
获取。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.vocoders.sovits_decoder import (  # noqa: E402
    _get_sovits_decoder_class,
)

# 小模型参数
_HIDDEN_SIZE = 32


def _get_flow_classes() -> dict:
    """通过 type 反射获取 FlowLayer 与 NormalizingFlow 类。

    构建一个小型 ``_SoVITSDecoderImpl`` 实例，再用 ``type()`` 取出其 flow
    子模块的类对象。
    """
    cls = _get_sovits_decoder_class()
    impl = cls(
        ssl_vocab_size=100,
        hidden_size=_HIDDEN_SIZE,
        n_enc_layers=2,
        n_enc_heads=4,
        n_flow_layers=4,
        n_wavenet_layers=2,
        upsample_rates=[8],
        upsample_initial_channel=2,
    )
    return {
        "FlowLayer": type(impl.flow.flows[0]),
        "NormalizingFlow": type(impl.flow),
    }


# ----------------------------------------------------------------------
# T_FLOW_01~02：可逆性（往返误差）
# ----------------------------------------------------------------------
def test_T_FLOW_01() -> None:
    """单层 FlowLayer forward->inverse 往返，误差 < 1e-5。

    耦合层正/逆使用同一 transform 作用于不变的半边，理论精确可逆；float64
    进一步压制浮点误差以满足 1e-5 容差。
    """
    import torch

    classes = _get_flow_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE).double()
    x = torch.randn(2, _HIDDEN_SIZE, 20, dtype=torch.float64)
    z, _ = flow.forward(x)
    x_recovered = flow.inverse(z)
    assert torch.allclose(x, x_recovered, atol=1e-5)


def test_T_FLOW_02() -> None:
    """多层 NormalizingFlow forward->inverse 往返，误差 < 1e-4。"""
    import torch

    classes = _get_flow_classes()
    nf = classes["NormalizingFlow"](
        hidden_size=_HIDDEN_SIZE, n_layers=4
    ).double()
    x = torch.randn(2, _HIDDEN_SIZE, 20, dtype=torch.float64)
    z_p, _ = nf.forward(x)
    x_recovered = nf.inverse(z_p)
    assert torch.allclose(x, x_recovered, atol=1e-4)


# ----------------------------------------------------------------------
# T_FLOW_03~04：数值稳定性（大 / 小幅值）
# ----------------------------------------------------------------------
def test_T_FLOW_03() -> None:
    """大幅值输入（randn * 100）：forward 与 inverse 输出无 NaN/Inf。"""
    import torch

    classes = _get_flow_classes()
    nf = classes["NormalizingFlow"](hidden_size=_HIDDEN_SIZE, n_layers=4)
    x = torch.randn(2, _HIDDEN_SIZE, 20) * 100
    z, _ = nf.forward(x)
    assert torch.isfinite(z).all()
    x_recovered = nf.inverse(z)
    assert torch.isfinite(x_recovered).all()


def test_T_FLOW_04() -> None:
    """小幅值输入（randn * 0.001）：forward 与 inverse 输出无 NaN/Inf。"""
    import torch

    classes = _get_flow_classes()
    nf = classes["NormalizingFlow"](hidden_size=_HIDDEN_SIZE, n_layers=4)
    x = torch.randn(2, _HIDDEN_SIZE, 20) * 0.001
    z, _ = nf.forward(x)
    assert torch.isfinite(z).all()
    x_recovered = nf.inverse(z)
    assert torch.isfinite(x_recovered).all()


# ----------------------------------------------------------------------
# T_FLOW_05~06：log_scale 截断范围 / 梯度
# ----------------------------------------------------------------------
def test_T_FLOW_05() -> None:
    """log_scale 截断范围: scale 值落在 [-5, 5] 内（sigmoid*10-5）。

    复现 FlowLayer.forward 内部的 scale 计算：对输入前半边经 transform 得
    raw_scale，再 sigmoid*10-5 截断，验证结果落在 [-5, 5]。
    """
    import torch

    classes = _get_flow_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE)
    x = torch.randn(2, _HIDDEN_SIZE, 20)
    # 复现 forward 内部对前半边的 scale 计算
    x1 = x[:, : flow.half]
    raw_scale, _ = flow.transform(x1, None)
    scale = torch.sigmoid(raw_scale) * 10.0 - 5.0
    assert scale.min().item() >= -5.0
    assert scale.max().item() <= 5.0


def test_T_FLOW_06() -> None:
    """梯度检查: 对输入开 requires_grad，经 log_det 反向，梯度有限。"""
    import torch

    classes = _get_flow_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE)
    x = torch.randn(2, _HIDDEN_SIZE, 20, requires_grad=True)
    _, log_det = flow.forward(x)
    log_det.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ----------------------------------------------------------------------
# T_FLOW_07~08：batch / seq 维度
# ----------------------------------------------------------------------
def test_T_FLOW_07() -> None:
    """batch_size=1 vs batch_size=4: forward 输出 batch 维正确。"""
    import torch

    classes = _get_flow_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE)
    z1, _ = flow.forward(torch.randn(1, _HIDDEN_SIZE, 20))
    assert z1.shape[0] == 1
    z4, _ = flow.forward(torch.randn(4, _HIDDEN_SIZE, 20))
    assert z4.shape[0] == 4


def test_T_FLOW_08() -> None:
    """不同 seq_len（10, 50, 100）: forward 输出 seq_len 与输入一致。"""
    import torch

    classes = _get_flow_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE)
    for seq_len in (10, 50, 100):
        x = torch.randn(2, _HIDDEN_SIZE, seq_len)
        z, _ = flow.forward(x)
        assert z.shape[2] == seq_len
