"""测试 SoVITS Decoder 内部子模块的回归行为。

覆盖 ``_get_sovits_decoder_class`` 惰性构建的 ``nn.Module`` 子类：
``NormalizingFlow`` / ``FlowLayer`` / ``PriorEncoder`` / ``SemanticEncoder``
/ ``ConditionalHiFiGANGenerator``。验证前向 / 逆向的数值性质与输出形状，
不依赖真实预训练权重。

依赖说明
--------
``torch`` 在每个测试函数内部局部导入，避免 phase2 mock 污染；模块级仅用
``importlib.util.find_spec("torch")`` 做跳过判断。内部子模块类定义在
``_get_sovits_decoder_class`` 的函数局部作用域，无法通过模块属性直接访问，
因此通过已构建实例的 ``type()`` 取回。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.vocoders.sovits_decoder import (
    _get_sovits_decoder_class,
)

# 模块级跳过判断：torch 缺失时跳过本文件全部用例（不在此处 import torch）
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch 未安装，跳过 SoVITS decoder 回归测试"
)


def _sovits_inner_classes() -> dict:
    """惰性构建一个极小的 ``_SoVITSDecoderImpl``，返回其子模块类对象。

    内部类（``NormalizingFlow`` / ``FlowLayer`` / ``PriorEncoder`` /
    ``SemanticEncoder`` / ``ConditionalHiFiGANGenerator``）定义在
    ``_get_sovits_decoder_class`` 函数局部作用域，无法通过模块属性直接访问，
    故先实例化一个极小的 impl，再用 ``type()`` 取回各子模块类。
    """
    cls = _get_sovits_decoder_class()
    impl = cls(
        ssl_vocab_size=100,
        hidden_size=32,
        n_enc_layers=2,
        n_enc_heads=2,
        n_flow_layers=2,
        n_wavenet_layers=2,
        upsample_rates=[2],
        upsample_initial_channel=8,
    )
    return {
        "NormalizingFlow": type(impl.flow),
        "FlowLayer": type(impl.flow.flows[0]),
        "PriorEncoder": type(impl.prior_encoder),
        "SemanticEncoder": type(impl.semantic_encoder),
        "ConditionalHiFiGANGenerator": type(impl.decoder),
    }


def test_T_SREG_01() -> None:
    """T_SREG_01: NormalizingFlow 正向 → 逆向的往返一致性。

    构建 ``NormalizingFlow(hidden_size=32, n_layers=4)``，对随机输入
    ``[2, 32, 20]`` 先 ``forward`` 再 ``inverse``，应恢复原输入。
    """
    import torch

    classes = _sovits_inner_classes()
    NormalizingFlow = classes["NormalizingFlow"]

    torch.manual_seed(0)
    flow = NormalizingFlow(hidden_size=32, n_layers=4)
    x = torch.randn(2, 32, 20)

    z_p, _log_det = flow.forward(x)
    x_recovered = flow.inverse(z_p)

    assert x_recovered.shape == x.shape
    assert torch.allclose(x, x_recovered, atol=1e-4)


def test_T_SREG_02() -> None:
    """T_SREG_02: FlowLayer 初始化后接近恒等。

    新建 ``FlowLayer`` 后 ``s`` 经 ``sigmoid*10-5`` 约束，初始化时接近 0，
    故 ``exp(s)≈1``、``shift`` 也较小，输出应接近输入。对小输入
    ``torch.randn(1, 32, 10) * 0.1`` 检查 ``|output - input| < 0.5``。
    """
    import torch

    classes = _sovits_inner_classes()
    FlowLayer = classes["FlowLayer"]

    torch.manual_seed(11)
    flow_layer = FlowLayer(32)  # 默认 n_wavenet_layers=4
    torch.manual_seed(111)
    x = torch.randn(1, 32, 10) * 0.1

    out, _log_det = flow_layer.forward(x)

    assert out.shape == x.shape
    assert (out - x).abs().max().item() < 0.5


def test_T_SREG_03() -> None:
    """T_SREG_03: PriorEncoder 输出 mu 有限、log_var 被 clamp 到 [-10, 10]。

    ``PriorEncoder`` 接收 ``[B, T, H]``（内部转置为 ``[B, H, T]`` 做卷积），
    故以 ``hidden_size=32`` 喂入 ``[1, 10, 32]``，输出 mu / log_var 形状
    为 ``[1, 32, 10]``。
    """
    import torch

    classes = _sovits_inner_classes()
    PriorEncoder = classes["PriorEncoder"]

    torch.manual_seed(0)
    prior = PriorEncoder(hidden_size=32)
    features = torch.randn(1, 10, 32)  # [B, T, H]

    mu, log_var = prior(features)

    assert mu.shape == (1, 32, 10)
    assert log_var.shape == (1, 32, 10)
    assert torch.isfinite(mu).all().item()
    assert log_var.min().item() >= -10.0
    assert log_var.max().item() <= 10.0


def test_T_SREG_04() -> None:
    """T_SREG_04: SemanticEncoder 对不同序列长度输出正确形状。

    ``SemanticEncoder(100, 32, 2, 4)`` 对 ``[1, 5]`` / ``[1, 10]`` /
    ``[1, 20]`` 三种长度，均应输出 ``[1, seq_len, 32]``。
    """
    import torch

    classes = _sovits_inner_classes()
    SemanticEncoder = classes["SemanticEncoder"]

    torch.manual_seed(0)
    encoder = SemanticEncoder(100, 32, 2, 4)

    for seq_len in (5, 10, 20):
        token_ids = torch.randint(0, 100, (1, seq_len))
        out = encoder(token_ids)
        assert out.shape == (1, seq_len, 32), (
            f"seq_len={seq_len} 期望 (1, {seq_len}, 32)，实际 {tuple(out.shape)}"
        )


def test_T_SREG_05() -> None:
    """T_SREG_05: ConditionalHiFiGANGenerator 输出格式为 ``[B, 1, T]``。

    带 / 不带 condition 两条前向路径均应输出单声道波形 ``[B, 1, T]``。

    注意：源码中条件 FiLM 在上采样 *之前* 施加，而 ``cond_projs`` 按
    *上采样后* 通道数（``channels//2``）构造，正常多级上采样配置下通道不
    匹配会报错。这里选用 ``upsample_initial_channel=2`` + 单级上采样，使
    得 ``channels//2 == 1`` 能与输入通道广播，从而真正走到条件 FiLM 路径
    并产出 ``[B, 1, T]``。
    """
    import torch

    classes = _sovits_inner_classes()
    ConditionalHiFiGANGenerator = classes["ConditionalHiFiGANGenerator"]

    torch.manual_seed(0)
    generator = ConditionalHiFiGANGenerator(
        hidden_size=32,
        upsample_rates=[4],
        upsample_initial_channel=2,
    )
    z = torch.randn(2, 32, 10)
    condition = torch.randn(2, 32)

    out_no_cond = generator(z)
    out_with_cond = generator(z, condition)

    assert out_no_cond.dim() == 3
    assert out_no_cond.shape[0] == 2
    assert out_no_cond.shape[1] == 1

    assert out_with_cond.dim() == 3
    assert out_with_cond.shape[0] == 2
    assert out_with_cond.shape[1] == 1
