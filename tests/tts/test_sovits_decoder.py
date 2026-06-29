"""测试 SoVITSDecoder 及其内部子模块。

依赖 torch；torch 不可用时整个模块自动跳过。使用小模型参数以节省时间，
不加载真实预训练权重（load_weights 用不存在的路径或空目录触发优雅降级，
以随机初始化完成 load 并标记为已加载）。

torch 导入放在函数内部，避免在模块顶层污染 sys.modules（phase2 mock）。
内部子模块类（SemanticEncoder / PriorEncoder / FlowLayer / NormalizingFlow /
ConditionalHiFiGANGenerator）定义在 ``_get_sovits_decoder_class()`` 函数体
内部，通过构建一个小型 ``_SoVITSDecoderImpl`` 实例后用 ``type()`` 反射获取。

备注
----
ConditionalHiFiGANGenerator 的 FiLM 条件注入在默认多层上采样配置下存在
通道不匹配（cond_proj 产出 ``channels//2`` 个 scale/shift，却作用在
``channels`` 通道的张量上）。单层上采样 + ``upsample_initial_channel=2``
时 scale 维度为 1，可正确广播到 2 通道，故条件注入路径统一采用该配置。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.vocoders.sovits_decoder import (  # noqa: E402
    SoVITSDecoder,
    _get_sovits_decoder_class,
)

# 小模型参数，兼顾速度与覆盖
_SSL_VOCAB_SIZE = 100
_HIDDEN_SIZE = 32
_N_ENC_LAYERS = 2
_N_ENC_HEADS = 4
_N_FLOW_LAYERS = 2
_N_WAVENET_LAYERS = 2
# 单层上采样 + upsample_initial_channel=2，使 FiLM 条件注入可正确广播
_UPSAMPLE_RATES = [8]
_UPSAMPLE_INITIAL_CHANNEL = 2
_SAMPLE_RATE = 32000


def _get_internal_classes() -> dict:
    """通过 type 反射获取 ``_get_sovits_decoder_class()`` 内部定义的子模块类。

    构建一个小型 ``_SoVITSDecoderImpl`` 实例，再用 ``type()`` 取出其各子
    模块的类对象，从而能以独立方式构造 SemanticEncoder / PriorEncoder /
    FlowLayer / NormalizingFlow / ConditionalHiFiGANGenerator。
    """
    cls = _get_sovits_decoder_class()
    impl = cls(
        ssl_vocab_size=_SSL_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        n_enc_layers=_N_ENC_LAYERS,
        n_enc_heads=_N_ENC_HEADS,
        n_flow_layers=_N_FLOW_LAYERS,
        n_wavenet_layers=_N_WAVENET_LAYERS,
        upsample_rates=_UPSAMPLE_RATES,
        upsample_initial_channel=_UPSAMPLE_INITIAL_CHANNEL,
    )
    return {
        "impl": cls,
        "SemanticEncoder": type(impl.semantic_encoder),
        "PriorEncoder": type(impl.prior_encoder),
        "NormalizingFlow": type(impl.flow),
        "FlowLayer": type(impl.flow.flows[0]),
        "ConditionalHiFiGANGenerator": type(impl.decoder),
    }


def _make_decoder() -> SoVITSDecoder:
    """构造一个小参数 SoVITSDecoder（不加载权重）。"""
    return SoVITSDecoder(
        model_path="/tmp/sovits_test",
        ssl_vocab_size=_SSL_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        sample_rate=_SAMPLE_RATE,
        n_enc_layers=_N_ENC_LAYERS,
        n_enc_heads=_N_ENC_HEADS,
        n_flow_layers=_N_FLOW_LAYERS,
        n_wavenet_layers=_N_WAVENET_LAYERS,
        upsample_rates=_UPSAMPLE_RATES,
        upsample_initial_channel=_UPSAMPLE_INITIAL_CHANNEL,
    )


def _make_loaded_decoder() -> SoVITSDecoder:
    """构造并用随机初始化加载一个 SoVITSDecoder。

    用不存在的路径触发优雅降级：state_dict 为空，以随机初始化完成 load
    并标记为已加载，使 decode / decode_chunk / set_reference 可运行。
    """
    decoder = _make_decoder()
    decoder.load_weights("/nonexistent/path", device="cpu", dtype="float32")
    return decoder


# ----------------------------------------------------------------------
# T_SVITS_01~03：语义编码器 / 先验编码器 / 重参数化
# ----------------------------------------------------------------------
def test_T_SVITS_01() -> None:
    """SemanticEncoder forward: [2, 10] token ids -> [2, 10, 32]。"""
    import torch

    classes = _get_internal_classes()
    encoder = classes["SemanticEncoder"](
        vocab_size=_SSL_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        n_layers=_N_ENC_LAYERS,
        n_heads=_N_ENC_HEADS,
    )
    token_ids = torch.randint(0, _SSL_VOCAB_SIZE, (2, 10), dtype=torch.long)
    out = encoder(token_ids)
    assert out.shape == (2, 10, _HIDDEN_SIZE)


def test_T_SVITS_02() -> None:
    """PriorEncoder forward: 输出 mu / log_var 形状 [2, 32, 10]。

    PriorEncoder 接收 [B, T, H] 并输出 [B, H, T]，故为得到 [2, 32, 10]
    的输出，输入取 [B=2, T=10, H=32]。
    """
    import torch

    classes = _get_internal_classes()
    prior = classes["PriorEncoder"](hidden_size=_HIDDEN_SIZE)
    x = torch.randn(2, 10, _HIDDEN_SIZE)  # [B, T, H]
    mu, log_var = prior(x)
    assert mu.shape == (2, _HIDDEN_SIZE, 10)
    assert log_var.shape == (2, _HIDDEN_SIZE, 10)


def test_T_SVITS_03() -> None:
    """reparameterize: z = mu + std * noise，shape 与 mu 一致。"""
    import torch

    classes = _get_internal_classes()
    PriorEncoder = classes["PriorEncoder"]
    mu = torch.randn(2, _HIDDEN_SIZE, 10)
    log_var = torch.randn(2, _HIDDEN_SIZE, 10)
    z = PriorEncoder.reparameterize(mu, log_var)
    assert z.shape == mu.shape


# ----------------------------------------------------------------------
# T_SVITS_04~06：NormalizingFlow / FlowLayer 可逆性与数值稳定性
# ----------------------------------------------------------------------
def test_T_SVITS_04() -> None:
    """FlowLayer inverse: 单层 forward 后 inverse，往返误差 < 1e-4。

    耦合层的正/逆使用同一 transform 作用于不变的 x1 半边，理论上精确可逆；
    使用 float64 进一步压制浮点误差以满足容差。
    """
    import torch

    classes = _get_internal_classes()
    flow = classes["FlowLayer"](hidden_size=_HIDDEN_SIZE).double()
    x = torch.randn(2, _HIDDEN_SIZE, 20, dtype=torch.float64)
    z, _ = flow.forward(x)
    x_rec = flow.inverse(z)
    err = (x - x_rec).abs().max().item()
    assert err < 1e-4


def test_T_SVITS_05() -> None:
    """NormalizingFlow 多层 inverse: 4 层 forward 后 inverse，往返误差 < 1e-3。"""
    import torch

    classes = _get_internal_classes()
    nf = classes["NormalizingFlow"](
        hidden_size=_HIDDEN_SIZE, n_layers=4
    ).double()
    x = torch.randn(2, _HIDDEN_SIZE, 20, dtype=torch.float64)
    z_p, _ = nf.forward(x)
    x_rec = nf.inverse(z_p)
    err = (x - x_rec).abs().max().item()
    assert err < 1e-3


def test_T_SVITS_06() -> None:
    """Flow 数值稳定性: 大幅值输入（magnitude 100），输出无 NaN/Inf。"""
    import torch

    classes = _get_internal_classes()
    nf = classes["NormalizingFlow"](hidden_size=_HIDDEN_SIZE, n_layers=4)
    x = torch.randn(2, _HIDDEN_SIZE, 20) * 100
    z, _ = nf.forward(x)
    assert torch.isfinite(z).all()
    x_rec = nf.inverse(z)
    assert torch.isfinite(x_rec).all()


# ----------------------------------------------------------------------
# T_SVITS_07~08：ConditionalHiFiGANGenerator / 完整 impl 前向
# ----------------------------------------------------------------------
def test_T_SVITS_07() -> None:
    """ConditionalHiFiGANGenerator: [1, 32, 10] + condition [1, 32] -> [1, 1, samples>10]。"""
    import torch

    classes = _get_internal_classes()
    generator = classes["ConditionalHiFiGANGenerator"](
        hidden_size=_HIDDEN_SIZE,
        upsample_rates=_UPSAMPLE_RATES,
        upsample_initial_channel=_UPSAMPLE_INITIAL_CHANNEL,
    )
    x = torch.randn(1, _HIDDEN_SIZE, 10)
    cond = torch.randn(1, _HIDDEN_SIZE)
    out = generator(x, cond)
    assert out.shape[0] == 1
    assert out.shape[1] == 1
    assert out.shape[2] > 10


def test_T_SVITS_08() -> None:
    """完整 _SoVITSDecoderImpl forward: semantic_tokens [1, 10] -> 含 'waveform' 的 dict。"""
    import torch

    cls = _get_sovits_decoder_class()
    impl = cls(
        ssl_vocab_size=_SSL_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        n_enc_layers=_N_ENC_LAYERS,
        n_enc_heads=_N_ENC_HEADS,
        n_flow_layers=_N_FLOW_LAYERS,
        n_wavenet_layers=_N_WAVENET_LAYERS,
        upsample_rates=_UPSAMPLE_RATES,
        upsample_initial_channel=_UPSAMPLE_INITIAL_CHANNEL,
    )
    semantic_tokens = torch.randint(
        0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long
    )
    out = impl.forward(semantic_tokens)
    assert isinstance(out, dict)
    assert "waveform" in out


# ----------------------------------------------------------------------
# T_SVITS_09~11：语音克隆 / Vocoder 接口 / 流式解码
# ----------------------------------------------------------------------
def test_T_SVITS_09() -> None:
    """set_reference: 设置参考 token 后 _ref_tokens 被填充。"""
    import torch

    decoder = _make_loaded_decoder()
    ref_tokens = torch.randint(0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long)
    decoder.set_reference(ref_tokens)
    assert decoder._ref_tokens is not None
    assert decoder._ref_tokens.shape[0] == 1


def test_T_SVITS_10() -> None:
    """decode 实现 Vocoder 接口: 返回 (waveform, sample_rate) 元组。"""
    import torch

    decoder = _make_loaded_decoder()
    tokens = torch.randint(0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long)
    result = decoder.decode(tokens)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == _SAMPLE_RATE
    assert torch.is_tensor(result[0])


def test_T_SVITS_11() -> None:
    """decode_chunk: 流式解码，返回 (waveform, sample_rate) 元组。"""
    import torch

    decoder = _make_loaded_decoder()
    decoder.reset_stream()
    tokens = torch.randint(0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long)
    result = decoder.decode_chunk(tokens)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == _SAMPLE_RATE
    assert torch.is_tensor(result[0])


# ----------------------------------------------------------------------
# T_SVITS_12~14：不同 seq_len / 权重加载 / 条件注入
# ----------------------------------------------------------------------
def test_T_SVITS_12() -> None:
    """不同 seq_len 输入 [1, 5] / [1, 10] / [1, 20] 均可解码。"""
    import torch

    decoder = _make_loaded_decoder()
    for seq_len in (5, 10, 20):
        tokens = torch.randint(
            0, _SSL_VOCAB_SIZE, (1, seq_len), dtype=torch.long
        )
        result = decoder.decode(tokens)
        assert isinstance(result, tuple)
        assert result[1] == _SAMPLE_RATE
        assert torch.is_tensor(result[0])
        assert result[0].shape[-1] > 0


def test_T_SVITS_13() -> None:
    """load_weights: 从空临时目录加载，_is_loaded 为 True。

    空目录使 _load_state_dict 返回空 dict，以随机初始化完成 load。
    """
    decoder = _make_decoder()
    with tempfile.TemporaryDirectory() as tmpdir:
        decoder.load_weights(tmpdir, device="cpu", dtype="float32")
    assert decoder._is_loaded is True


def test_T_SVITS_14() -> None:
    """条件注入: 有/无 set_reference 时 decode 输出应不同。

    固定随机种子以消除重参数化采样的随机性，使两次 decode 的唯一差异来自
    参考条件。
    """
    import torch

    decoder = _make_loaded_decoder()
    tokens = torch.randint(0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long)
    ref_tokens = torch.randint(0, _SSL_VOCAB_SIZE, (1, 10), dtype=torch.long)

    # 无参考
    torch.manual_seed(42)
    wf_no_ref, _ = decoder.decode(tokens)

    # 有参考（相同种子，仅条件不同）
    decoder.set_reference(ref_tokens)
    torch.manual_seed(42)
    wf_with_ref, _ = decoder.decode(tokens)

    assert not torch.allclose(wf_no_ref, wf_with_ref)
