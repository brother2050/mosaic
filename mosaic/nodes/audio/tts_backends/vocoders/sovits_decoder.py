# mosaic/nodes/audio/tts_backends/vocoders/sovits_decoder.py
"""GPT-SoVITS 声码器：SoVITS Decoder。

Layer 3: 声码器层。输入语义 token ids，输出音频波形。

GPT-SoVITS 的 SoVITS 部分将 GPT 生成的语义 token 解码为波形，核心流程::

    semantic_tokens → SemanticEncoder → PriorEncoder → NormalizingFlow.inverse
        → ConditionalHiFiGANDecoder → waveform

架构
----
::

    SemanticEncoder
        Embedding(ssl_vocab_size, hidden_size)
        + PositionalEncoding
        + TransformerEncoder(n_layers, n_heads, hidden_size)

    PriorEncoder
        Conv1D(hidden, hidden) → LayerNorm → ReLU
        → mu_proj(Conv1D(hidden, hidden))
        → logvar_proj(Conv1D(hidden, hidden))

    NormalizingFlow (n_layers_flow 个 FlowLayer)
        每个 FlowLayer: Affine Coupling Layer
        transform_net: WaveNet(dilated causal conv + FiLM conditioning)
        forward:  z = x2 * exp(s) + t,  log_det += sum(s)
        inverse:  x2 = (z2 - t) * exp(-s)

    ConditionalHiFiGANDecoder
        Conv1D(hidden, upsample_initial_channel)
        → UpsampleBlocks (with FiLM condition injection)
        → Conv1D(channels, 1) → tanh
        → waveform

数值稳定性
----------
* ``log_var`` 被 clamp 到 ``[-10, 10]``
* Flow 的 ``s`` 被 clamp 到 ``[-5, 5]``（先 sigmoid 再缩放）
* 采样时使用 ``torch.randn`` 并乘以 ``exp(0.5 * log_var)``

设计要点
--------
* ``torch`` / ``safetensors`` 采用惰性导入：模块顶层不导入这些重依赖，
  真正的 ``nn.Module`` 子类在首次 :meth:`load_weights` 时通过
  :func:`_get_sovits_decoder_class` 惰性构建。
* :class:`SoVITSDecoder` 继承 :class:`Vocoder` 抽象基类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现。
* :meth:`decode` / :meth:`decode_chunk` 返回 ``(waveform, sample_rate)``
  元组，兼容 :meth:`TTSBackend._coerce_vocoder_output`。
* ``features`` / ``semantic_tokens`` 等参数类型用 :data:`~typing.Any` 标注。
"""

from __future__ import annotations

import os
from typing import Any

from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder

__all__ = ["SoVITSDecoder"]


# 内部缓存的 nn.Module 子类（惰性创建）
_SoVITSImplClass: Any = None


def _get_sovits_decoder_class() -> Any:
    """惰性创建并返回 SoVITS Decoder 的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义所有内部子模块与主实现类，随后缓存到
    全局变量 :data:`_SoVITSImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_SoVITSDecoderImpl``。
    """
    global _SoVITSImplClass
    if _SoVITSImplClass is not None:
        return _SoVITSImplClass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # CausalConv1D：左填充因果卷积
    # ------------------------------------------------------------------
    class CausalConv1d(nn.Module):
        """因果一维卷积：仅在左侧填充，保证输出位置 t 只依赖输入 ≤ t。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        kernel_size : int
            卷积核大小。
        dilation : int
            膨胀率，默认 1。
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
        ) -> None:
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )

        def forward(self, x: Any) -> Any:
            """前向计算：左侧零填充后卷积。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, channels, seq_len]``。

            Returns
            -------
            torch.Tensor
                ``[batch, out_channels, seq_len]``。
            """
            x = F.pad(x, (self.pad, 0))
            return self.conv(x)

    # ------------------------------------------------------------------
    # WaveNet 风格的 transform_net（带 FiLM 条件注入）
    # ------------------------------------------------------------------
    class WaveNetTransform(nn.Module):
        """WaveNet 风格的膨胀因果卷积网络，用于 Flow 的 transform_net。

        每层：``CausalConv1D → FiLM(condition) → gated activation →
        1x1 Conv → 残差 + skip``

        Parameters
        ----------
        hidden_size : int
            隐藏维度。
        n_layers : int
            WaveNet 层数。
        kernel_size : int
            卷积核大小，默认 5。
        dilation_cycle : int
            膨胀率循环周期，默认 4。
        """

        def __init__(
            self,
            hidden_size: int,
            n_layers: int = 4,
            kernel_size: int = 5,
            dilation_cycle: int = 4,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.n_layers = n_layers

            # 输入投影
            self.in_proj = nn.Conv1d(hidden_size, hidden_size, 1)

            # 每层的卷积和门控
            self.conv_filters = nn.ModuleList()
            self.conv_gates = nn.ModuleList()
            self.cond_proj = nn.ModuleList()  # FiLM condition
            self.residual_proj = nn.ModuleList()
            self.skip_proj = nn.ModuleList()

            for i in range(n_layers):
                dilation = 2 ** (i % dilation_cycle)
                self.conv_filters.append(
                    CausalConv1d(hidden_size, hidden_size, kernel_size, dilation)
                )
                self.conv_gates.append(
                    CausalConv1d(hidden_size, hidden_size, kernel_size, dilation)
                )
                # FiLM: condition → (scale, shift)
                self.cond_proj.append(nn.Linear(hidden_size, hidden_size * 2))
                self.residual_proj.append(nn.Conv1d(hidden_size, hidden_size, 1))
                self.skip_proj.append(nn.Conv1d(hidden_size, hidden_size, 1))

            # 输出投影：hidden → 2*hidden (scale s 和 shift t)
            self.out_proj = nn.Conv1d(hidden_size, hidden_size * 2, 1)

        def forward(self, x: Any, condition: Any = None) -> tuple[Any, Any]:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, hidden, seq_len]``，输入的一半。
            condition : torch.Tensor | None
                ``[batch, hidden, seq_len]`` 或 ``[batch, hidden]``，
                FiLM 条件。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(scale, shift)``，各 ``[batch, hidden, seq_len]``。
            """
            h = self.in_proj(x)
            skip_sum = torch.zeros_like(h)

            for i in range(self.n_layers):
                # 门控激活
                hf = self.conv_filters[i](h)
                hg = self.conv_gates[i](h)

                # FiLM 条件注入
                if condition is not None:
                    cond = condition
                    if cond.dim() == 2:
                        cond = cond.unsqueeze(-1).expand(-1, -1, h.size(-1))
                    film = self.cond_proj[i](
                        cond.transpose(1, 2)
                    )  # [B, T, 2*H]
                    film = film.transpose(1, 2)  # [B, 2*H, T]
                    scale, shift = film.chunk(2, dim=1)
                    hf = hf * (1 + scale) + shift
                    hg = hg * (1 + scale) + shift

                # 门控：tanh(filter) * sigmoid(gate)
                act = torch.tanh(hf) * torch.sigmoid(hg)

                # 残差和 skip
                h = self.residual_proj[i](act) + h
                skip_sum = skip_sum + self.skip_proj[i](act)

            out = self.out_proj(F.relu(skip_sum))
            scale, shift = out.chunk(2, dim=1)
            return scale, shift

    # ------------------------------------------------------------------
    # FlowLayer：仿射耦合层
    # ------------------------------------------------------------------
    class FlowLayer(nn.Module):
        """仿射耦合层（Affine Coupling Layer）。

        将输入沿通道维度分成两半 ``x1, x2``，用 ``x1`` 经过
        :class:`WaveNetTransform` 计算 ``s, t``，然后：

        * forward:  ``z2 = x2 * exp(s) + t``,  ``log_det += sum(s)``
        * inverse: ``x2 = (z2 - t) * exp(-s)``

        ``s`` 经 sigmoid + 缩放约束到 ``[-5, 5]``。

        Parameters
        ----------
        hidden_size : int
            隐藏维度（须为偶数）。
        n_wavenet_layers : int
            WaveNet 层数。
        """

        def __init__(
            self, hidden_size: int, n_wavenet_layers: int = 4
        ) -> None:
            super().__init__()
            assert hidden_size % 2 == 0, "hidden_size must be even for FlowLayer"
            self.half = hidden_size // 2
            self.transform = WaveNetTransform(
                self.half, n_layers=n_wavenet_layers
            )

        def forward(
            self, x: Any, condition: Any = None
        ) -> tuple[Any, Any]:
            """正向：x → z，返回 z 和 log_det。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, hidden, seq_len]``。
            condition : torch.Tensor | None
                FiLM 条件。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(z, log_det)``。
            """
            x1, x2 = x[:, : self.half], x[:, self.half :]
            scale, shift = self.transform(x1, condition)
            # 约束 s 到 [-5, 5]
            scale = torch.sigmoid(scale) * 10.0 - 5.0
            z2 = x2 * torch.exp(scale) + shift
            z = torch.cat([x1, z2], dim=1)
            log_det = torch.sum(scale, dim=(1, 2))
            return z, log_det

        def inverse(
            self, z: Any, condition: Any = None
        ) -> Any:
            """逆向：z → x，返回 x。

            Parameters
            ----------
            z : torch.Tensor
                ``[batch, hidden, seq_len]``。
            condition : torch.Tensor | None
                FiLM 条件。

            Returns
            -------
            torch.Tensor
                ``[batch, hidden, seq_len]``。
            """
            z1, z2 = z[:, : self.half], z[:, self.half :]
            scale, shift = self.transform(z1, condition)
            scale = torch.sigmoid(scale) * 10.0 - 5.0
            x2 = (z2 - shift) * torch.exp(-scale)
            x = torch.cat([z1, x2], dim=1)
            return x

    # ------------------------------------------------------------------
    # NormalizingFlow：多个 FlowLayer 堆叠
    # ------------------------------------------------------------------
    class NormalizingFlow(nn.Module):
        """Normalizing Flow：多个 :class:`FlowLayer` 串行堆叠。

        提供 ``forward()``（训练时计算 log-likelihood）和 ``inverse()``
        （推理时从先验采样到数据空间）。

        Parameters
        ----------
        hidden_size : int
            隐藏维度。
        n_layers : int
            Flow 层数。
        n_wavenet_layers : int
            每个 FlowLayer 内 WaveNet 的层数。
        """

        def __init__(
            self,
            hidden_size: int,
            n_layers: int = 4,
            n_wavenet_layers: int = 4,
        ) -> None:
            super().__init__()
            self.flows = nn.ModuleList(
                [
                    FlowLayer(hidden_size, n_wavenet_layers)
                    for _ in range(n_layers)
                ]
            )

        def forward(
            self, x: Any, condition: Any = None
        ) -> tuple[Any, Any]:
            """正向：x → z_p，返回 z_p 和总 log_det。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, hidden, seq_len]``（后验空间）。
            condition : torch.Tensor | None
                FiLM 条件。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(z_p, total_log_det)``。
            """
            total_log_det = torch.zeros(
                x.size(0), device=x.device, dtype=x.dtype
            )
            for flow in self.flows:
                x, log_det = flow.forward(x, condition)
                total_log_det = total_log_det + log_det
            return x, total_log_det

        def inverse(
            self, z: Any, condition: Any = None
        ) -> Any:
            """逆向：z_p → x（从先验空间到数据空间）。

            逆序遍历所有 FlowLayer。

            Parameters
            ----------
            z : torch.Tensor
                ``[batch, hidden, seq_len]``（先验空间）。
            condition : torch.Tensor | None
                FiLM 条件。

            Returns
            -------
            torch.Tensor
                ``[batch, hidden, seq_len]``（后验空间）。
            """
            for flow in reversed(self.flows):
                z = flow.inverse(z, condition)
            return z

    # ------------------------------------------------------------------
    # SemanticEncoder
    # ------------------------------------------------------------------
    class SemanticEncoder(nn.Module):
        """语义编码器：将语义 token ids 编码为连续特征。

        结构：``Embedding → + PositionalEncoding → TransformerEncoder``

        Parameters
        ----------
        vocab_size : int
            语义 token 词表大小（SSL 码本大小）。
        hidden_size : int
            隐藏维度。
        n_layers : int
            Transformer 层数。
        n_heads : int
            注意力头数。
        max_len : int
            最大位置编码长度。
        """

        def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layers: int = 6,
            n_heads: int = 8,
            max_len: int = 2048,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size)

            # 正弦位置编码
            pe = torch.zeros(max_len, hidden_size)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, hidden_size, 2).float()
                * (-torch.log(torch.tensor(10000.0)) / hidden_size)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

        def forward(self, token_ids: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            token_ids : torch.Tensor
                ``[batch, seq_len]``，语义 token ids。

            Returns
            -------
            torch.Tensor
                ``[batch, seq_len, hidden_size]``。
            """
            x = self.embedding(token_ids)
            x = x + self.pe[:, : x.size(1)]
            x = self.transformer(x)
            return x

    # ------------------------------------------------------------------
    # PriorEncoder
    # ------------------------------------------------------------------
    class PriorEncoder(nn.Module):
        """先验编码器：从语义特征生成 mu 和 log_var。

        结构：``Conv1D → LayerNorm → ReLU → (mu_proj, logvar_proj)``

        Parameters
        ----------
        hidden_size : int
            隐藏维度。
        """

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(hidden_size, hidden_size, 1)
            self.norm = nn.LayerNorm(hidden_size)
            self.mu_proj = nn.Conv1d(hidden_size, hidden_size, 1)
            self.logvar_proj = nn.Conv1d(hidden_size, hidden_size, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, seq_len, hidden_size]``（SemanticEncoder 输出）。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(mu, log_var)``，各 ``[batch, hidden_size, seq_len]``。
            """
            # [B, T, H] → [B, H, T]
            x = x.transpose(1, 2)
            x = self.conv(x)
            # LayerNorm 期望最后一维是特征维
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = F.relu(x)
            x = x.transpose(1, 2)  # [B, H, T]

            mu = self.mu_proj(x)
            log_var = self.logvar_proj(x)
            # 数值稳定性：clamp log_var
            log_var = torch.clamp(log_var, -10.0, 10.0)
            return mu, log_var

        @staticmethod
        def reparameterize(mu: Any, log_var: Any) -> Any:
            """重参数化采样：``z = mu + exp(0.5 * log_var) * noise``。

            Parameters
            ----------
            mu : torch.Tensor
                均值 ``[batch, hidden, seq_len]``。
            log_var : torch.Tensor
                对数方差 ``[batch, hidden, seq_len]``。

            Returns
            -------
            torch.Tensor
                采样结果 ``[batch, hidden, seq_len]``。
            """
            std = torch.exp(0.5 * log_var)
            noise = torch.randn_like(std)
            return mu + std * noise

    # ------------------------------------------------------------------
    # ConditionalHiFiGANDecoder（带 FiLM 条件注入的 HiFi-GAN）
    # ------------------------------------------------------------------
    class _CondResBlock(nn.Module):
        """带 FiLM 条件注入的 HiFi-GAN 残差块。"""

        def __init__(
            self,
            channels: int,
            kernel_size: int,
            dilation_rates: list[int],
            cond_dim: int,
        ) -> None:
            super().__init__()
            self.convs1 = nn.ModuleList()
            self.convs2 = nn.ModuleList()
            self.cond_projs = nn.ModuleList()
            for _ in dilation_rates:
                self.convs1.append(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=dilation_rates[0],
                        padding=(kernel_size - 1) * dilation_rates[0] // 2,
                    )
                )
                self.convs2.append(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=1, padding=(kernel_size - 1) // 2,
                    )
                )
                self.cond_projs.append(nn.Linear(cond_dim, channels * 2))

        def forward(self, x: Any, cond: Any = None) -> Any:
            for c1, c2, cp in zip(self.convs1, self.convs2, self.cond_projs):
                xt = F.leaky_relu(x, 0.1)
                xt = c1(xt)
                if cond is not None:
                    film = cp(cond)  # [B, 2*C]
                    scale, shift = film.chunk(2, dim=-1)
                    scale = scale.unsqueeze(-1)
                    shift = shift.unsqueeze(-1)
                    xt = xt * (1 + scale) + shift
                xt = F.leaky_relu(xt, 0.1)
                xt = c2(xt)
                x = xt + x
            return x

    class ConditionalHiFiGANGenerator(nn.Module):
        """带 FiLM 条件注入的 HiFi-GAN Generator。

        与标准 HiFi-GAN 相比，每个上采样块额外接收条件向量，通过
        FiLM（scale + shift）调制中间特征。

        Parameters
        ----------
        hidden_size : int
            条件维度（latent 维度）。
        upsample_rates : list[int]
            上采样倍率。
        upsample_initial_channel : int
            初始通道数。
        resblock_kernel_sizes : list[int]
            残差块核大小。
        resblock_dilation_sizes : list[list[int]]
            残差块膨胀率。
        """

        def __init__(
            self,
            hidden_size: int = 192,
            upsample_rates: list[int] | None = None,
            upsample_initial_channel: int = 512,
            resblock_kernel_sizes: list[int] | None = None,
            resblock_dilation_sizes: list[list[int]] | None = None,
        ) -> None:
            super().__init__()
            if upsample_rates is None:
                upsample_rates = [8, 8, 2, 2]
            if resblock_kernel_sizes is None:
                resblock_kernel_sizes = [3, 7, 11]
            if resblock_dilation_sizes is None:
                resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

            self.hidden_size = hidden_size
            self.upsample_rates = list(upsample_rates)
            self.num_resblocks = len(resblock_kernel_sizes)

            self.conv_pre = nn.Conv1d(
                hidden_size, upsample_initial_channel, 7, 1, 3
            )

            self.ups = nn.ModuleList()
            self.resblocks = nn.ModuleList()
            self.cond_projs = nn.ModuleList()

            channels = upsample_initial_channel
            for upsample_rate in self.upsample_rates:
                self.ups.append(
                    nn.ConvTranspose1d(
                        channels, channels // 2,
                        kernel_size=upsample_rate * 2,
                        stride=upsample_rate,
                        padding=upsample_rate // 2,
                    )
                )
                # 条件投影：hidden_size → channels//2 * 2
                self.cond_projs.append(
                    nn.Linear(hidden_size, (channels // 2) * 2)
                )
                channels = channels // 2
                for k, d in zip(
                    resblock_kernel_sizes, resblock_dilation_sizes
                ):
                    self.resblocks.append(
                        _CondResBlock(channels, k, d, hidden_size)
                    )

            self.conv_post = nn.Conv1d(channels, 1, 7, 1, 3)

        def forward(self, x: Any, condition: Any = None) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, hidden, seq_len]``（latent）。
            condition : torch.Tensor | None
                ``[batch, hidden]``，全局条件向量。

            Returns
            -------
            torch.Tensor
                波形 ``[batch, 1, samples]``。
            """
            x = self.conv_pre(x)

            for i, up in enumerate(self.ups):
                # FiLM 条件注入
                cond = None
                if condition is not None:
                    film = self.cond_projs[i](condition)
                    scale, shift = film.chunk(2, dim=-1)
                    scale = scale.unsqueeze(-1)
                    shift = shift.unsqueeze(-1)
                    cond = condition  # 传给 resblock
                    x = x * (1 + scale) + shift

                x = F.leaky_relu(x, 0.1)
                x = up(x)

                xs: Any = None
                for j in range(self.num_resblocks):
                    rb = self.resblocks[i * self.num_resblocks + j]
                    out = rb(x, cond)
                    xs = out if xs is None else xs + out
                x = xs / self.num_resblocks

            x = F.leaky_relu(x, 0.1)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x

    # ------------------------------------------------------------------
    # _SoVITSDecoderImpl：主实现类
    # ------------------------------------------------------------------
    class _SoVITSDecoderImpl(nn.Module):
        """SoVITS Decoder 真实实现（``nn.Module`` 子类）。

        将语义 token ids 解码为音频波形。

        Parameters
        ----------
        ssl_vocab_size : int
            语义 token 词表大小。
        hidden_size : int
            隐藏维度。
        n_enc_layers : int
            SemanticEncoder 的 Transformer 层数。
        n_enc_heads : int
            SemanticEncoder 的注意力头数。
        n_flow_layers : int
            NormalizingFlow 的层数。
        n_wavenet_layers : int
            每个 FlowLayer 内 WaveNet 的层数。
        upsample_rates : list[int]
            HiFi-GAN 上采样倍率。
        upsample_initial_channel : int
            HiFi-GAN 初始通道数。
        """

        def __init__(
            self,
            ssl_vocab_size: int = 768,
            hidden_size: int = 192,
            n_enc_layers: int = 6,
            n_enc_heads: int = 8,
            n_flow_layers: int = 4,
            n_wavenet_layers: int = 4,
            upsample_rates: list[int] | None = None,
            upsample_initial_channel: int = 512,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size

            self.semantic_encoder = SemanticEncoder(
                vocab_size=ssl_vocab_size,
                hidden_size=hidden_size,
                n_layers=n_enc_layers,
                n_heads=n_enc_heads,
            )

            self.prior_encoder = PriorEncoder(hidden_size)

            self.flow = NormalizingFlow(
                hidden_size=hidden_size,
                n_layers=n_flow_layers,
                n_wavenet_layers=n_wavenet_layers,
            )

            self.decoder = ConditionalHiFiGANGenerator(
                hidden_size=hidden_size,
                upsample_rates=upsample_rates,
                upsample_initial_channel=upsample_initial_channel,
            )

            # 上采样总倍率
            rates = upsample_rates if upsample_rates else [8, 8, 2, 2]
            self.hop_length = 1
            for r in rates:
                self.hop_length *= r

        def forward(
            self,
            semantic_tokens: Any,
            ref_tokens: Any | None = None,
            ref_mel: Any | None = None,
        ) -> dict[str, Any]:
            """训练前向计算。

            Parameters
            ----------
            semantic_tokens : torch.Tensor
                ``[batch, seq_len]``，语义 token ids。
            ref_tokens : torch.Tensor | None
                参考语义 token ids。
            ref_mel : torch.Tensor | None
                参考音频的 mel 频谱（用于后验编码，简化版未实现）。

            Returns
            -------
            dict[str, torch.Tensor]
                包含 ``waveform``、``mu``、``log_var``、``z_p``、
                ``log_det`` 等。
            """
            # 1. 语义编码
            features = self.semantic_encoder(semantic_tokens)

            # 2. 先验编码
            mu, log_var = self.prior_encoder(features)

            # 3. 重参数化采样
            z = PriorEncoder.reparameterize(mu, log_var)

            # 4. Flow 正向：z → z_p
            z_p, log_det = self.flow.forward(z)

            # 5. 解码
            # condition: 使用 mu 的均值作为全局条件
            cond = mu.mean(dim=2)  # [B, H]
            waveform = self.decoder(z, cond)

            return {
                "waveform": waveform,
                "mu": mu,
                "log_var": log_var,
                "z": z,
                "z_p": z_p,
                "log_det": log_det,
            }

        @torch.no_grad()
        def infer(
            self,
            semantic_tokens: Any,
            ref_tokens: Any | None = None,
        ) -> Any:
            """推理前向：从语义 token 生成波形。

            流程：
            1. 语义编码 → 先验编码 → mu, log_var
            2. 如果有参考：用参考的先验作为条件
            3. 从先验采样 z_p
            4. Flow 逆向：z_p → z
            5. 解码：z → 波形

            Parameters
            ----------
            semantic_tokens : torch.Tensor
                ``[batch, seq_len]``。
            ref_tokens : torch.Tensor | None
                参考语义 token ids（用于语音克隆）。

            Returns
            -------
            torch.Tensor
                波形 ``[batch, 1, samples]``。
            """
            # 1. 语义编码
            features = self.semantic_encoder(semantic_tokens)

            # 2. 先验编码
            mu, log_var = self.prior_encoder(features)

            # 3. 语音克隆：使用参考 token 调整先验
            cond = mu.mean(dim=2)  # [B, H]
            if ref_tokens is not None:
                ref_features = self.semantic_encoder(ref_tokens)
                ref_mu, ref_log_var = self.prior_encoder(ref_features)
                # 用参考的先验均值作为额外条件
                ref_cond = ref_mu.mean(dim=2)  # [B, H]
                cond = (cond + ref_cond) / 2.0

            # 4. 从先验采样
            z_p = PriorEncoder.reparameterize(mu, log_var)

            # 5. Flow 逆向：z_p → z
            z = self.flow.inverse(z_p)

            # 6. 解码
            waveform = self.decoder(z, cond)
            return waveform

    _SoVITSImplClass = _SoVITSDecoderImpl
    return _SoVITSImplClass


class SoVITSDecoder(Vocoder):
    """GPT-SoVITS SoVITS 声码器。

    继承 :class:`Vocoder` 抽象基类，将语义 token ids 解码为音频波形。

    权重加载策略（:meth:`load_weights`）：

    1. 惰性创建 SoVITS Decoder ``nn.Module`` 子类实例；
    2. 从 safetensors / pytorch checkpoint 载入权重（``strict=False``）；
    3. 剥离 ``dec.`` / ``sovits.`` 等前缀；
    4. 移动到目标 device / dtype，切换为 eval。

    语音克隆通过 :meth:`set_reference` 设置参考语义 token，在
    :meth:`decode` 时自动注入条件。

    Parameters
    ----------
    model_path : str
        SoVITS 模型权重路径。
    ssl_vocab_size : int
        语义 token 词表大小（SSL 码本大小），默认 ``768``。
    hidden_size : int
        隐藏维度，默认 ``192``。
    sample_rate : int
        输出采样率，默认 ``32000``。
    n_enc_layers : int
        SemanticEncoder Transformer 层数。
    n_enc_heads : int
        SemanticEncoder 注意力头数。
    n_flow_layers : int
        NormalizingFlow 层数。
    n_wavenet_layers : int
        WaveNet 层数。
    upsample_rates : list[int] | None
        HiFi-GAN 上采样倍率。
    upsample_initial_channel : int
        HiFi-GAN 初始通道数。

    Attributes
    ----------
    vocoder_type : str
        固定为 ``"sovits_decoder"``。
    input_type : str
        固定为 ``"vq_tokens"``。
    """

    vocoder_type: str = "sovits_decoder"
    input_type: str = "vq_tokens"

    def __init__(
        self,
        model_path: str,
        ssl_vocab_size: int = 768,
        hidden_size: int = 192,
        sample_rate: int = 32000,
        n_enc_layers: int = 6,
        n_enc_heads: int = 8,
        n_flow_layers: int = 4,
        n_wavenet_layers: int = 4,
        upsample_rates: list[int] | None = None,
        upsample_initial_channel: int = 512,
    ) -> None:
        self.model_path = model_path
        self.ssl_vocab_size = ssl_vocab_size
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.n_enc_layers = n_enc_layers
        self.n_enc_heads = n_enc_heads
        self.n_flow_layers = n_flow_layers
        self.n_wavenet_layers = n_wavenet_layers
        self.upsample_rates: list[int] = (
            upsample_rates if upsample_rates is not None else [8, 8, 2, 2]
        )
        self.upsample_initial_channel = upsample_initial_channel

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

        # 参考信息（语音克隆）
        self._ref_tokens: Any = None

        # 流式重叠状态
        self._overlap_frames: int = 4
        self._token_buffer: Any = None

    # ------------------------------------------------------------------
    # 代理转发
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """将未在本类找到的属性转发给内部 ``nn.Module`` 实现。"""
        impl = self.__dict__.get("_impl")
        if impl is not None:
            return getattr(impl, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """转发调用，触发内部 ``nn.Module.__call__``。"""
        if self._impl is not None:
            return self._impl(*args, **kwargs)
        raise RuntimeError(
            "SoVITSDecoder is not loaded. Call load_weights() before "
            "calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载 SoVITS Decoder 权重。

        Parameters
        ----------
        weights_path : str
            权重文件路径或目录。
        device : str
            目标设备；无 GPU 时自动降级为 CPU。
        dtype : str
            数据精度，``"float16"`` / ``"float32"`` / ``"bfloat16"``。

        Raises
        ------
        ImportError
            ``torch`` 未安装。
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        resolved = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            resolved = "cpu"
        self._device = resolved
        self._dtype = dtype

        # 惰性创建实现实例
        cls = _get_sovits_decoder_class()
        impl = cls(
            ssl_vocab_size=self.ssl_vocab_size,
            hidden_size=self.hidden_size,
            n_enc_layers=self.n_enc_layers,
            n_enc_heads=self.n_enc_heads,
            n_flow_layers=self.n_flow_layers,
            n_wavenet_layers=self.n_wavenet_layers,
            upsample_rates=self.upsample_rates,
            upsample_initial_channel=self.upsample_initial_channel,
        )

        state_dict = self._load_state_dict(weights_path)
        if state_dict:
            state_dict = self._filter_and_strip(state_dict)
            impl.load_state_dict(state_dict, strict=False)

        impl = impl.to(device=resolved, dtype=torch_dtype)
        impl.eval()
        self._impl = impl
        self._is_loaded = True

    def unload_weights(self) -> None:
        """释放权重。"""
        try:
            import torch

            if self._impl is not None:
                try:
                    self._impl.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            from mosaic.core.device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass
        self._impl = None
        self._is_loaded = False
        self._ref_tokens = None
        self._token_buffer = None

    # ------------------------------------------------------------------
    # 语音克隆
    # ------------------------------------------------------------------
    def set_reference(
        self,
        ref_semantic_tokens: Any,
        ref_mel: Any | None = None,
    ) -> None:
        """设置参考语义 token，用于零样本语音克隆。

        Parameters
        ----------
        ref_semantic_tokens : torch.Tensor
            参考音频的语义 token ids，shape ``[1, ref_len]``。
        ref_mel : torch.Tensor | None
            参考音频的 mel 频谱（当前版本未使用，预留接口）。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SoVITSDecoder is not loaded. Call load_weights() "
                "before set_reference()."
            )
        import torch

        tokens = ref_semantic_tokens
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(self._device)
        self._ref_tokens = tokens

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(self, features: Any) -> tuple[Any, int]:
        """将语义 token ids 解码为波形。

        Parameters
        ----------
        features : torch.Tensor
            语义 token ids，shape ``[batch, seq_len]`` 或 ``[seq_len]``。

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(waveform, sample_rate)``，waveform 形状
            ``[batch, samples]`` 或 ``[samples]``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SoVITSDecoder is not loaded. Call load_weights() "
                "before decode()."
            )
        import torch

        tokens = features
        squeeze = False
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        tokens = tokens.to(self._device).long()

        with torch.no_grad():
            waveform = self._impl.infer(
                tokens, ref_tokens=self._ref_tokens
            )
        # [B, 1, samples] → [B, samples]
        waveform = waveform.squeeze(1)

        if squeeze:
            waveform = waveform.squeeze(0)
        return (waveform, self.sample_rate)

    def decode_chunk(self, features: Any) -> tuple[Any, int]:
        """流式解码：处理一小段语义 token，输出一小段波形。

        使用重叠 token 策略减少边界伪影。

        Parameters
        ----------
        features : torch.Tensor
            一小段语义 token ids。

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(waveform, sample_rate)``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SoVITSDecoder is not loaded. Call load_weights() "
                "before decode_chunk()."
            )
        import torch

        tokens = features
        squeeze = False
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        tokens = tokens.to(self._device).long()

        overlap = self._overlap_frames
        prepended = 0
        if self._token_buffer is not None and overlap > 0:
            # 拼接上一块的末尾作为重叠上下文
            buf = self._token_buffer
            if buf.size(-1) >= overlap:
                buf = buf[..., -overlap:]
            tokens = torch.cat([buf, tokens], dim=-1)
            prepended = buf.size(-1)

        with torch.no_grad():
            waveform = self._impl.infer(
                tokens, ref_tokens=self._ref_tokens
            )
        waveform = waveform.squeeze(1)

        # 跳过与重叠 token 对应的样本
        if prepended > 0:
            hop = self._impl.hop_length
            skip = prepended * hop
            if skip < waveform.shape[-1]:
                waveform = waveform[..., skip:]

        # 更新 token 缓冲区
        if overlap > 0 and tokens.size(-1) >= overlap:
            self._token_buffer = tokens[..., -overlap:].detach()

        if squeeze:
            waveform = waveform.squeeze(0)
        return (waveform, self.sample_rate)

    def reset_stream(self) -> None:
        """重置流式重叠缓冲区。"""
        self._token_buffer = None

    # ------------------------------------------------------------------
    # 训练前向（可选）
    # ------------------------------------------------------------------
    def forward(
        self,
        semantic_tokens: Any,
        ref_tokens: Any | None = None,
        ref_mel: Any | None = None,
    ) -> dict[str, Any]:
        """训练前向计算。

        Parameters
        ----------
        semantic_tokens : torch.Tensor
            ``[batch, seq_len]``。
        ref_tokens : torch.Tensor | None
            参考语义 token ids。
        ref_mel : torch.Tensor | None
            参考音频 mel 频谱。

        Returns
        -------
        dict[str, torch.Tensor]
            包含 waveform、mu、log_var、z、z_p、log_det 等。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SoVITSDecoder is not loaded. Call load_weights() "
                "before forward()."
            )
        return self._impl.forward(
            semantic_tokens, ref_tokens=ref_tokens, ref_mel=ref_mel
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _load_state_dict(weights_path: str) -> dict[str, Any]:
        """读取权重为 state_dict。"""
        import torch

        state_dict: dict[str, Any] = {}

        def _unwrap(ckpt: Any) -> dict[str, Any]:
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt and isinstance(
                    ckpt["state_dict"], dict
                ):
                    return ckpt["state_dict"]
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    return ckpt["model"]
                if "weight" in ckpt and isinstance(ckpt, dict):
                    return ckpt
            if isinstance(ckpt, dict):
                return ckpt
            raise ValueError(
                f"无法识别的 checkpoint 格式: {type(ckpt)}"
            )

        if os.path.isfile(weights_path):
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(weights_path)
            elif weights_path.endswith((".pt", ".pth", ".bin")):
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = _unwrap(ckpt)
        elif os.path.isdir(weights_path):
            for fname in (
                "sovits.safetensors",
                "vocoder.safetensors",
                "decoder.safetensors",
                "model.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in (
                    "sovits.pth",
                    "vocoder.bin",
                    "decoder.bin",
                    "model.bin",
                ):
                    fpath = os.path.join(weights_path, fname)
                    if os.path.isfile(fpath):
                        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                        state_dict = _unwrap(ckpt)
                        break
        return state_dict

    @staticmethod
    def _filter_and_strip(
        state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """过滤无关权重，剥离前缀。

        1. 丢弃以 ``discriminator.`` / ``disc.`` / ``mpd.`` / ``msd.``
           开头的 key；
        2. 剥离 ``dec.`` / ``sovits.`` / ``vits.`` 等前缀。
        """
        disc_prefixes = (
            "discriminator.", "disc.", "mpd.", "msd.",
        )
        gen_prefixes = ("dec.", "sovits.", "vits.", "decoder.")

        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            if any(key.startswith(p) for p in disc_prefixes):
                continue
            new_key = key
            for p in gen_prefixes:
                if key.startswith(p):
                    new_key = key[len(p):]
                    break
            filtered[new_key] = value
        return filtered
