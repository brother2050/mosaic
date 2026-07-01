# mosaic/nodes/audio/tts_backends/acoustic_models/flow_matching.py
"""Flow Matching 声学模型（CosyVoice 核心）。

Layer 2: 声学模型层。输入文本特征 + 条件信息，输出 mel spectrogram。

本模块实现 CosyVoice 的条件流匹配（Conditional Flow Matching）声学模型，
通过 ODE 求解从高斯噪声一次性生成完整 mel 频谱，而非逐 token 自回归。

Flow Matching 数学原理
-----------------------
训练时学习速度场 ``v(z_t, t, condition)``，使得：

* ``t=0`` 时 ``z_0`` 是数据分布（mel spectrogram）
* ``t=1`` 时 ``z_1`` 是标准高斯分布 ``N(0, I)``

推理时从 ``z_1`` 出发，沿速度场积分回到 ``z_0``：

::

    z_1 ~ N(0, I)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        v = FlowEstimator(z_t, t, condition)
        z_t = z_t - v * dt    (Euler 方法)
    mel = z_0

CosyVoice 2 使用 rectified flow 训练策略，使 ODE 轨迹接近直线，
因此只需 10 步左右即可获得高质量结果。

ODE 步数与质量/速度权衡
------------------------
+---------------+---------+--------+---------------------------+
| num_ode_steps | 延迟    | 质量   | 适用场景                  |
+===============+=========+========+===========================+
| 5             | ~50ms   | 中等   | 实时对话、低延迟场景      |
| 10            | ~100ms  | 好     | 推荐默认值，质量/速度均衡 |
| 20            | ~200ms  | 最高   | 离线合成、高质量要求      |
| 50            | ~500ms  | 极高   | 研究对比、极限质量        |
+---------------+---------+--------+---------------------------+

显存需求
--------
* ``float16`` 精度：约 2-4 GB GPU 显存
* ``float32`` 精度：约 4-8 GB GPU 显存

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 采用惰性导入。
* :class:`FlowMatchingModel` 直接继承 :class:`AcousticModel`（不继承
  :class:`LlamaARModelBase`），因为 Flow Matching 不是自回归。
* ``generate`` 返回 mel spectrogram（而非 token ids），与 AR 模型不同。
* ``generate_stream`` 实现了 Chunk-aware ODE 求解策略。
* FlowEstimator 使用 Transformer + Adaptive LayerNorm (FiLM) 条件化。
* 数值稳定性：velocity 被 clamp 到 ``[-10, 10]``，防止 ODE 发散。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel

logger = logging.getLogger(__name__)

__all__ = ["FlowMatchingModel"]


# 内部缓存的 nn.Module 子类（惰性创建）
_FlowMatchingImplClass: Any = None


def _get_flow_matching_class() -> Any:
    """惰性创建并返回 Flow Matching 的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义所有内部子模块与主实现类，随后缓存到
    全局变量 :data:`_FlowMatchingImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_FlowMatchingModelImpl``。
    """
    global _FlowMatchingImplClass
    if _FlowMatchingImplClass is not None:
        return _FlowMatchingImplClass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # SinusoidalPosEmb：正弦位置编码（用于时间步嵌入）
    # ------------------------------------------------------------------
    class SinusoidalPosEmb(nn.Module):
        """正弦位置编码，用于时间步 ``t`` 的嵌入。

        Parameters
        ----------
        dim : int
            嵌入维度。
        """

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, t: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            t : torch.Tensor
                时间步标量或张量，形状 ``[batch]`` 或标量。

            Returns
            -------
            torch.Tensor
                时间步嵌入 ``[batch, dim]`` 或 ``[dim]``。
            """
            device = t.device if hasattr(t, "device") else "cpu"
            half = self.dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half - 1)
            emb = torch.exp(torch.arange(half, device=device) * -emb)
            if t.dim() == 0:
                emb = emb * t
            else:
                emb = emb * t.unsqueeze(-1)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1))
            return emb

    # ------------------------------------------------------------------
    # AdaptiveLayerNorm：FiLM 条件化
    # ------------------------------------------------------------------
    class AdaptiveLayerNorm(nn.Module):
        """Adaptive Layer Normalization (FiLM)。

        通过时间步嵌入生成 scale (γ) 和 shift (β)，对特征做仿射变换：
        ``x = x * (1 + γ(t)) + β(t)``

        Parameters
        ----------
        hidden_size : int
            特征维度。
        cond_dim : int
            条件维度（时间步嵌入维度）。
        """

        def __init__(self, hidden_size: int, cond_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.proj = nn.Linear(cond_dim, hidden_size * 2)

        def forward(self, x: Any, cond: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, seq_len, hidden]``。
            cond : torch.Tensor
                ``[batch, cond_dim]``。

            Returns
            -------
            torch.Tensor
                条件化后的特征 ``[batch, seq_len, hidden]``。
            """
            h = self.norm(x)
            scale_shift = self.proj(cond)  # [B, 2*H]
            scale, shift = scale_shift.chunk(2, dim=-1)
            return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    # ------------------------------------------------------------------
    # TransformerBlock：自注意力 + 交叉注意力 + FFN + AdaLN
    # ------------------------------------------------------------------
    class TransformerBlock(nn.Module):
        """Flow Estimator 的 Transformer 块。

        包含：Self-Attention → Cross-Attention（关注条件）→ FFN，
        每层使用 AdaptiveLayerNorm 做时间步条件化。

        Parameters
        ----------
        hidden_size : int
            隐藏维度。
        num_heads : int
            注意力头数。
        cond_dim : int
            条件维度。
        ff_mult : int
            FFN 扩展倍率，默认 4。
        """

        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            cond_dim: int,
            ff_mult: int = 4,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads

            # Self-Attention
            self.norm1 = AdaptiveLayerNorm(hidden_size, cond_dim)
            self.self_attn = nn.MultiheadAttention(
                hidden_size, num_heads, batch_first=True
            )

            # Cross-Attention（关注条件）
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads, kdim=cond_dim, vdim=cond_dim,
                batch_first=True,
            )

            # FFN
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * ff_mult),
                nn.GELU(),
                nn.Linear(hidden_size * ff_mult, hidden_size),
            )

        def forward(
            self, x: Any, cond_embed: Any, cond_kv: Any
        ) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, seq_len, hidden]``。
            cond_embed : torch.Tensor
                时间步嵌入 ``[batch, cond_dim]``，用于 AdaLN。
            cond_kv : torch.Tensor
                交叉注意力的 key/value ``[batch, cond_len, cond_dim]``。

            Returns
            -------
            torch.Tensor
                ``[batch, seq_len, hidden]``。
            """
            # Self-Attention with AdaLN
            h = self.norm1(x, cond_embed)
            attn_out, _ = self.self_attn(h, h, h, need_weights=False)
            x = x + attn_out

            # Cross-Attention
            h = self.norm2(x)
            cross_out, _ = self.cross_attn(
                h, cond_kv, cond_kv, need_weights=False
            )
            x = x + cross_out

            # FFN
            h = self.norm3(x)
            ff_out = self.ff(h)
            x = x + ff_out

            return x

    # ------------------------------------------------------------------
    # FlowEstimator：预测速度场 v(z_t, t, condition)
    # ------------------------------------------------------------------
    class FlowEstimator(nn.Module):
        """速度场预测网络。

        输入当前 mel 状态 ``z_t``、时间步 ``t`` 和条件特征 ``condition``，
        输出速度场 ``v``，用于 ODE 求解。

        结构：输入投影 → 时间步嵌入 → 条件投影 → Transformer 块 → 输出投影

        Parameters
        ----------
        mel_bins : int
            mel 维度（输入/输出通道数）。
        hidden_size : int
            隐藏维度。
        num_layers : int
            Transformer 层数。
        num_heads : int
            注意力头数。
        cond_dim : int
            条件维度。
        """

        # 数值稳定性常量：velocity 截断范围，防止 ODE 发散（见 E2-1）。
        VELOCITY_CLAMP_MIN: float = -10.0
        VELOCITY_CLAMP_MAX: float = 10.0

        def __init__(
            self,
            mel_bins: int = 80,
            hidden_size: int = 512,
            num_layers: int = 8,
            num_heads: int = 8,
            cond_dim: int = 512,
        ) -> None:
            super().__init__()
            self.mel_bins = mel_bins
            self.hidden_size = hidden_size

            # 输入投影：mel_bins → hidden_size
            self.in_proj = nn.Conv1d(mel_bins, hidden_size, 1)

            # 时间步嵌入
            self.time_emb = SinusoidalPosEmb(hidden_size)
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )

            # 条件投影
            self.cond_proj = nn.Linear(cond_dim, hidden_size)

            # Transformer 块
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, cond_dim)
                for _ in range(num_layers)
            ])

            # 最终 LayerNorm + 输出投影
            self.final_norm = nn.LayerNorm(hidden_size)
            self.out_proj = nn.Conv1d(hidden_size, mel_bins, 1)

            # 缩放因子：t 接近 1 时缩小 velocity（接近噪声时变化慢）
            self.scale_proj = nn.Linear(hidden_size, 1)

        def forward(
            self,
            z_t: Any,
            t: Any,
            condition: Any,
        ) -> Any:
            """前向计算：预测速度场。

            Parameters
            ----------
            z_t : torch.Tensor
                当前 mel 状态 ``[batch, mel_bins, seq_len]``。
            t : torch.Tensor
                时间步标量或 ``[batch]`` 张量，范围 ``[0, 1]``。
            condition : torch.Tensor
                条件特征 ``[batch, cond_dim, cond_len]`` 或
                ``[batch, cond_len, cond_dim]``。

            Returns
            -------
            torch.Tensor
                速度场 ``[batch, mel_bins, seq_len]``。
            """
            B, _, T = z_t.shape

            # 1. 输入投影
            h = self.in_proj(z_t)  # [B, H, T]
            h = h.transpose(1, 2)  # [B, T, H]

            # 2. 时间步嵌入
            if t.dim() == 0:
                t_batched = t.unsqueeze(0).expand(B)
            else:
                t_batched = t
            t_emb = self.time_emb(t_batched)  # [B, H]
            t_emb = self.time_mlp(t_emb)       # [B, H]

            # 3. 条件投影
            cond = condition
            if cond.dim() == 3 and cond.shape[1] != cond.shape[2]:
                # [B, cond_dim, cond_len] → [B, cond_len, cond_dim]
                if cond.shape[1] == self.cond_proj.in_features:
                    cond = cond.transpose(1, 2)
            elif cond.dim() == 3 and cond.shape[1] == cond.shape[2]:
                # 模糊情况：假设最后一维是 cond_dim
                pass
            cond_proj = self.cond_proj(cond)  # [B, cond_len, H]

            # 4. Transformer 块
            for block in self.blocks:
                h = block(h, t_emb, cond)

            # 5. 输出投影
            h = self.final_norm(h)
            velocity = self.out_proj(h.transpose(1, 2))  # [B, mel_bins, T]

            # 6. 缩放因子
            scale = self.scale_proj(t_emb.mean(dim=0, keepdim=True))
            scale = torch.sigmoid(scale).squeeze()
            if scale.dim() == 0:
                velocity = velocity * scale
            else:
                velocity = velocity * scale.view(-1, 1, 1)

            # 数值稳定性：clamp velocity，防止 ODE 发散
            velocity = torch.clamp(
                velocity, self.VELOCITY_CLAMP_MIN, self.VELOCITY_CLAMP_MAX
            )

            return velocity

    # ------------------------------------------------------------------
    # _FlowMatchingModelImpl：主实现类
    # ------------------------------------------------------------------
    class _FlowMatchingModelImpl(nn.Module):
        """Flow Matching 模型真实实现（``nn.Module`` 子类）。

        将 FlowEstimator 与条件融合逻辑封装在一起，提供完整的前向计算
        （训练）和 ODE 求解（推理）能力。

        Parameters
        ----------
        mel_bins : int
            mel 维度。
        hidden_size : int
            隐藏维度。
        num_layers : int
            Transformer 层数。
        num_heads : int
            注意力头数。
        cond_dim : int
            条件维度。
        """

        def __init__(
            self,
            mel_bins: int = 80,
            hidden_size: int = 512,
            num_layers: int = 8,
            num_heads: int = 8,
            cond_dim: int = 512,
        ) -> None:
            super().__init__()
            self.mel_bins = mel_bins
            self.hidden_size = hidden_size

            self.estimator = FlowEstimator(
                mel_bins=mel_bins,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                cond_dim=cond_dim,
            )

            # 条件融合投影
            self.text_proj = nn.Linear(cond_dim, cond_dim)
            self.ref_proj = nn.Linear(cond_dim, cond_dim)
            self.spk_proj = nn.Linear(cond_dim, cond_dim)

        def fuse_condition(
            self,
            text_feats: Any,
            ref_feats: Any | None = None,
            speaker_embedding: Any | None = None,
        ) -> Any:
            """融合多路条件特征。

            将文本特征、参考音频特征和说话人嵌入融合为统一的条件向量。

            Parameters
            ----------
            text_feats : torch.Tensor
                文本特征 ``[batch, text_len, cond_dim]``。
            ref_feats : torch.Tensor | None
                参考音频特征 ``[batch, ref_len, cond_dim]``。
            speaker_embedding : torch.Tensor | None
                说话人嵌入 ``[batch, spk_dim]``。

            Returns
            -------
            torch.Tensor
                融合后的条件 ``[batch, total_len, cond_dim]``。
            """
            parts: list[Any] = []

            # 文本特征
            t = self.text_proj(text_feats)
            parts.append(t)

            # 参考音频特征
            if ref_feats is not None:
                r = self.ref_proj(ref_feats)
                parts.append(r)

            # 说话人嵌入（扩展为序列）
            if speaker_embedding is not None:
                s = self.spk_proj(speaker_embedding)
                # 扩展到序列长度 1
                if s.dim() == 2:
                    s = s.unsqueeze(1)  # [B, 1, D]
                parts.append(s)

            return torch.cat(parts, dim=1)  # [B, total_len, D]

        def velocity_fn(
            self,
            z_t: Any,
            t: float,
            condition: Any,
        ) -> Any:
            """速度场函数（供 ODE 求解器调用）。

            Parameters
            ----------
            z_t : torch.Tensor
                当前状态 ``[batch, mel_bins, seq_len]``。
            t : float
                当前时间步 ``[0, 1]``。
            condition : torch.Tensor
                条件特征。

            Returns
            -------
            torch.Tensor
                速度场 ``[batch, mel_bins, seq_len]``。
            """
            import torch

            t_tensor = torch.tensor(t, device=z_t.device, dtype=z_t.dtype)
            return self.estimator(z_t, t_tensor, condition)

        @torch.no_grad()
        def solve_ode(
            self,
            condition: Any,
            target_len: int,
            num_steps: int = 10,
            solver: str = "euler",
        ) -> Any:
            """ODE 求解：从高斯噪声生成 mel。

            Parameters
            ----------
            condition : torch.Tensor
                条件特征 ``[batch, total_len, cond_dim]``。
            target_len : int
                目标 mel 帧数。
            num_steps : int
                ODE 求解步数。
            solver : str
                求解器：``"euler"`` / ``"midpoint"`` / ``"rk4"``。

            Returns
            -------
            torch.Tensor
                生成的 mel ``[batch, mel_bins, target_len]``。
            """
            B = condition.shape[0]
            device = condition.device
            dtype = condition.dtype

            # 初始化 z_1 ~ N(0, I)
            z = torch.randn(
                B, self.mel_bins, target_len, device=device, dtype=dtype
            )

            dt = 1.0 / num_steps

            for i in range(num_steps):
                t = 1.0 - i * dt

                if solver == "euler":
                    v = self.velocity_fn(z, t, condition)
                    z = z - v * dt

                elif solver == "midpoint":
                    v1 = self.velocity_fn(z, t, condition)
                    z_mid = z - v1 * (dt / 2)
                    v2 = self.velocity_fn(z_mid, t - dt / 2, condition)
                    z = z - v2 * dt

                elif solver == "rk4":
                    k1 = self.velocity_fn(z, t, condition)
                    k2 = self.velocity_fn(
                        z - k1 * dt / 2, t - dt / 2, condition
                    )
                    k3 = self.velocity_fn(
                        z - k2 * dt / 2, t - dt / 2, condition
                    )
                    k4 = self.velocity_fn(z - k3 * dt, t - dt, condition)
                    z = z - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                else:
                    raise ValueError(
                        f"Unknown solver: {solver!r}. "
                        f"Supported: 'euler', 'midpoint', 'rk4'."
                    )

            return z  # z_0 = mel

        @torch.no_grad()
        def solve_ode_stream(
            self,
            condition: Any,
            total_len: int,
            chunk_frames: int = 150,
            overlap_frames: int = 15,
            num_steps: int = 10,
            solver: str = "euler",
        ) -> Iterator[Any]:
            """Chunk-aware ODE 求解（流式）。

            将目标 mel 分为多个 chunk，每个 chunk 独立做 ODE 求解，
            chunk 边界做 overlap-add 平滑。

            Parameters
            ----------
            condition : torch.Tensor
                全局条件特征。
            total_len : int
                目标 mel 总帧数。
            chunk_frames : int
                每个 chunk 的帧数（约 150 帧 ≈ 1.5 秒）。
            overlap_frames : int
                chunk 间重叠帧数。
            num_steps : int
                ODE 求解步数。
            solver : str
                ODE 求解器。

            Yields
            ------
            torch.Tensor
                每个 chunk 的 mel ``[batch, mel_bins, chunk_frames]``。
            """
            B = condition.shape[0]
            device = condition.device
            dtype = condition.dtype

            step = chunk_frames - overlap_frames
            num_chunks = max(1, (total_len + step - 1) // step)

            prev_tail: Any = None

            for chunk_idx in range(num_chunks):
                start = chunk_idx * step
                end = min(start + chunk_frames, total_len)
                actual_len = end - start

                # 初始化独立噪声
                z = torch.randn(
                    B, self.mel_bins, actual_len,
                    device=device, dtype=dtype,
                )

                # ODE 求解
                dt = 1.0 / num_steps
                for i in range(num_steps):
                    t = 1.0 - i * dt
                    v = self.velocity_fn(z, t, condition)
                    z = z - v * dt

                mel_chunk = z  # [B, mel_bins, actual_len]

                # Overlap-add 平滑
                if prev_tail is not None and overlap_frames > 0:
                    ov = min(overlap_frames, mel_chunk.shape[-1])
                    if ov > 0 and prev_tail.shape[-1] >= ov:
                        fade_in = torch.linspace(
                            0.0, 1.0, ov, device=device, dtype=dtype
                        )
                        fade_out = 1.0 - fade_in
                        mel_chunk[..., :ov] = (
                            fade_out * prev_tail[..., -ov:]
                            + fade_in * mel_chunk[..., :ov]
                        )

                # 保存尾部用于下一次平滑
                if overlap_frames > 0 and mel_chunk.shape[-1] >= overlap_frames:
                    prev_tail = mel_chunk[..., -overlap_frames:].clone()
                elif mel_chunk.shape[-1] > 0:
                    prev_tail = mel_chunk[..., -1:].clone()

                yield mel_chunk

        def forward(
            self,
            mel: Any,
            t: Any,
            condition: Any,
        ) -> Any:
            """训练前向计算：预测速度场。

            Parameters
            ----------
            mel : torch.Tensor
                mel spectrogram ``[batch, mel_bins, seq_len]``。
            t : torch.Tensor
                时间步 ``[batch]`` 或标量。
            condition : torch.Tensor
                条件特征。

            Returns
            -------
            torch.Tensor
                预测的速度场。
            """
            return self.estimator(mel, t, condition)

    _FlowMatchingImplClass = _FlowMatchingModelImpl
    return _FlowMatchingImplClass


class FlowMatchingModel(AcousticModel):
    """CosyVoice Flow Matching 声学模型。

    继承 :class:`AcousticModel`（不继承 :class:`LlamaARModelBase`），
    通过 ODE 求解从高斯噪声一次性生成完整 mel spectrogram。

    与自回归模型（ChatTTS / Fish / GPT-SoVITS）的根本区别：

    * **非自回归**：不逐 token 生成，而是一次性生成完整 mel。
    * **ODE 求解**：从高斯噪声出发，沿速度场积分到数据空间。
    * **输出**：返回 mel spectrogram（而非 token ids）。
    * **流式策略**：使用 Chunk-aware ODE 求解（非逐 token 流式）。

    Parameters
    ----------
    model_path : str
        Flow Matching 模型权重路径。
    llm_model_path : str | None
        LLM 模型路径（如果文本理解部分独立加载）。
    in_channels : int
        mel 频谱维度，默认 ``80``。
    hidden_size : int
        Flow 网络隐藏维度，默认 ``512``。
    num_layers : int
        Flow Transformer 层数，默认 ``8``。
    num_heads : int
        注意力头数，默认 ``8``。
    condition_dim : int
        条件特征维度，默认 ``512``。
    num_ode_steps : int
        ODE 求解步数，默认 ``10``。
    ode_solver : str
        ODE 求解器，``"euler"`` / ``"midpoint"`` / ``"rk4"``。
    target_length_seconds : float
        最大生成时长（秒），默认 ``30.0``。

    Attributes
    ----------
    model_type : str
        固定为 ``"flow_matching"``。
    """

    model_type: str = "flow_matching"

    # Mel 帧率（CosyVoice 默认 86.13 fps），用于由目标时长换算 mel 帧数
    # （见 E2-1：原为 __init__ 内魔法数字 86.13）。
    MEL_FPS: float = 86.13

    def __init__(
        self,
        model_path: str,
        llm_model_path: str | None = None,
        in_channels: int = 80,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        condition_dim: int = 512,
        num_ode_steps: int = 10,
        ode_solver: str = "euler",
        target_length_seconds: float = 30.0,
    ) -> None:
        self._model_path: str = model_path
        self._llm_model_path: str | None = llm_model_path
        self._in_channels: int = in_channels
        self._hidden_size: int = hidden_size
        self._num_layers: int = num_layers
        self._num_heads: int = num_heads
        self._condition_dim: int = condition_dim
        self._num_ode_steps: int = num_ode_steps
        self._ode_solver: str = ode_solver
        self._target_length_seconds: float = target_length_seconds

        # 模型实例（load_weights 后填充）
        self._impl: Any = None
        self._llm: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

        # mel 帧率（帧/秒）：hop_length / sample_rate
        # CosyVoice 默认 24000Hz, hop=256 → ~94 fps
        self._mel_fps: float = self.MEL_FPS

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
            "FlowMatchingModel is not loaded. Call load_weights() "
            "before calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载 Flow Matching 模型权重。

        Parameters
        ----------
        weights_path : str
            权重文件路径或目录。
        device : str
            目标设备；无 GPU 时自动降级为 CPU。
        dtype : str
            数据精度。

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
        cls = _get_flow_matching_class()
        impl = cls(
            mel_bins=self._in_channels,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            cond_dim=self._condition_dim,
        )

        state_dict = self._load_state_dict(weights_path)
        if state_dict:
            # strict=False 以兼容不同 checkpoint（缺失/多余键），但需报告
            # 不匹配情况，避免静默加载错误权重（见 E4-1）。
            result = impl.load_state_dict(state_dict, strict=False)
            try:
                missing = list(result.missing_keys)
                unexpected = list(result.unexpected_keys)
            except (AttributeError, TypeError):
                # result 不是标准 _IncompatibleKeys（如 mock 环境），跳过报告
                missing, unexpected = [], []
            if missing or unexpected:
                logger.warning(
                    "FlowMatching model loaded with non-strict matching: "
                    "missing=%d keys, unexpected=%d keys",
                    len(missing),
                    len(unexpected),
                )

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
            if self._llm is not None:
                try:
                    if hasattr(self._llm, "to"):
                        self._llm.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            from mosaic.core._device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass
        self._impl = None
        self._llm = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs: Any,
    ) -> Any:
        """生成 mel spectrogram。

        与 AR 模型不同，Flow Matching 一次性生成完整 mel，不逐 token 产出。

        Parameters
        ----------
        token_ids : Any
            文本特征或 token ids。对于 CosyVoice，这里传入的是
            ``text_features``（LLM 编码后的文本隐藏状态）或 token ids。
        speaker_embedding : Any | None
            说话人信息 dict，可包含：
            * ``ref_speech_tokens``：参考音频语音 token
            * ``speaker_embedding``：说话人嵌入向量
            或直接是特征张量。
        max_new_tokens : int
            在 Flow Matching 中作为目标 mel 帧数的提示（非严格限制）。
        temperature : float
            噪声缩放因子（影响生成多样性）。
        top_p : float
            未使用（保留接口兼容性）。
        top_k : int
            未使用（保留接口兼容性）。
        **kwargs : Any
            额外参数：``num_ode_steps``、``ode_solver``、
            ``target_length``、``condition`` 等。

        Returns
        -------
        torch.Tensor
            mel spectrogram ``[batch, mel_bins, gen_len]``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "FlowMatchingModel is not loaded. Call load_weights() "
                "before generate()."
            )
        import torch

        # 解析条件特征
        condition = self._prepare_condition(
            token_ids, speaker_embedding, kwargs
        )

        # 计算目标长度
        target_len = kwargs.get("target_length")
        if target_len is None:
            target_len = int(max_new_tokens)
        if target_len <= 0:
            target_len = int(self._target_length_seconds * self._mel_fps)

        # ODE 步数
        num_steps = kwargs.get("num_ode_steps", self._num_ode_steps)
        solver = kwargs.get("ode_solver", self._ode_solver)

        # ODE 求解
        mel = self._impl.solve_ode(
            condition=condition,
            target_len=target_len,
            num_steps=num_steps,
            solver=solver,
        )

        # temperature 缩放
        if temperature != 1.0:
            mel = mel * temperature

        return mel

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 24,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Chunk-aware 流式生成。

        将目标 mel 分为多个 chunk，每个 chunk 独立做 ODE 求解。
        这是 **分块一次性生成**，不是逐 token 流式。

        Parameters
        ----------
        token_ids : Any
            文本特征或 token ids。
        speaker_embedding : Any | None
            说话人信息。
        stream_batch : int
            在 Flow Matching 中作为 chunk 帧数的提示。
        **kwargs : Any
            额外参数。

        Yields
        ------
        torch.Tensor
            每个 chunk 的 mel ``[batch, mel_bins, chunk_frames]``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "FlowMatchingModel is not loaded. Call load_weights() "
                "before generate_stream()."
            )

        # Flow Matching 支持流式（覆写基类的 NotImplementedError）
        condition = self._prepare_condition(
            token_ids, speaker_embedding, kwargs
        )

        target_len = kwargs.get("target_length")
        if target_len is None:
            target_len = int(self._target_length_seconds * self._mel_fps)

        chunk_frames = kwargs.get("chunk_size_frames", 150)
        overlap_frames = kwargs.get("overlap_frames", 15)
        num_steps = kwargs.get("num_ode_steps", self._num_ode_steps)
        solver = kwargs.get("ode_solver", self._ode_solver)

        yield from self._impl.solve_ode_stream(
            condition=condition,
            total_len=target_len,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            num_steps=num_steps,
            solver=solver,
        )

    def get_input_embeddings(self) -> Any:
        """返回输入投影层。"""
        if self._impl is not None:
            return self._impl.estimator.in_proj
        return None

    def get_output_head(self) -> Any:
        """返回输出投影层。"""
        if self._impl is not None:
            return self._impl.estimator.out_proj
        return None

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _prepare_condition(
        self,
        token_ids: Any,
        speaker_embedding: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """准备条件特征。

        如果 kwargs 中有 ``condition``，直接使用；
        否则从 token_ids 和 speaker_embedding 构造。

        Parameters
        ----------
        token_ids : Any
            文本特征或 token ids。
        speaker_embedding : Any | None
            说话人信息。
        kwargs : dict
            额外参数。

        Returns
        -------
        torch.Tensor
            条件特征 ``[batch, cond_len, cond_dim]``。
        """
        import torch
        import torch.nn.functional as F

        # 如果直接传了 condition
        if "condition" in kwargs:
            cond = kwargs["condition"]
            if cond.dim() == 2:
                cond = cond.unsqueeze(0)
            return cond

        # 从 token_ids 构造文本特征
        text_feats = token_ids
        if hasattr(text_feats, "dim"):
            if text_feats.dim() == 1:
                text_feats = text_feats.unsqueeze(0)
            if text_feats.dtype in (
                torch.long, torch.int, torch.int32, torch.int64,
            ):
                # token ids → embedding
                if self._impl is not None:
                    embed = self._impl.estimator.cond_proj
                    text_feats = embed(
                        F.one_hot(
                            text_feats.long(),
                            num_classes=self._condition_dim,
                        ).float()
                    )
                else:
                    text_feats = torch.randn(
                        text_feats.shape[0], text_feats.shape[1],
                        self._condition_dim,
                        device=text_feats.device,
                    )
            elif text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(0)

        # 从 speaker_embedding 提取
        ref_feats = None
        spk_emb = None
        if speaker_embedding is not None:
            if isinstance(speaker_embedding, dict):
                ref_feats = speaker_embedding.get("ref_speech_tokens")
                spk_emb = speaker_embedding.get("speaker_embedding")
            elif hasattr(speaker_embedding, "dim"):
                spk_emb = speaker_embedding

        # 融合条件
        if self._impl is not None:
            condition = self._impl.fuse_condition(
                text_feats, ref_feats, spk_emb
            )
        else:
            condition = text_feats

        return condition

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
                "flow_matching.safetensors",
                "acoustic_model.safetensors",
                "model.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in (
                    "flow_matching.pth",
                    "acoustic_model.bin",
                    "model.bin",
                ):
                    fpath = os.path.join(weights_path, fname)
                    if os.path.isfile(fpath):
                        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                        state_dict = _unwrap(ckpt)
                        break
        return state_dict
