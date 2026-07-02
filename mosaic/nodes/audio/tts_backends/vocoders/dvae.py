# mosaic/nodes/audio/tts_backends/vocoders/dvae.py
"""ChatTTS DVAE 解码器。

Layer 3 前置步骤：将 VQ 音频码 token 解码为 mel 频谱。

本模块实现 ChatTTS 离散变分自编码器（Discrete VAE, DVAE）的**解码部分**，
结构与官方 ``ChatTTS/model/dvae.py`` 完全一致，``state_dict`` key 与官方
``DVAE.safetensors`` 权重文件一一对应，可直接通过 ``load_state_dict`` 加载，
无需前缀剥离，也无需依赖官方 ``ChatTTS`` 包。

架构（与官方一致）
------------------
顶层模块（对应官方 ``DVAE``）包含：

* ``coef`` —— buffer，形状 ``[1, mel_bins, 1]``，对输出 mel 做逐通道缩放；
* ``downsample_conv`` —— 仅在含 encoder 时存在，
  ``Conv1d -> GELU -> Conv1d(stride=2) -> GELU``，对输入 mel 下采样；
* ``encoder`` —— ``DVAEDecoder`` 子模块（encode 路径，decode 时不使用）；
* ``decoder`` —— ``DVAEDecoder`` 子模块（decode 主干）；
* ``out_conv`` —— ``Conv1d(dim, mel_bins, 3, 1, 1, bias=False)``；
* ``vq_layer`` —— ``GFSQ``，自实现 ``GroupedResidualFSQ``（不依赖
  ``vector_quantize_pytorch``），将 token ids 嵌入为连续特征。

``DVAEDecoder`` 子模块结构：

* ``conv_in`` —— ``Sequential(Conv1d, GELU, Conv1d)``；
* ``decoder_block`` —— ``ModuleList`` of ``ConvNeXtBlock``；
* ``conv_out`` —— ``Conv1d(hidden, odim, 1, bias=False)``。

``ConvNeXtBlock`` 结构（注意 layer scale 参数名为 ``weight``，非 ``gamma``）：

* ``dwconv`` —— Depthwise ``Conv1d``（膨胀，``groups=dim``）；
* ``norm`` —— ``LayerNorm``；
* ``pwconv1`` —— ``Linear(dim, dim*4)``；
* ``act`` —— ``GELU``；
* ``pwconv2`` —— ``Linear(dim*4, dim)``；
* ``weight`` —— layer scale ``Parameter``，形状 ``[dim]``。

decode 前向逻辑（与官方一致）
-----------------------------
``token_ids[B, num_vq, T]`` -> ``GFSQ._embed`` -> ``[B, 2*dim, T]`` ->
reshape 拆成 2 组、时间维度翻倍 -> ``[B, dim, T*2]`` -> ``decoder`` ->
``out_conv`` -> ``× coef`` -> ``[B, mel_bins, T*2]``。

设计要点
--------
* ``torch`` / ``safetensors`` / ``vector_quantize_pytorch`` 采用惰性导入，
  真正的 ``nn.Module`` 子类在首次实例化时通过 :func:`_get_dvae_class`
  惰性构建，使本模块在未安装 ``torch`` 时仍可被导入。
* :class:`DVAEDecoder` 是代理类，将属性访问与方法调用转发给内部
  ``nn.Module`` 实例，从而既保持 ``nn.Module`` 一致的行为，又避免在模块
  顶层硬依赖 ``torch``。
* 权重加载直接 ``load_state_dict(strict=False)``，无需前缀剥离或官方包。
* ``token_ids`` / ``mel`` 等参数类型用 :data:`~typing.Any` 标注。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["DVAEDecoder"]


# 内部缓存的 nn.Module 子类（惰性创建）
_DVAEDecoderClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_dvae_class() -> Any:
    """惰性创建并返回与官方 ChatTTS DVAE 结构完全一致的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义 :class:`_ConvNeXtBlock`、
    :class:`_DVAEDecoderModule`、:class:`_GFSQ` 与顶层 :class:`_DVAEDecoder`
    （对应官方 ``DVAE``），随后缓存到全局变量 :data:`_DVAEDecoderClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_DVAEDecoder``，其 ``state_dict`` key 与官方
        ``DVAE.safetensors`` 完全一致。
    """
    global _DVAEDecoderClass
    if _DVAEDecoderClass is None:
        import math
        import torch
        import torch.nn as nn

        class _ConvNeXtBlock(nn.Module):
            """官方 ChatTTS ConvNeXt 块。

            ``dwconv -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> *weight
            -> + residual``。注意 layer scale 参数名为 ``weight``（与官方
            一致，非 ``gamma``）。
            """

            def __init__(
                self,
                dim: int,
                intermediate_dim: int,
                kernel: int,
                dilation: int,
                layer_scale_init_value: float = 1e-6,
            ) -> None:
                super().__init__()
                # Depthwise 膨胀卷积（same padding，保持帧长不变）
                self.dwconv = nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=kernel,
                    padding=dilation * (kernel // 2),
                    dilation=dilation,
                    groups=dim,
                )
                self.norm = nn.LayerNorm(dim, eps=1e-6)
                self.pwconv1 = nn.Linear(dim, intermediate_dim)
                self.act = nn.GELU()
                self.pwconv2 = nn.Linear(intermediate_dim, dim)
                self.weight = (
                    nn.Parameter(layer_scale_init_value * torch.ones(dim))
                    if layer_scale_init_value > 0
                    else None
                )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, dim, frames]``。

                Returns
                -------
                torch.Tensor
                    同形状 ``[batch, dim, frames]``。
                """
                residual = x
                y = self.dwconv(x)
                y.transpose_(1, 2)  # [B, T, C]
                x = self.norm(y)
                y = self.pwconv1(x)
                x = self.act(y)
                y = self.pwconv2(x)
                if self.weight is not None:
                    y *= self.weight
                y.transpose_(1, 2)  # [B, C, T]
                x = y + residual
                return x

        class _DVAEDecoderModule(nn.Module):
            """官方 ChatTTS ``DVAEDecoder`` 子模块。

            ``conv_in`` (Sequential) + ``decoder_block`` (ModuleList[
            ConvNeXtBlock]) + ``conv_out``。该子模块既用于 ``encoder`` 也
            用于 ``decoder``，对应官方 ``DVAEDecoder`` 类。
            """

            def __init__(
                self,
                idim: int,
                odim: int,
                n_layer: int = 12,
                bn_dim: int = 64,
                hidden: int = 256,
                kernel: int = 7,
                dilation: int = 2,
                up: bool = False,
            ) -> None:
                super().__init__()
                self.conv_in = nn.Sequential(
                    nn.Conv1d(idim, bn_dim, 3, 1, 1),  # conv_in.0
                    nn.GELU(),  # conv_in.1
                    nn.Conv1d(bn_dim, hidden, 3, 1, 1),  # conv_in.2
                )
                self.decoder_block = nn.ModuleList(
                    [
                        _ConvNeXtBlock(hidden, hidden * 4, kernel, dilation)
                        for _ in range(n_layer)
                    ]
                )
                self.conv_out = nn.Conv1d(
                    hidden, odim, kernel_size=1, bias=False
                )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, idim, frames]``。

                Returns
                -------
                torch.Tensor
                    ``[batch, odim, frames]``。
                """
                y = self.conv_in(x)
                for f in self.decoder_block:
                    y = f(y)
                x = self.conv_out(y)
                return x

        class _ResidualFSQ(nn.Module):
            """对应官方 ``ResidualFSQ``，仅保留推理（decode）所需结构。

            可学习参数只有 ``project_in`` 和 ``project_out``（当
            ``codebook_dim != dim`` 时）。FSQ 量化本身无可学习参数。
            state_dict key: ``rvqs.{g}.project_in/project_out.{weight,bias}``
            """

            def __init__(self, dim: int, levels: list, num_quantizers: int):
                super().__init__()
                codebook_dim = len(levels)
                self.dim = dim
                self.codebook_dim = codebook_dim
                self.num_quantizers = num_quantizers
                self.levels = levels

                requires_projection = codebook_dim != dim
                self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
                self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
                self.has_projections = requires_projection

                # 非持久化 buffer（不出现在 state_dict 中，与官方一致）
                levels_t = torch.tensor(levels, dtype=torch.float32)
                basis = torch.cumprod(
                    torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0
                )
                self.register_buffer("_basis", basis, persistent=False)
                scales = torch.stack([levels_t ** (-i) for i in range(num_quantizers)])
                self.register_buffer("scales", scales, persistent=False)

            def _indices_to_codes(self, flat_indices: Any) -> Any:
                """将 flat index 转为量化值（preserve_symmetry=True）。

                levels=5 时: index {0..624} → 4 维向量, 每维 ∈ {-1, -0.5, 0, 0.5, 1}
                """
                # flat_indices: [...] → [..., 1]
                idx = flat_indices.unsqueeze(-1)
                level_indices = (idx // self._basis) % torch.tensor(
                    self.levels, device=idx.device
                )
                # preserve_symmetry: code = level_idx * (2/(levels-1)) - 1
                levels_t = torch.tensor(
                    self.levels, dtype=torch.float32, device=idx.device
                )
                codes = level_indices.float() * (2.0 / (levels_t - 1)) - 1.0
                return codes

            def get_output_from_indices(self, indices: Any) -> Any:
                """indices: [B, T, R] → [B, T, dim]"""
                # 对每个残差量化器，取出索引、转为量化值、按 scales 缩放
                codes_summed = None
                for r in range(self.num_quantizers):
                    flat_idx = indices[..., r]  # [B, T]
                    codes = self._indices_to_codes(flat_idx)  # [B, T, codebook_dim]
                    codes = codes * self.scales[r]  # 逐维缩放
                    codes_summed = codes if codes_summed is None else codes_summed + codes
                # 确保与权重 dtype 一致（量化计算可能产生 float32，
                # 但模型权重可能是 float16）
                w = self.project_out.weight
                if codes_summed.dtype != w.dtype:
                    codes_summed = codes_summed.to(w.dtype)
                # project_out: [B, T, dim]
                return self.project_out(codes_summed)

        class _GroupedResidualFSQ(nn.Module):
            """对应官方 ``GroupedResidualFSQ``。

            state_dict key: ``quantizer.rvqs.{g}.project_in/project_out.*``
            """

            def __init__(self, dim: int, levels: list, num_quantizers: int, groups: int):
                super().__init__()
                self.dim = dim
                self.groups = groups
                dim_per_group = dim // groups
                self.rvqs = nn.ModuleList([
                    _ResidualFSQ(dim_per_group, levels, num_quantizers)
                    for _ in range(groups)
                ])

            def get_output_from_indices(self, indices: Any) -> Any:
                """indices: [G, B, T, R] → [B, T, dim]"""
                outputs = [
                    rvq.get_output_from_indices(indices[g])
                    for g, rvq in enumerate(self.rvqs)
                ]
                return torch.cat(outputs, dim=-1)  # [B, T, dim]

        class _GFSQ(nn.Module):
            """官方 ChatTTS ``GFSQ``：自实现，不依赖 vector_quantize_pytorch。

            将 token ids ``[B, num_vq, T]``（``num_vq = G * R``）嵌入为连续
            特征 ``[B, dim, T]``。

            state_dict key: ``vq_layer.quantizer.rvqs.{g}.project_in/out.*``
            """

            def __init__(
                self,
                dim: int,
                levels: tuple,
                G: int,
                R: int,
                eps: float = 1e-5,
                transpose: bool = True,
            ) -> None:
                super().__init__()
                self.quantizer = _GroupedResidualFSQ(
                    dim=dim,
                    levels=list(levels),
                    num_quantizers=R,
                    groups=G,
                )
                self.n_ind = math.prod(levels)
                self.eps = eps
                self.transpose = transpose
                self.G = G
                self.R = R

            def _embed(self, x: Any) -> Any:
                """将 token ids 嵌入为连续特征。

                Parameters
                ----------
                x : torch.Tensor
                    ``[B, num_vq, T]``，``num_vq = G * R``。

                Returns
                -------
                torch.Tensor
                    ``[B, dim, T]``（``transpose=True`` 时）。
                """
                # x: [B, num_vq, T]
                if self.transpose:
                    x = x.transpose(1, 2)  # [B, T, num_vq]
                x = x.view(
                    x.size(0), x.size(1), self.G, self.R
                ).permute(2, 0, 1, 3)  # [G, B, T, R]
                feat = self.quantizer.get_output_from_indices(
                    x
                )  # [B, T, dim]
                if self.transpose:
                    return feat.transpose_(1, 2)  # [B, dim, T]
                return feat

        class _DVAEDecoder(nn.Module):
            """与官方 ChatTTS ``DVAE`` 结构完全一致的解码器实现。

            顶层模块持有 ``coef`` / ``downsample_conv`` / ``encoder`` /
            ``decoder`` / ``out_conv`` / ``vq_layer``，``state_dict`` key
            与官方 ``DVAE.safetensors`` 一一对应。

            Parameters
            ----------
            dim : int
                主干维度，等于 ``decoder.idim`` 与 ``decoder.odim``；
                VQ 维度为 ``2 * dim``。官方默认 ``512``。
            n_layer : int
                ConvNeXt 块数，官方默认 ``12``。
            hidden : int | None
                ConvNeXt 内部隐藏维度；``None`` 时取 ``dim // 2``（官方
                ``dim=512`` 时为 ``256``）。
            bn_dim : int | None
                ``conv_in`` 瓶颈维度；``None`` 时取 ``dim // 4``（官方
                ``dim=512`` 时为 ``128``）。
            kernel : int
                Depthwise 卷积核大小，官方默认 ``7``。
            dilation : int
                膨胀率，官方默认 ``2``。
            mel_bins : int
                mel 频谱维度（``coef`` 与 ``out_conv`` 输出通道数），官方
                ``100``。
            levels : tuple
                GFSQ 各码本层级，官方 ``(5, 5, 5, 5)``。
            G : int
                GFSQ 分组数，官方 ``2``。
            R : int
                GFSQ 每组残差量化器数，官方 ``2``；``num_vq = G * R``。
            with_encoder : bool
                是否构建 ``downsample_conv`` + ``encoder``（含 encoder 的
                ``DVAE.safetensors`` 需 ``True`` 以匹配全部 key）。
            coef : torch.Tensor | None
                ``coef`` 初值；``None`` 时随机初始化（加载权重后覆盖）。
            """

            def __init__(
                self,
                dim: int = 512,
                n_layer: int = 12,
                hidden: int | None = None,
                bn_dim: int | None = None,
                kernel: int = 7,
                dilation: int = 2,
                mel_bins: int = 100,
                levels: tuple = (5, 5, 5, 5),
                G: int = 2,
                R: int = 2,
                with_encoder: bool = True,
                coef: Any = None,
            ) -> None:
                super().__init__()
                if hidden is None:
                    hidden = dim // 2
                if bn_dim is None:
                    bn_dim = dim // 4
                vq_dim = 2 * dim  # GFSQ 输出维度，官方 1024

                # coef buffer: [1, mel_bins, 1]
                if coef is None:
                    coef = torch.rand(mel_bins)
                self.register_buffer("coef", coef.unsqueeze(0).unsqueeze_(2))

                if with_encoder:
                    self.downsample_conv = nn.Sequential(
                        nn.Conv1d(mel_bins, dim, 3, 1, 1),  # downsample_conv.0
                        nn.GELU(),  # downsample_conv.1
                        nn.Conv1d(dim, dim, 4, 2, 1),  # downsample_conv.2
                        nn.GELU(),  # downsample_conv.3
                    )
                    self.encoder = _DVAEDecoderModule(
                        idim=dim,
                        odim=2 * dim,
                        n_layer=n_layer,
                        bn_dim=bn_dim,
                        hidden=hidden,
                        kernel=kernel,
                        dilation=dilation,
                    )

                self.decoder = _DVAEDecoderModule(
                    idim=dim,
                    odim=dim,
                    n_layer=n_layer,
                    bn_dim=bn_dim,
                    hidden=hidden,
                    kernel=kernel,
                    dilation=dilation,
                )
                self.out_conv = nn.Conv1d(
                    dim, mel_bins, 3, 1, 1, bias=False
                )
                self.vq_layer = _GFSQ(dim=vq_dim, levels=levels, G=G, R=R)

                self._is_loaded: bool = False

            # ----------------------------------------------------------
            # 前向
            # ----------------------------------------------------------
            def forward(self, inp: Any, mode: str = "decode") -> Any:
                """将 VQ token ids 解码为 mel 频谱（官方 decode 逻辑）。

                Parameters
                ----------
                inp : torch.Tensor
                    ``[num_vq, frames]`` 或 ``[batch, num_vq, frames]``，
                    ``num_vq = G * R``。
                mode : str
                    仅支持 ``"decode"``。

                Returns
                -------
                torch.Tensor
                    输入 2D 时返回 ``[mel_bins, frames*2]``；
                    输入 3D 时返回 ``[batch, mel_bins, frames*2]``。
                    时间维度因 reshape 翻倍（与官方一致）。
                """
                squeeze = inp.dim() == 2
                if squeeze:
                    inp = inp.unsqueeze(0)  # [1, num_vq, T]

                if mode == "decode":
                    if self.vq_layer is not None:
                        vq_feats = self.vq_layer._embed(inp)  # [B, 2*dim, T]
                    else:
                        vq_feats = inp
                    # 把 hidden 维度拆成 2 组，时间维度翻倍
                    vq_feats = vq_feats.view(
                        (
                            vq_feats.size(0),
                            2,
                            vq_feats.size(1) // 2,
                            vq_feats.size(2),
                        )
                    ).permute(0, 2, 3, 1).flatten(
                        2
                    )  # [B, dim, T*2]
                    dec_out = self.out_conv(
                        self.decoder(vq_feats)
                    )  # [B, mel_bins, T*2]
                    out = torch.mul(dec_out, self.coef)
                else:
                    raise NotImplementedError(
                        f"unsupported mode: {mode!r}"
                    )

                if squeeze:
                    out = out.squeeze(0)
                return out

            def forward_chunk(self, token_ids: Any) -> Any:
                """流式解码兼容接口。

                官方 DVAE 解码为无状态的全卷积网络，此处直接对当前块执行
                完整 decode（与 :meth:`forward` 等价），保留接口供复合声码器
                流式管线调用。
                """
                return self.forward(token_ids)

            def reset_stream_buffer(self) -> None:
                """重置流式状态（官方 DVAE 无状态，此处为接口兼容空实现）。"""
                pass

            # ----------------------------------------------------------
            # 权重加载 / 释放
            # ----------------------------------------------------------
            def load_weights(
                self,
                weights_path: str,
                device: str = "cuda",
                dtype: str = "float16",
            ) -> None:
                """从 safetensors / pytorch checkpoint 加载权重。

                加载步骤：

                1. 解析 dtype 字符串为 torch dtype；
                2. 无 GPU 时将设备降级为 CPU；
                3. 读取权重（safetensors 优先，回退到 .pt/.pth/.bin）；
                4. 直接 ``load_state_dict(strict=False)`` 载入（key 与官方
                   完全匹配，无需前缀剥离）；
                5. 移动到目标 device / dtype，切换为 eval。

                Parameters
                ----------
                weights_path : str
                    权重文件路径或目录（目录下查找 ``dvae.safetensors`` /
                    ``decoder.safetensors``）。
                device : str
                    目标设备；无 GPU 时自动降级为 CPU。
                dtype : str
                    数据精度，``"float16"`` / ``"float32"`` / ``"bfloat16"``。
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

                state_dict = self._load_state_dict(weights_path)
                if state_dict:
                    # key 与官方 DVAE.safetensors 完全匹配，直接载入
                    self.load_state_dict(state_dict, strict=False)

                self.to(device=resolved, dtype=torch_dtype)
                self.eval()
                self._is_loaded = True

            def unload_weights(self) -> None:
                """释放权重：移至 CPU 并清空 CUDA 缓存。"""
                try:
                    import torch

                    self.to("cpu")
                    from mosaic.core.device_utils import empty_device_cache

                    empty_device_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._is_loaded = False

            # ----------------------------------------------------------
            # 内部辅助：权重读取
            # ----------------------------------------------------------
            @staticmethod
            def _load_state_dict(weights_path: str) -> dict[str, Any]:
                """读取权重为 state_dict。"""
                import torch

                state_dict: dict[str, Any] = {}
                if os.path.isfile(weights_path):
                    if weights_path.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        state_dict = load_file(weights_path)
                    elif weights_path.endswith((".pt", ".pth", ".bin")):
                        ckpt = torch.load(
                            weights_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        state_dict = _unwrap_ckpt(ckpt)
                elif os.path.isdir(weights_path):
                    for fname in ("dvae.safetensors", "decoder.safetensors"):
                        fpath = os.path.join(weights_path, fname)
                        if os.path.isfile(fpath):
                            from safetensors.torch import load_file

                            state_dict = load_file(fpath)
                            break
                    if not state_dict:
                        for fname in ("dvae.bin", "decoder.bin"):
                            fpath = os.path.join(weights_path, fname)
                            if os.path.isfile(fpath):
                                ckpt = torch.load(
                                    fpath,
                                    map_location="cpu",
                                    weights_only=False,
                                )
                                state_dict = _unwrap_ckpt(ckpt)
                                break
                return state_dict

        _DVAEDecoderClass = _DVAEDecoder
    return _DVAEDecoderClass


class DVAEDecoder:
    """DVAE 解码器代理类。

    在首次实例化时惰性创建与官方 ChatTTS DVAE 结构完全一致的
    ``nn.Module`` 子类实例，并将属性访问与方法调用转发给内部实现。这样
    既能让本模块在未安装 ``torch`` 时被导入，又能在 ``torch`` 可用时获得
    完整的 ``nn.Module`` 行为（包括 ``isinstance`` 检查与 ``forward``
    调用的一致性），同时 ``state_dict`` key 与官方 ``DVAE.safetensors``
    完全匹配。

    Parameters
    ----------
    num_vq : int
        VQ 码本组数，``num_vq = G * R``，默认 ``4``（对应官方 ``G=2, R=2``）。
        其它偶数值按 ``G = num_vq // 2, R = 2`` 拆分。
    num_audio_tokens : int
        兼容旧接口的参数（官方 GFSQ 由 ``levels`` 决定码字数，此处保留
        但不参与建模）。
    hidden_size : int
        主干维度 ``dim``，等于 ``decoder.idim`` / ``decoder.odim``，官方
        ``512``。
    mel_bins : int
        mel 频谱维度（``coef`` 与 ``out_conv`` 输出通道数），官方 ``100``。
    num_layers : int
        ConvNeXt 块数 ``n_layer``，官方 ``12``。
    dilation_rates : list[int] | None
        兼容旧接口的参数（官方固定 ``dilation=2``，此处保留但不参与建模）。
    output_length_factor : int
        兼容旧接口的参数（官方 decode 通过 reshape 使时间翻倍，此处保留
        但不参与建模）。
    """

    def __init__(
        self,
        num_vq: int = 4,
        num_audio_tokens: int = 1024,
        hidden_size: int = 512,
        mel_bins: int = 80,
        num_layers: int = 12,
        dilation_rates: list[int] | None = None,
        output_length_factor: int = 1,
    ) -> None:
        # 由 num_vq 推导 G、R（num_vq = G * R）；官方 num_vq=4 -> G=2, R=2
        if num_vq >= 2 and num_vq % 2 == 0:
            G, R = num_vq // 2, 2
        else:
            G, R = 1, max(1, num_vq)
        levels = (5,) * (G * R)

        cls = _get_dvae_class()
        self._impl = cls(
            dim=hidden_size,
            n_layer=num_layers,
            mel_bins=mel_bins,
            levels=levels,
            G=G,
            R=R,
            with_encoder=True,
        )

    def __getattr__(self, name: str) -> Any:
        """将未在代理类上找到的属性转发给内部 ``nn.Module`` 实现。"""
        impl = self.__dict__.get("_impl")
        if impl is not None:
            return getattr(impl, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """转发调用，触发 ``nn.Module.__call__``（含 forward hooks）。"""
        return self._impl(*args, **kwargs)

    # 显式转发关键方法
    def forward(self, token_ids: Any) -> Any:
        """将 VQ token ids 解码为 mel 频谱（官方 decode 逻辑）。"""
        return self._impl.forward(token_ids)

    def forward_chunk(self, token_ids: Any) -> Any:
        """流式解码兼容接口。"""
        return self._impl.forward_chunk(token_ids)

    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """从 safetensors / pytorch checkpoint 加载权重。"""
        return self._impl.load_weights(weights_path, device, dtype)

    def unload_weights(self) -> None:
        """释放权重。"""
        return self._impl.unload_weights()

    def reset_stream_buffer(self) -> None:
        """重置流式状态（接口兼容）。"""
        return self._impl.reset_stream_buffer()
