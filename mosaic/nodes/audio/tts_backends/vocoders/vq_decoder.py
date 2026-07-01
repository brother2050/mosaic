# mosaic/nodes/audio/tts_backends/vocoders/vq_decoder.py
"""Fish Speech VQ 解码器。

Layer 3 前置步骤：将 VQ 音频码 token 解码为 mel 频谱。

本模块实现 Fish Speech 的 VQ 解码器，将多组 VQ 码本 token ids 解码为 mel
频谱。它是声码器管线中的「第一步」（token -> mel），其后通常接 Vocos /
HiFi-GAN 等声码器（mel -> waveform）。

与 :class:`~mosaic.nodes.audio.tts_backends.vocoders.dvae.DVAEDecoder` 的
关键差异：

1. 码本查找采用**拼接**（``torch.cat``）而非逐元素求和；
2. 解码网络采用**残差卷积块**（Conv1D + BatchNorm1d + GELU + Conv1D +
   残差连接）而非 ConvNeXt 膨胀卷积；
3. 投影层从 ``codebook_dim × num_codebooks`` 到 ``hidden_size``。

架构
----
A) 码本查找层
   - ``num_codebooks`` 个 :class:`torch.nn.Embedding`
     ``(codebook_size, codebook_dim)``；
   - 多组码本嵌入在最后一维**拼接**，得到
     ``[batch, frames, codebook_dim × num_codebooks]``。

B) 投影层
   - :class:`torch.nn.Linear`
     ``(codebook_dim × num_codebooks, hidden_size)``。

C) 残差卷积网络
   - 每层：Conv1D(kernel=3, padding=1) → BatchNorm1d → GELU →
     Conv1D(kernel=3, padding=1) → 残差连接；
   - 重复 ``num_layers`` 次。

D) 输出层
   - :class:`torch.nn.Conv1d`
     ``(hidden_size, mel_bins, kernel=1)``，输出
     ``[batch, mel_bins, frames]``。

设计要点
--------
* ``torch`` / ``safetensors`` 采用惰性导入：模块顶层不导入这些重依赖，
  真正的 ``nn.Module`` 子类在首次实例化时通过 :func:`_get_vq_class`
  惰性构建，使本模块在未安装 ``torch`` 时仍可被导入。
* :class:`VQDecoder` 是代理类，将属性访问与方法调用转发给内部
  ``nn.Module`` 实例，从而既保持 ``nn.Module`` 一致的行为，又避免在模块
  顶层硬依赖 ``torch``。
* ``token_ids`` / ``mel`` 等参数类型用 :data:`~typing.Any` 标注。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["VQDecoder"]


# 内部缓存的 nn.Module 子类（惰性创建）
_VQDecoderClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_vq_class() -> Any:
    """惰性创建并返回真正的 VQ 解码器 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义 :class:`_ResidualConvBlock` 与
    :class:`_VQDecoder`，随后缓存到全局变量 :data:`_VQDecoderClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_VQDecoder``。
    """
    global _VQDecoderClass
    if _VQDecoderClass is None:
        import torch
        import torch.nn as nn

        class _ResidualConvBlock(nn.Module):
            """残差卷积块。

            结构：Conv1D(kernel=3, padding=1) → BatchNorm1d → GELU →
            Conv1D(kernel=3, padding=1) → 残差连接。

            所有运算保持 ``hidden_size`` 维度不变，残差可直接相加。
            """

            def __init__(
                self, hidden_size: int, kernel_size: int = 3
            ) -> None:
                super().__init__()
                padding = kernel_size // 2
                self.conv1 = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                )
                self.norm = nn.BatchNorm1d(hidden_size)
                self.act = nn.GELU()
                self.conv2 = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, hidden_size, frames]``。

                Returns
                -------
                torch.Tensor
                    同形状 ``[batch, hidden_size, frames]``。
                """
                residual = x
                x = self.conv1(x)            # [B, H, T]
                x = self.norm(x)             # BatchNorm1d 按通道归一化
                x = self.act(x)              # GELU
                x = self.conv2(x)            # [B, H, T]
                return residual + x

        class _VQDecoder(nn.Module):
            """VQ 解码器真实实现（``nn.Module`` 子类）。

            将多组 VQ 码本 token 解码为 mel 频谱。码本嵌入采用拼接策略，
            解码主干为残差卷积块。
            """

            def __init__(
                self,
                codec_type: str = "dac",
                codebook_size: int = 1024,
                codebook_dim: int = 8,
                num_codebooks: int = 1,
                hidden_size: int = 512,
                mel_bins: int = 80,
                num_layers: int = 6,
                output_length_factor: int = 1,
            ) -> None:
                super().__init__()
                # 参数记录
                self.codec_type = codec_type
                self.codebook_size = codebook_size
                self.codebook_dim = codebook_dim
                self.num_codebooks = num_codebooks
                self.hidden_size = hidden_size
                self.mel_bins = mel_bins
                self.num_layers = num_layers
                self.output_length_factor = output_length_factor
                self._kernel_size = 3

                # A) 码本查找层（拼接而非求和）
                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(codebook_size, codebook_dim)
                        for _ in range(num_codebooks)
                    ]
                )

                # B) 投影层：codebook_dim × num_codebooks -> hidden_size
                self.proj = nn.Linear(
                    codebook_dim * num_codebooks, hidden_size
                )

                # C) 残差卷积网络
                self.blocks = nn.ModuleList(
                    [
                        _ResidualConvBlock(hidden_size, self._kernel_size)
                        for _ in range(num_layers)
                    ]
                )

                # D) 输出层
                self.out_conv = nn.Conv1d(
                    hidden_size, mel_bins, kernel_size=1
                )

                # 流式缓冲区（forward_chunk 使用）
                self._stream_buffer: Any = None
                self._is_loaded: bool = False

            # ----------------------------------------------------------
            # 辅助
            # ----------------------------------------------------------
            def _receptive_field(self) -> int:
                """计算残差卷积网络的时间感受野（帧数）。

                每个残差块包含 2 个 ``kernel=3`` 的 Conv1D，每个贡献
                ``(kernel - 1) = 2`` 的感受野增量，故每层增量为 4。
                """
                rf = 0
                for _ in range(self.num_layers):
                    rf += (self._kernel_size - 1) * 2
                return rf

            def _embed_tokens(
                self, token_ids: Any
            ) -> tuple[Any, bool]:
                """将 VQ token ids 转为拼接后的嵌入。

                与 DVAEDecoder 不同，此处将多组码本嵌入在最后一维
                **拼接**（``torch.cat``）而非逐元素求和。

                Parameters
                ----------
                token_ids : torch.Tensor
                    ``[num_codebooks, frames]`` 或
                    ``[batch, num_codebooks, frames]``。

                Returns
                -------
                emb : torch.Tensor
                    ``[batch, frames, codebook_dim × num_codebooks]``。
                squeeze_batch : bool
                    输入是否为 2D（需要在输出时 squeeze 回去）。
                """
                squeeze_batch = False
                if token_ids.dim() == 2:
                    # [num_codebooks, frames] -> [1, num_codebooks, frames]
                    token_ids = token_ids.unsqueeze(0)
                    squeeze_batch = True
                # [batch, num_codebooks, frames]
                n_cb = min(self.num_codebooks, token_ids.size(1))
                embs: list[Any] = []
                for i in range(n_cb):
                    e = self.embeddings[i](
                        token_ids[:, i, :].long()
                    )  # [batch, frames, codebook_dim]
                    embs.append(e)
                # 拼接而非求和
                emb = torch.cat(embs, dim=-1)
                return emb, squeeze_batch

            def _decode_hidden(self, emb: Any) -> Any:
                """对嵌入执行投影 + 残差卷积 + 输出投影。

                Parameters
                ----------
                emb : torch.Tensor
                    ``[batch, frames, codebook_dim × num_codebooks]``。

                Returns
                -------
                torch.Tensor
                    ``[batch, mel_bins, frames]``。
                """
                import torch

                x = self.proj(emb)             # [B, frames, hidden]
                x = x.transpose(1, 2)          # [B, hidden, frames]
                for block in self.blocks:
                    x = block(x)
                mel = self.out_conv(x)         # [B, mel_bins, frames]
                if self.output_length_factor != 1:
                    mel = torch.nn.functional.interpolate(
                        mel,
                        scale_factor=float(self.output_length_factor),
                        mode="linear",
                        align_corners=False,
                    )
                return mel

            # ----------------------------------------------------------
            # 前向
            # ----------------------------------------------------------
            def forward(self, token_ids: Any) -> Any:
                """将 VQ token ids 解码为 mel 频谱。

                Parameters
                ----------
                token_ids : torch.Tensor
                    ``[num_codebooks, frames]`` 或
                    ``[batch, num_codebooks, frames]``。

                Returns
                -------
                torch.Tensor
                    输入 2D 时返回 ``[mel_bins, frames]``；
                    输入 3D 时返回 ``[batch, mel_bins, frames]``。
                """
                emb, squeeze_batch = self._embed_tokens(token_ids)
                mel = self._decode_hidden(emb)
                if squeeze_batch:
                    mel = mel.squeeze(0)
                return mel

            def forward_chunk(self, token_ids: Any) -> Any:
                """流式解码：维护残差卷积感受野缓冲区。

                每次 ``forward_chunk`` 调用时，将上一块末尾的若干帧嵌入缓存
                作为左侧上下文拼接到当前块前，使卷积获得正确感受野，从而
                减小块边界伪影。仅输出当前块对应的新帧。

                使用 :meth:`reset_stream_buffer` 重置流式状态。

                Parameters
                ----------
                token_ids : torch.Tensor
                    ``[num_codebooks, chunk_frames]`` 或
                    ``[batch, num_codebooks, chunk_frames]``。

                Returns
                -------
                torch.Tensor
                    当前块对应的 mel，``[mel_bins, chunk_frames]`` 或
                    ``[batch, mel_bins, chunk_frames]``。
                """
                import torch

                emb, squeeze_batch = self._embed_tokens(token_ids)

                # 拼接上一块缓冲区作为左侧上下文
                buffer_frames = 0
                if self._stream_buffer is not None:
                    emb = torch.cat([self._stream_buffer, emb], dim=1)
                    buffer_frames = self._stream_buffer.shape[1]

                mel = self._decode_hidden(emb)  # [B, mel_bins, total]

                # 丢弃缓冲区对应的（已在上一次输出过的）部分
                if buffer_frames > 0 and mel.shape[-1] > buffer_frames:
                    mel = mel[..., buffer_frames:]

                # 更新缓冲区：保留最近感受野帧
                rf = self._receptive_field()
                if rf > 0 and emb.shape[1] >= rf:
                    self._stream_buffer = emb[:, -rf:, :].detach()
                else:
                    self._stream_buffer = emb.detach()

                if squeeze_batch:
                    mel = mel.squeeze(0)
                return mel

            def reset_stream_buffer(self) -> None:
                """重置流式缓冲区。"""
                self._stream_buffer = None

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
                4. 剥离常见前缀后 ``strict=False`` 载入；
                5. 移动到目标 device / dtype，切换为 eval。

                Parameters
                ----------
                weights_path : str
                    权重文件路径或目录（目录下查找 ``vq.safetensors`` /
                    ``decoder.safetensors`` / ``codec.safetensors`` /
                    ``dac.safetensors``）。
                device : str
                    目标设备；无 GPU 时自动降级为 CPU。
                dtype : str
                    数据精度，``"float16"`` / ``"float32"`` / ``"bfloat16"``。

                Raises
                ------
                ImportError
                    ``torch`` / ``safetensors`` 未安装。
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
                    state_dict = self._strip_prefix(state_dict)
                    self.load_state_dict(state_dict, strict=False)

                self.to(device=resolved, dtype=torch_dtype)
                self.eval()
                self._is_loaded = True

            def unload_weights(self) -> None:
                """释放权重：移至 CPU 并清空 CUDA 缓存。"""
                try:
                    import torch

                    self.to("cpu")
                    from mosaic.core._device_utils import empty_device_cache

                    empty_device_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._stream_buffer = None
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
                        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                        state_dict = _unwrap_ckpt(ckpt)
                elif os.path.isdir(weights_path):
                    for fname in (
                        "vq.safetensors",
                        "decoder.safetensors",
                        "codec.safetensors",
                        "dac.safetensors",
                    ):
                        fpath = os.path.join(weights_path, fname)
                        if os.path.isfile(fpath):
                            from safetensors.torch import load_file

                            state_dict = load_file(fpath)
                            break
                    if not state_dict:
                        for fname in (
                            "vq.bin",
                            "decoder.bin",
                            "codec.bin",
                            "dac.bin",
                        ):
                            fpath = os.path.join(weights_path, fname)
                            if os.path.isfile(fpath):
                                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                                state_dict = _unwrap_ckpt(ckpt)
                                break
                return state_dict

            @staticmethod
            def _strip_prefix(
                state_dict: dict[str, Any]
            ) -> dict[str, Any]:
                """剥离常见前缀以提升匹配率。

                当且仅当 state_dict 中存在以这些前缀开头的 key 时才剥离，
                避免破坏本就无前缀的权重。
                """
                prefixes = (
                    "vq.",
                    "decoder.",
                    "codec.",
                    "dac.",
                    "generator.",
                )
                has_prefix = any(
                    any(k.startswith(p) for k in state_dict)
                    for p in prefixes
                )
                if not has_prefix:
                    return state_dict
                stripped: dict[str, Any] = {}
                for key, value in state_dict.items():
                    new_key = key
                    for p in prefixes:
                        if key.startswith(p):
                            new_key = key[len(p):]
                            break
                    stripped[new_key] = value
                return stripped

        _VQDecoderClass = _VQDecoder
    return _VQDecoderClass


class VQDecoder:
    """VQ 解码器代理类。

    在首次实例化时惰性创建真正的 ``nn.Module`` 子类实例，并将属性访问与
    方法调用转发给内部实现。这样既能让本模块在未安装 ``torch`` 时被导入，
    又能在 ``torch`` 可用时获得完整的 ``nn.Module`` 行为（包括
    ``isinstance`` 检查与 ``forward`` 调用的一致性）。

    Parameters
    ----------
    codec_type : str
        编码器/解码器类型，默认 ``"dac"``。
    codebook_size : int
        码本大小（每个码本的码字数），默认 ``1024``。
    codebook_dim : int
        每个码字的维度，默认 ``8``。
    num_codebooks : int
        码本数量，默认 ``1``。
    hidden_size : int
        隐藏层维度，默认 ``512``。
    mel_bins : int
        mel 频谱维度，默认 ``80``。
    num_layers : int
        残差卷积块数量，默认 ``6``。
    output_length_factor : int
        输出长度倍率，默认 ``1``（不缩放）。
    """

    def __init__(
        self,
        codec_type: str = "dac",
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        num_codebooks: int = 1,
        hidden_size: int = 512,
        mel_bins: int = 80,
        num_layers: int = 6,
        output_length_factor: int = 1,
    ) -> None:
        cls = _get_vq_class()
        self._impl = cls(
            codec_type=codec_type,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_codebooks=num_codebooks,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            num_layers=num_layers,
            output_length_factor=output_length_factor,
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
        """将 VQ token ids 解码为 mel 频谱。"""
        return self._impl.forward(token_ids)

    def forward_chunk(self, token_ids: Any) -> Any:
        """流式解码：维护残差卷积感受野缓冲区。"""
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
        """重置流式缓冲区。"""
        return self._impl.reset_stream_buffer()
