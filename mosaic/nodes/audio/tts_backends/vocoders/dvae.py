# mosaic/nodes/audio/tts_backends/vocoders/dvae.py
"""ChatTTS DVAE 解码器。

Layer 3 前置步骤：将 VQ 音频码 token 解码为 mel 频谱。

本模块实现 ChatTTS 离散变分自编码器（Discrete VAE, DVAE）的**解码部分**：
将多组 VQ 码本 token ids 解码为 mel 频谱。它是声码器管线中的「第一步」
（token -> mel），其后通常接 Vocos / HiFi-GAN 等声码器（mel -> waveform）。

架构
----
A) 向量量化嵌入层（GFSQ 解码部分）
   - ``num_vq`` 个 :class:`torch.nn.Embedding`，每码本形状
     ``(num_audio_tokens, hidden_size)``；
   - 多组码本嵌入逐元素求和，得到 ``[batch, frames, hidden_size]``。

B) ConvNeXt 解码主干
   - 每层：Depthwise Conv1D（膨胀卷积）-> LayerNorm -> Pointwise Conv(1x1)
     -> GELU -> 残差连接；
   - 各层膨胀率由 ``dilation_rates`` 指定，逐步扩大感受野；
   - 主干末端附加一层 :class:`torch.nn.LayerNorm` 以稳定数值。

C) 输出投影层
   - :class:`torch.nn.Linear` ``(hidden_size, mel_bins)``，将隐藏特征映射
     为 mel 频谱，输出 ``[batch, mel_bins, frames]``。

设计要点
--------
* ``torch`` / ``safetensors`` 采用惰性导入：模块顶层不导入这些重依赖，
  真正的 ``nn.Module`` 子类在首次实例化时通过 :func:`_get_dvae_class`
  惰性构建，使本模块在未安装 ``torch`` 时仍可被导入。
* :class:`DVAEDecoder` 是代理类，将属性访问与方法调用转发给内部
  ``nn.Module`` 实例，从而既保持 ``nn.Module`` 一致的行为，又避免在模块
  顶层硬依赖 ``torch``。
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
    """惰性创建并返回真正的 DVAE 解码器 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义 :class:`_ConvNeXtBlock` 与
    :class:`_DVAEDecoder`，随后缓存到全局变量 :data:`_DVAEDecoderClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_DVAEDecoder``。
    """
    global _DVAEDecoderClass
    if _DVAEDecoderClass is None:
        import torch.nn as nn

        class _ConvNeXtBlock(nn.Module):
            """ConvNeXt 解码块。

            结构：Depthwise Conv1D（膨胀）-> LayerNorm -> Pointwise Conv(1x1)
            -> GELU -> 残差连接。所有运算保持 ``hidden_size`` 维度不变，
            残差可直接相加。
            """

            def __init__(
                self,
                hidden_size: int,
                dilation_rate: int = 1,
                kernel_size: int = 7,
            ) -> None:
                super().__init__()
                # Depthwise 膨胀卷积（same padding，保持帧长不变）
                padding = (kernel_size - 1) * dilation_rate // 2
                self.dwconv = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation_rate,
                    groups=hidden_size,
                )
                self.norm = nn.LayerNorm(hidden_size)
                # Pointwise 1x1 卷积
                self.pwconv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
                self.act = nn.GELU()

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
                x = self.dwconv(x)            # [B, H, T]
                x = x.transpose(1, 2)         # [B, T, H]
                x = self.norm(x)
                x = x.transpose(1, 2)         # [B, H, T]
                x = self.pwconv(x)            # pointwise 1x1
                x = self.act(x)
                return residual + x

        class _DVAEDecoder(nn.Module):
            """DVAE 解码器真实实现（``nn.Module`` 子类）。

            将多组 VQ 音频码 token 解码为 mel 频谱。
            """

            def __init__(
                self,
                num_vq: int = 4,
                num_audio_tokens: int = 1024,
                hidden_size: int = 512,
                mel_bins: int = 80,
                num_layers: int = 6,
                dilation_rates: list[int] | None = None,
                output_length_factor: int = 1,
            ) -> None:
                super().__init__()
                if dilation_rates is None:
                    dilation_rates = [1, 2, 4, 8, 1, 2]
                # 参数记录
                self.num_vq = num_vq
                self.num_audio_tokens = num_audio_tokens
                self.hidden_size = hidden_size
                self.mel_bins = mel_bins
                self.num_layers = num_layers
                self.dilation_rates: list[int] = list(dilation_rates)
                self.output_length_factor = output_length_factor
                self._kernel_size = 7

                # A) 向量量化嵌入层（GFSQ 解码部分）
                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_audio_tokens, hidden_size)
                        for _ in range(num_vq)
                    ]
                )

                # B) ConvNeXt 解码主干
                self.blocks = nn.ModuleList(
                    [
                        _ConvNeXtBlock(
                            hidden_size,
                            dilation_rate=self.dilation_rates[
                                i % len(self.dilation_rates)
                            ],
                            kernel_size=self._kernel_size,
                        )
                        for i in range(num_layers)
                    ]
                )
                # 主干末端归一化，稳定数值
                self.final_norm = nn.LayerNorm(hidden_size)

                # C) 输出投影层
                self.out_proj = nn.Linear(hidden_size, mel_bins)

                # 流式缓冲区（forward_chunk 使用）
                self._stream_buffer: Any = None
                self._is_loaded: bool = False

            # ----------------------------------------------------------
            # 辅助
            # ----------------------------------------------------------
            def _receptive_field(self) -> int:
                """计算主干网络的时间感受野（需要的左侧上下文帧数）。"""
                rf = 0
                for i in range(self.num_layers):
                    d = self.dilation_rates[i % len(self.dilation_rates)]
                    rf += (self._kernel_size - 1) * d
                return rf

            def _embed_tokens(self, token_ids: Any) -> tuple[Any, bool]:
                """将 VQ token ids 转为求和后的嵌入。

                Parameters
                ----------
                token_ids : torch.Tensor
                    ``[num_vq, frames]`` 或 ``[batch, num_vq, frames]``。

                Returns
                -------
                emb : torch.Tensor
                    ``[batch, frames, hidden_size]``。
                squeeze_batch : bool
                    输入是否为 2D（需要在输出时 squeeze 回去）。
                """
                squeeze_batch = False
                if token_ids.dim() == 2:
                    # [num_vq, frames] -> [1, num_vq, frames]
                    token_ids = token_ids.unsqueeze(0)
                    squeeze_batch = True
                # [batch, num_vq, frames]
                n_vq = min(self.num_vq, token_ids.size(1))
                emb: Any = None
                for i in range(n_vq):
                    # clamp 防止 AR 模型生成的越界 token id 导致 Embedding 查表报错
                    safe_ids = token_ids[:, i, :].long().clamp(
                        min=0, max=self.num_audio_tokens - 1
                    )
                    e = self.embeddings[i](safe_ids)
                    emb = e if emb is None else emb + e
                return emb, squeeze_batch

            def _decode_hidden(self, emb: Any) -> Any:
                """对 ``[batch, frames, hidden]`` 嵌入执行主干 + 输出投影。

                Returns
                -------
                torch.Tensor
                    ``[batch, mel_bins, frames]``。
                """
                import torch

                x = emb.transpose(1, 2)            # [B, H, T]
                for block in self.blocks:
                    x = block(x)
                x = x.transpose(1, 2)              # [B, T, H]
                x = self.final_norm(x)
                mel = self.out_proj(x)             # [B, T, mel_bins]
                mel = mel.transpose(1, 2)          # [B, mel_bins, T]
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
                    ``[num_vq, frames]`` 或 ``[batch, num_vq, frames]``。

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
                """流式解码：维护膨胀卷积感受野缓冲区。

                每次 ``forward_chunk`` 调用时，将上一块末尾的若干帧嵌入缓存
                作为左侧上下文拼接到当前块前，使膨胀卷积获得正确感受野，从而
                减小块边界伪影。仅输出当前块对应的新帧。

                使用 :meth:`reset_stream_buffer` 重置流式状态。

                Parameters
                ----------
                token_ids : torch.Tensor
                    ``[num_vq, chunk_frames]`` 或
                    ``[batch, num_vq, chunk_frames]``。

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
                4. 剥离 ``dvae.`` / ``decoder.`` 前缀后 ``strict=False`` 载入；
                5. 移动到目标 device / dtype，切换为 eval。

                Parameters
                ----------
                weights_path : str
                    权重文件路径或目录（目录下查找 ``dvae.safetensors`` /
                    ``decoder.safetensors`` / ``dvae.bin``）。
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
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
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
                                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                                state_dict = _unwrap_ckpt(ckpt)
                                break
                return state_dict

            @staticmethod
            def _strip_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
                """剥离常见前缀（``dvae.`` / ``decoder.``）以提升匹配率。

                当且仅当 state_dict 中存在以这些前缀开头的 key 时才剥离，
                避免破坏本就无前缀的权重。
                """
                prefixes = ("dvae.", "decoder.")
                has_prefix = any(
                    any(k.startswith(p) for k in state_dict) for p in prefixes
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

        _DVAEDecoderClass = _DVAEDecoder
    return _DVAEDecoderClass


class DVAEDecoder:
    """DVAE 解码器代理类。

    在首次实例化时惰性创建真正的 ``nn.Module`` 子类实例，并将属性访问与
    方法调用转发给内部实现。这样既能让本模块在未安装 ``torch`` 时被导入，
    又能在 ``torch`` 可用时获得完整的 ``nn.Module`` 行为（包括
    ``isinstance`` 检查与 ``forward`` 调用的一致性）。

    Parameters
    ----------
    num_vq : int
        VQ 码本组数，默认 ``4``。
    num_audio_tokens : int
        每个码本的码字数，默认 ``1024``。
    hidden_size : int
        隐藏层维度，默认 ``512``。
    mel_bins : int
        mel 频谱维度，默认 ``80``。
    num_layers : int
        ConvNeXt 解码层数，默认 ``6``。
    dilation_rates : list[int] | None
        各层膨胀率；``None`` 时使用 ``[1, 2, 4, 8, 1, 2]``。
    output_length_factor : int
        输出长度缩放因子，默认 ``1``（不缩放）。
    """

    def __init__(
        self,
        num_vq: int = 4,
        num_audio_tokens: int = 1024,
        hidden_size: int = 512,
        mel_bins: int = 80,
        num_layers: int = 6,
        dilation_rates: list[int] | None = None,
        output_length_factor: int = 1,
    ) -> None:
        cls = _get_dvae_class()
        self._impl = cls(
            num_vq=num_vq,
            num_audio_tokens=num_audio_tokens,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            num_layers=num_layers,
            dilation_rates=dilation_rates,
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
        """流式解码：维护膨胀卷积感受野缓冲区。"""
        return self._impl.forward_chunk(token_ids)

    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """从 safetensors / pytorch checkpoint 加载权重。"""
        return self._impl.load_weights(weights_path, device, dtype)

    def unload_weights(self) -> None:
        """释放权重。"""
        return self._impl.unload_weights()
