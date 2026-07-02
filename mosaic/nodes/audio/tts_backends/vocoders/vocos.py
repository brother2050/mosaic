# mosaic/nodes/audio/tts_backends/vocoders/vocos.py
"""Vocos 声码器。

Layer 3: 声码器层。将 mel 频谱解码为音频波形。

本模块实现基于 iSTFT 的 Vocos 声码器，将 mel 频谱 ``[batch, mel_bins,
frames]`` 解码为波形 ``[batch, samples]``。Vocos 通过预测幅度与相位谱再
做逆短时傅里叶变换（iSTFT）重建波形，相比 HiFi-GAN 计算更轻、质量更稳。

架构
----
A) 主干网络（VocosBackbone / ConvNeXt 块）
   - ``embed``: :class:`torch.nn.Conv1d` ``(input_channels -> dim)``，
     kernel=7, padding=3，将 mel 投影到隐藏空间；
   - ``norm``: :class:`torch.nn.LayerNorm` (eps=1e-6)；
   - ``convnext``: ``num_layers`` 个 ConvNeXt 块，每层 Depthwise Conv1D
     (kernel=7, padding=3, groups=dim) -> LayerNorm(eps=1e-6) ->
     Pointwise Linear(dim->intermediate_dim) -> GELU ->
     Pointwise Linear(intermediate_dim->dim) -> gamma 缩放 -> 残差连接；
   - ``final_layer_norm``: :class:`torch.nn.LayerNorm` (eps=1e-6)。

B) 傅里叶头部（ISTFTHead）
   - ``out``: :class:`torch.nn.Linear` ``(dim -> n_fft+2)``，投影为幅度与
     相位谱；
   - ``magnitude = exp(mag)`` 并 ``clip(max=1e2)``，``phase`` 直接作为弧度；
   - 用 ``cos/sin`` 构造复数谱（与官方一致，避免 ``torch.polar`` 在 CUDA
     float16 下不支持的问题）；
   - ``istft``: 自实现的 :class:`_ISTFT`，``"same"`` padding 下用
     ``torch.fft.irfft`` + ``fold`` 重建波形。

state_dict key 完全匹配官方 vocos 包
-----------------------------------
* ``backbone.embed.{weight,bias}``
* ``backbone.norm.{weight,bias}``
* ``backbone.convnext.{i}.{dwconv,norm,pwconv1,pwconv2}.{weight,bias}``
* ``backbone.convnext.{i}.gamma``
* ``backbone.final_layer_norm.{weight,bias}``
* ``head.out.{weight,bias}``
* ``head.istft.window`` (buffer)

因此 :meth:`load_weights` 可直接用 ``load_state_dict(strict=False)``
加载官方 ``Vocos.safetensors``，无需前缀剥离。

设计要点
--------
* ``torch`` / ``safetensors`` 采用惰性导入：模块顶层不导入这些重依赖，
  真正的 ``nn.Module`` 子类在首次 :meth:`load_weights` 时通过
  :func:`_get_vocos_class` 惰性构建，使本模块在未安装这些依赖时仍可被
  导入与实例化。实现中不 import 官方 ``vocos`` 包。
* :class:`VocosVocoder` 继承 :class:`Vocoder` 抽象基类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现。
* :meth:`decode` / :meth:`decode_chunk` 返回 ``(waveform, sample_rate)``
  元组，兼容 :meth:`TTSBackend._coerce_vocoder_output`。
* ``features`` / ``mel`` 等参数类型用 :data:`~typing.Any` 标注。
"""

from __future__ import annotations

import os
from typing import Any

from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder

__all__ = ["VocosVocoder"]


# 内部缓存的自实现 nn.Module 子类（惰性创建）
_VocosImplClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_vocos_class() -> Any:
    """惰性创建并返回 Vocos 自实现的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义 :class:`_ISTFT`、:class:`_ISTFTHead`、
    :class:`_ConvNeXtBlock`、:class:`_VocosBackbone` 与 :class:`_VocosImpl`，
    随后缓存到全局变量 :data:`_VocosImplClass`。返回类的 ``state_dict`` key
    与官方 ``vocos`` 包完全一致（``backbone.*`` / ``head.*``），因此可直接
    ``load_state_dict(strict=False)`` 加载官方 ``Vocos.safetensors``。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_VocosImpl``。
    """
    global _VocosImplClass
    if _VocosImplClass is None:
        import torch
        import torch.nn as nn

        class _ISTFT(nn.Module):
            """自实现逆短时傅里叶变换（不依赖官方 vocos 包）。

            ``"same"`` padding 模式下用 :func:`torch.fft.irfft` 得到时域帧
            再用 :func:`torch.nn.functional.fold` 重叠相加，最后除以窗能量
            包络；``"center"`` 模式直接调用 :func:`torch.istft`。
            """

            def __init__(
                self,
                n_fft: int,
                hop_length: int,
                win_length: int,
                padding: str = "same",
            ) -> None:
                super().__init__()
                self.padding = padding
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.register_buffer(
                    "window", torch.hann_window(win_length)
                )

            def forward(self, spec: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                spec : torch.Tensor
                    复数谱 ``[batch, n_bins, frames]``。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, samples]``。
                """
                # torch.fft 不支持 float16 / complex32：先上采到 float32
                if spec.real.dtype == torch.float16:
                    spec = spec.to(torch.complex64)
                win = self.window.to(spec.real.dtype)

                if self.padding == "center":
                    return torch.istft(
                        spec,
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        win,
                        center=True,
                    )

                # "same" padding
                pad = (self.win_length - self.hop_length) // 2
                _, _, T = spec.shape
                ifft = torch.fft.irfft(
                    spec, self.n_fft, dim=1, norm="backward"
                )
                ifft = ifft * win[None, :, None]
                output_size = (T - 1) * self.hop_length + self.win_length
                y = torch.nn.functional.fold(
                    ifft,
                    output_size=(1, output_size),
                    kernel_size=(1, self.win_length),
                    stride=(1, self.hop_length),
                )[:, 0, 0, pad:-pad]
                window_sq = (
                    win.square().expand(1, T, -1).transpose(1, 2)
                )
                window_envelope = (
                    torch.nn.functional.fold(
                        window_sq,
                        output_size=(1, output_size),
                        kernel_size=(1, self.win_length),
                        stride=(1, self.hop_length),
                    ).squeeze()[pad:-pad]
                )
                y = y / window_envelope
                return y

        class _ISTFTHead(nn.Module):
            """傅里叶头部：隐藏特征 -> 幅度与相位 -> iSTFT -> 波形。

            与官方 ``ISTFTHead`` 完全一致：``out`` 线性层输出 ``n_fft+2`` 维，
            前半为对数幅度、后半为相位（弧度），用 ``cos/sin`` 构造复数谱。
            """

            def __init__(
                self,
                dim: int,
                n_fft: int,
                hop_length: int,
                padding: str = "same",
            ) -> None:
                super().__init__()
                out_dim = n_fft + 2
                self.out = nn.Linear(dim, out_dim)
                self.istft = _ISTFT(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    padding=padding,
                )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, frames, dim]``。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, samples]``。
                """
                x = self.out(x).transpose(1, 2)  # [B, n_fft+2, T]
                mag, p = x.chunk(2, dim=1)
                mag = torch.exp(mag)
                mag = torch.clip(mag, max=1e2)
                real = torch.cos(p)
                imag = torch.sin(p)
                # 用 cos/sin 构造复数谱（与官方一致；torch.polar 在 CUDA
                # float16 下不支持）
                S = mag * (real + 1j * imag)
                audio = self.istft(S)  # [B, samples]
                return audio

        class _ConvNeXtBlock(nn.Module):
            """ConvNeXt 主干块（与官方 ``vocos.modules.ConvNeXtBlock`` 一致）。

            Depthwise Conv1D (kernel=7, padding=3, groups=dim) ->
            LayerNorm(eps=1e-6) -> Pointwise Linear(dim->intermediate_dim)
            -> GELU -> Pointwise Linear(intermediate_dim->dim) -> gamma
            缩放 -> 残差连接。
            """

            def __init__(
                self,
                dim: int,
                intermediate_dim: int,
                layer_scale_init_value: float,
                adanorm_num_embeddings: int | None = None,
            ) -> None:
                super().__init__()
                self.dwconv = nn.Conv1d(
                    dim, dim, kernel_size=7, padding=3, groups=dim
                )
                self.norm = nn.LayerNorm(dim, eps=1e-6)
                self.pwconv1 = nn.Linear(dim, intermediate_dim)
                self.act = nn.GELU()
                self.pwconv2 = nn.Linear(intermediate_dim, dim)
                self.gamma = (
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
                x = self.dwconv(x)
                x = x.transpose(1, 2)  # [B, T, C]
                x = self.norm(x)
                x = self.pwconv1(x)
                x = self.act(x)
                x = self.pwconv2(x)
                if self.gamma is not None:
                    x = self.gamma * x
                x = x.transpose(1, 2)  # [B, C, T]
                x = residual + x
                return x

        class _VocosBackbone(nn.Module):
            """Vocos 主干网络（与官方 ``vocos.models.VocosBackbone`` 一致）。"""

            def __init__(
                self,
                input_channels: int,
                dim: int,
                intermediate_dim: int,
                num_layers: int,
                layer_scale_init_value: float | None = None,
                adanorm_num_embeddings: int | None = None,
            ) -> None:
                super().__init__()
                self.embed = nn.Conv1d(
                    input_channels, dim, kernel_size=7, padding=3
                )
                self.norm = nn.LayerNorm(dim, eps=1e-6)
                layer_scale_init_value = (
                    layer_scale_init_value or 1 / num_layers
                )
                self.convnext = nn.ModuleList(
                    [
                        _ConvNeXtBlock(
                            dim,
                            intermediate_dim,
                            layer_scale_init_value,
                            adanorm_num_embeddings,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, input_channels, frames]``。

                Returns
                -------
                torch.Tensor
                    ``[batch, frames, dim]``。
                """
                x = self.embed(x)
                x = self.norm(x.transpose(1, 2))
                x = x.transpose(1, 2)
                for conv_block in self.convnext:
                    x = conv_block(x)
                x = self.final_layer_norm(x.transpose(1, 2))  # [B, T, H]
                return x

        class _VocosImpl(nn.Module):
            """Vocos 声码器自实现（``nn.Module`` 子类）。

            顶层包含 ``backbone`` (:class:`_VocosBackbone`) 与 ``head``
            (:class:`_ISTFTHead`)，state_dict key 与官方 vocos 包完全一致。
            """

            def __init__(
                self,
                input_channels: int = 100,
                dim: int = 512,
                intermediate_dim: int = 1536,
                num_layers: int = 8,
                n_fft: int = 1024,
                hop_length: int = 256,
                padding: str = "same",
            ) -> None:
                super().__init__()
                self.backbone = _VocosBackbone(
                    input_channels, dim, intermediate_dim, num_layers
                )
                self.head = _ISTFTHead(
                    dim, n_fft, hop_length, padding=padding
                )

            def forward(self, mel: Any) -> Any:
                """前向计算：mel -> backbone -> head -> 波形。

                Parameters
                ----------
                mel : torch.Tensor
                    ``[batch, input_channels, frames]``。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, samples]``。
                """
                x = self.backbone(mel)  # [B, T, H]
                audio = self.head(x)  # [B, samples]
                return audio

            def decode(self, mel: Any) -> Any:
                """与 :meth:`forward` 等价（mel -> 波形）。"""
                return self.forward(mel)

        _VocosImplClass = _VocosImpl
    return _VocosImplClass


class VocosVocoder(Vocoder):
    """Vocos 声码器。

    继承 :class:`Vocoder` 抽象基类，将 mel 频谱解码为音频波形。

    内部 ``nn.Module`` 实现（:class:`_VocosImpl`）的 state_dict key 与官方
    ``vocos`` 包完全一致，因此 :meth:`load_weights` 可直接用
    ``load_state_dict(strict=False)`` 加载 ChatTTS 的 ``Vocos.safetensors``，
    无需前缀剥离，也不依赖官方 ``vocos`` 包。

    默认配置匹配 ChatTTS 实际使用的 Vocos 权重：input_channels(mel)=100、
    dim=512、intermediate_dim=1536、num_layers=8、n_fft=1024、
    hop_length=256、padding="same"。

    Parameters
    ----------
    model_path : str
        Vocos 模型权重路径（仅记录，实际加载以 :meth:`load_weights` 的
        ``weights_path`` 参数为准）。
    n_fft : int
        FFT 点数，默认 ``1024``。
    hop_length : int
        帧移，默认 ``256``。
    win_length : int
        窗长，默认 ``1024``（官方 ISTFT 取 ``win_length = n_fft``）。
    n_mels : int
        mel 频谱维度（backbone 的 input_channels），默认 ``100``。
    sample_rate : int
        输出采样率，默认 ``24000``。
    hidden_size : int
        隐藏维度（backbone 的 dim），默认 ``512``。
    num_layers : int
        ConvNeXt 块数量，默认 ``8``。
    intermediate_dim : int
        ConvNeXt 块 pointwise 中间维度，默认 ``1536``。
    dilation_rates : list[int] | None
        保留以兼容旧接口；官方结构不使用膨胀卷积，该参数被忽略。

    Attributes
    ----------
    vocoder_type : str
        固定为 ``"vocos"``。
    input_type : str
        固定为 ``"mel"``。
    """

    vocoder_type: str = "vocos"
    input_type: str = "mel"

    def __init__(
        self,
        model_path: str,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        sample_rate: int = 24000,
        hidden_size: int = 512,
        num_layers: int = 8,
        intermediate_dim: int = 1536,
        dilation_rates: list[int] | None = None,
    ) -> None:
        # 注意：此处不导入 torch；内部 nn.Module 在 load_weights 时创建
        self.model_path = model_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        # 官方结构不使用膨胀卷积；保留以兼容旧接口
        self.dilation_rates: list[int] | None = dilation_rates

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

        # 流式重叠状态
        self._overlap_frames: int = 4
        self._mel_buffer: Any = None

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
            "VocosVocoder is not loaded. Call load_weights() before calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放（实现 Vocoder 抽象方法）
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载 Vocos 权重。

        加载步骤：

        1. 解析 dtype 字符串为 torch dtype，无 GPU 时设备降级为 CPU；
        2. 通过 :func:`_get_vocos_class` 构建自实现 ``_VocosImpl``（key 与
           官方 vocos 包一致）；
        3. 读取权重为 state_dict，直接 ``load_state_dict(strict=False)``
           加载（无需前缀剥离）；
        4. 移动到目标 device / dtype，切换为 eval。

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

        cls = _get_vocos_class()
        impl = cls(
            input_channels=self.n_mels,
            dim=self.hidden_size,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding="same",
        )
        state_dict = self._load_state_dict(weights_path)
        if state_dict:
            impl.load_state_dict(state_dict, strict=False)
            import logging
            logger = logging.getLogger(__name__)
            model_keys = set(impl.state_dict().keys())
            file_keys = set(state_dict.keys())
            matched = model_keys & file_keys
            missing = model_keys - file_keys
            unexpected = file_keys - model_keys
            logger.info(
                "Vocos 权重加载: 模型 %d key, 文件 %d key, "
                "匹配 %d, 缺失 %d, 多余 %d",
                len(model_keys), len(file_keys),
                len(matched), len(missing), len(unexpected),
            )
            if missing:
                logger.warning("Vocos 缺失 key (前10): %s",
                               sorted(missing)[:10])
            if unexpected:
                logger.warning("Vocos 多余 key (前10): %s",
                               sorted(unexpected)[:10])
        impl = impl.to(device=resolved, dtype=torch_dtype)
        impl.eval()
        self._impl = impl
        self._is_loaded = True

    def unload_weights(self) -> None:
        """释放权重：将内部模型移至 CPU 并清空 CUDA 缓存。"""
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
        self._mel_buffer = None

    # ------------------------------------------------------------------
    # 解码（实现 Vocoder 抽象方法）
    # ------------------------------------------------------------------
    def decode(self, features: Any) -> tuple[Any, int]:
        """将 mel 频谱解码为波形。

        Parameters
        ----------
        features : torch.Tensor
            mel 频谱 ``[batch, mel_bins, frames]`` 或 ``[mel_bins, frames]``。

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(waveform, sample_rate)``，waveform 形状 ``[batch, samples]``
            或 ``[samples]``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "VocosVocoder is not loaded. Call load_weights() before decode()."
            )
        import torch

        mel = features
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        with torch.no_grad():
            waveform = self._run_model(mel)

        if squeeze:
            waveform = waveform.squeeze(0)
        return (waveform, self.sample_rate)

    def decode_chunk(self, features: Any) -> tuple[Any, int]:
        """流式解码：重叠处理避免边界伪影。

        每次调用时，将上一块末尾的若干 mel 帧作为重叠上下文拼接到当前块
        前，解码后丢弃重叠部分对应的样本，仅输出当前块的新样本。使用
        :meth:`reset_stream` 重置流式状态。

        Parameters
        ----------
        features : torch.Tensor
            一小块 mel 频谱。

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(waveform, sample_rate)``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "VocosVocoder is not loaded. Call load_weights() before decode_chunk()."
            )
        import torch

        mel = features
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        overlap = self._overlap_frames
        prepended = 0
        if self._mel_buffer is not None and overlap > 0:
            mel = torch.cat([self._mel_buffer, mel], dim=-1)
            prepended = overlap

        with torch.no_grad():
            waveform = self._run_model(mel)

        # 跳过与重叠 mel 帧对应的样本（"same" padding 下每帧恰好对应
        # hop_length 个样本）
        if prepended > 0:
            skip = prepended * self.hop_length
            if skip < waveform.shape[-1]:
                waveform = waveform[..., skip:]

        # 更新 mel 缓冲区：保留最近 overlap 帧
        if overlap > 0 and mel.shape[-1] >= overlap:
            self._mel_buffer = mel[..., -overlap:].detach()

        if squeeze:
            waveform = waveform.squeeze(0)
        return (waveform, self.sample_rate)

    def reset_stream(self) -> None:
        """重置流式重叠缓冲区。"""
        self._mel_buffer = None

    # ------------------------------------------------------------------
    # mel 滤波器组
    # ------------------------------------------------------------------
    def get_mel_basis(
        self, n_fft: int, sample_rate: int, n_mels: int
    ) -> Any:
        """返回 mel 滤波器组矩阵。

        使用 HTK 式 mel 频率刻度构造三角滤波器组。

        Parameters
        ----------
        n_fft : int
            FFT 点数。
        sample_rate : int
            采样率。
        n_mels : int
            mel 滤波器数量。

        Returns
        -------
        torch.Tensor
            mel 滤波器组，形状 ``[n_mels, n_fft // 2 + 1]``。
        """
        import numpy as np
        import torch

        n_bins = n_fft // 2 + 1
        f_min = 0.0
        f_max = sample_rate / 2.0

        def hz_to_mel(f: float) -> float:
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

        fb = np.zeros((n_mels, n_bins), dtype=np.float32)
        for m in range(n_mels):
            left = hz_points[m]
            center = hz_points[m + 1]
            right = hz_points[m + 2]
            for k in range(n_bins):
                f = fft_freqs[k]
                if f < left or f > right:
                    continue
                if f <= center:
                    if center > left:
                        fb[m, k] = (f - left) / (center - left)
                else:
                    if right > center:
                        fb[m, k] = (right - f) / (right - center)
        return torch.from_numpy(fb)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _run_model(self, mel: Any) -> Any:
        """运行内部 ``nn.Module`` 实现（forward/decode）。"""
        if self._impl is None:
            raise RuntimeError("No vocoder model available.")
        return self._impl(mel)

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
                "vocoder.safetensors",
                "vocos.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in ("vocoder.bin", "vocos.bin"):
                    fpath = os.path.join(weights_path, fname)
                    if os.path.isfile(fpath):
                        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                        state_dict = _unwrap_ckpt(ckpt)
                        break
        return state_dict
