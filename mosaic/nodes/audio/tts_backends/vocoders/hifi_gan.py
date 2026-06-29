# mosaic/nodes/audio/tts_backends/vocoders/hifi_gan.py
"""HiFi-GAN 声码器。

Layer 3: 声码器层。将 mel 频谱解码为音频波形。

本模块实现 HiFi-GAN Generator，将 mel 频谱 ``[batch, mel_bins, frames]``
解码为波形 ``[batch, samples]``。HiFi-GAN 通过多上采样 + 多感受野融合
残差块（Multi-Receptive Field Fusion, MRF）的纯卷积生成器直接从 mel
重建波形，相比 iSTFT 类声码器（如 Vocos）计算更重但质量更高。

架构
----
::

    mel → Conv1D(upsample_initial_channel)
        → UpsampleBlocks
        → Conv1D(1)
        → waveform

每个 UpsampleBlock::

    ConvTranspose1D(channels, channels//2,
                    kernel=stride*2, stride=upsample_rate)
    MRF(channels//2, resblock_kernel_sizes, resblock_dilation_sizes)
    channels = channels // 2

MRF (Multi-ReceptiveFieldFusion)::

    多个不同核大小和膨胀率的残差块并行，结果求和
    每个残差块:
        Conv1D(channels, channels, kernel=k, dilation=d)
        LeakyReLU(0.1)
        Conv1D(channels, channels, kernel=k, dilation=1)
        LeakyReLU(0.1)
        残差连接

设计要点
--------
* ``torch`` / ``safetensors`` 采用惰性导入：模块顶层不导入这些重依赖，
  真正的 ``nn.Module`` 子类在首次 :meth:`load_weights` 时通过
  :func:`_get_hifi_gan_class` 惰性构建。
* :class:`HiFiGanVocoder` 继承 :class:`Vocoder` 抽象基类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现。
* :meth:`decode` / :meth:`decode_chunk` 返回 ``(waveform, sample_rate)``
  元组，兼容 :meth:`TTSBackend._coerce_vocoder_output`。
* 使用 ``LeakyReLU(0.1)`` 作为激活函数（HiFi-GAN 标准）。
* ``features`` / ``mel`` 等参数类型用 :data:`~typing.Any` 标注。
"""

from __future__ import annotations

import os
from typing import Any

from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder

__all__ = ["HiFiGanVocoder"]


# 内部缓存的 nn.Module 子类（惰性创建）
_HiFiGanImplClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_hifi_gan_class() -> Any:
    """惰性创建并返回 HiFi-GAN Generator 的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并定义 :class:`_ResBlock` 与
    :class:`_HiFiGanGenerator`，随后缓存到全局变量
    :data:`_HiFiGanImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_HiFiGanGenerator``。
    """
    global _HiFiGanImplClass
    if _HiFiGanImplClass is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _ResBlock(nn.Module):
            """HiFi-GAN 残差块。

            对每组膨胀率 ``d`` 依次执行：

            ``LeakyReLU(0.1) → Conv1D(k, dilation=d) → LeakyReLU(0.1)
            → Conv1D(k, dilation=1) → 残差连接``

            多组膨胀率串行累加残差，逐步扩大感受野。
            """

            def __init__(
                self,
                channels: int,
                kernel_size: int,
                dilation_rates: list[int],
            ) -> None:
                super().__init__()
                self.convs1 = nn.ModuleList()
                self.convs2 = nn.ModuleList()
                for d in dilation_rates:
                    self.convs1.append(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            dilation=d,
                            padding=(kernel_size - 1) * d // 2,
                        )
                    )
                    self.convs2.append(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            dilation=1,
                            padding=(kernel_size - 1) // 2,
                        )
                    )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, channels, frames]``。

                Returns
                -------
                torch.Tensor
                    同形状 ``[batch, channels, frames]``。
                """
                for c1, c2 in zip(self.convs1, self.convs2):
                    xt = F.leaky_relu(x, 0.1)
                    xt = c1(xt)
                    xt = F.leaky_relu(xt, 0.1)
                    xt = c2(xt)
                    x = xt + x
                return x

        class _HiFiGanGenerator(nn.Module):
            """HiFi-GAN Generator 真实实现（``nn.Module`` 子类）。

            将 mel 频谱解码为音频波形。结构为：预卷积 → 多上采样块
            （每个含 ConvTranspose1D + MRF）→ 后卷积 → tanh。
            """

            def __init__(
                self,
                n_mels: int = 80,
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

                self.n_mels = n_mels
                self.upsample_rates: list[int] = list(upsample_rates)
                self.upsample_initial_channel = upsample_initial_channel
                self.resblock_kernel_sizes: list[int] = list(
                    resblock_kernel_sizes
                )
                self.resblock_dilation_sizes: list[list[int]] = [
                    list(d) for d in resblock_dilation_sizes
                ]
                self.num_resblocks = len(self.resblock_kernel_sizes)

                # 预卷积：mel -> upsample_initial_channel
                self.conv_pre = nn.Conv1d(
                    n_mels,
                    upsample_initial_channel,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                )

                # 上采样块 + MRF
                self.ups = nn.ModuleList()
                self.resblocks = nn.ModuleList()

                channels = upsample_initial_channel
                for i, upsample_rate in enumerate(self.upsample_rates):
                    # ConvTranspose1D：channels -> channels // 2
                    self.ups.append(
                        nn.ConvTranspose1d(
                            channels,
                            channels // 2,
                            kernel_size=upsample_rate * 2,
                            stride=upsample_rate,
                            padding=upsample_rate // 2,
                        )
                    )
                    channels = channels // 2
                    # MRF：每个 (kernel, dilation) 对应一个残差块
                    for k, d in zip(
                        self.resblock_kernel_sizes,
                        self.resblock_dilation_sizes,
                    ):
                        self.resblocks.append(_ResBlock(channels, k, d))

                # 后卷积：channels -> 1
                self.conv_post = nn.Conv1d(
                    channels, 1, kernel_size=7, stride=1, padding=3
                )

            def forward(self, x: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    mel 频谱 ``[batch, n_mels, frames]``。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, 1, samples]``。
                """
                x = self.conv_pre(x)            # [B, C, T]
                for i, up in enumerate(self.ups):
                    x = F.leaky_relu(x, 0.1)
                    x = up(x)                    # 上采样
                    # MRF：多个残差块并行求和后取平均
                    xs: Any = None
                    for j in range(self.num_resblocks):
                        resblock = self.resblocks[
                            i * self.num_resblocks + j
                        ]
                        out = resblock(x)
                        xs = out if xs is None else xs + out
                    x = xs / self.num_resblocks
                x = F.leaky_relu(x, 0.1)
                x = self.conv_post(x)           # [B, 1, samples]
                x = torch.tanh(x)
                return x

        _HiFiGanImplClass = _HiFiGanGenerator
    return _HiFiGanImplClass


class HiFiGanVocoder(Vocoder):
    """HiFi-GAN 声码器。

    继承 :class:`Vocoder` 抽象基类，将 mel 频谱解码为音频波形。

    权重加载策略（:meth:`load_weights`）：

    1. 惰性创建 HiFi-GAN Generator ``nn.Module`` 子类实例；
    2. 从 safetensors / pytorch checkpoint 载入权重（``strict=False``），
       **只加载 Generator 权重**，过滤掉 Discriminator 相关 key；
    3. 剥离 ``generator.`` / ``hifi_gan.`` / ``vocoder.`` 等前缀；
    4. 移动到目标 device / dtype，切换为 eval。

    Parameters
    ----------
    model_path : str
        HiFi-GAN 模型权重路径（仅记录，实际加载以 :meth:`load_weights`
        的 ``weights_path`` 参数为准）。
    sample_rate : int
        输出采样率，默认 ``22050``。
    n_fft : int
        FFT 点数，默认 ``1024``。
    hop_length : int
        帧移，默认 ``256``。
    win_length : int
        窗长，默认 ``1024``。
    n_mels : int
        mel 频谱维度，默认 ``80``。
    upsample_rates : list[int] | None
        上采样倍率，默认 ``[8, 8, 2, 2]``。
    upsample_initial_channel : int
        初始通道数，默认 ``512``。
    resblock_kernel_sizes : list[int] | None
        残差块核大小，默认 ``[3, 7, 11]``。
    resblock_dilation_sizes : list[list[int]] | None
        残差块膨胀率，默认 ``[[1,3,5],[1,3,5],[1,3,5]]``。

    Attributes
    ----------
    vocoder_type : str
        固定为 ``"hifi_gan"``。
    input_type : str
        固定为 ``"mel"``。
    """

    vocoder_type: str = "hifi_gan"
    input_type: str = "mel"

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        upsample_rates: list[int] | None = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
    ) -> None:
        # 注意：此处不导入 torch；内部 nn.Module 在 load_weights 时创建
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.upsample_rates: list[int] = (
            upsample_rates if upsample_rates is not None else [8, 8, 2, 2]
        )
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes: list[int] = (
            resblock_kernel_sizes
            if resblock_kernel_sizes is not None
            else [3, 7, 11]
        )
        self.resblock_dilation_sizes: list[list[int]] = (
            resblock_dilation_sizes
            if resblock_dilation_sizes is not None
            else [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

        # 每帧 mel 对应的采样点数 = ∏(upsample_rates)
        self._samples_per_frame: int = 1
        for r in self.upsample_rates:
            self._samples_per_frame *= r

        # 流式重叠状态
        self._overlap_frames: int = 8
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
            "HiFiGanVocoder is not loaded. Call load_weights() before "
            "calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放（实现 Vocoder 抽象方法）
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载 HiFi-GAN Generator 权重。

        加载步骤：

        1. 解析 dtype 字符串为 torch dtype，无 GPU 时设备降级为 CPU；
        2. 惰性创建 Generator ``nn.Module`` 实例；
        3. 读取权重（safetensors 优先，回退到 .pt/.pth/.bin）；
        4. **过滤掉 Discriminator 权重**，剥离 ``generator.`` 等前缀后
           ``strict=False`` 载入；
        5. 移动到目标 device / dtype，切换为 eval。

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

        # 惰性创建 Generator 实例
        cls = _get_hifi_gan_class()
        impl = cls(
            n_mels=self.n_mels,
            upsample_rates=self.upsample_rates,
            upsample_initial_channel=self.upsample_initial_channel,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
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
        """释放权重：将内部模型移至 CPU 并清空 CUDA 缓存。"""
        try:
            import torch

            if self._impl is not None:
                try:
                    self._impl.to("cpu")
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
            ``(waveform, sample_rate)``，waveform 形状
            ``[batch, samples]`` 或 ``[samples]``。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "HiFiGanVocoder is not loaded. Call load_weights() "
                "before decode()."
            )
        import torch

        mel = features
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        with torch.no_grad():
            waveform = self._impl(mel)          # [B, 1, samples]
        waveform = waveform.squeeze(1)          # [B, samples]

        if squeeze:
            waveform = waveform.squeeze(0)      # [samples]
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
                "HiFiGanVocoder is not loaded. Call load_weights() "
                "before decode_chunk()."
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
            waveform = self._impl(mel)          # [B, 1, samples]
        waveform = waveform.squeeze(1)          # [B, samples]

        # 跳过与重叠 mel 帧对应的样本
        if prepended > 0:
            skip = prepended * self._samples_per_frame
            if skip < waveform.shape[-1]:
                waveform = waveform[..., skip:]

        # 更新 mel 缓冲区：保留最近 overlap 帧
        if overlap > 0 and mel.shape[-1] >= overlap:
            self._mel_buffer = mel[..., -overlap:].detach()

        if squeeze:
            waveform = waveform.squeeze(0)      # [samples]
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
                ckpt = torch.load(weights_path, map_location="cpu")
                state_dict = _unwrap_ckpt(ckpt)
        elif os.path.isdir(weights_path):
            for fname in (
                "generator.safetensors",
                "hifi_gan.safetensors",
                "vocoder.safetensors",
                "g_0.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in (
                    "generator.bin",
                    "hifi_gan.bin",
                    "vocoder.bin",
                    "g_0.bin",
                ):
                    fpath = os.path.join(weights_path, fname)
                    if os.path.isfile(fpath):
                        ckpt = torch.load(fpath, map_location="cpu")
                        state_dict = _unwrap_ckpt(ckpt)
                        break
        return state_dict

    @staticmethod
    def _filter_and_strip(
        state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """过滤掉判别器权重，剥离生成器前缀。

        HiFi-GAN 完整 checkpoint 通常同时包含 Generator 和 Discriminator
        权重。本方法：

        1. 丢弃以 ``discriminator.`` / ``disc.`` / ``mpd.`` / ``msd.``
           开头的 key（判别器）；
        2. 剥离 ``generator.`` / ``hifi_gan.`` / ``vocoder.`` 等前缀，
           使剩余 key 与 Generator 模块结构匹配。
        """
        disc_prefixes = ("discriminator.", "disc.", "mpd.", "msd.")
        gen_prefixes = ("generator.", "hifi_gan.", "vocoder.")

        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 跳过判别器权重
            if any(key.startswith(p) for p in disc_prefixes):
                continue
            # 剥离生成器前缀
            new_key = key
            for p in gen_prefixes:
                if key.startswith(p):
                    new_key = key[len(p):]
                    break
            filtered[new_key] = value
        return filtered
