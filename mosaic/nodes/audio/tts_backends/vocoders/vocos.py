# mosaic/nodes/audio/tts_backends/vocoders/vocos.py
"""Vocos 声码器。

Layer 3: 声码器层。将 mel 频谱解码为音频波形。

本模块实现基于 iSTFT 的 Vocos 声码器，将 mel 频谱 ``[batch, mel_bins,
frames]`` 解码为波形 ``[batch, samples]``。Vocos 通过预测幅度与相位谱再
做逆短时傅里叶变换（iSTFT）重建波形，相比 HiFi-GAN 计算更轻、质量更稳。

架构
----
A) 特征提取器
   - :class:`torch.nn.Conv1d` ``(n_mels -> hidden_size)``，将 mel 投影到
     隐藏空间。

B) 主干网络（ConvNeXt 块）
   - 每层：Depthwise Conv1D（膨胀卷积）-> LayerNorm -> Pointwise Conv(1x1)
     -> GELU -> 残差连接；
   - 主干末端附加一层 :class:`torch.nn.LayerNorm`。

C) 傅里叶头部（iSTFTHead）
   - 隐藏特征 -> 线性投影为幅度与相位谱；
   - ``magnitude = exp(mag_raw)``，``phase = tanh(phase_raw) * pi``；
   - 组成复数谱后调用 :func:`torch.istft` 重建波形。

设计要点
--------
* ``torch`` / ``safetensors`` / ``vocos`` 均为惰性导入：模块顶层不导入
  这些重依赖，真正的 ``nn.Module`` 子类在首次 :meth:`load_weights` 时通过
  :func:`_get_vocos_class` 惰性构建，使本模块在未安装这些依赖时仍可被
  导入与实例化。
* :class:`VocosVocoder` 继承 :class:`Vocoder` 抽象基类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现
  （自实现版本或官方 ``vocos.Vocos``）。
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

    首次调用时导入 ``torch`` 并定义 :class:`_ConvNeXtBlock`、
    :class:`_ISTFTHead` 与 :class:`_VocosImpl`，随后缓存到全局变量
    :data:`_VocosImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_VocosImpl``。
    """
    global _VocosImplClass
    if _VocosImplClass is None:
        import torch
        import torch.nn as nn

        class _ConvNeXtBlock(nn.Module):
            """ConvNeXt 主干块。

            结构：Depthwise Conv1D（膨胀）-> LayerNorm -> Pointwise Conv(1x1)
            -> GELU -> 残差连接。所有运算保持 ``hidden_size`` 维度不变。
            """

            def __init__(
                self,
                hidden_size: int,
                dilation_rate: int = 1,
                kernel_size: int = 7,
            ) -> None:
                super().__init__()
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
                x = self.dwconv(x)
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
                x = self.pwconv(x)
                x = self.act(x)
                return residual + x

        class _ISTFTHead(nn.Module):
            """傅里叶头部：隐藏特征 -> 幅度与相位 -> iSTFT -> 波形。"""

            def __init__(
                self,
                hidden_size: int,
                n_fft: int,
                hop_length: int,
                win_length: int,
            ) -> None:
                super().__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                n_bins = n_fft // 2 + 1
                # 同时投影幅度与相位（各 n_bins 维）
                self.proj = nn.Linear(hidden_size, n_bins * 2)

            def forward(self, x: Any, window: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                x : torch.Tensor
                    ``[batch, frames, hidden_size]``。
                window : torch.Tensor
                    分析窗（长度 ``win_length``）。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, samples]``。
                """
                import torch

                x = self.proj(x)                       # [B, T, n_bins*2]
                mag_raw, phase_raw = torch.chunk(x, 2, dim=-1)
                mag = torch.exp(mag_raw)               # 正幅度
                phase = torch.tanh(phase_raw) * torch.pi
                # -> [B, n_bins, T]
                mag = mag.transpose(1, 2)
                phase = phase.transpose(1, 2)
                # torch.polar 在 CUDA 上不支持 float16，先转 float32
                mag = mag.float()
                phase = phase.float()
                complex_spec = torch.polar(mag, phase)  # 复数谱 [B, n_bins, T]
                waveform = torch.istft(
                    complex_spec,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=True,
                    return_complex=False,
                )
                return waveform  # [B, samples]

        class _VocosImpl(nn.Module):
            """Vocos 声码器自实现（``nn.Module`` 子类）。"""

            def __init__(
                self,
                n_mels: int = 80,
                hidden_size: int = 384,
                num_layers: int = 6,
                n_fft: int = 1024,
                hop_length: int = 256,
                win_length: int = 1024,
                dilation_rates: list[int] | None = None,
            ) -> None:
                super().__init__()
                if dilation_rates is None:
                    dilation_rates = [1, 2, 8, 2, 1, 2]
                self.n_mels = n_mels
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.dilation_rates: list[int] = list(dilation_rates)
                self._kernel_size = 7

                # A) 特征提取器
                self.input_conv = nn.Conv1d(
                    n_mels,
                    hidden_size,
                    kernel_size=self._kernel_size,
                    padding=self._kernel_size // 2,
                )
                # B) 主干网络
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
                self.final_norm = nn.LayerNorm(hidden_size)
                # C) 傅里叶头部
                self.istft_head = _ISTFTHead(
                    hidden_size, n_fft, hop_length, win_length
                )
                # iSTFT 分析窗（随模型一起移动设备）
                self.register_buffer(
                    "window", torch.hann_window(win_length)
                )

            def forward(self, mel: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                mel : torch.Tensor
                    ``[batch, n_mels, frames]``。

                Returns
                -------
                torch.Tensor
                    波形 ``[batch, samples]``。
                """
                x = self.input_conv(mel)              # [B, H, T]
                for block in self.blocks:
                    x = block(x)
                x = x.transpose(1, 2)                 # [B, T, H]
                x = self.final_norm(x)
                waveform = self.istft_head(x, self.window)  # [B, samples]
                return waveform

        _VocosImplClass = _VocosImpl
    return _VocosImplClass


class VocosVocoder(Vocoder):
    """Vocos 声码器。

    继承 :class:`Vocoder` 抽象基类，将 mel 频谱解码为音频波形。

    权重加载策略（:meth:`load_weights`）：

    1. 优先尝试 ``from vocos import Vocos``，若官方包可用且 ``weights_path``
       为可被 ``Vocos.from_pretrained`` 加载的目录，则包装官方模型；
    2. 否则使用本模块自实现的 iSTFT 声码器，从 safetensors / pytorch
       checkpoint 载入权重（``strict=False``）。

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
        窗长，默认 ``1024``。
    n_mels : int
        mel 频谱维度，默认 ``80``。
    sample_rate : int
        输出采样率，默认 ``24000``。

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
        n_mels: int = 80,
        sample_rate: int = 24000,
        hidden_size: int = 384,
        num_layers: int = 6,
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
        self.dilation_rates: list[int] | None = dilation_rates

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._official: Any = None
        self._use_official: bool = False
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
        """将未在本类找到的属性转发给内部 ``nn.Module`` 实现。

        优先转发给自实现版本，其次官方 ``vocos.Vocos``。
        """
        impl = self.__dict__.get("_impl")
        if impl is not None:
            return getattr(impl, name)
        official = self.__dict__.get("_official")
        if official is not None:
            return getattr(official, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """转发调用，触发内部 ``nn.Module.__call__``。"""
        if self._use_official and self._official is not None:
            return self._official(*args, **kwargs)
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
        2. 尝试官方 ``vocos`` 包：若可用且 ``weights_path`` 为目录，则用
           ``Vocos.from_pretrained`` 加载并包装；
        3. 否则使用自实现版本，从 safetensors / pytorch checkpoint 载入
           权重（``strict=False``），剥离 ``vocos.`` / ``decoder.`` 前缀；
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

        # 优先尝试官方 vocos 包
        official = self._try_load_official(
            weights_path, resolved, torch_dtype
        )
        if official is not None:
            self._official = official
            self._use_official = True
            self._impl = None
            self._is_loaded = True
            return

        # 回退到自实现版本
        cls = _get_vocos_class()
        impl = cls(
            n_mels=self.n_mels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            dilation_rates=self.dilation_rates,
        )
        state_dict = self._load_state_dict(weights_path)
        if state_dict:
            state_dict = self._strip_prefix(state_dict)
            impl.load_state_dict(state_dict, strict=False)
        impl = impl.to(device=resolved, dtype=torch_dtype)
        impl.eval()
        self._impl = impl
        self._official = None
        self._use_official = False
        self._is_loaded = True

    def unload_weights(self) -> None:
        """释放权重：将内部模型移至 CPU 并清空 CUDA 缓存。"""
        try:
            import torch

            if self._official is not None:
                try:
                    self._official.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            if self._impl is not None:
                try:
                    self._impl.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            from mosaic.core.device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass
        self._official = None
        self._impl = None
        self._use_official = False
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

        # 跳过与重叠 mel 帧对应的样本（center=True STFT 下近似为
        # overlap * hop_length）
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
        """运行当前可用的内部模型（官方或自实现）。"""
        if self._use_official and self._official is not None:
            # 官方 Vocos.decode() 跳过 feature_extractor，直接走 backbone + head
            # forward() 会先运行 feature_extractor（期望原始音频），不能用
            return self._official.decode(mel)
        if self._impl is not None:
            return self._impl(mel)
        raise RuntimeError("No vocoder model available.")

    def _try_load_official(
        self, weights_path: str, device: str, torch_dtype: Any
    ) -> Any:
        """尝试加载官方 ``vocos.Vocos`` 模型，失败返回 ``None``。

        使用 ChatTTS.config.Config 的默认 Vocos 配置创建官方 Vocos 实例，
        确保结构与权重文件完全匹配。不依赖外部 yaml 文件。
        """
        try:
            from vocos import Vocos as VocosOfficial
            from vocos.pretrained import instantiate_class
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(
                "vocos 包不可用，回退到自实现: %s", e
            )
            return None

        try:
            import os
            import torch
            from dataclasses import asdict

            if os.path.isdir(weights_path):
                model = VocosOfficial.from_pretrained(weights_path)
            else:
                # 用 ChatTTS 默认 Config 创建官方 Vocos
                try:
                    from ChatTTS.config import Config
                    config = Config()
                    feature_extractor = instantiate_class(
                        args=(), init=asdict(config.vocos.feature_extractor)
                    )
                    backbone = instantiate_class(
                        args=(), init=asdict(config.vocos.backbone)
                    )
                    head = instantiate_class(
                        args=(), init=asdict(config.vocos.head)
                    )
                    model = VocosOfficial(
                        feature_extractor=feature_extractor,
                        backbone=backbone,
                        head=head,
                    )
                except ImportError:
                    # ChatTTS 不可用时，尝试从 yaml 加载
                    weights_dir = os.path.dirname(weights_path)
                    parent_dir = os.path.dirname(weights_dir)
                    basename = os.path.splitext(
                        os.path.basename(weights_path)
                    )[0]
                    config_candidates = [
                        os.path.join(weights_dir, "config.yaml"),
                        os.path.join(
                            parent_dir, "config", f"{basename.lower()}.yaml"
                        ),
                        os.path.join(parent_dir, "config", "vocos.yaml"),
                    ]
                    config_path = None
                    for c in config_candidates:
                        if os.path.isfile(c):
                            config_path = c
                            break
                    if config_path is None:
                        return None
                    model = VocosOfficial.from_hparams(config_path)

                # 加载权重文件
                if weights_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                else:
                    state_dict = torch.load(
                        weights_path, map_location="cpu", weights_only=False
                    )
                model.load_state_dict(state_dict, strict=True)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Vocos official load failed: %s", e
            )
            return None

        try:
            model = model.to(device=device, dtype=torch_dtype)
            model.eval()
            return model
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Vocos device transfer failed: %s", e
            )
            return None

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

    @staticmethod
    def _strip_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
        """剥离常见前缀（``vocos.`` / ``decoder.``）以提升匹配率。"""
        prefixes = ("vocos.", "decoder.")
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
