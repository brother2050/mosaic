# mosaic/nodes/audio/tts_backends/acoustic_models/speaker_encoder.py
"""说话人编码器：从参考音频中提取说话人嵌入向量。

本模块是一个独立的辅助类（**非** :class:`AcousticModel` 子类），用于把
参考音频波形编码为一个固定维度的说话人嵌入向量，供声学模型做语音克隆时
作为说话人条件。

背景
----
在零样本 / 少样本语音克隆 TTS 中，通常需要先用一个说话人编码器（如
ECAPA-TDNN / CAM++ / x-vector）从参考音频中提取一个全局的说话人嵌入，
再将该嵌入作为条件注入声学模型。本模块提供对该说话人编码语义的统一封装。

内部实现
--------
内部 ``nn.Module`` 实现一个简化的 ECAPA-TDNN::

    waveform
        → Conv1D 前端（1 通道 → hidden）
        → 3 个 SE-Res2Net 残差块（多尺度感受野 + 通道注意力）
        → Attentive Statistics Pooling（注意力统计量池化）
        → Linear 投影到 embedding_dim
        → speaker embedding ``[1, embedding_dim]``

各子模块说明：

* **SE-Res2Net 块**：1x1 卷积扩张后按通道分组，每组串行 3x3 膨胀卷积
  （Res2Net 多尺度），再 1x1 卷积回原通道，经 SE（Squeeze-and-Excitation）
  通道注意力调制后做残差连接。
* **Attentive Statistics Pooling**：对时间维做注意力加权，同时聚合加权
  均值与加权标准差，拼成 ``[batch, 2 * hidden]``。
* **Linear 投影**：``2 * hidden → embedding_dim``。

设计要点
--------
* ``torch`` / ``safetensors`` / ``transformers`` 采用惰性导入：模块顶层
  不导入这些重依赖，真正的 ``nn.Module`` 子类在首次 :meth:`load_weights`
  时通过 :func:`_get_speaker_encoder_class` 惰性构建。
* :class:`SpeakerEncoder` 是一个独立的辅助类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现。
* 说话人编码器通常工作在 **16 kHz**，:meth:`encode` 会在必要时把输入
  音频重采样到 :attr:`sample_rate`（默认 ``16000``）。
* ``waveform`` / ``embedding`` 等参数类型用 :data:`~typing.Any` 标注，
  避免在模块顶层硬依赖 ``torch``。
* 即便未安装 ``torch``，本模块仍可被正常导入。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅用于类型注解，运行时惰性导入
    from mosaic.core.types import AudioData

__all__ = ["SpeakerEncoder"]


# 内部缓存的 nn.Module 子类（惰性创建）
_SpeakerEncoderImplClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_speaker_encoder_class() -> Any:
    """惰性创建并返回说话人编码器的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并在函数内部定义 :class:`_SEBlock` /
    :class:`_SERes2NetBlock` / :class:`_AttentiveStatsPooling` /
    :class:`_SpeakerEncoderImpl`（简化的 ECAPA-TDNN），随后缓存到全局
    变量 :data:`_SpeakerEncoderImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_SpeakerEncoderImpl``。
    """
    global _SpeakerEncoderImplClass
    if _SpeakerEncoderImplClass is not None:
        return _SpeakerEncoderImplClass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # SEBlock：Squeeze-and-Excitation 通道注意力
    # ------------------------------------------------------------------
    class _SEBlock(nn.Module):
        """Squeeze-and-Excitation 通道注意力模块。

        对时间维做全局平均池化得到通道描述子，经两层 1x1 卷积 + sigmoid
        得到通道权重，再逐通道缩放输入。

        Parameters
        ----------
        channels : int
            通道数。
        se_ratio : int
            SE 瓶颈维度的缩减比，瓶颈维度为 ``channels // se_ratio``。
        """

        def __init__(self, channels: int, se_ratio: int = 8) -> None:
            super().__init__()
            mid = max(channels // se_ratio, 8)
            self.fc1 = nn.Conv1d(channels, mid, kernel_size=1)
            self.fc2 = nn.Conv1d(mid, channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, channels, time]``。

            Returns
            -------
            torch.Tensor
                通道注意力调制后的结果，同形状。
            """
            w = x.mean(dim=2, keepdim=True)              # [B, C, 1]
            w = F.relu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))
            return x * w

    # ------------------------------------------------------------------
    # SERes2NetBlock：带 SE 的 Res2Net 多尺度残差块
    # ------------------------------------------------------------------
    class _SERes2NetBlock(nn.Module):
        """SE-Res2Net 残差块（ECAPA-TDNN 的核心构建单元）。

        结构::

            x → Conv1x1(channels → channels*scale) → ReLU
              → 按通道分成 scale 组，第 0 组直通，其余组依次串行 3x3 膨胀卷积
              → concat → Conv1x1(channels*scale → channels)
              → SE 通道注意力
              → + 残差(x) → ReLU

        Parameters
        ----------
        channels : int
            通道数。
        kernel_size : int
            分支内 3x3 卷积的核大小。
        dilation : int
            分支内 3x3 卷积的膨胀率。
        scale : int
            Res2Net 分组数（多尺度）。
        se_ratio : int
            SE 缩减比。
        """

        def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            scale: int = 8,
            se_ratio: int = 8,
        ) -> None:
            super().__init__()
            self.scale = scale
            hidden = channels * scale
            self.split_size = channels                  # 每组通道数

            # 1x1 扩张
            self.conv1 = nn.Conv1d(channels, hidden, kernel_size=1)
            # scale-1 个分支卷积（第 0 组直通）
            pad = (kernel_size - 1) * dilation // 2
            self.branch_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.split_size,
                        self.split_size,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=pad,
                    )
                    for _ in range(scale - 1)
                ]
            )
            # 1x1 回到 channels
            self.conv2 = nn.Conv1d(hidden, channels, kernel_size=1)
            self.se = _SEBlock(channels, se_ratio)
            self.relu = nn.ReLU()

        def forward(self, x: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, channels, time]``。

            Returns
            -------
            torch.Tensor
                同形状 ``[batch, channels, time]``。
            """
            residual = x
            y = self.relu(self.conv1(x))                # [B, hidden, T]
            splits = y.split(self.split_size, dim=1)    # scale 组各 [B, C, T]

            # Res2Net 多尺度：第 0 组直通，其余组依次卷积
            outs = [splits[0]]
            cur = splits[0]
            for conv, s in zip(self.branch_convs, splits[1:]):
                cur = conv(cur)
                # 与同组原始特征相加，增强多尺度表达
                cur = cur + s
                outs.append(cur)
            y = torch.cat(outs, dim=1)                  # [B, hidden, T]

            y = self.conv2(y)                            # [B, C, T]
            y = self.se(y)
            y = y + residual
            return self.relu(y)

    # ------------------------------------------------------------------
    # AttentiveStatsPooling：注意力统计量池化
    # ------------------------------------------------------------------
    class _AttentiveStatsPooling(nn.Module):
        """注意力统计量池化（Attentive Statistics Pooling）。

        对时间维做注意力加权，同时聚合加权均值与加权标准差，拼成
        ``[batch, 2 * channels]``。

        Parameters
        ----------
        channels : int
            输入通道数。
        hidden : int
            注意力打分网络的隐层维度。
        """

        def __init__(self, channels: int, hidden: int = 128) -> None:
            super().__init__()
            self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
            self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """前向计算。

            Parameters
            ----------
            x : torch.Tensor
                ``[batch, channels, time]``。

            Returns
            -------
            torch.Tensor
                ``[batch, 2 * channels]``，前半为加权均值，后半为加权标准差。
            """
            w = torch.tanh(self.fc1(x))                  # [B, hidden, T]
            w = self.fc2(w)                               # [B, C, T]
            w = F.softmax(w, dim=2)                       # 时间维注意力
            mean = (x * w).sum(dim=2)                     # [B, C]
            var = ((x - mean.unsqueeze(2)) ** 2 * w).sum(dim=2)
            std = torch.sqrt(var.clamp(min=1e-9))
            return torch.cat([mean, std], dim=1)          # [B, 2C]

    # ------------------------------------------------------------------
    # _SpeakerEncoderImpl：主实现类（简化 ECAPA-TDNN）
    # ------------------------------------------------------------------
    class _SpeakerEncoderImpl(nn.Module):
        """说话人编码器真实实现（``nn.Module`` 子类）。

        简化的 ECAPA-TDNN：Conv1D 前端 → 3 个 SE-Res2Net 块 → 注意力
        统计量池化 → Linear 投影。

        Parameters
        ----------
        embedding_dim : int
            输出说话人嵌入维度。
        hidden_size : int
            主干隐藏维度。
        n_blocks : int
            SE-Res2Net 残差块数量。
        scale : int
            Res2Net 多尺度分组数。
        """

        def __init__(
            self,
            embedding_dim: int = 192,
            hidden_size: int = 512,
            n_blocks: int = 3,
            scale: int = 8,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size

            # 前端：1 通道波形 → hidden
            self.frontend = nn.Conv1d(
                1, hidden_size, kernel_size=5, stride=1, padding=2
            )
            # 3 个 SE-Res2Net 块，膨胀率递增（1, 2, 4）
            self.blocks = nn.ModuleList(
                [
                    _SERes2NetBlock(
                        hidden_size,
                        kernel_size=3,
                        dilation=2 ** i,
                        scale=scale,
                    )
                    for i in range(n_blocks)
                ]
            )
            # 注意力统计量池化 → [B, 2 * hidden]
            self.pool = _AttentiveStatsPooling(hidden_size)
            # 投影到 embedding_dim
            self.proj = nn.Linear(hidden_size * 2, embedding_dim)

        def forward(self, waveform: Any) -> Any:
            """前向计算：从波形提取说话人嵌入。

            Parameters
            ----------
            waveform : torch.Tensor
                波形 ``[batch, samples]`` 或 ``[samples]``。

            Returns
            -------
            torch.Tensor
                说话人嵌入 ``[batch, embedding_dim]``（单条输入时为
                ``[1, embedding_dim]``）。
            """
            x = waveform
            if x.dim() == 1:
                x = x.unsqueeze(0)                        # [B, samples]
            x = x.unsqueeze(1)                            # [B, 1, samples]

            x = self.frontend(x)                          # [B, hidden, T]
            for block in self.blocks:
                x = block(x)
            pooled = self.pool(x)                         # [B, 2*hidden]
            embedding = self.proj(pooled)                 # [B, embedding_dim]
            return embedding

    _SpeakerEncoderImplClass = _SpeakerEncoderImpl
    return _SpeakerEncoderImplClass


class SpeakerEncoder:
    """说话人编码器：从参考音频中提取说话人嵌入向量。

    独立的辅助类（**非** :class:`AcousticModel` 子类），封装说话人编码器
    （如 ECAPA-TDNN / CAM++ / x-vector）的语义，用于把参考音频压缩为
    一个固定维度的说话人嵌入向量，作为声学模型的说话人条件。

    权重加载策略（:meth:`load_weights`）：

    1. 解析 dtype 字符串为 torch dtype，无 GPU 时设备降级为 CPU；
    2. 惰性创建内部 ``nn.Module`` 子类实例（简化 ECAPA-TDNN）；
    3. 读取权重（safetensors 优先，回退到 ``.pt`` / ``.pth`` / ``.bin``）；
    4. 剥离 ``speaker_encoder.`` / ``encoder.`` 等前缀后 ``strict=False``
       载入；
    5. 移动到目标 device / dtype，切换为 eval。

    Parameters
    ----------
    model_type : str
        编码器类型，``"campp"`` / ``"ecapa"`` / ``"xvector"``，仅用于标识，
        不影响内部结构（内部统一使用简化 ECAPA-TDNN）。
    embedding_dim : int
        输出说话人嵌入维度，默认 ``192``。
    sample_rate : int
        期望的音频采样率，默认 ``16000``（说话人编码器通常使用 16 kHz）。

    Attributes
    ----------
    model_type : str
        编码器类型标识。
    embedding_dim : int
        输出嵌入维度。
    sample_rate : int
        期望采样率。
    """

    def __init__(
        self,
        model_type: str = "campp",
        embedding_dim: int = 192,
        sample_rate: int = 16000,
    ) -> None:
        # 注意：此处不导入 torch；内部 nn.Module 在 load_weights 时创建
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate

        # 主干隐藏维度（与 embedding_dim 解耦，便于承载更宽的特征）
        self.hidden_size: int = 192
        # SE-Res2Net 残差块数量
        self.n_blocks: int = 3
        # Res2Net 多尺度分组数
        self.scale: int = 8

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

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
            "SpeakerEncoder is not loaded. Call load_weights() before "
            "calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载说话人编码器权重。

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
        cls = _get_speaker_encoder_class()
        impl = cls(
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            n_blocks=self.n_blocks,
            scale=self.scale,
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
                except Exception:  # noqa: BLE001
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._impl = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode(self, audio: Any) -> Any:
        """从音频中提取说话人嵌入向量。

        接受 :class:`~mosaic.core.types.AudioData`、``torch.Tensor`` 或
        ``numpy.ndarray`` 作为输入。若输入采样率与 :attr:`sample_rate`
        不一致，会先重采样（说话人编码器通常需要 16 kHz）。

        Parameters
        ----------
        audio : AudioData | torch.Tensor | numpy.ndarray
            输入音频。

        Returns
        -------
        torch.Tensor
            说话人嵌入 ``[1, embedding_dim]``。

        Raises
        ------
        RuntimeError
            模型未加载。
        TypeError
            不支持的输入类型。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SpeakerEncoder is not loaded. Call load_weights() "
                "before encode()."
            )
        import torch

        waveform, sample_rate = self._coerce_audio(audio)
        if sample_rate != self.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.sample_rate)
        waveform = waveform.to(self._device)

        with torch.no_grad():
            embedding = self._impl.forward(waveform)
        return embedding

    # ------------------------------------------------------------------
    # 内部辅助：音频规整与重采样
    # ------------------------------------------------------------------
    def _coerce_audio(self, audio: Any) -> tuple[Any, int]:
        """将多种音频输入规整为 ``(mono waveform tensor, sample_rate)``。

        Parameters
        ----------
        audio : AudioData | torch.Tensor | numpy.ndarray
            输入音频。

        Returns
        -------
        tuple[torch.Tensor, int]
            单声道波形 ``[samples]`` 与采样率。
        """
        import numpy as np
        import torch

        sample_rate = self.sample_rate
        waveform: Any = audio

        try:
            from mosaic.core.types import AudioData

            if isinstance(audio, AudioData):
                waveform = audio.waveform
                sample_rate = audio.sample_rate
        except ImportError:
            pass

        if isinstance(waveform, np.ndarray):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.to(torch.float32)
        else:
            # 兜底：尝试转换为 tensor
            waveform = torch.as_tensor(waveform, dtype=torch.float32)

        # 统一为单声道 [samples]
        if waveform.dim() == 2:
            # (channels, samples) → mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
        elif waveform.dim() == 0:
            raise TypeError("音频波形不能是标量")
        # dim == 1 直接使用
        return waveform, sample_rate

    @staticmethod
    def _resample(
        waveform: Any, orig_sr: int, target_sr: int
    ) -> Any:
        """重采样单声道波形。

        优先使用 ``torchaudio``，未安装时回退到线性插值。

        Parameters
        ----------
        waveform : torch.Tensor
            单声道波形 ``[samples]``。
        orig_sr : int
            原始采样率。
        target_sr : int
            目标采样率。

        Returns
        -------
        torch.Tensor
            重采样后的单声道波形 ``[samples]``。
        """
        import torch
        import torch.nn.functional as F

        if orig_sr == target_sr:
            return waveform

        # 优先 torchaudio
        try:
            import torchaudio

            x = waveform.unsqueeze(0).unsqueeze(0)        # [1, 1, L]
            y = torchaudio.functional.resample(x, orig_sr, target_sr)
            return y.squeeze(0).squeeze(0)
        except Exception:  # noqa: BLE001
            pass

        # 回退：线性插值
        scale = target_sr / orig_sr
        new_len = max(1, int(round(waveform.shape[-1] * scale)))
        x = waveform.unsqueeze(0).unsqueeze(0)            # [1, 1, L]
        y = F.interpolate(
            x, size=new_len, mode="linear", align_corners=False
        )
        return y.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # 内部辅助：权重读取
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
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                state_dict = _unwrap_ckpt(ckpt)
        elif os.path.isdir(weights_path):
            for fname in (
                "speaker_encoder.safetensors",
                "spk_encoder.safetensors",
                "encoder.safetensors",
                "model.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in (
                    "speaker_encoder.bin",
                    "spk_encoder.bin",
                    "encoder.bin",
                    "model.bin",
                ):
                    fpath = os.path.join(weights_path, fname)
                    if os.path.isfile(fpath):
                        ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                        state_dict = _unwrap_ckpt(ckpt)
                        break
        return state_dict

    @staticmethod
    def _filter_and_strip(state_dict: dict[str, Any]) -> dict[str, Any]:
        """剥离外层包装前缀，使剩余 key 与内部模块结构匹配。

        会剥离 ``speaker_encoder.`` / ``spk_encoder.`` / ``encoder.`` /
        ``model.`` 等前缀。
        """
        gen_prefixes = (
            "speaker_encoder.",
            "spk_encoder.",
            "encoder.",
            "model.",
        )

        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = key
            for p in gen_prefixes:
                if key.startswith(p):
                    new_key = key[len(p):]
                    break
            filtered[new_key] = value
        return filtered
