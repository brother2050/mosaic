# mosaic/nodes/audio/tts_backends/acoustic_models/speech_tokenizer.py
"""语音 Tokenizer：将参考音频编码为离散语音 token。

本模块是一个独立的辅助类（**非** :class:`AcousticModel` 子类），用于把
参考音频波形编码为离散的语音 token ids，供声学模型做语音克隆 / 上下文
 conditioning 时使用。

背景
----
在零样本语音克隆 TTS（如 CosyVoice / Fish Speech / GPT-SoVITS）中，通常
需要先用一个 codec 模型（如 EnCodec / DAC / WavTokenizer）把参考音频
压缩为离散 token，再将这些 token 作为条件喂给声学模型。本模块提供对该
codec 语义的统一封装。

内部实现
--------
内部 ``nn.Module`` 采用一个简化的 2 层 RVQ（Residual Vector
Quantization）::

    waveform
        → Conv1D 编码器（1 通道 → hidden，按 hop_length 下采样）
        → 第 1 层量化：在 81 个码本中心中找最近 → layer1_id
        → 残差 = z - quantized_1
        → 第 2 层量化：在 81 个码本中心中找最近 → layer2_id
        → token_id = layer1_id * 81 + layer2_id        # 81 * 81 = 6561
        → ConvTranspose1D 解码器（hidden → 1 通道）→ 重建波形

当 ``num_codebooks == 1``（默认）时，两层 RVQ 的结果被合并为单条
token 流，词表大小为 ``codebook_size``（``81 * 81 = 6561``）；当
``num_codebooks == 2`` 时，输出两条独立 token 流。

设计要点
--------
* ``torch`` / ``safetensors`` / ``transformers`` 采用惰性导入：模块顶层
  不导入这些重依赖，真正的 ``nn.Module`` 子类在首次 :meth:`load_weights`
  时通过 :func:`_get_speech_tokenizer_class` 惰性构建。
* :class:`SpeechTokenizer` 是一个独立的辅助类，并使用代理模式：
  ``__getattr__`` 将未在本类找到的属性转发给内部 ``nn.Module`` 实现。
* ``waveform`` / ``token_ids`` 等参数类型用 :data:`~typing.Any` 标注，
  避免在模块顶层硬依赖 ``torch``。
* 即便未安装 ``torch``，本模块仍可被正常导入。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from mosaic.core.onnx_utils import (
    create_inference_session,
    get_onnx_providers,
    is_onnxruntime_usable,
)

if TYPE_CHECKING:  # 仅用于类型注解，运行时惰性导入
    from mosaic.core.types import AudioData

__all__ = ["SpeechTokenizer"]

logger = logging.getLogger(__name__)


# 内部缓存的 nn.Module 子类（惰性创建）
_SpeechTokenizerImplClass: Any = None


def _unwrap_ckpt(ckpt: Any) -> dict[str, Any]:
    """解包 ``{"state_dict": ...}`` / ``{"model": ...}`` 包装格式。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")


def _get_speech_tokenizer_class() -> Any:
    """惰性创建并返回语音 Tokenizer 的 ``nn.Module`` 子类。

    首次调用时导入 ``torch`` 并在函数内部定义 :class:`_SpeechTokenizerImpl`
    （含 2 层 RVQ 的编/解码器），随后缓存到全局变量
    :data:`_SpeechTokenizerImplClass`。

    Returns
    -------
    type
        ``nn.Module`` 子类 ``_SpeechTokenizerImpl``。
    """
    global _SpeechTokenizerImplClass
    if _SpeechTokenizerImplClass is not None:
        return _SpeechTokenizerImplClass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _SpeechTokenizerImpl(nn.Module):
        """语音 Tokenizer 真实实现（``nn.Module`` 子类）。

        采用 2 层 RVQ：Conv1D 编码器 → 量化到 81 个中心 → 残差 → 量化到
        81 个中心 → ``token_id = layer1_id * 81 + layer2_id``。

        Parameters
        ----------
        hidden_size : int
            隐藏维度。
        codebook_size : int
            合并后的词表大小，须为完全平方数（``81 * 81 = 6561``）。
        num_codebooks : int
            输出 token 流数量。``1`` 表示合并为单条流，``2`` 表示输出两条
            独立流（分别对应两层 RVQ）。
        hop_length : int
            编码器下采样步长（每帧对应的采样点数）。
        """

        def __init__(
            self,
            hidden_size: int = 512,
            codebook_size: int = 6561,
            num_codebooks: int = 1,
            hop_length: int = 320,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.codebook_size = codebook_size
            self.num_codebooks = num_codebooks
            self.hop_length = hop_length

            # 码本中心数 = sqrt(codebook_size)，例如 6561 → 81
            num_centers = int(round(codebook_size ** 0.5))
            if num_centers * num_centers != codebook_size:
                # 非完全平方数时回退到 81，保证 2 层 RVQ 结构成立
                num_centers = 81
            self.num_centers = num_centers

            # 编码器：1 通道波形 → hidden，按 hop_length 下采样
            self.encoder = nn.Conv1d(
                1,
                hidden_size,
                kernel_size=hop_length * 2,
                stride=hop_length,
                padding=hop_length // 2,
            )
            # 解码器：hidden → 1 通道波形（上采样回原始分辨率）
            self.decoder = nn.ConvTranspose1d(
                hidden_size,
                1,
                kernel_size=hop_length * 2,
                stride=hop_length,
                padding=hop_length // 2,
            )

            # 两层 RVQ 码本：每个 [num_centers, hidden_size]
            self.codebook1 = nn.Parameter(
                torch.randn(num_centers, hidden_size) * 0.02
            )
            self.codebook2 = nn.Parameter(
                torch.randn(num_centers, hidden_size) * 0.02
            )

        # ------------------------------------------------------------------
        # 量化核心
        # ------------------------------------------------------------------
        def _quantize(self, z: Any, codebook: Any) -> tuple[Any, Any]:
            """在码本中查找最近中心。

            Parameters
            ----------
            z : torch.Tensor
                ``[batch, frames, hidden]``。
            codebook : torch.Tensor
                ``[num_centers, hidden]``。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(ids, quantized)``，``ids`` 形状 ``[batch, frames]``，
                ``quantized`` 形状 ``[batch, frames, hidden]``。
            """
            batch, frames, hidden = z.shape
            z_flat = z.reshape(-1, hidden)               # [B*T, H]
            # 欧氏距离 [1, B*T, C]
            dist = torch.cdist(
                z_flat.unsqueeze(0), codebook.unsqueeze(0)
            ).squeeze(0)                                  # [B*T, C]
            ids = dist.argmin(dim=-1).reshape(batch, frames)
            quantized = codebook[ids]                     # [B, T, H]
            return ids, quantized

        # ------------------------------------------------------------------
        # encode / decode / forward
        # ------------------------------------------------------------------
        def encode(self, waveform: Any) -> Any:
            """将波形编码为 token ids。

            Parameters
            ----------
            waveform : torch.Tensor
                波形 ``[batch, samples]`` 或 ``[samples]``。

            Returns
            -------
            torch.Tensor
                ``num_codebooks == 1`` 时为 ``[batch, num_tokens]`` 的合并
                token ids（取值 ``[0, codebook_size)``）；``num_codebooks
                == 2`` 时为 ``[batch, 2, num_tokens]`` 的两层 token ids。
            """
            x = waveform
            if x.dim() == 1:
                x = x.unsqueeze(0)                        # [B, samples]
            x = x.unsqueeze(1)                            # [B, 1, samples]

            z = self.encoder(x)                           # [B, H, T]
            z = z.transpose(1, 2)                         # [B, T, H]

            ids1, q1 = self._quantize(z, self.codebook1)
            residual = z - q1
            ids2, _q2 = self._quantize(residual, self.codebook2)

            if self.num_codebooks == 1:
                # 合并为单条 token 流
                token_ids = ids1 * self.num_centers + ids2   # [B, T]
            else:
                # 多码本：堆叠为 [B, 2, T]
                token_ids = torch.stack([ids1, ids2], dim=1)
            return token_ids

        def decode(self, token_ids: Any) -> Any:
            """将 token ids 解码为波形（主要用于调试）。

            Parameters
            ----------
            token_ids : torch.Tensor
                ``[batch, num_tokens]``（合并流）或 ``[batch, 2, num_tokens]``
                （双流）。

            Returns
            -------
            torch.Tensor
                重建波形 ``[batch, samples]``。
            """
            tokens = token_ids
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)              # [1, T]

            if self.num_codebooks == 1 or tokens.dim() == 2:
                ids1 = tokens // self.num_centers
                ids2 = tokens % self.num_centers
            else:
                # [B, 2, T]
                ids1 = tokens[:, 0]
                ids2 = tokens[:, 1]

            quantized = self.codebook1[ids1] + self.codebook2[ids2]  # [B, T, H]
            z = quantized.transpose(1, 2)                 # [B, H, T]
            waveform = self.decoder(z)                    # [B, 1, samples]
            return waveform.squeeze(1)                    # [B, samples]

        def forward(self, waveform: Any) -> tuple[Any, Any]:
            """前向计算：编码后立即解码，返回 token 与重建波形。

            Parameters
            ----------
            waveform : torch.Tensor
                波形 ``[batch, samples]`` 或 ``[samples]``。

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(token_ids, reconstructed_audio)``。
            """
            token_ids = self.encode(waveform)
            reconstructed = self.decode(token_ids)
            return token_ids, reconstructed

    _SpeechTokenizerImplClass = _SpeechTokenizerImpl
    return _SpeechTokenizerImplClass


class SpeechTokenizer:
    """语音 Tokenizer：将参考音频编码为离散语音 token。

    独立的辅助类（**非** :class:`AcousticModel` 子类），封装 codec 模型
    （如 EnCodec / DAC / WavTokenizer）的编/解码语义，用于把参考音频
    压缩为离散 token ids 供声学模型做语音克隆。

    权重加载策略（:meth:`load_weights`）：

    1. 解析 dtype 字符串为 torch dtype，无 GPU 时设备降级为 CPU；
    2. 惰性创建内部 ``nn.Module`` 子类实例（2 层 RVQ 编/解码器）；
    3. 读取权重（safetensors 优先，回退到 ``.pt`` / ``.pth`` / ``.bin``）；
    4. 剥离 ``tokenizer.`` / ``codec.`` 等前缀后 ``strict=False`` 载入；
    5. 移动到目标 device / dtype，切换为 eval。

    Parameters
    ----------
    model_type : str
        tokenizer 类型，如 ``"cosyvoice"`` / ``"encodec"`` / ``"dac"``，
        仅用于标识，不影响内部结构。
    codebook_size : int
        合并后的词表大小，默认 ``6561``（``81 * 81``，对应 2 层 RVQ）。
    num_codebooks : int
        输出 token 流数量，默认 ``1``（合并为单条流）。
    hidden_size : int
        隐藏维度，默认 ``512``。
    sample_rate : int
        期望的音频采样率，默认 ``22050``。

    Attributes
    ----------
    model_type : str
        tokenizer 类型标识。
    codebook_size : int
        合并词表大小。
    num_codebooks : int
        输出 token 流数量。
    hidden_size : int
        隐藏维度。
    sample_rate : int
        期望采样率。
    """

    def __init__(
        self,
        model_type: str = "cosyvoice",
        codebook_size: int = 6561,
        num_codebooks: int = 1,
        hidden_size: int = 512,
        sample_rate: int = 22050,
    ) -> None:
        # 注意：此处不导入 torch；内部 nn.Module 在 load_weights 时创建
        self.model_type = model_type
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate

        # 编码器下采样步长（每帧对应的采样点数）
        self.hop_length: int = 320

        # 内部模型实例（load_weights 后填充）
        self._impl: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False
        # ONNX Runtime 推理会话（加载 .onnx 权重时填充，PyTorch 路径为 None）
        self._onnx_session: Any = None

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
            "SpeechTokenizer is not loaded. Call load_weights() before "
            "calling it."
        )

    # ------------------------------------------------------------------
    # 权重加载 / 释放
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载语音 Tokenizer 权重。

        当 ``weights_path`` 指向 ``.onnx`` 文件时，优先使用
        ``onnxruntime`` 加载并推理（跳过 PyTorch 实现）；若 onnxruntime
        不可用，则记录 warning 并回退到 PyTorch 路径，在同目录查找
        ``.pt`` / ``.safetensors`` 替代权重。非 ``.onnx`` 文件走原有
        PyTorch 路径。

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
        # ONNX 路径：若权重为 .onnx 文件，使用 onnxruntime 加载
        if os.path.isfile(weights_path) and weights_path.lower().endswith(".onnx"):
            if is_onnxruntime_usable():
                self._onnx_session = create_inference_session(
                    weights_path, providers=get_onnx_providers(device)
                )
                self._device = device
                self._dtype = dtype
                self._is_loaded = True
                logger.info(
                    "SpeechTokenizer ONNX 模型已加载 "
                    "(path=%s, providers=%s)",
                    weights_path,
                    self._onnx_session.get_providers(),
                )
                return
            # onnxruntime 不可用 → 回退到 PyTorch 路径，尝试查找替代权重
            logger.warning(
                "onnxruntime 不可用，无法加载 ONNX 模型 %s，"
                "尝试回退到 PyTorch 权重（.pt/.safetensors）。",
                weights_path,
            )
            parent = os.path.dirname(weights_path)
            if parent and os.path.isdir(parent):
                weights_path = parent

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
        cls = _get_speech_tokenizer_class()
        impl = cls(
            hidden_size=self.hidden_size,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            hop_length=self.hop_length,
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
        # 释放 ONNX 推理会话
        self._onnx_session = None
        try:
            import torch

            if self._impl is not None:
                try:
                    self._impl.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            from mosaic.core._device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass
        self._impl = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def encode(self, audio: Any) -> Any:
        """将音频编码为离散语音 token ids。

        接受 :class:`~mosaic.core.types.AudioData`、``torch.Tensor`` 或
        ``numpy.ndarray`` 作为输入。若输入采样率与 :attr:`sample_rate`
        不一致，会先重采样。

        Parameters
        ----------
        audio : AudioData | torch.Tensor | numpy.ndarray
            输入音频。

        Returns
        -------
        torch.Tensor
            token ids，``num_codebooks == 1`` 时形状 ``[1, num_tokens]``，
            ``num_codebooks == 2`` 时形状 ``[2, num_tokens]``。

        Raises
        ------
        RuntimeError
            模型未加载。
        TypeError
            不支持的输入类型。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SpeechTokenizer is not loaded. Call load_weights() "
                "before encode()."
            )

        waveform, sample_rate = self._coerce_audio(audio)
        if sample_rate != self.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.sample_rate)

        # ONNX 推理路径
        if self._onnx_session is not None:
            return self._encode_onnx(waveform)

        import torch

        waveform = waveform.to(self._device)

        with torch.no_grad():
            tokens = self._impl.encode(waveform)
        return tokens

    def decode(self, token_ids: Any) -> AudioData:
        """将 token ids 解码为音频（主要用于调试）。

        Parameters
        ----------
        token_ids : torch.Tensor
            token ids，``[batch, num_tokens]`` 或 ``[num_tokens]``。

        Returns
        -------
        AudioData
            重建音频，采样率为 :attr:`sample_rate`。

        Raises
        ------
        RuntimeError
            模型未加载。
        """
        if not self._is_loaded:
            raise RuntimeError(
                "SpeechTokenizer is not loaded. Call load_weights() "
                "before decode()."
            )
        import numpy as np
        import torch
        from mosaic.core.types import AudioData

        tokens = torch.as_tensor(token_ids, dtype=torch.long)

        # ONNX 推理路径
        if self._onnx_session is not None:
            tokens_np = tokens.detach().cpu().long().numpy()
            if tokens_np.ndim == 1:
                tokens_np = tokens_np[np.newaxis, :]        # [1, T]
            feeds = self._build_onnx_feeds(self._onnx_session, tokens_np)
            outputs = self._onnx_session.run(None, feeds)
            waveform_np = np.asarray(outputs[0], dtype=np.float32)
            if waveform_np.ndim == 2:
                waveform_np = waveform_np[0]                 # [samples]
            return AudioData(waveform=waveform_np, sample_rate=self.sample_rate)

        tokens = tokens.to(self._device)

        with torch.no_grad():
            waveform = self._impl.decode(tokens)         # [B, samples]
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)                # [samples]
        waveform_np = waveform.detach().cpu().float().numpy()
        if waveform_np.ndim == 2:
            waveform_np = waveform_np[0]
        return AudioData(waveform=waveform_np, sample_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # 内部辅助：ONNX 推理
    # ------------------------------------------------------------------
    def _encode_onnx(self, waveform: Any) -> Any:
        """ONNX 推理：将波形编码为 token ids。

        Parameters
        ----------
        waveform : torch.Tensor
            单声道波形 ``[samples]`` 或 ``[batch, samples]``。

        Returns
        -------
        torch.Tensor
            token ids（结构与 PyTorch 路径一致）。
        """
        import numpy as np
        import torch

        wav_np = waveform.detach().cpu().float().numpy()
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]                  # [1, samples]

        feeds = self._build_onnx_feeds(self._onnx_session, wav_np)
        outputs = self._onnx_session.run(None, feeds)
        # 取第一个输出作为 token ids，转回 torch.Tensor 保持接口一致
        return torch.as_tensor(outputs[0])

    @staticmethod
    def _build_onnx_feeds(session: Any, data_np: Any) -> dict[str, Any]:
        """构建 ONNX 推理输入字典。

        将主数据（波形或 token ids）喂给第一个非 ``length`` 输入；若模型
        含 ``length`` 输入（部分 codec / 说话人编码器导出），自动填充为
        主数据最后一维的长度。

        Parameters
        ----------
        session : onnxruntime.InferenceSession
            ONNX 推理会话。
        data_np : numpy.ndarray
            主输入数据（波形或 token ids）。

        Returns
        -------
        dict[str, Any]
            供 ``session.run`` 使用的输入字典。
        """
        import numpy as np

        feeds: dict[str, Any] = {}
        inputs = session.get_inputs()
        if not inputs:
            return feeds

        audio_inp: Any = None
        len_inps: list[Any] = []
        for inp in inputs:
            if "len" in inp.name.lower():
                len_inps.append(inp)
            elif audio_inp is None:
                audio_inp = inp
        if audio_inp is None:
            audio_inp = inputs[0]
        feeds[audio_inp.name] = data_np

        length = int(data_np.shape[-1]) if data_np.ndim else 0
        for inp in len_inps:
            feeds[inp.name] = np.array([length], dtype=np.int32)
        return feeds

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
                "tokenizer.safetensors",
                "codec.safetensors",
                "speech_tokenizer.safetensors",
                "model.safetensors",
            ):
                fpath = os.path.join(weights_path, fname)
                if os.path.isfile(fpath):
                    from safetensors.torch import load_file

                    state_dict = load_file(fpath)
                    break
            if not state_dict:
                for fname in (
                    "tokenizer.bin",
                    "codec.bin",
                    "speech_tokenizer.bin",
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

        会剥离 ``tokenizer.`` / ``codec.`` / ``speech_tokenizer.`` /
        ``quantizer.`` / ``model.`` 等前缀；丢弃以 ``discriminator.`` /
        ``disc.`` 开头的判别器权重。
        """
        disc_prefixes = ("discriminator.", "disc.")
        gen_prefixes = (
            "tokenizer.",
            "codec.",
            "speech_tokenizer.",
            "quantizer.",
            "model.",
        )

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
