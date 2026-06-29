# mosaic/nodes/audio/tts_backends/vocoders/base.py
"""TTS 声码器层抽象基类。

Layer 3: 声码器层。输入 mel spectrogram 或 VQ tokens，输出 waveform。

本模块定义 :class:`Vocoder` 抽象基类，将声学特征（mel spectrogram 或 VQ
tokens）转换为音频波形。支持基于 mel 的声码器（vocos / hifi_gan）与基于
VQ tokens 的声码器（sovits_decoder）。

设计要点
--------
* ``torch`` / ``numpy`` 等重依赖采用惰性导入，使本模块在未安装这些依赖时
  仍可被导入与继承。
* features 等参数类型用 :data:`~typing.Any` 标注，避免在模块顶层硬依赖
  ``torch``。
* 提供流式解码（``decode_chunk``）的默认实现，子类可覆写以维护内部状态
  实现更高效的流式解码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Vocoder"]


class Vocoder(ABC):
    """TTS 声码器抽象基类。

    负责将声学特征（mel spectrogram 或 VQ tokens）转换为音频波形。

    支持的声码器类型：

    - ``"vocos"``：基于 mel 的声码器
    - ``"hifi_gan"``：基于 mel 的声码器
    - ``"sovits_decoder"``：基于 VQ tokens 的声码器

    Attributes
    ----------
    vocoder_type : str
        声码器类型。
    input_type : str
        输入特征类型，``"mel"`` 或 ``"vq_tokens"``。
    sample_rate : int
        输出采样率。
    """

    vocoder_type: str = "hifi_gan"
    input_type: str = "mel"
    sample_rate: int = 24000

    @abstractmethod
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载声码器权重。

        Parameters
        ----------
        weights_path : str
            权重文件路径（safetensors 格式）。
        device : str
            目标设备。
        dtype : str
            数据精度。
        """

    @abstractmethod
    def unload_weights(self) -> None:
        """释放声码器权重。"""

    @abstractmethod
    def decode(self, features: Any) -> Any:
        """将声学特征解码为波形。

        Parameters
        ----------
        features : torch.Tensor
            声学特征：

            - mel spectrogram: ``[batch, mel_bins, frames]``
            - VQ tokens: ``[batch, seq_len]``

        Returns
        -------
        torch.Tensor
            音频波形 ``[batch, samples]``。
        """

    def decode_chunk(self, features: Any) -> Any:
        """流式解码：处理一小块特征，输出一小段波形。

        用于流式合成场景。
        默认实现直接调用 :meth:`decode`。
        子类可覆写以实现更高效的流式解码（如维护内部状态）。

        Parameters
        ----------
        features : torch.Tensor
            一小块声学特征。

        Returns
        -------
        torch.Tensor
            一小段音频波形。
        """
        return self.decode(features)

    def get_mel_basis(
        self, n_fft: int, sample_rate: int, n_mels: int
    ) -> Any:
        """返回 mel 滤波器组。

        如果声码器需要 mel 滤波器组，子类可覆写此方法。
        默认实现返回 ``None``。

        Returns
        -------
        torch.Tensor | None
            mel 滤波器组矩阵；``None`` 表示不需要。
        """
        return None
