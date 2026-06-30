# mosaic/nodes/audio/tts_backends/acoustic_models/base.py
"""TTS 声学模型层抽象基类。

Layer 2: 声学模型层。输入 token ids + 条件信息，输出 mel frames 或 VQ token ids。

本模块定义 :class:`AcousticModel` 抽象基类，将文本 token ids 转换为声学特征
（mel spectrogram 或 VQ tokens）。支持自回归（``"ar"``）与流匹配
（``"flow_matching"``）两种模型类型。

设计要点
--------
* ``torch`` / ``numpy`` 等重依赖采用惰性导入，使本模块在未安装这些依赖时
  仍可被导入与继承。
* token ids 等参数类型用 :data:`~typing.Any` 标注，避免在模块顶层硬依赖
  ``torch``。
* 流匹配模型不支持流式生成，调用 ``generate_stream`` 会抛出
  :class:`NotImplementedError`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["AcousticModel"]


class AcousticModel(ABC):
    """TTS 声学模型抽象基类。

    负责将文本 token ids 转换为声学特征（mel spectrogram 或 VQ tokens）。

    支持两种模型类型：

    - ``"ar"``：自回归模型，逐 token 生成音频码
    - ``"flow_matching"``：流匹配模型，一次性从噪声生成 mel

    Attributes
    ----------
    model_type : str
        模型类型，``"ar"`` 或 ``"flow_matching"``。
    vocab_size : int
        词表大小。
    hidden_size : int
        隐藏层维度。
    """

    model_type: str = "ar"
    vocab_size: int = 0
    hidden_size: int = 0

    @abstractmethod
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载模型权重。

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
        """释放模型权重。"""

    @abstractmethod
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
        """生成声学特征。

        对于自回归模型：逐 token 生成音频码，返回 VQ token ids。
        对于流匹配模型：一次性从噪声生成 mel，返回 mel spectrogram。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本 token ids，形状 ``[seq_len]``。
        speaker_embedding : torch.Tensor | None
            说话人嵌入向量。
        max_new_tokens : int
            最大生成 token 数（仅 AR 模型）。
        temperature : float
            采样温度。
        top_p : float
            nucleus sampling 的 p 值。
        top_k : int
            top-k 采样的 k 值。

        Returns
        -------
        torch.Tensor
            生成的声学特征（mel 或 VQ token ids）。
        """

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 24,
        **kwargs: Any,
    ) -> Iterator[np.ndarray | torch.Tensor]:
        """流式生成声学特征。

        每生成 ``stream_batch`` 个 token yield 一次。
        仅自回归模型支持流式生成。
        流匹配模型不支持流式，调用此方法时抛出 :class:`NotImplementedError`。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本 token ids。
        speaker_embedding : torch.Tensor | None
            说话人嵌入向量。
        stream_batch : int
            每次 yield 的 token 数量。

        Yields
        ------
        torch.Tensor
            一小段声学特征。

        Raises
        ------
        NotImplementedError
            如果模型类型为 ``"flow_matching"``。
        """
        if self.model_type == "flow_matching":
            raise NotImplementedError(
                "Flow matching models do not support streaming generation. "
                "Use generate() instead."
            )
        # 默认实现：子类应覆写以实现真正的流式生成
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented generate_stream()."
        )

    def get_input_embeddings(self) -> Any:
        """返回输入嵌入层。

        用于权重转换时检查结构一致性。
        默认实现返回 ``None``。
        """
        return None

    def get_output_head(self) -> Any:
        """返回输出头。

        用于权重转换时检查结构一致性。
        默认实现返回 ``None``。
        """
        return None
