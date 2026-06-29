# mosaic/nodes/audio/tts_backends/text_frontends/base.py
"""TTS 文本前端层抽象基类。

Layer 1: 文本前端层。负责文本清洗 -> 韵律标注 -> 分词/音素化 -> token ids。

本模块定义 :class:`TextFrontend` 抽象基类，将原始文本转换为模型可处理的
token ids。子类需实现具体的 tokenize / detokenize 逻辑，并可覆写文本清洗、
说话人编码、韵律标记插入等辅助方法。

设计要点
--------
* ``torch`` / ``numpy`` 等重依赖采用惰性导入，使本模块在未安装这些依赖时
  仍可被导入与继承（仅在实际调用 tokenize 等方法时才报依赖缺失）。
* token ids 等参数类型用 :data:`~typing.Any` 标注，避免在模块顶层硬依赖
  ``torch``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["TextFrontend"]


class TextFrontend(ABC):
    """TTS 文本前端抽象基类。

    负责将原始文本转换为模型可处理的 token ids。
    子类需要实现具体的 tokenize 逻辑。

    Attributes
    ----------
    vocab_size : int
        词表大小。
    special_tokens : dict[str, int]
        特殊标记映射，如 ``{"<pad>": 0, "<eos>": 1, ...}``。
    """

    # 类属性
    vocab_size: int = 0
    special_tokens: dict[str, int] = {}

    @abstractmethod
    def tokenize(self, text: str, language: str = "zh", **kwargs: Any) -> Any:
        """将文本转换为 token ids。

        Parameters
        ----------
        text : str
            输入文本。
        language : str
            语言代码。

        Returns
        -------
        torch.Tensor
            token ids 张量，形状 ``[seq_len]``。
        """

    @abstractmethod
    def detokenize(self, token_ids: Any) -> str:
        """将 token ids 转回文本（用于调试）。

        Parameters
        ----------
        token_ids : torch.Tensor
            token ids 张量。

        Returns
        -------
        str
            还原的文本。
        """

    def encode_speaker(self, speaker_id: str | None) -> Any | None:
        """将说话人 ID 编码为嵌入向量。

        默认实现返回 ``None``（不支持说话人编码）。
        子类可覆写以实现具体的说话人嵌入。

        Parameters
        ----------
        speaker_id : str | None
            说话人标识符。

        Returns
        -------
        torch.Tensor | None
            说话人嵌入向量；``None`` 表示不使用说话人条件。
        """
        return None

    def insert_prosody_tokens(self, text: str, prosody_prompt: str) -> str:
        """在文本中插入韵律控制标记。

        默认实现直接拼接 ``prosody_prompt`` 到文本前。
        子类可覆写以实现具体的韵律标记插入逻辑。

        Parameters
        ----------
        text : str
            原始文本。
        prosody_prompt : str
            韵律提示文本。

        Returns
        -------
        str
            插入韵律标记后的文本。
        """
        if prosody_prompt:
            return f"{prosody_prompt} {text}"
        return text

    def preprocess(self, text: str) -> str:
        """文本清洗：去除特殊字符、标准化标点、处理数字和缩写。

        默认实现执行基本清洗：

        - 去除多余空白
        - 标准化 Unicode
        - 去除控制字符

        子类可覆写以实现语言特定的清洗逻辑。

        Parameters
        ----------
        text : str
            原始文本。

        Returns
        -------
        str
            清洗后的文本。
        """
        import unicodedata

        # Unicode 标准化（NFC）
        text = unicodedata.normalize("NFC", text)
        # 去除控制字符（保留换行和空格）
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C"
            or ch in ("\n", " ")
        )
        # 合并多余空白
        import re

        text = re.sub(r"\s+", " ", text).strip()
        return text
