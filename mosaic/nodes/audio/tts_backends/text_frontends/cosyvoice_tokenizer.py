# mosaic/nodes/audio/tts_backends/text_frontends/cosyvoice_tokenizer.py
"""CosyVoice 文本前端实现。

本模块实现 :class:`CosyVoiceTokenizer`，基于 :class:`TextFrontend` 抽象基类，
为 CosyVoice 声学模型提供文本前端能力，主要职责包括：

* 文本清洗与 Unicode 标准化、去除多余空白。
* 韵律控制：CosyVoice 通过标点符号控制韵律，不使用 ChatTTS 风格的特殊
  韵律标记，因此 :meth:`insert_prosody_tokens` 仅在提供 ``prosody_prompt``
  时将其拼接到文本前。
* 复用 LLM 自带的 tokenizer（如 ``Qwen/Qwen2.5-1.5B-Instruct``）对文本
  进行 BPE 子词分词，并将文本映射为 token ids。
* 说话人条件：通过参考音频的语音 token 与说话人嵌入实现音色克隆。本前端
  仅做接口预留，实际的语音 token 提取与说话人嵌入计算由后端的
  ``SpeechTokenizer`` / ``SpeakerEncoder`` 完成。
* 组装 CosyVoice LLM 所需的输入序列：

  ``<sos> text_tokens <flow>``

  其中 ``<flow>`` 标记流匹配（Flow Matching）声学模型生成的起点。

Token 空间设计
--------------
CosyVoice 将文本 token、语音 token 与结构标记统一在同一词表空间内：

.. code-block:: text

    [0, llm_vocab_size)                          -> LLM 文本 token
    [llm_vocab_size, llm_vocab_size              -> 语音 token
              + speech_token_size)
    [sos_token_id]                               -> 序列开始
    [eos_token_id]                               -> 序列结束
    [flow_token_id]                              -> 流匹配生成起点标记
    [spk_token_id]                               -> 说话人标记

即文本 token 紧贴词表起始、语音 token 紧随其后、4 个特殊标记位于词表
末尾。总词表大小为::

    vocab_size = llm_vocab_size + speech_token_size + 4

设计要点
--------
* ``torch`` / ``transformers`` 等重依赖采用惰性导入，使本模块在未安装这些
  依赖时仍可被导入与实例化（仅在实际调用 ``tokenize`` / ``load_weights``
  等方法时才报依赖缺失）。
* 在 LLM tokenizer 尚未加载（或 ``transformers`` 不可用）时，文本编码退
  化为简单的字符级 dummy 分词，便于在无权重环境下进行接口测试与调试。
* token ids 等参数类型用 :data:`~typing.Any` 标注，避免在模块顶层硬依赖
  ``torch``。
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from mosaic.nodes.audio.tts_backends.text_frontends.base import TextFrontend

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["CosyVoiceTokenizer"]


class CosyVoiceTokenizer(TextFrontend):
    """CosyVoice 文本前端。

    负责将原始文本清洗、分词后转换为 CosyVoice LLM 可处理的 token ids
    张量，并维护统一的文本/语音/特殊标记词表空间。

    Parameters
    ----------
    llm_model_path : str, default "Qwen/Qwen2.5-1.5B-Instruct"
        LLM 模型路径（HuggingFace repo id 或本地目录），用于加载
        ``AutoTokenizer`` 进行文本分词。
    speech_token_size : int, default 6561
        语音 token 词表大小（CosyVoice 语义 codec 码本大小，通常为
        ``3**8 = 6561``）。
    speech_token_offset : int, default 151665
        语音 token 在总词表中的起始偏移。按 Token 空间设计，语音 token
        紧跟 LLM 文本词表，因此该值在加载真实 tokenizer 后会与
        ``llm_vocab_size`` 保持一致；在加载前作为 ``llm_vocab_size`` 的
        假设值。
    use_llm_tokenizer : bool, default True
        是否直接使用 LLM 自带的 tokenizer 进行文本编码。为 ``False`` 或
        tokenizer 未加载时，退化为字符级 dummy 分词。

    Attributes
    ----------
    vocab_size : int
        总词表大小（= ``llm_vocab_size + speech_token_size + 特殊标记数``）。
    special_tokens : dict[str, int]
        特殊标记映射，包含 ``<sos>`` / ``<eos>`` / ``<flow>`` / ``<spk>``。
    llm_vocab_size : int
        LLM 文本词表大小。
    speech_token_size : int
        语音 token 词表大小。
    speech_token_offset : int
        语音 token 在总词表中的起始偏移（= ``llm_vocab_size``）。
    sos_token_id / eos_token_id / flow_token_id / spk_token_id : int
        各特殊标记的 token id。
    model_type : str
        模型类型标识，固定为 ``"llm"``。
    """

    # 模型类型标识
    model_type: str = "llm"

    # 特殊标记数量：<sos> / <eos> / <flow> / <spk>
    _NUM_SPECIAL: int = 4

    _logger: logging.Logger = logging.getLogger("mosaic.tts.cosyvoice_tokenizer")

    # ==================================================================
    # 构造函数
    # ==================================================================
    def __init__(
        self,
        llm_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        speech_token_size: int = 6561,
        speech_token_offset: int = 151665,
        use_llm_tokenizer: bool = True,
    ) -> None:
        self.llm_model_path: str = llm_model_path
        self.speech_token_size: int = speech_token_size
        self.speech_token_offset: int = speech_token_offset
        self.use_llm_tokenizer: bool = use_llm_tokenizer

        # LLM 词表大小：在加载真实 tokenizer 前以 speech_token_offset
        # 作为假设值（语音 token 紧跟 LLM 词表）。
        self.llm_vocab_size: int = speech_token_offset

        # LLM tokenizer 实例（惰性加载，load_weights 时填充）
        self._tokenizer: Any = None

        # 根据当前词表布局计算特殊标记 id 与总词表大小
        self._rebuild_vocab()

    # ==================================================================
    # 词表布局
    # ==================================================================
    def _rebuild_vocab(self) -> None:
        """根据当前 ``llm_vocab_size`` / ``speech_token_size`` 重新计算
        特殊标记 id 与总词表大小。

        Token 空间布局::

            [0, llm_vocab_size)                              -> 文本 token
            [llm_vocab_size, llm_vocab_size                  -> 语音 token
                      + speech_token_size)
            sos = llm_vocab_size + speech_token_size
            eos = sos + 1
            flow = sos + 2
            spk = sos + 3
            vocab_size = sos + 4
        """
        base = self.llm_vocab_size + self.speech_token_size
        self.sos_token_id: int = base
        self.eos_token_id: int = base + 1
        self.flow_token_id: int = base + 2
        self.spk_token_id: int = base + 3
        self.special_tokens: dict[str, int] = {
            "<sos>": self.sos_token_id,
            "<eos>": self.eos_token_id,
            "<flow>": self.flow_token_id,
            "<spk>": self.spk_token_id,
        }
        self.vocab_size: int = base + self._NUM_SPECIAL

    # ==================================================================
    # 文本编码（内部辅助）
    # ==================================================================
    def _encode_text(self, text: str) -> list[int]:
        """将纯文本编码为 token id 列表。

        匹配优先级：

        1. 若 ``use_llm_tokenizer`` 为 ``True`` 且 LLM tokenizer 已加载，
           使用 ``AutoTokenizer.encode``（不附加特殊标记）进行 BPE 子词
           切分；
        2. 否则退化为简单的字符级 dummy 分词：``id = ord(ch) % llm_vocab_size``，
           跳过空白字符。便于在无 ``transformers`` / 无权重环境下进行接口
           测试与调试。

        Returns
        -------
        list[int]
            文本 token id 列表，所有 id 落在 ``[0, llm_vocab_size)`` 区间。
        """
        if self.use_llm_tokenizer and self._tokenizer is not None:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            return [int(i) for i in ids]

        # 回退：字符级 dummy 分词
        return [
            ord(ch) % self.llm_vocab_size
            for ch in text
            if not ch.isspace()
        ]

    # ==================================================================
    # 抽象方法实现：tokenize / detokenize
    # ==================================================================
    def tokenize(
        self, text: str, language: str = "zh", **kwargs: Any
    ) -> Any:
        """将文本转换为 token ids 张量。

        处理流程：

        1. 调用 :meth:`preprocess` 清洗文本（Unicode 标准化 + 去多余空白）。
        2. 使用 LLM tokenizer（或字符级回退）将文本编码为 token ids。
        3. 构造输入序列：``<sos> text_tokens <flow>``，其中 ``<flow>``
           标记流匹配声学模型生成的起点。
        4. 返回 ``torch.Tensor``，形状 ``[1, seq_len]``。

        Parameters
        ----------
        text : str
            输入文本。
        language : str, default "zh"
            语言代码（CosyVoice 的文本编码由 LLM tokenizer 统一处理，
            此参数当前仅作记录，不影响编码逻辑）。

        Returns
        -------
        torch.Tensor
            token ids 张量，形状 ``[1, seq_len]``。
        """
        import torch

        # A2-4: CosyVoice 模型内部处理多语言，language 参数仅作记录
        self._logger.debug(
            "CosyVoice tokenizer: language=%r recorded but does not "
            "affect encoding (model handles multilingual internally).",
            language,
        )

        # 1. 文本清洗
        text = self.preprocess(text)

        # 2. 文本编码
        text_ids = self._encode_text(text)

        # 3. 组装序列: <sos> text_tokens <flow>
        ids = [self.sos_token_id, *text_ids, self.flow_token_id]

        # 4. 返回 [1, seq_len] 张量
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def detokenize(self, token_ids: Any) -> str:
        """将 token ids 转回文本（用于调试）。

        - 自动展平 ``[1, seq_len]`` 形状的嵌套输入；
        - 提取 ``<sos>`` 与 ``<flow>`` 之间的文本 token ids（跳过语音
          token 与特殊标记）；
        - 使用 LLM tokenizer 解码；若 tokenizer 不可用，则按字符级回退
          还原。

        Parameters
        ----------
        token_ids : torch.Tensor | list
            token ids。

        Returns
        -------
        str
            还原的文本。
        """
        import torch

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 容错：tokenize 返回形状 [1, seq_len]，tolist() 得到嵌套列表
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        # 定位 <sos> 与 <flow> 之间的文本 token ids
        try:
            start = token_ids.index(self.sos_token_id) + 1
        except ValueError:
            start = 0
        try:
            end = token_ids.index(self.flow_token_id)
        except ValueError:
            end = len(token_ids)

        # 仅保留文本 token（id < llm_vocab_size），跳过语音 token 与特殊标记
        text_ids = [
            int(i)
            for i in token_ids[start:end]
            if 0 <= int(i) < self.llm_vocab_size
        ]

        if self.use_llm_tokenizer and self._tokenizer is not None:
            return self._tokenizer.decode(text_ids)

        # 回退：字符级还原
        return "".join(
            chr(i) for i in text_ids if 0 <= i < 0x110000
        )

    # ==================================================================
    # 说话人编码
    # ==================================================================
    def encode_speaker(
        self, speaker_id: str | np.ndarray | torch.Tensor | None
    ) -> torch.Tensor | None:
        """编码说话人信息（语音克隆参考）。

        CosyVoice 的"说话人"由参考音频的语音 token（``ref_speech_tokens``）
        与说话人嵌入向量（``speaker_embedding``）共同表示。本前端仅做接口
        预留，实际的语音 token 提取与说话人嵌入计算由后端的
        ``SpeechTokenizer`` / ``SpeakerEncoder`` 完成。

        Parameters
        ----------
        speaker_id : str | dict | None
            说话人标识符：

            * ``None`` 表示不使用说话人条件；
            * 字符串表示参考音频文件路径；
            * dict 表示预计算的特征（如 ``{"ref_speech_tokens": ...,
              "speaker_embedding": ...}``）。

        Returns
        -------
        Any | None
            * ``None``：``speaker_id`` 为 ``None``，不使用说话人条件；
            * dict：占位特征 ``{"ref_speech_tokens": None,
              "speaker_embedding": None}``（字符串输入，实际编码由后端
              完成）或预计算特征 dict（dict 输入，原样返回）；
            * 其他类型（如 tensor）：原样返回。
        """
        if speaker_id is None:
            return None

        if isinstance(speaker_id, dict):
            # 预计算特征，原样返回
            return speaker_id

        if isinstance(speaker_id, str):
            # 参考音频路径：实际编码由后端 SpeechTokenizer / SpeakerEncoder
            # 完成，此处返回占位特征 dict。
            return {
                "ref_speech_tokens": None,
                "speaker_embedding": None,
            }

        # 其他类型（如预编码 tensor）原样返回
        return speaker_id

    # ==================================================================
    # 韵律标记插入
    # ==================================================================
    def insert_prosody_tokens(self, text: str, prosody_prompt: str) -> str:
        """在文本中插入韵律控制标记。

        CosyVoice 的韵律控制方式与 ChatTTS 不同：

        - 不使用 ``[laugh_X]`` / ``[break_X]`` 等特殊韵律标记；
        - 通过标点符号（逗号、句号、问号等）自然控制停顿与节奏；
        - 若提供 ``prosody_prompt``，直接将其拼接到文本前（调用基类
          行为）。

        Parameters
        ----------
        text : str
            原始文本。
        prosody_prompt : str
            韵律提示文本（如额外的标点或引导语）。

        Returns
        -------
        str
            插入韵律标记后的文本。
        """
        if prosody_prompt:
            return f"{prosody_prompt} {text}"
        return text

    # ==================================================================
    # 文本预处理
    # ==================================================================
    def preprocess(self, text: str) -> str:
        """文本清洗与标准化。

        在基类清洗基础上执行：

        1. 调用 ``super().preprocess()`` 做 Unicode NFC 标准化、去除控制
           字符并合并多余空白；
        2. 额外去除多余空白（合并连续空白、去除首尾空白）。

        Parameters
        ----------
        text : str
            原始文本。

        Returns
        -------
        str
            清洗后的文本。
        """
        # 1. 基类清洗（Unicode 标准化 + 去控制字符 + 合并空白）
        text = super().preprocess(text)

        # 2. 额外去除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ==================================================================
    # 权重加载 / 释放
    # ==================================================================
    def load_weights(
        self,
        weights_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        """加载 LLM tokenizer 并更新词表大小。

        从 ``llm_model_path``（或 ``weights_path`` 覆盖）加载 LLM 的
        ``AutoTokenizer``，并根据 tokenizer 的真实词表大小更新
        ``llm_vocab_size``、``speech_token_offset``、特殊标记 id 与总
        ``vocab_size``。

        Parameters
        ----------
        weights_path : str
            模型路径（HuggingFace repo id 或本地目录）。非空时覆盖
            ``llm_model_path``；为空字符串时使用构造时传入的
            ``llm_model_path``。
        device : str, default "cuda"
            设备字符串。tokenizer 不占用显存，此参数仅为与声学模型 /
            声码器的 ``load_weights`` 接口保持一致而保留，当前忽略。
        dtype : str, default "float16"
            数据类型字符串。tokenizer 不涉及张量精度，此参数仅为接口
            一致性保留，当前忽略。
        """
        from transformers import AutoTokenizer

        path = weights_path if weights_path else self.llm_model_path
        self._tokenizer = AutoTokenizer.from_pretrained(path)

        # 更新 LLM 词表大小
        llm_vocab = getattr(self._tokenizer, "vocab_size", None)
        if not isinstance(llm_vocab, int) or llm_vocab <= 0:
            # 回退：从词表长度推断
            try:
                llm_vocab = len(self._tokenizer)
            except Exception:
                llm_vocab = self.llm_vocab_size
        self.llm_vocab_size = int(llm_vocab)

        # 语音 token 紧跟 LLM 词表，偏移与 llm_vocab_size 保持一致
        self.speech_token_offset = self.llm_vocab_size

        # 重新计算特殊标记 id 与总词表大小
        self._rebuild_vocab()

    def unload_weights(self) -> None:
        """释放 LLM tokenizer。

        将内部 tokenizer 引用置空，便于显存/内存回收与权重热切换。
        释放后 ``tokenize`` / ``detokenize`` 将退化为字符级 dummy 分词。
        """
        self._tokenizer = None
