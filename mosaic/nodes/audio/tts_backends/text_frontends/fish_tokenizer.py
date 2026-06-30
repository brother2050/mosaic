# mosaic/nodes/audio/tts_backends/text_frontends/fish_tokenizer.py
"""Fish Speech 文本前端实现。

本模块实现 :class:`FishTokenizer`，基于 :class:`TextFrontend` 抽象基类，
为 Fish Speech 声学模型提供文本前端能力，主要职责包括：

* 文本清洗与标点标准化（全角 -> 半角、全角数字 -> 半角）。
* 韵律控制（通过标点符号控制停顿，不使用 ChatTTS 风格的标记）。
* BPE 子词或字符级分词，并将文本映射为 token ids。
* 语音克隆：通过参考音频的 codec tokens 实现音色克隆。
* 组装 Fish Speech 自回归（AR）模型所需的输入序列：

  - 无语音克隆: ``<s> <lang> text_tokens <audio>``
  - 有语音克隆: ``<s> <clone> ref_tokens <lang> text_tokens <audio>``

设计要点
--------
* ``torch`` / ``numpy`` 等重依赖采用惰性导入，使本模块在未安装这些依赖时
  仍可被导入与实例化（仅在实际调用 tokenize 等方法时才报依赖缺失）。
* 词表为空（``vocab_path=''`` 或文件不存在）时退化为简单的字符级分词，
  便于在无权重环境下进行接口测试与调试。
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

__all__ = ["FishTokenizer"]


class FishTokenizer(TextFrontend):
    """Fish Speech 文本前端。

    负责将原始文本清洗、分词后转换为 Fish Speech AR 模型可处理的
    token ids 张量，并支持语音克隆（参考音频 codec tokens）。

    Parameters
    ----------
    vocab_path : str
        词表文件路径。支持 JSON 字典（``{"token": id}``）或每行一个 token
        的纯文本文件（行号即 id）；为空字符串或文件不存在时退化为字符级
        分词。
    text_vocab_size : int
        文本词表大小。文本 token id 范围为 ``[0, text_vocab_size)``。
    audio_vocab_size : int
        音频 codec 词表大小。音频 token id 范围为
        ``[text_vocab_size, text_vocab_size + audio_vocab_size)``。
    language : str, default "zh"
        默认语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"``）。

    Attributes
    ----------
    vocab_size : int
        总词表大小（= text_vocab_size + audio_vocab_size）。
    special_tokens : dict[str, int]
        Fish Speech 特殊标记映射表。
    model_type : str
        模型类型标识，固定为 ``"ar"``。
    """

    # ==================================================================
    # 特殊标记映射表
    # ==================================================================
    SPECIAL_TOKENS: dict[str, int] = {
        "<s>": 0,       # 序列开始
        "</s>": 1,      # 序列结束
        "<text>": 2,    # 文本段开始
        "<audio>": 3,   # 音频段开始（标记从此处开始生成音频）
        "<clone>": 4,   # 语音克隆标记（标记参考音频的开始）
        "<pad>": 5,     # 填充
        "<zh>": 6,      # 中文语言标记
        "<en>": 7,      # 英文语言标记
        "<ja>": 8,      # 日文语言标记
        "<ko>": 9,      # 韩文语言标记
    }

    # 特殊标记映射（与 SPECIAL_TOKENS 一致，供基类协议使用）
    special_tokens: dict[str, int] = SPECIAL_TOKENS

    # 模型类型标识
    model_type: str = "ar"

    _logger: logging.Logger = logging.getLogger("mosaic.tts.fish_tokenizer")

    # 语言代码 -> 语言标记字符串
    _LANG_TOKEN_MAP: dict[str, str] = {
        "zh": "<zh>",
        "en": "<en>",
        "ja": "<ja>",
        "ko": "<ko>",
    }

    # 全角 -> 半角标点映射
    _PUNCT_FULL_TO_HALF: dict[str, str] = {
        "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
        "（": "(", "）": ")", "｛": "{", "｝": "}", "［": "[", "］": "]",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "～": "~", "—": "-", "–": "-",
        "《": "<", "》": ">", "「": '"', "」": '"', "『": '"', "』": '"',
        "【": "[", "】": "]",
        "、": ",",  # 顿号
    }

    # ==================================================================
    # 构造函数
    # ==================================================================
    def __init__(
        self,
        vocab_path: str,
        text_vocab_size: int,
        audio_vocab_size: int,
        language: str = "zh",
    ) -> None:
        self.vocab_path = vocab_path
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.language = language

        # 词表（token -> id）及其逆映射
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self._load_vocab(vocab_path)

        # 总词表大小 = 文本词表 + 音频词表
        self.vocab_size = text_vocab_size + audio_vocab_size

    # ==================================================================
    # 词表加载
    # ==================================================================
    def _load_vocab(self, path: str) -> None:
        """加载词表文件。

        支持 JSON 字典（``{"token": id}``）或每行一个 token 的纯文本文件
        （行号即 id）。路径为空或文件不存在时保持空词表，退化为字符级分词。
        """
        if not path:
            return
        import os

        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return
            if content.startswith("{"):
                import json

                self.vocab = {
                    str(k): int(v) for k, v in json.loads(content).items()
                }
            else:
                idx = 0
                for line in content.splitlines():
                    tok = line.strip()
                    if tok:
                        self.vocab[tok] = idx
                        idx += 1
        except Exception:  # noqa: BLE001
            self.vocab = {}

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ==================================================================
    # 文本编码
    # ==================================================================
    def _encode_text(self, text: str) -> list[int]:
        """将纯文本编码为 token id 列表。

        匹配优先级：

        1. 特殊标记（如 ``<zh>``）-> 对应特殊标记 id；
        2. 若加载了词表，使用贪心最长匹配（近似 BPE 子词切分）；
        3. 否则退化为字符级分词，字符 id = 特殊标记数量 + 字符码点。

        这样在无词表时也能正确识别文本中插入的特殊标记。
        """
        # 按长度降序排列特殊标记，避免 ``<s>`` 被 ``<`` 拆开
        specials = sorted(self.SPECIAL_TOKENS.keys(), key=len, reverse=True)
        base = len(self.SPECIAL_TOKENS)

        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            # 1. 特殊标记匹配
            matched = False
            for tok in specials:
                if text.startswith(tok, i):
                    ids.append(self.SPECIAL_TOKENS[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue

            # 2. 词表贪心最长匹配
            if self.vocab:
                j = min(n, i + 20)
                while j > i:
                    piece = text[i:j]
                    if piece in self.vocab:
                        ids.append(self.vocab[piece])
                        i = j
                        matched = True
                        break
                    j -= 1
                if matched:
                    continue
                # 未知字符：跳过
                i += 1
                continue

            # 3. 字符级分词（字符 id = 特殊标记数量 + 字符码点）
            #    注意：不在此处对越界 id 做截断或取模，以保持 detokenize 的
            #    往返一致性。越界 id 的安全性由模型端 Embedding 层的 clamp
            #    机制保障（见 UnifiedEmbedding.forward）。
            cid = base + ord(text[i])
            ids.append(cid)
            i += 1

        return ids

    # ==================================================================
    # 抽象方法实现：tokenize / detokenize
    # ==================================================================
    def tokenize(self, text: str, language: str = "zh", **kwargs: Any) -> Any:
        """将文本转换为 token ids 张量。

        处理流程：

        1. 调用 :meth:`preprocess` 清洗文本。
        2. 依据 ``language`` 参数选择语言标记（``<zh>`` / ``<en>`` /
           ``<ja>`` / ``<ko>``）。
        3. BPE / 字符级分词。
        4. 构造输入序列：

           - 无语音克隆: ``<s> <lang> text_tokens <audio>``
           - 有语音克隆（``kwargs`` 含 ``ref_tokens``）:
             ``<s> <clone> ref_tokens <lang> text_tokens <audio>``

        5. 返回 ``torch.Tensor``，形状 ``[1, seq_len]``。

        Parameters
        ----------
        text : str
            输入文本。
        language : str, default "zh"
            语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"``）。

        Keyword Arguments
        -----------------
        ref_tokens : torch.Tensor | list[int] | None
            参考音频的 codec token ids，用于语音克隆。提供时在序列中插入
            ``<clone>`` 标记和参考音频 tokens。

        Returns
        -------
        torch.Tensor
            token ids 张量，形状 ``[1, seq_len]``。
        """
        import torch

        # 1. 文本清洗
        text = self.preprocess(text)

        # 2. 语言标记（A2-2: 未知语言发出警告并回退到中文）
        if language not in self._LANG_TOKEN_MAP:
            self._logger.warning(
                "Language %r not supported by Fish tokenizer; "
                "falling back to 'zh'. Supported: %s",
                language, list(self._LANG_TOKEN_MAP.keys()),
            )
        lang_tok_str = self._LANG_TOKEN_MAP.get(language, "<zh>")
        lang_id = self.special_tokens[lang_tok_str]

        # 3. 分词
        text_ids = self._encode_text(text)

        # 4. 结构标记
        bos = self.special_tokens["<s>"]
        audio = self.special_tokens["<audio>"]

        # 5. 语音克隆
        ref_tokens = kwargs.get("ref_tokens", None)
        if ref_tokens is not None:
            clone = self.special_tokens["<clone>"]
            # 将 ref_tokens 展平为一维 int 列表
            ref_list = self._flatten_token_ids(ref_tokens)
            # 组装序列: <s> <clone> ref_tokens <lang> text_tokens <audio>
            ids = [bos, clone, *ref_list, lang_id, *text_ids, audio]
        else:
            # 组装序列: <s> <lang> text_tokens <audio>
            ids = [bos, lang_id, *text_ids, audio]

        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def detokenize(self, token_ids: Any) -> str:
        """将 token ids 转回文本（用于调试）。

        - 自动跳过音频 token（``id >= text_vocab_size`` 的不解码）。
        - 跳过特殊标记（不还原为标记字符串）。
        - 其余通过逆词表或字符级回退还原。
        """
        import torch

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 容错：tokenize 返回形状 [1, seq_len]，tolist() 得到嵌套列表
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        inv_special = {v: k for k, v in self.special_tokens.items()}
        base = len(self.SPECIAL_TOKENS)

        result: list[str] = []
        for tid in token_ids:
            # 跳过音频 token
            if tid >= self.text_vocab_size:
                continue
            # 跳过特殊标记
            if tid in inv_special:
                continue
            # 逆词表查找
            if self.inv_vocab and tid in self.inv_vocab:
                result.append(self.inv_vocab[tid])
            elif tid >= base:
                # 字符级回退
                try:
                    result.append(chr(tid - base))
                except (ValueError, OverflowError):
                    pass
            # 其余未知 token 跳过

        return "".join(result)

    # ==================================================================
    # 辅助方法
    # ==================================================================
    @staticmethod
    def _flatten_token_ids(ref_tokens: Any) -> list[int]:
        """将参考音频 token ids 展平为一维 int 列表。

        支持 ``torch.Tensor``、嵌套 ``list`` 等输入。
        """
        # torch.Tensor -> list（通过 duck typing 避免硬依赖 torch）
        if hasattr(ref_tokens, "tolist"):
            ref_tokens = ref_tokens.tolist()

        # 递归展平嵌套列表
        flat: list[int] = []

        def _flatten(items: Any) -> None:
            if isinstance(items, list):
                for item in items:
                    _flatten(item)
            else:
                flat.append(int(items))

        _flatten(ref_tokens)
        return flat

    # ==================================================================
    # 说话人编码
    # ==================================================================
    def encode_speaker(
        self, speaker_id: str | np.ndarray | torch.Tensor | None
    ) -> torch.Tensor | None:
        """编码说话人信息（语音克隆参考）。

        Fish Speech 的"说话人"是参考音频的 codec tokens，而非传统的
        说话人嵌入向量。

        * ``speaker_id`` 为 ``None`` 时返回 ``None``（无克隆，使用默认音色）。
        * ``speaker_id`` 为预编码的 token ids tensor 时，直接返回。
        * ``speaker_id`` 为字符串（音频文件路径）时，返回路径字符串本身。
          实际的音频编码逻辑在
          ``FishLlamaARModel.encode_reference_audio`` 中实现，
          此处仅做接口预留。

        Parameters
        ----------
        speaker_id : str | torch.Tensor | None
            说话人标识符：``None`` 表示无克隆；字符串表示音频文件路径；
            tensor 表示预编码的参考音频 codec tokens。

        Returns
        -------
        torch.Tensor | str | None
            说话人编码结果。
        """
        if speaker_id is None:
            return None

        if isinstance(speaker_id, str):
            # 音频文件路径，由后端（FishLlamaARModel.encode_reference_audio）
            # 负责实际的音频编码
            return speaker_id

        # 预编码的 token ids tensor，直接返回
        return speaker_id

    # ==================================================================
    # 韵律标记插入
    # ==================================================================
    def insert_prosody_tokens(self, text: str, prosody_prompt: str) -> str:
        """在文本中插入韵律控制标记。

        Fish Speech 的韵律控制方式与 ChatTTS 不同：

        - 不使用 ``[laugh_X]`` 等 ChatTTS 风格的标记。
        - ``prosody_prompt`` 可以包含标点符号用于控制停顿。
        - 直接将 ``prosody_prompt`` 拼接到文本前（调用基类方法）。

        Parameters
        ----------
        text : str
            原始文本。
        prosody_prompt : str
            韵律提示文本（如标点符号）。

        Returns
        -------
        str
            插入韵律标记后的文本。
        """
        return super().insert_prosody_tokens(text, prosody_prompt)

    # ==================================================================
    # 文本预处理
    # ==================================================================
    def preprocess(self, text: str) -> str:
        """文本清洗与标准化。

        在基类清洗基础上执行：

        1. 调用 ``super().preprocess()`` 做 Unicode 标准化与空白合并；
        2. 中文标点标准化（全角 -> 半角统一）；
        3. 数字处理（全角数字 -> 半角，简单实现）；
        4. 过滤不支持的字符（保留 CJK、日文假名、韩文、英文字母、数字、
           常见标点、空白）。
        """
        # 1. 基本清洗（Unicode NFC + 去控制字符 + 合并空白）
        text = super().preprocess(text)

        # 2. 全角 -> 半角标点
        text = "".join(self._PUNCT_FULL_TO_HALF.get(ch, ch) for ch in text)

        # 3. 全角数字 -> 半角（简单实现）
        text = "".join(
            chr(ord(ch) - 0xFEE0) if "０" <= ch <= "９" else ch
            for ch in text
        )

        # 4. 过滤不支持的字符
        #    保留：CJK 统一表意文字、平假名、片假名、韩文音节、
        #          英文字母、数字、常见标点、空白
        text = re.sub(
            "[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
            "a-zA-Z0-9,.!?;:'\"()\\[\\]\\s]",
            "",
            text,
        )

        return text.strip()
