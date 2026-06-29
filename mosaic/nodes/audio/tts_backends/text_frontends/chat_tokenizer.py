"""ChatTTS 文本前端实现。

本模块实现 :class:`ChatTokenizer`，基于 :class:`TextFrontend` 抽象基类，
为 ChatTTS 声学模型提供文本前端能力，主要职责包括：

* 文本清洗与标点标准化（全角 -> 半角、阿拉伯数字 -> 中文读法）。
* 韵律控制标记的插入（停顿 / 笑声 / 口语化 / 语速）。
* BPE 子词或字符级分词，并将文本映射为 token ids。
* 说话人嵌入的解码（Base16384 -> LZMA2 -> tensor）。
* 组装 ChatTTS 自回归（AR）模型所需的输入序列：

  ``[Stts] [spk_emb]/[empty_spk] <语言标记> <文本 token> [Ptts]``

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

import re
from typing import Any

from mosaic.nodes.audio.tts_backends.text_frontends.base import TextFrontend

__all__ = ["ChatTokenizer"]


class ChatTokenizer(TextFrontend):
    """ChatTTS 文本前端。

    负责将原始文本清洗、标注韵律、分词后转换为 ChatTTS AR 模型可处理的
    token ids 张量，并支持说话人嵌入的解码。

    Parameters
    ----------
    vocab_path : str
        词表文件路径。支持 JSON 字典（``{"token": id}``）或每行一个 token
        的纯文本文件（行号即 id）；为空字符串或文件不存在时退化为字符级
        分词。
    merges_path : str, optional
        BPE 合并规则文件路径（可选；当前为简化实现，仅记录于实例属性）。
    num_vq : int, default 4
        VQ 码本组数。
    sample_rate : int, default 24000
        采样率。

    Attributes
    ----------
    vocab_size : int
        词表大小（含特殊标记）。
    special_tokens : dict[str, int]
        完整的特殊标记映射表。
    model_type : str
        模型类型标识，固定为 ``"ar"``。
    """

    # ==================================================================
    # 特殊标记映射表
    # ==================================================================
    SPECIAL_TOKENS: dict[str, int] = {
        # 结构标记
        "[Stts]": 0,       # 文本开始
        "[Ptts]": 1,       # 文本结束
        "[spk_emb]": 2,    # 说话人嵌入位置
        "[empty_spk]": 3,  # 空说话人
        # 情感标记
        "[laugh_0]": 4,    # 轻微笑声
        "[laugh_1]": 5,    # 中等笑声
        "[laugh_2]": 6,    # 明显笑声
        # 停顿标记
        "[uv_break]": 7,   # 呼吸停顿
        "[lbreak]": 8,     # 长停顿
        # 可调节停顿 break_0~break_7 (200-800ms)
        "[break_0]": 9, "[break_1]": 10, "[break_2]": 11, "[break_3]": 12,
        "[break_4]": 13, "[break_5]": 14, "[break_6]": 15, "[break_7]": 16,
        # 口语化程度 oral_0~oral_9
        "[oral_0]": 17, "[oral_1]": 18, "[oral_2]": 19, "[oral_3]": 20,
        "[oral_4]": 21, "[oral_5]": 22, "[oral_6]": 23, "[oral_7]": 24,
        "[oral_8]": 25, "[oral_9]": 26,
        # 语速控制 speed_0~speed_9
        "[speed_0]": 27, "[speed_1]": 28, "[speed_2]": 29, "[speed_3]": 30,
        "[speed_4]": 31, "[speed_5]": 32, "[speed_6]": 33, "[speed_7]": 34,
        "[speed_8]": 35, "[speed_9]": 36,
    }

    # 特殊标记映射（与 SPECIAL_TOKENS 一致，供基类协议使用）
    special_tokens: dict[str, int] = SPECIAL_TOKENS

    # 模型类型标识
    model_type: str = "ar"

    # 全角 -> 半角标点映射
    _PUNCT_FULL_TO_HALF: dict[str, str] = {
        "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
        "（": "(", "）": ")", "｛": "{", "｝": "}", "［": "[", "］": "]",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "～": "~", "—": "-", "–": "-",
        "《": "<", "》": ">", "「": '"', "」": '"', "『": '"', "』": '"',
        "【": "[", "】": "]",
    }

    # 阿拉伯数字 -> 中文数字（简单逐字符替换）
    _DIGIT_TO_CN: dict[str, str] = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
    }

    # Base16384 字符基址（使用 CJK 统一表意文字区段，共 16384 个码位）
    _BASE16384_OFFSET: int = 0x4E00

    # ==================================================================
    # 构造函数
    # ==================================================================
    def __init__(
        self,
        vocab_path: str,
        merges_path: str | None = None,
        num_vq: int = 4,
        sample_rate: int = 24000,
        num_text_tokens: int = 21178,
    ) -> None:
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.num_vq = num_vq
        self.sample_rate = sample_rate
        self.text_vocab_size = num_text_tokens

        # 词表（token -> id）及其逆映射
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.unk_token: str = "<unk>"
        self._load_vocab(vocab_path)

        # 计算词表大小（含特殊标记）
        if self.vocab:
            self.vocab_size = max(max(self.vocab.values()) + 1,
                                  len(self.SPECIAL_TOKENS))
        else:
            self.vocab_size = len(self.SPECIAL_TOKENS)

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
        except Exception:
            self.vocab = {}

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ==================================================================
    # 文本编码
    # ==================================================================
    def _encode_text(self, text: str) -> list[int]:
        """将纯文本编码为 token id 列表。

        匹配优先级：

        1. 特殊标记（如 ``[break_2]``）-> 对应特殊标记 id；
        2. 若加载了词表，使用贪心最长匹配（近似 BPE 子词切分）；
        3. 否则退化为字符级分词，字符 id = 特殊标记数量 + 字符码点。

        这样在无词表时也能正确识别文本中插入的韵律标记。
        """
        # 按长度降序排列特殊标记，避免 ``[break_0]`` 被 ``[`` 拆开
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
                # 未知字符：使用 unk 标记（若词表中存在），否则跳过该字符
                if self.unk_token in self.vocab:
                    ids.append(self.vocab[self.unk_token])
                i += 1
                continue

            # 3. 字符级分词（字符 id = 特殊标记数量 + 字符码点）
            #    注意：不在此处对越界 id 做截断或取模，以保持 detokenize 的
            #    往返一致性。越界 id 的安全性由模型端 Embedding 层的 clamp
            #    机制保障（见 DualEmbedding / UnifiedEmbedding.forward）。
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
        2. 依据 ``kwargs['prosody_prompt']`` 调用
           :meth:`insert_prosody_tokens` 插入韵律标记。
        3. BPE / 字符级分词（自动识别已插入的韵律特殊标记）。
        4. 添加结构标记 ``[Stts]``（开始）与 ``[Ptts]``（结束）。
        5. 添加说话人标记：``[spk_emb]`` 或 ``[empty_spk]``。
        6. 添加语言标记（``zh`` / ``en``）。
        7. 返回 ``torch.Tensor``，形状 ``[1, seq_len]``。

        Parameters
        ----------
        text : str
            输入文本。
        language : str, default "zh"
            语言代码（``"zh"`` 或 ``"en"``）。

        Keyword Arguments
        -----------------
        prosody_prompt : str
            韵律提示，如 ``"[oral_2][laugh_0][break_4]"``。
        speaker_id : str | None
            说话人标识。非空时使用 ``[spk_emb]``，为空时使用 ``[empty_spk]``。

        Returns
        -------
        torch.Tensor
            token ids 张量，形状 ``[1, seq_len]``。
        """
        import torch

        # 1. 文本清洗
        text = self.preprocess(text)

        # 2. 插入韵律标记
        prosody_prompt = kwargs.get("prosody_prompt", "")
        if prosody_prompt:
            text = self.insert_prosody_tokens(text, prosody_prompt)

        # 3. 分词（自动识别韵律特殊标记）
        text_ids = self._encode_text(text)

        # 4. 结构标记
        stts = self.special_tokens["[Stts]"]
        ptts = self.special_tokens["[Ptts]"]

        # 5. 说话人标记
        speaker_id = kwargs.get("speaker_id", None)
        if speaker_id is not None:
            spk_token = self.special_tokens["[spk_emb]"]
        else:
            spk_token = self.special_tokens["[empty_spk]"]

        # 6. 语言标记（将 "zh"/"en" 编码为 token，作为语言条件信号）
        lang_ids = self._encode_text(language) if language else []

        # 7. 组装序列：[Stts] [spk] <lang> <text> [Ptts]
        ids = [stts, spk_token] + lang_ids + text_ids + [ptts]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def detokenize(self, token_ids: Any) -> str:
        """将 token ids 转回文本（用于调试）。

        特殊标记还原为标记字符串，其余通过逆词表或字符级回退还原。
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
            if tid in inv_special:
                result.append(inv_special[tid])
            elif self.inv_vocab and tid in self.inv_vocab:
                result.append(self.inv_vocab[tid])
            elif tid >= base:
                # 字符级回退
                try:
                    result.append(chr(tid - base))
                except (ValueError, OverflowError):
                    result.append(f"<{tid}>")
            else:
                result.append(f"<{tid}>")
        return "".join(result)

    # ==================================================================
    # 说话人编码
    # ==================================================================
    def encode_speaker(self, speaker_id: str | None) -> Any | None:
        """将说话人 ID 解码为嵌入张量。

        * ``speaker_id`` 为 ``None`` 时返回 ``None``（使用 ``[empty_spk]``）。
        * ``speaker_id`` 为字符串（Base16384 编码的说话人嵌入）时，按
          ``Base16384 解码 -> LZMA2 解压 -> torch.tensor`` 流程还原。

        Parameters
        ----------
        speaker_id : str | None
            说话人标识符。

        Returns
        -------
        torch.Tensor | None
            说话人嵌入张量；``None`` 表示不使用说话人条件。
        """
        if speaker_id is None:
            return None

        if isinstance(speaker_id, str):
            try:
                return self._decode_speaker(speaker_id)
            except Exception:
                # 解码失败时优雅降级，返回 None
                return None

        return None

    def _decode_speaker(self, s: str) -> Any:
        """解码 Base16384 + LZMA2 编码的说话人嵌入。

        解码流程与 ChatTTS 编码端对称：

        1. Base16384 解码为原始压缩字节；
        2. LZMA2（FORMAT_RAW）解压；
        3. 字节按 float16 解释为 ``torch.Tensor``。
        """
        import lzma
        import struct
        import torch

        # 1. Base16384 解码
        raw = self._base16384_decode(s)
        if not raw:
            # 无效输入（无可解码字符）-> 抛出，交由 encode_speaker 降级为 None
            raise ValueError("empty base16384 payload")

        # 2. LZMA2 解压（FORMAT_RAW，与 ChatTTS 编码端一致）
        data = lzma.decompress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[
                {"id": lzma.FILTER_LZMA2,
                 "preset": 7 | lzma.PRESET_EXTREME}
            ],
        )

        # 3. 字节 -> tensor（ChatTTS 使用 float16 存储说话人嵌入）
        n = len(data) // 2
        if n == 0:
            raise ValueError("empty decompressed payload")
        # 'e' 为 IEEE 754 半精度浮点（Python 3.6+ 支持）
        values = struct.unpack("<" + "e" * n, bytes(data[: n * 2]))
        return torch.tensor(values, dtype=torch.float16)

    def _base16384_decode(self, s: str) -> bytes:
        """简化的 Base16384 解码。

        编码方案：将每 7 字节（56 bit）拆分为 4 个 14-bit 值，每个值映射到
        一个 Unicode 字符（码点 = ``_BASE16384_OFFSET + 值``）。解码执行
        逆过程：每 4 个字符还原为 7 字节。

        说明：此处为简化实现，使用连续的 CJK 区段作为字母表；与官方
        ``base16384`` 库在字母表顺序上可能不同，但编解码自洽，接口正确。
        """
        values: list[int] = []
        for ch in s:
            v = ord(ch) - self._BASE16384_OFFSET
            if 0 <= v < 16384:
                values.append(v)
            # 非字母表字符忽略

        out = bytearray()
        # 每 4 个 14-bit 值 -> 56 bit -> 7 字节
        full = len(values) - (len(values) % 4)
        for i in range(0, full, 4):
            v0, v1, v2, v3 = (
                values[i], values[i + 1], values[i + 2], values[i + 3]
            )
            bits = (v0 << 42) | (v1 << 28) | (v2 << 14) | v3
            out.extend(bits.to_bytes(7, "big"))

        # 处理剩余的 1~3 个值（对应 2/4/5 字节，按可用比特尽力还原）
        rem = values[full:]
        if rem:
            bits = 0
            for v in rem:
                bits = (bits << 14) | v
            nbits = 14 * len(rem)
            nbytes = nbits // 8
            if nbytes > 0:
                bits >>= nbits - nbytes * 8
                out.extend(bits.to_bytes(nbytes, "big"))

        return bytes(out)

    # ==================================================================
    # 韵律标记插入
    # ==================================================================
    def insert_prosody_tokens(self, text: str, prosody_prompt: str) -> str:
        """在文本中插入韵律控制标记。

        解析 ``prosody_prompt`` 中的标记并按类型插入：

        * ``[oral_X]`` / ``[speed_X]``：插入到文本最开头（全局风格控制）；
        * ``[laugh_X]``：插入到句子开头（口语化/语速标记之后）；
        * ``[break_X]``：插入到每个标点之后（中英文标点均识别）。

        Parameters
        ----------
        text : str
            原始文本。
        prosody_prompt : str
            韵律提示，如 ``"[oral_2][laugh_0][break_4]"``。

        Returns
        -------
        str
            插入韵律标记后的文本。
        """
        if not prosody_prompt:
            return text

        # 解析所有形如 [xxx] 的标记
        tokens = re.findall(r"\[[^\]]+\]", prosody_prompt)

        break_tokens = [t for t in tokens if t.startswith("[break_")]
        laugh_tokens = [t for t in tokens if t.startswith("[laugh_")]
        oral_tokens = [t for t in tokens if t.startswith("[oral_")]
        speed_tokens = [t for t in tokens if t.startswith("[speed_")]

        # 口语化与语速标记插入到文本最开头；笑声标记紧随其后（句子开头）
        prefix = "".join(oral_tokens) + "".join(speed_tokens)
        prefix += "".join(laugh_tokens)
        result = prefix + text

        # 停顿标记插入到每个标点之后（同时识别半角与全角标点）
        if break_tokens:
            break_str = "".join(break_tokens)
            result = re.sub(
                r"([,.!?;:。！？；：、])",
                lambda m: m.group(1) + break_str,
                result,
            )

        return result

    # ==================================================================
    # 文本预处理
    # ==================================================================
    def preprocess(self, text: str) -> str:
        """文本清洗与标准化。

        在基类清洗基础上执行：

        1. 调用 ``super().preprocess()`` 做 Unicode 标准化与空白合并；
        2. 中文标点标准化（全角 -> 半角统一）；
        3. 数字处理：阿拉伯数字转中文读法（简单逐字符替换）；
        4. 过滤不支持的字符（保留中英文、标点、数字、空格）。
        """
        # 1. 基本清洗
        text = super().preprocess(text)

        # 2. 全角 -> 半角标点
        text = "".join(self._PUNCT_FULL_TO_HALF.get(ch, ch) for ch in text)

        # 3. 阿拉伯数字 -> 中文读法
        text = "".join(self._DIGIT_TO_CN.get(ch, ch) for ch in text)

        # 4. 过滤不支持的字符（保留 CJK、英文字母、数字、常见标点、空白）
        text = re.sub(
            "[^\u4e00-\u9fffa-zA-Z0-9,.!?;:'\"()\\[\\]\\s]",
            "",
            text,
        )

        return text.strip()
