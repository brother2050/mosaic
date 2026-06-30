# mosaic/nodes/audio/tts_backends/text_frontends/sovits_tokenizer.py
"""GPT-SoVITS 文本前端：音素级文本处理。

文件路径: mosaic/nodes/audio/tts_backends/text_frontends/sovits_tokenizer.py

Layer 1: 文本前端层。负责文本清洗 → 音素化（拼音/ARPAbet）→ token ids。

与 ChatTTS/Fish 的关键差异
---------------------------
* GPT-SoVITS 使用**音素级**文本处理，不是字符级或 BPE。
* 中文使用**带声调的拼音**作为音素（如 ``ni3 hao3``）。
* 英文使用 **ARPAbet** 音素（如 ``HH AH L OW``）。
* 文本中嵌入语言标记 ``[ZH]`` / ``[EN]`` / ``[JA]`` 和停顿标记 ``_``。

G2P 实现说明
------------
* 中文 G2P 不依赖 ``pypinyin``，内置简化拼音映射表。如果 ``pypinyin`` 可用
  则自动使用以获得更好的多音字消歧。
* 英文 G2P 使用内置极简 CMU 词典子集 + 字母规则回退。
* 内置 G2P 的质量有限，在 docstring 中标注了局限性。
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from mosaic.nodes.audio.tts_backends.text_frontends.base import TextFrontend

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["SoVITSTokenizer"]


# ---------------------------------------------------------------------------
# 特殊标记
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: dict[str, int] = {
    "<s>": 0,
    "</s>": 1,
    "_": 2,       # 空白/停顿
    "#": 3,       # 声调标记前缀
    "[ZH]": 4,    # 中文语言标记
    "[EN]": 5,    # 英文语言标记
    "[JA]": 6,    # 日文语言标记
    "[SPLIT]": 7, # 句子分割标记
    "[SPK]": 8,   # 说话人标记位置
}

_NUM_SPECIAL = len(SPECIAL_TOKENS)  # 9

# ---------------------------------------------------------------------------
# 中文拼音映射表（简化版）
# ---------------------------------------------------------------------------
# 声母（含翘舌音 zh/ch/sh）
_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "w", "y",
]
# 声母优先匹配顺序（长前短后，保证 zh 优先于 z）
_INITIAL_ORDER = sorted(_INITIALS, key=len, reverse=True)
# 韵母
_FINALS = [
    "a", "o", "e", "i", "u", "v",  # v 代表 ü
    "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ve", "er",
    "an", "en", "in", "un", "vn",
    "ang", "eng", "ing", "ong",
]

# 常用汉字→拼音映射（极小子集，实际使用 pypinyin 效果更好）
_COMMON_PINYIN: dict[str, str] = {
    "你": "ni3", "好": "hao3", "世": "shi4", "界": "jie4",
    "中": "zhong1", "国": "guo2", "人": "ren2", "大": "da4",
    "小": "xiao3", "的": "de5", "是": "shi4", "不": "bu4",
    "我": "wo3", "他": "ta1", "她": "ta1", "它": "ta1",
    "们": "men5", "有": "you3", "一": "yi1", "个": "ge4",
    "上": "shang4", "下": "xia4", "左": "zuo3", "右": "you4",
    "前": "qian2", "后": "hou4", "里": "li3", "外": "wai4",
    "天": "tian1", "地": "di4", "日": "ri4", "月": "yue4",
    "水": "shui3", "火": "huo3", "山": "shan1", "河": "he2",
    "说": "shuo1", "话": "hua4", "听": "ting1", "看": "kan4",
    "来": "lai2", "去": "qu4", "吃": "chi1", "喝": "he1",
    "走": "zou3", "跑": "pao3", "飞": "fei1", "坐": "zuo4",
    "学": "xue2", "教": "jiao1", "读": "du2", "写": "xie3",
    "音": "yin1", "乐": "le4", "声": "sheng1", "调": "diao4",
    "高": "gao1", "低": "di1", "快": "kuai4", "慢": "man4",
    "长": "chang2", "短": "duan3", "多": "duo1", "少": "shao3",
    "好": "hao3", "坏": "huai4", "美": "mei3", "丑": "chou3",
    "新": "xin1", "旧": "jiu4", "早": "zao3", "晚": "wan3",
    "今": "jin1", "明": "ming2", "昨": "zuo2", "年": "nian2",
    "月": "yue4", "日": "ri4", "时": "shi2", "分": "fen1",
    "秒": "miao3", "点": "dian3", "钟": "zhong1",
}

# 数字→中文读法
_DIGIT_TO_CN: dict[str, str] = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

# ---------------------------------------------------------------------------
# 英文 ARPAbet 音素
# ---------------------------------------------------------------------------
_ARPABET: list[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH", "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

# 极简 CMU 词典（常见词）
_CMU_DICT: dict[str, list[str]] = {
    "HELLO": ["HH", "AH", "L", "OW"],
    "WORLD": ["W", "ER", "L", "D"],
    "THE": ["DH", "AH"],
    "A": ["AH"],
    "IS": ["IH", "Z"],
    "IT": ["IH", "T"],
    "AND": ["AE", "N", "D"],
    "TO": ["T", "UW"],
    "OF": ["AH", "V"],
    "IN": ["IH", "N"],
    "FOR": ["F", "AO", "R"],
    "ON": ["AA", "N"],
    "WITH": ["W", "IH", "DH"],
    "YES": ["Y", "EH", "S"],
    "NO": ["N", "OW"],
    "GOOD": ["G", "UH", "D"],
    "MORNING": ["M", "AO", "R", "N", "IH", "NG"],
    "HELLO": ["HH", "AH", "L", "OW"],
    "HI": ["HH", "AY"],
    "BYE": ["B", "AY"],
    "THANK": ["TH", "AE", "NG", "K"],
    "YOU": ["Y", "UW"],
    "ME": ["M", "IY"],
    "WE": ["W", "IY"],
    "THEY": ["DH", "EY"],
    "HE": ["HH", "IY"],
    "SHE": ["SH", "IY"],
    "ARE": ["AA", "R"],
    "WAS": ["W", "AH", "Z"],
    "WERE": ["W", "ER"],
    "HAVE": ["HH", "AE", "V"],
    "HAS": ["HH", "AE", "Z"],
    "HAD": ["HH", "AE", "D"],
    "DO": ["D", "UW"],
    "DOES": ["D", "AH", "Z"],
    "DID": ["D", "IH", "D"],
    "WILL": ["W", "IH", "L"],
    "WOULD": ["W", "UH", "D"],
    "CAN": ["K", "AE", "N"],
    "COULD": ["K", "UH", "D"],
    "SHOULD": ["SH", "UH", "D"],
    "MAY": ["M", "EY"],
    "MIGHT": ["M", "AY", "T"],
    "MUST": ["M", "AH", "S", "T"],
    "THIS": ["DH", "IH", "S"],
    "THAT": ["DH", "AE", "T"],
    "THESE": ["DH", "IY", "Z"],
    "THOSE": ["DH", "OW", "Z"],
}

# 字母→ARPAbet 回退
_LETTER_TO_ARPABET: dict[str, list[str]] = {
    "A": ["EY"], "B": ["B", "IY"], "C": ["S", "IY"],
    "D": ["D", "IY"], "E": ["IY"], "F": ["EH", "F"],
    "G": ["JH", "IY"], "H": ["EY", "CH"], "I": ["AY"],
    "J": ["JH", "EY"], "K": ["K", "EY"], "L": ["EH", "L"],
    "M": ["EH", "M"], "N": ["EH", "N"], "O": ["OW"],
    "P": ["P", "IY"], "Q": ["K", "Y", "UW"], "R": ["AA", "R"],
    "S": ["EH", "S"], "T": ["T", "IY"], "U": ["Y", "UW"],
    "V": ["V", "IY"], "W": ["D", "AH", "B", "AH", "L", "Y", "UW"],
    "X": ["EH", "K", "S"], "Y": ["W", "AY"], "Z": ["Z", "IY"],
}

# 全角→半角标点映射
_FULLWIDTH_PUNCT: dict[str, str] = {
    "，": ",", "。": ".", "！": "!", "？": "?",
    "；": ";", "：": ":", "（": "(", "）": ")",
    "「": '"', "」": '"', "『": '"', "』": '"',
    "【": "[", "】": "]", "《": "<", "》": ">",
    "、": ",", "～": "~",
}


class SoVITSTokenizer(TextFrontend):
    """GPT-SoVITS 音素级文本前端。

    将原始文本转换为音素 token ids。中文使用带声调的拼音（如 ``ni3``），
    英文使用 ARPAbet 音素（如 ``HH AH L OW``）。

    Attributes
    ----------
    model_type : str
        固定为 ``"ar"``。
    special_tokens : dict[str, int]
        特殊标记映射。
    vocab_size : int
        词表大小。
    """

    model_type: str = "ar"
    special_tokens: dict[str, int] = SPECIAL_TOKENS

    _logger: logging.Logger = logging.getLogger("mosaic.tts.sovits_tokenizer")

    def __init__(
        self,
        vocab_path: str = "",
        language: str = "zh",
        bert_model: str | None = None,
        add_blank: bool = True,
    ) -> None:
        """初始化 SoVITSTokenizer。

        Parameters
        ----------
        vocab_path : str
            音素词表文件路径。空字符串时使用内置词表。
        language : str
            默认语言代码。
        bert_model : str | None
            BERT 模型路径（GPT-SoVITS v2 辅助特征用）。
        add_blank : bool
            是否在音素间插入空白 token。
        """
        self._language: str = language
        self._bert_model: str | None = bert_model
        self._add_blank: bool = add_blank

        # 构建内置音素词表
        self._phoneme_to_id: dict[str, int] = {}
        # 特殊标记
        for tok, tid in SPECIAL_TOKENS.items():
            self._phoneme_to_id[tok] = tid
        # 中文拼音音素（声母+韵母+声调组合）
        idx = _NUM_SPECIAL
        for init in _INITIALS:
            for tone in range(1, 6):
                self._phoneme_to_id[f"{init}{tone}"] = idx
                idx += 1
        for fin in _FINALS:
            for tone in range(1, 6):
                self._phoneme_to_id[f"{fin}{tone}"] = idx
                idx += 1
        # 英文 ARPAbet
        for ph in _ARPABET:
            self._phoneme_to_id[ph] = idx
            idx += 1

        self.vocab_size = idx

        # 尝试加载外部词表
        if vocab_path:
            import json
            import os

            if os.path.exists(vocab_path):
                with open(vocab_path, encoding="utf-8") as f:
                    ext_vocab = json.load(f)
                self._phoneme_to_id.update(ext_vocab)
                self.vocab_size = max(
                    self._phoneme_to_id.values(), default=0
                ) + 1

        # 逆向映射
        self._id_to_phoneme: dict[int, str] = {
            v: k for k, v in self._phoneme_to_id.items()
        }

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------
    def tokenize(
        self, text: str, language: str = "zh", **kwargs: Any
    ) -> Any:
        """将文本转换为音素 token ids。

        Parameters
        ----------
        text : str
            输入文本。
        language : str
            语言代码（``"zh"`` / ``"en"`` / ``"ja"``）。
        **kwargs : Any
            ``ref_phonemes``：参考音频的音素列表（语音克隆用）。

        Returns
        -------
        torch.Tensor
            token ids，shape ``[1, seq_len]``。
        """
        import torch

        processed = self.preprocess(text)
        ref_phonemes = kwargs.get("ref_phonemes")

        # 音素化
        if language == "zh":
            phonemes = self._g2p_chinese(processed)
            lang_token = SPECIAL_TOKENS["[ZH]"]
        elif language == "en":
            phonemes = self._g2p_english(processed)
            lang_token = SPECIAL_TOKENS["[EN]"]
        elif language == "ja":
            # 日文简化处理：按字符分，使用 [JA] 标记
            phonemes = list(processed.replace(" ", ""))
            lang_token = SPECIAL_TOKENS["[JA]"]
        else:
            # A2-3: ko/yue 等声明支持但实际走中文 G2P，发出警告
            self._logger.warning(
                "Language %r declared as supported but uses Chinese G2P "
                "as fallback; pronunciation may not be accurate.",
                language,
            )
            phonemes = self._g2p_chinese(processed)
            lang_token = SPECIAL_TOKENS["[ZH]"]

        # 查词表
        token_ids: list[int] = [SPECIAL_TOKENS["<s>"], lang_token]
        for ph in phonemes:
            tid = self._phoneme_to_id.get(ph, SPECIAL_TOKENS["_"])
            token_ids.append(tid)
            if self._add_blank:
                token_ids.append(SPECIAL_TOKENS["_"])
        token_ids.append(SPECIAL_TOKENS["</s>"])

        # 语音克隆：在前面插入参考音素
        if ref_phonemes is not None:
            ref_ids: list[int] = []
            for ph in ref_phonemes:
                tid = self._phoneme_to_id.get(ph, SPECIAL_TOKENS["_"])
                ref_ids.append(tid)
                if self._add_blank:
                    ref_ids.append(SPECIAL_TOKENS["_"])
            token_ids = (
                [SPECIAL_TOKENS["<s>"]]
                + ref_ids
                + [SPECIAL_TOKENS["[SPLIT]"]]
                + token_ids[1:]
            )

        return torch.tensor([token_ids], dtype=torch.long)

    def detokenize(self, token_ids: Any) -> str:
        """将 token ids 转回音素字符串。

        Parameters
        ----------
        token_ids : torch.Tensor | list
            token ids。

        Returns
        -------
        str
            音素字符串。
        """
        if hasattr(token_ids, "tolist"):
            ids = token_ids.tolist()
        elif isinstance(token_ids, list):
            ids = token_ids
        else:
            ids = list(token_ids)

        # Flatten if nested (e.g., [[1, 2, 3]] -> [1, 2, 3])
        if ids and isinstance(ids[0], list):
            ids = ids[0]

        phonemes: list[str] = []
        for tid in ids:
            ph = self._id_to_phoneme.get(tid, "")
            if ph in ("<s>", "</s>", "_", "#"):
                continue
            if ph.startswith("[") and ph.endswith("]"):
                continue
            if ph:
                phonemes.append(ph)
        return " ".join(phonemes)

    # ------------------------------------------------------------------
    # G2P 实现
    # ------------------------------------------------------------------
    def _split_pinyin(self, pinyin: str) -> list[str]:
        """将完整拼音音节拆分为声母+韵母（均带声调）。

        例如::

            "ni3"   -> ["n3", "i3"]
            "hao3"  -> ["h3", "ao3"]
            "shi4"  -> ["sh4", "i4"]
            "zhong1"-> ["zh1", "ong1"]
            "a1"    -> ["a1"]       # 无声母
            "er2"   -> ["er2"]      # 无声母

        Parameters
        ----------
        pinyin : str
            带声调数字的拼音音节，如 ``"ni3"``。

        Returns
        -------
        list[str]
            拆分后的音素列表，每个音素都带声调数字。
        """
        # 提取声调数字
        tone = ""
        for c in reversed(pinyin):
            if c.isdigit():
                tone = c + tone
            else:
                break

        if not tone:
            tone = "5"  # 轻声

        syllable = pinyin[: -len(tone)] if tone else pinyin
        if not syllable:
            return []

        # 尝试匹配声母（长前短后，保证 zh 优先于 z）
        initial = ""
        for ini in _INITIAL_ORDER:
            if syllable.startswith(ini):
                initial = ini
                break

        final = syllable[len(initial):] if initial else syllable

        result: list[str] = []
        if initial:
            result.append(f"{initial}{tone}")
        if final:
            result.append(f"{final}{tone}")
        return result

    # ------------------------------------------------------------------
    def _g2p_chinese(self, text: str) -> list[str]:
        """中文文本转带声调拼音音素。

        使用内置简化映射表。如果 ``pypinyin`` 可用则自动使用。
        每个拼音音节会通过 :meth:`_split_pinyin` 拆分为声母+韵母。

        Parameters
        ----------
        text : str
            中文文本。

        Returns
        -------
        list[str]
            拼音音素列表，如 ``["n3", "i3", "h3", "ao3"]``。

        Notes
        -----
        内置映射表覆盖的汉字有限，多音字使用默认读音。
        生产环境建议安装 ``pypinyin``。
        """
        # 尝试 pypinyin
        try:
            from pypinyin import pinyin, Style  # type: ignore

            result: list[str] = []
            for char in text:
                if char.isspace():
                    continue
                if char in _FULLWIDTH_PUNCT or char in ",.!?;:":
                    result.append("_")
                    continue
                if "\u4e00" <= char <= "\u9fff":
                    py_list = pinyin(char, style=Style.TONE3)
                    if py_list and py_list[0]:
                        result.extend(self._split_pinyin(py_list[0][0].lower()))
                elif char.isascii() and char.isalpha():
                    # 英文字符切换到英文 G2P
                    result.extend(self._g2p_english(char))
                elif char.isdigit():
                    cn_char = _DIGIT_TO_CN.get(char, char)
                    if cn_char in _COMMON_PINYIN:
                        result.extend(self._split_pinyin(_COMMON_PINYIN[cn_char]))
            return result
        except ImportError:
            pass

        # 回退到内置映射表
        result: list[str] = []
        for char in text:
            if char.isspace():
                continue
            if char in _FULLWIDTH_PUNCT:
                result.append("_")
                continue
            if char in ",.!?;:":
                result.append("_")
                continue
            if char in _COMMON_PINYIN:
                result.extend(self._split_pinyin(_COMMON_PINYIN[char]))
            elif "\u4e00" <= char <= "\u9fff":
                # 未在映射表中的汉字，使用默认读音
                result.extend(self._split_pinyin("a1"))
            elif char.isdigit():
                cn_char = _DIGIT_TO_CN.get(char, char)
                if cn_char in _COMMON_PINYIN:
                    result.extend(self._split_pinyin(_COMMON_PINYIN[cn_char]))
            elif char.isascii() and char.isalpha():
                result.extend(self._g2p_english(char))
        return result

    def _g2p_english(self, text: str) -> list[str]:
        """英文文本转 ARPAbet 音素。

        使用内置极简 CMU 词典 + 字母规则回退。

        Parameters
        ----------
        text : str
            英文文本。

        Returns
        -------
        list[str]
            ARPAbet 音素列表。

        Notes
        -----
        内置词典覆盖的词有限，未覆盖的词按字母回退。
        生产环境建议安装完整的 CMU 词典。
        """
        result: list[str] = []
        # 按空格分词
        # B3-2: 改进英文分词正则，识别连字符、数字、常见符号
        words = re.findall(
            r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[,.!?;:$%@]",
            text,
        )
        for word in words:
            if word in ",.!?;:":
                result.append("_")
                continue
            upper = word.upper()
            if upper in _CMU_DICT:
                result.extend(_CMU_DICT[upper])
            else:
                # 字母回退
                for letter in upper:
                    if letter in _LETTER_TO_ARPABET:
                        result.extend(_LETTER_TO_ARPABET[letter])
        return result

    # ------------------------------------------------------------------
    # 说话人编码
    # ------------------------------------------------------------------
    def encode_speaker(
        self, speaker_id: str | np.ndarray | torch.Tensor | None
    ) -> torch.Tensor | None:
        """编码说话人标识。

        GPT-SoVITS 的说话人表示是参考音频的 SSL 语义 tokens + speaker embedding。
        此方法只做接口预留，实际编码由后端完成。

        Parameters
        ----------
        speaker_id : str | None
            说话人标识。可以是音频路径、预设名称或 ``None``。

        Returns
        -------
        Any | None
            ``None`` 时表示不使用语音克隆。路径/名称原样返回，由后端处理。
        """
        if speaker_id is None:
            return None
        # 路径或名称原样返回
        return speaker_id

    # ------------------------------------------------------------------
    # 韵律控制
    # ------------------------------------------------------------------
    def insert_prosody_tokens(
        self, text: str, prosody_prompt: str
    ) -> str:
        """在文本中插入韵律控制标记。

        GPT-SoVITS 的韵律控制主要通过标点和停顿实现。

        Parameters
        ----------
        text : str
            原始文本。
        prosody_prompt : str
            韵律提示（如语速、停顿位置）。

        Returns
        -------
        str
            插入韵律标记后的文本。
        """
        if prosody_prompt:
            return f"{prosody_prompt} {text}"
        return text

    # ------------------------------------------------------------------
    # 文本预处理
    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """文本清洗。

        Parameters
        ----------
        text : str
            原始文本。

        Returns
        -------
        str
            清洗后的文本。
        """
        # 基类清洗
        text = super().preprocess(text)

        # 全角标点转半角
        for full, half in _FULLWIDTH_PUNCT.items():
            text = text.replace(full, half)

        # 全角数字转半角
        text = text.translate(
            str.maketrans(
                "０１２３４５６７８９",
                "0123456789",
            )
        )

        # 全角字母转半角
        text = text.translate(
            str.maketrans(
                "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
                "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            )
        )

        # 过滤不支持的字符（保留 CJK/假名/英文/数字/标点/空白）
        result: list[str] = []
        for ch in text:
            cat = ch
            if (
                "\u4e00" <= ch <= "\u9fff"  # CJK 统一汉字
                or "\u3040" <= ch <= "\u30ff"  # 假名
                or "\uac00" <= ch <= "\ud7af"  # 韩文
                or ch.isascii()  # ASCII
            ):
                result.append(ch)
        return "".join(result)
