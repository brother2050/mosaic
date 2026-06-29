"""测试 SoVITSTokenizer（音素级中文/英文 G2P、特殊标记、说话人编码）。

依赖 torch；torch 不可用时整个模块自动跳过。torch 的导入放在函数内部，
避免在模块顶层污染 sys.modules（phase2 mock pollution）。模块级仅用
``importlib.util.find_spec`` 探测 torch 是否存在。

注意：``pypinyin`` 为可选依赖，未安装时 SoVITSTokenizer 回退到内置简化拼音
映射表，本测试的断言在两种路径下均成立。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.text_frontends.sovits_tokenizer import (
    SPECIAL_TOKENS,
    SoVITSTokenizer,
)


@pytest.fixture
def tokenizer() -> SoVITSTokenizer:
    """返回默认参数的 SoVITSTokenizer（空词表，使用内置音素词表）。"""
    return SoVITSTokenizer()


# ----------------------------------------------------------------------
# T_STKN_01~05：基本分词 / 拼音 / 声调 / ARPAbet / 中英混合
# ----------------------------------------------------------------------
def test_STKN_01(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_01：基本中文分词返回 tensor。"""
    import torch

    ids = tokenizer.tokenize("你好", language="zh")
    assert torch.is_tensor(ids)
    assert ids.ndim == 2
    assert ids.shape[0] == 1  # [1, seq_len]
    assert ids.shape[1] > 0


def test_STKN_02(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_02：拼音转换拆分为声母+韵母（带声调）。

    ``_g2p_chinese("你好")`` 应包含 ``n3`` / ``i3`` 这样的音素。
    """
    phonemes = tokenizer._g2p_chinese("你好")
    assert isinstance(phonemes, list)
    assert "n3" in phonemes
    assert "i3" in phonemes


def test_STKN_03(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_03：声调 1-5 均出现在 _g2p_chinese 输出中。

    选取 ``中(1) 国(2) 你(3) 是(4) 的(5)`` 五个字覆盖全部声调。
    """
    phonemes = tokenizer._g2p_chinese("中国你是的")
    tones_found: set[str] = set()
    for ph in phonemes:
        if ph and ph[-1].isdigit():
            tones_found.add(ph[-1])
    for tone in ("1", "2", "3", "4", "5"):
        assert tone in tones_found, f"声调 {tone} 未出现在 {phonemes}"


def test_STKN_04(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_04：英文 ARPAbet 音素转换。"""
    phonemes = tokenizer._g2p_english("hello")
    assert isinstance(phonemes, list)
    # CMU 词典子集：HELLO -> HH AH L OW
    assert "HH" in phonemes
    assert "AH" in phonemes
    assert "L" in phonemes
    assert "OW" in phonemes


def test_STKN_05(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_05：中英混合文本分词应正常工作。"""
    import torch

    ids = tokenizer.tokenize("你好hello", language="zh")
    assert torch.is_tensor(ids)
    assert ids.shape[0] == 1
    # 中文音素 + 英文字母音素，序列应明显长于结构标记
    assert ids.shape[1] > 4


# ----------------------------------------------------------------------
# T_STKN_06~07：多音字 / 数字转拼音
# ----------------------------------------------------------------------
def test_STKN_06(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_06：多音字处理（内置表给出默认读音）。

    ``乐`` 有 ``le4``（快乐）/ ``yue4``（音乐）两种读音，内置表取 ``le4``。
    """
    phonemes = tokenizer._g2p_chinese("乐")
    assert len(phonemes) > 0
    # le4 拆分为 l4 / e4
    assert "l4" in phonemes
    assert "e4" in phonemes


def test_STKN_07(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_07：数字转拼音应产出音素。

    ``123`` 中 ``1`` -> ``一`` -> ``yi1`` -> ``y1``/``i1``。
    """
    phonemes = tokenizer._g2p_chinese("123")
    assert len(phonemes) > 0
    # 至少存在一个带声调数字后缀的音素
    assert any(ph and ph[-1].isdigit() for ph in phonemes)


# ----------------------------------------------------------------------
# T_STKN_08：add_blank 参数
# ----------------------------------------------------------------------
def test_STKN_08() -> None:
    """T_STKN_08：add_blank=True 时在音素间插入空白 token。"""
    import torch

    tk_blank = SoVITSTokenizer(add_blank=True)
    tk_noblank = SoVITSTokenizer(add_blank=False)

    ids_blank = tk_blank.tokenize("你好", language="zh")
    ids_noblank = tk_noblank.tokenize("你好", language="zh")

    # 插入空白后序列更长
    assert ids_blank.shape[1] > ids_noblank.shape[1]
    # 空白 token id == SPECIAL_TOKENS["_"] == 2，应出现在 blank 版本中
    assert 2 in ids_blank[0].tolist()
    # 不插入空白时不应有空白 token
    assert 2 not in ids_noblank[0].tolist()


# ----------------------------------------------------------------------
# T_STKN_09~10：特殊标记 / 语言标记
# ----------------------------------------------------------------------
def test_STKN_09() -> None:
    """T_STKN_09：SPECIAL_TOKENS 包含 <s>、</s>、[SPLIT]、[SPK]。"""
    assert "<s>" in SPECIAL_TOKENS
    assert "</s>" in SPECIAL_TOKENS
    assert "[SPLIT]" in SPECIAL_TOKENS
    assert "[SPK]" in SPECIAL_TOKENS
    # id 为非负整数
    for key in ("<s>", "</s>", "[SPLIT]", "[SPK]"):
        assert isinstance(SPECIAL_TOKENS[key], int)
        assert SPECIAL_TOKENS[key] >= 0


def test_STKN_10() -> None:
    """T_STKN_10：语言标记 [ZH] 和 [EN] 存在于 SPECIAL_TOKENS。"""
    assert "[ZH]" in SPECIAL_TOKENS
    assert "[EN]" in SPECIAL_TOKENS
    assert SPECIAL_TOKENS["[ZH]"] != SPECIAL_TOKENS["[EN]"]


# ----------------------------------------------------------------------
# T_STKN_11~12：说话人编码
# ----------------------------------------------------------------------
def test_STKN_11(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_11：encode_speaker(None) 返回 None。"""
    assert tokenizer.encode_speaker(None) is None


def test_STKN_12(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_12：encode_speaker 接收音频路径时优雅处理（原样返回）。"""
    result = tokenizer.encode_speaker("/nonexistent/path/to/audio.wav")
    # 路径/名称原样返回，由后端负责实际加载
    assert result == "/nonexistent/path/to/audio.wav"


# ----------------------------------------------------------------------
# T_STKN_13：文本预处理
# ----------------------------------------------------------------------
def test_STKN_13(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_13：preprocess 全角转半角、空白归一化。"""
    cleaned = tokenizer.preprocess("你好，世界！ １２３")
    # 全角标点转半角
    assert "，" not in cleaned
    assert "！" not in cleaned
    # 全角数字转半角
    assert "123" in cleaned
    # 多余空白被归一化为单个空格
    assert "  " not in cleaned


# ----------------------------------------------------------------------
# T_STKN_14：detokenize 往返
# ----------------------------------------------------------------------
def test_STKN_14(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_14：detokenize 将 token ids 转回音素字符串。"""
    import torch

    ids = tokenizer.tokenize("你好", language="zh")
    text = tokenizer.detokenize(ids)
    assert isinstance(text, str)
    assert len(text) > 0
    # 应包含拼音音素 n3 / i3
    assert "n3" in text
    assert "i3" in text


# ----------------------------------------------------------------------
# T_STKN_15：空文本处理
# ----------------------------------------------------------------------
def test_STKN_15(tokenizer: SoVITSTokenizer) -> None:
    """T_STKN_15：空文本 tokenize 不崩溃，至少包含结构标记。"""
    import torch

    ids = tokenizer.tokenize("", language="zh")
    assert torch.is_tensor(ids)
    assert ids.shape[0] == 1
    # <s> + [ZH] + </s> 至少 3 个 token
    assert ids.shape[1] >= 3
    # 首尾为 <s> / </s>
    assert int(ids[0, 0]) == SPECIAL_TOKENS["<s>"]
    assert int(ids[0, -1]) == SPECIAL_TOKENS["</s>"]
