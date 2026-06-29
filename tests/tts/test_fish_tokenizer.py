"""测试 FishTokenizer。

依赖 torch；torch 不可用时整个模块自动跳过。torch 导入放在函数内部，
避免在模块顶层污染 sys.modules，从而不影响 phase2 的 mock 测试。
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.text_frontends.fish_tokenizer import (
    FishTokenizer,
)

# FishTokenizer 特殊标记 id（与 SPECIAL_TOKENS 一致）
_S_ID = 0          # <s> 序列开始
_AUDIO_ID = 3      # <audio> 音频段开始
_CLONE_ID = 4      # <clone> 语音克隆标记
_ZH_ID = 6         # <zh> 中文语言标记
_EN_ID = 7         # <en> 英文语言标记
_JA_ID = 8         # <ja> 日文语言标记

# 构造参数：文本词表 1000 + 音频词表 1024 = 总词表 2024
_TEXT_VOCAB_SIZE = 1000
_AUDIO_VOCAB_SIZE = 1024
_VOCAB_SIZE = _TEXT_VOCAB_SIZE + _AUDIO_VOCAB_SIZE  # 2024


@pytest.fixture
def tokenizer() -> FishTokenizer:
    """返回字符级 FishTokenizer（空词表，便于接口测试）。"""
    return FishTokenizer(
        vocab_path="",
        text_vocab_size=_TEXT_VOCAB_SIZE,
        audio_vocab_size=_AUDIO_VOCAB_SIZE,
    )


# ----------------------------------------------------------------------
# FTKN_01~05：基本 / 中文 / 英文 / 日文 / 混合分词
# ----------------------------------------------------------------------
def test_FTKN_01(tokenizer: FishTokenizer) -> None:
    """基本分词，输出 tensor。"""
    import torch

    ids = tokenizer.tokenize("你好", language="zh")
    assert torch.is_tensor(ids)
    assert ids.ndim == 2
    assert ids.shape[0] == 1  # [1, seq_len]


def test_FTKN_02(tokenizer: FishTokenizer) -> None:
    """中文文本分词正确，包含 <s>(0) 与 <audio>(3)。"""
    ids = tokenizer.tokenize("你好世界", language="zh")
    flat = ids[0].tolist()
    assert _S_ID in flat
    assert _AUDIO_ID in flat


def test_FTKN_03(tokenizer: FishTokenizer) -> None:
    """英文文本分词，包含 <en>(7) 标记。"""
    ids = tokenizer.tokenize("hello world", language="en")
    flat = ids[0].tolist()
    assert _EN_ID in flat


def test_FTKN_04(tokenizer: FishTokenizer) -> None:
    """日文文本分词，包含 <ja>(8) 标记。"""
    ids = tokenizer.tokenize("こんにちは", language="ja")
    flat = ids[0].tolist()
    assert _JA_ID in flat


def test_FTKN_05(tokenizer: FishTokenizer) -> None:
    """中英混合文本分词，输出不为空。"""
    ids = tokenizer.tokenize("你好hello", language="zh")
    assert ids.shape[1] > 0


# ----------------------------------------------------------------------
# FTKN_06~08：token 范围 / 特殊标记 / 语言标记
# ----------------------------------------------------------------------
def test_FTKN_06(tokenizer: FishTokenizer) -> None:
    """token ids 在正确范围内：所有 id >= 0 且 < vocab_size(2024)。

    注：空词表时退化为字符级分词，字符 id = 特殊标记数(10) + 字符码点。
    CJK 字符码点远大于 vocab_size，会越界，因此此处使用 ASCII 文本验证
    token id 范围约束。
    """
    ids = tokenizer.tokenize("hello", language="en")
    flat = ids[0].tolist()
    assert all(0 <= int(tid) < _VOCAB_SIZE for tid in flat)


def test_FTKN_07(tokenizer: FishTokenizer) -> None:
    """特殊标记正确添加：序列以 <s>(0) 开头，以 <audio>(3) 结尾。"""
    ids = tokenizer.tokenize("你好", language="zh")
    assert int(ids[0, 0]) == _S_ID
    assert int(ids[0, -1]) == _AUDIO_ID


def test_FTKN_08(tokenizer: FishTokenizer) -> None:
    """语言标记插入：不同 language 对应不同语言标记 id。"""
    ids_zh = tokenizer.tokenize("hi", language="zh")
    ids_en = tokenizer.tokenize("hi", language="en")
    ids_ja = tokenizer.tokenize("hi", language="ja")
    lang_zh = int(ids_zh[0, 1])
    lang_en = int(ids_en[0, 1])
    lang_ja = int(ids_ja[0, 1])
    assert lang_zh == _ZH_ID
    assert lang_en == _EN_ID
    assert lang_ja == _JA_ID
    # 三者互不相同
    assert len({lang_zh, lang_en, lang_ja}) == 3


# ----------------------------------------------------------------------
# FTKN_09：文本清洗
# ----------------------------------------------------------------------
def test_FTKN_09(tokenizer: FishTokenizer) -> None:
    """preprocess 文本清洗：去除控制字符。"""
    cleaned = tokenizer.preprocess("\x00你好\x01世界\x02")
    # 控制字符已被清除
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "\x02" not in cleaned
    # 正文字符保留
    assert "你好" in cleaned


# ----------------------------------------------------------------------
# FTKN_10~11：说话人编码
# ----------------------------------------------------------------------
def test_FTKN_10(tokenizer: FishTokenizer) -> None:
    """encode_speaker(None) 返回 None。"""
    assert tokenizer.encode_speaker(None) is None


def test_FTKN_11(tokenizer: FishTokenizer) -> None:
    """encode_speaker(音频路径字符串) 返回路径字符串本身。"""
    path = "/data/ref_audio.wav"
    assert tokenizer.encode_speaker(path) == path


# ----------------------------------------------------------------------
# FTKN_12：detokenize 与 tokenize 对称
# ----------------------------------------------------------------------
def test_FTKN_12(tokenizer: FishTokenizer) -> None:
    """detokenize 与 tokenize 的对称性：文本部分可恢复。

    注：detokenize 会跳过 id >= text_vocab_size 的音频 token。字符级分词
    下 CJK 字符 id = 10 + 码点，远大于 text_vocab_size(1000) 会被跳过，
    因此使用 ASCII 文本验证往返对称性。
    """
    ids = tokenizer.tokenize("hello", language="en")
    text = tokenizer.detokenize(ids)
    # 文本部分可恢复
    assert "hello" in text


# ----------------------------------------------------------------------
# FTKN_13：语音克隆模式
# ----------------------------------------------------------------------
def test_FTKN_13(tokenizer: FishTokenizer) -> None:
    """voice clone 模式：传入 ref_tokens，序列包含 <clone>(4) 标记。"""
    ids = tokenizer.tokenize(
        "hello", language="en", ref_tokens=[100, 200, 300]
    )
    flat = ids[0].tolist()
    assert _CLONE_ID in flat
    # 克隆模式序列结构：<s> <clone> ref_tokens <lang> text_tokens <audio>
    assert int(flat[0]) == _S_ID
    assert int(flat[1]) == _CLONE_ID
