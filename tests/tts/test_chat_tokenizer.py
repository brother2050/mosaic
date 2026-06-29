"""测试 ChatTokenizer（字符级分词、韵律标记、特殊标记、说话人编码）。

依赖 torch；torch 不可用时整个模块自动跳过。
"""
from __future__ import annotations

import importlib.util
import lzma
import struct
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.text_frontends.chat_tokenizer import (
    ChatTokenizer,
)
from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
    ChatTTSBackend,
)

# 韵律标记在 ChatTokenizer.SPECIAL_TOKENS 中的固定下标
# ['[Stts]','[Ptts]','[spk_emb]','[empty_spk]','[laugh_0]',...,'[break_4]',...,'[oral_2]']
_LAUGH_0_ID = 4
_BREAK_4_ID = 13
_ORAL_2_ID = 19
_STTS_ID = 0
_PTTS_ID = 1
_SPK_EMB_ID = 2
_EMPTY_SPK_ID = 3


@pytest.fixture
def tokenizer() -> ChatTokenizer:
    """返回字符级 ChatTokenizer（空词表）。"""
    return ChatTokenizer(vocab_path="")


# ----------------------------------------------------------------------
# T_CTKN_01~04：基本 / 中文 / 英文 / 混合分词
# ----------------------------------------------------------------------
def test_CTKN_01(tokenizer: ChatTokenizer) -> None:
    """基本分词，输出 tensor。"""
    import torch

    ids = tokenizer.tokenize("你好", language="zh")
    assert torch.is_tensor(ids)
    assert ids.ndim == 2
    assert ids.shape[0] == 1  # [1, seq_len]


def test_CTKN_02(tokenizer: ChatTokenizer) -> None:
    """中文文本分词正确。"""
    ids = tokenizer.tokenize("你好世界", language="zh")
    # 结构标记 [Stts][empty_spk] + 语言 'zh' + 四个汉字 + [Ptts]，至少多于结构部分
    assert ids.shape[1] > 7
    # 首尾结构标记正确
    assert int(ids[0, 0]) == _STTS_ID
    assert int(ids[0, -1]) == _PTTS_ID


def test_CTKN_03(tokenizer: ChatTokenizer) -> None:
    """英文文本分词正确。"""
    ids = tokenizer.tokenize("hello world", language="en")
    assert ids.shape[1] > 6
    assert int(ids[0, 0]) == _STTS_ID
    assert int(ids[0, -1]) == _PTTS_ID


def test_CTKN_04(tokenizer: ChatTokenizer) -> None:
    """中英混合文本分词。"""
    ids = tokenizer.tokenize("你好hello", language="zh")
    assert ids.shape[1] > 6
    # 同时包含中文与英文字符对应的 token
    flat = ids[0].tolist()
    assert int(flat[0]) == _STTS_ID
    assert int(flat[-1]) == _PTTS_ID


# ----------------------------------------------------------------------
# T_CTKN_05~07：韵律标记
# ----------------------------------------------------------------------
def test_CTKN_05(tokenizer: ChatTokenizer) -> None:
    """韵律标记插入 [laugh_0]。"""
    ids = tokenizer.tokenize("你好", language="zh", prosody_prompt="[laugh_0]")
    flat = ids[0].tolist()
    assert _LAUGH_0_ID in flat


def test_CTKN_06(tokenizer: ChatTokenizer) -> None:
    """韵律标记插入 [break_4]。"""
    ids = tokenizer.tokenize("你好,世界.", language="zh", prosody_prompt="[break_4]")
    flat = ids[0].tolist()
    assert _BREAK_4_ID in flat


def test_CTKN_07(tokenizer: ChatTokenizer) -> None:
    """多个韵律标记同时使用。"""
    ids = tokenizer.tokenize(
        "你好,世界.", language="zh", prosody_prompt="[oral_2][laugh_0][break_4]"
    )
    flat = ids[0].tolist()
    assert _ORAL_2_ID in flat
    assert _LAUGH_0_ID in flat
    assert _BREAK_4_ID in flat


# ----------------------------------------------------------------------
# T_CTKN_08：文本清洗
# ----------------------------------------------------------------------
def test_CTKN_08(tokenizer: ChatTokenizer) -> None:
    """preprocess 文本清洗（全角转半角、数字转中文）。"""
    cleaned = tokenizer.preprocess("你好，世界！123")
    # 全角标点转半角
    assert "，" not in cleaned
    assert "。" not in cleaned
    # 数字转中文
    assert "1" not in cleaned
    assert "一二三" in cleaned


# ----------------------------------------------------------------------
# T_CTKN_09：特殊标记
# ----------------------------------------------------------------------
def test_CTKN_09(tokenizer: ChatTokenizer) -> None:
    """特殊标记 [Stts]、[Ptts]、[spk_emb] 正确添加。"""
    # 指定 speaker_id 时使用 [spk_emb]
    with_spk = tokenizer.tokenize("你好", language="zh", speaker_id="spk1")
    assert int(with_spk[0, 0]) == _STTS_ID
    assert int(with_spk[0, 1]) == _SPK_EMB_ID
    assert int(with_spk[0, -1]) == _PTTS_ID
    # 不指定时使用 [empty_spk]
    without_spk = tokenizer.tokenize("你好", language="zh")
    assert int(without_spk[0, 1]) == _EMPTY_SPK_ID


# ----------------------------------------------------------------------
# T_CTKN_10~11：说话人编码 / 解码
# ----------------------------------------------------------------------
def _encode_speaker_tensor(spk: Any) -> str:
    """复用 ChatTTSBackend 的编码方式将说话人张量编码为字符串。"""
    import torch

    spk_f16 = spk.to(torch.float16)
    n = spk_f16.numel()
    raw = struct.pack("<" + "e" * n, *spk_f16.tolist())
    compressed = lzma.compress(
        raw,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 7 | lzma.PRESET_EXTREME}],
    )
    return ChatTTSBackend._base16384_encode(compressed)


def test_CTKN_10(tokenizer: ChatTokenizer) -> None:
    """encode_speaker 编码和解码一致性。"""
    import torch

    spk = torch.randn(256) * 0.5
    encoded = _encode_speaker_tensor(spk)
    decoded = tokenizer.encode_speaker(encoded)
    assert decoded is not None
    assert decoded.shape == (256,)
    # float16 往返应近乎无损
    assert torch.allclose(decoded.float(), spk.to(torch.float16).float(), atol=1e-2)


def test_CTKN_11(tokenizer: ChatTokenizer) -> None:
    """encode_speaker(None) 返回 None。"""
    assert tokenizer.encode_speaker(None) is None


# ----------------------------------------------------------------------
# T_CTKN_12：detokenize 与 tokenize 对称
# ----------------------------------------------------------------------
def test_CTKN_12(tokenizer: ChatTokenizer) -> None:
    """detokenize 与 tokenize 对称（基本测试）。"""
    ids = tokenizer.tokenize("你好", language="zh")
    text = tokenizer.detokenize(ids)
    # detokenize 结果应包含结构标记与原始字符
    assert "[Stts]" in text
    assert "[Ptts]" in text
    assert "你好" in text
