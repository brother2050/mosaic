"""测试 CosyVoiceTokenizer（LLM 词表布局、特殊标记、字符级回退分词）。

依赖 torch；torch 不可用时整个模块自动跳过。torch 导入放在函数内部，
避免在模块顶层污染 sys.modules（phase2 mock pollution）。模块级仅用
``importlib.util.find_spec`` 探测 torch 是否存在。

本测试在不加载真实 LLM tokenizer 权重的前提下，验证 CosyVoiceTokenizer
的字符级回退分词、词表空间布局（文本 / 语音 / 特殊标记）、说话人编码接口
与文本预处理行为。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.text_frontends.cosyvoice_tokenizer import (
    CosyVoiceTokenizer,
)


class TestCosyVoiceTokenizer:
    """CosyVoiceTokenizer 组件测试。

    使用默认参数构造（``speech_token_offset=151665``、``speech_token_size=6561``），
    不调用 ``load_weights``，因此走字符级回退分词路径：
    ``id = ord(ch) % llm_vocab_size``，跳过空白字符。
    """

    def _make_tokenizer(self) -> CosyVoiceTokenizer:
        """构造默认参数的 CosyVoiceTokenizer（不加载权重，使用字符级回退）。"""
        return CosyVoiceTokenizer()

    # ------------------------------------------------------------------
    # T_CVTOK_01~04：基本分词 / 中文 / 英文 / 中英混合
    # ------------------------------------------------------------------
    def test_T_CVTOK_01(self) -> None:
        """T_CVTOK_01：基本分词返回 tensor。

        不加载权重，调用 ``tokenize("hello")``，验证返回 ``torch.Tensor``。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("hello")
        assert torch.is_tensor(ids)

    def test_T_CVTOK_02(self) -> None:
        """T_CVTOK_02：中文文本分词，首尾为特殊标记。

        ``tokenize("你好世界")`` 输出首元素为 ``sos_token_id``、尾元素为
        ``flow_token_id``。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("你好世界")
        assert torch.is_tensor(ids)
        flat = ids[0].tolist()
        assert flat[0] == tokenizer.sos_token_id
        assert flat[-1] == tokenizer.flow_token_id

    def test_T_CVTOK_03(self) -> None:
        """T_CVTOK_03：英文文本分词，输出形状与内容正确。

        ``"hello world"`` 经字符级回退产生 10 个非空白字符 token，加上
        ``[sos]`` 与 ``[flow]`` 共 12 个。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("hello world")
        assert torch.is_tensor(ids)
        assert ids.shape[0] == 1  # [1, seq_len]
        # h e l l o w o r l d = 10 字符 + sos + flow = 12
        assert ids.shape[1] == 12

    def test_T_CVTOK_04(self) -> None:
        """T_CVTOK_04：中英混合文本分词。

        ``"你好hello世界"`` 共 9 个非空白字符 + sos + flow = 11。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("你好hello世界")
        assert torch.is_tensor(ids)
        assert ids.shape[0] == 1
        # 你 好 h e l l o 世 界 = 9 字符 + sos + flow = 11
        assert ids.shape[1] == 11

    # ------------------------------------------------------------------
    # T_CVTOK_05~06：特殊标记 / 词表范围
    # ------------------------------------------------------------------
    def test_T_CVTOK_05(self) -> None:
        """T_CVTOK_05：特殊标记 [sos] 和 [flow] 正确添加。

        检查首元素 == ``sos_token_id``，尾元素 == ``flow_token_id``。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("test")
        flat = ids[0].tolist()
        assert flat[0] == tokenizer.sos_token_id
        assert flat[-1] == tokenizer.flow_token_id

    def test_T_CVTOK_06(self) -> None:
        """T_CVTOK_06：所有 token id 落在 ``[0, vocab_size)`` 区间。"""
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("hello world 你好")
        assert torch.is_tensor(ids)
        flat = ids[0].tolist()
        for tid in flat:
            assert 0 <= tid < tokenizer.vocab_size

    # ------------------------------------------------------------------
    # T_CVTOK_07~08：说话人编码
    # ------------------------------------------------------------------
    def test_T_CVTOK_07(self) -> None:
        """T_CVTOK_07：``encode_speaker(None)`` 返回 ``None``。"""
        tokenizer = self._make_tokenizer()
        assert tokenizer.encode_speaker(None) is None

    def test_T_CVTOK_08(self) -> None:
        """T_CVTOK_08：``encode_speaker`` 接收音频路径返回占位 dict。

        返回的 dict 包含 ``ref_speech_tokens`` 和 ``speaker_embedding`` 两个键，
        值均为 ``None``（实际编码由后端 SpeechTokenizer / SpeakerEncoder 完成）。
        """
        tokenizer = self._make_tokenizer()
        result = tokenizer.encode_speaker("path/to/audio.wav")
        assert isinstance(result, dict)
        assert "ref_speech_tokens" in result
        assert "speaker_embedding" in result
        assert result["ref_speech_tokens"] is None
        assert result["speaker_embedding"] is None

    # ------------------------------------------------------------------
    # T_CVTOK_09~10：预处理 / detokenize
    # ------------------------------------------------------------------
    def test_T_CVTOK_09(self) -> None:
        """T_CVTOK_09：preprocess 清洗多余空白。

        连续空白被合并为单个空格，首尾空白被去除。
        """
        tokenizer = self._make_tokenizer()
        cleaned = tokenizer.preprocess("你好   世界  hello")
        assert "  " not in cleaned
        assert cleaned == "你好 世界 hello"

    def test_T_CVTOK_10(self) -> None:
        """T_CVTOK_10：detokenize 返回文本（字符级回退可还原）。

        由于未加载 LLM tokenizer，``detokenize`` 走字符级回退，对 ASCII
        字符可完整还原。此处验证返回非空字符串。
        """
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("hello")
        text = tokenizer.detokenize(ids)
        assert isinstance(text, str)
        assert len(text) > 0

    # ------------------------------------------------------------------
    # T_CVTOK_11~12：长文本 / 空文本
    # ------------------------------------------------------------------
    def test_T_CVTOK_11(self) -> None:
        """T_CVTOK_11：长文本（>500 字符）分词不崩溃且输出足够长。"""
        import torch

        tokenizer = self._make_tokenizer()
        long_text = "a" * 600
        ids = tokenizer.tokenize(long_text)
        assert torch.is_tensor(ids)
        # 600 字符 + sos + flow = 602
        assert ids.shape[1] > 500

    def test_T_CVTOK_12(self) -> None:
        """T_CVTOK_12：空文本仍产生 ``[sos, flow]``（2 个元素）。"""
        import torch

        tokenizer = self._make_tokenizer()
        ids = tokenizer.tokenize("")
        assert torch.is_tensor(ids)
        assert ids.shape[0] == 1
        assert ids.shape[1] == 2
        flat = ids[0].tolist()
        assert flat[0] == tokenizer.sos_token_id
        assert flat[1] == tokenizer.flow_token_id
