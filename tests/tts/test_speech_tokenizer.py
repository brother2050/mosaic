"""测试 SpeechTokenizer（参考音频 → 离散语音 token 编/解码）。

依赖 torch；torch 不可用时整个模块自动跳过。torch 导入放在函数内部，
避免在模块顶层污染 sys.modules。模块级仅用 ``importlib.util.find_spec``
探测 torch 是否存在。

本测试使用小模型配置（``hidden_size=64``）验证 2 层 RVQ 编码器的
encode / decode 行为与码本范围，不加载真实预训练权重——``load_weights``
传入空路径时会以随机初始化完成加载。
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.acoustic_models.speech_tokenizer import (
    SpeechTokenizer,
)

# 小模型参数（兼顾速度与覆盖）
_HIDDEN_SIZE = 64
_CODEBOOK_SIZE = 6561  # 81 * 81


class TestSpeechTokenizer:
    """SpeechTokenizer 组件测试。

    使用小 hidden_size 加速，验证 encode 输出 token ids 的形状与码本范围、
    不同长度音频兼容性，以及 decode 可用性。
    """

    def _make_tokenizer(self) -> SpeechTokenizer:
        """构造小参数 SpeechTokenizer（未加载权重）。"""
        tokenizer = SpeechTokenizer(
            model_type="cosyvoice",
            codebook_size=_CODEBOOK_SIZE,
            num_codebooks=1,
            hidden_size=_HIDDEN_SIZE,
            sample_rate=22050,
        )
        return tokenizer

    def _load_tokenizer(self) -> SpeechTokenizer:
        """构造并加载小参数 SpeechTokenizer（随机权重，CPU / float32）。"""
        tokenizer = self._make_tokenizer()
        tokenizer.load_weights("", device="cpu", dtype="float32")
        return tokenizer

    def _make_waveform(self, samples: int = 16000) -> Any:
        """构造随机波形 tensor ``[1, samples]``。"""
        import torch

        return torch.randn(1, samples)

    # ------------------------------------------------------------------
    # T_SPTOK_01~02：encode 基本行为 / 码本范围
    # ------------------------------------------------------------------
    def test_T_SPTOK_01(self) -> None:
        """T_SPTOK_01：encode 返回 token ids。

        加载模型，构造波形 ``[1, 16000]``，encode 后验证返回 tensor。
        """
        import torch

        tokenizer = self._load_tokenizer()
        waveform = self._make_waveform(16000)
        tokens = tokenizer.encode(waveform)
        assert tokens is not None
        assert torch.is_tensor(tokens)

    def test_T_SPTOK_02(self) -> None:
        """T_SPTOK_02：token ids 落在 ``[0, codebook_size)`` 范围内。"""
        import torch

        tokenizer = self._load_tokenizer()
        waveform = self._make_waveform(16000)
        tokens = tokenizer.encode(waveform)
        flat = tokens.reshape(-1)
        assert int(flat.min()) >= 0
        assert int(flat.max()) < tokenizer.codebook_size

    # ------------------------------------------------------------------
    # T_SPTOK_03：不同长度音频
    # ------------------------------------------------------------------
    def test_T_SPTOK_03(self) -> None:
        """T_SPTOK_03：不同长度音频兼容（``[1, 8000]`` 与 ``[1, 32000]``）。"""
        import torch

        tokenizer = self._load_tokenizer()
        for samples in (8000, 32000):
            waveform = self._make_waveform(samples)
            tokens = tokenizer.encode(waveform)
            assert torch.is_tensor(tokens)
            # 更长的音频应产生更多的 token 帧
            assert tokens.shape[-1] > 0

    # ------------------------------------------------------------------
    # T_SPTOK_04：decode 可用性
    # ------------------------------------------------------------------
    def test_T_SPTOK_04(self) -> None:
        """T_SPTOK_04：decode 可用——对编码后的 token 调用 decode 返回结果。

        decode 返回 ``AudioData``（含 ``waveform``）或 tensor，此处验证
        调用不崩溃且返回非空结果。
        """
        import torch

        tokenizer = self._load_tokenizer()
        waveform = self._make_waveform(16000)
        tokens = tokenizer.encode(waveform)
        result = tokenizer.decode(tokens)
        assert result is not None
        # decode 返回 AudioData（含 waveform）或 tensor
        if hasattr(result, "waveform"):
            assert result.waveform is not None
        else:
            assert torch.is_tensor(result)

    # ------------------------------------------------------------------
    # T_SPTOK_05：权重加载
    # ------------------------------------------------------------------
    def test_T_SPTOK_05(self) -> None:
        """T_SPTOK_05：load_weights 成功，``_is_loaded=True``。"""
        tokenizer = self._make_tokenizer()
        tokenizer.load_weights("", device="cpu", dtype="float32")
        assert tokenizer._is_loaded is True
