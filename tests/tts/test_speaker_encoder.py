"""测试 SpeakerEncoder（参考音频 → 说话人嵌入向量）。

依赖 torch；torch 不可用时整个模块自动跳过。torch 导入放在函数内部，
避免在模块顶层污染 sys.modules。模块级仅用 ``importlib.util.find_spec``
探测 torch 是否存在。

本测试使用小主干配置（``hidden_size=64``、``n_blocks=2``、``scale=4``）验证
ECAPA-TDNN 说话人编码器的 encode 行为、输出形状、确定性与权重加载，
不加载真实预训练权重——``load_weights`` 传入空路径时会以随机初始化完成加载。
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

from mosaic.nodes.audio.tts_backends.acoustic_models.speaker_encoder import (
    SpeakerEncoder,
)

# 输出嵌入维度（保持与默认一致，用于属性校验）
_EMBEDDING_DIM = 512


class TestSpeakerEncoder:
    """SpeakerEncoder 组件测试。

    使用小主干（``hidden_size=64``、``n_blocks=2``、``scale=4``）加速，
    保持 ``embedding_dim=512`` 以验证输出形状，验证 encode 的确定性、
    区分性与权重加载。
    """

    def _make_encoder(self) -> SpeakerEncoder:
        """构造 SpeakerEncoder（未加载权重），并缩减主干参数以加速。"""
        encoder = SpeakerEncoder(
            model_type="campp",
            embedding_dim=_EMBEDDING_DIM,
            sample_rate=16000,
        )
        # 缩减主干参数，兼顾测试速度与覆盖
        encoder.hidden_size = 64
        encoder.n_blocks = 2
        encoder.scale = 4
        return encoder

    def _load_encoder(self) -> SpeakerEncoder:
        """构造并加载 SpeakerEncoder（随机权重，CPU / float32）。"""
        encoder = self._make_encoder()
        encoder.load_weights("", device="cpu", dtype="float32")
        return encoder

    def _make_waveform(self, samples: int = 16000) -> Any:
        """构造随机波形 tensor ``[1, samples]``。"""
        import torch

        return torch.randn(1, samples)

    # ------------------------------------------------------------------
    # T_SPKENC_01~02：encode 基本行为 / 输出形状
    # ------------------------------------------------------------------
    def test_T_SPKENC_01(self) -> None:
        """T_SPKENC_01：encode 返回嵌入向量。

        加载模型，构造波形 ``[1, 16000]``，encode 后验证返回 tensor。
        """
        import torch

        encoder = self._load_encoder()
        waveform = self._make_waveform(16000)
        embedding = encoder.encode(waveform)
        assert embedding is not None
        assert torch.is_tensor(embedding)

    def test_T_SPKENC_02(self) -> None:
        """T_SPKENC_02：输出形状 ``[1, embedding_dim]``。"""
        import torch

        encoder = self._load_encoder()
        waveform = self._make_waveform(16000)
        embedding = encoder.encode(waveform)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == encoder.embedding_dim

    # ------------------------------------------------------------------
    # T_SPKENC_03~04：确定性 / 区分性
    # ------------------------------------------------------------------
    def test_T_SPKENC_03(self) -> None:
        """T_SPKENC_03：相同音频两次编码结果一致（eval 模式确定性）。

        在 eval + no_grad 下网络无随机性，相同输入应产生相同输出。
        """
        import torch

        encoder = self._load_encoder()
        waveform = self._make_waveform(16000)
        emb_a = encoder.encode(waveform)
        emb_b = encoder.encode(waveform)
        # 形状一致
        assert emb_a.shape == emb_b.shape
        # 数值一致（eval 模式确定性）
        assert torch.allclose(emb_a, emb_b, atol=1e-6)

    def test_T_SPKENC_04(self) -> None:
        """T_SPKENC_04：不同音频产生不同嵌入。

        编码两段不同的随机波形，验证输出不相等。
        """
        import torch

        encoder = self._load_encoder()
        torch.manual_seed(0)
        waveform_a = torch.randn(1, 16000)
        torch.manual_seed(1)
        waveform_b = torch.randn(1, 16000)
        # 确保两段音频确实不同
        assert not torch.equal(waveform_a, waveform_b)

        emb_a = encoder.encode(waveform_a)
        emb_b = encoder.encode(waveform_b)
        assert not torch.allclose(emb_a, emb_b, atol=1e-6)

    # ------------------------------------------------------------------
    # T_SPKENC_05：不同长度音频
    # ------------------------------------------------------------------
    def test_T_SPKENC_05(self) -> None:
        """T_SPKENC_05：不同长度音频可正常编码（``[1, 32000]``）。"""
        import torch

        encoder = self._load_encoder()
        waveform = self._make_waveform(32000)
        embedding = encoder.encode(waveform)
        assert torch.is_tensor(embedding)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == encoder.embedding_dim

    # ------------------------------------------------------------------
    # T_SPKENC_06：权重加载
    # ------------------------------------------------------------------
    def test_T_SPKENC_06(self) -> None:
        """T_SPKENC_06：load_weights 成功，``_is_loaded=True``。"""
        encoder = self._make_encoder()
        encoder.load_weights("", device="cpu", dtype="float32")
        assert encoder._is_loaded is True
