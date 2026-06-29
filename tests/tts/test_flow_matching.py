"""测试 FlowMatchingModel（CosyVoice 条件流匹配声学模型）。

依赖 torch；torch 不可用时整个模块自动跳过。torch 导入放在函数内部，
避免在模块顶层污染 sys.modules。模块级仅用 ``importlib.util.find_spec``
探测 torch 是否存在。

本测试使用小模型配置（``hidden_size=64``、``num_layers=2``、``num_heads=4``、
``condition_dim=64``）以兼顾速度与覆盖，不加载真实预训练权重——
``load_weights`` 传入空路径时会以随机初始化完成加载。
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

from mosaic.nodes.audio.tts_backends.acoustic_models.flow_matching import (
    FlowMatchingModel,
)

# 小模型参数（兼顾速度与覆盖）
_IN_CHANNELS = 80
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_CONDITION_DIM = 64


class TestFlowMatchingModel:
    """FlowMatchingModel 组件测试。

    使用小模型配置验证 ODE 求解、流式生成、ODE 步数 / 求解器切换、
    条件注入与数值稳定性。
    """

    def _make_model(self) -> FlowMatchingModel:
        """构造小参数 FlowMatchingModel（未加载权重）。"""
        return FlowMatchingModel(
            model_path="/tmp/test_flow",
            in_channels=_IN_CHANNELS,
            hidden_size=_HIDDEN_SIZE,
            num_layers=_NUM_LAYERS,
            num_heads=_NUM_HEADS,
            condition_dim=_CONDITION_DIM,
            num_ode_steps=10,
            ode_solver="euler",
            target_length_seconds=30.0,
        )

    def _load_model(self) -> FlowMatchingModel:
        """构造并加载小参数 FlowMatchingModel（随机权重，CPU / float32）。"""
        model = self._make_model()
        model.load_weights("", device="cpu", dtype="float32")
        return model

    def _make_condition(self, seq_len: int = 10) -> Any:
        """构造条件特征 tensor ``[1, seq_len, condition_dim]``。"""
        import torch

        return torch.randn(1, seq_len, _CONDITION_DIM)

    def _make_token_ids(self) -> Any:
        """构造 dummy token ids ``[1, 5]``。"""
        import torch

        return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # ------------------------------------------------------------------
    # T_FLOWM_01~02：权重加载 / 释放
    # ------------------------------------------------------------------
    def test_T_FLOWM_01(self) -> None:
        """T_FLOWM_01：load_weights 成功，``_is_loaded=True``。"""
        model = self._make_model()
        model.load_weights("", device="cpu", dtype="float32")
        assert model._is_loaded is True

    def test_T_FLOWM_02(self) -> None:
        """T_FLOWM_02：unload_weights 后 ``_is_loaded=False``、``_impl=None``。"""
        model = self._load_model()
        assert model._is_loaded is True
        model.unload_weights()
        assert model._is_loaded is False
        assert model._impl is None

    # ------------------------------------------------------------------
    # T_FLOWM_03~05：generate 基本行为 / 形状 / 数值稳定性
    # ------------------------------------------------------------------
    def test_T_FLOWM_03(self) -> None:
        """T_FLOWM_03：generate 返回 mel。

        加载模型，构造条件 tensor ``[1, 10, 64]``，通过 ``condition`` kwarg
        传入 generate，验证返回非空 mel。
        """
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50
        )
        assert mel is not None
        assert torch.is_tensor(mel)

    def test_T_FLOWM_04(self) -> None:
        """T_FLOWM_04：输出 mel 形状 ``[batch, 80, frames]``。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50
        )
        assert mel.shape[0] == 1  # batch
        assert mel.shape[1] == _IN_CHANNELS  # mel_bins == 80

    def test_T_FLOWM_05(self) -> None:
        """T_FLOWM_05：输出无 NaN / Inf。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50
        )
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()

    # ------------------------------------------------------------------
    # T_FLOWM_06~08：ODE 步数
    # ------------------------------------------------------------------
    def test_T_FLOWM_06(self) -> None:
        """T_FLOWM_06：``num_ode_steps=5``（最快）可正常生成。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50,
            num_ode_steps=5,
        )
        assert torch.is_tensor(mel)
        assert mel.shape[1] == _IN_CHANNELS

    def test_T_FLOWM_07(self) -> None:
        """T_FLOWM_07：``num_ode_steps=10``（推荐默认）可正常生成。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50,
            num_ode_steps=10,
        )
        assert torch.is_tensor(mel)
        assert mel.shape[1] == _IN_CHANNELS

    def test_T_FLOWM_08(self) -> None:
        """T_FLOWM_08：``num_ode_steps=20``（高质量）可正常生成。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        mel = model.generate(
            token_ids, condition=cond, target_length=50,
            num_ode_steps=20,
        )
        assert torch.is_tensor(mel)
        assert mel.shape[1] == _IN_CHANNELS

    # ------------------------------------------------------------------
    # T_FLOWM_09：ODE 求解器
    # ------------------------------------------------------------------
    def test_T_FLOWM_09(self) -> None:
        """T_FLOWM_09：不同 ode_solver（euler / midpoint）均可工作。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        for solver in ("euler", "midpoint"):
            mel = model.generate(
                token_ids, condition=cond, target_length=50,
                num_ode_steps=5, ode_solver=solver,
            )
            assert torch.is_tensor(mel)
            assert mel.shape[1] == _IN_CHANNELS

    # ------------------------------------------------------------------
    # T_FLOWM_10：条件注入
    # ------------------------------------------------------------------
    def test_T_FLOWM_10(self) -> None:
        """T_FLOWM_10：不同条件产生不同输出。

        使用相同随机种子控制 ODE 初始噪声，仅改变条件，验证输出不同。
        """
        import torch

        model = self._load_model()
        token_ids = self._make_token_ids()

        torch.manual_seed(42)
        cond_a = torch.randn(1, 10, _CONDITION_DIM)
        cond_b = torch.randn(1, 10, _CONDITION_DIM)
        # 确保两个条件确实不同
        assert not torch.equal(cond_a, cond_b)

        torch.manual_seed(123)
        mel_a = model.generate(
            token_ids, condition=cond_a, target_length=50, num_ode_steps=5,
        )
        torch.manual_seed(123)
        mel_b = model.generate(
            token_ids, condition=cond_b, target_length=50, num_ode_steps=5,
        )
        assert not torch.equal(mel_a, mel_b)

    # ------------------------------------------------------------------
    # T_FLOWM_11~12：流式生成
    # ------------------------------------------------------------------
    def test_T_FLOWM_11(self) -> None:
        """T_FLOWM_11：generate_stream 产出 mel chunk。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        result = model.generate_stream(
            token_ids, condition=cond, target_length=300,
            chunk_size_frames=150, overlap_frames=15, num_ode_steps=5,
        )
        chunk = next(result)
        assert chunk is not None
        assert torch.is_tensor(chunk)

    def test_T_FLOWM_12(self) -> None:
        """T_FLOWM_12：generate_stream 产出多个 chunk（``len > 1``）。"""
        import torch

        model = self._load_model()
        cond = self._make_condition(seq_len=10)
        token_ids = self._make_token_ids()
        result = model.generate_stream(
            token_ids, condition=cond, target_length=300,
            chunk_size_frames=150, overlap_frames=15, num_ode_steps=5,
        )
        chunks = list(result)
        assert len(chunks) > 1

    # ------------------------------------------------------------------
    # T_FLOWM_13：不同条件长度兼容
    # ------------------------------------------------------------------
    def test_T_FLOWM_13(self) -> None:
        """T_FLOWM_13：不同 text_features 长度兼容。

        尝试 condition ``[1, 5, 64]`` 与 ``[1, 20, 64]``，均可正常生成。
        """
        import torch

        model = self._load_model()
        token_ids = self._make_token_ids()
        for seq_len in (5, 20):
            cond = torch.randn(1, seq_len, _CONDITION_DIM)
            mel = model.generate(
                token_ids, condition=cond, target_length=50, num_ode_steps=5,
            )
            assert torch.is_tensor(mel)
            assert mel.shape[1] == _IN_CHANNELS

    # ------------------------------------------------------------------
    # T_FLOWM_14：模型类型标识
    # ------------------------------------------------------------------
    def test_T_FLOWM_14(self) -> None:
        """T_FLOWM_14：``model_type == "flow_matching"``，acoustic_type 若可用则一致。"""
        model = self._make_model()
        assert model.model_type == "flow_matching"
        # acoustic_type 并非所有实现都定义；若类层级可用则应与 model_type 一致
        acoustic_type = getattr(type(model), "acoustic_type", None)
        if acoustic_type is not None:
            assert acoustic_type == "flow_matching"

    # ------------------------------------------------------------------
    # T_FLOWM_15：EventBus 事件
    # ------------------------------------------------------------------
    def test_T_FLOWM_15(self, sample_token_ids: Any) -> None:
        """T_FLOWM_15：EventBus 事件——``event_bus=None`` 时 generate 不报错。

        若 ``EventBus`` 不可导入则跳过。同时使用 MagicMock 构造一个 mock
        事件总线，验证 generate 对其同样兼容。
        """
        try:
            from mosaic.core.events import EventBus  # noqa: F401
        except ImportError:
            pytest.skip("EventBus 不可用")

        import torch
        from unittest.mock import MagicMock

        model = self._load_model()
        cond = self._make_condition(seq_len=10)

        # event_bus=None 不应导致 generate 报错
        mel = model.generate(
            sample_token_ids, condition=cond, target_length=50,
            num_ode_steps=5, event_bus=None,
        )
        assert mel is not None
        assert torch.is_tensor(mel)

        # 使用 MagicMock 构造 mock 事件总线，同样不应报错
        mock_bus = MagicMock()
        mel2 = model.generate(
            sample_token_ids, condition=cond, target_length=50,
            num_ode_steps=5, event_bus=mock_bus,
        )
        assert mel2 is not None
