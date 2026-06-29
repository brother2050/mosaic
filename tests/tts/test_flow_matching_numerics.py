"""测试 Flow Matching 声学模型 ODE 求解的数值稳定性。

聚焦 Conditional Flow Matching 从高斯噪声积分到 mel 空间过程中的数值
性质：Euler / Midpoint 求解器产出是否有限、velocity 是否被正确截断、
不同 ODE 步数 / batch / target_length 下的稳定性，以及时间步嵌入在
边界 ``t=0`` / ``t=1`` 处的良定义性。不依赖真实预训练权重（``load_weights``
传入空目录走随机初始化分支，仍完整覆盖 ODE 求解代码路径）。

torch 采用惰性探测：模块级仅用 ``importlib.util.find_spec`` 检查是否存在，
不在模块顶层导入，避免 phase2 mock 污染 ``sys.modules``。``torch`` 的实际
导入放在每个需要它的测试方法内部。``FlowMatchingModel`` 的模块导入不触发
torch 加载（torch 在 ``_get_flow_matching_class()`` 与各方法内部惰性导入），
因此可在模块顶层安全导入。
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

sys.path.insert(0, "/workspace/mosaic")

# 检查 torch 是否可用（不导入，避免污染全局 sys.modules）
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch 未安装")

from mosaic.nodes.audio.tts_backends.acoustic_models.flow_matching import (  # noqa: E402
    FlowMatchingModel,
)

# ----------------------------------------------------------------------
# 小模型参数（不依赖真实权重，兼顾速度与覆盖）
# ----------------------------------------------------------------------
_IN_CHANNELS = 80
_HIDDEN_SIZE = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_COND_DIM = 64


def _make_loaded_model(tmp_path) -> FlowMatchingModel:
    """构造并加载一个小参数 FlowMatchingModel（CPU / float32，随机初始化）。

    ``load_weights`` 传入空临时目录时无权重文件，走随机初始化分支，
    仍会创建内部 ``_impl``（``_FlowMatchingModelImpl`` nn.Module）并完成
    device / dtype 迁移与 ``eval()`` 置位。
    """
    model = FlowMatchingModel(
        model_path=str(tmp_path),
        in_channels=_IN_CHANNELS,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        condition_dim=_COND_DIM,
        num_ode_steps=10,
        ode_solver="euler",
    )
    model.load_weights(str(tmp_path), device="cpu", dtype="float32")
    return model


class TestFlowMatchingNumerics:
    """Flow Matching ODE 求解数值稳定性测试（T_FMNUM_01~09）。"""

    # ------------------------------------------------------------------
    # T_FMNUM_01：Euler 单步轨迹平滑
    # ------------------------------------------------------------------
    def test_T_FMNUM_01(self, tmp_path) -> None:
        """Euler 10 步求解：输出 mel 有限且形状正确。

        velocity 被 clamp 到 [-10, 10]，单步 ``dt=0.1`` 最多改变 1.0，
        10 步内轨迹不会发散。此处验证最终 mel 有限、形状为
        ``[1, mel_bins, target_len]`` 且幅值在合理范围内。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.randn(1, 10, _COND_DIM)

        mel = model._impl.solve_ode(
            condition=condition,
            target_len=50,
            num_steps=10,
            solver="euler",
        )

        assert mel.shape == (1, _IN_CHANNELS, 50)
        assert torch.isfinite(mel).all()
        # velocity 截断 + 有限步数 -> 幅值不会爆炸
        assert mel.abs().max().item() < 1000.0

    # ------------------------------------------------------------------
    # T_FMNUM_02：Midpoint 与 Euler 均产出有效输出
    # ------------------------------------------------------------------
    def test_T_FMNUM_02(self, tmp_path) -> None:
        """Euler 5 步与 Midpoint 5 步均产出有限 mel。

        真正的精度对比需要训练好的权重（rectified flow 使轨迹接近直线），
        此处仅验证两种求解器在随机初始化下都能正常完成 ODE 积分并产出
        有限、同形状的结果。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.randn(1, 10, _COND_DIM)
        token_ids = torch.tensor([[1, 2, 3]])

        mel_euler = model.generate(
            token_ids,
            condition=condition,
            target_length=50,
            num_ode_steps=5,
            ode_solver="euler",
        )
        mel_mid = model.generate(
            token_ids,
            condition=condition,
            target_length=50,
            num_ode_steps=5,
            ode_solver="midpoint",
        )

        assert torch.isfinite(mel_euler).all()
        assert torch.isfinite(mel_mid).all()
        assert mel_euler.shape == (1, _IN_CHANNELS, 50)
        assert mel_mid.shape == (1, _IN_CHANNELS, 50)

    # ------------------------------------------------------------------
    # T_FMNUM_03：velocity 输出无 NaN/Inf 且被截断
    # ------------------------------------------------------------------
    def test_T_FMNUM_03(self, tmp_path) -> None:
        """直接调用 FlowEstimator forward：velocity 有限且落在 [-10, 10]。

        验证速度场预测 ``v(z_t, t, condition)`` 的数值稳定性：输出无
        NaN/Inf，且经 ``torch.clamp(velocity, -10, 10)`` 后幅值不超过 10。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        estimator = model._impl.estimator

        z_t = torch.randn(1, _IN_CHANNELS, 50)
        t = torch.tensor(0.5)
        condition = torch.randn(1, 10, _COND_DIM)

        velocity = estimator(z_t, t, condition)

        assert velocity.shape == (1, _IN_CHANNELS, 50)
        assert torch.isfinite(velocity).all()
        # velocity 被 clamp 到 [-10, 10]
        assert velocity.abs().max().item() <= 10.0

    # ------------------------------------------------------------------
    # T_FMNUM_04：ODE 过程中 z_t 无 NaN/Inf（检查最终 mel）
    # ------------------------------------------------------------------
    def test_T_FMNUM_04(self, tmp_path) -> None:
        """ODE 求解后最终 mel 无 NaN/Inf。

        无法在不修改源码的情况下直接观测中间 ``z_t``，但 velocity 截断
        保证每步更新有限，因此最终 mel 有限即等价于整个轨迹未发散。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.randn(1, 10, _COND_DIM)
        token_ids = torch.tensor([[1, 2, 3]])

        mel = model.generate(
            token_ids,
            condition=condition,
            target_length=50,
            num_ode_steps=10,
            ode_solver="euler",
        )

        assert mel.shape == (1, _IN_CHANNELS, 50)
        assert torch.isfinite(mel).all()

    # ------------------------------------------------------------------
    # T_FMNUM_05：更多步数 -> 更稳定（均有限）
    # ------------------------------------------------------------------
    def test_T_FMNUM_05(self, tmp_path) -> None:
        """5 / 10 / 20 步 ODE 求解均产出有限 mel。

        不对未训练权重断言单调性，仅验证不同步数下输出都有限（步数变化
        不引发数值异常），并记录各步数输出的标准差供观察。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.randn(1, 10, _COND_DIM)
        token_ids = torch.tensor([[1, 2, 3]])

        for num_steps in (5, 10, 20):
            mel = model.generate(
                token_ids,
                condition=condition,
                target_length=50,
                num_ode_steps=num_steps,
                ode_solver="euler",
            )
            assert torch.isfinite(mel).all(), (
                f"num_steps={num_steps} 产生 NaN/Inf"
            )
            assert mel.shape == (1, _IN_CHANNELS, 50)
            # 标准差有限（非负，供观察）
            std = mel.std().item()
            assert std >= 0.0

    # ------------------------------------------------------------------
    # T_FMNUM_06：时间步嵌入在 t=0 与 t=1 处良定义
    # ------------------------------------------------------------------
    def test_T_FMNUM_06(self, tmp_path) -> None:
        """SinusoidalPosEmb 在 t=0.0 与 t=1.0 处输出有限。

        时间步嵌入是 ODE 求解的条件输入，必须在区间端点良定义。
        ``t=0`` 时正弦项为 0、余弦项为 1；``t=1`` 时为一般值，均应有限。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        time_emb = model._impl.estimator.time_emb

        emb_0 = time_emb(torch.tensor(0.0))
        emb_1 = time_emb(torch.tensor(1.0))

        assert torch.isfinite(emb_0).all()
        assert torch.isfinite(emb_1).all()
        # 标量 t -> [dim] 嵌入
        assert emb_0.shape[-1] == _HIDDEN_SIZE
        assert emb_1.shape[-1] == _HIDDEN_SIZE

    # ------------------------------------------------------------------
    # T_FMNUM_07：零条件仍可生成
    # ------------------------------------------------------------------
    def test_T_FMNUM_07(self, tmp_path) -> None:
        """全零条件 [1,10,64] 下 ODE 仍产出有限 mel。

        零条件使交叉注意力 key/value 为零投影，但 ``z_1`` 仍为高斯噪声，
        velocity 截断保证积分稳定。验证退化条件下不崩溃。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.zeros(1, 10, _COND_DIM)
        token_ids = torch.tensor([[1, 2, 3]])

        mel = model.generate(
            token_ids,
            condition=condition,
            target_length=50,
            num_ode_steps=10,
            ode_solver="euler",
        )

        assert mel.shape == (1, _IN_CHANNELS, 50)
        assert torch.isfinite(mel).all()

    # ------------------------------------------------------------------
    # T_FMNUM_08：batch_size=1 与 batch_size>1 兼容
    # ------------------------------------------------------------------
    def test_T_FMNUM_08(self, tmp_path) -> None:
        """batch=1 与 batch=2 条件下 generate 均正常。

        ``solve_ode`` 以 ``condition.shape[0]`` 推断 batch 维并初始化噪声，
        验证两种 batch 下输出 batch 维正确且有限。
        """
        import torch

        model = _make_loaded_model(tmp_path)

        cond_b1 = torch.randn(1, 10, _COND_DIM)
        cond_b2 = torch.randn(2, 10, _COND_DIM)

        mel_b1 = model.generate(
            torch.tensor([[1, 2, 3]]),
            condition=cond_b1,
            target_length=50,
        )
        mel_b2 = model.generate(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            condition=cond_b2,
            target_length=50,
        )

        assert mel_b1.shape == (1, _IN_CHANNELS, 50)
        assert mel_b2.shape == (2, _IN_CHANNELS, 50)
        assert torch.isfinite(mel_b1).all()
        assert torch.isfinite(mel_b2).all()

    # ------------------------------------------------------------------
    # T_FMNUM_09：不同 target_length 兼容
    # ------------------------------------------------------------------
    def test_T_FMNUM_09(self, tmp_path) -> None:
        """target_length=50/100/200 下输出长度匹配且有限。

        ``solve_ode`` 用 ``target_len`` 初始化噪声序列长度，验证不同
        目标长度下输出最后一维与 target_length 一致。
        """
        import torch

        model = _make_loaded_model(tmp_path)
        condition = torch.randn(1, 10, _COND_DIM)
        token_ids = torch.tensor([[1, 2, 3]])

        for target_len in (50, 100, 200):
            mel = model.generate(
                token_ids,
                condition=condition,
                target_length=target_len,
            )
            assert mel.shape == (1, _IN_CHANNELS, target_len)
            assert torch.isfinite(mel).all()
