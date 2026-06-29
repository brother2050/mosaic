# tests/tts/test_four_backends_regression.py
"""四后端共存回归测试。

验证 ChatTTS / Fish / GPT-SoVITS / CosyVoice 四个后端在注册表中共存、
独立加载卸载、自动选择与回归不缺失。

CosyVoice 作为第四个后端加入后，需确保原有三个后端不受影响，
且 Flow Matching（cosyvoice）与自回归（chattts/fish/sovits）后端
能在同一注册表中按声学类型正确区分。
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/workspace/mosaic")

_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _reset_registry() -> Any:
    """重置注册表单例并重新注册全部内置后端，返回 registry。"""
    from mosaic.nodes.audio.tts_backends.registry import TTSBackendRegistry
    import mosaic.nodes.audio.tts_backends.registry as reg_module

    registry = TTSBackendRegistry()
    registry._backends.clear()
    registry._specs.clear()
    reg_module._backends_registered = False
    registry._ensure_builtin_registered()
    return registry


# ----------------------------------------------------------------------
# T_4BE_01 ~ T_4BE_03: 注册表完整性
# ----------------------------------------------------------------------
class TestFourBackendsRegistry:
    """四后端注册表完整性测试。"""

    def test_T_4BE_01(self) -> None:
        """T_4BE_01：四个后端均已注册。"""
        registry = _reset_registry()
        names = list(registry._backends.keys())
        assert "chattts" in names
        assert "fish" in names
        assert "sovits" in names
        assert "cosyvoice" in names

    def test_T_4BE_02(self) -> None:
        """T_4BE_02：list_backends() 返回数量 >= 4。"""
        registry = _reset_registry()
        specs = registry.list_backends()
        assert len(specs) >= 4

    def test_T_4BE_03(self) -> None:
        """T_4BE_03：各后端 acoustic_type 正确。"""
        registry = _reset_registry()
        assert registry._specs["chattts"].acoustic_type == "ar"
        assert registry._specs["fish"].acoustic_type == "ar"
        assert registry._specs["sovits"].acoustic_type == "ar"
        assert registry._specs["cosyvoice"].acoustic_type == "flow_matching"


# ----------------------------------------------------------------------
# T_4BE_04 ~ T_4BE_07: 独立加载卸载与共存
# ----------------------------------------------------------------------
class TestFourBackendsCoexistence:
    """四后端独立加载卸载与共存测试。"""

    def test_T_4BE_04(self) -> None:
        """T_4BE_04：CosyVoice 与 GPT-SoVITS 可同时创建（不加载）。"""
        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        cosyvoice = CosyVoiceBackend(model_path="/tmp/fake_cosyvoice")
        sovits = GPTSoVITSBackend(model_path="/tmp/fake_sovits")
        assert cosyvoice is not None
        assert sovits is not None
        assert cosyvoice.name == "cosyvoice"
        assert sovits.name == "sovits"
        assert cosyvoice.name != sovits.name

    def test_T_4BE_05(self) -> None:
        """T_4BE_05：调度器管理 GPU 显存，创建时被存储。"""
        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )

        mock_scheduler = MagicMock()
        backend = CosyVoiceBackend(
            model_path="/tmp/fake_cosyvoice", scheduler=mock_scheduler
        )
        assert backend._scheduler is mock_scheduler

    def test_T_4BE_06(self) -> None:
        """T_4BE_06：注册表 _backends 字典包含全部四个后端。"""
        registry = _reset_registry()
        assert isinstance(registry._backends, dict)
        for name in ("chattts", "fish", "sovits", "cosyvoice"):
            assert name in registry._backends

    def test_T_4BE_07(self) -> None:
        """T_4BE_07：卸载一个后端实例，注册表中其他后端仍可用。"""
        registry = _reset_registry()

        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )

        backend = CosyVoiceBackend(model_path="/tmp/fake_cosyvoice")
        backend.is_loaded = True
        backend.unload()
        assert backend.is_loaded is False
        # 注册表中 fish 仍可用
        assert "fish" in registry._backends


# ----------------------------------------------------------------------
# T_4BE_08: 自动选择
# ----------------------------------------------------------------------
class TestFourBackendsSelect:
    """四后端自动选择测试。"""

    def test_T_4BE_08(self) -> None:
        """T_4BE_08：auto_select 按需求选择正确后端。

        quality 优先选 CosyVoice（Flow Matching +5.0），
        low_latency 优先选 ChatTTS（+5.0）。
        """
        registry = _reset_registry()

        quality_result = registry.auto_select(
            {"language": "zh", "quality": True}
        )
        assert quality_result == "cosyvoice"

        low_latency_result = registry.auto_select(
            {"language": "zh", "low_latency": True}
        )
        assert low_latency_result == "chattts"


# ----------------------------------------------------------------------
# T_4BE_09 ~ T_4BE_12: 回归不缺失
# ----------------------------------------------------------------------
class TestFourBackendsRegression:
    """四后端回归不缺失测试。"""

    def test_T_4BE_09(self) -> None:
        """T_4BE_09：ChatTTS 仍注册（回归）。"""
        registry = _reset_registry()
        assert "chattts" in registry._backends
        assert registry._specs["chattts"].name == "chattts"

    def test_T_4BE_10(self) -> None:
        """T_4BE_10：Fish 仍注册（回归）。"""
        registry = _reset_registry()
        assert "fish" in registry._backends
        assert registry._specs["fish"].name == "fish"

    def test_T_4BE_11(self) -> None:
        """T_4BE_11：GPT-SoVITS 仍注册（回归）。"""
        registry = _reset_registry()
        assert "sovits" in registry._backends
        assert registry._specs["sovits"].name == "sovits"

    def test_T_4BE_12(self) -> None:
        """T_4BE_12：CosyVoice 已注册且 acoustic_type 为 flow_matching。"""
        registry = _reset_registry()
        assert "cosyvoice" in registry._backends
        assert registry._specs["cosyvoice"].acoustic_type == "flow_matching"
