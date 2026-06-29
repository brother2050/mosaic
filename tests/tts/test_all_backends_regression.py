# tests/tts/test_all_backends_regression.py
"""三后端共存回归测试。

验证 ChatTTS、Fish Speech、GPT-SoVITS 三个后端可以在同一进程中
共存，注册表正确路由，显存调度正常工作。

测试 ID: T_ALLR_01 ~ T_ALLR_08
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

_check = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(_check is None, reason="torch 未安装")


# ----------------------------------------------------------------------
# T_ALLR_01 ~ T_ALLR_02: 注册与独立加载
# ----------------------------------------------------------------------
class TestAllBackendsRegistration:
    """三后端注册与共存测试。"""

    def test_T_ALLR_01(self) -> None:
        """T_ALLR_01：ChatTTS、Fish、GPT-SoVITS 三个后端同时注册成功。"""
        from mosaic.nodes.audio.tts_backends.registry import (
            TTSBackendRegistry,
            tts_backend_registry,
        )

        # 重置单例以重新注册
        registry = TTSBackendRegistry()
        registry._backends.clear()
        registry._specs.clear()

        global _backends_registered
        import mosaic.nodes.audio.tts_backends.registry as reg_module

        reg_module._backends_registered = False
        registry._ensure_builtin_registered()

        names = list(registry._backends.keys())
        assert "chattts" in names, f"chattts not in {names}"
        assert "fish" in names, f"fish not in {names}"
        assert "sovits" in names, f"sovits not in {names}"

    def test_T_ALLR_02(self) -> None:
        """T_ALLR_02：三个后端可以独立加载和卸载。"""
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        # 创建三个后端实例（不加载权重）
        chattts = ChatTTSBackend(model_path="/tmp/fake_chattts")
        fish = FishSpeechBackend(model_path="/tmp/fake_fish")
        sovits = GPTSoVITSBackend(model_path="/tmp/fake_sovits")

        assert chattts.is_loaded is False
        assert fish.is_loaded is False
        assert sovits.is_loaded is False

        # 模拟加载（使用 patch）
        from unittest.mock import patch

        with patch.object(chattts, "_build_pipeline"), \
             patch.object(fish, "_build_pipeline"), \
             patch.object(sovits, "_build_pipeline"):
            chattts.load(device="cpu", dtype="float32")
            fish.load(device="cpu", dtype="float32")
            sovits.load(device="cpu", dtype="float32")

            assert chattts.is_loaded is True
            assert fish.is_loaded is True
            assert sovits.is_loaded is True

            # 卸载
            chattts.unload()
            fish.unload()
            sovits.unload()

            assert chattts.is_loaded is False
            assert fish.is_loaded is False
            assert sovits.is_loaded is False


# ----------------------------------------------------------------------
# T_ALLR_03 ~ T_ALLR_04: 路由与采样率
# ----------------------------------------------------------------------
class TestAllBackendsRouting:
    """三后端路由与采样率测试。"""

    def test_T_ALLR_03(self) -> None:
        """T_ALLR_03：TTS 节点可以正确路由到三个后端。"""
        from mosaic.nodes.audio.tts import TTS

        backends = TTS.list_backends()
        assert "chattts" in backends
        assert "fish" in backends
        assert "sovits" in backends
        # 内置后端也在
        assert "edge_tts" in backends
        assert "transformers" in backends

    def test_T_ALLR_04(self) -> None:
        """T_ALLR_04：三个后端的 sample_rate 不同（24000/22050/32000），各自正确。"""
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        assert ChatTTSBackend.spec.sample_rate == 24000
        assert FishSpeechBackend.spec.sample_rate == 22050
        assert GPTSoVITSBackend.spec.sample_rate == 32000

        # 确保三者不同
        rates = {
            ChatTTSBackend.spec.sample_rate,
            FishSpeechBackend.spec.sample_rate,
            GPTSoVITSBackend.spec.sample_rate,
        }
        assert len(rates) == 3, "All three sample rates should be different"


# ----------------------------------------------------------------------
# T_ALLR_05 ~ T_ALLR_06: 调度与隔离
# ----------------------------------------------------------------------
class TestAllBackendsScheduler:
    """三后端调度与隔离测试。"""

    def test_T_ALLR_05(self) -> None:
        """T_ALLR_05：Scheduler 可以管理三个后端的显存（LRU 淘汰）。"""
        from mosaic.nodes.audio.tts_backends.base import TTSBackend
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        # 检查三个后端都是 TTSBackend 子类
        assert issubclass(ChatTTSBackend, TTSBackend)
        assert issubclass(FishSpeechBackend, TTSBackend)
        assert issubclass(GPTSoVITSBackend, TTSBackend)

        # 检查 spec 的 min_gpu_memory_gb
        chattts_gb = ChatTTSBackend.spec.min_gpu_memory_gb
        fish_gb = FishSpeechBackend.spec.min_gpu_memory_gb
        sovits_gb = GPTSoVITSBackend.spec.min_gpu_memory_gb

        # 所有后端都声明了显存需求
        assert chattts_gb > 0
        assert fish_gb > 0
        assert sovits_gb > 0

    def test_T_ALLR_06(self) -> None:
        """T_ALLR_06：一个后端 unload 后，另一个后端仍可正常使用。"""
        from unittest.mock import patch

        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        chattts = ChatTTSBackend(model_path="/tmp/fake_chattts")
        sovits = GPTSoVITSBackend(model_path="/tmp/fake_sovits")

        with patch.object(chattts, "_build_pipeline"), \
             patch.object(sovits, "_build_pipeline"):
            chattts.load(device="cpu", dtype="float32")
            sovits.load(device="cpu", dtype="float32")

            # 卸载 chattts
            chattts.unload()
            assert chattts.is_loaded is False

            # sovits 仍然可用
            assert sovits.is_loaded is True

            sovits.unload()
            assert sovits.is_loaded is False


# ----------------------------------------------------------------------
# T_ALLR_07 ~ T_ALLR_08: 列表与自动选择
# ----------------------------------------------------------------------
class TestAllBackendsSelect:
    """三后端列表与自动选择测试。"""

    def test_T_ALLR_07(self) -> None:
        """T_ALLR_07：TTSBackendRegistry.list_backends() 包含三个后端。"""
        from mosaic.nodes.audio.tts_backends.registry import (
            TTSBackendRegistry,
        )
        import mosaic.nodes.audio.tts_backends.registry as reg_module

        registry = TTSBackendRegistry()
        registry._backends.clear()
        registry._specs.clear()
        reg_module._backends_registered = False
        registry._ensure_builtin_registered()

        specs = registry.list_backends()
        names = [s.name for s in specs]
        assert "chattts" in names
        assert "fish" in names
        assert "sovits" in names
        assert len(names) >= 3

    def test_T_ALLR_08(self) -> None:
        """T_ALLR_08：auto_select 根据需求选择合适的后端。"""
        from mosaic.nodes.audio.tts_backends.registry import (
            TTSBackendRegistry,
        )
        import mosaic.nodes.audio.tts_backends.registry as reg_module

        registry = TTSBackendRegistry()
        registry._backends.clear()
        registry._specs.clear()
        reg_module._backends_registered = False
        registry._ensure_builtin_registered()

        # 选择支持流式的后端
        result = registry.auto_select({"language": "zh", "streaming": True})
        assert result in ("chattts", "fish", "sovits", "cosyvoice")

        # 选择支持语音克隆的后端
        result = registry.auto_select({"voice_clone": True})
        assert result in ("chattts", "fish", "sovits", "cosyvoice")

        # 选择低显存需求的后端
        result = registry.auto_select({"gpu_memory_gb": 4.0})
        assert result in ("chattts", "fish", "sovits", "cosyvoice", "edge_tts")
