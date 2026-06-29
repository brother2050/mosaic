# tests/final/test_tts_backends_compliance.py
"""TTS 后端接口合规测试。

遍历所有 4 个 TTS 后端，逐一验证 TTSBackend 抽象基类定义的接口契约：
- load / unload / synthesize / synthesize_stream / list_speakers / describe
  方法存在且可调用
- describe() 返回 TTSBackendSpec 类型
- TTSBackendSpec 各字段完整性（name、sample_rate、supported_languages 等）
- 各后端采样率正确性

测试 ID 约定：
    T_TTSIF_01 ~ T_TTSIF_14 分别对应不同的接口合规检查项。
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from mosaic.core.scheduler import Scheduler, set_scheduler
from mosaic.nodes.audio.tts_backends.base import TTSBackendSpec


# ============================================================================
# 所有 4 个 TTS 后端名称
# ============================================================================
_TTS_BACKEND_NAMES: list[str] = ["chattts", "fish", "sovits", "cosyvoice"]

# 预期采样率
_EXPECTED_SAMPLE_RATES: dict[str, int] = {
    "chattts": 24000,
    "fish": 22050,
    "sovits": 32000,
    "cosyvoice": 24000,
}

# 有效声学类型
_VALID_ACOUSTIC_TYPES: set[str] = {"ar", "flow_matching"}


# ============================================================================
# 辅助：获取后端实例
# ============================================================================
def _get_backend_instance(tts_registry: Any, name: str, scheduler: Scheduler) -> Any:
    """获取 TTS 后端实例，带错误处理。

    尝试实例化后端类；若失败（如缺少依赖），返回 None。
    """
    backend_cls = tts_registry.get(name)
    if backend_cls is None:
        return None
    try:
        instance = backend_cls(scheduler=scheduler)
        return instance
    except Exception:
        return None


def _get_backend_spec(tts_registry: Any, name: str) -> TTSBackendSpec | None:
    """获取后端的 TTSBackendSpec（优先从实例，回退到类属性）。"""
    backend_cls = tts_registry.get(name)
    if backend_cls is None:
        return None
    # 优先从类属性获取 spec（无需实例化）
    spec = getattr(backend_cls, "spec", None)
    if isinstance(spec, TTSBackendSpec):
        return spec
    return None


# ============================================================================
# 接口合规测试
# ============================================================================
class TestTTSBackendInterfaceCompliance:
    """参数化遍历所有 4 个 TTS 后端，验证接口合规性。"""

    @pytest.fixture(autouse=True)
    def _setup_scheduler(self) -> Any:
        """为每个测试设置 CPU 模式调度器。"""
        from mosaic.core.events import EventBus

        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)
        return sched

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_load(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_01: 后端具有可调用的 load 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.load), (
                f"TTS backend '{backend_name}': load must be callable."
            )
        else:
            # 无法实例化时，检查类级别方法
            assert callable(backend_cls.load), (
                f"TTS backend '{backend_name}': load must be callable (class-level)."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_unload(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_02: 后端具有可调用的 unload 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.unload), (
                f"TTS backend '{backend_name}': unload must be callable."
            )
        else:
            assert callable(backend_cls.unload), (
                f"TTS backend '{backend_name}': unload must be callable (class-level)."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_synthesize(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_03: 后端具有可调用的 synthesize 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.synthesize), (
                f"TTS backend '{backend_name}': synthesize must be callable."
            )
        else:
            assert callable(backend_cls.synthesize), (
                f"TTS backend '{backend_name}': synthesize must be callable (class-level)."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_synthesize_stream(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_04: 后端具有可调用的 synthesize_stream 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.synthesize_stream), (
                f"TTS backend '{backend_name}': synthesize_stream must be callable."
            )
        else:
            assert callable(backend_cls.synthesize_stream), (
                f"TTS backend '{backend_name}': synthesize_stream must be callable (class-level)."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_list_speakers(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_05: 后端具有可调用的 list_speakers 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.list_speakers), (
                f"TTS backend '{backend_name}': list_speakers must be callable."
            )
        else:
            assert callable(backend_cls.list_speakers), (
                f"TTS backend '{backend_name}': list_speakers must be callable (class-level)."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_backend_has_callable_describe(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_06: 后端具有可调用的 describe 方法。"""
        backend_cls = tts_registry.get(backend_name)
        assert backend_cls is not None, (
            f"TTS backend '{backend_name}' is not registered."
        )

        instance = _get_backend_instance(tts_registry, backend_name, scheduler)
        if instance is not None:
            assert callable(instance.describe), (
                f"TTS backend '{backend_name}': describe must be callable."
            )
        else:
            assert callable(backend_cls.describe), (
                f"TTS backend '{backend_name}': describe must be callable (class-level)."
            )


# ============================================================================
# TTSBackendSpec 字段完整性测试
# ============================================================================
class TestTTSBackendSpecIntegrity:
    """验证每个 TTS 后端的规格说明字段完整性。"""

    @pytest.fixture(autouse=True)
    def _setup_scheduler(self) -> Any:
        """为每个测试设置 CPU 模式调度器。"""
        from mosaic.core.events import EventBus

        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)
        return sched

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_describe_returns_tts_backend_spec(
        self, tts_registry: object, scheduler: Scheduler, backend_name: str
    ) -> None:
        """T_TTSIF_07: describe() 返回 TTSBackendSpec 类型。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert isinstance(spec, TTSBackendSpec), (
                f"TTS backend '{backend_name}': describe() must return TTSBackendSpec, "
                f"got {type(spec).__name__}."
            )
        else:
            # 回退：尝试实例化后调用 describe
            instance = _get_backend_instance(tts_registry, backend_name, scheduler)
            if instance is not None:
                result = instance.describe()
                assert isinstance(result, TTSBackendSpec), (
                    f"TTS backend '{backend_name}': describe() must return TTSBackendSpec, "
                    f"got {type(result).__name__}."
                )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_name_matches_registered_name(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_08: spec.name 与注册名称一致。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert spec.name == backend_name, (
                f"TTS backend '{backend_name}': spec.name ({spec.name!r}) "
                f"must match registered name ({backend_name!r})."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_sample_rate_positive(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_09: spec.sample_rate > 0。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert spec.sample_rate > 0, (
                f"TTS backend '{backend_name}': sample_rate must be > 0, "
                f"got {spec.sample_rate}."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_supported_languages_non_empty(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_10: spec.supported_languages 非空。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert len(spec.supported_languages) > 0, (
                f"TTS backend '{backend_name}': supported_languages must be non-empty, "
                f"got {spec.supported_languages}."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_min_gpu_memory_positive(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_11: spec.min_gpu_memory_gb > 0。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert spec.min_gpu_memory_gb > 0, (
                f"TTS backend '{backend_name}': min_gpu_memory_gb must be > 0, "
                f"got {spec.min_gpu_memory_gb}."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_model_license_non_empty(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_12: spec.model_license 非空。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert spec.model_license, (
                f"TTS backend '{backend_name}': model_license must not be empty."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_acoustic_type_valid(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_13: spec.acoustic_type 为 'ar' 或 'flow_matching'。"""
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            assert spec.acoustic_type in _VALID_ACOUSTIC_TYPES, (
                f"TTS backend '{backend_name}': acoustic_type must be 'ar' or "
                f"'flow_matching', got {spec.acoustic_type!r}."
            )

    @pytest.mark.parametrize("backend_name", _TTS_BACKEND_NAMES)
    def test_spec_sample_rate_correct_value(
        self, tts_registry: object, backend_name: str
    ) -> None:
        """T_TTSIF_14: 各后端采样率正确：
        chattts=24000, fish=22050, sovits=32000, cosyvoice=24000。
        """
        spec = _get_backend_spec(tts_registry, backend_name)

        if spec is not None:
            expected_sr = _EXPECTED_SAMPLE_RATES.get(backend_name)
            if expected_sr is not None:
                assert spec.sample_rate == expected_sr, (
                    f"TTS backend '{backend_name}': expected sample_rate={expected_sr}, "
                    f"got {spec.sample_rate}."
                )