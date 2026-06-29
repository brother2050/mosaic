# mosaic/nodes/audio/tts_backends/registry.py
"""TTS 后端注册表。

提供后端的注册、查询、按需求自动选择与可用性检查能力。采用单例模式，
全局实例 :data:`tts_backend_registry` 供各处共享。

设计要点
--------
* 单例模式：所有 :class:`TTSBackendRegistry` 实例共享同一份注册表。
* 维护两张表：``name -> backend_class`` 与 ``name -> TTSBackendSpec``。
* :meth:`auto_select` 根据语言/流式/克隆/显存等需求筛选并排序后端，
  无可用项时回退到 ``"edge_tts"``（云端、免 GPU）。
* :meth:`is_available` 在注册基础上，若后端类提供 ``check_dependencies``
  类方法，则进一步校验运行时依赖是否就绪。
"""

from __future__ import annotations

import logging
from typing import Any

from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec

__all__ = ["TTSBackendRegistry", "tts_backend_registry"]

logger = logging.getLogger("mosaic.tts.backends.registry")


class TTSBackendRegistry:
    """TTS 后端注册表（单例模式）。

    维护 ``name -> backend_class`` 与 ``name -> TTSBackendSpec`` 两张表，
    支持按语言/流式/克隆/显存等需求自动选择最优后端。

    Examples
    --------
    >>> from mosaic.nodes.audio.tts_backends.registry import tts_backend_registry
    >>> tts_backend_registry.register("my_tts", MyTTSBackend)
    >>> cls = tts_backend_registry.get("my_tts")
    >>> best = tts_backend_registry.auto_select({"language": "zh"})
    """

    _instance: TTSBackendRegistry | None = None

    def __new__(cls) -> TTSBackendRegistry:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._backends = {}  # name -> backend_class
            instance._specs = {}  # name -> TTSBackendSpec
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        # __new__ 已完成初始化；此处避免重复初始化（每次 TTSBackendRegistry()
        # 调用都会触发 __init__）
        if not hasattr(self, "_backends"):
            self._backends: dict[str, type[TTSBackend]] = {}
            self._specs: dict[str, TTSBackendSpec] = {}

    def _ensure_builtin_registered(self) -> None:
        """确保内置后端已注册（延迟注册）。

        在首次访问注册表（查询/选择/可用性检查）时触发内置后端的注册，
        避免在模块加载阶段产生对可选依赖（如 ``torch``）的硬依赖。
        """
        global _backends_registered
        if not _backends_registered:
            _register_builtin_backends()
            _backends_registered = True

    # ------------------------------------------------------------------
    # 注册与查询
    # ------------------------------------------------------------------
    def register(self, name: str, backend_class: type[TTSBackend]) -> None:
        """注册一个 TTS 后端。

        若 ``backend_class`` 拥有 ``spec`` 类属性且为 :class:`TTSBackendSpec`
        实例，则同时登记其规格信息。

        Parameters
        ----------
        name:
            后端唯一名称（非空字符串）。
        backend_class:
            :class:`TTSBackend` 子类。

        Raises
        ------
        ValueError
            ``name`` 为空。
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Backend name must be a non-empty string.")
        self._backends[name] = backend_class
        spec = getattr(backend_class, "spec", None)
        if isinstance(spec, TTSBackendSpec):
            self._specs[name] = spec
        logger.info("Registered TTS backend %r.", name)

    def get(self, name: str) -> type[TTSBackend] | None:
        """按名称获取后端类。

        Parameters
        ----------
        name:
            后端名称。

        Returns
        -------
        type[TTSBackend] | None
            后端类；未注册时返回 ``None``。
        """
        self._ensure_builtin_registered()
        return self._backends.get(name)

    def list_backends(self) -> list[TTSBackendSpec]:
        """列出所有已注册后端的规格信息。

        Returns
        -------
        list[TTSBackendSpec]
            已注册后端的规格列表。
        """
        self._ensure_builtin_registered()
        result: list[TTSBackendSpec] = []
        for name, backend_class in self._backends.items():
            spec = self._specs.get(name)
            if spec is None:
                # 兜底：再次从类属性读取（可能 register 时缺少 spec）
                spec = getattr(backend_class, "spec", None)
            if isinstance(spec, TTSBackendSpec):
                result.append(spec)
        return result

    # ------------------------------------------------------------------
    # 自动选择与可用性
    # ------------------------------------------------------------------
    def auto_select(self, requirements: dict[str, Any]) -> str:
        """根据需求自动选择最优后端。

        Parameters
        ----------
        requirements:
            需求字典，可包含：

            * ``language`` (str): 要求支持的语言代码。
            * ``streaming`` (bool): 是否需要流式合成。
            * ``voice_clone`` (bool): 是否需要语音克隆。
            * ``gpu_memory_gb`` (float): 可用 GPU 显存（GB）。

        Returns
        -------
        str
            最优后端名称；无可用后端时返回 ``"edge_tts"``（回退）。
        """
        self._ensure_builtin_registered()
        language = requirements.get("language")
        streaming = bool(requirements.get("streaming", False))
        voice_clone = bool(requirements.get("voice_clone", False))
        gpu_memory_gb = requirements.get("gpu_memory_gb")

        candidates: list[tuple[str, TTSBackendSpec]] = []
        for name, backend_class in self._backends.items():
            spec = self._specs.get(name)
            if spec is None:
                spec = getattr(backend_class, "spec", None)
            if not isinstance(spec, TTSBackendSpec):
                continue

            # 语言过滤：后端声明了语言列表且不含目标语言则跳过；
            # 未声明语言列表（空）时不做语言过滤（视为兼容）。
            if (
                language
                and spec.supported_languages
                and language not in spec.supported_languages
            ):
                continue
            # 流式过滤
            if streaming and not spec.supports_streaming:
                continue
            # 克隆过滤
            if voice_clone and not spec.supports_voice_clone:
                continue
            # 显存过滤
            if (
                gpu_memory_gb is not None
                and spec.min_gpu_memory_gb > float(gpu_memory_gb)
            ):
                continue

            candidates.append((name, spec))

        if not candidates:
            logger.info(
                "No TTS backend matches requirements %r; "
                "fallback to 'edge_tts'.",
                requirements,
            )
            return "edge_tts"

        # 排序：优先显存需求小（资源占用少），其次按名称稳定排序
        candidates.sort(key=lambda item: (item[1].min_gpu_memory_gb, item[0]))
        best = candidates[0][0]
        logger.info(
            "Auto-selected TTS backend %r for requirements %r.",
            best,
            requirements,
        )
        return best

    def is_available(self, name: str) -> bool:
        """检查指定后端是否已注册且依赖可用。

        Parameters
        ----------
        name:
            后端名称。

        Returns
        -------
        bool
            已注册且依赖可用返回 ``True``；否则 ``False``。

        Notes
        -----
        若后端类提供 ``check_dependencies`` 类方法（返回 bool），则委托其
        做运行时依赖校验；否则仅以注册状态为准。
        """
        self._ensure_builtin_registered()
        backend_class = self._backends.get(name)
        if backend_class is None:
            return False
        check = getattr(backend_class, "check_dependencies", None)
        if callable(check):
            try:
                return bool(check())
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Dependency check failed for %r: %s", name, exc
                )
                return False
        return True

    # ------------------------------------------------------------------
    # 便捷协议
    # ------------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._backends

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"<TTSBackendRegistry backends={list(self._backends)}>"


# 全局单例
tts_backend_registry = TTSBackendRegistry()


# 注册内置后端
def _register_builtin_backends() -> None:
    """注册内置 TTS 后端。

    延迟注册：仅在实际需要时才导入后端实现，避免硬依赖。当可选依赖
    （如 ``torch``）或后端实现模块不可用时，静默跳过。
    """
    try:
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )

        tts_backend_registry.register("chattts", ChatTTSBackend)
    except Exception:
        pass  # 依赖不可用时静默跳过

    try:
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )

        tts_backend_registry.register("fish", FishSpeechBackend)
    except Exception:
        pass  # 依赖不可用时静默跳过


# 不在模块加载时注册，而是在首次使用时延迟注册
_backends_registered = False


def _ensure_backends_registered() -> None:
    """确保内置后端已注册（延迟注册）。"""
    global _backends_registered
    if not _backends_registered:
        _register_builtin_backends()
        _backends_registered = True
