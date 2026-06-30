"""跨节点的模型/Pipeline 实例缓存。

避免同一模型被多个节点实例重复加载，通过 (pipeline_class_name, model_name, dtype) 
作为缓存键共享已加载的 Pipeline 实例。
"""
from __future__ import annotations
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ModelCache:
    """模型实例缓存（线程安全）。

    以 (pipeline_class_name, model_name, dtype_str) 为键缓存已加载的
    Pipeline/Model 实例，避免同模型重复加载。

    使用方式::

        from mosaic.core.model_cache import model_cache

        # 尝试从缓存获取
        pipe = model_cache.get("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16")
        if pipe is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(...)
            model_cache.put("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", pipe)
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()
        self._enabled: bool = True

    def _make_key(
        self, pipeline_class: str | type, model_name: str, dtype: str
    ) -> tuple[str, str, str]:
        """生成缓存键。"""
        class_name = pipeline_class if isinstance(pipeline_class, str) else pipeline_class.__name__
        return (class_name, model_name, dtype)

    def get(
        self, pipeline_class: str | type, model_name: str, dtype: str
    ) -> Any | None:
        """从缓存获取模型实例。

        Returns
        -------
        Any | None
            缓存的模型实例，未命中返回 None。
        """
        if not self._enabled:
            return None
        key = self._make_key(pipeline_class, model_name, dtype)
        with self._lock:
            obj = self._cache.get(key)
            if obj is not None:
                logger.info(
                    "Model cache hit: %s/%s (%s)", key[0], key[1], key[2]
                )
            return obj

    def put(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        obj: Any,
    ) -> None:
        """将模型实例放入缓存。"""
        if not self._enabled:
            return
        key = self._make_key(pipeline_class, model_name, dtype)
        with self._lock:
            if key in self._cache:
                logger.debug("Model cache already has key %s, replacing", key)
            self._cache[key] = obj
            logger.info(
                "Model cache stored: %s/%s (%s), total=%d",
                key[0], key[1], key[2], len(self._cache),
            )

    def remove(
        self, pipeline_class: str | type, model_name: str, dtype: str
    ) -> None:
        """从缓存移除模型实例。"""
        key = self._make_key(pipeline_class, model_name, dtype)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info("Model cache removed: %s/%s", key[0], key[1])

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            if count:
                logger.info("Model cache cleared (%d entries)", count)

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用缓存。"""
        self._enabled = enabled
        if not enabled:
            self.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple[str, str, str]) -> bool:
        return key in self._cache


# 全局单例
model_cache = ModelCache()
