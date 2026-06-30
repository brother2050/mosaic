"""跨节点的模型/Pipeline 实例缓存。

避免同一模型被多个节点实例重复加载，通过
``(pipeline_class_name, model_name, dtype, device)`` 作为缓存键共享已加载
的 Pipeline 实例。缓存设有最大容量与 LRU 淘汰策略，防止显存被无限制占用。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ModelCache:
    """模型实例缓存（线程安全，带 LRU 淘汰）。

    以 ``(pipeline_class_name, model_name, dtype_str, device)`` 为键缓存
    已加载的 Pipeline/Model 实例，避免同模型重复加载。``device`` 纳入缓存键
    可区分同一模型加载到不同设备上的实例。

    当缓存条目数达到 :attr:`MAX_CACHE_SIZE` 时，按 LRU 策略淘汰最久未访问
    的条目。

    使用方式::

        from mosaic.core.model_cache import model_cache

        # 尝试从缓存获取
        pipe = model_cache.get("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", "cuda")
        if pipe is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(...)
            model_cache.put("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", "cuda", pipe)
    """

    #: 缓存最大条目数，超出后按 LRU 淘汰最久未使用的条目。
    MAX_CACHE_SIZE: int = 10

    def __init__(self, max_cache_size: int | None = None) -> None:
        # OrderedDict 保持访问顺序：最近访问的在右端，最久未访问的在左端。
        self._cache: "OrderedDict[tuple[str, str, str, str], Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._enabled: bool = True
        if max_cache_size is not None:
            self.MAX_CACHE_SIZE = int(max_cache_size)

    def _make_key(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        device: str | None = None,
    ) -> tuple[str, str, str, str]:
        """生成缓存键。

        ``device`` 纳入键中，以区分同一模型加载到不同设备（如 ``cuda`` 与
        ``cpu``）的实例；为 ``None`` 时以字符串 ``"None"`` 占位，保持向后兼容。
        """
        class_name = (
            pipeline_class
            if isinstance(pipeline_class, str)
            else pipeline_class.__name__
        )
        return (class_name, model_name, dtype, str(device))

    def get(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        device: str | None = None,
    ) -> Any | None:
        """从缓存获取模型实例。

        命中时将该条目移动到 LRU 最近访问端。

        Returns
        -------
        Any | None
            缓存的模型实例，未命中返回 None。
        """
        if not self._enabled:
            return None
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            obj = self._cache.get(key)
            if obj is not None:
                self._cache.move_to_end(key)
                logger.info(
                    "Model cache hit: %s/%s (%s, %s)",
                    key[0], key[1], key[2], key[3],
                )
            return obj

    def put(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        obj: Any,
        device: str | None = None,
    ) -> None:
        """将模型实例放入缓存。

        若缓存已达 :attr:`MAX_CACHE_SIZE` 且为新键，则淘汰最久未使用的条目。
        """
        if not self._enabled:
            return
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            if key in self._cache:
                logger.debug("Model cache already has key %s, replacing", key)
                self._cache.move_to_end(key)
            else:
                # 容量上限：淘汰最久未使用（左端）的条目
                while len(self._cache) >= self.MAX_CACHE_SIZE:
                    evicted_key, _evicted_obj = self._cache.popitem(last=False)
                    logger.info(
                        "Model cache LRU evicted: %s/%s (%s, %s)",
                        evicted_key[0], evicted_key[1],
                        evicted_key[2], evicted_key[3],
                    )
            self._cache[key] = obj
            logger.info(
                "Model cache stored: %s/%s (%s, %s), total=%d",
                key[0], key[1], key[2], key[3], len(self._cache),
            )

    def remove(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        device: str | None = None,
    ) -> None:
        """从缓存移除模型实例。"""
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(
                    "Model cache removed: %s/%s (%s, %s)",
                    key[0], key[1], key[2], key[3],
                )

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

    def __contains__(self, key: Any) -> bool:
        return key in self._cache


# 全局单例
model_cache = ModelCache()
