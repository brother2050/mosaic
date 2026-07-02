"""跨节点的模型/Pipeline 实例缓存。

避免同一模型被多个节点实例重复加载，通过
``(pipeline_class_name, model_name, dtype, device)`` 作为缓存键共享已加载
的 Pipeline 实例。缓存设有最大容量与 LRU 淘汰策略，防止显存被无限制占用。

引用计数
--------
当多个节点共享同一缓存实例时（如多个文本节点都用 Qwen-7B），缓存通过
引用计数跟踪活跃使用者。``get`` 命中时引用 +1，``remove`` 时引用 -1，
仅当引用归零才真正删除条目。这避免了一个节点 ``unload`` 把共享对象
``.to("cpu")`` 后破坏其他正在使用该对象的节点。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class _CacheEntry:
    """缓存条目：持有模型对象与引用计数。"""

    __slots__ = ("obj", "refcount")

    def __init__(self, obj: Any) -> None:
        self.obj: Any = obj
        self.refcount: int = 1


class ModelCache:
    """模型实例缓存（线程安全，带 LRU 淘汰与引用计数）。

    以 ``(pipeline_class_name, model_name, dtype_str, device)`` 为键缓存
    已加载的 Pipeline/Model 实例，避免同模型重复加载。``device`` 纳入缓存键
    可区分同一模型加载到不同设备上的实例。

    当缓存条目数达到 :attr:`MAX_CACHE_SIZE` 时，按 LRU 策略淘汰最久未访问
    的条目（仅淘汰引用计数为 0 的条目；若所有条目均有活跃引用，跳过淘汰
    并记录警告）。

    引用计数
    ~~~~~~~~
    ``get`` 命中时引用 +1，``put`` 新建时引用 =1，``remove`` 时引用 -1。
    仅当引用归零时 ``remove`` 才真正删除条目并返回 ``True``，调用方可据此
    判断是否安全地对模型执行 ``.to("cpu")`` 等搬运操作——避免破坏仍在被
    其他节点使用的共享对象。

    使用方式::

        from mosaic.core.model_cache import model_cache

        # 尝试从缓存获取（命中时引用 +1）
        pipe = model_cache.get("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", "cuda")
        if pipe is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(...)
            model_cache.put("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", "cuda", pipe)

        # 卸载时引用 -1，仅当返回 True（引用归零）才安全搬运到 CPU
        released = model_cache.remove("StableDiffusionXLPipeline", "stabilityai/sdxl", "fp16", "cuda")
        if released:
            pipe.to("cpu")
    """

    #: 缓存最大条目数，超出后按 LRU 淘汰最久未使用的条目。
    MAX_CACHE_SIZE: int = 10

    def __init__(self, max_cache_size: int | None = None) -> None:
        # OrderedDict 保持访问顺序：最近访问的在右端，最久未访问的在左端。
        self._cache: "OrderedDict[tuple[str, str, str, str], _CacheEntry]" = OrderedDict()
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
        ``pipeline_class`` 可以是字符串或类对象；对不支持 ``__name__`` 的
        对象（如 unittest.mock.MagicMock）回退到 ``str()``。
        """
        if isinstance(pipeline_class, str):
            class_name = pipeline_class
        else:
            name = getattr(pipeline_class, "__name__", None)
            class_name = name if isinstance(name, str) else str(pipeline_class)
        return (class_name, model_name, dtype, str(device))

    def get(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        device: str | None = None,
    ) -> Any | None:
        """从缓存获取模型实例。

        命中时将该条目移动到 LRU 最近访问端，并**引用 +1**。

        Returns
        -------
        Any | None
            缓存的模型实例，未命中返回 None。
        """
        if not self._enabled:
            return None
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.refcount += 1
                self._cache.move_to_end(key)
                logger.info(
                    "Model cache hit: %s/%s (%s, %s), refcount=%d",
                    key[0], key[1], key[2], key[3], entry.refcount,
                )
                return entry.obj
            return None

    def contains(
        self,
        pipeline_class: str | type | None,
        model_name: str,
        dtype: str | None,
        device: str | None = None,
    ) -> bool:
        """探测缓存中是否存在指定键的条目（不增加引用计数）。

        用于 scheduler 在淘汰决策时判断目标节点是否会 cache hit，
        而不会像 :meth:`get` 那样增加 refcount。

        Returns
        -------
        bool
            存在返回 True，不存在或缓存禁用返回 False。
        """
        if not self._enabled:
            return False
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            return key in self._cache

    def put(
        self,
        pipeline_class: str | type,
        model_name: str,
        dtype: str,
        obj: Any,
        device: str | None = None,
    ) -> None:
        """将模型实例放入缓存。

        若键已存在，仅更新对象（保留原引用计数）并移到 LRU 最近端。
        若为新键且缓存已达上限，按 LRU 淘汰引用计数为 0 的最旧条目。
        新条目的初始引用计数为 1（对应首次加载的节点）。
        """
        if not self._enabled:
            return
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            if key in self._cache:
                logger.debug("Model cache already has key %s, replacing", key)
                self._cache[key].obj = obj
                self._cache.move_to_end(key)
            else:
                # 容量上限：淘汰最久未使用（左端）且引用计数为 0 的条目
                while len(self._cache) >= self.MAX_CACHE_SIZE:
                    # 寻找第一个 refcount=0 的可淘汰条目
                    evicted = False
                    for ek, ev in list(self._cache.items()):
                        if ev.refcount <= 0:
                            del self._cache[ek]
                            logger.info(
                                "Model cache LRU evicted: %s/%s (%s, %s)",
                                ek[0], ek[1], ek[2], ek[3],
                            )
                            evicted = True
                            break
                    if not evicted:
                        logger.warning(
                            "Model cache full (%d entries, all in-use); "
                            "cannot evict for new key %s/%s",
                            len(self._cache), key[0], key[1],
                        )
                        break
                self._cache[key] = _CacheEntry(obj)
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
    ) -> bool:
        """从缓存减少引用计数，归零时删除条目。

        Returns
        -------
        bool
            ``True`` 表示引用已归零、条目已删除——调用方可安全地对模型
            执行 ``.to("cpu")`` 等搬运操作。``False`` 表示仍有其他节点
            在使用该共享对象，调用方**不应**搬运或修改对象。
        """
        key = self._make_key(pipeline_class, model_name, dtype, device)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                logger.debug(
                    "Model cache remove miss: %s/%s (%s, %s)",
                    key[0], key[1], key[2], key[3],
                )
                return False
            entry.refcount -= 1
            if entry.refcount <= 0:
                del self._cache[key]
                logger.info(
                    "Model cache removed (refcount→0): %s/%s (%s, %s)",
                    key[0], key[1], key[2], key[3],
                )
                return True
            logger.info(
                "Model cache release (refcount=%d): %s/%s (%s, %s)",
                entry.refcount, key[0], key[1], key[2], key[3],
            )
            return False

    def clear(self) -> None:
        """清空缓存并释放 GPU 显存。"""
        with self._lock:
            count = len(self._cache)
            # 将所有 Pipeline/模型移至 CPU 加速显存回收
            for _key, entry in self._cache.items():
                try:
                    if hasattr(entry.obj, "to"):
                        entry.obj.to("cpu")
                except Exception:
                    pass
            self._cache.clear()
            if count:
                logger.info("Model cache cleared (%d entries)", count)
        # 在锁外触发 GPU 显存回收（CUDA/MPS）
        from mosaic.core.device_utils import empty_device_cache

        empty_device_cache()

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
