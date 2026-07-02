# mosaic/core/scheduler.py
"""Mosaic 显存调度器。

管理多个 AI 模型的 GPU 显存生命周期，目标是用户无需手动 ``.to("cuda")``：
节点模型由调度器按需加载，显存不足时按 LRU 策略淘汰最近最少使用的节点。

设计要点
--------
* :class:`Scheduler` 跟踪每个节点的显存占用（来自 ``NodeSpec.model_info``
  的 ``vram_gb``，缺失时回退到 ``torch.cuda`` 实测）。
* ``ensure_loaded`` 实现按需加载 + LRU 淘汰：显存不足时依次卸载最久未
  使用的节点，直到腾出足够空间或无可卸载者。
* **线程安全**：所有公开方法均由 ``RLock`` 保护，支持 Pipeline 并行执行。
* **无 GPU 优雅降级**：检测不到 CUDA 时进入 CPU 模式，跳过显存计量，
  节点照常加载（到 CPU），并发出提示。
* 与 :mod:`mosaic.core.events` 集成：模型加载/卸载时发布 ``model_load``
  / ``model_unload`` 事件。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec

__all__ = [
    "Scheduler",
    "SchedulerStatus",
    "get_scheduler",
]


# ---------------------------------------------------------------------------
# 状态信息
# ---------------------------------------------------------------------------
class SchedulerStatus:
    """调度器状态快照的字典结构（类型注解用）。"""

    def __init__(self) -> None:
        self.mode: str = "cpu"  # "gpu" 或 "cpu"
        self.device: str = "cpu"
        self.memory_total_gb: float = 0.0
        self.memory_used_gb: float = 0.0
        self.memory_limit_gb: float = 0.0
        self.tracked_nodes: list[str] = []
        self.loaded_nodes: list[str] = []
        self.node_memory: dict[str, float] = {}

    def to_dict(self) -> dict[str, Any]:
        """转为纯字典。"""
        return {
            "mode": self.mode,
            "device": self.device,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_limit_gb": self.memory_limit_gb,
            "tracked_nodes": list(self.tracked_nodes),
            "loaded_nodes": list(self.loaded_nodes),
            "node_memory": dict(self.node_memory),
        }


# ---------------------------------------------------------------------------
# Scheduler — 显存调度器
# ---------------------------------------------------------------------------
class Scheduler:
    """GPU 显存调度器，管理节点模型生命周期。

    Parameters
    ----------
    bus:
        事件总线，用于发布 ``model_load``/``model_unload`` 事件。
        ``None`` 表示使用全局单例。
    memory_limit_gb:
        显存使用上限（GB）。``None`` 表示使用 GPU 实际总量（CPU 模式下为 0）。
    device:
        强制指定设备。``None`` 表示自动检测（有 CUDA 用 ``"cuda"``
        否则 ``"cpu"``）。

    线程安全
    --------
    所有公开方法均由可重入锁 ``RLock`` 保护，可在多线程 Pipeline 中安全调用。
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        memory_limit_gb: float | None = None,
        device: str | None = None,
    ) -> None:
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger("mosaic.scheduler")

        # 跟踪状态
        self._tracked: dict[str, Node] = {}  # name -> Node
        self._node_memory: dict[str, float] = {}  # name -> 估算显存(GB)
        self._loaded_names: set = set()
        # LRU 访问顺序：最近访问的在右端
        self._lru: deque[str] = deque()
        # 配置：未显式传入时回退到 MosaicEnv 集中读取的环境变量
        from mosaic.core.env import MosaicEnv

        if device is None:
            device = MosaicEnv.get_device()
        self._device: str = device if device is not None else self._detect_device()
        self._is_gpu: bool = self._device.startswith("cuda")
        self._memory_total_gb: float = self._query_total_memory() if self._is_gpu else 0.0
        if memory_limit_gb is None:
            memory_limit_gb = MosaicEnv.get_memory_limit()
        self._memory_limit_gb: float = (
            memory_limit_gb
            if memory_limit_gb is not None
            else self._memory_total_gb
        )
        # 锁
        self._lock = threading.RLock()

        if not self._is_gpu:
            self._logger.warning(
                "No CUDA-capable GPU detected. Scheduler is running in CPU mode; "
                "memory tracking is disabled and nodes will load onto CPU."
            )

    # -- 设备与显存检测 ----------------------------------------------------
    @staticmethod
    def _detect_device() -> str:
        """自动检测计算设备。"""
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _device_index(self) -> int:
        """从 ``self._device`` 解析 GPU 设备索引。

        支持 ``"cuda"`` / ``"cuda:0"`` / ``"cuda:2"`` 等形式；无 GPU 或
        解析失败时返回 ``0``，避免显存查询硬编码 device 0 而无法支持多 GPU。
        """
        if isinstance(self._device, str) and self._device.startswith("cuda:"):
            try:
                return int(self._device.split(":", 1)[1])
            except (IndexError, ValueError):
                pass
        return 0

    def _query_total_memory(self) -> float:
        """查询 GPU 显存总量（GB）。无 GPU 时返回 0。"""
        if not self._is_gpu:
            return 0.0
        try:
            import torch  # type: ignore

            return (
                torch.cuda.get_device_properties(self._device_index()).total_memory
                / (1024 ** 3)
            )
        except Exception:  # noqa: BLE001
            return 0.0

    def _query_used_memory(self) -> float:
        """查询当前 GPU 已用显存（GB）。无 GPU 时返回 0。"""
        if not self._is_gpu:
            return 0.0
        try:
            import torch  # type: ignore

            return torch.cuda.memory_allocated(self._device_index()) / (1024 ** 3)
        except Exception:  # noqa: BLE001
            return 0.0

    # -- 配置 --------------------------------------------------------------
    @property
    def device(self) -> str:
        """当前调度设备（``"cuda"`` 或 ``"cpu"``）。"""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """是否运行在 GPU 模式。"""
        return self._is_gpu

    def set_memory_limit(self, limit_gb: float) -> None:
        """设置显存使用上限（GB）。

        Parameters
        ----------
        limit_gb:
            新的显存上限。设为 ``0`` 或负数表示不限制（CPU 模式下忽略）。

        Raises
        ------
        ValueError
            上限低于当前已加载节点总占用。
        """
        with self._lock:
            if limit_gb <= 0:
                self._memory_limit_gb = self._memory_total_gb
                return
            current_used = sum(
                self._node_memory.get(n, 0.0) for n in self._loaded_names
            )
            if self._is_gpu and limit_gb < current_used:
                raise ValueError(
                    f"Memory limit {limit_gb}GB is smaller than currently "
                    f"loaded nodes ({current_used}GB). Unload some nodes first."
                )
            self._memory_limit_gb = limit_gb

    # -- 节点跟踪 ----------------------------------------------------------
    def track(self, node: Node) -> None:
        """注册一个节点，跟踪其显存占用。

        若节点已跟踪，则刷新其显存估算值。显存占用优先取
        ``NodeSpec.model_info["vram_gb"]``，缺失则回退到 0（CPU 模式下统一为 0）。

        Parameters
        ----------
        node:
            待跟踪的节点实例。
        """
        with self._lock:
            name = node.name
            self._tracked[name] = node
            self._node_memory[name] = self._estimate_memory(node)

    def _estimate_memory(self, node: Node) -> float:
        """估算节点的显存占用（GB）。"""
        if not self._is_gpu:
            return 0.0
        try:
            spec: NodeSpec = node.describe()
        except Exception:  # noqa: BLE001
            return 0.0
        vram = spec.model_info.get("vram_gb", 0.0) if spec.model_info else 0.0
        try:
            return float(vram)
        except (TypeError, ValueError):
            return 0.0

    # -- 加载与释放 --------------------------------------------------------
    def ensure_loaded(self, node: Node) -> None:
        """确保节点模型已加载到 GPU（按需加载 + LRU 淘汰）。

        * 已加载：直接返回，并刷新 LRU 访问顺序。
        * 未加载：检查显存是否足够；不足则按 LRU 卸载其他节点腾出空间，
          然后加载目标节点。若腾不出足够空间（无可卸载者或单节点即超限），
          在 GPU 模式下抛出 :class:`MemoryError`，CPU 模式下照常加载。

        Parameters
        ----------
        node:
            需要加载的节点。若未 :meth:`track`，会自动跟踪。

        Raises
        ------
        MemoryError
            GPU 模式下显存不足以容纳该节点且无法通过淘汰腾出空间。
        """
        with self._lock:
            name = node.name
            if name not in self._tracked:
                self.track(node)

            # 已加载：刷新 LRU
            if node.is_loaded():
                self._touch_lru(name)
                return

            needed = self._node_memory.get(name, 0.0)

            # GPU 模式：检查并腾出空间（淘汰在锁内完成，保证 LRU 一致）
            if self._is_gpu:
                # 优化：如果目标节点的模型已被另一个已加载节点共享（model_cache
                # 会命中），则实际不需要额外显存，跳过淘汰——避免"先淘汰旧节点
                # （缓存引用归零被删除）→ 再加载新节点（cache miss 从磁盘重载）"
                # 的抖动问题。
                model_name = getattr(node, "_model_name", None)
                if model_name:
                    for loaded_name in list(self._loaded_names):
                        loaded_node = self._nodes.get(loaded_name)
                        if (
                            loaded_node is not None
                            and loaded_name != name
                            and getattr(loaded_node, "_model_name", None)
                            == model_name
                        ):
                            needed = 0.0
                            break
                self._ensure_capacity(needed, exclude=name)

        # 实际加载在锁外执行：``node.load()`` 可能耗时（权重加载/设备迁移），
        # 持锁会阻塞其他调度操作成为性能瓶颈。加载完成后再重新获取锁更新
        # 调度器状态（见 :meth:`_do_load`）。
        # NOTE: 这放弃了“容量检查→加载”的原子性；在多线程并发加载同一节点
        # 或加载期间该节点被淘汰的极端场景下可能产生状态竞争，但显著降低
        # 了锁持有时间。当前调用方（Pipeline）按节点串行加载，可安全受益。
        self._do_load(node, needed)

    def _ensure_capacity(self, needed_gb: float, exclude: str) -> None:
        """确保有 ``needed_gb`` 的可用显存，必要时按 LRU 淘汰其他节点。

        Parameters
        ----------
        needed_gb:
            目标节点需要的显存。
        exclude:
            淘汰时跳过的节点名（目标节点本身，虽此时未加载）。

        Raises
        ------
        MemoryError
            腾不出足够空间。
        """
        # 须持锁调用
        if self._has_capacity(needed_gb):
            return
        # LRU 淘汰：从最久未使用（左端）开始卸载
        evicted: list[str] = []
        while not self._has_capacity(needed_gb) and self._lru:
            # 找到第一个不是 exclude 的元素淘汰。
            # 不使用 rotate(1)——那会把队首元素搬到队尾，破坏 LRU 访问顺序
            # （见 A1）。exclude 通常是目标节点本身，正常情况下不在 LRU 中。
            for i, victim in enumerate(self._lru):
                if victim != exclude:
                    del self._lru[i]
                    self._evict(victim)
                    evicted.append(victim)
                    break
            else:
                # 队列中全是 exclude，无可淘汰者
                break

        if not self._has_capacity(needed_gb):
            total_loaded = sum(
                self._node_memory.get(n, 0.0) for n in self._loaded_names
            )
            raise MemoryError(
                f"Cannot free enough GPU memory for node {exclude!r} "
                f"(needs {needed_gb:.2f}GB). Current loaded={total_loaded:.2f}GB, "
                f"limit={self._memory_limit_gb:.2f}GB. "
                f"Evicted {len(evicted)} node(s): {evicted}."
            )

    def _has_capacity(self, needed_gb: float) -> bool:
        """综合静态估算与实际显存查询判断是否有 ``needed_gb`` 的可用空间。

        须持锁调用。优先依据静态估算（节点声明的 ``vram_gb`` 之和，
        见 :meth:`_fits`）；当能查询到有效的实际显存占用（``> 0``）时，
        作为辅助判断——即便静态估算认为足够，若实际显存已超限也视为
        无足够空间，从而在静态估算偏小时仍能触发淘汰（见 A2）。
        """
        if not self._fits(needed_gb):
            return False
        if self._is_gpu and self._memory_limit_gb > 0:
            actual_used = self._query_used_memory()
            # 仅在获取到有效读数时启用辅助判断：查询失败、CPU 模式或
            # 模拟环境（返回 0）时退化为仅依赖静态估算，避免误判。
            if actual_used > 0:
                if actual_used + needed_gb > self._memory_limit_gb + 1e-9:
                    return False
        return True

    def _fits(self, needed_gb: float) -> bool:
        """判断加入 ``needed_gb`` 后是否仍在显存上限内。须持锁调用。"""
        if self._memory_limit_gb <= 0:
            return True  # 不限制
        current = sum(
            self._node_memory.get(n, 0.0) for n in self._loaded_names
        )
        return current + needed_gb <= self._memory_limit_gb + 1e-9

    def _do_load(self, node: Node, memory_gb: float) -> None:
        """实际加载节点并记录状态、发布事件。

        ``node.load()`` 在锁外执行（见 :meth:`ensure_loaded`）；加载完成后
        重新获取锁更新调度器状态。加载失败时若节点被部分加载，则卸载以
        保持状态一致（见 A6），避免 ``node.is_loaded()`` 为真但调度器
        ``_loaded_names`` 未记录的不一致。
        """
        name = node.name
        try:
            node.load()
        except Exception:  # noqa: BLE001
            # 加载失败：若 node.load() 部分成功（已置 _loaded=True），卸载
            # 以保持节点状态与调度器记录一致。
            try:
                if node.is_loaded():
                    node.unload()
            except Exception:  # noqa: BLE001
                pass
            raise
        with self._lock:
            self._loaded_names.add(name)
            self._touch_lru(name)
            used = memory_gb if self._is_gpu else self._query_used_memory()
            self._bus.emit(
                EventType.MODEL_LOAD,
                node_name=name,
                device=self._device,
                memory_used=used,
            )

    def release(self, node: Node) -> None:
        """主动释放节点显存。

        若节点未加载则无操作。释放后从 LRU 与已加载集合中移除，并发布
        ``model_unload`` 事件。

        Parameters
        ----------
        node:
            待释放的节点。
        """
        with self._lock:
            name = node.name
            if not node.is_loaded():
                self._loaded_names.discard(name)
                self._remove_from_lru(name)
                return
            freed = self._node_memory.get(name, 0.0)
            try:
                node.unload()
            except Exception:
                self._logger.warning(
                    "Node %s unload raised an exception", name, exc_info=True,
                )
            self._loaded_names.discard(name)
            self._remove_from_lru(name)
            if not self._is_gpu:
                freed = 0.0
            self._bus.emit(
                EventType.MODEL_UNLOAD,
                node_name=name,
                memory_freed=freed,
            )

    def _evict(self, name: str) -> None:
        """淘汰一个已加载节点（LRU 内部调用）。须持锁调用。"""
        node = self._tracked.get(name)
        if node is None or not node.is_loaded():
            self._remove_from_lru(name)
            self._loaded_names.discard(name)
            return
        freed = self._node_memory.get(name, 0.0)
        try:
            node.unload()
        except Exception:
            self._logger.warning(
                "Node %s unload raised an exception during eviction",
                name, exc_info=True,
            )
        self._loaded_names.discard(name)
        self._remove_from_lru(name)
        self._bus.emit(
            EventType.MODEL_UNLOAD,
            node_name=name,
            memory_freed=freed,
        )
        self._logger.info(
            "LRU evicted node %r to free ~%.2fGB GPU memory.", name, freed
        )

    # -- LRU 维护 ----------------------------------------------------------
    def _touch_lru(self, name: str) -> None:
        """将节点移到 LRU 最近访问端（右端）。须持锁调用。"""
        self._remove_from_lru(name)
        self._lru.append(name)

    def _remove_from_lru(self, name: str) -> None:
        """从 LRU 队列移除指定节点。须持锁调用。"""
        try:
            self._lru.remove(name)
        except ValueError:
            pass

    # -- 状态查询 ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """返回当前显存使用情况快照。

        Returns
        -------
        dict[str, Any]
            含 ``mode``/``device``/``memory_total_gb``/``memory_used_gb``
            /``memory_limit_gb``/``tracked_nodes``/``loaded_nodes``
            /``node_memory`` 等字段。
        """
        with self._lock:
            st = SchedulerStatus()
            st.mode = "gpu" if self._is_gpu else "cpu"
            st.device = self._device
            st.memory_total_gb = self._memory_total_gb
            st.memory_limit_gb = self._memory_limit_gb
            st.node_memory = {n: m for n, m in self._node_memory.items()}
            st.tracked_nodes = list(self._tracked.keys())
            st.loaded_nodes = [n for n in self._lru if n in self._loaded_names]
            if self._is_gpu:
                # 优先用 GPU 实测已用，回退到加载节点估算之和
                queried = self._query_used_memory()
                if queried > 0:
                    st.memory_used_gb = queried
                else:
                    st.memory_used_gb = sum(
                        self._node_memory.get(n, 0.0) for n in self._loaded_names
                    )
            else:
                st.memory_used_gb = 0.0
            return st.to_dict()

    # -- 便捷方法 ----------------------------------------------------------
    def loaded_count(self) -> int:
        """返回当前已加载节点数。"""
        with self._lock:
            return len(self._loaded_names)

    def release_all(self) -> int:
        """释放所有已加载节点。

        Returns
        -------
        int
            实际卸载的节点数。
        """
        with self._lock:
            names = [n for n in list(self._lru) if n in self._loaded_names]
            count = 0
            for name in names:
                node = self._tracked.get(name)
                if node is not None and node.is_loaded():
                    self._evict(name)
                    count += 1
            return count

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<Scheduler mode={'gpu' if self._is_gpu else 'cpu'} "
                f"device={self._device!r} "
                f"tracked={len(self._tracked)} loaded={len(self._loaded_names)} "
                f"limit={self._memory_limit_gb:.2f}GB>"
            )


# ---------------------------------------------------------------------------
# 全局默认调度器
# ---------------------------------------------------------------------------
_default_scheduler: Scheduler | None = None
_default_scheduler_lock = threading.Lock()


def get_scheduler() -> Scheduler:
    """返回全局默认 :class:`Scheduler` 单例。"""
    global _default_scheduler
    if _default_scheduler is None:
        with _default_scheduler_lock:
            if _default_scheduler is None:
                _default_scheduler = Scheduler()
    return _default_scheduler


def set_scheduler(scheduler: Scheduler | None) -> None:
    """替换全局默认调度器（主要供测试使用）。"""
    global _default_scheduler
    with _default_scheduler_lock:
        _default_scheduler = scheduler
