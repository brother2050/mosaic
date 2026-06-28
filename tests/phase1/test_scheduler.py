# tests/phase1/test_scheduler.py
"""Phase 1 调度器测试。

覆盖 Scheduler 的 track、status、ensure_loaded、release、LRU 淘汰与优雅降级。
"""

from __future__ import annotations

import pytest

from mosaic.core.scheduler import Scheduler, get_scheduler, set_scheduler
from mosaic.core.events import EventBus, EventType
from mosaic.core.node import Node, NodeSpec
from mosaic.core.types import MosaicData, TextData


# ===========================================================================
# 辅助节点类
# ===========================================================================
class _TestNode(Node):
    """用于调度器测试的节点。"""

    name = "test-node"
    domain = "test"
    description = "Test node"
    version = "0.1.0"
    input_types = ["text"]
    output_types = ["text"]

    def __init__(self, name="test-node", vram_gb=4.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self._vram_gb = vram_gb
        self._load_count = 0
        self._unload_count = 0

    def load(self):
        self._load_count += 1
        self._loaded = True

    def unload(self):
        self._unload_count += 1
        self._loaded = False

    def run(self, input_data):
        return input_data

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info={"vram_gb": self._vram_gb},
        )


class _HeavyNode(_TestNode):
    """显存占用较大的节点（用于测试 LRU 淘汰）。"""

    name = "heavy-node"

    def __init__(self, name="heavy-node", vram_gb=8.0, **kwargs):
        super().__init__(name=name, vram_gb=vram_gb, **kwargs)


# ===========================================================================
# T_SCHED_01: track 和 status 基本功能
# ===========================================================================
class TestSchedulerTrackAndStatus:
    """调度器 track 和 status 测试。"""

    def test_track_registers_node(self, cpu_scheduler):
        """T_SCHED_01: track 注册节点。"""
        node = _TestNode(name="n1")
        cpu_scheduler.track(node)
        st = cpu_scheduler.status()
        assert "n1" in st["tracked_nodes"]

    def test_status_returns_dict(self, cpu_scheduler):
        """T_SCHED_01: status 返回字典格式。"""
        st = cpu_scheduler.status()
        assert isinstance(st, dict)
        assert "mode" in st
        assert "device" in st
        assert "loaded_nodes" in st

    def test_cpu_mode_status(self, cpu_scheduler):
        """T_SCHED_01: CPU 模式状态正确。"""
        st = cpu_scheduler.status()
        assert st["mode"] == "cpu"
        assert st["device"] == "cpu"

    def test_loaded_count(self, cpu_scheduler):
        """T_SCHED_01: loaded_count 返回已加载节点数。"""
        assert cpu_scheduler.loaded_count() == 0
        node = _TestNode(name="n1")
        cpu_scheduler.track(node)
        cpu_scheduler.ensure_loaded(node)
        assert cpu_scheduler.loaded_count() == 1

    def test_repr(self, cpu_scheduler):
        """T_SCHED_01: repr 包含模式信息。"""
        r = repr(cpu_scheduler)
        assert "cpu" in r


# ===========================================================================
# T_SCHED_02: ensure_loaded 加载模型
# ===========================================================================
class TestSchedulerEnsureLoaded:
    """调度器 ensure_loaded 测试。"""

    def test_ensure_loaded_loads_node(self, cpu_scheduler):
        """T_SCHED_02: ensure_loaded 加载节点。"""
        node = _TestNode(name="n1")
        assert not node.is_loaded()
        cpu_scheduler.ensure_loaded(node)
        assert node.is_loaded()
        assert node._load_count == 1

    def test_ensure_loaded_already_loaded(self, cpu_scheduler):
        """T_SCHED_02: 已加载节点 ensure_loaded 不重复加载。"""
        node = _TestNode(name="n1")
        cpu_scheduler.ensure_loaded(node)
        assert node._load_count == 1
        cpu_scheduler.ensure_loaded(node)
        assert node._load_count == 1  # 不应重复加载

    def test_ensure_loaded_auto_tracks(self, cpu_scheduler):
        """T_SCHED_02: ensure_loaded 自动 track 未注册节点。"""
        node = _TestNode(name="n2")
        assert "n2" not in cpu_scheduler.status()["tracked_nodes"]
        cpu_scheduler.ensure_loaded(node)
        assert "n2" in cpu_scheduler.status()["tracked_nodes"]

    def test_ensure_loaded_fires_event(self, cpu_scheduler):
        """T_SCHED_02: ensure_loaded 触发 model_load 事件。"""
        events = []
        cpu_scheduler._bus.on(EventType.MODEL_LOAD, lambda e: events.append(e))
        node = _TestNode(name="n1")
        cpu_scheduler.ensure_loaded(node)
        assert len(events) == 1
        assert events[0].type == EventType.MODEL_LOAD


# ===========================================================================
# T_SCHED_03: release 释放模型
# ===========================================================================
class TestSchedulerRelease:
    """调度器 release 测试。"""

    def test_release_unloads_node(self, cpu_scheduler):
        """T_SCHED_03: release 卸载节点。"""
        node = _TestNode(name="n1")
        cpu_scheduler.ensure_loaded(node)
        assert node.is_loaded()
        cpu_scheduler.release(node)
        assert not node.is_loaded()
        assert node._unload_count == 1

    def test_release_unloaded_node_noop(self, cpu_scheduler):
        """T_SCHED_03: 释放未加载节点无操作。"""
        node = _TestNode(name="n1")
        cpu_scheduler.track(node)
        cpu_scheduler.release(node)
        assert not node.is_loaded()

    def test_release_fires_event(self, cpu_scheduler):
        """T_SCHED_03: release 触发 model_unload 事件。"""
        events = []
        cpu_scheduler._bus.on(EventType.MODEL_UNLOAD, lambda e: events.append(e))
        node = _TestNode(name="n1")
        cpu_scheduler.ensure_loaded(node)
        cpu_scheduler.release(node)
        assert len(events) == 1
        assert events[0].type == EventType.MODEL_UNLOAD

    def test_release_all(self, cpu_scheduler):
        """T_SCHED_03: release_all 释放所有节点。"""
        n1 = _TestNode(name="n1")
        n2 = _TestNode(name="n2")
        cpu_scheduler.ensure_loaded(n1)
        cpu_scheduler.ensure_loaded(n2)
        assert cpu_scheduler.loaded_count() == 2
        count = cpu_scheduler.release_all()
        assert count == 2
        assert cpu_scheduler.loaded_count() == 0


# ===========================================================================
# T_SCHED_04: 显存不足时 LRU 淘汰
# ===========================================================================
class TestSchedulerLRU:
    """调度器 LRU 淘汰测试。"""

    def test_lru_eviction_on_memory_limit(self, gpu_scheduler):
        """T_SCHED_04: 显存不足时按 LRU 淘汰。"""
        # 每个节点 4GB，limit=10GB，只能容纳 2 个
        n1 = _TestNode(name="n1", vram_gb=4.0)
        n2 = _TestNode(name="n2", vram_gb=4.0)
        n3 = _TestNode(name="n3", vram_gb=4.0)

        gpu_scheduler.ensure_loaded(n1)  # 最新
        gpu_scheduler.ensure_loaded(n2)  # 最新
        # n1 是最久未使用的
        gpu_scheduler.ensure_loaded(n1)  # 刷新 n1 为最新
        # 现在 n2 是最久未使用的
        gpu_scheduler.ensure_loaded(n3)  # 需要淘汰 n2

        assert n1.is_loaded()
        assert not n2.is_loaded()  # 被淘汰
        assert n3.is_loaded()

    def test_lru_eviction_fires_events(self, gpu_scheduler):
        """T_SCHED_04: LRU 淘汰触发 model_unload 事件。"""
        events = []
        gpu_scheduler._bus.on(EventType.MODEL_UNLOAD, lambda e: events.append(e))

        n1 = _TestNode(name="n1", vram_gb=4.0)
        n2 = _TestNode(name="n2", vram_gb=4.0)
        n3 = _TestNode(name="n3", vram_gb=4.0)

        gpu_scheduler.ensure_loaded(n1)
        gpu_scheduler.ensure_loaded(n2)
        gpu_scheduler.ensure_loaded(n3)  # 应淘汰 n1

        # 至少有 1 个 model_unload 事件（淘汰）
        unload_events = [e for e in events if e.type == EventType.MODEL_UNLOAD]
        assert len(unload_events) >= 1

    def test_memory_error_when_cannot_evict(self, gpu_scheduler):
        """T_SCHED_04: 无法腾出空间时抛出 MemoryError。"""
        # 单个节点 12GB，超过 limit=10GB
        huge = _TestNode(name="huge", vram_gb=12.0)
        with pytest.raises(MemoryError, match="Cannot free enough GPU memory"):
            gpu_scheduler.ensure_loaded(huge)


# ===========================================================================
# T_SCHED_05: set_memory_limit 生效
# ===========================================================================
class TestSchedulerMemoryLimit:
    """调度器 set_memory_limit 测试。"""

    def test_set_memory_limit(self, gpu_scheduler):
        """T_SCHED_05: set_memory_limit 修改上限。"""
        gpu_scheduler.set_memory_limit(5.0)
        st = gpu_scheduler.status()
        assert st["memory_limit_gb"] == 5.0

    def test_set_memory_limit_zero_no_limit(self, gpu_scheduler):
        """T_SCHED_05: 设为 0 表示不限制。"""
        gpu_scheduler.set_memory_limit(0)
        st = gpu_scheduler.status()
        assert st["memory_limit_gb"] == gpu_scheduler._memory_total_gb

    def test_set_memory_limit_below_current_raises(self, gpu_scheduler):
        """T_SCHED_05: 上限低于当前占用抛出 ValueError。"""
        n1 = _TestNode(name="n1", vram_gb=4.0)
        n2 = _TestNode(name="n2", vram_gb=4.0)
        gpu_scheduler.ensure_loaded(n1)
        gpu_scheduler.ensure_loaded(n2)
        # 当前占用 8GB，设置 5GB 上限
        with pytest.raises(ValueError, match="smaller than currently loaded"):
            gpu_scheduler.set_memory_limit(5.0)


# ===========================================================================
# T_SCHED_06: 无 GPU 时优雅降级
# ===========================================================================
class TestSchedulerGracefulDegradation:
    """调度器优雅降级测试。"""

    def test_cpu_mode_no_memory_error(self, cpu_scheduler):
        """T_SCHED_06: CPU 模式不抛出 MemoryError。"""
        node = _TestNode(name="n1", vram_gb=100.0)  # 超大显存
        cpu_scheduler.ensure_loaded(node)  # CPU 模式不检查显存
        assert node.is_loaded()

    def test_cpu_mode_device_is_cpu(self, cpu_scheduler):
        """T_SCHED_06: CPU 模式 device 为 'cpu'。"""
        assert cpu_scheduler.device == "cpu"
        assert not cpu_scheduler.is_gpu

    def test_cpu_mode_memory_used_is_zero(self, cpu_scheduler):
        """T_SCHED_06: CPU 模式 memory_used_gb 为 0。"""
        node = _TestNode(name="n1")
        cpu_scheduler.ensure_loaded(node)
        st = cpu_scheduler.status()
        assert st["memory_used_gb"] == 0.0

    def test_global_scheduler_is_cpu(self):
        """T_SCHED_06: 全局调度器在无 GPU 环境为 CPU 模式。"""
        sched = get_scheduler()
        assert sched.device == "cpu"

    def test_set_scheduler_replaces_global(self):
        """T_SCHED_06: set_scheduler 替换全局调度器。"""
        old = get_scheduler()
        new = Scheduler(device="cpu")
        set_scheduler(new)
        assert get_scheduler() is new
        set_scheduler(old)  # 恢复