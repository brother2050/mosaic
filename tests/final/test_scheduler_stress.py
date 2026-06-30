# tests/final/test_scheduler_stress.py
"""Mosaic 调度器压力测试。

覆盖 Scheduler 的并发加载、LRU 淘汰、内存压力、死锁检测等场景。

测试 ID 约定：
    T_STRESS_01 ~ T_STRESS_06 分别对应不同的调度器压力测试项。
"""

from __future__ import annotations

import threading
import time

import pytest

from mosaic.core import EventBus, Scheduler, get_scheduler, set_scheduler
from mosaic.core.node import Node, NodeSpec
from mosaic.core.types import MosaicData


# ============================================================================
# 辅助节点类
# ============================================================================
class _StressTestNode(Node):
    """用于调度器压力测试的节点。"""

    name = "stress-test-node"
    domain = "test"
    description = "Stress test node for scheduler."
    version = "0.1.0"
    input_types = ["text"]
    output_types = ["text"]

    def __init__(self, name="stress-test-node", vram_gb=0.5, **kwargs):
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


# ============================================================================
# T_STRESS_01：5 个节点顺序加载，调度器正确管理
# ============================================================================
class TestSequentialLoading:
    """验证调度器顺序加载多个节点的行为。"""

    def test_load_five_nodes_sequentially(self):
        """T_STRESS_01: 顺序加载 5 个节点，调度器正确管理。"""
        # 创建全新调度器
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)

        # 创建 5 个 mock 节点
        nodes = [
            _StressTestNode(name=f"stress-node-{i}", vram_gb=0.5)
            for i in range(5)
        ]

        # 逐个 track 并 ensure_loaded
        for node in nodes:
            sched.track(node)
            sched.ensure_loaded(node)

        # 验证 loaded_count == 5
        assert sched.loaded_count() == 5, (
            f"Expected 5 loaded nodes, but got {sched.loaded_count()}. "
            f"Loaded: {sched.status()['loaded_nodes']}"
        )

        # 验证 status() 返回 dict 且包含预期键
        st = sched.status()
        assert isinstance(st, dict), "status() should return a dict."
        expected_keys = {"mode", "device", "memory_total_gb", "memory_used_gb",
                         "memory_limit_gb", "tracked_nodes", "loaded_nodes",
                         "node_memory"}
        for key in expected_keys:
            assert key in st, (
                f"status() dict missing expected key '{key}'. "
                f"Got keys: {list(st.keys())}"
            )

        # 验证 tracked_nodes 包含所有节点
        assert len(st["tracked_nodes"]) == 5, (
            f"Expected 5 tracked nodes, got {len(st['tracked_nodes'])}."
        )
        for i in range(5):
            assert f"stress-node-{i}" in st["tracked_nodes"], (
                f"stress-node-{i} should be in tracked_nodes."
            )

        # 清理
        sched.release_all()


# ============================================================================
# T_STRESS_02：内存压力下的 LRU 淘汰
# ============================================================================
class TestLRUEvictionOnMemoryPressure:
    """验证内存压力下 LRU 淘汰行为。"""

    def test_lru_eviction_on_small_memory_limit(self):
        """T_STRESS_02: 内存压力下 LRU 淘汰生效，加载数不超过上限。"""
        # 创建极小内存限制的 GPU 调度器 (0.001 GB ≈ 1 MB)
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cuda", memory_limit_gb=0.001)
        set_scheduler(sched)

        # 每个节点占用 0.0005 GB，极限下最多容纳 2 个
        node_vram = 0.0005
        nodes = [
            _StressTestNode(name=f"evict-node-{i}", vram_gb=node_vram)
            for i in range(6)
        ]

        # 逐个加载并检查 loaded_count
        loaded_counts = []
        for i, node in enumerate(nodes):
            sched.track(node)
            sched.ensure_loaded(node)
            count = sched.loaded_count()
            loaded_counts.append(count)

        # 验证加载数从未超过 2（0.0005 * 2 = 0.001 ≤ limit）
        max_loaded = max(loaded_counts)
        assert max_loaded <= 2, (
            f"Loaded count should never exceed 2 with memory limit 0.001 GB "
            f"and node vram 0.0005 GB, but max was {max_loaded}. "
            f"Counts per step: {loaded_counts}"
        )

        # 验证发生了 LRU 淘汰（最终加载数 < 总加载次数）
        # 因为加载了 6 个节点，但内存只能容纳 2 个，所以一定发生了淘汰
        eviction_occurred = any(
            loaded_counts[i] < i + 1 for i in range(2, len(loaded_counts))
        )
        assert eviction_occurred, (
            "LRU eviction should have occurred: at some point loaded_count "
            f"should be less than total load attempts. Counts: {loaded_counts}"
        )

        # 最终加载数应在 1-2 之间
        final_count = sched.loaded_count()
        assert 1 <= final_count <= 2, (
            f"Final loaded count should be 1 or 2, but got {final_count}."
        )

        # 清理
        sched.release_all()


# ============================================================================
# T_STRESS_03：并发加载/卸载无死锁
# ============================================================================
class TestConcurrentLoadUnload:
    """验证多线程并发加载/卸载无死锁。"""

    def test_concurrent_load_unload_no_deadlock(self):
        """T_STRESS_03: 3 线程并发加载/卸载，无死锁，全部在超时内完成。"""
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)

        # 创建一批节点
        nodes = [
            _StressTestNode(name=f"concurrent-node-{i}", vram_gb=0.5)
            for i in range(6)
        ]
        for node in nodes:
            sched.track(node)

        errors = []
        timeout = 15  # 秒

        def worker(worker_id, node_list):
            """每个线程反复加载/卸载节点。"""
            try:
                for _ in range(20):
                    for node in node_list:
                        sched.ensure_loaded(node)
                        time.sleep(0.001)
                        sched.release(node)
            except Exception as exc:  # noqa: BLE001
                errors.append((worker_id, str(exc)))

        # 创建 3 个线程，各负责不同的节点子集
        threads = []
        for tid in range(3):
            subset = nodes[tid * 2 : tid * 2 + 2]
            t = threading.Thread(
                target=worker, args=(tid, subset), daemon=True
            )
            threads.append(t)

        # 启动所有线程
        start = time.time()
        for t in threads:
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=timeout)

        elapsed = time.time() - start

        # 验证所有线程在超时内完成
        for idx, t in enumerate(threads):
            assert not t.is_alive(), (
                f"Thread {idx} did not complete within {timeout}s timeout. "
                f"Deadlock suspected."
            )

        # 验证没有异常
        assert not errors, (
            f"Errors occurred during concurrent load/unload: {errors}"
        )

        # 验证总耗时在合理范围内
        assert elapsed < timeout, (
            f"Concurrent test took {elapsed:.2f}s, expected < {timeout}s."
        )

        # 清理
        sched.release_all()


# ============================================================================
# T_STRESS_04：加载后释放，内存完全归还
# ============================================================================
class TestLoadThenUnload:
    """验证加载后释放的完整生命周期。"""

    def test_load_then_unload_memory_released(self):
        """T_STRESS_04: 加载后释放，内存完全归还。"""
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)

        nodes = [
            _StressTestNode(name=f"release-node-{i}", vram_gb=0.5)
            for i in range(4)
        ]

        for node in nodes:
            # 初始状态：loaded_count == 0
            assert sched.loaded_count() == 0, (
                f"Expected loaded_count=0 before loading {node.name}, "
                f"got {sched.loaded_count()}."
            )

            # 加载
            sched.track(node)
            sched.ensure_loaded(node)
            assert node.is_loaded(), (
                f"Node {node.name} should be loaded after ensure_loaded."
            )
            assert sched.loaded_count() == 1, (
                f"Expected loaded_count=1 after loading {node.name}, "
                f"got {sched.loaded_count()}."
            )

            # 释放
            sched.release(node)
            assert not node.is_loaded(), (
                f"Node {node.name} should NOT be loaded after release."
            )
            assert sched.loaded_count() == 0, (
                f"Expected loaded_count=0 after releasing {node.name}, "
                f"got {sched.loaded_count()}. "
                f"Loaded: {sched.status()['loaded_nodes']}"
            )

        # 最终状态
        st = sched.status()
        assert len(st["loaded_nodes"]) == 0, (
            "No nodes should be loaded after all releases."
        )
        assert len(st["tracked_nodes"]) == 4, (
            f"All 4 nodes should still be tracked, "
            f"got {len(st['tracked_nodes'])}."
        )

        # 清理
        sched.release_all()


# ============================================================================
# T_STRESS_05：调度器状态报告准确
# ============================================================================
class TestSchedulerStatusReport:
    """验证调度器状态报告的准确性。"""

    def test_status_report_is_accurate(self):
        """T_STRESS_05: status() 返回准确的状态信息。"""
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        sched = Scheduler(bus=bus, device="cpu")
        set_scheduler(sched)

        # 创建并 track 5 个节点
        nodes = [
            _StressTestNode(name=f"status-node-{i}", vram_gb=0.5)
            for i in range(5)
        ]
        for node in nodes:
            sched.track(node)

        # 加载前 3 个
        for node in nodes[:3]:
            sched.ensure_loaded(node)

        # 获取状态
        st = sched.status()

        # 验证 device 和 is_gpu 相关信息
        assert "device" in st, "status() should contain 'device' key."
        assert st["device"] == "cpu", (
            f"Expected device='cpu', got {st['device']}."
        )
        assert st["mode"] == "cpu", (
            f"Expected mode='cpu', got {st['mode']}."
        )

        # 验证 loaded_count 与 status 一致
        assert sched.loaded_count() == 3, (
            f"Expected 3 loaded nodes, got {sched.loaded_count()}."
        )
        assert len(st["loaded_nodes"]) == 3, (
            f"status() loaded_nodes should have 3 entries, "
            f"got {len(st['loaded_nodes'])}."
        )

        # 验证 tracked_nodes 有 5 个
        assert len(st["tracked_nodes"]) == 5, (
            f"Expected 5 tracked nodes, got {len(st['tracked_nodes'])}."
        )

        # 验证 memory_limit_gb 存在
        assert "memory_limit_gb" in st, (
            "status() should contain 'memory_limit_gb' key."
        )
        assert "memory_used_gb" in st, (
            "status() should contain 'memory_used_gb' key."
        )
        assert "memory_total_gb" in st, (
            "status() should contain 'memory_total_gb' key."
        )

        # 验证 node_memory 包含所有 tracked 节点
        assert len(st["node_memory"]) == 5, (
            f"node_memory should have 5 entries, got {len(st['node_memory'])}."
        )
        for i in range(5):
            assert f"status-node-{i}" in st["node_memory"], (
                f"status-node-{i} should be in node_memory."
            )

        # 清理
        sched.release_all()


# ============================================================================
# T_STRESS_06：多 TTS 后端 LRU 淘汰顺序正确
# ============================================================================
class TestTTSBackendLRUOrder:
    """验证多个 TTS 后端的 LRU 淘汰顺序。"""

    def test_tts_backend_lru_eviction_order(self):
        """T_STRESS_06: 4 个 TTS 后端 LRU 淘汰顺序正确。"""
        EventBus._reset_singleton()
        bus = EventBus()
        bus.clear()
        # 使用 GPU 模式，内存限制为 4 GB
        sched = Scheduler(bus=bus, device="cuda", memory_limit_gb=4.0)
        set_scheduler(sched)

        # 模拟 4 个 TTS 后端，每个占用 2 GB
        backends = [
            _StressTestNode(name="chattts", vram_gb=2.0),
            _StressTestNode(name="fish", vram_gb=2.0),
            _StressTestNode(name="sovits", vram_gb=2.0),
            _StressTestNode(name="cosyvoice", vram_gb=2.0),
        ]

        # 按顺序加载，内存只能容纳 2 个（2.0 * 2 = 4.0 ≤ 4.0）
        for backend in backends:
            sched.track(backend)
            sched.ensure_loaded(backend)

        # 最终只能加载 2 个节点
        assert sched.loaded_count() <= 2, (
            f"Expected at most 2 loaded nodes with 4 GB limit, "
            f"got {sched.loaded_count()}."
        )

        # 验证 LRU 顺序：后加载的节点应该保留，先加载的被淘汰
        loaded = sched.status()["loaded_nodes"]
        # chattts 和 fish 先加载，应该被淘汰
        # sovits 和 cosyvoice 后加载，应该保留
        for early in ["chattts", "fish"]:
            assert early not in loaded, (
                f"'{early}' should have been evicted (loaded first), "
                f"but it is still loaded. Loaded: {loaded}"
            )
        for late in ["sovits", "cosyvoice"]:
            assert late in loaded, (
                f"'{late}' should still be loaded (loaded last), "
                f"but it was evicted. Loaded: {loaded}"
            )

        # 验证被淘汰的节点确实已卸载
        assert not backends[0].is_loaded(), (
            "chattts should be unloaded after eviction."
        )
        assert not backends[1].is_loaded(), (
            "fish should be unloaded after eviction."
        )
        assert backends[2].is_loaded(), (
            "sovits should still be loaded."
        )
        assert backends[3].is_loaded(), (
            "cosyvoice should still be loaded."
        )

        # 清理
        sched.release_all()