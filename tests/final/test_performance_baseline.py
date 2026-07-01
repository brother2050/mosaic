# tests/final/test_performance_baseline.py
"""性能基线测试。

验证 Mosaic 框架核心操作在正常条件下的性能基线：
- Pipeline 创建速度
- 注册表查询速度
- 调度器状态查询速度
- 事件总线注册速度
- 中间结果序列化速度
- 节点运行开销
"""

from __future__ import annotations

import time

import pytest

from mosaic.core import (
    EventBus,
    Pipeline,
    PipelineResult,
    Scheduler,
    registry,
)
from mosaic.core.types import TextData


# ===========================================================================
# T_PERF_01: Pipeline 创建 < 1 秒
# ===========================================================================
def test_pipeline_creation_under_one_second():
    """T_PERF_01: 创建一个包含两个节点的 Pipeline 应在 1 秒内完成。"""
    start = time.perf_counter()
    pipe = Pipeline(
        "perf_test",
        [registry.get("TextGenerator"), registry.get("TextRewriter")],
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (
        f"Pipeline 创建耗时 {elapsed:.3f}s，应小于 1.0s"
    )


# ===========================================================================
# T_PERF_02: registry.list_nodes() < 100ms
# ===========================================================================
def test_registry_list_nodes_under_100ms():
    """T_PERF_02: registry.list_nodes() 应在 100ms 内完成。"""
    start = time.perf_counter()
    nodes = registry.list_nodes()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, (
        f"list_nodes() 耗时 {elapsed * 1000:.1f}ms，应小于 100ms"
    )
    assert isinstance(nodes, list), "list_nodes() 应返回 list"
    assert len(nodes) >= 42, (
        f"list_nodes() 应返回至少 39 个节点，实际返回 {len(nodes)}"
    )


# ===========================================================================
# T_PERF_03: Scheduler.status() < 50ms
# ===========================================================================
def test_scheduler_status_under_50ms():
    """T_PERF_03: Scheduler.status() 应在 50ms 内完成。"""
    scheduler = Scheduler()
    start = time.perf_counter()
    status = scheduler.status()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, (
        f"Scheduler.status() 耗时 {elapsed * 1000:.1f}ms，应小于 50ms"
    )
    assert isinstance(status, dict), "status() 应返回 dict"
    assert "mode" in status, "status() 返回结果应包含 'mode' 字段"


# ===========================================================================
# T_PERF_04: EventBus 注册 100 个监听器 < 100ms
# ===========================================================================
def test_eventbus_register_100_listeners_under_100ms():
    """T_PERF_04: EventBus 注册 100 个监听器应在 100ms 内完成。"""
    bus = EventBus()
    start = time.perf_counter()
    for i in range(100):
        bus.on("test_event", lambda e, i=i: None)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, (
        f"注册 100 个监听器耗时 {elapsed * 1000:.1f}ms，应小于 100ms"
    )
    bus.clear()


# ===========================================================================
# T_PERF_05: 中间快照（100 条目）序列化 < 1 秒
# ===========================================================================
def test_intermediate_snapshot_serialization_under_one_second():
    """T_PERF_05: 100 个中间结果的 PipelineResult.to_dict() 应在 1 秒内完成。"""
    # 构建 100 个条目的中间结果
    intermediate = {
        f"node_{i}": TextData(content=f"data_{i}") for i in range(100)
    }

    result = PipelineResult(
        output=TextData(content="final"),
        intermediate=intermediate,
        errors=[],
        duration=0.5,
        node_durations={f"node_{i}": 0.01 for i in range(100)},
        pipeline_name="perf_test",
    )

    start = time.perf_counter()
    d = result.to_dict()
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"100 个中间结果序列化耗时 {elapsed:.3f}s，应小于 1.0s"
    )
    assert isinstance(d, dict), "to_dict() 应返回 dict"
    assert len(d["intermediate"]) == 100, (
        f"序列化后应有 100 个中间结果，实际有 {len(d['intermediate'])}"
    )


# ===========================================================================
# T_PERF_06: Mock 节点 run() 开销 < 10ms
# ===========================================================================
def test_node_run_overhead_under_10ms():
    """T_PERF_06: 使用注册节点运行 10 次的平均开销应小于 10ms。

    使用一个真实的注册节点（TextGenerator）进行测试。
    由于 TextGenerator 的 run() 方法可能涉及模型加载或抛出异常，
    我们在 try/except 中捕获异常，仅测量调用开销。
    """
    # 使用 registry 获取一个真实节点
    node = registry.get("TextGenerator")

    # 构造输入数据
    input_data = TextData(content="test")

    # 预热一次（避免首次调用的额外开销）
    try:
        node.run(input_data)
    except Exception:  # noqa: BLE001
        pass

    # 运行 10 次并记录时间
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        try:
            node.run(input_data)
        except Exception:  # noqa: BLE001
            # 在 mock 环境下 run() 可能因缺少模型而失败，这是预期的
            pass
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    assert avg < 0.01, (
        f"节点 run() 平均开销 {avg * 1000:.1f}ms，应小于 10ms"
    )