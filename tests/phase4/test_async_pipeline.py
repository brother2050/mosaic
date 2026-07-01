# tests/phase4/test_async_pipeline.py
"""Pipeline.run_async() 异步管道执行测试。

测试 Pipeline 的异步执行方法：返回 AsyncTask、获取结果、
并行执行、EventBus 事件转发等。
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import PipelineResult
from mosaic.core.task import AsyncTask, TaskStatus
from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# Mock 节点定义
# ---------------------------------------------------------------------------
class MockFastNode(Node):
    """快速执行的 mock 节点。"""

    name = "mock-fast"
    domain = "test"
    description = "A fast mock node for async pipeline testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        # 通过 EventBus 发出 NODE_START 和 NODE_COMPLETE 事件
        bus = get_event_bus()
        bus.emit(EventType.NODE_START, node_name=self.name)
        result = MosaicData(
            prompt="mock output",
            input_tokens=10,
            output_tokens=20,
        )
        bus.emit(EventType.NODE_COMPLETE, node_name=self.name)
        return result

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockSlowNode(Node):
    """慢速执行的 mock 节点。"""

    name = "mock-slow"
    domain = "test"
    description = "A slow mock node for async pipeline testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        time.sleep(0.5)
        return MosaicData(
            prompt="slow output",
            input_tokens=10,
            output_tokens=20,
        )

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_pipeline(name: str = "test-pipe", node: Node | None = None) -> Pipeline:
    """创建一条单节点管道。"""
    if node is None:
        node = MockFastNode()
    return Pipeline(name=name, elements=[node])


# ---------------------------------------------------------------------------
# TestAsyncPipeline
# ---------------------------------------------------------------------------
class TestAsyncPipeline:
    """Pipeline.run_async() 测试。"""

    # T_APIPE_01
    def test_run_async_returns_async_task(self, fresh_bus):
        """T_APIPE_01：run_async() 返回 AsyncTask 实例。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        task = pipe.run_async(input_data)

        assert isinstance(task, AsyncTask), "run_async() 应返回 AsyncTask"
        assert task.pipeline_name == "test-pipe", "pipeline_name 应正确"
        assert task.task_id is not None, "task_id 不应为 None"

        # 清理
        task.wait(timeout=10)

    # T_APIPE_02
    def test_async_execution_yields_result(self, fresh_bus):
        """T_APIPE_02：异步执行最终产出 PipelineResult。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        task = pipe.run_async(input_data)
        result = task.wait(timeout=10)

        assert isinstance(result, PipelineResult), "wait() 应返回 PipelineResult"
        assert result.success, "管道应成功执行"
        assert result.pipeline_name == "test-pipe", "pipeline_name 应正确"
        assert result.output is not None, "output 不应为 None"

    # T_APIPE_03
    def test_multiple_async_tasks_parallel(self, fresh_bus):
        """T_APIPE_03：多个异步任务可并行运行。"""
        slow_node = MockSlowNode()
        pipe = _make_pipeline(name="parallel-pipe", node=slow_node)

        input_1 = MosaicData(prompt="first")
        input_2 = MosaicData(prompt="second")
        input_3 = MosaicData(prompt="third")

        # 提交三个异步任务
        task1 = pipe.run_async(input_1)
        task2 = pipe.run_async(input_2)
        task3 = pipe.run_async(input_3)

        # 验证三个任务有不同 task_id
        assert task1.task_id != task2.task_id, "task1 和 task2 应有不同 ID"
        assert task2.task_id != task3.task_id, "task2 和 task3 应有不同 ID"
        assert task1.task_id != task3.task_id, "task1 和 task3 应有不同 ID"

        # 并行等待所有任务完成（总时间应接近单个任务时间，而非三倍）
        t0 = time.time()
        result1 = task1.wait(timeout=10)
        result2 = task2.wait(timeout=10)
        result3 = task3.wait(timeout=10)
        elapsed = time.time() - t0

        assert result1.success, "task1 应成功"
        assert result2.success, "task2 应成功"
        assert result3.success, "task3 应成功"

        # 并行执行的总时间应小于三个任务串行时间之和
        assert elapsed < 2.5, "并行执行时间应显著小于串行执行时间"

    # T_APIPE_04
    def test_eventbus_events_forwarded(self, fresh_bus):
        """T_APIPE_04：EventBus 事件在异步执行期间被转发。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        # 收集事件
        events = []
        fresh_bus.on(EventType.NODE_START, lambda e: events.append(("start", e)))
        fresh_bus.on(EventType.NODE_COMPLETE, lambda e: events.append(("complete", e)))

        task = pipe.run_async(input_data)
        task.wait(timeout=10)

        start_events = [e for e_type, e in events if e_type == "start"]
        complete_events = [e for e_type, e in events if e_type == "complete"]

        assert len(start_events) > 0, "NODE_START 事件应被转发"
        assert len(complete_events) > 0, "NODE_COMPLETE 事件应被转发"


# ---------------------------------------------------------------------------
# TestAsyncPipelineEdgeCases
# ---------------------------------------------------------------------------
class TestAsyncPipelineEdgeCases:
    """Pipeline.run_async() 边界条件测试。"""

    def test_async_with_empty_pipeline(self, fresh_bus):
        """空管道异步执行。"""
        pipe = Pipeline(name="empty-pipe", elements=[])
        input_data = MosaicData(prompt="test")

        task = pipe.run_async(input_data)
        result = task.wait(timeout=10)

        assert isinstance(result, PipelineResult), "空管道也应返回 PipelineResult"
        assert result.success, "空管道应成功执行"

    def test_async_status_transitions(self, fresh_bus):
        """异步任务状态从 pending 到 completed 的转换。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        # 创建任务但不启动，检查 pending 状态
        task = AsyncTask(
            pipeline_name=pipe.name,
            pipeline=pipe,
            input_data=input_data,
            bus=fresh_bus,
        )

        assert task.status == TaskStatus.PENDING, "初始状态应为 pending"

        task._start()
        task.wait(timeout=10)
        assert task.status == TaskStatus.COMPLETED, "完成后状态应为 completed"

    def test_async_task_repr(self, fresh_bus):
        """AsyncTask 的 repr 包含关键信息。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        task = pipe.run_async(input_data)
        repr_str = repr(task)

        assert "AsyncTask" in repr_str, "repr 应包含类型名"
        task.wait(timeout=10)