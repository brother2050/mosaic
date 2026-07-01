# tests/phase4/test_task_manager.py
"""TaskManager 任务管理器测试。

测试 TaskManager 的任务提交、查询、过滤、取消、清理和统计功能。
使用合成 Pipeline + Mock 节点，不依赖真实模型。
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
from mosaic.core.task import AsyncTask, TaskCancelledError, TaskStatus
from mosaic.core.task_manager import TaskManager
from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# Mock 节点定义
# ---------------------------------------------------------------------------
class MockFastNode(Node):
    """快速执行的 mock 节点。"""

    name = "mock-fast"
    domain = "test"
    description = "A fast mock node for TM testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        return MosaicData(
            prompt="mock output",
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


class MockSlowNode(Node):
    """慢速执行的 mock 节点。"""

    name = "mock-slow"
    domain = "test"
    description = "A slow mock node for TM testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        time.sleep(3.0)
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
# TestTaskManager
# ---------------------------------------------------------------------------
class TestTaskManager:
    """TaskManager 核心功能测试。"""

    # T_TM_01
    def test_submit_returns_async_task(self, fresh_bus):
        """T_TM_01：submit() 返回 AsyncTask。"""
        tm = TaskManager(bus=fresh_bus)
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        task = tm.submit(pipe, input_data)

        assert isinstance(task, AsyncTask), "submit() 应返回 AsyncTask"
        assert task.task_id is not None, "task_id 不应为 None"
        assert task.pipeline_name == "test-pipe", "pipeline_name 应正确"

        # 清理
        task.wait(timeout=10)

    # T_TM_02
    def test_get_retrieves_correct_task(self, fresh_bus):
        """T_TM_02：get(task_id) 检索到正确的任务。"""
        tm = TaskManager(bus=fresh_bus)
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="test")

        task = tm.submit(pipe, input_data)
        task_id = task.task_id

        retrieved = tm.get(task_id)
        assert retrieved is not None, "get() 应返回任务"
        assert retrieved.task_id == task_id, "检索到的 task_id 应匹配"
        assert retrieved is task, "应为同一任务实例"

        # 清理
        task.wait(timeout=10)

    # T_TM_03
    def test_list_tasks_returns_all(self, fresh_bus):
        """T_TM_03：list_tasks() 返回所有任务。"""
        tm = TaskManager(bus=fresh_bus)

        pipe1 = _make_pipeline(name="pipe-1")
        pipe2 = _make_pipeline(name="pipe-2")
        pipe3 = _make_pipeline(name="pipe-3")

        task1 = tm.submit(pipe1, MosaicData(prompt="a"))
        task2 = tm.submit(pipe2, MosaicData(prompt="b"))
        task3 = tm.submit(pipe3, MosaicData(prompt="c"))

        # 等待所有任务完成
        for t in [task1, task2, task3]:
            t.wait(timeout=10)

        all_tasks = tm.list_tasks()
        assert len(all_tasks) == 3, "list_tasks() 应返回 3 个任务"

        task_ids = {t.task_id for t in all_tasks}
        assert task1.task_id in task_ids, "task1 应在列表中"
        assert task2.task_id in task_ids, "task2 应在列表中"
        assert task3.task_id in task_ids, "task3 应在列表中"

    # T_TM_04
    def test_list_tasks_filter_by_status(self, fresh_bus):
        """T_TM_04：list_tasks(status="running") 正确过滤。"""
        tm = TaskManager(bus=fresh_bus)

        # 提交一个慢任务（保持 running 状态）
        slow_pipe = _make_pipeline(name="slow-pipe", node=MockSlowNode())
        running_task = tm.submit(slow_pipe, MosaicData(prompt="slow"))

        # 提交一个快任务（很快完成）
        fast_pipe = _make_pipeline(name="fast-pipe")
        completed_task = tm.submit(fast_pipe, MosaicData(prompt="fast"))
        completed_task.wait(timeout=10)

        # 过滤 running 状态
        running_tasks = tm.list_tasks(status="running")
        assert len(running_tasks) >= 1, "至少应有 1 个 running 任务"

        running_ids = {t.task_id for t in running_tasks}
        assert running_task.task_id in running_ids, "慢任务应在 running 列表中"

        # 过滤 completed 状态
        completed_tasks = tm.list_tasks(status="completed")
        completed_ids = {t.task_id for t in completed_tasks}
        assert completed_task.task_id in completed_ids, "快任务应在 completed 列表中"

        # 清理
        running_task.cancel()

    # T_TM_05
    def test_cancel_specific_task(self, fresh_bus):
        """T_TM_05：cancel(task_id) 取消指定任务。"""
        tm = TaskManager(bus=fresh_bus)

        slow_pipe = _make_pipeline(name="slow-pipe", node=MockSlowNode())
        task = tm.submit(slow_pipe, MosaicData(prompt="test"))

        time.sleep(0.05)
        result = tm.cancel(task.task_id)

        assert result is True, "cancel() 应返回 True"
        assert task.is_cancelled, "任务应被取消"

    # T_TM_06
    def test_cancel_all_returns_count(self, fresh_bus):
        """T_TM_06：cancel_all() 取消所有 running 任务，返回数量。"""
        tm = TaskManager(bus=fresh_bus)
        slow_node = MockSlowNode()

        pipe1 = _make_pipeline(name="slow-1", node=slow_node)
        pipe2 = _make_pipeline(name="slow-2", node=slow_node)
        pipe3 = _make_pipeline(name="slow-3", node=slow_node)

        task1 = tm.submit(pipe1, MosaicData(prompt="a"))
        task2 = tm.submit(pipe2, MosaicData(prompt="b"))
        task3 = tm.submit(pipe3, MosaicData(prompt="c"))

        time.sleep(0.05)
        count = tm.cancel_all()

        assert count >= 0, "cancel_all() 应返回非负整数"
        # 任务应该被取消（至少部分）
        assert task1.is_cancelled or task2.is_cancelled or task3.is_cancelled, (
            "至少有一个任务应被取消"
        )

    # T_TM_07
    def test_cleanup_removes_completed_tasks(self, fresh_bus):
        """T_TM_07：cleanup(max_age=0.001) 移除已完成任务。"""
        tm = TaskManager(bus=fresh_bus)

        pipe = _make_pipeline()
        task = tm.submit(pipe, MosaicData(prompt="test"))
        task.wait(timeout=10)

        # 任务已完成
        assert task.status == TaskStatus.COMPLETED, "任务应已完成"

        # 使用极小的 max_age 确保清理
        time.sleep(0.01)
        removed = tm.cleanup(max_age=0.001)

        assert removed >= 1, "cleanup() 应至少清理 1 个任务"

        # 清理后任务应不可检索
        retrieved = tm.get(task.task_id)
        assert retrieved is None, "清理后 get() 应返回 None"

    # T_TM_08
    def test_status_summary_counts(self, fresh_bus):
        """T_TM_08：status_summary() 返回正确的计数字典。"""
        tm = TaskManager(bus=fresh_bus)

        # 提交并等待完成一些任务
        pipe = _make_pipeline()
        for i in range(3):
            task = tm.submit(pipe, MosaicData(prompt=f"test-{i}"))
            task.wait(timeout=10)

        summary = tm.status_summary()

        assert isinstance(summary, dict), "status_summary() 应返回字典"
        required_keys = ["pending", "running", "completed", "failed", "cancelled", "total"]
        for key in required_keys:
            assert key in summary, f"summary 应包含键 '{key}'"

        assert summary["completed"] == 3, "completed 应为 3"
        assert summary["total"] == 3, "total 应为 3"
        assert sum(summary[k] for k in ["pending", "running", "completed", "failed", "cancelled"]) == summary["total"], (
            "各状态计数之和应等于 total"
        )


# ---------------------------------------------------------------------------
# TestTaskManagerEdgeCases
# ---------------------------------------------------------------------------
class TestTaskManagerEdgeCases:
    """TaskManager 边界条件测试。"""

    def test_get_nonexistent_task(self, fresh_bus):
        """get() 不存在的任务返回 None。"""
        tm = TaskManager(bus=fresh_bus)
        result = tm.get("nonexistent-id")
        assert result is None, "不存在的任务应返回 None"

    def test_cancel_nonexistent_task(self, fresh_bus):
        """cancel() 不存在的任务返回 False。"""
        tm = TaskManager(bus=fresh_bus)
        result = tm.cancel("nonexistent-id")
        assert result is False, "不存在的任务 cancel() 应返回 False"

    def test_list_tasks_empty(self, fresh_bus):
        """空管理器的 list_tasks() 返回空列表。"""
        tm = TaskManager(bus=fresh_bus)
        tasks = tm.list_tasks()
        assert tasks == [], "空管理器应返回空列表"

    def test_status_summary_empty(self, fresh_bus):
        """空管理器的 status_summary() 各计数为 0。"""
        tm = TaskManager(bus=fresh_bus)
        summary = tm.status_summary()

        assert summary["total"] == 0, "total 应为 0"
        for status in ["pending", "running", "completed", "failed", "cancelled"]:
            assert summary[status] == 0, f"{status} 应为 0"

    def test_cleanup_empty_manager(self, fresh_bus):
        """空管理器的 cleanup() 返回 0。"""
        tm = TaskManager(bus=fresh_bus)
        removed = tm.cleanup(max_age=0.001)
        assert removed == 0, "空管理器 cleanup() 应返回 0"

    def test_submit_multiple_same_pipeline(self, fresh_bus):
        """同一管道多次提交返回不同 AsyncTask。"""
        tm = TaskManager(bus=fresh_bus)
        pipe = _make_pipeline()

        task1 = tm.submit(pipe, MosaicData(prompt="a"))
        task2 = tm.submit(pipe, MosaicData(prompt="b"))

        assert task1.task_id != task2.task_id, "不同提交应有不同 task_id"

        task1.wait(timeout=10)
        task2.wait(timeout=10)