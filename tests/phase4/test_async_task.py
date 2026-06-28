# tests/phase4/test_async_task.py
"""AsyncTask 异步任务测试。

测试 AsyncTask 的完整生命周期：状态转换、阻塞等待、超时、取消、
回调机制、序列化等。使用合成 Pipeline + Mock 节点，不依赖真实模型。
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
from mosaic.core.task import AsyncTask, TaskCancelledError, TaskStatus
from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# Mock 节点定义
# ---------------------------------------------------------------------------
class MockFastNode(Node):
    """快速执行的 mock 节点（用于大多数测试）。"""

    name = "mock-fast"
    domain = "test"
    description = "A fast mock node for async testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        return MosaicData(
            generated_text="mock output",
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
    """慢速执行的 mock 节点（用于超时测试）。"""

    name = "mock-slow"
    domain = "test"
    description = "A slow mock node for timeout testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        time.sleep(5.0)  # 慢速执行
        return MosaicData(
            generated_text="slow output",
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


class MockFailingNode(Node):
    """会失败的 mock 节点（用于错误回调测试）。"""

    name = "mock-failing"
    domain = "test"
    description = "A failing mock node for error testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        raise ValueError("Simulated node failure.")

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
    """创建一条单节点管道的辅助函数。"""
    if node is None:
        node = MockFastNode()
    return Pipeline(name=name, elements=[node])


def _create_async_task(
    pipe: Pipeline,
    input_data: MosaicData | None = None,
    bus: EventBus | None = None,
) -> AsyncTask:
    """创建并启动 AsyncTask 的辅助函数。"""
    if input_data is None:
        input_data = MosaicData(prompt="test prompt")
    task = AsyncTask(
        pipeline_name=pipe.name,
        pipeline=pipe,
        input_data=input_data,
        bus=bus or get_event_bus(),
    )
    task._start()
    return task


# ---------------------------------------------------------------------------
# TestAsyncTask
# ---------------------------------------------------------------------------
class TestAsyncTask:
    """AsyncTask 核心功能测试。"""

    # T_ASYNC_01
    def test_initial_status(self, fresh_bus):
        """T_ASYNC_01：创建 AsyncTask，初始状态为 pending，progress=0.0，current_node=None。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="hello")

        task = AsyncTask(
            pipeline_name=pipe.name,
            pipeline=pipe,
            input_data=input_data,
            bus=fresh_bus,
        )

        assert task.status == TaskStatus.PENDING, "初始状态应为 pending"
        assert task.progress == 0.0, "初始 progress 应为 0.0"
        assert task.current_node is None, "初始 current_node 应为 None"
        assert task.pipeline_name == "test-pipe", "pipeline_name 应正确"
        assert task.task_id is not None, "task_id 不应为 None"
        assert isinstance(task.task_id, str), "task_id 应为字符串"
        assert len(task.task_id) > 0, "task_id 不应为空"

    # T_ASYNC_02
    def test_status_running_then_completed(self, fresh_bus):
        """T_ASYNC_02：启动后，状态先变为 running 然后变为 completed。"""
        pipe = _make_pipeline()
        task = _create_async_task(pipe, bus=fresh_bus)

        # 等待任务完成
        task.wait(timeout=10)

        assert task.status == TaskStatus.COMPLETED, "完成后状态应为 completed"
        assert task.is_ready(), "is_ready() 应为 True"

    # T_ASYNC_03
    def test_wait_returns_pipeline_result(self, fresh_bus):
        """T_ASYNC_03：wait() 阻塞返回 PipelineResult。"""
        pipe = _make_pipeline()
        task = _create_async_task(pipe, bus=fresh_bus)

        result = task.wait(timeout=10)

        assert isinstance(result, PipelineResult), "wait() 应返回 PipelineResult"
        assert result.success, "管道应成功执行"
        assert result.pipeline_name == "test-pipe", "pipeline_name 应正确"

    # T_ASYNC_04
    def test_wait_timeout_raises(self, fresh_bus):
        """T_ASYNC_04：wait(timeout=0.1) 在长任务上抛出 TimeoutError。"""
        slow_node = MockSlowNode()
        pipe = _make_pipeline(name="slow-pipe", node=slow_node)
        task = _create_async_task(pipe, bus=fresh_bus)

        with pytest.raises(TimeoutError):
            task.wait(timeout=0.1)

    # T_ASYNC_05
    def test_cancel_sets_flag(self, fresh_bus):
        """T_ASYNC_05：cancel() 设置取消标志，is_cancelled 变为 True。"""
        slow_node = MockSlowNode()
        pipe = _make_pipeline(name="cancel-pipe", node=slow_node)
        task = _create_async_task(pipe, bus=fresh_bus)

        # 等待进入 running 状态
        time.sleep(0.05)
        assert task.cancel(), "cancel() 应返回 True"

        assert task.is_cancelled, "is_cancelled 应为 True"

    # T_ASYNC_06
    def test_on_complete_callback(self, fresh_bus):
        """T_ASYNC_06：on_complete 回调被触发，参数为 PipelineResult。"""
        pipe = _make_pipeline()
        task = _create_async_task(pipe, bus=fresh_bus)

        callback_results = []

        def _on_complete(result: PipelineResult) -> None:
            callback_results.append(result)

        task.on_complete(_on_complete)
        task.wait(timeout=10)

        assert len(callback_results) == 1, "完成回调应被触发一次"
        assert isinstance(callback_results[0], PipelineResult), "回调参数应为 PipelineResult"

    # T_ASYNC_07
    def test_on_error_callback(self, fresh_bus):
        """T_ASYNC_07：on_error 回调在管道失败时被触发。"""
        failing_node = MockFailingNode()
        pipe = _make_pipeline(name="fail-pipe", node=failing_node)
        task = _create_async_task(pipe, bus=fresh_bus)

        callback_errors = []

        def _on_error(error: BaseException) -> None:
            callback_errors.append(error)

        task.on_error(_on_error)

        # wait() 对失败任务会抛出 RuntimeError
        with pytest.raises(RuntimeError):
            task.wait(timeout=10)

        assert len(callback_errors) == 1, "错误回调应被触发一次"
        assert isinstance(callback_errors[0], Exception), "回调参数应为 Exception"

    # T_ASYNC_08
    def test_on_progress_callback(self, fresh_bus):
        """T_ASYNC_08：on_progress 回调接收进度更新 (float, str)。"""
        # 使用一个会在 run() 中通过 EventBus 发出 PROGRESS 事件的节点
        class ProgressNode(Node):
            name = "mock-progress"
            domain = "test"
            description = "Node that emits progress."
            version = "0.1.0"
            input_types = ["text", "mosaic"]
            output_types = ["text", "mosaic"]

            def load(self) -> None:
                self._loaded = True

            def unload(self) -> None:
                self._loaded = False

            def run(self, input_data: MosaicData) -> MosaicData:
                # 在节点执行期间发出 PROGRESS 事件
                bus = get_event_bus()
                bus.emit(
                    EventType.PROGRESS,
                    node_name="mock-progress",
                    current=3,
                    total=10,
                )
                return MosaicData(
                    generated_text="progress output",
                    input_tokens=10,
                    output_tokens=20,
                )

            def describe(self) -> NodeSpec:
                return NodeSpec(
                    name=self.name, domain=self.domain,
                    description=self.description, version=self.version,
                    input_types=list(self.input_types),
                    output_types=list(self.output_types),
                )

        pipe = Pipeline(name="progress-pipe", elements=[ProgressNode()])

        # 先创建任务（不启动），注册回调后再启动
        input_data = MosaicData(prompt="test")
        task = AsyncTask(
            pipeline_name=pipe.name,
            pipeline=pipe,
            input_data=input_data,
            bus=fresh_bus,
        )

        progress_updates = []

        def _on_progress(progress: float, node_name: str) -> None:
            progress_updates.append((progress, node_name))

        task.on_progress(_on_progress)

        # 启动任务（回调已注册，ProgressNode 发出的 PROGRESS 事件会被捕获）
        task._start()
        task.wait(timeout=10)

        # 进度回调应被触发
        assert len(progress_updates) > 0, "进度回调应至少被触发一次"
        for progress, node_name in progress_updates:
            assert isinstance(progress, float), "progress 应为 float"
            assert 0.0 <= progress <= 1.0, "progress 应在 0.0~1.0 之间"
            assert isinstance(node_name, str), "node_name 应为 str"

    # T_ASYNC_09
    def test_is_ready_reflects_status(self, fresh_bus):
        """T_ASYNC_09：is_ready() 正确反映状态。"""
        pipe = _make_pipeline()
        input_data = MosaicData(prompt="hello")

        # 创建任务但不启动，检查 pending 状态
        task = AsyncTask(
            pipeline_name=pipe.name,
            pipeline=pipe,
            input_data=input_data,
            bus=fresh_bus,
        )

        assert not task.is_ready(), "pending 状态 is_ready() 应为 False"

        # 启动并等待完成
        task._start()
        task.wait(timeout=10)
        assert task.is_ready(), "completed 状态 is_ready() 应为 True"

    # T_ASYNC_10
    def test_to_dict_serialization(self, fresh_bus):
        """T_ASYNC_10：to_dict() 序列化正确（所有键存在，值匹配）。"""
        pipe = _make_pipeline()
        task = _create_async_task(pipe, bus=fresh_bus)
        task.wait(timeout=10)

        d = task.to_dict()

        required_keys = [
            "task_id",
            "status",
            "progress",
            "current_node",
            "pipeline_name",
            "created_at",
            "started_at",
            "completed_at",
            "duration",
            "is_cancelled",
            "error",
        ]
        for key in required_keys:
            assert key in d, f"to_dict() 应包含键 '{key}'"

        assert d["task_id"] == task.task_id, "task_id 应匹配"
        assert d["status"] == TaskStatus.COMPLETED, "status 应为 completed"
        assert d["progress"] == 1.0, "progress 应为 1.0"
        assert d["pipeline_name"] == "test-pipe", "pipeline_name 应匹配"
        assert d["is_cancelled"] is False, "is_cancelled 应为 False"
        assert d["error"] is None, "error 应为 None"
        assert isinstance(d["created_at"], float), "created_at 应为 float"
        assert d["started_at"] is not None, "started_at 不应为 None"
        assert d["completed_at"] is not None, "completed_at 不应为 None"
        assert d["duration"] is not None, "duration 不应为 None"
        assert d["duration"] >= 0, "duration 应 >= 0"


# ---------------------------------------------------------------------------
# TestAsyncTaskEdgeCases
# ---------------------------------------------------------------------------
class TestAsyncTaskEdgeCases:
    """AsyncTask 边界条件测试。"""

    def test_task_id_auto_generated(self, fresh_bus):
        """不提供 task_id 时自动生成 UUID。"""
        pipe = _make_pipeline()
        task = AsyncTask(
            pipeline_name="test",
            pipeline=pipe,
            input_data=MosaicData(prompt="test"),
            bus=fresh_bus,
        )
        assert task.task_id, "task_id 应自动生成"
        assert len(task.task_id) >= 32, "task_id 应为 UUID 格式"

    def test_task_id_custom(self, fresh_bus):
        """提供自定义 task_id。"""
        pipe = _make_pipeline()
        custom_id = "my-custom-task-id"
        task = AsyncTask(
            pipeline_name="test",
            pipeline=pipe,
            input_data=MosaicData(prompt="test"),
            task_id=custom_id,
            bus=fresh_bus,
        )
        assert task.task_id == custom_id, "task_id 应为自定义值"

    def test_cancel_completed_task_returns_false(self, fresh_bus):
        """对已完成任务调用 cancel() 返回 False。"""
        pipe = _make_pipeline()
        task = _create_async_task(pipe, bus=fresh_bus)
        task.wait(timeout=10)

        assert task.cancel() is False, "已完成任务 cancel() 应返回 False"

    def test_created_at_is_timestamp(self, fresh_bus):
        """created_at 是合理的时间戳。"""
        before = time.time()
        pipe = _make_pipeline()
        task = AsyncTask(
            pipeline_name="test",
            pipeline=pipe,
            input_data=MosaicData(prompt="test"),
            bus=fresh_bus,
        )
        after = time.time()

        assert before <= task.created_at <= after, "created_at 应在合理时间范围内"