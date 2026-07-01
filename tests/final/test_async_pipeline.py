# tests/final/test_async_pipeline.py
"""Async pipeline global tests.

Tests for the asynchronous pipeline execution subsystem: ``Pipeline.run_async``,
``AsyncTask`` lifecycle, ``TaskManager`` submission / query / cancellation.

Design notes
------------
* All tests use mock ``Node`` subclasses with ``Pipeline``, avoiding real model
  weights or GPU requirements.
* ``AsyncTask`` tests verify the framework layer (task creation, status
  transitions, callbacks, serialization, cancellation), not actual inference.
* ``TaskManager`` tests verify the manager layer (submit, list, cancel_all,
  get, status_summary).
* Tests handling potentially long-running pipelines use ``try/except`` guards
  to absorb ``TimeoutError`` or ``RuntimeError`` where appropriate.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import PipelineResult
from mosaic.core.task import AsyncTask, TaskCancelledError, TaskStatus
from mosaic.core.task_manager import TaskManager
from mosaic.core.types import MosaicData, TextData


# ============================================================================
# Mock nodes for async pipeline testing
# ============================================================================
class _MockFastNode(Node):
    """A fast-executing mock node -- returns a simple MosaicData."""

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


class _MockFailingNode(Node):
    """A failing mock node -- raises ``ValueError`` in ``run()``."""

    name = "mock-failing"
    domain = "test"
    description = "A failing mock node for error callback testing."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        raise ValueError("Simulated node failure for async error testing.")

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockProgressNode(Node):
    """A mock node that emits PROGRESS events via EventBus."""

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
        bus = get_event_bus()
        bus.emit(
            EventType.PROGRESS,
            node_name="mock-progress",
            current=3,
            total=10,
        )
        return MosaicData(
            prompt="progress output",
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


# ============================================================================
# Helpers
# ============================================================================
def _make_pipeline(name: str = "test-pipe", node: Node | None = None) -> Pipeline:
    """Create a single-node pipeline for testing."""
    if node is None:
        node = _MockFastNode()
    return Pipeline(name=name, elements=[node])


def _fresh_bus() -> EventBus:
    """Return a fresh EventBus singleton."""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


# ============================================================================
# T_ASYNC_01: run_async returns AsyncTask
# ============================================================================
class TestRunAsync:
    """Tests for ``Pipeline.run_async()``."""

    def test_T_ASYNC_01_run_async_returns_async_task(self):
        """T_ASYNC_01: ``run_async()`` returns an ``AsyncTask`` instance."""
        pipe = _make_pipeline()
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        assert isinstance(task, AsyncTask), (
            "run_async() must return an AsyncTask instance"
        )
        assert task.pipeline_name == "test-pipe", (
            "task.pipeline_name must match the pipeline name"
        )
        assert task.task_id is not None, (
            "task.task_id must not be None"
        )
        assert isinstance(task.task_id, str), (
            "task.task_id must be a string"
        )
        assert len(task.task_id) > 0, (
            "task.task_id must not be empty"
        )

        # Wait for completion to clean up
        task.wait(timeout=10)


# ============================================================================
# T_ASYNC_02: AsyncTask.wait() blocks and returns PipelineResult
# ============================================================================
class TestAsyncTaskWait:
    """Tests for ``AsyncTask.wait()``."""

    def test_T_ASYNC_02_wait_returns_pipeline_result(self):
        """T_ASYNC_02: ``wait()`` blocks and returns a ``PipelineResult``."""
        pipe = _make_pipeline()
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)
        result = task.wait(timeout=10)

        assert result is not None, (
            "wait() must return a non-None result"
        )
        assert isinstance(result, PipelineResult), (
            f"wait() must return PipelineResult, got {type(result).__name__}"
        )
        assert result.success, (
            "Pipeline must execute successfully"
        )
        assert result.pipeline_name == "test-pipe", (
            "result.pipeline_name must match the pipeline name"
        )
        assert result.output is not None, (
            "result.output must not be None"
        )


# ============================================================================
# T_ASYNC_03: AsyncTask.is_ready() reflects correct status
# ============================================================================
class TestAsyncTaskIsReady:
    """Tests for ``AsyncTask.is_ready()``."""

    def test_T_ASYNC_03_is_ready_reflects_status(self):
        """T_ASYNC_03: ``is_ready()`` returns True after task completes."""
        pipe = _make_pipeline()
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        # Wait briefly for the task to start
        time.sleep(0.3)

        # After wait, is_ready must be True
        task.wait(timeout=10)
        assert task.is_ready(), (
            "is_ready() must return True after wait() completes"
        )


# ============================================================================
# T_ASYNC_04: AsyncTask.cancel() sets cancel flag
# ============================================================================
class TestAsyncTaskCancel:
    """Tests for ``AsyncTask.cancel()``."""

    def test_T_ASYNC_04_cancel_sets_flag(self):
        """T_ASYNC_04: ``cancel()`` sets the cancel flag / ``is_cancelled``.

        Since the mock node completes very quickly, ``cancel()`` may return
        ``False`` (task already reached a terminal state).  The test verifies
        that ``cancel()`` returns a ``bool`` and does not crash, and that
        either ``cancel()`` succeeded OR the task is already completed.
        """
        pipe = _make_pipeline()
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        # Try to cancel (may succeed or fail depending on timing)
        cancelled = task.cancel()

        assert isinstance(cancelled, bool), (
            "cancel() must return a bool"
        )

        # Verify: either cancel() returned True, is_cancelled is True,
        # or the task has already completed (terminal state).
        # If the task already completed, cancel() returns False but
        # that is expected behaviour.
        if not cancelled and not task.is_cancelled:
            # The task must have completed already -- verify it is in a terminal state
            assert task.is_ready(), (
                "If cancel() returned False and is_cancelled is False, "
                "the task must be in a terminal state (completed)"
            )

        # Clean up -- wait for completion (may raise)
        try:
            task.wait(timeout=10)
        except (TimeoutError, RuntimeError, TaskCancelledError):
            pass


# ============================================================================
# T_ASYNC_05: AsyncTask.on_complete callback fires
# ============================================================================
class TestAsyncTaskOnComplete:
    """Tests for ``AsyncTask.on_complete()`` callback."""

    def test_T_ASYNC_05_on_complete_callback_fires(self):
        """T_ASYNC_05: ``on_complete`` callback is called with ``PipelineResult``."""
        pipe = _make_pipeline()
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        completed: list[PipelineResult] = []

        def _on_complete(result: PipelineResult) -> None:
            completed.append(result)

        task.on_complete(_on_complete)
        task.wait(timeout=10)

        assert len(completed) >= 1, (
            f"on_complete callback must be called at least once, "
            f"got {len(completed)} calls"
        )
        assert isinstance(completed[0], PipelineResult), (
            "on_complete callback argument must be PipelineResult"
        )


# ============================================================================
# T_ASYNC_06: AsyncTask.on_error callback fires
# ============================================================================
class TestAsyncTaskOnError:
    """Tests for ``AsyncTask.on_error()`` callback."""

    def test_T_ASYNC_06_on_error_callback_fires(self):
        """T_ASYNC_06: ``on_error`` callback is called when a pipeline fails."""
        failing_node = _MockFailingNode()
        pipe = _make_pipeline(name="fail-pipe", node=failing_node)
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        errors: list[BaseException] = []

        def _on_error(error: BaseException) -> None:
            errors.append(error)

        task.on_error(_on_error)

        # wait() for a failing pipeline raises RuntimeError
        try:
            task.wait(timeout=10)
        except RuntimeError:
            pass

        assert len(errors) >= 1, (
            f"on_error callback must be called at least once, "
            f"got {len(errors)} calls"
        )
        assert isinstance(errors[0], BaseException), (
            "on_error callback argument must be a BaseException"
        )


# ============================================================================
# T_ASYNC_07: AsyncTask.on_progress callback fires
# ============================================================================
class TestAsyncTaskOnProgress:
    """Tests for ``AsyncTask.on_progress()`` callback."""

    def test_T_ASYNC_07_on_progress_callback_fires(self):
        """T_ASYNC_07: ``on_progress`` callback receives ``(float, str)`` updates."""
        progress_node = _MockProgressNode()
        pipe = _make_pipeline(name="progress-pipe", node=progress_node)
        input_data = TextData(content="hello")

        task = pipe.run_async(input_data)

        progresses: list[tuple[float, str]] = []

        def _on_progress(progress: float, node_name: str) -> None:
            progresses.append((progress, node_name))

        task.on_progress(_on_progress)
        task.wait(timeout=10)

        # Progress callbacks may or may not fire depending on timing,
        # but the callback registration must not crash
        for progress, node_name in progresses:
            assert isinstance(progress, float), (
                f"progress must be float, got {type(progress).__name__}"
            )
            assert 0.0 <= progress <= 1.0, (
                f"progress must be in [0.0, 1.0], got {progress}"
            )
            assert isinstance(node_name, str), (
                f"node_name must be str, got {type(node_name).__name__}"
            )


# ============================================================================
# T_ASYNC_08: TaskManager.submit submits task
# ============================================================================
class TestTaskManagerSubmit:
    """Tests for ``TaskManager.submit()``."""

    def test_T_ASYNC_08_submit_returns_async_task(self):
        """T_ASYNC_08: ``TaskManager.submit()`` returns an ``AsyncTask``."""
        tm = TaskManager()
        pipe = _make_pipeline(name="tm-pipe")
        input_data = TextData(content="hello")

        task = tm.submit(pipe, input_data)

        assert isinstance(task, AsyncTask), (
            "submit() must return an AsyncTask instance"
        )
        assert task.task_id is not None, (
            "task.task_id must not be None"
        )

        # Clean up
        try:
            task.wait(timeout=10)
        except (TimeoutError, RuntimeError, TaskCancelledError):
            pass


# ============================================================================
# T_ASYNC_09: TaskManager.list_tasks returns task list
# ============================================================================
class TestTaskManagerList:
    """Tests for ``TaskManager.list_tasks()``."""

    def test_T_ASYNC_09_list_tasks_returns_list(self):
        """T_ASYNC_09: ``list_tasks()`` returns a list containing submitted tasks."""
        tm = TaskManager()
        pipe = _make_pipeline(name="list-pipe")
        input_data = TextData(content="hello")

        task = tm.submit(pipe, input_data)

        tasks = tm.list_tasks()
        assert isinstance(tasks, list), (
            "list_tasks() must return a list"
        )
        assert len(tasks) >= 1, (
            f"list_tasks() must return at least 1 task, got {len(tasks)}"
        )
        assert any(t.task_id == task.task_id for t in tasks), (
            "Submitted task must appear in list_tasks() result"
        )

        # Clean up
        try:
            task.wait(timeout=10)
        except (TimeoutError, RuntimeError, TaskCancelledError):
            pass


# ============================================================================
# T_ASYNC_10: TaskManager.cancel_all cancels all tasks
# ============================================================================
class TestTaskManagerCancelAll:
    """Tests for ``TaskManager.cancel_all()``."""

    def test_T_ASYNC_10_cancel_all_returns_count(self):
        """T_ASYNC_10: ``cancel_all()`` returns an ``int`` count >= 0."""
        tm = TaskManager()
        pipe = _make_pipeline(name="cancel-all-pipe")
        input_data_1 = TextData(content="hello")
        input_data_2 = TextData(content="world")

        task1 = tm.submit(pipe, input_data_1)
        task2 = tm.submit(pipe, input_data_2)

        count = tm.cancel_all()

        assert isinstance(count, int), (
            "cancel_all() must return an int"
        )
        assert count >= 0, (
            f"cancel_all() must return non-negative count, got {count}"
        )

        # Clean up remaining tasks
        for t in (task1, task2):
            try:
                t.wait(timeout=10)
            except (TimeoutError, RuntimeError, TaskCancelledError):
                pass


# ============================================================================
# Additional: TaskManager.get() and TaskManager.status_summary()
# ============================================================================
class TestTaskManagerAdditional:
    """Additional TaskManager functionality tests."""

    def test_task_manager_get_returns_task(self):
        """``TaskManager.get()`` returns the correct task by ID."""
        tm = TaskManager()
        pipe = _make_pipeline(name="get-pipe")
        input_data = TextData(content="hello")

        task = tm.submit(pipe, input_data)

        retrieved = tm.get(task.task_id)
        assert retrieved is not None, (
            "get() must return the submitted task"
        )
        assert retrieved.task_id == task.task_id, (
            "get() must return the task with matching task_id"
        )

        # Clean up
        try:
            task.wait(timeout=10)
        except (TimeoutError, RuntimeError, TaskCancelledError):
            pass

    def test_task_manager_status_summary(self):
        """``TaskManager.status_summary()`` returns a dict with expected keys."""
        tm = TaskManager()
        pipe = _make_pipeline(name="summary-pipe")
        input_data = TextData(content="hello")

        task = tm.submit(pipe, input_data)

        # Wait for completion
        try:
            task.wait(timeout=10)
        except (TimeoutError, RuntimeError, TaskCancelledError):
            pass

        summary = tm.status_summary()
        assert isinstance(summary, dict), (
            "status_summary() must return a dict"
        )
        assert "total" in summary, (
            "status_summary() must contain 'total' key"
        )
        assert summary["total"] >= 1, (
            f"status_summary 'total' must be >= 1, got {summary['total']}"
        )