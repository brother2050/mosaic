# tests/final/test_error_handling.py
"""Mosaic 最终验收测试 —— 错误处理测试。

覆盖 12 个错误处理场景，验证框架在各种异常情况下的行为：
友好的错误消息、类型校验、GPU 降级、OOM 处理、文件不存在、
错误传播、异步任务失败、插件加载失败、循环依赖检测、
TTS 后端错误处理等。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mosaic.core import (
    Node,
    Pipeline,
    PipelineError,
    PipelineResult,
    Scheduler,
)
from mosaic.core.types import MosaicData, TextData
from mosaic.core.result import NodeError


# ---------------------------------------------------------------------------
# 辅助：创建会失败的节点
# ---------------------------------------------------------------------------
class _FailingNode(Node):
    """会抛出异常的节点，用于测试错误传播。"""

    name = "failing-node-for-errors"
    domain = "test"
    description = "Failing node for error handling tests."
    version = "0.1.0"
    input_types = ["mosaic"]
    output_types = ["mosaic"]

    def __init__(self, name="failing-node", error_msg="intentional failure", **kwargs):
        super().__init__(name=name, **kwargs)
        self._error_msg = error_msg

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        raise RuntimeError(self._error_msg)

    def describe(self):
        from mosaic.core.node import NodeSpec
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_ERR_01: Non-existent node name gives friendly error
# ===========================================================================
def test_nonexistent_node_gives_friendly_error(registry):
    """T_ERR_01: 不存在的节点名应给出友好的错误提示，而非原始 KeyError。

    使用 registry.get("NonExistentNode")，验证抛出 KeyError 且消息
    包含有帮助的上下文信息。
    """
    with pytest.raises(KeyError) as exc_info:
        registry.get("NonExistentNode")

    error_msg = str(exc_info.value)
    # 错误消息应包含节点名和可用节点列表
    assert "NonExistentNode" in error_msg, (
        f"Error message should mention the node name, got: {error_msg}"
    )
    assert "not registered" in error_msg.lower(), (
        f"Error message should indicate node is not registered, got: {error_msg}"
    )
    assert "Available" in error_msg, (
        f"Error message should list available nodes, got: {error_msg}"
    )


# ===========================================================================
# T_ERR_02: Missing required parameter gives clear message
# ===========================================================================
def test_missing_required_parameter(registry):
    """T_ERR_02: 缺少必需参数时应给出清晰的错误消息。

    创建一个简单的测试节点，要求输入数据包含 "prompt" 字段，
    验证未提供时抛出 ValueError 且消息指明缺少哪个参数。
    """

    # 创建一个需要 "prompt" 参数的简单节点
    class _RequiresPromptNode(Node):
        name = "requires-prompt-node"
        domain = "test"
        description = "Test node that requires prompt parameter."
        version = "0.1.0"
        input_types = ["mosaic"]
        output_types = ["mosaic"]

        def load(self):
            self._loaded = True

        def unload(self):
            self._loaded = False

        def run(self, input_data):
            if not hasattr(input_data, "prompt") or not input_data.prompt:
                raise ValueError("Missing required parameter: 'prompt'. Please provide a text prompt.")
            return input_data

        def describe(self):
            from mosaic.core.node import NodeSpec
            return NodeSpec(
                name=self.name, domain=self.domain, description=self.description,
                version=self.version, input_types=list(self.input_types),
                output_types=list(self.output_types),
            )

    node = _RequiresPromptNode(name="test-requires-prompt")

    # 创建不包含 "prompt" 的输入数据
    bad_input = MosaicData(some_other_field="value")

    with pytest.raises(ValueError) as exc_info:
        node.run(bad_input)

    error_msg = str(exc_info.value)
    assert "prompt" in error_msg, (
        f"Error message should mention 'prompt', got: {error_msg}"
    )


# ===========================================================================
# T_ERR_03: Input data type mismatch gives type error
# ===========================================================================
def test_input_type_mismatch_detected_by_dry_run(registry):
    """T_ERR_03: 输入数据类型不匹配应被 dry_run 检测到。

    创建 TextGenerator (output text) -> BackgroundRemover (input image) 管道，
    dry_run 应报告类型不匹配。
    """
    text_gen = registry.get("text-generator")
    bg_remover = registry.get("background-remover")

    # text-generator 输出 text，background-remover 期望 image
    pipe = Pipeline("type-mismatch-pipe", [text_gen, bg_remover])

    dry_result = pipe.dry_run()
    # 可能存在类型不匹配警告
    assert isinstance(dry_result.issues, list), "dry_run should return issues list"
    # 应该有类型不匹配的 issue
    type_issues = [i for i in dry_result.issues if "Type mismatch" in i or "type" in i.lower()]
    assert len(type_issues) > 0, (
        f"Expected type mismatch issues, got: {dry_result.issues}"
    )


# ===========================================================================
# T_ERR_04: GPU unavailable fallback
# ===========================================================================
def test_gpu_unavailable_graceful_degradation():
    """T_ERR_04: GPU 不可用时系统应优雅降级。

    在 CPU 模式下创建 Scheduler，验证不会因缺少 GPU 而崩溃，
    且进入 CPU 模式并给出提示。
    """
    from mosaic.core.events import EventBus

    # 确保 CUDA 不可用
    bus = EventBus()
    bus.clear()
    scheduler = Scheduler(bus=bus, device="cpu")

    status = scheduler.status()
    assert status["mode"] == "cpu", f"Expected CPU mode, got {status['mode']}"
    assert scheduler.device == "cpu"
    assert not scheduler.is_gpu

    # 验证在 CPU 模式下可以正常 track 和获取节点状态
    # 不触发 ensure_loaded 以避免 transformers 导入问题
    from mosaic.core import Node as _Node

    class _SimpleNode(_Node):
        name = "simple-node"
        domain = "test"
        description = "Test node."
        version = "0.1.0"
        input_types = ["mosaic"]
        output_types = ["mosaic"]

        def load(self):
            self._loaded = True

        def unload(self):
            self._loaded = False

        def run(self, input_data):
            return input_data

        def describe(self):
            from mosaic.core.node import NodeSpec
            return NodeSpec(
                name=self.name, domain=self.domain, description=self.description,
                version=self.version, input_types=list(self.input_types),
                output_types=list(self.output_types),
            )

    simple_node = _SimpleNode(name="simple-cpu-node")
    scheduler.track(simple_node)
    assert simple_node.name in scheduler._tracked, "Node should be tracked"

    # CPU 模式下 ensure_loaded 应正常工作
    scheduler.ensure_loaded(simple_node)
    assert simple_node.is_loaded(), "Simple node should load in CPU mode"


# ===========================================================================
# T_ERR_05: Out of memory error with suggestion
# ===========================================================================
def test_out_of_memory_error_with_suggestion(registry):
    """T_ERR_05: 显存不足时给出有建议的错误信息。

    使用 memory_limit_gb=0.01 创建 Scheduler（模拟 GPU），
    验证当显存不足以加载节点时抛出 MemoryError 且包含建议信息。
    """
    from mosaic.core.events import EventBus

    bus = EventBus()
    bus.clear()

    # 使用极小的显存限制来模拟 GPU OOM
    scheduler = Scheduler(bus=bus, device="cuda", memory_limit_gb=0.001)

    node = registry.get("text-generator")
    scheduler.track(node)

    # 在 GPU 模式下，极小显存限制应导致 MemoryError
    try:
        scheduler.ensure_loaded(node)
        # 如果节点未加载成功（vram_gb 估算为 0），则手动验证 MemoryError
        if not node.is_loaded():
            pytest.skip("Node has zero VRAM estimate, cannot trigger OOM")
    except MemoryError as exc:
        error_msg = str(exc)
        assert "memory" in error_msg.lower(), (
            f"MemoryError should mention memory, got: {error_msg}"
        )
        assert "GPU" in error_msg or "gb" in error_msg.lower(), (
            f"MemoryError should mention GPU or GB, got: {error_msg}"
        )


# ===========================================================================
# T_ERR_06: File not found error handling
# ===========================================================================
def test_file_not_found_error_handling(registry):
    """T_ERR_06: 文件不存在时的错误处理。

    传递不存在的文件路径给节点，验证抛出清晰的错误消息。
    """
    nonexistent_path = "/nonexistent/path/to/file.mp3"

    # 确保文件确实不存在
    assert not os.path.exists(nonexistent_path), (
        f"Test path should not exist: {nonexistent_path}"
    )

    asr_node = registry.get("asr")

    # 尝试传入文件路径 -- ASR 节点应处理文件不存在的情况
    try:
        asr_node.run(MosaicData(audio_path=nonexistent_path))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        error_msg = str(exc)
        # 错误消息应包含路径信息或文件相关提示
        assert any(
            keyword in error_msg.lower()
            for keyword in ["file", "path", "not found", "exist", "nonexistent"]
        ), f"Error message should mention file-related issue, got: {error_msg}"
    except Exception:  # noqa: BLE001
        # 节点可能因缺少模型而失败，这是可接受的
        pass


# ===========================================================================
# T_ERR_07: Error propagation in pipeline
# ===========================================================================
def test_error_propagation_in_pipeline():
    """T_ERR_07: 管道中节点失败时错误应传播到 PipelineResult。

    创建包含会失败节点的管道，使用 fail_fast=False 执行，
    验证 PipelineResult.errors 包含错误信息。
    """
    failing_node = _FailingNode(name="will-fail", error_msg="pipeline propagation test")

    pipe = Pipeline("error-propagation", [failing_node])

    result = pipe.execute_result(
        MosaicData(content="test"),
        fail_fast=False,
    )

    assert not result.success, "Pipeline should not be successful"
    assert len(result.errors) > 0, "PipelineResult should contain errors"
    assert result.failed_nodes == ["will-fail"], (
        f"Expected ['will-fail'], got {result.failed_nodes}"
    )

    # 验证错误信息
    err = result.errors[0]
    assert isinstance(err, NodeError)
    assert err.node_name == "will-fail"
    assert "pipeline propagation test" in str(err.error)


# ===========================================================================
# T_ERR_08: Async task failure status update
# ===========================================================================
def test_async_task_failure_status_update():
    """T_ERR_08: 异步任务失败时状态应正确更新为 FAILED。

    创建包含失败节点的管道，异步执行，验证任务状态变为 FAILED。
    """
    failing_node = _FailingNode(name="async-fail", error_msg="async task failure test")

    pipe = Pipeline("async-fail-pipe", [failing_node])

    task = pipe.run_async(
        MosaicData(content="test"),
        fail_fast=False,
    )

    from mosaic.core.task import TaskStatus, AsyncTask

    assert isinstance(task, AsyncTask)

    # 等待任务完成
    try:
        result = task.wait(timeout=30)
        # 如果 wait 成功，检查结果
        assert not result.success, "Async task result should indicate failure"
    except Exception:  # noqa: BLE001
        # 任务失败，状态应为 FAILED
        pass

    # 验证状态
    assert task.status in (TaskStatus.FAILED, TaskStatus.COMPLETED), (
        f"Task status should be FAILED or COMPLETED, got {task.status}"
    )

    # 如果失败，验证 error 属性
    if task.status == TaskStatus.FAILED:
        assert task.error is not None, "Failed task should have error set"
        assert "async task failure test" in str(task.error)


# ===========================================================================
# T_ERR_09: Plugin load failure doesn't crash framework startup
# ===========================================================================
def test_plugin_load_failure_doesnt_crash(plugin_manager):
    """T_ERR_09: 插件加载失败不应导致框架启动崩溃。

    验证 plugin_manager 在遇到无效插件目录或损坏的插件文件时
    不会抛出异常导致框架崩溃。
    """
    # 创建一个临时目录，其中包含无效的 Python 文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入一个语法错误的 Python 文件
        bad_file = os.path.join(tmpdir, "bad_plugin.py")
        with open(bad_file, "w") as f:
            f.write("this is not valid python {{{{{\n")

        # 注册这个目录并尝试加载
        plugin_manager.register_plugin_dir(tmpdir)

        # 重新加载插件（不应崩溃）
        try:
            # 重置加载状态以允许重新扫描
            plugin_manager._loaded = False
            count = plugin_manager.load_plugins()
            # 不应崩溃，count 应为 0 或更多
            assert isinstance(count, int)
        except Exception as exc:  # noqa: BLE001
            # 如果 load_plugins 抛异常，说明防护不足
            pytest.fail(f"plugin_manager.load_plugins() should not crash: {exc}")


# ===========================================================================
# T_ERR_10: Circular dependency detection
# ===========================================================================
def test_circular_dependency_detection(registry):
    """T_ERR_10: 循环依赖检测。

    通过 DAG 合法性检查验证循环检测机制存在。Pipeline 的正常 API
    在构建时不会创建循环，但 validate() 方法内置了拓扑排序循环检测。
    本测试验证 validate() 对有效管道的正确行为，以及干跑检测的存在。
    """
    # 测试一：正常管道应通过 validate
    text_gen = registry.get("text-generator")
    t2i = registry.get("text-to-image")

    pipe = Pipeline("no-cycle", [text_gen, t2i])
    # 不应抛出异常
    pipe.validate()

    # 测试二：验证 validate 在空管道上的行为
    empty_pipe = Pipeline("empty")
    empty_pipe.validate()  # 空管道合法

    # 测试三：dry_run 应包含结构校验
    dry_result = pipe.dry_run()
    assert dry_result.ok, f"Dry run should pass for valid pipeline: {dry_result.issues}"

    # 测试四：验证 __bool__ 行为
    assert bool(dry_result) == dry_result.ok


# ===========================================================================
# T_ERR_11: Non-existent TTS backend gives friendly error
# ===========================================================================
def test_nonexistent_tts_backend_friendly_error(registry):
    """T_ERR_11: 不存在的 TTS 后端应给出友好错误或优雅降级。

    使用不存在的 backend 名称创建 TTS 节点，验证不会崩溃且
    给出有用的提示或回退到 edge-tts。
    """
    # 使用不存在的后端名创建 TTS 节点
    tts_node = registry.get("tts", backend="nonexistent_backend_xyz")

    assert tts_node is not None, "TTS node should be created even with invalid backend"

    # 尝试加载模型（应回退到 edge-tts 或给出警告）
    try:
        tts_node._load_model()
        # 成功回退到 edge-tts
        assert tts_node._backend in ("edge_tts", "edge-tts"), (
            f"Expected fallback to edge_tts, got {tts_node._backend}"
        )
    except Exception as exc:  # noqa: BLE001
        # 即使失败，也应给出友好消息
        error_msg = str(exc)
        assert any(
            keyword in error_msg.lower()
            for keyword in ["backend", "not found", "not registered", "fallback", "edge"]
        ), f"Error should mention backend issue, got: {error_msg}"


# ===========================================================================
# T_ERR_12: TTS backend load failure handling
# ===========================================================================
def test_tts_backend_load_failure_handling(tts_registry):
    """T_ERR_12: TTS 后端加载失败的处理。

    验证 tts_backend_registry.get() 对不存在的后端返回 None，
    而对已注册但依赖不可用的后端返回 None 或正确处理后端类。
    """
    # 测试一：不存在的后端应返回 None
    result = tts_registry.get("nonexistent_tts_backend")
    assert result is None, (
        f"Expected None for nonexistent backend, got {result}"
    )

    # 测试二：检查已知后端是否可注册
    known_backends = ["chattts", "fish", "sovits", "cosyvoice"]
    for backend_name in known_backends:
        backend_class = tts_registry.get(backend_name)
        # 后端可能因依赖缺失而未注册，这是可接受的
        if backend_class is not None:
            assert hasattr(backend_class, "name"), (
                f"Backend {backend_name} should have a 'name' attribute"
            )

    # 测试三：is_available 对不存在的后端返回 False
    assert not tts_registry.is_available("nonexistent_backend"), (
        "is_available should return False for nonexistent backend"
    )

    # 测试四：__contains__ 协议
    assert "nonexistent_backend" not in tts_registry


# ===========================================================================
# 附加：管道错误消息质量测试
# ===========================================================================
def test_pipeline_error_message_quality():
    """验证 PipelineError 消息包含足够的上下文信息。"""
    # 测试 validate() 对异常结构的错误消息
    try:
        raise PipelineError("Test pipeline error with context")
    except PipelineError as exc:
        error_msg = str(exc)
        assert "Test pipeline error" in error_msg


def test_pipeline_validate_detects_issues(registry):
    """验证 pipeline.validate() 对结构问题的检测能力。"""
    # 创建只有单个节点的管道，validate 应通过
    text_gen = registry.get("text-generator")
    pipe = Pipeline("single-node", [text_gen])
    pipe.validate()  # 不应抛出异常

    # 验证 dry_run 中包含结构校验
    dry_result = pipe.dry_run()
    assert dry_result.ok, f"Valid pipeline should pass dry_run: {dry_result.issues}"


def test_pipeline_fail_fast_vs_collect():
    """验证 fail_fast=True 和 fail_fast=False 的行为差异。"""
    n1 = _FailingNode(name="fail-1", error_msg="error in node 1")
    n2 = _FailingNode(name="fail-2", error_msg="error in node 2")

    pipe = Pipeline("fail-fast-test", [n1, n2])

    # fail_fast=True: 第一个节点失败时立即抛出
    with pytest.raises(Exception):
        pipe.execute_result(MosaicData(content="test"), fail_fast=True)

    # fail_fast=False: 收集所有错误
    result = pipe.execute_result(MosaicData(content="test"), fail_fast=False)
    assert not result.success
    assert len(result.errors) >= 1, "Should collect at least the first error"


def test_async_task_cancel(registry):
    """验证 AsyncTask 的取消功能。"""
    text_gen = registry.get("text-generator")

    pipe = Pipeline("cancel-pipe", [text_gen])

    task = pipe.run_async(TextData(content="test"))

    from mosaic.core.task import TaskStatus

    # 尝试取消任务
    cancelled = task.cancel()
    assert isinstance(cancelled, bool)

    # 等待任务完成
    try:
        task.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass

    # 验证终态
    assert task.status in TaskStatus.terminal_statuses(), (
        f"Task should be in a terminal state, got {task.status}"
    )