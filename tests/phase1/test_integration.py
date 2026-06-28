# tests/phase1/test_integration.py
"""Phase 1 端到端集成测试。

覆盖多节点管道、事件触发与中间结果访问。
使用 mock 节点测试管道编排能力，不依赖真实模型。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.pipeline import Pipeline, Branch, Merge
from mosaic.core.types import MosaicData, TextData
from mosaic.core.events import EventBus, EventType
from mosaic.core.scheduler import Scheduler, set_scheduler
from mosaic.core.node import Node, NodeSpec


# ===========================================================================
# 辅助节点：用于管道编排测试，输入/输出键兼容
# ===========================================================================
class _GenNode(Node):
    """模拟文本生成节点：输入 prompt → 输出 text。"""

    name = "mock-generator"
    domain = "text"
    description = "Mock text generator"
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, name="mock-generator", content="generated content", **kwargs):
        super().__init__(name=name, **kwargs)
        self._content = content

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        prompt = input_data.get("prompt", "")
        result = MosaicData(
            text=f"[GEN:{prompt}→{self._content}]",
            generated_text=self._content,
        )
        # 透传下游节点需要的键
        for key in ("labels", "target_language", "max_length", "style"):
            if key in input_data:
                result[key] = input_data[key]
        return result

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _TranslateNode(Node):
    """模拟翻译节点：输入 text → 输出 translated_text。"""

    name = "mock-translator"
    domain = "text"
    description = "Mock translator"
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, name="mock-translator", target="zh", **kwargs):
        super().__init__(name=name, **kwargs)
        self._target = target

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        text = input_data.get("text", "")
        return MosaicData(
            translated_text=f"[TRANS:{text}→{self._target}]",
            target_language=self._target,
        )

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _SummarizeNode(Node):
    """模拟摘要节点：输入 text → 输出 summary。"""

    name = "mock-summarizer"
    domain = "text"
    description = "Mock summarizer"
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, name="mock-summarizer", **kwargs):
        super().__init__(name=name, **kwargs)

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        text = input_data.get("text", "")
        result = MosaicData(
            summary=f"[SUMMARY:{text[:30]}...]",
            compression_ratio=0.5,
        )
        # 透传下游节点需要的键
        for key in ("labels", "target_language"):
            if key in input_data:
                result[key] = input_data[key]
        return result

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _ClassifyNode(Node):
    """模拟分类节点：输入 text + labels → 输出 predicted_label。"""

    name = "mock-classifier"
    domain = "text"
    description = "Mock classifier"
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, name="mock-classifier", **kwargs):
        super().__init__(name=name, **kwargs)

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        text = input_data.get("text", "")
        labels = input_data.get("labels", ["default"])
        return MosaicData(
            predicted_label=labels[0],
            scores={lbl: (1.0 if lbl == labels[0] else 0.0) for lbl in labels},
            method="mock",
        )

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def integration_bus():
    """集成测试用的事件总线。"""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


@pytest.fixture
def integration_scheduler(integration_bus):
    """集成测试用的 CPU 调度器。"""
    sched = Scheduler(bus=integration_bus, device="cpu")
    set_scheduler(sched)
    return sched


# ===========================================================================
# T_E2E_01: "文本生成 → 翻译" 管道完整运行
# ===========================================================================
@pytest.mark.integration
class TestE2EGenerationTranslation:
    """端到端：文本生成 → 翻译。"""

    def test_gen_translate_pipeline(self, integration_scheduler, integration_bus):
        """T_E2E_01: "文本生成 → 翻译" 管道完整运行。"""
        gen = _GenNode(name="generator", content="Hello, world!")
        translator = _TranslateNode(name="translator", target="zh")

        pipe = Pipeline("gen2translate", [gen, translator])
        result = pipe.execute(MosaicData(
            prompt="Introduce yourself.",
            target_language="zh",
        ))

        assert "translated_text" in result
        assert isinstance(result["translated_text"], str)
        assert "[TRANS:" in result["translated_text"]

    def test_intermediate_result_accessible(self, integration_scheduler, integration_bus):
        """T_E2E_01: 运行后可以获取 generation 节点的中间结果。"""
        gen = _GenNode(name="generator", content="test content")
        translator = _TranslateNode(name="translator", target="zh")

        pipe = Pipeline("gen2translate", [gen, translator])
        pipe.execute(MosaicData(
            prompt="Say hello.",
            target_language="zh",
        ))

        # 获取中间生成结果
        assert len(pipe.intermediate_names) > 0
        assert any("generator" in name for name in pipe.intermediate_names)


# ===========================================================================
# T_E2E_02: "文本生成 → 摘要 → 分类" 三节点管道
# ===========================================================================
@pytest.mark.integration
class TestE2EThreeNodePipeline:
    """端到端：文本生成 → 摘要 → 分类。"""

    def test_three_node_pipeline(self, integration_scheduler, integration_bus):
        """T_E2E_02: "文本生成 → 摘要 → 分类" 三节点管道。"""
        gen = _GenNode(name="generator", content="long text about AI")
        summarizer = _SummarizeNode(name="summarizer")
        classifier = _ClassifyNode(name="classifier")

        pipe = Pipeline("gen2sum2cls", [gen, summarizer, classifier])
        result = pipe.execute(MosaicData(
            prompt="Write about AI.",
            labels=["科技", "教育", "娱乐"],
        ))

        assert "predicted_label" in result
        assert result["predicted_label"] in ["科技", "教育", "娱乐"]
        assert "scores" in result
        assert "method" in result

    def test_pipeline_as_node_nesting(self, integration_scheduler, integration_bus):
        """T_E2E_02: 管道作为 Node 嵌套到更大的管道中。"""
        gen = _GenNode(name="generator", content="ML content")
        translator = _TranslateNode(name="translator", target="zh")
        summarizer = _SummarizeNode(name="summarizer")

        # 子管道：生成 → 翻译
        sub_pipe = Pipeline("gen2translate", [gen, translator])
        # 父管道：子管道 → 摘要
        parent_pipe = Pipeline("parent", [sub_pipe, summarizer])

        result = parent_pipe.execute(MosaicData(
            prompt="Write about machine learning.",
            target_language="zh",
        ))

        assert "summary" in result
        assert "compression_ratio" in result


# ===========================================================================
# T_E2E_03: 运行过程中事件被正确触发
# ===========================================================================
@pytest.mark.integration
class TestE2EEvents:
    """端到端：事件触发测试。"""

    def test_events_fired_during_pipeline(self, integration_scheduler, integration_bus):
        """T_E2E_03: 运行过程中事件被正确触发（通过 Context）。"""
        from mosaic.core.context import Context

        gen = _GenNode(name="generator", content="hello")
        translator = _TranslateNode(name="translator", target="zh")

        events_received = []
        ctx = Context()
        ctx.on_event(lambda e: events_received.append(e.type))

        pipe = Pipeline("event-test", [gen, translator])
        pipe.execute(MosaicData(
            prompt="Say hello.",
            target_language="zh",
        ), context=ctx)

        # 至少应有 pipeline_start, node_end (x2), pipeline_end
        assert "pipeline_start" in events_received, f"Got: {events_received}"
        assert "pipeline_end" in events_received, f"Got: {events_received}"
        node_end_count = sum(1 for e in events_received if e == "node_end")
        assert node_end_count >= 2, f"Expected >=2 node_end events, got {node_end_count}"

    def test_no_error_events_on_success(self, integration_scheduler, integration_bus):
        """T_E2E_03: 成功运行不产生错误事件。"""
        gen = _GenNode(name="generator", content="ok")
        translator = _TranslateNode(name="translator", target="zh")

        error_events = []
        integration_bus.on(EventType.NODE_ERROR, lambda e: error_events.append(e))

        pipe = Pipeline("success-test", [gen, translator])
        pipe.execute(MosaicData(
            prompt="Hello.",
            target_language="zh",
        ))

        assert len(error_events) == 0, f"Expected 0 error events, got {len(error_events)}"

    def test_context_events_include_pipeline_lifecycle(self, integration_scheduler, integration_bus):
        """T_E2E_03: Context 产生 pipeline_start/pipeline_end 事件。"""
        from mosaic.core.context import Context

        gen = _GenNode(name="generator", content="test")

        events = []
        ctx = Context()
        ctx.on_event(lambda e: events.append(e.type))

        pipe = Pipeline("lifecycle", [gen])
        pipe.execute(MosaicData(prompt="test"), context=ctx)

        assert "pipeline_start" in events
        assert "pipeline_end" in events


# ===========================================================================
# T_E2E_04: 运行结束后中间结果可访问
# ===========================================================================
@pytest.mark.integration
class TestE2EIntermediates:
    """端到端：中间结果访问测试。"""

    def test_all_intermediates_accessible(self, integration_scheduler, integration_bus):
        """T_E2E_04: 运行结束后中间结果可访问。"""
        gen = _GenNode(name="generator", content="AI ethics content")
        summarizer = _SummarizeNode(name="summarizer")
        classifier = _ClassifyNode(name="classifier")

        pipe = Pipeline("all-three", [gen, summarizer, classifier])
        pipe.execute(MosaicData(
            prompt="Write about AI ethics.",
            labels=["科技", "哲学", "法律"],
        ))

        # 中间产物应包含所有节点
        assert len(pipe.intermediate_names) >= 3

        # 可以获取第一个节点的中间产物
        gen_names = [n for n in pipe.intermediate_names if "generator" in n]
        assert len(gen_names) >= 1
        gen_output = pipe.get_intermediate(gen_names[0])
        assert "text" in gen_output or "generated_text" in gen_output

        # 可以获取第二个节点的中间产物
        sum_names = [n for n in pipe.intermediate_names if "summarizer" in n]
        assert len(sum_names) >= 1
        sum_output = pipe.get_intermediate(sum_names[0])
        assert "summary" in sum_output

        # 最终结果包含分类标签
        cls_names = [n for n in pipe.intermediate_names if "classifier" in n]
        assert len(cls_names) >= 1

    def test_operator_chain_intermediates(self, integration_scheduler, integration_bus):
        """T_E2E_04: | 运算符链式管道中间结果可访问。"""
        gen = _GenNode(name="generator", content="hello")
        translator = _TranslateNode(name="translator", target="zh")

        pipe = gen | translator
        pipe.execute(MosaicData(
            prompt="Hello.",
            target_language="zh",
        ))

        assert len(pipe.intermediate_names) >= 2

    def test_branch_merge_intermediates(self, integration_scheduler, integration_bus):
        """T_E2E_04: 分支管道中间结果可访问。"""
        gen = _GenNode(name="generator", content="shared")
        branch_a = _TranslateNode(name="translator-a", target="en")
        branch_b = _TranslateNode(name="translator-b", target="ja")
        merge = Merge(strategy="dict")

        pipe = Pipeline("branch-test", [gen, Branch(a=branch_a, b=branch_b), merge])
        pipe.execute(MosaicData(prompt="test"))

        assert len(pipe.intermediate_names) >= 4  # gen + 2 translators + merge