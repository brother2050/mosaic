# /workspace/mosaic/tests/phase2/test_integration.py
"""Phase 2 端到端集成测试。

覆盖跨域管道、图像域管道、并行分支、合并、中间产物保存、事件触发
与 PipelineResult 摘要信息。使用 mock 节点，不依赖真实模型。
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
from PIL import Image

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.pipeline import Pipeline, Branch, Merge
from mosaic.core.types import MosaicData, ImageData
from mosaic.core.result import PipelineResult
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec


# ===========================================================================
# 辅助 mock 节点
# ===========================================================================
class _MockTextNode(Node):
    """模拟文本生成节点：输入 prompt → 输出 content。"""

    name = "mock-text-gen"
    domain = "text"
    description = "Mock text generator for integration tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def __init__(self, name="mock-text-gen", content="generated text", **kwargs):
        super().__init__(name=name, **kwargs)
        self._content = content
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        prompt = input_data.get("prompt", "")
        return MosaicData(
            content=f"[TEXT:{prompt}->{self._content}]",
            text=self._content,
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


class _MockImageGenNode(Node):
    """模拟图像生成节点：输入 content → 输出 PIL Image。"""

    name = "mock-image-gen"
    domain = "image"
    description = "Mock image generator for integration tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-image-gen", color=(100, 150, 200), **kwargs):
        super().__init__(name=name, **kwargs)
        self._color = color
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        content = input_data.get("content", "default")
        img = Image.new("RGB", (256, 256), color=self._color)
        return ImageData(
            image=img,
            size=(256, 256),
            metadata={"source": content, "node": self.name},
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


class _MockImageProcessorNode(Node):
    """模拟图像处理节点（如背景去除、风格化）：输入 image → 输出处理后的 PIL Image。"""

    name = "mock-image-processor"
    domain = "image"
    description = "Mock image processor for integration tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-image-processor", color=(50, 200, 100), tag="processed", **kwargs):
        super().__init__(name=name, **kwargs)
        self._color = color
        self._tag = tag
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        # 获取输入图像，若不存在则创建新图
        img = input_data.get("image")
        if img is None:
            img = Image.new("RGB", (256, 256), color=self._color)
        else:
            # 模拟处理：改变尺寸
            img = img.resize((128, 128))
        return ImageData(
            image=img,
            size=img.size,
            metadata={"tag": self._tag, "node": self.name},
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
# T_E2E_P2_01: 跨域管道：文本生成 → 图像生成
# ===========================================================================
@pytest.mark.integration
def test_cross_domain_text_to_image(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_01: 跨域管道：文本生成 → 图像生成。

    使用 mock 文本节点和 mock 图像生成节点，验证输出从文本域
    流向图像域。
    """
    text_node = _MockTextNode(name="text-gen", content="a beautiful sunset")
    image_node = _MockImageGenNode(name="image-gen", color=(255, 128, 64))

    pipe = Pipeline("cross-domain", [text_node, image_node])
    result = pipe.execute_result(MosaicData(prompt="sunset over mountains"))

    # 验证管道成功
    assert result.success
    assert result.output is not None

    # 输出应为图像数据
    assert "image" in result.output
    assert isinstance(result.output["image"], Image.Image)
    assert result.output["size"] == (256, 256)

    # 验证中间产物包含两个节点
    assert len(result.intermediate) >= 2
    assert any("text-gen" in name for name in result.intermediate)
    assert any("image-gen" in name for name in result.intermediate)

    # 验证文本节点的中间输出包含生成的内容
    text_intermediate = result.get_intermediate("text-gen")
    assert "a beautiful sunset" in text_intermediate.get("text", "")


# ===========================================================================
# T_E2E_P2_02: 图像域管道：text-to-image → 背景去除
# ===========================================================================
@pytest.mark.integration
def test_image_domain_pipeline(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_02: 图像域管道：text-to-image → 背景去除。

    使用两个 mock 图像节点串联，验证输出从前一个流向后一个。
    """
    t2i = _MockImageGenNode(name="t2i", color=(100, 200, 255))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")

    pipe = Pipeline("image-pipe", [t2i, bg_remover])
    result = pipe.execute_result(MosaicData(content="generate an image"))

    assert result.success
    assert result.output is not None

    # 输出应为处理后的图像（128x128，因为 bg_remover 会 resize）
    assert "image" in result.output
    assert isinstance(result.output["image"], Image.Image)

    # 验证中间产物
    assert len(result.intermediate) >= 2
    t2i_output = result.get_intermediate("t2i")
    assert "image" in t2i_output
    # t2i 输出 256x256
    assert t2i_output["size"] == (256, 256)

    # 最终输出 metadata 应包含 bg-remover 的 tag
    assert result.output["metadata"]["tag"] == "no-bg"


# ===========================================================================
# T_E2E_P2_03: 并行分支：text-to-image → (背景去除 + 风格化)
# ===========================================================================
@pytest.mark.integration
def test_parallel_branch_two_paths(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_03: 并行分支：text-to-image → (背景去除 + 风格化)。

    使用 Branch 创建两条并行路径，验证两条路径都执行。
    """
    t2i = _MockImageGenNode(name="t2i", color=(100, 200, 255))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")
    stylizer = _MockImageProcessorNode(name="stylizer", color=(255, 0, 128), tag="styled")

    pipe = Pipeline("parallel-branch", [
        t2i,
        Branch(
            bg=bg_remover,
            style=stylizer,
        ),
    ])

    result = pipe.execute_result(MosaicData(content="generate an image"))

    assert result.success
    assert result.output is not None

    # 多终点管道：output 以标签聚合，键为 "bg" 和 "style"
    assert "bg" in result.output
    assert "style" in result.output

    # 两条路径的输出都应存在
    bg_out = result.output["bg"]
    style_out = result.output["style"]
    assert isinstance(bg_out, MosaicData)
    assert isinstance(style_out, MosaicData)

    # 验证两个分支的 tag 不同
    assert bg_out["metadata"]["tag"] == "no-bg"
    assert style_out["metadata"]["tag"] == "styled"

    # 验证中间产物包含所有节点
    assert len(result.intermediate) >= 3  # t2i + bg-remover + stylizer
    assert any("bg-remover" in name for name in result.intermediate)
    assert any("stylizer" in name for name in result.intermediate)


# ===========================================================================
# T_E2E_P2_04: 并行分支 + Merge —— 验证结果可访问
# ===========================================================================
@pytest.mark.integration
def test_parallel_branch_with_merge(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_04: 并行分支 + Merge —— 验证结果可访问。

    使用 Branch + Merge，验证输出中可访问两个分支的结果。
    """
    t2i = _MockImageGenNode(name="t2i", color=(100, 200, 255))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")
    stylizer = _MockImageProcessorNode(name="stylizer", color=(255, 0, 128), tag="styled")
    merge = Merge(strategy="dict")

    pipe = Pipeline("branch-merge", [
        t2i,
        Branch(
            bg=bg_remover,
            style=stylizer,
        ),
        merge,
    ])

    result = pipe.execute_result(MosaicData(content="generate an image"))

    assert result.success
    assert result.output is not None

    # Merge 以 dict 策略合并，output 应包含 "bg" 和 "style" 键
    assert "bg" in result.output
    assert "style" in result.output

    # 可分别访问两个分支的结果
    bg_data = result.output["bg"]
    style_data = result.output["style"]
    assert isinstance(bg_data, MosaicData)
    assert isinstance(style_data, MosaicData)
    assert "image" in bg_data
    assert "image" in style_data

    # 验证中间产物包含 merge 节点
    assert any("merge" in name for name in result.intermediate)


# ===========================================================================
# T_E2E_P2_05: 中间产物可保存为文件
# ===========================================================================
@pytest.mark.integration
def test_intermediate_artifact_saved_as_file(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_05: 中间产物可保存为文件。

    运行管道，获取中间图像数据，保存到文件，验证文件存在。
    """
    t2i = _MockImageGenNode(name="t2i", color=(128, 64, 200))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")

    pipe = Pipeline("save-artifact", [t2i, bg_remover])
    result = pipe.execute_result(MosaicData(content="test image"))

    assert result.success

    # 获取中间产物（t2i 节点的输出）
    t2i_output = result.get_intermediate("t2i")
    assert "image" in t2i_output
    img = t2i_output["image"]
    assert isinstance(img, Image.Image)

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path, format="PNG")

    try:
        # 验证文件存在且非空
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # 验证可以重新加载
        reloaded = Image.open(tmp_path)
        assert reloaded.size == (256, 256)
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# T_E2E_P2_06: 事件触发验证（node_start, node_complete）
# ===========================================================================
@pytest.mark.integration
def test_event_triggers_node_start_and_complete(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_06: 事件触发验证（node_start, node_complete）。

    订阅 EventBus，收集事件到列表，验证每个节点都触发了
    node_start 和 node_complete 事件。

    由于 Pipeline 通过 Context 发出事件，本测试通过 Context
    回调将事件转发到 EventBus，再通过 EventBus 订阅收集。
    """
    from mosaic.core.context import Context

    events = []

    def collect_event(event):
        events.append(event)

    # 订阅 EventBus 上的 node_start 和 node_end 事件
    # （Pipeline 实际发出的是 "node_start" 和 "node_end"，
    #  EventType.NODE_COMPLETE = "node_complete" 此处也一并订阅以覆盖 EventType 常量）
    fresh_bus.on(EventType.NODE_START, collect_event)
    fresh_bus.on(EventType.NODE_COMPLETE, collect_event)
    fresh_bus.on("node_end", collect_event)

    # 创建 Context，注册回调将 Context 事件转发到 EventBus
    ctx = Context()

    def forward_to_bus(event):
        """将 Context 的 Event 转发为 EventBus 的 MosaicEvent。"""
        fresh_bus.emit(event.event_type, node_name=event.node_name, **event.payload)

    ctx.on_event(forward_to_bus)

    t2i = _MockImageGenNode(name="t2i", color=(100, 200, 255))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")

    pipe = Pipeline("event-test", [t2i, bg_remover])
    result = pipe.execute_result(MosaicData(content="test image"), context=ctx)

    assert result.success

    # 收集的事件中应包含 node_start 和 node_end
    start_events = [e for e in events if e.event_type == EventType.NODE_START]
    end_events = [e for e in events if e.event_type == "node_end"]

    # 每个节点至少触发一次 start 和 end
    assert len(start_events) >= 2, f"Expected >=2 node_start events, got {len(start_events)}"
    assert len(end_events) >= 2, f"Expected >=2 node_end events, got {len(end_events)}"

    # 验证事件中包含节点名信息
    node_names = {e.payload.get("node_name", "") for e in start_events}
    assert "t2i" in node_names, f"Expected 't2i' in node names, got {node_names}"


# ===========================================================================
# T_E2E_P2_07: PipelineResult 包含正确的摘要信息
# ===========================================================================
@pytest.mark.integration
def test_pipeline_result_summary_info(
    cpu_scheduler,
    fresh_bus,
):
    """T_E2E_P2_07: PipelineResult 包含正确的摘要信息。

    验证 summary() 和 to_dict() 包含正确的管道名、节点数等信息。
    """
    t2i = _MockImageGenNode(name="t2i", color=(100, 200, 255))
    bg_remover = _MockImageProcessorNode(name="bg-remover", color=(0, 255, 0), tag="no-bg")
    stylizer = _MockImageProcessorNode(name="stylizer", color=(255, 0, 128), tag="styled")
    merge = Merge(strategy="dict")

    pipe = Pipeline("summary-test-pipe", [
        t2i,
        Branch(
            bg=bg_remover,
            style=stylizer,
        ),
        merge,
    ])

    result = pipe.execute_result(MosaicData(content="test image"))

    assert result.success

    # 验证 to_dict() 包含正确字段
    d = result.to_dict()
    assert d["pipeline_name"] == "summary-test-pipe"
    assert d["success"] is True
    assert d["node_count"] >= 4  # t2i + bg-remover + stylizer + merge
    assert isinstance(d["duration"], float)
    assert d["duration"] > 0
    assert len(d["errors"]) == 0
    assert "output" in d
    assert "intermediate" in d
    assert "node_durations" in d

    # 验证 summary() 返回字符串且包含关键信息
    summary = result.summary()
    assert isinstance(summary, str)
    assert "summary-test-pipe" in summary
    assert "SUCCESS" in summary
    assert "Duration" in summary or "duration" in summary.lower()
    assert "Nodes executed" in summary or "node" in summary.lower()

    # 验证 node_count 属性
    assert result.node_count >= 4