# tests/phase2/test_pipeline_result.py
"""Phase 2 PipelineResult 测试。

覆盖 PipelineResult 的 success、duration、node_durations、summary、
to_dict、failed_nodes 等属性与方法。
"""

from __future__ import annotations

import json

import pytest

from mosaic.core.pipeline import Pipeline
from mosaic.core.types import MosaicData
from mosaic.core.result import PipelineResult, NodeError


# ===========================================================================
# T_RES_01: success == True after successful run
# ===========================================================================
def test_success_true_after_successful_run(MockImageNode):
    """T_RES_01: 成功运行后 result.success 为 True。

    使用 Pipeline 包含 2 个 MockImageNode，执行 execute_result()，
    验证 result.success 为 True。
    """
    n1 = MockImageNode(name="step1", tag="A")
    n2 = MockImageNode(name="step2", tag="B")
    pipe = Pipeline("test-success", [n1, n2])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data)
    assert result.success is True


# ===========================================================================
# T_RES_02: success == False after failed run
# ===========================================================================
def test_success_false_after_failed_run(FailingNode):
    """T_RES_02: 失败运行后 result.success 为 False。

    使用 Pipeline 包含 _FailingNode，执行 execute_result(fail_fast=False)，
    验证 result.success 为 False。
    """
    failing = FailingNode(name="bad-node", error_msg="intentional failure")
    pipe = Pipeline("test-fail", [failing])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data, fail_fast=False)
    assert result.success is False


# ===========================================================================
# T_RES_03: duration > 0 after run
# ===========================================================================
def test_duration_positive_after_run(MockImageNode):
    """T_RES_03: 运行后 result.duration 为正浮点数。"""
    n1 = MockImageNode(name="step1", tag="A")
    n2 = MockImageNode(name="step2", tag="B")
    pipe = Pipeline("test-duration", [n1, n2])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data)
    assert isinstance(result.duration, float)
    assert result.duration > 0.0


# ===========================================================================
# T_RES_04: node_durations contains duration for each node
# ===========================================================================
def test_node_durations_contains_all_nodes(MockImageNode):
    """T_RES_04: node_durations 包含每个节点的耗时。

    验证 dict 的 keys 包含所有节点的 id。
    """
    n1 = MockImageNode(name="step1", tag="A")
    n2 = MockImageNode(name="step2", tag="B")
    pipe = Pipeline("test-node-durations", [n1, n2])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data)
    # 验证 node_durations 包含所有节点的耗时
    assert len(result.node_durations) == 2
    assert "step1" in result.node_durations
    assert "step2" in result.node_durations
    # 每个耗时都应为正浮点数
    for dur in result.node_durations.values():
        assert isinstance(dur, float)
        assert dur > 0.0


# ===========================================================================
# T_RES_05: summary() outputs readable text
# ===========================================================================
def test_summary_outputs_readable_text(MockImageNode):
    """T_RES_05: summary() 输出可读文本。

    验证包含 pipeline name、status、duration 等关键信息。
    """
    n1 = MockImageNode(name="step1", tag="A")
    n2 = MockImageNode(name="step2", tag="B")
    pipe = Pipeline("test-summary", [n1, n2])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data)
    text = result.summary()
    assert isinstance(text, str)
    assert "test-summary" in text
    assert "SUCCESS" in text
    assert "Duration" in text
    assert "Nodes executed" in text


# ===========================================================================
# T_RES_06: to_dict() is serializable
# ===========================================================================
def test_to_dict_is_serializable(MockImageNode):
    """T_RES_06: to_dict() 返回字典且可 JSON 序列化。"""
    n1 = MockImageNode(name="step1", tag="A")
    pipe = Pipeline("test-serial", [n1])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data)
    d = result.to_dict()
    assert isinstance(d, dict)
    # json.dumps 不应抛出异常
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


# ===========================================================================
# T_RES_07: failed_nodes returns correct list when there are errors
# ===========================================================================
def test_failed_nodes_returns_correct_list(FailingNode):
    """T_RES_07: 有错误时 failed_nodes 返回正确的失败节点名列表。"""
    failing = FailingNode(name="bad-node", error_msg="intentional failure")
    pipe = Pipeline("test-failed-nodes", [failing])
    input_data = MosaicData(content="hello")
    result = pipe.execute_result(input_data, fail_fast=False)
    failed = result.failed_nodes
    assert isinstance(failed, list)
    assert "bad-node" in failed
    assert len(failed) == 1