# tests/phase5/test_rag_pipeline.py
"""RAG 管道组合测试。

测试使用 Pipeline 类声明式组装 RAG 流程，包括多次查询、
跨域管道组合、中间产物检查等。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.pipeline import Pipeline
from mosaic.core.types import MosaicData
from mosaic.core.result import PipelineResult
from mosaic.core.node import Node, NodeSpec


# ===========================================================================
# 辅助 mock 节点
# ===========================================================================
class _MockQueryGeneratorNode(Node):
    """模拟查询生成节点：将用户输入转换为查询。"""

    name = "mock-query-gen"
    domain = "text"
    description = "Mock query generator for RAG pipeline tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text", "mosaic"]

    def __init__(self, name="mock-query-gen", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        prompt = input_data.get("prompt", "")
        return MosaicData(query=f"{prompt}", content=prompt)

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockRetrieverNode(Node):
    """模拟检索节点：返回固定检索结果。"""

    name = "mock-retriever"
    domain = "rag"
    description = "Mock retriever for RAG pipeline tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["rag_query_result", "mosaic"]

    def __init__(self, name="mock-retriever", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        query = input_data.get("query", "")
        return MosaicData(
            query=query,
            results=[
                {
                    "content": f"检索结果1: {query}",
                    "score": 0.95,
                    "source": "doc1.txt",
                    "metadata": {"topic": "AI"},
                },
                {
                    "content": f"检索结果2: {query}",
                    "score": 0.85,
                    "source": "doc2.txt",
                    "metadata": {"topic": "NLP"},
                },
            ],
            result_count=2,
            top_score=0.95,
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


class _MockCitationNode(Node):
    """模拟引用生成节点：返回带引用的回答。"""

    name = "mock-citation-gen"
    domain = "rag"
    description = "Mock citation generator for RAG pipeline tests."
    version = "0.1.0"
    input_types = ["rag_query_result", "mosaic"]
    output_types = ["text", "mosaic"]

    def __init__(self, name="mock-citation-gen", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        query = input_data.get("query", "")
        results = input_data.get("results", [])
        return MosaicData(
            answer=f"回答: {query} [1]",
            citations=[
                {
                    "citation_id": 1,
                    "source": results[0]["source"] if results else "unknown",
                    "content": results[0]["content"] if results else "",
                    "score": results[0]["score"] if results else 0.0,
                }
            ],
            query=query,
            sources_used=1,
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
# T_RPIPE_01：用 Pipeline 声明式组装完整 RAG 流程
# ===========================================================================
class TestRAGPipelineAssembly:
    """RAG 管道组装测试。"""

    def test_pipeline_assembly_full_rag(self, cpu_scheduler, fresh_bus):
        """T_RPIPE_01：用 Pipeline 声明式组装完整 RAG 流程。

        使用 mock 节点组装：查询生成 → 检索 → 引用生成。
        """
        query_gen = _MockQueryGeneratorNode()
        retriever = _MockRetrieverNode()
        citation_gen = _MockCitationNode()

        pipe = Pipeline("full-rag", [query_gen, retriever, citation_gen])
        result = pipe.execute_result(MosaicData(prompt="什么是机器学习？"))

        assert result.success, "管道应成功执行"
        assert result.output is not None, "输出不应为空"
        assert "answer" in result.output, "输出应包含 answer"
        assert "citations" in result.output, "输出应包含 citations"

        # 中间产物应包含所有节点
        assert len(result.intermediate) >= 3, f"应至少有 3 个中间产物，得到 {len(result.intermediate)}"


# ===========================================================================
# T_RPIPE_02：RAG 管道支持多次查询（索引一次，查询多次）
# ===========================================================================
class TestRAGPipelineMultiQuery:
    """RAG 管道多次查询测试。"""

    def test_multi_query_same_index(self, cpu_scheduler, fresh_bus):
        """T_RPIPE_02：RAG 管道支持多次查询（索引一次，查询多次）。

        使用 mock 节点：检索 → 引用生成，多次传入不同 query。
        """
        retriever = _MockRetrieverNode()
        citation_gen = _MockCitationNode()

        pipe = Pipeline("multi-query", [retriever, citation_gen])

        # 第一次查询
        result1 = pipe.execute_result(MosaicData(query="什么是机器学习？"))
        assert result1.success, "第一次查询应成功"
        assert "answer" in result1.output, "第一次查询应包含 answer"

        # 第二次查询
        result2 = pipe.execute_result(MosaicData(query="什么是深度学习？"))
        assert result2.success, "第二次查询应成功"
        assert "answer" in result2.output, "第二次查询应包含 answer"

        # 两次查询结果应不同
        assert result1.output["answer"] != result2.output["answer"], "不同查询应有不同结果"


# ===========================================================================
# T_RPIPE_03：RAG 管道与文本域管道组合
# ===========================================================================
class TestRAGPipelineCrossDomain:
    """RAG 管道跨域组合测试。"""

    def test_rag_with_text_domain(self, cpu_scheduler, fresh_bus):
        """T_RPIPE_03：RAG 管道与文本域管道组合（生成查询 → RAG 检索 → 回答）。

        使用 mock 节点：文本查询生成 → RAG 检索 → 引用生成。
        """
        query_gen = _MockQueryGeneratorNode()
        retriever = _MockRetrieverNode()
        citation_gen = _MockCitationNode()

        pipe = Pipeline("text-rag", [query_gen, retriever, citation_gen])
        result = pipe.execute_result(MosaicData(prompt="机器学习的基础概念"))

        assert result.success, "跨域管道应成功执行"
        assert "answer" in result.output, "最终输出应包含 answer"
        assert "citations" in result.output, "最终输出应包含 citations"

        # 验证中间产物
        # 查询生成输出
        query_output = result.get_intermediate("mock-query-gen")
        assert "query" in query_output, "查询生成节点应输出 query"

        # 检索输出
        retriever_output = result.get_intermediate("mock-retriever")
        assert "results" in retriever_output, "检索节点应输出 results"
        assert retriever_output["result_count"] == 2, "检索结果数量应为 2"

        # 引用生成输出
        citation_output = result.get_intermediate("mock-citation-gen")
        assert "answer" in citation_output, "引用生成节点应输出 answer"


# ===========================================================================
# T_RPIPE_04：中间产物可检查
# ===========================================================================
class TestRAGPipelineIntermediates:
    """RAG 管道中间产物测试。"""

    def test_intermediate_artifacts_inspectable(self, cpu_scheduler, fresh_bus):
        """T_RPIPE_04：中间产物可检查（索引数据、检索结果可单独取出）。"""
        query_gen = _MockQueryGeneratorNode()
        retriever = _MockRetrieverNode()
        citation_gen = _MockCitationNode()

        pipe = Pipeline("artifact-check", [query_gen, retriever, citation_gen])
        result = pipe.execute_result(MosaicData(prompt="AI发展历史"))

        assert result.success, "管道应成功执行"

        # 使用 Pipeline.get_intermediate 检查中间产物
        # 检索结果
        retriever_out = result.get_intermediate("mock-retriever")
        assert "results" in retriever_out, "检索中间产物应包含 results"
        assert isinstance(retriever_out["results"], list), "results 应为 list"
        assert len(retriever_out["results"]) == 2, "应有 2 个检索结果"
        assert retriever_out["results"][0]["score"] == 0.95, "第一个结果分数应为 0.95"

        # 引用生成结果
        citation_out = result.get_intermediate("mock-citation-gen")
        assert "citations" in citation_out, "引用生成中间产物应包含 citations"
        assert len(citation_out["citations"]) == 1, "应有 1 个引用"

        # 使用 PipelineResult.to_dict 验证
        d = result.to_dict()
        assert d["pipeline_name"] == "artifact-check", "pipeline_name 应正确"
        assert d["success"] is True, "success 应为 True"
        assert d["node_count"] >= 3, "node_count 应 >= 3"

    def test_pipeline_intermediate_count(self, cpu_scheduler, fresh_bus):
        """验证中间产物数量正确。"""
        retriever = _MockRetrieverNode()
        citation_gen = _MockCitationNode()

        pipe = Pipeline("intermediate-count", [retriever, citation_gen])
        result = pipe.execute_result(MosaicData(query="测试"))

        assert result.success, "管道应成功执行"

        # 中间产物应包含每个已执行节点
        intermediate_names = result.list_intermediate()
        assert len(intermediate_names) >= 2, f"应至少有 2 个中间产物，得到 {len(intermediate_names)}"