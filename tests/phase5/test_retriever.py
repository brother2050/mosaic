# tests/phase5/test_retriever.py
"""Retriever 节点测试。

测试 Retriever 的向量检索功能，包括基本检索、参数控制、分数过滤、
结果排序等。使用 conftest 中注入的 mock faiss/sentence_transformers。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.types import MosaicData
from mosaic.nodes.rag.retriever import Retriever


# ===========================================================================
# 辅助：创建已注入 mock 数据的 Retriever
# ===========================================================================
def _create_mock_retriever(cpu_scheduler, fresh_bus, chunks=None, index_type="faiss"):
    """创建 Retriever，加载后注入 mock 内部状态。"""
    retriever = Retriever(
        index_type=index_type,
        bus=fresh_bus,
        scheduler=cpu_scheduler,
    )
    # 创建 mock 嵌入模型
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        return_value=np.random.randn(1, 384).astype(np.float32)
    )
    retriever._model = mock_model

    # 注入 mock 数据
    if chunks is None:
        chunks = [
            {
                "content": "机器学习是人工智能的一个子领域，专注于从数据中学习模式。",
                "source": "doc1.txt",
                "metadata": {"topic": "AI", "page": 1},
            },
            {
                "content": "深度学习使用多层神经网络来处理复杂的数据表示。",
                "source": "doc1.txt",
                "metadata": {"topic": "AI", "page": 2},
            },
            {
                "content": "自然语言处理（NLP）是AI的重要组成部分。",
                "source": "doc2.txt",
                "metadata": {"topic": "NLP", "page": 1},
            },
            {
                "content": "Transformer架构改变了NLP领域的格局。",
                "source": "doc2.txt",
                "metadata": {"topic": "NLP", "page": 2},
            },
            {
                "content": "Python是一种流行的编程语言，广泛用于数据科学。",
                "source": "doc3.txt",
                "metadata": {"topic": "Programming", "page": 1},
            },
        ]

    # 创建 mock FAISS 索引
    mock_index = MagicMock()
    def _search(query_vec, k):
        actual_k = min(k, len(chunks))
        scores = np.array([[0.95, 0.85, 0.75, 0.65, 0.55][:actual_k]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, 4][:actual_k]], dtype=np.int64)
        return scores, indices
    mock_index.search = MagicMock(side_effect=_search)

    retriever._collections = {"default": mock_index}
    retriever._chunk_store = {"default": chunks}
    retriever._loaded = True
    return retriever


# ===========================================================================
# TestRetrieverBasic: T_RET_01 ~ T_RET_08
# ===========================================================================
class TestRetrieverBasic:
    """Retriever 基本检索测试。"""

    def test_basic_retrieval(self, cpu_scheduler, fresh_bus):
        """T_RET_01：基本检索，返回 results 列表。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？"))

        assert "results" in result, "输出应包含 results"
        results = result["results"]
        assert isinstance(results, list), "results 应为 list"
        assert len(results) > 0, "results 不应为空"

    def test_top_k_parameter(self, cpu_scheduler, fresh_bus):
        """T_RET_02：top_k 参数生效（返回数量正确，不超过 top_k）。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=3))
        results = result["results"]
        assert len(results) <= 3, f"返回数量 {len(results)} 不应超过 top_k=3"

    def test_score_threshold_filter(self, cpu_scheduler, fresh_bus):
        """T_RET_03：score_threshold 过滤生效（高分过滤后只保留高分结果）。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=5, score_threshold=0.80))
        results = result["results"]
        for item in results:
            assert item["score"] >= 0.80, f"分数 {item['score']} 应 >= 0.80"

    def test_results_sorted_by_score(self, cpu_scheduler, fresh_bus):
        """T_RET_04：结果按 score 降序排列。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=5))
        results = result["results"]
        if len(results) > 1:
            scores = [item["score"] for item in results]
            assert scores == sorted(scores, reverse=True), f"分数应降序排列，得到 {scores}"

    def test_top_score_is_highest(self, cpu_scheduler, fresh_bus):
        """T_RET_05：top_score 是最高分。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=5))
        results = result["results"]
        top_score = result["top_score"]
        if results:
            max_score = max(item["score"] for item in results)
            assert top_score == max_score, f"top_score={top_score} 应等于最高分={max_score}"

    def test_result_count_consistent(self, cpu_scheduler, fresh_bus):
        """T_RET_06：result_count 与实际结果数量一致。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=5))
        assert result["result_count"] == len(result["results"]), (
            f"result_count={result['result_count']} 应等于 len(results)={len(result['results'])}"
        )

    def test_query_field_echoed(self, cpu_scheduler, fresh_bus):
        """T_RET_07：query 字段正确回传。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        query_text = "什么是深度学习？"
        result = retriever(MosaicData(query=query_text))
        assert result["query"] == query_text, f"query 应回传 '{query_text}'，得到 '{result['query']}'"

    def test_each_result_has_required_fields(self, cpu_scheduler, fresh_bus):
        """T_RET_08：每个结果包含 content、score、source、metadata 四个字段。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(query="什么是机器学习？", top_k=5))
        required_fields = {"content", "score", "source", "metadata"}
        for item in result["results"]:
            for field in required_fields:
                assert field in item, f"结果应包含 '{field}' 字段，item keys={list(item.keys())}"


# ===========================================================================
# TestRetrieverAdvanced: T_RET_09 ~ T_RET_10
# ===========================================================================
class TestRetrieverAdvanced:
    """Retriever 高级功能测试。"""

    def test_filter_metadata(self, cpu_scheduler, fresh_bus):
        """T_RET_09：filter_metadata 元信息过滤。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        result = retriever(MosaicData(
            query="什么是NLP？",
            top_k=5,
            filter_metadata={"topic": "NLP"},
        ))
        results = result["results"]
        for item in results:
            assert item["metadata"].get("topic") == "NLP", (
                f"过滤后 topic 应为 'NLP'，得到 {item['metadata'].get('topic')}"
            )

    def test_collection_name(self, cpu_scheduler, fresh_bus):
        """T_RET_10：collection_name 指定不同集合。"""
        chunk_store2 = [
            {
                "content": "Python适合数据科学和机器学习。",
                "source": "python_doc.txt",
                "metadata": {"topic": "Python", "page": 1},
            },
            {
                "content": "NumPy是Python科学计算的基础库。",
                "source": "python_doc.txt",
                "metadata": {"topic": "Python", "page": 2},
            },
            {
                "content": "Pandas提供DataFrame数据结构。",
                "source": "python_doc.txt",
                "metadata": {"topic": "Python", "page": 3},
            },
        ]

        mock_index2 = MagicMock()
        def _search2(query_vec, k):
            actual_k = min(k, len(chunk_store2))
            scores = np.array([[0.90, 0.80, 0.70][:actual_k]], dtype=np.float32)
            indices = np.array([[0, 1, 2][:actual_k]], dtype=np.int64)
            return scores, indices
        mock_index2.search = MagicMock(side_effect=_search2)

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.random.randn(1, 384).astype(np.float32))

        retriever = Retriever(index_type="faiss", bus=fresh_bus, scheduler=cpu_scheduler)
        retriever._model = mock_model
        retriever._collections = {"default": MagicMock(), "python_docs": mock_index2}
        retriever._chunk_store = {"default": [], "python_docs": chunk_store2}
        retriever._loaded = True

        result = retriever(MosaicData(
            query="Python有哪些库？",
            collection_name="python_docs",
            top_k=3,
        ))
        results = result["results"]
        assert len(results) > 0, "指定 collection 检索应返回结果"
        for item in results:
            assert item["metadata"].get("topic") == "Python", "结果应来自 python_docs collection"


# ===========================================================================
# TestRetrieverErrors
# ===========================================================================
class TestRetrieverErrors:
    """Retriever 错误处理测试。"""

    def test_missing_query_raises_error(self, cpu_scheduler, fresh_bus):
        """缺少 query 时应抛出 ValueError。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        with pytest.raises(ValueError, match="query"):
            retriever(MosaicData())

    def test_nonexistent_collection_raises_error(self, cpu_scheduler, fresh_bus):
        """不存在的 collection 应抛出 ValueError。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        with pytest.raises(ValueError, match="not found"):
            retriever(MosaicData(query="测试", collection_name="nonexistent"))


# ===========================================================================
# TestRetrieverDescribe
# ===========================================================================
class TestRetrieverDescribe:
    """Retriever describe 测试。"""

    def test_describe_returns_correct_info(self, cpu_scheduler, fresh_bus):
        """describe 返回正确的节点信息。"""
        retriever = _create_mock_retriever(cpu_scheduler, fresh_bus)
        spec = retriever.describe()
        assert spec.name == "retriever"
        assert spec.domain == "rag"
        assert "rag_query_result" in spec.output_types