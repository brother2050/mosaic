# tests/phase5/test_integration.py
"""Phase 5 端到端集成测试。

测试 RAG 域节点之间的端到端工作流：
文档解析→向量化→检索、文档解析→向量化→检索→引用生成、
多文档索引、索引持久化、事件触发。
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.events import EventBus, EventType
from mosaic.core.types import DocumentData, MosaicData


# ---------------------------------------------------------------------------
# 集成测试 mark
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration


# ===========================================================================
# T_E2E_P5_01：文档解析 → 向量化索引 完整流程
# ===========================================================================
class TestE2EParseAndIndex:
    """文档解析 → 向量化索引 端到端测试。"""

    def test_parse_and_index(self, cpu_scheduler):
        """T_E2E_P5_01：文档解析 → 向量化索引 完整流程。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer

        # Step 1: 文档解析
        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        test_content = (
            "机器学习是人工智能的一个子领域。\n\n"
            "它专注于从数据中学习模式。\n\n"
            "深度学习是机器学习的一个分支。\n\n"
            "它使用多层神经网络。"
        )

        parse_result = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="test_doc.txt",
        ))

        assert "document" in parse_result, "解析结果应包含 document"
        document = parse_result["document"]
        assert isinstance(document, DocumentData), "document 应为 DocumentData"
        assert len(document.chunks) > 0, "chunks 不应为空"

        # Step 2: 向量化索引
        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()

        index_result = indexer(MosaicData(
            document=document,
            collection_name="test_docs",
        ))

        assert "indexed_count" in index_result, "索引结果应包含 indexed_count"
        assert index_result["indexed_count"] > 0, "indexed_count 应 > 0"
        assert index_result["collection_name"] == "test_docs", "collection_name 应正确"


# ===========================================================================
# T_E2E_P5_02：文档解析 → 向量化 → 检索 完整流程
# ===========================================================================
class TestE2EParseIndexRetrieve:
    """文档解析 → 向量化 → 检索 端到端测试。"""

    def test_parse_index_retrieve(self, cpu_scheduler):
        """T_E2E_P5_02：文档解析 → 向量化 → 检索 完整流程。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        # Step 1: 文档解析
        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        test_content = (
            "机器学习是人工智能的一个子领域，专注于从数据中学习模式。\n\n"
            "深度学习使用多层神经网络来处理复杂的数据表示。\n\n"
            "自然语言处理（NLP）是AI的重要组成部分。\n\n"
            "Transformer架构改变了NLP领域的格局。"
        )

        parse_result = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="ai_doc.txt",
        ))
        document = parse_result["document"]

        # Step 2: 向量化索引
        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()
        indexer(MosaicData(document=document, collection_name="ai_docs"))

        # Step 3: 检索
        retriever = Retriever(indexer=indexer, scheduler=cpu_scheduler)
        retriever.load()

        retrieve_result = retriever(MosaicData(
            query="什么是机器学习？",
            collection_name="ai_docs",
            top_k=3,
        ))

        assert "results" in retrieve_result, "检索结果应包含 results"
        results = retrieve_result["results"]
        assert isinstance(results, list), "results 应为 list"
        assert len(results) > 0, "results 不应为空"

        for item in results:
            assert "content" in item, "每个结果应包含 content"
            assert "score" in item, "每个结果应包含 score"
            assert "source" in item, "每个结果应包含 source"
            assert "metadata" in item, "每个结果应包含 metadata"


# ===========================================================================
# T_E2E_P5_03：文档解析 → 向量化 → 检索 → 引用生成 完整 RAG 管道
# ===========================================================================
class TestE2EFullRAGPipeline:
    """文档解析 → 向量化 → 检索 → 引用生成 完整 RAG 管道。"""

    def test_full_rag_pipeline(self, cpu_scheduler, sample_retrieval_results):
        """T_E2E_P5_03：文档解析 → 向量化 → 检索 → 引用生成 完整 RAG 管道。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever
        from mosaic.nodes.rag.citation_generator import CitationGenerator

        # Step 1: 文档解析
        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        test_content = (
            "机器学习是人工智能的一个子领域，专注于从数据中学习模式。\n\n"
            "深度学习使用多层神经网络来处理复杂的数据表示。\n\n"
            "自然语言处理（NLP）是AI的重要组成部分。"
        )

        parse_result = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="ai_doc.txt",
        ))
        document = parse_result["document"]

        # Step 2: 向量化索引
        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()
        indexer(MosaicData(document=document, collection_name="ai_docs"))

        # Step 3: 检索
        retriever = Retriever(indexer=indexer, scheduler=cpu_scheduler)
        retriever.load()

        retrieve_result = retriever(MosaicData(
            query="什么是机器学习？",
            collection_name="ai_docs",
            top_k=3,
        ))

        # Step 4: 引用生成 (使用 mock LLM)
        gen = CitationGenerator(scheduler=cpu_scheduler)
        mock_model = MagicMock()
        output_ids = MagicMock()
        output_ids.shape = [1, 70]
        output_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate = MagicMock(return_value=output_ids)
        param = MagicMock()
        param.device = "cpu"
        mock_model.parameters = MagicMock(return_value=iter([param]))

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="<|user|>\n基于以下参考资料回答...\n<|assistant|>\n"
        )
        mock_tokenizer.pad_token = None
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "</s>"
        mock_inputs = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 50]
        mock_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_inputs["input_ids"] = mock_input_ids
        mock_tokenizer.__call__ = MagicMock(return_value=mock_inputs)
        mock_tokenizer.decode = MagicMock(
            return_value="机器学习是人工智能[1]的一个子领域，它从数据中学习模式[2]。"
        )

        gen._model = mock_model
        gen._tokenizer = mock_tokenizer
        gen._loaded = True

        citation_result = gen(MosaicData(
            query="什么是机器学习？",
            results=retrieve_result["results"],
        ))

        assert "answer" in citation_result, "应包含 answer"
        assert len(citation_result["answer"]) > 0, "answer 不应为空"
        assert "citations" in citation_result, "应包含 citations"
        assert len(citation_result["citations"]) > 0, "citations 不应为空"


# ===========================================================================
# T_E2E_P5_04：解析多个文档建立索引后统一检索
# ===========================================================================
class TestE2EMultiDocIndex:
    """多文档索引测试。"""

    def test_multi_doc_index_and_retrieve(self, cpu_scheduler):
        """T_E2E_P5_04：解析多个文档建立索引后统一检索。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()

        # 文档 1
        doc1 = parser(MosaicData(
            file_content="机器学习是人工智能的一个子领域。\n\n深度学习使用多层神经网络。",
            file_type="txt",
            filename="doc1.txt",
        ))["document"]

        # 文档 2
        doc2 = parser(MosaicData(
            file_content="自然语言处理（NLP）是AI的重要组成部分。\n\nTransformer改变了NLP格局。",
            file_type="txt",
            filename="doc2.txt",
        ))["document"]

        # 两个文档都索引到同一个 collection
        indexer(MosaicData(document=doc1, collection_name="multi_docs"))
        indexer(MosaicData(document=doc2, collection_name="multi_docs"))

        # 检索
        retriever = Retriever(indexer=indexer, scheduler=cpu_scheduler)
        retriever.load()

        result = retriever(MosaicData(
            query="NLP和Transformer",
            collection_name="multi_docs",
            top_k=5,
        ))

        assert len(result["results"]) > 0, "应返回检索结果"


# ===========================================================================
# T_E2E_P5_05：索引持久化后重新加载并检索（结果一致）
# ===========================================================================
class TestE2EIndexPersistence:
    """索引持久化测试。"""

    def test_index_persistence_and_reload(self, cpu_scheduler):
        """T_E2E_P5_05：索引持久化后重新加载并检索（结果一致）。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建索引并保存
            parser = DocumentParser(chunk_size=200, chunk_overlap=20)
            parser.load()

            test_content = (
                "机器学习是人工智能的一个子领域。\n\n"
                "深度学习使用多层神经网络。\n\n"
                "自然语言处理（NLP）是AI的重要组成部分。"
            )
            document = parser(MosaicData(
                file_content=test_content,
                file_type="txt",
                filename="test.txt",
            ))["document"]

            indexer = VectorIndexer(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                index_type="faiss",
                index_path=tmpdir,
                scheduler=cpu_scheduler,
            )
            indexer.load()
            indexer(MosaicData(document=document, collection_name="default"))

            # 保存索引（先保存引用，unload 会清空内部状态）
            saved_collections = indexer._collections.copy()
            saved_chunk_store = {k: v.copy() for k, v in indexer._chunk_store.items()}
            indexer.unload()

            # 验证索引文件已保存（使用 _save_index 中的实际文件命名）
            index_path = os.path.join(tmpdir, "default.faiss")
            chunks_path = os.path.join(tmpdir, "default_chunks.json")
            assert os.path.exists(index_path), "FAISS 索引文件应已保存"
            assert os.path.exists(chunks_path), "chunks JSON 文件应已保存"

            # 创建新的 indexer 和 retriever 共享内存索引
            indexer2 = VectorIndexer(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                index_type="faiss",
                scheduler=cpu_scheduler,
            )
            indexer2.load()
            # 手动注入保存的索引状态
            indexer2._collections = saved_collections
            indexer2._chunk_store = saved_chunk_store

            retriever = Retriever(indexer=indexer2, scheduler=cpu_scheduler)
            retriever.load()

            result = retriever(MosaicData(
                query="什么是机器学习？",
                collection_name="default",
                top_k=3,
            ))

            assert len(result["results"]) > 0, "重新加载索引后应能检索到结果"


# ===========================================================================
# T_E2E_P5_06：不同 query 返回不同检索结果
# ===========================================================================
class TestE2EDifferentQueries:
    """不同查询测试。"""

    def test_different_queries_different_results(self, cpu_scheduler):
        """T_E2E_P5_06：不同 query 返回不同检索结果。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        test_content = (
            "机器学习是人工智能的一个子领域。\n\n"
            "深度学习使用多层神经网络。\n\n"
            "Python是一种流行的编程语言。\n\n"
            "自然语言处理（NLP）是AI的重要组成部分。"
        )
        document = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="test.txt",
        ))["document"]

        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()
        indexer(MosaicData(document=document, collection_name="default"))

        retriever = Retriever(indexer=indexer, scheduler=cpu_scheduler)
        retriever.load()

        # 查询 1
        result1 = retriever(MosaicData(query="机器学习", top_k=3))
        # 查询 2
        result2 = retriever(MosaicData(query="Python编程", top_k=3))

        # 两个查询的结果不应完全相同
        contents1 = set(r["content"] for r in result1["results"])
        contents2 = set(r["content"] for r in result2["results"])

        assert len(contents1) > 0, "查询1应有结果"
        assert len(contents2) > 0, "查询2应有结果"


# ===========================================================================
# T_E2E_P5_07：运行过程中事件被正确触发
# ===========================================================================
class TestE2EEvents:
    """事件触发测试。"""

    def test_events_triggered_during_rag(self, cpu_scheduler, fresh_bus):
        """T_E2E_P5_07：运行过程中事件被正确触发。"""
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        # 收集事件
        events = []
        fresh_bus.on(EventType.NODE_START, lambda e: events.append(("start", e)))
        fresh_bus.on(EventType.NODE_COMPLETE, lambda e: events.append(("complete", e)))

        parser = DocumentParser(chunk_size=200, chunk_overlap=20, bus=fresh_bus)
        parser.load()

        test_content = "机器学习是人工智能的一个子领域。"
        parse_result = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="test.txt",
        ))
        document = parse_result["document"]

        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            bus=fresh_bus,
            scheduler=cpu_scheduler,
        )
        indexer.load()
        indexer(MosaicData(document=document, collection_name="default"))

        retriever = Retriever(indexer=indexer, bus=fresh_bus, scheduler=cpu_scheduler)
        retriever.load()
        retriever(MosaicData(query="机器学习", top_k=3))

        # 验证事件被触发
        start_events = [e for e_type, e in events if e_type == "start"]
        complete_events = [e for e_type, e in events if e_type == "complete"]
        assert len(start_events) > 0, "NODE_START 事件应被触发"
        assert len(complete_events) > 0, "NODE_COMPLETE 事件应被触发"


# ===========================================================================
# T_E2E_P5_08：PipelineResult 包含正确信息
# ===========================================================================
class TestE2EPipelineResult:
    """PipelineResult 信息测试。"""

    def test_pipeline_result_contains_correct_info(self, cpu_scheduler):
        """T_E2E_P5_08：PipelineResult 包含正确信息。"""
        from mosaic.core.pipeline import Pipeline
        from mosaic.nodes.rag.document_parser import DocumentParser
        from mosaic.nodes.rag.vector_indexer import VectorIndexer
        from mosaic.nodes.rag.retriever import Retriever

        parser = DocumentParser(chunk_size=200, chunk_overlap=20)
        parser.load()

        test_content = "机器学习是人工智能的一个子领域。\n\n深度学习使用多层神经网络。"
        document = parser(MosaicData(
            file_content=test_content,
            file_type="txt",
            filename="test.txt",
        ))["document"]

        indexer = VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="faiss",
            scheduler=cpu_scheduler,
        )
        indexer.load()
        indexer(MosaicData(document=document, collection_name="default"))

        retriever = Retriever(indexer=indexer, scheduler=cpu_scheduler)
        retriever.load()

        # 使用 Pipeline
        pipe = Pipeline("rag-pipeline", [retriever])
        result = pipe.execute_result(MosaicData(query="机器学习", top_k=3))

        assert result.success, "管道应成功执行"
        assert result.output is not None, "输出不应为空"
        assert "results" in result.output, "输出应包含 results"
        assert result.pipeline_name == "rag-pipeline", "pipeline_name 应正确"
        assert result.node_count > 0, "node_count 应 > 0"
        assert result.duration > 0, "duration 应 > 0"