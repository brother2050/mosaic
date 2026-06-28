# tests/phase5/test_rag_types.py
"""Phase 5 RAG 数据类型测试。

测试 DocumentData 和 RagQueryResult 的创建、序列化/反序列化与校验。
"""

from __future__ import annotations

import pytest

from mosaic.core.types import DocumentData, RagQueryResult, data_from_dict


# ===================================================================
# DocumentData 测试
# ===================================================================
class TestDocumentDataCreation:
    """T_RAGTYPE_01：DocumentData 创建测试。"""

    def test_create_with_chunks_and_metadata(self):
        # T_RAGTYPE_01：DocumentData 创建，包含 chunks 和 metadata
        chunks = [
            "人工智能是计算机科学的重要分支。",
            "机器学习是人工智能的核心技术。",
            "深度学习通过多层神经网络实现模式识别。",
        ]
        metadata = {"filename": "test.txt", "author": "tester"}
        chunk_metadata = [
            {"source": "test.txt", "page": 1},
            {"source": "test.txt", "page": 1},
            {"source": "test.txt", "page": 2},
        ]

        doc = DocumentData(
            chunks=chunks,
            metadata=metadata,
            chunk_metadata=chunk_metadata,
        )

        assert doc.data_type == "document", "data_type 应为 'document'"
        assert len(doc.chunks) == 3, f"chunks 长度应为 3，实际 {len(doc.chunks)}"
        assert doc.chunks[0] == chunks[0], "chunks[0] 内容不匹配"
        assert doc.metadata["filename"] == "test.txt", "metadata filename 不匹配"
        assert doc.metadata["author"] == "tester", "metadata author 不匹配"
        assert len(doc.chunk_metadata) == 3, (
            f"chunk_metadata 长度应为 3，实际 {len(doc.chunk_metadata)}"
        )
        assert doc.chunk_metadata[0]["source"] == "test.txt", (
            "chunk_metadata[0] source 不匹配"
        )

    def test_create_with_defaults(self):
        """使用默认参数创建 DocumentData。"""
        doc = DocumentData()
        assert doc.chunks == [], "默认 chunks 应为空列表"
        assert doc.metadata == {}, "默认 metadata 应为空字典"
        assert doc.chunk_metadata == [], "默认 chunk_metadata 应为空列表"

    def test_dict_like_access(self, sample_document):
        """DocumentData 支持字典式访问。"""
        doc = sample_document
        assert "chunks" in doc, "应包含 'chunks' 键"
        assert "metadata" in doc, "应包含 'metadata' 键"
        assert "chunk_metadata" in doc, "应包含 'chunk_metadata' 键"
        assert isinstance(doc["chunks"], list), "chunks 应为列表"
        assert isinstance(doc["metadata"], dict), "metadata 应为字典"


class TestDocumentDataSerialization:
    """T_RAGTYPE_02：DocumentData 序列化/反序列化测试。"""

    def test_roundtrip(self, sample_document):
        # T_RAGTYPE_02：DocumentData 序列化/反序列化
        original = sample_document
        d = original.to_dict()

        assert "__data_type__" in d, "序列化后应包含 __data_type__"
        assert d["__data_type__"] == "document", "data_type 应为 'document'"
        assert "chunks" in d, "序列化后应包含 chunks"
        assert "metadata" in d, "序列化后应包含 metadata"

        restored = data_from_dict(d)
        assert isinstance(restored, DocumentData), (
            "反序列化后应为 DocumentData，实际 " + str(type(restored))
        )
        assert restored.chunks == original.chunks, "反序列化后 chunks 不一致"
        assert restored.metadata == original.metadata, "反序列化后 metadata 不一致"
        assert len(restored.chunk_metadata) == len(original.chunk_metadata), (
            "反序列化后 chunk_metadata 长度不一致"
        )

    def test_empty_document_roundtrip(self):
        """空 DocumentData 的序列化/反序列化。"""
        doc = DocumentData()
        d = doc.to_dict()
        restored = data_from_dict(d)
        assert isinstance(restored, DocumentData), "空文档反序列化后应为 DocumentData"
        assert restored.chunks == [], "空文档 chunks 应为空列表"
        assert restored.metadata == {}, "空文档 metadata 应为空字典"


class TestDocumentDataChunkMetadata:
    """T_RAGTYPE_03：chunk_metadata 与 chunks 长度一致性测试。"""

    def test_chunk_metadata_length_matches(self, sample_document):
        # T_RAGTYPE_03：chunk_metadata 与 chunks 长度一致
        doc = sample_document
        assert len(doc.chunk_metadata) == len(doc.chunks), (
            f"chunk_metadata 长度 ({len(doc.chunk_metadata)}) "
            f"应与 chunks 长度 ({len(doc.chunks)}) 一致"
        )

    def test_chunk_metadata_elements_are_dicts(self, sample_document):
        """chunk_metadata 中每个元素应为字典。"""
        doc = sample_document
        for i, cm in enumerate(doc.chunk_metadata):
            assert isinstance(cm, dict), (
                f"chunk_metadata[{i}] 应为 dict，实际 {type(cm)}"
            )

    def test_chunk_metadata_can_be_empty(self):
        """chunk_metadata 可以为空列表。"""
        doc = DocumentData(chunks=["chunk1", "chunk2"], chunk_metadata=[])
        assert doc.chunk_metadata == []
        # 长度可以不一致（例如用户未提供 chunk_metadata）
        assert len(doc.chunks) == 2


class TestDocumentDataEmpty:
    """T_RAGTYPE_04：空文档的 DocumentData 处理。"""

    def test_empty_document(self):
        # T_RAGTYPE_04：空文档的 DocumentData 处理
        doc = DocumentData()
        assert doc.chunks == [], "空文档 chunks 应为空列表"
        assert doc.metadata == {}, "空文档 metadata 应为空字典"
        assert doc.chunk_metadata == [], "空文档 chunk_metadata 应为空列表"

    def test_empty_chunks_with_metadata(self):
        """chunks 为空但 metadata 有内容。"""
        doc = DocumentData(
            chunks=[],
            metadata={"filename": "empty.txt"},
            chunk_metadata=[],
        )
        assert doc.chunks == []
        assert doc.metadata["filename"] == "empty.txt"

    def test_len_zero(self):
        """空文档 len 为 3（chunks, metadata, chunk_metadata）。"""
        doc = DocumentData()
        assert len(doc) == 3, f"空 DocumentData 应有 3 个字段，实际 {len(doc)}"


class TestDocumentDataValidation:
    """DocumentData 校验测试。"""

    def test_validate_valid(self, sample_document):
        """有效的 DocumentData 应通过校验。"""
        assert DocumentData.validate(sample_document), (
            "有效 DocumentData 应通过校验"
        )

    def test_validate_empty(self):
        """空 DocumentData 应通过校验。"""
        doc = DocumentData()
        assert DocumentData.validate(doc), "空 DocumentData 应通过校验"

    def test_validate_non_document(self):
        """非 DocumentData 类型不应通过校验。"""
        from mosaic.core.types import TextData
        text = TextData(content="test")
        assert not DocumentData.validate(text), (
            "TextData 不应通过 DocumentData 校验"
        )

    def test_validate_chunks_not_strings(self):
        """chunks 包含非字符串元素应不通过校验。"""
        doc = DocumentData(chunks=["valid", 123])  # type: ignore
        assert not DocumentData.validate(doc), "包含非字符串 chunk 应不通过校验"


# ===================================================================
# RagQueryResult 测试
# ===================================================================
class TestRagQueryResultCreation:
    """RagQueryResult 创建测试。"""

    def test_create_basic(self):
        """RagQueryResult 基本创建。"""
        result = RagQueryResult(
            query="什么是人工智能？",
            results=[
                {
                    "content": "人工智能是计算机科学的重要分支。",
                    "score": 0.95,
                    "source": "doc1.txt",
                    "metadata": {"page": 1},
                },
                {
                    "content": "机器学习是人工智能的核心技术。",
                    "score": 0.87,
                    "source": "doc1.txt",
                    "metadata": {"page": 2},
                },
            ],
        )

        assert result.data_type == "rag_query_result", (
            "data_type 应为 'rag_query_result'"
        )
        assert result.query == "什么是人工智能？", "query 不匹配"
        assert len(result.results) == 2, f"results 长度应为 2，实际 {len(result.results)}"
        assert result.results[0]["content"] is not None, "results[0].content 不应为 None"
        assert result.results[0]["score"] == 0.95, "results[0].score 不匹配"
        assert result.answer is None, "默认 answer 应为 None"
        assert result.citations is None, "默认 citations 应为 None"

    def test_create_with_answer_and_citations(self):
        """创建带 answer 和 citations 的 RagQueryResult。"""
        result = RagQueryResult(
            query="什么是深度学习？",
            results=[
                {
                    "content": "深度学习通过多层神经网络实现模式识别。",
                    "score": 0.92,
                    "source": "doc2.txt",
                    "metadata": {},
                },
            ],
            answer="深度学习是机器学习的一个子领域，通过多层神经网络学习数据的层次化表示。",
            citations=[
                {
                    "citation_id": "c1",
                    "source": "doc2.txt",
                    "content": "深度学习通过多层神经网络实现模式识别。",
                    "score": 0.92,
                },
            ],
        )

        assert result.answer is not None, "answer 不应为 None"
        assert "深度学习是" in result.answer, "answer 内容不匹配"
        assert result.citations is not None, "citations 不应为 None"
        assert len(result.citations) == 1, "citations 长度应为 1"
        assert result.citations[0]["citation_id"] == "c1", "citation_id 不匹配"

    def test_create_with_defaults(self):
        """使用默认参数创建 RagQueryResult。"""
        result = RagQueryResult()
        assert result.query == "", "默认 query 应为空字符串"
        assert result.results == [], "默认 results 应为空列表"
        assert result.answer is None, "默认 answer 应为 None"


class TestRagQueryResultSerialization:
    """RagQueryResult 序列化/反序列化测试。"""

    def test_roundtrip(self):
        """RagQueryResult 序列化/反序列化。"""
        original = RagQueryResult(
            query="什么是人工智能？",
            results=[
                {
                    "content": "人工智能是计算机科学的重要分支。",
                    "score": 0.95,
                    "source": "doc1.txt",
                    "metadata": {"page": 1},
                },
            ],
            answer="人工智能是计算机科学的重要分支。",
            citations=[
                {
                    "citation_id": "c1",
                    "source": "doc1.txt",
                    "content": "人工智能是计算机科学的重要分支。",
                    "score": 0.95,
                },
            ],
        )

        d = original.to_dict()
        assert "__data_type__" in d, "序列化后应包含 __data_type__"
        assert d["__data_type__"] == "rag_query_result", (
            "data_type 应为 'rag_query_result'"
        )

        restored = data_from_dict(d)
        assert isinstance(restored, RagQueryResult), (
            "反序列化后应为 RagQueryResult，实际 " + str(type(restored))
        )
        assert restored.query == original.query, "反序列化后 query 不一致"
        assert restored.results == original.results, "反序列化后 results 不一致"
        assert restored.answer == original.answer, "反序列化后 answer 不一致"
        assert restored.citations == original.citations, (
            "反序列化后 citations 不一致"
        )

    def test_roundtrip_no_answer(self):
        """无 answer 和 citations 的序列化/反序列化。"""
        original = RagQueryResult(
            query="测试",
            results=[{"content": "test", "score": 0.5, "source": "s", "metadata": {}}],
        )
        d = original.to_dict()
        restored = data_from_dict(d)
        assert restored.answer is None, "answer 应为 None"
        assert restored.citations is None, "citations 应为 None"

    def test_dict_like_access(self):
        """RagQueryResult 支持字典式访问。"""
        result = RagQueryResult(
            query="test",
            results=[{"content": "c", "score": 0.9, "source": "s", "metadata": {}}],
        )
        assert result["query"] == "test", "字典式访问 query 失败"
        assert "results" in result, "应包含 'results' 键"
        assert len(result["results"]) == 1, "results 长度应为 1"


class TestRagQueryResultValidation:
    """RagQueryResult validate 校验测试。"""

    def test_validate_valid(self):
        """有效的 RagQueryResult 应通过校验。"""
        result = RagQueryResult(
            query="test",
            results=[
                {"content": "c", "score": 0.9, "source": "s", "metadata": {}},
            ],
        )
        assert RagQueryResult.validate(result), "有效 RagQueryResult 应通过校验"

    def test_validate_empty_results(self):
        """空 results 应通过校验。"""
        result = RagQueryResult(query="test", results=[])
        assert RagQueryResult.validate(result), "空 results 应通过校验"

    def test_validate_missing_content(self):
        """results 中缺少 content 键应不通过校验。"""
        result = RagQueryResult(
            query="test",
            results=[{"score": 0.9, "source": "s"}],
        )
        assert not RagQueryResult.validate(result), (
            "缺少 content 键应不通过校验"
        )

    def test_validate_missing_score(self):
        """results 中缺少 score 键应不通过校验。"""
        result = RagQueryResult(
            query="test",
            results=[{"content": "c", "source": "s"}],
        )
        assert not RagQueryResult.validate(result), (
            "缺少 score 键应不通过校验"
        )

    def test_validate_non_rag_query_result(self):
        """非 RagQueryResult 类型不应通过校验。"""
        from mosaic.core.types import TextData
        text = TextData(content="test")
        assert not RagQueryResult.validate(text), (
            "TextData 不应通过 RagQueryResult 校验"
        )

    def test_validate_results_not_list(self):
        """results 不是列表不应通过校验。"""
        result = RagQueryResult(query="test")
        result["results"] = "not a list"
        assert not RagQueryResult.validate(result), "results 非列表应不通过校验"

    def test_validate_results_item_not_dict(self):
        """results 中元素不是 dict 不应通过校验。"""
        result = RagQueryResult(query="test", results=["not a dict"])  # type: ignore
        assert not RagQueryResult.validate(result), (
            "results 元素非 dict 应不通过校验"
        )


class TestRagQueryResultWithFixture:
    """使用 sample_retrieval_results fixture 的 RagQueryResult 测试。"""

    def test_create_from_fixture(self, sample_retrieval_results):
        """使用 sample_retrieval_results 创建 RagQueryResult。"""
        result = RagQueryResult(
            query="人工智能有哪些关键技术？",
            results=sample_retrieval_results,
        )
        assert len(result.results) == 5, "results 长度应为 5"
        assert result.results[0]["score"] == 0.95, "最高分应为 0.95"
        assert result.results[0]["source"] == "doc1.txt", "source 不匹配"
        assert RagQueryResult.validate(result), "应通过校验"