# mosaic/nodes/rag/__init__.py
"""RAG 检索增强生成域节点包。

提供完整的 RAG 流水线节点：
    DocumentParser → VectorIndexer → Retriever → CitationGenerator

节点列表
--------
- :class:`DocumentParser`   — 文档解析（PDF/DOCX/TXT/MD/HTML/CSV → DocumentData）
- :class:`VectorIndexer`    — 向量化索引（文本块 → FAISS/ChromaDB 索引）
- :class:`Retriever`        — 向量检索（查询 → 相关文本块）
- :class:`CitationGenerator`— 引用生成（检索结果 + 问题 → 带引用的回答）

示例
--------
    >>> from mosaic.nodes.rag import DocumentParser, VectorIndexer, Retriever, CitationGenerator
    >>> parser = DocumentParser(chunk_size=512)
    >>> indexer = VectorIndexer(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    >>> retriever = Retriever(indexer=indexer)
    >>> generator = CitationGenerator(llm_model="Qwen/Qwen2.5-7B-Instruct")
"""

from __future__ import annotations

from mosaic.nodes.rag._base import BaseRagNode
from mosaic.nodes.rag.citation_generator import CitationGenerator
from mosaic.nodes.rag.document_parser import DocumentParser
from mosaic.nodes.rag.retriever import Retriever
from mosaic.nodes.rag.vector_indexer import VectorIndexer

__all__ = [
    "BaseRagNode",
    "DocumentParser",
    "VectorIndexer",
    "Retriever",
    "CitationGenerator",
]
