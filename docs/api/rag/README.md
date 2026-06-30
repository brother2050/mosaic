# RAG 域 API 文档

本目录包含 `mosaic.nodes.rag` 下所有节点的自动生成 API 文档。

## 节点列表

| 节点 | 说明 |
|------|------|
| `DocumentParser` | 文档解析，支持 PDF/Word/TXT/Markdown |
| `VectorIndexer` | 向量索引，支持 FAISS/ChromaDB |
| `Retriever` | 检索器，语义搜索 + 重排序 |
| `CitationGenerator` | 引用生成，基于检索结果生成带引用的回答 |