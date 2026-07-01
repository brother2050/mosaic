# mosaic/nodes/rag/retriever.py
"""Retriever 节点 —— 向量检索。

根据查询文本从向量索引中检索相关文本块，支持 FAISS 和 ChromaDB 后端。

设计要点
--------
* 查询文本使用与索引时相同的嵌入模型进行向量化。
* FAISS 检索使用 ``index.search``，ChromaDB 使用 ``collection.query``。
* 结果按相似度分数降序排列。
* 支持 ``score_threshold`` 过滤和 ``filter_metadata`` 元信息过滤。
* 可与 :class:`VectorIndexer` 共享内存索引（通过 ``indexer`` 参数注入），
  也可从 ``index_path`` 独立加载。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData, RagQueryResult

from mosaic.nodes._coerce import safe_float, safe_int
from mosaic.nodes.rag._base import BaseRagNode, _EMBEDDING_VRAM

__all__ = ["Retriever"]


@registry.register
class Retriever(BaseRagNode):
    """向量检索节点。

    从向量索引中检索与查询最相关的文本块。

    Parameters
    ----------
    index_type:
        索引类型，``"faiss"`` 或 ``"chromadb"``，默认 ``"faiss"``。
    index_path:
        索引路径。``None`` 时需通过 ``indexer`` 参数注入内存索引。
    embedding_model:
        嵌入模型标识（需与索引时一致）。
    device:
        计算设备，默认 ``"cuda"``。
    metric:
        FAISS 相似度度量，``"ip"`` 或 ``"l2"``，默认 ``"ip"``。
    indexer:
        已加载的 :class:`VectorIndexer` 实例，用于共享内存索引。

    Examples
    --------
    >>> retriever = Retriever(indexer=indexer)
    >>> retriever.load()
    >>> result = retriever(MosaicData(
    ...     query="什么是机器学习？",
    ...     top_k=5,
    ... ))
    >>> result["results"]  # list[dict]
    >>> result["top_score"]  # float
    """

    name: str = "retriever"
    domain: str = "rag"
    description: str = (
        "Retrieve relevant text chunks from a vector index given a query. "
        "Supports FAISS and ChromaDB backends."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["text", "mosaic"]
    output_types: list[str] = ["rag_query_result", "mosaic"]

    def __init__(
        self,
        index_type: str = "faiss",
        index_path: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        metric: str = "ip",
        bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        indexer: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._index_type: str = index_type.lower()
        self._index_path: str | None = index_path
        self._model_name: str = embedding_model
        self._device: str = device
        self._metric: str = metric.lower()
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._indexer: Any = indexer  # VectorIndexer 实例（可选）

        # 运行时状态
        self._model: Any = None  # 嵌入模型
        self._collections: dict[str, Any] = {}
        self._chunk_store: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载嵌入模型和向量索引。

        * 如果注入了 ``indexer``，直接共享其模型和索引。
        * 否则独立加载嵌入模型，并从 ``index_path`` 加载索引。
        """
        self._scheduler.track(self)

        # 优先使用注入的 indexer
        if self._indexer is not None:
            if not self._indexer.is_loaded():
                self._indexer.load()
            self._model = self._indexer._model
            self._collections = self._indexer._collections
            self._chunk_store = self._indexer._chunk_store
            self._loaded = True
            self._logger.info(
                "Retriever using shared indexer (collections=%s).",
                list(self._collections.keys()),
            )
            return

        # 独立加载嵌入模型
        if self._model is not None:
            self._loaded = True
            return

        self._logger.info("Loading embedding model: %s", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        except ImportError:
            self._logger.warning(
                "sentence-transformers not found, falling back to "
                "transformers.AutoModel."
            )
            self._model = self._load_transformers_model()

        # 加载已有索引
        if self._index_path and os.path.exists(self._index_path):
            self._load_existing_index()

        self._loaded = True
        self._logger.info("Retriever ready (model=%s, type=%s).", self._model_name, self._index_type)

    def unload(self) -> None:
        """释放资源。

        如果通过 ``indexer`` 注入共享索引，不释放共享资源（由 indexer 负责）。
        """
        if self._indexer is None:
            # 独立加载模式：释放自己的模型
            self._model = None
            self._collections.clear()
            self._chunk_store.clear()
        # 共享模式：不释放 indexer 的资源
        self._loaded = False
        self._logger.info("Retriever unloaded.")

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行向量检索。

        Parameters
        ----------
        input_data:
            必须包含 ``query`` (str)。
            可选：``collection_name`` (str, 默认 "default")、
            ``top_k`` (int, 默认 5)、``score_threshold`` (float, 默认 0.0)、
            ``filter_metadata`` (dict)。

        Returns
        -------
        MosaicData
            包含 ``query``、``results`` (list[dict])、``result_count``、
            ``top_score``。

        Raises
        ------
        ValueError
            缺少 ``query`` 或索引为空。
        """
        self._scheduler.ensure_loaded(self)
        self._emit_start()
        t0 = time.perf_counter()

        try:
            query = input_data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Retriever requires 'query' (str), got {type(query).__name__}."
                )

            collection_name = input_data.get("collection_name", "default")
            top_k = safe_int(input_data.get("top_k"), "top_k", default=5)
            score_threshold = safe_float(
                input_data.get("score_threshold"), "score_threshold", default=0.0
            )
            filter_metadata = input_data.get("filter_metadata")

            # 计算查询向量
            query_embedding = self._compute_query_embedding(query)

            # 执行检索
            raw_results = self._search(
                collection_name, query_embedding, top_k, filter_metadata
            )

            # 构造结果
            results: list[dict[str, Any]] = []
            for item in raw_results:
                score = item.get("score", 0.0)
                if score >= score_threshold:
                    results.append({
                        "content": item.get("content", ""),
                        "score": float(score),
                        "source": item.get("source", "unknown"),
                        "metadata": item.get("metadata", {}),
                    })

            # 按分数降序
            results.sort(key=lambda x: x["score"], reverse=True)

            top_score = results[0]["score"] if results else 0.0

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                query=query,
                results=results,
                result_count=len(results),
                top_score=top_score,
            )

            self._emit_complete(
                duration=elapsed,
                output_summary={
                    "result_count": len(results),
                    "top_score": top_score,
                    "query": query[:100],
                },
            )
            return result

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        vram = _EMBEDDING_VRAM.get(self._model_name, 1.0)
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info={
                "name": self._model_name,
                "source": "HuggingFace",
                "license": "See model card on HuggingFace",
                "vram_gb": vram,
                "device": self._device,
            },
        )

    # ------------------------------------------------------------------
    # 嵌入计算
    # ------------------------------------------------------------------
    def _compute_query_embedding(self, query: str) -> Any:
        """计算查询文本的嵌入向量。"""
        import numpy as np  # type: ignore

        if hasattr(self._model, "encode"):
            emb = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=(self._metric == "ip"),
            )
            return emb[0]  # (dim,)
        else:
            # transformers AutoModel 回退
            emb = self._encode_with_transformers([query])
            return emb[0]

    def _encode_with_transformers(self, texts: list[str]) -> Any:
        """使用 transformers AutoModel + mean pooling 计算嵌入。"""
        import numpy as np  # type: ignore
        import torch  # type: ignore

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**encoded)

        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        if self._metric == "ip":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _load_transformers_model(self) -> Any:
        """使用 transformers AutoModel 加载嵌入模型。"""
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModel.from_pretrained(self._model_name)
        model.to(self._device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _search(
        self,
        collection_name: str,
        query_embedding: Any,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """执行检索，返回原始结果列表。"""
        import numpy as np  # type: ignore

        if self._index_type == "faiss":
            return self._search_faiss(
                collection_name, query_embedding, top_k, filter_metadata
            )
        elif self._index_type == "chromadb":
            return self._search_chromadb(
                collection_name, query_embedding, top_k, filter_metadata
            )
        else:
            raise ValueError(
                f"Unsupported index_type: {self._index_type!r}. "
                f"Supported: 'faiss', 'chromadb'."
            )

    def _search_faiss(
        self,
        collection_name: str,
        query_embedding: Any,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """FAISS 检索。"""
        import numpy as np  # type: ignore

        index = self._collections.get(collection_name)
        if index is None:
            raise ValueError(
                f"Collection {collection_name!r} not found. "
                f"Available: {list(self._collections.keys())}"
            )

        chunk_store = self._chunk_store.get(collection_name, [])
        if not chunk_store:
            return []

        # 限制 top_k
        actual_k = min(top_k, len(chunk_store))

        # FAISS search
        query_vec = np.array([query_embedding], dtype=np.float32)
        scores, indices = index.search(query_vec, actual_k)

        results: list[dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunk_store):
                continue
            item = chunk_store[idx]
            # 元信息过滤
            if filter_metadata and not self._match_metadata(item.get("metadata", {}), filter_metadata):
                continue
            results.append({
                "content": item["content"],
                "score": float(scores[0][i]),
                "source": item.get("source", "unknown"),
                "metadata": item.get("metadata", {}),
            })

        return results

    def _search_chromadb(
        self,
        collection_name: str,
        query_embedding: Any,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """ChromaDB 检索。"""
        collection = self._collections.get(collection_name)
        if collection is None:
            raise ValueError(
                f"Collection {collection_name!r} not found. "
                f"Available: {list(self._collections.keys())}"
            )

        # ChromaDB where 过滤
        where_clause = filter_metadata if filter_metadata else None

        results_raw = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
        )

        results: list[dict[str, Any]] = []
        documents = results_raw.get("documents", [[]])
        metadatas = results_raw.get("metadatas", [[]])
        distances = results_raw.get("distances", [[]])
        ids = results_raw.get("ids", [[]])

        if not documents or not documents[0]:
            return results

        for i in range(len(documents[0])):
            # ChromaDB L2 距离 → 相似度分数（距离越小越相似，取负作为分数）
            distance = float(distances[0][i]) if i < len(distances[0]) else 0.0
            score = -distance  # 转换为分数：距离越小分数越高
            meta = metadatas[0][i] if i < len(metadatas[0]) else {}
            results.append({
                "content": documents[0][i],
                "score": score,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
            })

        return results

    @staticmethod
    def _match_metadata(
        item_meta: dict[str, Any],
        filter_meta: dict[str, Any],
    ) -> bool:
        """检查 item 元信息是否匹配过滤条件。"""
        for key, value in filter_meta.items():
            if item_meta.get(key) != value:
                return False
        return True

    # ------------------------------------------------------------------
    # 索引加载
    # ------------------------------------------------------------------
    def _load_existing_index(self) -> None:
        """从 ``index_path`` 加载已有索引。"""
        if self._index_type == "faiss":
            import faiss  # type: ignore
            import json  # type: ignore

            # 加载 default collection
            faiss_path = os.path.join(self._index_path, "default.faiss")
            if os.path.exists(faiss_path):
                index = faiss.read_index(faiss_path)
                self._collections["default"] = index

                store_path = os.path.join(self._index_path, "default_chunks.json")
                if os.path.exists(store_path):
                    with open(store_path, "r", encoding="utf-8") as f:
                        self._chunk_store["default"] = json.load(f)

                self._logger.info("Loaded FAISS index from %s", faiss_path)

        elif self._index_type == "chromadb":
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=self._index_path)
            # 获取所有 collections（兼容 ChromaDB 0.4.x 和 0.5.x）
            for coll in client.list_collections():
                name = coll if isinstance(coll, str) else coll.name
                self._collections[name] = client.get_collection(name)
                self._chunk_store[name] = []

            self._logger.info("Loaded ChromaDB from %s", self._index_path)
