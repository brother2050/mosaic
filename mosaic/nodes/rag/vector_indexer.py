# mosaic/nodes/rag/vector_indexer.py
"""VectorIndexer 节点 —— 向量化索引。

将文本块转换为向量并建立索引，支持 FAISS（内积/L2）和 ChromaDB 两种后端。

设计要点
--------
* 嵌入模型优先使用 ``sentence-transformers`` 库加载，回退到
  ``transformers.AutoModel`` + mean pooling。
* FAISS 模式：使用 ``IndexFlatIP``（内积，需归一化向量）或
  ``IndexFlatL2``（L2 距离）。
* ChromaDB 模式：使用 ``chromadb.PersistentClient`` 或 ``EphemeralClient``。
* 支持增量索引：多次 ``run`` 往同一个 collection 添加。
* 嵌入计算支持批处理，避免一次处理过多导致 OOM。
* 索引持久化：FAISS 用 ``faiss.write_index``，ChromaDB 自带持久化。
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
from mosaic.core.types import DocumentData, MosaicData

from mosaic.nodes.rag._base import BaseRagNode, _EMBEDDING_VRAM

__all__ = ["VectorIndexer"]


@registry.register
class VectorIndexer(BaseRagNode):
    """向量化索引节点。

    将 ``DocumentData`` 中的文本块转换为嵌入向量并建立可检索的索引。

    Parameters
    ----------
    embedding_model:
        嵌入模型标识，默认 ``"sentence-transformers/all-MiniLM-L6-v2"``。
    index_type:
        索引类型，``"faiss"`` 或 ``"chromadb"``，默认 ``"faiss"``。
    index_path:
        索引持久化路径。``None`` 为纯内存索引。
    batch_size:
        嵌入计算批大小，默认 ``32``。
    device:
        计算设备，默认 ``"cuda"``。
    metric:
        FAISS 相似度度量，``"ip"``（内积）或 ``"l2"``，默认 ``"ip"``。

    Examples
    --------
    >>> indexer = VectorIndexer(
    ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ...     index_type="faiss",
    ... )
    >>> indexer.load()
    >>> result = indexer(MosaicData(
    ...     document=doc_data,
    ...     collection_name="my_docs",
    ... ))
    >>> result["indexed_count"]  # int
    >>> result["embedding_dim"]  # int, e.g. 384
    """

    name: str = "vector-indexer"
    domain: str = "rag"
    description: str = (
        "Convert text chunks into embeddings and build a vector index "
        "(FAISS or ChromaDB) for retrieval."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["document", "mosaic"]
    output_types: list[str] = ["mosaic"]

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "faiss",
        index_path: str | None = None,
        batch_size: int = 32,
        device: str = "cuda",
        metric: str = "ip",
        bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._model_name: str = embedding_model
        self._index_type: str = index_type.lower()
        self._index_path: str | None = index_path
        self._batch_size: int = max(1, batch_size)
        self._device: str = device
        self._metric: str = metric.lower()
        self._scheduler: Scheduler = scheduler or get_scheduler()

        # 运行时状态
        self._model: Any = None  # sentence-transformers 模型
        self._embedding_dim: int = 0
        self._collections: dict[str, Any] = {}  # collection_name -> index
        self._chunk_store: dict[str, list[dict[str, Any]]] = {}  # 存储 chunk 文本和元信息

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载嵌入模型并初始化索引。

        * 使用 ``sentence-transformers`` 加载嵌入模型（惰性导入）。
        * 根据 ``index_type`` 初始化 FAISS 或 ChromaDB。
        * 如果 ``index_path`` 指定且已有索引文件，尝试加载。
        """
        self._scheduler.track(self)

        if self._model is not None:
            self._loaded = True
            return

        self._logger.info("Loading embedding model: %s", self._model_name)

        # 加载嵌入模型
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

        # 推断嵌入维度
        self._embedding_dim = self._infer_embedding_dim()

        # 加载已有索引
        if self._index_path and os.path.exists(self._index_path):
            self._load_existing_index()

        self._loaded = True
        self._logger.info(
            "VectorIndexer ready (model=%s, dim=%d, type=%s).",
            self._model_name,
            self._embedding_dim,
            self._index_type,
        )

    def unload(self) -> None:
        """释放嵌入模型并保存索引（如果指定了 ``index_path``）。"""
        # 保存索引
        if self._index_path:
            try:
                self._save_index()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to save index: %s", exc)

        self._model = None
        self._collections.clear()
        self._chunk_store.clear()
        self._embedding_dim = 0
        self._loaded = False
        self._logger.info("VectorIndexer unloaded.")

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行向量化索引。

        Parameters
        ----------
        input_data:
            必须包含 ``document`` (DocumentData)。
            可选：``collection_name`` (str, 默认 "default")、
            ``metadata`` (list[dict])。

        Returns
        -------
        MosaicData
            包含 ``collection_name``、``indexed_count``、``embedding_dim``、
            ``index_type``、``index_path``。

        Raises
        ------
        ValueError
            缺少 ``document`` 或 ``document`` 中无 ``chunks``。
        """
        self._scheduler.ensure_loaded(self)
        self._emit_start()
        t0 = time.perf_counter()

        try:
            document = input_data.get("document")
            if not isinstance(document, DocumentData):
                # 尝试从 MosaicData 中获取 chunks
                chunks = input_data.get("chunks")
                if isinstance(chunks, list) and chunks:
                    document = DocumentData(chunks=chunks)
                else:
                    raise ValueError(
                        "VectorIndexer requires 'document' (DocumentData) "
                        "or 'chunks' (list[str])."
                    )

            chunks = document.chunks
            if not chunks:
                raise ValueError("Document has no chunks to index.")

            collection_name = input_data.get("collection_name", "default")
            extra_metadata = input_data.get("metadata", [])

            # 计算嵌入
            embeddings = self._compute_embeddings(chunks)

            # 获取或创建 collection
            index = self._get_or_create_collection(collection_name)

            # 添加到索引
            self._add_to_index(
                collection_name, index, embeddings, chunks,
                document.chunk_metadata, extra_metadata,
            )

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                collection_name=collection_name,
                indexed_count=len(chunks),
                embedding_dim=self._embedding_dim,
                index_type=self._index_type,
                index_path=self._index_path,
            )

            self._emit_complete(
                duration=elapsed,
                output_summary={
                    "indexed_count": len(chunks),
                    "collection_name": collection_name,
                    "embedding_dim": self._embedding_dim,
                },
            )
            return result

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=self._build_model_info(),
        )

    def _build_model_info(self) -> dict[str, Any]:
        """构造模型信息。"""
        vram = _EMBEDDING_VRAM.get(self._model_name, 1.0)
        return {
            "name": self._model_name,
            "source": "HuggingFace",
            "license": "See model card on HuggingFace",
            "vram_gb": vram,
            "device": self._device,
        }

    # ------------------------------------------------------------------
    # 嵌入计算
    # ------------------------------------------------------------------
    def _compute_embeddings(self, texts: list[str]) -> Any:
        """批量计算文本嵌入向量。

        Parameters
        ----------
        texts:
            文本列表。

        Returns
        -------
        numpy.ndarray
            形状 ``(N, dim)`` 的嵌入矩阵。
        """
        import numpy as np  # type: ignore

        all_embeddings: list[Any] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            if hasattr(self._model, "encode"):
                # sentence-transformers 接口
                emb = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=(self._metric == "ip"),
                )
            else:
                # transformers AutoModel 回退
                emb = self._encode_with_transformers(batch)

            all_embeddings.append(emb)

        return np.vstack(all_embeddings)

    def _encode_with_transformers(self, texts: list[str]) -> Any:
        """使用 transformers AutoModel + mean pooling 计算嵌入。"""
        import numpy as np  # type: ignore
        import torch  # type: ignore

        # tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**encoded)

        # mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # L2 归一化（如果用内积）
        if self._metric == "ip":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _load_transformers_model(self) -> Any:
        """使用 transformers AutoModel 加载嵌入模型（回退方案）。"""
        from transformers import AutoModel, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModel.from_pretrained(self._model_name)
        model.to(self._device)
        model.eval()
        return model

    def _infer_embedding_dim(self) -> int:
        """通过编码一个测试句子推断嵌入维度。"""
        try:
            emb = self._compute_embeddings(["test"])
            return int(emb.shape[-1])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # FAISS / ChromaDB 索引管理
    # ------------------------------------------------------------------
    def _get_or_create_collection(self, name: str) -> Any:
        """获取或创建一个索引 collection。"""
        if name in self._collections:
            return self._collections[name]

        if self._index_type == "faiss":
            index = self._create_faiss_index()
        elif self._index_type == "chromadb":
            index = self._create_chromadb_collection(name)
        else:
            raise ValueError(
                f"Unsupported index_type: {self._index_type!r}. "
                f"Supported: 'faiss', 'chromadb'."
            )

        self._collections[name] = index
        if name not in self._chunk_store:
            self._chunk_store[name] = []
        return index

    def _create_faiss_index(self) -> Any:
        """创建 FAISS 索引。"""
        import faiss  # type: ignore

        if self._embedding_dim <= 0:
            raise RuntimeError("Embedding dimension is 0; model not loaded properly.")

        if self._metric == "ip":
            index = faiss.IndexFlatIP(self._embedding_dim)
        else:
            index = faiss.IndexFlatL2(self._embedding_dim)
        return index

    def _create_chromadb_collection(self, name: str) -> Any:
        """创建 ChromaDB collection。"""
        import chromadb  # type: ignore

        if self._index_path:
            client = chromadb.PersistentClient(path=self._index_path)
        else:
            client = chromadb.EphemeralClient()

        collection = client.get_or_create_collection(name=name)
        return collection

    def _add_to_index(
        self,
        collection_name: str,
        index: Any,
        embeddings: Any,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]],
        extra_metadata: list[dict[str, Any]],
    ) -> None:
        """将嵌入向量和对应的文本/元信息添加到索引。"""
        import numpy as np  # type: ignore

        # 确保 embeddings 是 numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if self._index_type == "faiss":
            # FAISS: 添加向量
            emb_float32 = embeddings.astype(np.float32)
            index.add(emb_float32)

            # 存储对应的文本和元信息
            store = self._chunk_store[collection_name]
            base_id = len(store)
            for i, chunk in enumerate(chunks):
                meta = {}
                if i < len(chunk_metadata):
                    meta.update(chunk_metadata[i])
                if i < len(extra_metadata):
                    meta.update(extra_metadata[i])
                store.append({
                    "content": chunk,
                    "metadata": meta,
                    "source": meta.get("source", "unknown"),
                })

        elif self._index_type == "chromadb":
            # ChromaDB: 添加文档和嵌入
            ids = [f"{collection_name}_{len(self._chunk_store[collection_name]) + i}"
                   for i in range(len(chunks))]
            metadatas = []
            for i, chunk in enumerate(chunks):
                meta = {}
                if i < len(chunk_metadata):
                    meta.update(chunk_metadata[i])
                if i < len(extra_metadata):
                    meta.update(extra_metadata[i])
                metadatas.append(meta)
                self._chunk_store[collection_name].append({
                    "content": chunk,
                    "metadata": meta,
                    "source": meta.get("source", "unknown"),
                })

            index.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

    def _load_existing_index(self) -> None:
        """从 ``index_path`` 加载已有索引。"""
        if self._index_type == "faiss":
            import faiss  # type: ignore

            faiss_path = os.path.join(self._index_path, "faiss.index")
            if os.path.exists(faiss_path):
                index = faiss.read_index(faiss_path)
                self._collections["default"] = index
                self._chunk_store["default"] = []
                self._logger.info("Loaded FAISS index from %s", faiss_path)

        elif self._index_type == "chromadb":
            # ChromaDB PersistentClient 自动加载
            self._logger.info("ChromaDB will load from %s", self._index_path)

    def _save_index(self) -> None:
        """保存索引到 ``index_path``。"""
        if not self._index_path:
            return

        os.makedirs(self._index_path, exist_ok=True)

        if self._index_type == "faiss":
            import faiss  # type: ignore
            import json  # type: ignore

            for name, index in self._collections.items():
                faiss_path = os.path.join(self._index_path, f"{name}.faiss")
                faiss.write_index(index, faiss_path)

                # 保存 chunk store
                store_path = os.path.join(self._index_path, f"{name}_chunks.json")
                with open(store_path, "w", encoding="utf-8") as f:
                    json.dump(self._chunk_store.get(name, []), f, ensure_ascii=False)

            self._logger.info("Saved FAISS index to %s", self._index_path)

        elif self._index_type == "chromadb":
            # ChromaDB PersistentClient 自动持久化
            self._logger.info("ChromaDB index persisted to %s", self._index_path)

    # ------------------------------------------------------------------
    # 公共 API：供 Retriever 节点使用
    # ------------------------------------------------------------------
    def get_collection(self, name: str) -> Any:
        """获取指定名称的索引 collection（供 Retriever 使用）。"""
        return self._collections.get(name)

    def get_chunk_store(self, name: str) -> list[dict[str, Any]]:
        """获取指定 collection 的 chunk 存储（供 Retriever 使用）。"""
        return self._chunk_store.get(name, [])

    @property
    def embedding_dim(self) -> int:
        """嵌入维度。"""
        return self._embedding_dim
