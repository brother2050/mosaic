# tests/phase5/conftest.py
"""Phase 5 测试公共 fixtures。

提供 RAG 域测试所需的 mock 嵌入模型、mock FAISS、mock ChromaDB、
DocumentData 等共用 fixture。全部使用合成数据，不依赖外部文件或真实模型。

关键 mock 注入（session 级别）：
- sentence_transformers.SentenceTransformer -> 返回固定 384 维 numpy 数组
- faiss -> mock IndexFlatIP/IndexFlatL2，支持 add/search
- chromadb -> mock PersistentClient/EphemeralClient，支持 add/query
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.scheduler import Scheduler, get_scheduler, set_scheduler
from mosaic.core.types import DocumentData, MosaicData


# ---------------------------------------------------------------------------
# Mock transformers 注入
# ---------------------------------------------------------------------------
def _inject_mock_transformers():
    """注入 mock transformers 模块。"""
    if "transformers" not in sys.modules:
        tm = types.ModuleType("transformers")
        tm.AutoModelForCausalLM = MagicMock()
        tm.AutoModelForCausalLM.from_pretrained = MagicMock()
        tm.AutoTokenizer = MagicMock()
        tm.AutoTokenizer.from_pretrained = MagicMock()
        tm.AutoModel = MagicMock()
        tm.AutoModel.from_pretrained = MagicMock()
        tm.pipeline = MagicMock()
        sys.modules["transformers"] = tm
    else:
        tm = sys.modules["transformers"]
        if not hasattr(tm, "AutoModelForCausalLM"):
            tm.AutoModelForCausalLM = MagicMock()
            tm.AutoModelForCausalLM.from_pretrained = MagicMock()
        if not hasattr(tm, "AutoTokenizer"):
            tm.AutoTokenizer = MagicMock()
            tm.AutoTokenizer.from_pretrained = MagicMock()
        if not hasattr(tm, "AutoModel"):
            tm.AutoModel = MagicMock()
            tm.AutoModel.from_pretrained = MagicMock()
        if not hasattr(tm, "pipeline"):
            tm.pipeline = MagicMock()


_inject_mock_transformers()


# ---------------------------------------------------------------------------
# Mock torch 注入（session 作用域，与其他 Phase 兼容）
# ---------------------------------------------------------------------------
def _make_mock_tensor(numpy_array):
    """创建 mock torch.Tensor。"""
    mock_t = MagicMock()
    mock_t.cpu.return_value = mock_t
    mock_t.numpy.return_value = numpy_array
    mock_t.shape = numpy_array.shape
    mock_t.ndim = numpy_array.ndim
    return mock_t


@pytest.fixture(scope="session", autouse=True)
def _mock_torch_phase5():
    """注入/补齐 mock torch 模块。"""
    if "torch" not in sys.modules:
        mt = types.ModuleType("torch")
        _ctx = MagicMock()
        _ctx.__enter__ = MagicMock(return_value=None)
        _ctx.__exit__ = MagicMock(return_value=None)
        mt.inference_mode = MagicMock(return_value=_ctx)
        mt.no_grad = MagicMock(return_value=_ctx)
        mt.float16 = "float16"
        mt.float32 = "float32"
        mt.bfloat16 = "bfloat16"
        mt.Generator = MagicMock
        mt.Tensor = MagicMock
        mt.ones_like = MagicMock(return_value=MagicMock())
        mt.ones = MagicMock(return_value=MagicMock())
        mt.tensor = MagicMock(return_value=MagicMock())
        mt.from_numpy = MagicMock(
            side_effect=lambda x: _make_mock_tensor(x)
        )
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        _mcuda.empty_cache = MagicMock()
        mt.cuda = _mcuda
        sys.modules["torch"] = mt
        sys.modules["torch.cuda"] = _mcuda
    yield


# ---------------------------------------------------------------------------
# Mock sentence_transformers 注入
# ---------------------------------------------------------------------------
_EMBEDDING_DIM = 384


def _make_mock_sentence_transformer():
    """创建 mock SentenceTransformer 类。"""

    class _MockSentenceTransformer:
        def __init__(self, model_name_or_path, device="cpu", **kwargs):
            self.model_name = model_name_or_path
            self.device = device

        def encode(self, sentences, convert_to_numpy=True,
                   normalize_embeddings=False, **kwargs):
            if isinstance(sentences, str):
                sentences = [sentences]
            emb = np.random.random((len(sentences), _EMBEDDING_DIM)).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-9)
                emb = emb / norms
            return emb

        def to(self, device):
            self.device = device
            return self

    return _MockSentenceTransformer


def _inject_mock_sentence_transformers():
    """注入 mock sentence_transformers 模块。"""
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = _make_mock_sentence_transformer()
        sys.modules["sentence_transformers"] = st
    else:
        st = sys.modules["sentence_transformers"]
        if not hasattr(st, "SentenceTransformer"):
            st.SentenceTransformer = _make_mock_sentence_transformer()


_inject_mock_sentence_transformers()


# ---------------------------------------------------------------------------
# Mock faiss 注入
# ---------------------------------------------------------------------------
class _MockFaissIndex:
    """Mock FAISS IndexFlatIP/L2，支持 add 和 search。"""

    def __init__(self, dim):
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self.ntotal = 0

    def add(self, vectors):
        vectors = np.array(vectors).astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._vectors.append(vectors)
        self.ntotal += vectors.shape[0]

    def search(self, query, k):
        query = np.array(query).astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        nq = query.shape[0]
        distances = np.zeros((nq, k), dtype=np.float32)
        indices = np.zeros((nq, k), dtype=np.int64)
        for i in range(nq):
            for j in range(min(k, self.ntotal)):
                indices[i, j] = j
                distances[i, j] = 1.0 - j * 0.1
        return distances, indices


class _MockFaissModule:
    IndexFlatIP = _MockFaissIndex
    IndexFlatL2 = _MockFaissIndex

    @staticmethod
    def read_index(path):
        return _MockFaissIndex(384)

    @staticmethod
    def write_index(index, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("mock-faiss-index")


def _inject_mock_faiss():
    """注入 mock faiss 模块。"""
    if "faiss" not in sys.modules:
        fm = types.ModuleType("faiss")
        fm.IndexFlatIP = _MockFaissIndex
        fm.IndexFlatL2 = _MockFaissIndex
        fm.read_index = _MockFaissModule.read_index
        fm.write_index = _MockFaissModule.write_index
        sys.modules["faiss"] = fm
    else:
        fm = sys.modules["faiss"]
        if not hasattr(fm, "IndexFlatIP"):
            fm.IndexFlatIP = _MockFaissIndex
        if not hasattr(fm, "IndexFlatL2"):
            fm.IndexFlatL2 = _MockFaissIndex


_inject_mock_faiss()


# ---------------------------------------------------------------------------
# Mock chromadb 注入
# ---------------------------------------------------------------------------
class _MockChromaCollection:
    """Mock ChromaDB collection，支持 add 和 query。"""

    def __init__(self, name):
        self.name = name
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._embeddings: list[Any] = []
        self._metadatas: list[Dict] = []

    def add(self, ids, documents=None, embeddings=None, metadatas=None, **kwargs):
        self._ids.extend(ids)
        if documents:
            self._documents.extend(documents)
        if embeddings is not None:
            self._embeddings.extend(embeddings)
        if metadatas:
            self._metadatas.extend(metadatas)

    def query(self, query_embeddings=None, n_results=5, **kwargs):
        return {
            "ids": [self._ids[:n_results]],
            "documents": [self._documents[:n_results]],
            "distances": [[0.0] * min(n_results, len(self._ids))],
            "metadatas": [self._metadatas[:n_results]],
        }

    def count(self):
        return len(self._ids)


class _MockChromaClient:
    def __init__(self, **kwargs):
        self._collections: dict[str, _MockChromaCollection] = {}

    def get_or_create_collection(self, name, **kwargs):
        if name not in self._collections:
            self._collections[name] = _MockChromaCollection(name)
        return self._collections[name]

    def get_collection(self, name, **kwargs):
        return self._collections.get(name)


def _inject_mock_chromadb():
    """注入 mock chromadb 模块。"""
    if "chromadb" not in sys.modules:
        cm = types.ModuleType("chromadb")
        cm.PersistentClient = _MockChromaClient
        cm.EphemeralClient = _MockChromaClient
        sys.modules["chromadb"] = cm
    else:
        cm = sys.modules["chromadb"]
        if not hasattr(cm, "PersistentClient"):
            cm.PersistentClient = _MockChromaClient
        if not hasattr(cm, "EphemeralClient"):
            cm.EphemeralClient = _MockChromaClient


_inject_mock_chromadb()


# ---------------------------------------------------------------------------
# 合成文档数据 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_text() -> str:
    """返回一段中文测试文本（约 200 字）。"""
    return (
        "Mosaic 是一个多模态 AI 生成框架，支持文本、图像、音频、视频和字幕等多种模态。\n\n"
        "它提供了统一的节点抽象和管道编排机制，让开发者可以像搭积木一样构建复杂的 AI 工作流。\n\n"
        "每个节点都遵循一致的接口规范：load/unload/run/describe，通过事件总线进行通信。\n\n"
        "框架内置了 39 个节点，涵盖文本生成、图像生成、音频处理、视频编辑、字幕生成等领域。\n\n"
        "所有节点都通过注册表进行管理，支持动态发现和热加载。"
    )


@pytest.fixture
def sample_long_text() -> str:
    """返回一段超过 10000 字的长文本。"""
    base = (
        "Mosaic 是一个多模态 AI 生成框架，支持文本、图像、音频、视频和字幕等多种模态。"
        "它提供了统一的节点抽象和管道编排机制，让开发者可以像搭积木一样构建复杂的 AI 工作流。"
        "每个节点都遵循一致的接口规范：load/unload/run/describe，通过事件总线进行通信。"
        "框架内置了 39 个节点，涵盖文本生成、图像生成、音频处理、视频编辑、字幕生成等领域。"
        "所有节点都通过注册表进行管理，支持动态发现和热加载。"
    )
    # 重复以生成超过 10000 字
    return (base * 80)  # ~200 chars * 80 = ~16000 chars


@pytest.fixture
def sample_md_text() -> str:
    """返回一段 Markdown 格式文本。"""
    return (
        "# Mosaic 框架介绍\n\n"
        "## 概述\n\n"
        "Mosaic 是一个多模态 AI 生成框架。\n\n"
        "## 特性\n\n"
        "- 统一的节点抽象\n"
        "- 管道编排机制\n"
        "- 事件总线通信\n\n"
        "## 架构\n\n"
        "框架采用分层架构设计，包括核心层、节点层和管道层。\n\n"
        "核心层提供基础数据类型、事件系统和注册表。\n\n"
        "节点层提供各种 AI 节点的实现。\n\n"
        "管道层提供工作流编排和调度功能。"
    )


@pytest.fixture
def sample_csv_text() -> str:
    """返回一段 CSV 格式文本。"""
    return (
        "name,age,city\n"
        "Alice,30,Beijing\n"
        "Bob,25,Shanghai\n"
        "Charlie,35,Guangzhou\n"
        "Diana,28,Shenzhen\n"
    )


@pytest.fixture
def sample_document(sample_text) -> DocumentData:
    """创建 DocumentData，包含 5 个分块。"""
    chunks = [
        "Mosaic 是一个多模态 AI 生成框架，支持文本、图像、音频、视频和字幕等多种模态。",
        "它提供了统一的节点抽象和管道编排机制，让开发者可以像搭积木一样构建复杂的 AI 工作流。",
        "每个节点都遵循一致的接口规范：load/unload/run/describe，通过事件总线进行通信。",
        "框架内置了 39 个节点，涵盖文本生成、图像生成、音频处理、视频编辑、字幕生成等领域。",
        "所有节点都通过注册表进行管理，支持动态发现和热加载。",
    ]
    chunk_metadata = [
        {"chunk_index": i, "source": "test.txt", "file_type": "txt", "char_count": len(c)}
        for i, c in enumerate(chunks)
    ]
    return DocumentData(
        chunks=chunks,
        metadata={"filename": "test.txt", "source": "test"},
        chunk_metadata=chunk_metadata,
    )


@pytest.fixture
def sample_chunks() -> list[str]:
    """返回文本分块列表。"""
    return [
        "Mosaic 是一个多模态 AI 生成框架。",
        "它提供了统一的节点抽象和管道编排机制。",
        "每个节点都遵循一致的接口规范。",
    ]


@pytest.fixture
def tmp_dir():
    """临时目录，测试结束后自动清理。"""
    path = tempfile.mkdtemp(prefix="mosaic_phase5_test_")
    yield path
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 文件路径 fixtures（用于 _read_file 测试）
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_txt_file(tmp_path, sample_text):
    """创建临时 txt 文件，内容为 sample_text。"""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_markdown_file(tmp_path, sample_md_text):
    """创建临时 md 文件，内容为 sample_md_text。"""
    file_path = tmp_path / "sample.md"
    file_path.write_text(sample_md_text, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_text):
    """创建临时 csv 文件，内容为 sample_csv_text。"""
    file_path = tmp_path / "sample.csv"
    file_path.write_text(sample_csv_text, encoding="utf-8")
    return str(file_path)


# ---------------------------------------------------------------------------
# 检索结果 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_retrieval_results() -> list[dict[str, Any]]:
    """返回模拟的检索结果列表（5 条，含 content/score/source/metadata）。"""
    return [
        {
            "content": "Mosaic 是一个多模态 AI 生成框架，支持文本、图像、音频、视频等多种模态。",
            "score": 0.95,
            "source": "doc1.txt",
            "metadata": {"page": 1, "paragraph": 1},
        },
        {
            "content": "框架提供了统一的节点抽象和管道编排机制。",
            "score": 0.87,
            "source": "doc1.txt",
            "metadata": {"page": 1, "paragraph": 2},
        },
        {
            "content": "每个节点都遵循一致的接口规范：load/unload/run/describe。",
            "score": 0.82,
            "source": "doc2.txt",
            "metadata": {"page": 3, "paragraph": 1},
        },
        {
            "content": "框架内置了 39 个节点，涵盖文本生成、图像生成、音频处理等领域。",
            "score": 0.76,
            "source": "doc2.txt",
            "metadata": {"page": 5, "paragraph": 3},
        },
        {
            "content": "所有节点都通过注册表进行管理，支持动态发现和热加载。",
            "score": 0.71,
            "source": "doc3.txt",
            "metadata": {"page": 2, "paragraph": 1},
        },
    ]


# ---------------------------------------------------------------------------
# 调度器 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_bus():
    """新鲜的事件总线。"""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


@pytest.fixture
def cpu_scheduler(fresh_bus):
    """CPU 调度器。"""
    sched = Scheduler(bus=fresh_bus, device="cpu")
    set_scheduler(sched)
    return sched