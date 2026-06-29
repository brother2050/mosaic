# mosaic/nodes/rag/_base.py
"""RAG 域节点基类。

提取 :class:`DocumentParser`、:class:`VectorIndexer`、:class:`Retriever`
与 :class:`CitationGenerator` 共用的文档处理工具方法。

设计要点
--------
* 所有外部解析库（pdfplumber / python-docx / beautifulsoup4 等）均采用
  惰性导入，使本模块在依赖缺失时仍可被注册表发现与导入。
* 文档分块策略兼顾"按段落分块"（preserve_structure=True）和"固定字符
  数分块"（preserve_structure=False），后者通过滑动窗口实现重叠。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
"""

from __future__ import annotations

import abc
import logging
import os
import re
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

__all__ = ["BaseRagNode"]


# ---------------------------------------------------------------------------
# 常见嵌入模型的粗略显存估算（GB）
# ---------------------------------------------------------------------------
_EMBEDDING_VRAM: dict[str, float] = {
    "sentence-transformers/all-MiniLM-L6-v2": 0.5,
    "sentence-transformers/all-mpnet-base-v2": 1.5,
    "BAAI/bge-large-en-v1.5": 2.0,
    "BAAI/bge-small-en-v1.5": 0.5,
    "BAAI/bge-base-en-v1.5": 1.0,
}


class BaseRagNode(Node):
    """RAG 域节点抽象基类。

    提供文档处理工具方法（文件读取、文本分块、文本清洗、元信息提取）
    以及事件发射辅助。子类只需实现 :meth:`load`/:meth:`unload`/
    :meth:`run`/:meth:`describe`。

    Parameters
    ----------
    chunk_size:
        分块大小（字符数），默认 ``512``。
    chunk_overlap:
        分块重叠（字符数），默认 ``50``。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "rag"
    description: str = "Base RAG node."
    version: str = "0.1.0"
    input_types: list[str] = ["text", "mosaic"]
    output_types: list[str] = ["text", "mosaic"]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._chunk_size: int = max(1, chunk_size)
        self._chunk_overlap: int = max(0, min(chunk_overlap, self._chunk_size - 1))
        self._bus: EventBus = bus or get_event_bus()
        self._scheduler: Scheduler = get_scheduler()
        self._logger = logging.getLogger(f"mosaic.nodes.rag.{self.name}")

    # ------------------------------------------------------------------
    # 文档处理工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _read_file(path: str) -> str:
        """读取文件内容，根据扩展名选择解析器。

        支持 pdf / docx / txt / md / html / csv。各解析库惰性导入，
        缺失时抛出带安装提示的 :class:`ImportError`。

        Parameters
        ----------
        path:
            文件路径。

        Returns
        -------
        str
            解析出的纯文本内容。

        Raises
        ------
        FileNotFoundError
            文件不存在。
        ImportError
            对应格式的解析库未安装。
        ValueError
            不支持的文件格式。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower().removeprefix(".")
        if ext in ("txt", "md", "csv"):
            # 纯文本类直接读取
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        if ext == "pdf":
            return BaseRagNode._read_pdf(path)
        if ext == "docx":
            return BaseRagNode._read_docx(path)
        if ext in ("html", "htm"):
            return BaseRagNode._read_html(path)

        raise ValueError(
            f"Unsupported file format: .{ext}. "
            f"Supported: pdf, docx, txt, md, html, csv."
        )

    @staticmethod
    def _read_pdf(path: str) -> str:
        """使用 pdfplumber 或 PyPDF2 解析 PDF。"""
        # 优先 pdfplumber（表格支持更好）
        try:
            import pdfplumber  # type: ignore

            texts: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    texts.append(page_text)
            return "\n\n".join(texts)
        except ImportError:
            pass

        # 回退 PyPDF2
        try:
            import PyPDF2  # type: ignore

            texts: list[str] = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    texts.append(page.extract_text() or "")
            return "\n\n".join(texts)
        except ImportError:
            raise ImportError(
                "PDF parsing requires 'pdfplumber' or 'PyPDF2'. "
                "Install via: pip install pdfplumber"
            )

    @staticmethod
    def _read_docx(path: str) -> str:
        """使用 python-docx 解析 DOCX，保留段落结构。"""
        try:
            import docx  # type: ignore
        except ImportError:
            raise ImportError(
                "DOCX parsing requires 'python-docx'. "
                "Install via: pip install python-docx"
            )

        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    @staticmethod
    def _read_html(path: str) -> str:
        """使用 beautifulsoup4 提取 HTML 正文。"""
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise ImportError(
                "HTML parsing requires 'beautifulsoup4'. "
                "Install via: pip install beautifulsoup4"
            )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        # 移除 script 和 style 标签
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    @staticmethod
    def _split_text(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        preserve_structure: bool = True,
    ) -> list[str]:
        """将文本分块。

        Parameters
        ----------
        text:
            待分块的文本。
        chunk_size:
            每块最大字符数。
        chunk_overlap:
            相邻块之间的重叠字符数。
        preserve_structure:
            ``True`` 时优先按段落（双换行）分块，尽量不切断句子；
            ``False`` 时按固定字符数滑动窗口分块。

        Returns
        -------
        list[str]
            分块后的文本列表。
        """
        if not text or not text.strip():
            return []

        chunk_size = max(1, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        if preserve_structure:
            return BaseRagNode._split_by_paragraphs(text, chunk_size, chunk_overlap)
        return BaseRagNode._split_by_chars(text, chunk_size, chunk_overlap)

    @staticmethod
    def _split_by_paragraphs(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """按段落分块，尽量不切断句子。

        策略：
        1. 按双换行分割为段落。
        2. 依次累加段落，若累加后超过 ``chunk_size`` 则当前累加结果为一块。
        3. 下一段从当前块的末尾 ``chunk_overlap`` 字符处开始重叠。
        """
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            # 单段就超长：退化为字符分块
            if len(para) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    BaseRagNode._split_by_chars(para, chunk_size, chunk_overlap)
                )
                continue

            if current:
                candidate = current + "\n\n" + para
            else:
                candidate = para

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                chunks.append(current)
                # 重叠：取当前块末尾部分
                if chunk_overlap > 0 and len(current) > chunk_overlap:
                    overlap_text = current[-chunk_overlap:]
                    current = overlap_text + "\n\n" + para
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _split_by_chars(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """按固定字符数滑动窗口分块。"""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        step = chunk_size - chunk_overlap

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += step

        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗文本：去除多余空白、特殊字符等。

        * 连续空格压缩为单个空格。
        * 连续空行压缩为双换行。
        * 去除首尾空白。
        """
        if not text:
            return ""
        # 压缩连续空格（但不破坏换行）
        text = re.sub(r"[^\S\n]+", " ", text)
        # 压缩连续空行（3+ 换行 → 2 换行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_metadata(path: str) -> dict[str, Any]:
        """提取文件元信息。

        Parameters
        ----------
        path:
            文件路径。

        Returns
        -------
        dict[str, Any]
            包含 ``filename``、``file_size``、``modified_time``、
            ``extension`` 等字段。
        """
        stat = os.stat(path)
        return {
            "filename": os.path.basename(path),
            "file_path": os.path.abspath(path),
            "file_size": stat.st_size,
            "modified_time": stat.st_mtime,
            "extension": os.path.splitext(path)[1].lower().removeprefix("."),
        }

    # ------------------------------------------------------------------
    # 事件发射辅助
    # ------------------------------------------------------------------
    def _emit_start(self) -> None:
        """发出 node_start 事件。"""
        self._bus.emit(
            EventType.NODE_START,
            node_name=self.name,
            node_domain=self.domain,
        )

    def _emit_complete(self, duration: float, output_summary: Any) -> None:
        """发出 node_complete 事件。"""
        self._bus.emit(
            EventType.NODE_COMPLETE,
            node_name=self.name,
            duration=duration,
            output_summary=output_summary,
        )

    def _emit_error(self, error: BaseException) -> None:
        """发出 node_error 事件。"""
        self._bus.emit(
            EventType.NODE_ERROR,
            node_name=self.name,
            error=error,
        )

    # ------------------------------------------------------------------
    # Node 抽象方法
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """加载资源（子类实现）。"""

    @abc.abstractmethod
    def unload(self) -> None:
        """释放资源（子类实现）。"""

    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑（子类实现）。"""

    @abc.abstractmethod
    def describe(self) -> NodeSpec:
        """返回节点规格说明（子类实现）。"""

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"domain={self.domain!r} state={status}>"
        )
