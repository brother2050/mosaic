# mosaic/nodes/rag/document_parser.py
"""DocumentParser 节点 —— 文档解析。

解析各种格式的文档（PDF/DOCX/TXT/MD/HTML/CSV），输出结构化的
``DocumentData``（包含分块文本、文档元信息和逐块元信息）。

设计要点
--------
* 纯工程节点，不涉及 AI 模型推理，不需要 GPU。
* 各格式解析库（pdfplumber / python-docx / beautifulsoup4）惰性导入，
  缺失时给出友好安装提示。
* 分块策略：
    - ``preserve_structure=True``（默认）：按段落/标题分块，尽量不切断句子。
    - ``preserve_structure=False``：按固定字符数滑动窗口分块。
* 每个 chunk 保留来源信息（文件名、页码、段落序号）。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import DocumentData, MosaicData

from mosaic.nodes.rag._base import BaseRagNode

__all__ = ["DocumentParser"]


# 支持的文件格式
_SUPPORTED_FORMATS = {"pdf", "docx", "txt", "md", "html", "htm", "csv"}


@registry.register
class DocumentParser(BaseRagNode):
    """文档解析节点。

    将各种格式的文档解析为结构化的 ``DocumentData``，包含分块文本与
    逐块元信息。

    Parameters
    ----------
    chunk_size:
        分块大小（字符数），默认 ``512``。
    chunk_overlap:
        分块重叠（字符数），默认 ``50``。
    preserve_structure:
        是否保留文档结构（按段落分块），默认 ``True``。
    supported_formats:
        支持的文件格式列表，默认 ``["pdf", "docx", "txt", "md",
        "html", "csv"]``。
    bus:
        事件总线实例。

    Examples
    --------
    >>> parser = DocumentParser(chunk_size=512, chunk_overlap=50)
    >>> result = parser(MosaicData(file_path="/path/to/doc.pdf"))
    >>> result["document"]  # DocumentData
    >>> result["total_chunks"]  # int
    >>> result["file_type"]  # "pdf"
    """

    name: str = "document-parser"
    domain: str = "rag"
    description: str = (
        "Parse documents (PDF/DOCX/TXT/MD/HTML/CSV) into structured "
        "text chunks with metadata for RAG pipelines."
    )
    version: str = "0.1.0"
    input_types: tuple[str, ...] = ("text", "mosaic")
    output_types: tuple[str, ...] = ("document", "mosaic")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        preserve_structure: bool = True,
        supported_formats: list[str] | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            bus=bus,
            **kwargs,
        )
        self._preserve_structure: bool = preserve_structure
        self._supported_formats: set = set(
            f.lower().removeprefix(".") for f in (supported_formats or list(_SUPPORTED_FORMATS))
        )
        self._available_parsers: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载环境：检测各格式解析库可用性。

        不需要加载 AI 模型，仅检测依赖库是否安装。缺失的库会记录警告
        但不阻止节点加载（仅在解析对应格式时报错）。
        """
        self._available_parsers = {
            "txt": True,  # 纯文本无需第三方库
            "md": True,
            "csv": True,
        }

        # 检测 PDF 解析库
        for lib in ("pdfplumber", "PyPDF2"):
            try:
                __import__(lib)
                self._available_parsers["pdf"] = True
                break
            except ImportError:
                pass
        else:
            self._available_parsers["pdf"] = False
            self._logger.warning(
                "PDF parsing libraries not found. "
                "Install via: pip install pdfplumber"
            )

        # 检测 DOCX 解析库
        try:
            __import__("docx")
            self._available_parsers["docx"] = True
        except ImportError:
            self._available_parsers["docx"] = False
            self._logger.warning(
                "DOCX parsing library not found. "
                "Install via: pip install python-docx"
            )

        # 检测 HTML 解析库
        try:
            __import__("bs4")
            self._available_parsers["html"] = True
        except ImportError:
            self._available_parsers["html"] = False
            self._logger.warning(
                "HTML parsing library not found. "
                "Install via: pip install beautifulsoup4"
            )

        self._loaded = True
        self._logger.info("DocumentParser ready. Available parsers: %s", self._available_parsers)

    def unload(self) -> None:
        """释放资源（无持久化资源需要释放）。"""
        self._available_parsers.clear()
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文档解析。

        Parameters
        ----------
        input_data:
            必须包含 ``file_path`` (str) 或 ``file_content`` (str)。
            可选：``metadata_filter`` (dict)。

        Returns
        -------
        MosaicData
            包含 ``document`` (DocumentData)、``total_chunks`` (int)、
            ``total_chars`` (int)、``file_type`` (str)、``metadata`` (dict)。

        Raises
        ------
        ValueError
            缺少 ``file_path`` 且无 ``file_content``，或格式不支持。
        """
        self._emit_start()
        t0 = time.perf_counter()

        try:
            file_content = input_data.get("file_content")
            file_path = input_data.get("file_path")

            # 确定文本来源和文件类型
            if file_content is not None and isinstance(file_content, str):
                text = file_content
                file_type = input_data.get("file_type", "txt")
                metadata = {
                    "filename": input_data.get("filename", "inline_content"),
                    "source": "inline",
                }
            elif isinstance(file_path, str) and file_path:
                # 校验格式支持
                ext = os.path.splitext(file_path)[1].lower().removeprefix(".")
                if ext not in self._supported_formats:
                    raise ValueError(
                        f"Unsupported file format: .{ext}. "
                        f"Supported: {sorted(self._supported_formats)}"
                    )
                # 校验解析库可用
                if not self._available_parsers.get(ext, True):
                    raise ImportError(
                        f"Parser for .{ext} is not available. "
                        f"Please install the required library."
                    )
                text = self._read_file(file_path)
                file_type = ext
                metadata = self._extract_metadata(file_path)
            else:
                raise ValueError(
                    "DocumentParser requires 'file_path' (str) or "
                    "'file_content' (str)."
                )

            # 清洗文本
            text = self._clean_text(text)

            # 分块
            chunks = self._split_text(
                text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                preserve_structure=self._preserve_structure,
            )

            # 构造逐块元信息
            chunk_metadata = self._build_chunk_metadata(
                chunks, metadata, file_type
            )

            # 构造 DocumentData
            document = DocumentData(
                chunks=chunks,
                metadata=metadata,
                chunk_metadata=chunk_metadata,
            )

            total_chars = sum(len(c) for c in chunks)

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                document=document,
                total_chunks=len(chunks),
                total_chars=total_chars,
                file_type=file_type,
                metadata=metadata,
            )

            self._emit_complete(
                duration=elapsed,
                output_summary={
                    "total_chunks": len(chunks),
                    "total_chars": total_chars,
                    "file_type": file_type,
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
            model_info=None,  # 纯工程节点，无模型
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _build_chunk_metadata(
        self,
        chunks: list[str],
        doc_metadata: dict[str, Any],
        file_type: str,
    ) -> list[dict[str, Any]]:
        """为每个 chunk 构造元信息。

        Parameters
        ----------
        chunks:
            分块文本列表。
        doc_metadata:
            文档级元信息。
        file_type:
            文件类型。

        Returns
        -------
        list[dict[str, Any]]
            每个 chunk 的元信息，长度与 ``chunks`` 一致。
        """
        filename = doc_metadata.get("filename", "unknown")
        chunk_meta_list: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "chunk_index": i,
                "source": filename,
                "file_type": file_type,
                "char_count": len(chunk),
                "doc_metadata": doc_metadata,
            }
            chunk_meta_list.append(chunk_meta)

        return chunk_meta_list
