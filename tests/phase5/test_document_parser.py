# tests/phase5/test_document_parser.py
"""DocumentParser 节点单元测试。

测试覆盖：
- 不同文件格式解析（txt/md/csv）
- 分块策略（preserve_structure=True/False）
- chunk_size 参数生效
- file_content 直接传入
- chunk 元信息保留
- 大文件分块
- describe 描述信息

重要：DocumentParser 不需要 GPU，load() 只检查解析库可用性。
测试方法：mock _read_file 和 _extract_metadata 避免真实文件依赖，
但保留 load() 的真实行为。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.events import EventBus, get_event_bus
from mosaic.core.types import DocumentData, MosaicData
from mosaic.nodes.rag.document_parser import DocumentParser


# ---------------------------------------------------------------------------
# 辅助：mock _read_file + _extract_metadata
# ---------------------------------------------------------------------------
from contextlib import contextmanager


def _fake_metadata_for(filename, text):
    return {
        "filename": filename,
        "file_path": f"/fake/path/{filename}",
        "file_size": len(text),
        "modified_time": 1234567890.0,
        "extension": filename.split(".")[-1] if "." in filename else "txt",
    }


@contextmanager
def _mock_file_ops(parser, text, filename="test.txt"):
    """同时 mock _read_file 和 _extract_metadata。

    用法：
        with _mock_file_ops(parser, sample_text) as mocks:
            result = parser(MosaicData(file_path="/fake/path/test.txt"))
        mocks["_read_file"].assert_called_once()
    """
    fake_meta = _fake_metadata_for(filename, text)
    mock_read = MagicMock(return_value=text)
    mock_extract = MagicMock(return_value=fake_meta)
    with patch.object(parser, "_read_file", mock_read), \
         patch.object(parser, "_extract_metadata", mock_extract):
        yield {"_read_file": mock_read, "_extract_metadata": mock_extract}


# ---------------------------------------------------------------------------
# T_PARSER_01：解析 txt 文件，输出 DocumentData
# ---------------------------------------------------------------------------
def test_parse_txt_file_outputs_document_data(sample_text, fresh_bus):
    """# T_PARSER_01：解析 txt 文件，输出 DocumentData"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_text) as mocks:
        result = parser(MosaicData(file_path="/fake/path/test.txt"))

    assert "document" in result, "result should contain 'document' key"
    document = result["document"]
    assert isinstance(document, DocumentData), (
        f"Expected DocumentData, got {type(document).__name__}"
    )
    assert len(document.chunks) > 0, "document should have at least one chunk"
    mocks["_read_file"].assert_called_once_with("/fake/path/test.txt")


# ---------------------------------------------------------------------------
# T_PARSER_02：解析 md 文件，保留标题结构
# ---------------------------------------------------------------------------
def test_parse_md_file_preserves_header_structure(sample_md_text, fresh_bus):
    """# T_PARSER_02：解析 md 文件，保留标题结构"""
    parser = DocumentParser(preserve_structure=True, bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_md_text, filename="document.md"):
        result = parser(MosaicData(file_path="/fake/path/document.md"))

    document = result["document"]
    all_text = " ".join(document.chunks)
    assert "# Mosaic" in all_text, "Markdown heading should be preserved in chunks"
    assert "## 概述" in all_text, "Second-level heading should be preserved"
    assert "## 特性" in all_text, "Third-level heading should be preserved"
    assert result["file_type"] == "md", f"Expected file_type 'md', got {result['file_type']!r}"


# ---------------------------------------------------------------------------
# T_PARSER_03：解析 csv 文件
# ---------------------------------------------------------------------------
def test_parse_csv_file(sample_csv_text, fresh_bus):
    """# T_PARSER_03：解析 csv 文件"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_csv_text, filename="data.csv"):
        result = parser(MosaicData(file_path="/fake/path/data.csv"))

    document = result["document"]
    assert isinstance(document, DocumentData)
    assert result["file_type"] == "csv", f"Expected file_type 'csv', got {result['file_type']!r}"
    all_text = " ".join(document.chunks)
    assert "name" in all_text, "CSV header should be in chunks"
    assert "Alice" in all_text, "CSV data should be in chunks"


# ---------------------------------------------------------------------------
# T_PARSER_04：total_chunks > 0
# ---------------------------------------------------------------------------
def test_total_chunks_positive(sample_text, fresh_bus):
    """# T_PARSER_04：total_chunks > 0"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_text):
        result = parser(MosaicData(file_path="/fake/path/test.txt"))

    assert result["total_chunks"] > 0, (
        f"total_chunks should be > 0, got {result['total_chunks']}"
    )
    assert isinstance(result["total_chunks"], int), "total_chunks should be an integer"


# ---------------------------------------------------------------------------
# T_PARSER_05：total_chars 与原文长度一致（允许清洗导致的微小差异）
# ---------------------------------------------------------------------------
def test_total_chars_matches_original_length(sample_text, fresh_bus):
    """# T_PARSER_05：total_chars 与原文长度一致（允许清洗导致的微小差异）"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_text):
        result = parser(MosaicData(file_path="/fake/path/test.txt"))

    total_chars = result["total_chars"]
    original_len = len(sample_text)

    assert total_chars > 0, "total_chars should be positive"
    assert total_chars <= original_len, (
        f"total_chars ({total_chars}) should not exceed original length ({original_len})"
    )
    ratio = total_chars / original_len if original_len > 0 else 1.0
    assert ratio > 0.7, (
        f"total_chars ({total_chars}) is too far from original ({original_len}), "
        f"ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# T_PARSER_06：chunk_size 参数生效（较小 chunk_size 产生更多 chunk）
# ---------------------------------------------------------------------------
def test_smaller_chunk_size_produces_more_chunks(sample_text, fresh_bus):
    """# T_PARSER_06：chunk_size 参数生效（较小 chunk_size 产生更多 chunk）"""
    parser_large = DocumentParser(chunk_size=500, chunk_overlap=0, preserve_structure=False, bus=fresh_bus)
    parser_small = DocumentParser(chunk_size=50, chunk_overlap=0, preserve_structure=False, bus=fresh_bus)
    parser_large.load()
    parser_small.load()

    with _mock_file_ops(parser_large, sample_text):
        result_large = parser_large(MosaicData(file_path="/fake/path/test.txt"))
    with _mock_file_ops(parser_small, sample_text):
        result_small = parser_small(MosaicData(file_path="/fake/path/test.txt"))

    assert result_small["total_chunks"] > result_large["total_chunks"], (
        f"Smaller chunk_size (50) should produce more chunks than larger (500). "
        f"Got {result_small['total_chunks']} vs {result_large['total_chunks']}"
    )


# ---------------------------------------------------------------------------
# T_PARSER_07：preserve_structure=True 时按段落分块
# ---------------------------------------------------------------------------
def test_preserve_structure_true_split_by_paragraphs(sample_text, fresh_bus):
    """# T_PARSER_07：preserve_structure=True 时按段落分块"""
    parser = DocumentParser(chunk_size=500, chunk_overlap=0, preserve_structure=True, bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_text):
        result = parser(MosaicData(file_path="/fake/path/test.txt"))

    chunks = result["document"].chunks
    assert len(chunks) > 0, "Should have at least one chunk"


# ---------------------------------------------------------------------------
# T_PARSER_08：preserve_structure=False 时按固定长度分块
# ---------------------------------------------------------------------------
def test_preserve_structure_false_split_by_fixed_length(sample_text, fresh_bus):
    """# T_PARSER_08：preserve_structure=False 时按固定长度分块"""
    chunk_size = 100
    parser = DocumentParser(
        chunk_size=chunk_size, chunk_overlap=0,
        preserve_structure=False, bus=fresh_bus,
    )
    parser.load()

    with _mock_file_ops(parser, sample_text):
        result = parser(MosaicData(file_path="/fake/path/test.txt"))

    chunks = result["document"].chunks
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= chunk_size, (
            f"Chunk {i} length ({len(chunk)}) exceeds chunk_size ({chunk_size})"
        )


# ---------------------------------------------------------------------------
# T_PARSER_09：file_content 直接传入文本（跳过文件读取）
# ---------------------------------------------------------------------------
def test_file_content_directly_passed(sample_text, fresh_bus):
    """# T_PARSER_09：file_content 直接传入文本（跳过文件读取）"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    result = parser(MosaicData(file_content=sample_text, file_type="txt"))

    assert "document" in result
    document = result["document"]
    assert isinstance(document, DocumentData)
    assert len(document.chunks) > 0, "Should have chunks from file_content"
    assert result["file_type"] == "txt", f"Expected file_type 'txt', got {result['file_type']!r}"


# ---------------------------------------------------------------------------
# T_PARSER_10：每个 chunk 保留来源元信息
# ---------------------------------------------------------------------------
def test_each_chunk_preserves_source_metadata(sample_text, fresh_bus):
    """# T_PARSER_10：每个 chunk 保留来源元信息（chunk_index、source、file_type）"""
    parser = DocumentParser(bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_text, filename="important.txt"):
        result = parser(MosaicData(file_path="/fake/path/important.txt"))

    document = result["document"]
    chunk_metadata = document.chunk_metadata

    assert len(chunk_metadata) == len(document.chunks), (
        f"chunk_metadata length ({len(chunk_metadata)}) should match "
        f"chunks length ({len(document.chunks)})"
    )

    for i, meta in enumerate(chunk_metadata):
        assert "chunk_index" in meta, f"Chunk {i} metadata missing 'chunk_index'"
        assert meta["chunk_index"] == i, f"Chunk {i} has wrong chunk_index: {meta['chunk_index']}"
        assert "source" in meta, f"Chunk {i} metadata missing 'source'"
        assert "important.txt" in meta["source"], (
            f"Chunk {i} source should contain filename, got {meta['source']!r}"
        )
        assert "file_type" in meta, f"Chunk {i} metadata missing 'file_type'"
        assert meta["file_type"] == "txt", (
            f"Chunk {i} file_type should be 'txt', got {meta['file_type']!r}"
        )
        assert "char_count" in meta, f"Chunk {i} metadata missing 'char_count'"
        assert meta["char_count"] > 0, f"Chunk {i} char_count should be > 0"


# ---------------------------------------------------------------------------
# T_PARSER_11：大文件（>10000 字）正确分块
# ---------------------------------------------------------------------------
def test_large_file_correctly_chunked(sample_long_text, fresh_bus):
    """# T_PARSER_11：大文件（>10000 字）正确分块"""
    assert len(sample_long_text) > 10000, (
        f"Test requires text > 10000 chars, got {len(sample_long_text)}"
    )

    parser = DocumentParser(chunk_size=512, chunk_overlap=50, preserve_structure=False, bus=fresh_bus)
    parser.load()

    with _mock_file_ops(parser, sample_long_text, filename="large.txt"):
        result = parser(MosaicData(file_path="/fake/path/large.txt"))

    document = result["document"]
    assert len(document.chunks) > 10, (
        f"Large text should produce many chunks, got {len(document.chunks)}"
    )
    assert result["total_chunks"] > 0
    assert result["total_chars"] > 0


# ---------------------------------------------------------------------------
# T_PARSER_12：describe 返回正确信息
# ---------------------------------------------------------------------------
def test_describe_returns_correct_info(fresh_bus):
    """# T_PARSER_12：describe 返回正确信息"""
    parser = DocumentParser(bus=fresh_bus)
    spec = parser.describe()

    assert spec.name == "document-parser", f"Expected name 'document-parser', got {spec.name!r}"
    assert spec.domain == "rag", f"Expected domain 'rag', got {spec.domain!r}"
    assert "document" in spec.output_types, (
        f"Expected 'document' in output_types, got {spec.output_types}"
    )
    assert len(spec.description) > 0, "description should not be empty"
    assert spec.version == "0.1.0", f"Expected version '0.1.0', got {spec.version!r}"
    assert spec.model_info is None, (
        f"DocumentParser should have no model_info, got {spec.model_info!r}"
    )