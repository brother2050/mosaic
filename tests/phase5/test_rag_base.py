# tests/phase5/test_rag_base.py
"""Phase 5 RAG 基类静态方法测试。

测试 BaseRagNode 的 _read_file、_split_text、_clean_text、_extract_metadata
等静态工具方法。
"""

from __future__ import annotations

import os

import pytest

from mosaic.nodes.rag._base import BaseRagNode


# ===================================================================
# _read_file 测试
# ===================================================================
class TestReadFile:
    """T_RBASE_01-03, T_RBASE_09：_read_file 读取各种格式文件。"""

    def test_read_txt_file(self, sample_txt_file):
        # T_RBASE_01：_read_file 读取 txt 文件
        content = BaseRagNode._read_file(sample_txt_file)
        assert len(content) > 0, "读取的 txt 内容不应为空"
        assert "Mosaic" in content, "txt 内容应包含 'Mosaic'"
        assert "多模态" in content, "txt 内容应包含 '多模态'"

    def test_read_markdown_file(self, sample_markdown_file):
        # T_RBASE_02：_read_file 读取 md 文件
        content = BaseRagNode._read_file(sample_markdown_file)
        assert len(content) > 0, "读取的 md 内容不应为空"
        assert "# Mosaic 框架介绍" in content, "md 内容应包含标题"
        assert "## 概述" in content, "md 内容应包含二级标题"

    def test_read_csv_file(self, sample_csv_file):
        # T_RBASE_03：_read_file 读取 csv 文件
        content = BaseRagNode._read_file(sample_csv_file)
        assert len(content) > 0, "读取的 csv 内容不应为空"
        assert "name" in content, "csv 内容应包含 'name' 列名"
        assert "Alice" in content, "csv 内容应包含 'Alice' 数据"
        assert "Beijing" in content, "csv 内容应包含 'Beijing' 数据"

    def test_read_file_not_found(self):
        """读取不存在的文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="File not found"):
            BaseRagNode._read_file("/nonexistent/path/file.txt")

    def test_read_unsupported_format(self, tmp_path):
        # T_RBASE_09：_read_file 不支持的格式给出友好错误
        file_path = tmp_path / "test.xyz"
        file_path.write_text("dummy content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file format"):
            BaseRagNode._read_file(str(file_path))

    def test_read_txt_returns_string(self, sample_txt_file):
        """_read_file 读取 txt 返回 str 类型。"""
        content = BaseRagNode._read_file(sample_txt_file)
        assert isinstance(content, str), "_read_file 应返回 str 类型"

    def test_read_md_returns_string(self, sample_markdown_file):
        """_read_file 读取 md 返回 str 类型。"""
        content = BaseRagNode._read_file(sample_markdown_file)
        assert isinstance(content, str), "_read_file 应返回 str 类型"


# ===================================================================
# _split_text 测试
# ===================================================================
class TestSplitText:
    """T_RBASE_04-06：_split_text 分块测试。"""

    def test_split_by_fixed_length(self):
        # T_RBASE_04：_split_text 按固定长度分块
        text = "0123456789" * 20  # 200 字符
        chunk_size = 50
        chunks = BaseRagNode._split_text(
            text, chunk_size=chunk_size, chunk_overlap=0, preserve_structure=False
        )

        assert len(chunks) > 1, f"分块数应大于 1，实际 {len(chunks)}"
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= chunk_size, (
                f"chunk[{i}] 长度 {len(chunk)} 超过 chunk_size {chunk_size}"
            )
            assert len(chunk) > 0, f"chunk[{i}] 不应为空"

    def test_split_no_overlap_reconstruct(self):
        # T_RBASE_05：_split_text 分块结果包含完整原文（可拼接还原）
        text = "0123456789" * 30  # 300 字符
        chunk_size = 60
        chunks = BaseRagNode._split_text(
            text, chunk_size=chunk_size, chunk_overlap=0, preserve_structure=False
        )

        # 无重叠时拼接应还原原文
        reconstructed = "".join(chunks)
        assert reconstructed == text, (
            f"无重叠分块拼接后应与原文一致，"
            f"原文长度 {len(text)}，拼接长度 {len(reconstructed)}"
        )

    def test_split_with_overlap(self):
        # T_RBASE_06：_split_text chunk_overlap 生效（相邻块有重叠）
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 8  # 208 字符
        chunk_size = 30
        chunk_overlap = 10
        chunks = BaseRagNode._split_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_structure=False,
        )

        assert len(chunks) >= 2, f"分块数应至少为 2，实际 {len(chunks)}"

        # 验证相邻块有重叠
        step = chunk_size - chunk_overlap
        for i in range(len(chunks) - 1):
            # 实际重叠量：当前块的末尾与下一块的开头共享的字符数
            actual_overlap = max(0, len(chunks[i]) - step)
            if actual_overlap > 0:
                assert chunks[i][-actual_overlap:] == chunks[i + 1][:actual_overlap], (
                    f"chunk[{i}] 末尾与 chunk[{i+1}] 开头应有重叠，"
                    f"actual_overlap={actual_overlap}"
                )

    def test_split_text_empty(self):
        """空文本分块返回空列表。"""
        chunks = BaseRagNode._split_text("", chunk_size=100, chunk_overlap=0)
        assert chunks == [], "空文本分块应返回空列表"

    def test_split_text_whitespace_only(self):
        """纯空白文本分块返回空列表。"""
        chunks = BaseRagNode._split_text("   \n\n  ", chunk_size=100, chunk_overlap=0)
        assert chunks == [], "纯空白文本分块应返回空列表"

    def test_split_text_shorter_than_chunk(self):
        """文本短于 chunk_size 时返回单元素列表。"""
        text = "短文本"
        chunks = BaseRagNode._split_text(
            text, chunk_size=100, chunk_overlap=0, preserve_structure=False
        )
        assert len(chunks) == 1, f"短文本应返回 1 个 chunk，实际 {len(chunks)}"
        assert chunks[0] == text, "短文本 chunk 内容应与原文一致"

    def test_split_preserve_structure(self):
        """preserve_structure=True 时按段落分块。"""
        text = "段落一。\n\n段落二。\n\n段落三。"
        chunks = BaseRagNode._split_text(
            text, chunk_size=500, chunk_overlap=0, preserve_structure=True
        )
        assert len(chunks) > 0, "按段落分块结果不应为空"

    def test_split_chunk_size_boundary(self):
        """chunk_size 恰好等于文本长度。"""
        text = "0123456789" * 10  # 100 字符
        chunks = BaseRagNode._split_text(
            text, chunk_size=100, chunk_overlap=0, preserve_structure=False
        )
        assert len(chunks) == 1, "chunk_size 等于文本长度时应返回 1 个 chunk"
        assert chunks[0] == text, "chunk 内容应与原文一致"


# ===================================================================
# _clean_text 测试
# ===================================================================
class TestCleanText:
    """T_RBASE_07：_clean_text 清洗文本测试。"""

    def test_clean_extra_spaces(self):
        # T_RBASE_07：_clean_text 去除多余空白
        text = "这是  一段   有  多余  空格  的  文本。"
        cleaned = BaseRagNode._clean_text(text)
        # _clean_text 将连续空格压缩为单个空格
        assert "  " not in cleaned, "清洗后不应有连续两个空格"
        assert "这是" in cleaned and "一段" in cleaned, "清洗后应保留所有词语"
        # 所有连续空格被压缩为单个空格
        assert cleaned.count("  ") == 0, "不应有连续空格"

    def test_clean_extra_newlines(self):
        """去除多余空行。"""
        text = "第一行\n\n\n\n\n第二行\n\n\n第三行"
        cleaned = BaseRagNode._clean_text(text)
        assert "\n\n\n" not in cleaned, "清洗后不应有连续三个换行"
        assert cleaned == "第一行\n\n第二行\n\n第三行", (
            f"清洗结果不匹配：'{cleaned}'"
        )

    def test_clean_strip(self):
        """去除首尾空白。"""
        text = "  \n  内容  \n  "
        cleaned = BaseRagNode._clean_text(text)
        assert cleaned == "内容", f"清洗后首尾不应有空白，实际 '{cleaned}'"

    def test_clean_empty_string(self):
        """空字符串清洗返回空字符串。"""
        assert BaseRagNode._clean_text("") == "", "空字符串清洗应返回空字符串"
        assert BaseRagNode._clean_text("   ") == "", "空白字符串清洗应返回空字符串"

    def test_clean_preserve_single_newline(self):
        """清洗后保留单个换行。"""
        text = "第一行\n第二行"
        cleaned = BaseRagNode._clean_text(text)
        assert "\n" in cleaned, "清洗后应保留单个换行"

    def test_clean_mixed_whitespace(self):
        """混合多余空格和换行。"""
        text = "  段落一  。  \n\n\n\n  段落二  。  "
        cleaned = BaseRagNode._clean_text(text)
        # 连续空格被压缩为单个空格，连续空行被压缩为双换行，首尾空白被去除
        assert "段落一" in cleaned, "清洗后应保留 '段落一'"
        assert "段落二" in cleaned, "清洗后应保留 '段落二'"
        assert "\n\n\n" not in cleaned, "不应有连续三个换行"
        assert cleaned.startswith("段落一"), "开头不应有空白"
        assert cleaned.endswith("。"), "结尾不应有空白"
        assert "  " not in cleaned, "不应有连续两个空格"


# ===================================================================
# _extract_metadata 测试
# ===================================================================
class TestExtractMetadata:
    """T_RBASE_08：_extract_metadata 提取文件信息测试。"""

    def test_extract_metadata(self, sample_txt_file):
        # T_RBASE_08：_extract_metadata 提取文件信息
        meta = BaseRagNode._extract_metadata(sample_txt_file)

        assert "filename" in meta, "metadata 应包含 'filename'"
        assert "file_path" in meta, "metadata 应包含 'file_path'"
        assert "file_size" in meta, "metadata 应包含 'file_size'"
        assert "modified_time" in meta, "metadata 应包含 'modified_time'"
        assert "extension" in meta, "metadata 应包含 'extension'"

        assert meta["filename"] == "sample.txt", (
            f"filename 应为 'sample.txt'，实际 '{meta['filename']}'"
        )
        assert meta["extension"] == "txt", (
            f"extension 应为 'txt'，实际 '{meta['extension']}'"
        )
        assert meta["file_size"] > 0, "file_size 应大于 0"
        assert os.path.isabs(meta["file_path"]), "file_path 应为绝对路径"

    def test_extract_metadata_md(self, sample_markdown_file):
        """提取 md 文件的元信息。"""
        meta = BaseRagNode._extract_metadata(sample_markdown_file)
        assert meta["extension"] == "md", "md 文件的 extension 应为 'md'"
        assert meta["filename"] == "sample.md", "filename 应为 'sample.md'"

    def test_extract_metadata_csv(self, sample_csv_file):
        """提取 csv 文件的元信息。"""
        meta = BaseRagNode._extract_metadata(sample_csv_file)
        assert meta["extension"] == "csv", "csv 文件的 extension 应为 'csv'"
        assert meta["filename"] == "sample.csv", "filename 应为 'sample.csv'"

    def test_extract_metadata_file_not_found(self):
        """不存在的文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            BaseRagNode._extract_metadata("/nonexistent/file.txt")

    def test_extract_metadata_file_size_positive(self, sample_txt_file):
        """file_size 应为正整数。"""
        meta = BaseRagNode._extract_metadata(sample_txt_file)
        assert isinstance(meta["file_size"], int), "file_size 应为 int 类型"
        assert meta["file_size"] > 0, "file_size 应大于 0"