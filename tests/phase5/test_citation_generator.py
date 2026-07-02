# tests/phase5/test_citation_generator.py
"""CitationGenerator 节点测试。

测试基于检索结果的引用生成功能，包括基本引用生成、不同引用风格、
语言参数、空结果处理等。使用 mock LLM 模型，不依赖真实模型。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.types import MosaicData
from mosaic.nodes.rag.citation_generator import CitationGenerator


# ===========================================================================
# 辅助：创建已注入 mock 状态的 CitationGenerator
# ===========================================================================
def _create_mock_tokenizer():
    """创建 mock tokenizer。"""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(
        return_value="<|user|>\n基于以下参考资料回答...\n<|assistant|>\n"
    )
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"

    mock_inputs = MagicMock()
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 50]
    mock_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
    mock_inputs["input_ids"] = mock_input_ids
    tokenizer.__call__ = MagicMock(return_value=mock_inputs)
    tokenizer.decode = MagicMock(
        return_value="机器学习是人工智能[1]的一个子领域，它从数据中学习模式[2]。"
    )
    return tokenizer


def _create_mock_model():
    """创建 mock LLM model。"""
    model = MagicMock()
    output_ids = MagicMock()
    output_ids.shape = [1, 70]
    output_ids.__getitem__ = MagicMock(return_value=MagicMock())
    model.generate = MagicMock(return_value=output_ids)
    param = MagicMock()
    param.device = "cpu"
    model.parameters = MagicMock(return_value=iter([param]))
    return model


def _create_mock_generator(cpu_scheduler, fresh_bus, model=None, tokenizer=None, **kwargs):
    """创建 CitationGenerator 并直接注入 mock LLM 状态，绕过 load()。"""
    gen = CitationGenerator(
        bus=fresh_bus,
        scheduler=cpu_scheduler,
        **kwargs,
    )
    if model is None:
        model = _create_mock_model()
    if tokenizer is None:
        tokenizer = _create_mock_tokenizer()
    gen._model = model
    gen._tokenizer = tokenizer
    gen._loaded = True
    return gen


# ===========================================================================
# TestCitationGeneratorBasic: T_CITE_01 ~ T_CITE_07
# ===========================================================================
class TestCitationGeneratorBasic:
    """CitationGenerator 基本功能测试。"""

    def test_basic_citation_generation(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_01：基本引用生成，输出 answer 非空。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))

        assert "answer" in result, "输出应包含 answer"
        answer = result["answer"]
        assert answer is not None, "answer 不应为 None"
        assert isinstance(answer, str), "answer 应为字符串"
        assert len(answer) > 0, "answer 不应为空"

    def test_citations_list_non_empty(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_02：citations 列表非空。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))

        assert "citations" in result, "输出应包含 citations"
        citations = result["citations"]
        assert isinstance(citations, list), "citations 应为 list"
        assert len(citations) > 0, "citations 不应为空"

    def test_each_citation_has_required_fields(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_03：每个 citation 包含 citation_id、source、content、score。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        citations = result["citations"]

        required_fields = {"citation_id", "source", "content", "score"}
        for citation in citations:
            for field in required_fields:
                assert field in citation, (
                    f"citation 应包含 '{field}' 字段，keys={list(citation.keys())}"
                )

    def test_inline_style_contains_bracket_markers(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_04：inline 引用风格，answer 中包含 [1] 标记。"""
        gen = _create_mock_generator(
            cpu_scheduler, fresh_bus, citation_style="inline"
        )
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        answer = result["answer"]
        assert "[1]" in answer, f"inline 风格 answer 应包含 [1] 标记，得到: {answer}"

    def test_footnote_style(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_05：footnote 引用风格。"""
        gen = _create_mock_generator(
            cpu_scheduler, fresh_bus, citation_style="footnote"
        )
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        assert "answer" in result, "footnote 风格应生成 answer"
        assert len(result["answer"]) > 0, "footnote 风格 answer 不应为空"

    def test_academic_style(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_06：academic 引用风格。"""
        gen = _create_mock_generator(
            cpu_scheduler, fresh_bus, citation_style="academic"
        )
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        assert "answer" in result, "academic 风格应生成 answer"
        assert len(result["answer"]) > 0, "academic 风格 answer 不应为空"

    def test_sources_used_count(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_07：sources_used 数量正确。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        assert "sources_used" in result, "输出应包含 sources_used"
        assert result["sources_used"] == len(result["citations"]), (
            f"sources_used={result['sources_used']} 应等于 len(citations)={len(result['citations'])}"
        )


# ===========================================================================
# TestCitationGeneratorAdvanced: T_CITE_08 ~ T_CITE_11
# ===========================================================================
class TestCitationGeneratorAdvanced:
    """CitationGenerator 高级功能测试。"""

    def test_language_zh_uses_chinese_prompt(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_08：language 参数生效（zh 使用中文 prompt）。"""
        tokenizer = _create_mock_tokenizer()
        tokenizer.decode = MagicMock(
            return_value="机器学习是人工智能[1]的一个子领域，它从数据中学习模式[2]。"
        )
        gen = _create_mock_generator(cpu_scheduler, fresh_bus, tokenizer=tokenizer)

        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
            language="zh",
        ))
        assert "answer" in result, "中文回答应生成 answer"
        assert len(result["answer"]) > 0, "中文回答不应为空"

    def test_language_en_uses_english_prompt(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_08 续：language 参数生效（en 使用英文 prompt）。"""
        tokenizer = _create_mock_tokenizer()
        tokenizer.decode = MagicMock(
            return_value="Machine learning is a subfield of AI[1] that learns from data[2]."
        )
        gen = _create_mock_generator(cpu_scheduler, fresh_bus, tokenizer=tokenizer)

        result = gen(MosaicData(
            query="What is machine learning?",
            results=sample_retrieval_results,
            language="en",
        ))
        assert "answer" in result, "英文回答应生成 answer"
        assert len(result["answer"]) > 0, "英文回答不应为空"

    def test_empty_results_raises_error(self, cpu_scheduler, fresh_bus):
        """T_CITE_09：检索结果为空时的回答处理（应抛出 ValueError）。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        with pytest.raises(ValueError, match="results"):
            gen(MosaicData(query="什么是机器学习？", results=[]))

    def test_temperature_parameter(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """T_CITE_10：temperature 参数生效。"""
        gen = _create_mock_generator(
            cpu_scheduler, fresh_bus, temperature=0.0
        )
        result = gen(MosaicData(
            query="什么是机器学习？",
            results=sample_retrieval_results,
        ))
        assert "answer" in result, "应正常生成回答"

    def test_describe_returns_correct_info(self, cpu_scheduler, fresh_bus):
        """T_CITE_11：describe 返回正确信息。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        spec = gen.describe()
        assert spec.name == "citation-generator", f"name 应为 'citation-generator'，得到 '{spec.name}'"
        assert spec.domain == "rag", f"domain 应为 'rag'，得到 '{spec.domain}'"
        assert "text" in spec.output_types, "output_types 应包含 'text'"


# ===========================================================================
# TestCitationGeneratorParseCitations
# ===========================================================================
class TestCitationGeneratorParseCitations:
    """_parse_citations 方法测试。"""

    def test_parse_single_citation(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """_parse_citations 正确解析单个引用。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        citation_map = {1: 0, 2: 1, 3: 2}
        answer = "机器学习是人工智能[1]的一个子领域。"
        citations = gen._parse_citations(answer, sample_retrieval_results[:3], citation_map)
        assert len(citations) == 1, f"应解析出 1 个引用，得到 {len(citations)}"
        assert citations[0]["citation_id"] == 1

    def test_parse_multiple_citations(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """_parse_citations 正确解析多个引用。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        citation_map = {1: 0, 2: 1, 3: 2}
        answer = "机器学习是人工智能[1]的一个子领域，从数据中学习[2]模式。NLP[3]是重要应用。"
        citations = gen._parse_citations(answer, sample_retrieval_results[:3], citation_map)
        assert len(citations) == 3, f"应解析出 3 个引用，得到 {len(citations)}"
        assert all(c["citation_id"] in {1, 2, 3} for c in citations)

    def test_parse_range_citation(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """_parse_citations 正确解析范围引用 [1-3]。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        citation_map = {1: 0, 2: 1, 3: 2}
        answer = "多项研究[1-3]表明机器学习是重要领域。"
        citations = gen._parse_citations(answer, sample_retrieval_results[:3], citation_map)
        # range [1-3] is expanded into individual numbers 1, 2, 3
        assert len(citations) == 3, f"范围引用 [1-3] 应解析出 3 个引用，得到 {len(citations)}"

    def test_parse_no_citations(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """_parse_citations 无引用时返回空列表。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        citation_map = {1: 0, 2: 1}
        answer = "机器学习是人工智能的一个子领域。"
        citations = gen._parse_citations(answer, sample_retrieval_results[:2], citation_map)
        assert citations == [], f"无引用时应返回空列表，得到 {citations}"


# ===========================================================================
# TestCitationGeneratorBuildContext
# ===========================================================================
class TestCitationGeneratorBuildContext:
    """_build_context 方法测试。"""

    def test_build_context_creates_numbered_entries(self, sample_retrieval_results, cpu_scheduler, fresh_bus):
        """_build_context 创建编号条目。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        context, citation_map = gen._build_context(
            sample_retrieval_results[:3], "inline", "zh"
        )
        assert "[1]" in context, "context 应包含 [1]"
        assert "[2]" in context, "context 应包含 [2]"
        assert "[3]" in context, "context 应包含 [3]"
        assert len(citation_map) == 3, f"citation_map 应包含 3 个条目，得到 {len(citation_map)}"

    def test_build_prompt_zh(self, cpu_scheduler, fresh_bus):
        """_build_prompt 使用中文模板。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        prompt = gen._build_prompt("测试问题", "测试上下文", "zh")
        assert "参考资料" in prompt, f"中文 prompt 应包含 '参考资料'"

    def test_build_prompt_en(self, cpu_scheduler, fresh_bus):
        """_build_prompt 使用英文模板。"""
        gen = _create_mock_generator(cpu_scheduler, fresh_bus)
        prompt = gen._build_prompt("test query", "test context", "en")
        assert "Reference materials" in prompt, f"英文 prompt 应包含 'Reference materials'"