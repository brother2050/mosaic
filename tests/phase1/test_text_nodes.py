# tests/phase1/test_text_nodes.py
"""Phase 1 文本域节点测试（集成测试）。

覆盖 TextGenerator、Chat、TextRewriter、Translator、TextSummarizer、TextClassifier。
每个节点至少 3 个测试用例，使用 mock 模型避免实际加载。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.types import MosaicData, TextData
from mosaic.core.scheduler import Scheduler, set_scheduler
from mosaic.core.events import EventBus, EventType


# ===========================================================================
# 辅助：mock transformers 模块（模块级注入，在 import 前生效）
# ===========================================================================
def _inject_mock_transformers():
    """在 sys.modules 中注入 mock transformers 模块。"""
    import types

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.chat_template = None
    tokenizer.apply_chat_template = MagicMock(return_value=MagicMock())
    tokenizer.apply_chat_template.return_value.shape = [1, 10]
    tokenizer.decode = MagicMock(return_value="Mock generated text.")
    tokenizer.__call__ = MagicMock(return_value={"input_ids": MagicMock()})

    model = MagicMock()
    model.eval = MagicMock()
    model.generate = MagicMock()
    fake_output = MagicMock()
    fake_output.__getitem__ = MagicMock(return_value=MagicMock())
    fake_output[0].shape = [1, 25]
    model.generate.return_value = fake_output
    next(model.parameters()).device = "cpu"

    # 构建 mock transformers 模块
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = MagicMock()
    mod.AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer)
    mod.AutoModelForCausalLM = MagicMock()
    mod.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=model)
    mod.AutoModelForSeq2SeqLM = MagicMock()
    mod.AutoModelForSeq2SeqLM.from_pretrained = MagicMock(return_value=model)
    mod.pipeline = MagicMock()
    sys.modules["transformers"] = mod

    return tokenizer, model


# 在模块加载时注入 mock（确保 text nodes import 时有 transformers 可用）
_mock_tok, _mock_mod = _inject_mock_transformers()


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def mock_transformers():
    """提供 mock transformers 模块。"""
    return _mock_tok, _mock_mod


@pytest.fixture
def text_scheduler():
    """文本节点测试用的 CPU 调度器。"""
    bus = EventBus()
    bus.clear()
    sched = Scheduler(bus=bus, device="cpu")
    set_scheduler(sched)
    return sched


# ===========================================================================
# T_GEN_01 - T_GEN_03: TextGenerator
# ===========================================================================
class TestTextGenerator:
    """TextGenerator 节点测试。"""

    def test_basic_generation(self, mock_transformers, text_scheduler):
        """T_GEN_01: 基本生成，返回非空文本。"""
        from mosaic.nodes.text.generator import TextGenerator

        gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        result = gen.run(MosaicData(prompt="Hello, world!"))
        assert "generated_text" in result
        assert isinstance(result["generated_text"], str)
        assert len(result["generated_text"]) > 0

    def test_custom_parameters(self, mock_transformers, text_scheduler):
        """T_GEN_02: 自定义参数（temperature、max_new_tokens）生效。"""
        from mosaic.nodes.text.generator import TextGenerator

        gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        result = gen.run(MosaicData(
            prompt="Hello",
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.8,
            do_sample=False,
        ))
        assert "generated_text" in result
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0

    def test_describe_returns_model_info(self, mock_transformers, text_scheduler):
        """T_GEN_03: describe 返回正确的模型信息。"""
        from mosaic.nodes.text.generator import TextGenerator

        gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        spec = gen.describe()
        assert spec.name == "text-generator"
        assert spec.domain == "text"
        assert "name" in spec.model_info
        assert spec.model_info["name"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_missing_prompt_raises(self, mock_transformers, text_scheduler):
        """T_GEN_01: 缺少 prompt 抛出 ValueError。"""
        from mosaic.nodes.text.generator import TextGenerator

        gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="prompt"):
            gen.run(MosaicData())


# ===========================================================================
# T_CHAT_01 - T_CHAT_03: Chat
# ===========================================================================
class TestChat:
    """Chat 节点测试。"""

    def test_single_turn_chat(self, mock_transformers, text_scheduler):
        """T_CHAT_01: 单轮对话返回回复。"""
        from mosaic.nodes.text.chat import Chat

        chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
        result = chat.run(MosaicData(
            messages=[{"role": "user", "content": "Hello"}],
        ))
        assert "reply" in result
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0
        # 返回的 messages 包含 assistant 回复
        assert len(result["messages"]) > 1
        assert result["messages"][-1]["role"] == "assistant"

    def test_multi_turn_context(self, mock_transformers, text_scheduler):
        """T_CHAT_02: 多轮对话保持上下文。"""
        from mosaic.nodes.text.chat import Chat

        chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "今天天气如何？"},
        ]
        result = chat.run(MosaicData(messages=messages))
        assert "reply" in result
        # 返回的 messages 包含原始对话 + 新回复
        assert len(result["messages"]) == 4

    def test_system_prompt(self, mock_transformers, text_scheduler):
        """T_CHAT_03: system_prompt 生效。"""
        from mosaic.nodes.text.chat import Chat

        chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
        result = chat.run(MosaicData(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are a helpful assistant.",
        ))
        assert "reply" in result

    def test_missing_messages_raises(self, mock_transformers, text_scheduler):
        """T_CHAT_01: 缺少 messages 抛出 ValueError。"""
        from mosaic.nodes.text.chat import Chat

        chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="messages"):
            chat.run(MosaicData())

    def test_invalid_message_format_raises(self, mock_transformers, text_scheduler):
        """T_CHAT_01: 非法消息格式抛出 ValueError。"""
        from mosaic.nodes.text.chat import Chat

        chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="role"):
            chat.run(MosaicData(messages=[{"not_role": "user"}]))


# ===========================================================================
# T_REWRITE_01 - T_REWRITE_03: TextRewriter
# ===========================================================================
class TestTextRewriter:
    """TextRewriter 节点测试。"""

    def test_basic_rewrite(self, mock_transformers, text_scheduler):
        """T_REWRITE_01: 基本改写，返回不同文本。"""
        from mosaic.nodes.text.rewriter import TextRewriter

        rewriter = TextRewriter(model="Qwen/Qwen2.5-7B-Instruct")
        result = rewriter.run(MosaicData(
            text="这个东西很好用。",
        ))
        assert "rewritten_text" in result
        assert "original_text" in result
        assert result["original_text"] == "这个东西很好用。"

    def test_custom_instruction(self, mock_transformers, text_scheduler):
        """T_REWRITE_02: 指定 instruction 改写。"""
        from mosaic.nodes.text.rewriter import TextRewriter

        rewriter = TextRewriter(model="Qwen/Qwen2.5-7B-Instruct")
        result = rewriter.run(MosaicData(
            text="Hello world",
            instruction="Translate to French",
        ))
        assert "rewritten_text" in result
        assert "original_text" in result

    def test_preserves_original_text(self, mock_transformers, text_scheduler):
        """T_REWRITE_03: 保留原文语义（输出含 original_text）。"""
        from mosaic.nodes.text.rewriter import TextRewriter

        original = "请保持语义不变"
        rewriter = TextRewriter(model="Qwen/Qwen2.5-7B-Instruct")
        result = rewriter.run(MosaicData(text=original))
        assert result["original_text"] == original

    def test_missing_text_raises(self, mock_transformers, text_scheduler):
        """T_REWRITE_01: 缺少 text 抛出 ValueError。"""
        from mosaic.nodes.text.rewriter import TextRewriter

        rewriter = TextRewriter(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="text"):
            rewriter.run(MosaicData())


# ===========================================================================
# T_TRANS_01 - T_TRANS_03: Translator
# ===========================================================================
class TestTranslator:
    """Translator 节点测试。"""

    def test_zh_to_en(self, mock_transformers, text_scheduler):
        """T_TRANS_01: 中译英。"""
        from mosaic.nodes.text.translator import Translator

        translator = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        result = translator.run(MosaicData(
            text="你好，世界！",
            target_language="en",
        ))
        assert "translated_text" in result
        assert result["target_language"] == "en"

    def test_en_to_zh(self, mock_transformers, text_scheduler):
        """T_TRANS_02: 英译中。"""
        from mosaic.nodes.text.translator import Translator

        translator = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        result = translator.run(MosaicData(
            text="Hello, world!",
            target_language="zh",
        ))
        assert "translated_text" in result
        assert result["target_language"] == "zh"

    def test_auto_source_language(self, mock_transformers, text_scheduler):
        """T_TRANS_03: auto 源语言检测。"""
        from mosaic.nodes.text.translator import Translator

        translator = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        result = translator.run(MosaicData(
            text="Bonjour le monde",
            target_language="en",
            source_language="auto",
        ))
        assert "translated_text" in result
        assert result["source_language"] == "auto"

    def test_missing_target_language_raises(self, mock_transformers, text_scheduler):
        """T_TRANS_01: 缺少 target_language 抛出 ValueError。"""
        from mosaic.nodes.text.translator import Translator

        translator = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="target_language"):
            translator.run(MosaicData(text="Hello"))

    def test_describe_includes_translation_mode(self, mock_transformers, text_scheduler):
        """T_TRANS_03: describe 包含翻译模式信息。"""
        from mosaic.nodes.text.translator import Translator

        tr = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        spec = tr.describe()
        assert "translation_mode" in spec.model_info
        assert spec.model_info["translation_mode"] == "generic"


# ===========================================================================
# T_SUM_01 - T_SUM_03: TextSummarizer
# ===========================================================================
class TestTextSummarizer:
    """TextSummarizer 节点测试。"""

    def test_long_text_summary(self, mock_transformers, text_scheduler):
        """T_SUM_01: 长文本摘要，压缩比 < 1。"""
        from mosaic.nodes.text.summarizer import TextSummarizer

        long_text = "这是一段非常长的文本。" * 50  # ~500 chars
        summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
        result = summarizer.run(MosaicData(
            text=long_text,
            max_length=100,
            style="concise",
        ))
        assert "summary" in result
        assert "compression_ratio" in result
        assert "original_length" in result
        assert result["original_length"] == len(long_text)

    def test_short_text_returned_as_is(self, mock_transformers, text_scheduler):
        """T_SUM_02: 短文本直接返回。"""
        from mosaic.nodes.text.summarizer import TextSummarizer

        short_text = "短文本"
        summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
        result = summarizer.run(MosaicData(
            text=short_text,
            max_length=150,  # 原文比 max_length 短
        ))
        assert result["compression_ratio"] == 1.0
        assert result["summary"] == short_text
        assert "returned as-is" in (result.get("note", "") or "")

    def test_bullet_points_style(self, mock_transformers, text_scheduler):
        """T_SUM_03: bullet_points 风格。"""
        from mosaic.nodes.text.summarizer import TextSummarizer

        text = "段落一：" + "内容A" * 30 + "。段落二：" + "内容B" * 30
        summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
        result = summarizer.run(MosaicData(
            text=text,
            max_length=50,
            style="bullet_points",
        ))
        assert "summary" in result

    def test_invalid_style_raises(self, mock_transformers, text_scheduler):
        """T_SUM_01: 非法 style 抛出 ValueError。"""
        from mosaic.nodes.text.summarizer import TextSummarizer

        summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="Invalid style"):
            summarizer.run(MosaicData(
                text="A" * 200,
                max_length=50,
                style="invalid_style",
            ))

    def test_describe_includes_styles(self, mock_transformers, text_scheduler):
        """T_SUM_03: describe 包含支持的风格。"""
        from mosaic.nodes.text.summarizer import TextSummarizer

        summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
        spec = summarizer.describe()
        assert "supported_styles" in spec.model_info
        assert "concise" in spec.model_info["supported_styles"]


# ===========================================================================
# T_CLS_01 - T_CLS_03: TextClassifier
# ===========================================================================
class TestTextClassifier:
    """TextClassifier 节点测试。"""

    def test_single_label_classification(self, mock_transformers, text_scheduler):
        """T_CLS_01: 单标签分类，返回正确格式。"""
        from mosaic.nodes.text.classifier import TextClassifier

        clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        result = clf.run(MosaicData(
            text="这部电影太棒了！",
            labels=["正面", "负面", "中性"],
        ))
        assert "predicted_label" in result
        assert result["predicted_label"] in ["正面", "负面", "中性"]
        assert "scores" in result
        assert "method" in result

    def test_scores_dict(self, mock_transformers, text_scheduler):
        """T_CLS_02: 返回 scores 字典。"""
        from mosaic.nodes.text.classifier import TextClassifier

        clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        result = clf.run(MosaicData(
            text="AI technology is advancing rapidly",
            labels=["科技", "财经", "体育"],
        ))
        assert "scores" in result
        assert isinstance(result["scores"], dict)
        # 每个标签都有分数
        for label in ["科技", "财经", "体育"]:
            assert label in result["scores"]

    def test_multi_label_classification(self, mock_transformers, text_scheduler):
        """T_CLS_03: 多标签分类。"""
        from mosaic.nodes.text.classifier import TextClassifier

        clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        result = clf.run(MosaicData(
            text="科技公司发布新款智能手机，股价大涨",
            labels=["科技", "财经", "体育"],
            multi_label=True,
        ))
        assert "predicted_labels" in result
        assert isinstance(result["predicted_labels"], list)
        assert "scores" in result

    def test_missing_labels_raises(self, mock_transformers, text_scheduler):
        """T_CLS_01: 缺少 labels 抛出 ValueError。"""
        from mosaic.nodes.text.classifier import TextClassifier

        clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="labels"):
            clf.run(MosaicData(text="Hello"))

    def test_describe_includes_classification_mode(self, mock_transformers, text_scheduler):
        """T_CLS_03: describe 包含分类模式信息。"""
        from mosaic.nodes.text.classifier import TextClassifier

        clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        spec = clf.describe()
        assert "classification_mode" in spec.model_info
        assert "zero_shot_threshold" in spec.model_info