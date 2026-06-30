"""
examples/01_text_domain.py
文本域示例 —— 6 个文本节点 + 文本处理管道组合。

运行：
    python examples/01_text_domain.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic import Pipeline
from mosaic.core.types import MosaicData
from mosaic.nodes.text import (
    TextGenerator,
    Chat,
    TextRewriter,
    Translator,
    TextSummarizer,
    TextClassifier,
)


def example_1_text_generation():
    """示例 1：基本文本生成。"""
    print("\n=== 示例 1：基本文本生成 ===")

    gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
    result = gen.run(
        prompt="用三句话描述春天的早晨",
        temperature=0.7,
        seed=42,
    )

    text = result.get("text")
    print(f"生成的文本：\n{text}\n")
    return text


def example_2_chat():
    """示例 2：多轮对话。"""
    print("\n=== 示例 2：多轮对话 ===")

    chat = Chat(system_prompt="你是一个 Python 教学助手")

    r1 = chat.run(MosaicData(message="什么是装饰器？"))
    print(f"用户：什么是装饰器？")
    print(f"助手：{r1.get('reply')}\n")

    r2 = chat.run(MosaicData(message="能举个例子吗？"))
    print(f"用户：能举个例子吗？")
    print(f"助手：{r2.get('reply')}\n")


def example_3_rewriter():
    """示例 3：文本改写。"""
    print("\n=== 示例 3：文本改写 ===")

    rewriter = TextRewriter()
    original = "这个产品超棒，我非常喜欢，用起来很爽"

    result = rewriter.run(
        text=original,
        style="formal",
        requirement="增加专业术语",
    )

    print(f"原文：{original}")
    print(f"改写：{result.get('text')}\n")


def example_4_translator():
    """示例 4：翻译。"""
    print("\n=== 示例 4：翻译 ===")

    translator = Translator()

    for text, target in [
        ("你好世界", "en"),
        ("今天天气真好", "ja"),
        ("Hello world", "zh"),
    ]:
        result = translator.run(MosaicData(text=text, target_lang=target))
        print(f"{text} → [{target}]: {result.get('text')}")

    print()


def example_5_summarizer():
    """示例 5：摘要。"""
    print("\n=== 示例 5：摘要 ===")

    long_text = """
    Mosaic 是一个全模态生成式 AI 框架，将文本、图像、视频、音频等
    能力抽象为可独立注册、自由组合的"节点"（Node）。用户只需用
    Python 就能像搭积木一样把它们串成任意复杂的生成式 AI 流水线。
    核心理念是解耦：每个节点独立运行、独立测试、独立组合。
    """

    summarizer = TextSummarizer()
    result = summarizer.run(
        text=long_text,
        mode="bullet_points",
        max_length=100,
    )

    print(f"摘要：\n{result.get('text')}\n")


def example_6_classifier():
    """示例 6：分类。"""
    print("\n=== 示例 6：分类 ===")

    classifier = TextClassifier()

    # 情感分析
    result = classifier.run(MosaicData(text="这家餐厅的食物非常棒！", mode="sentiment"))
    print(f"情感分析：{result.get('label')} ({result.get('scores')})")

    # 零样本分类
    result = classifier.run(
        text="今天的会议讨论了产品路线图",
        labels=["技术", "商业", "运营", "人事"],
        mode="zero_shot",
    )
    print(f"零样本分类：{result.get('label')} (top-3: {result.get('top_k')})")

    print()


def example_7_combined_pipeline():
    """示例 7：文本处理管道（生成 → 翻译 → 摘要）。"""
    print("\n=== 示例 7：组合管道（生成 → 翻译 → 摘要）===")

    pipe = (
        TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        | Translator()
        | TextSummarizer()
    )

    result = pipe.run(
        prompt="用一段话介绍北京",
        target_lang="en",
        mode="concise",
    )

    print(f"最终结果：\n{result.get('text')}\n")


def main():
    """运行所有文本域示例。"""
    print("=" * 60)
    print("Mosaic 文本域示例")
    print("=" * 60)

    example_1_text_generation()
    example_2_chat()
    example_3_rewriter()
    example_4_translator()
    example_5_summarizer()
    example_6_classifier()
    example_7_combined_pipeline()

    print("=" * 60)
    print("所有文本域示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
