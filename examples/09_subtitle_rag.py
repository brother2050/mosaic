"""
examples/09_subtitle_rag.py
字幕与 RAG 域示例 —— 字幕生成/翻译/对齐，RAG 完整管道，组合应用。

运行：
    python examples/09_subtitle_rag.py
"""
from __future__ import annotations

from mosaic import Pipeline
from mosaic.core import MosaicData
from mosaic.nodes.audio import ASR
from mosaic.nodes.subtitle import (
    SubtitleGenerator,
    SubtitleTranslator,
    SubtitleAligner,
)
from mosaic.nodes.rag import (
    DocumentParser,
    VectorIndexer,
    Retriever,
    CitationGenerator,
)
from mosaic.nodes.text import TextGenerator


# ============= 字幕示例 =============

def example_1_subtitle_generation():
    """示例 1：字幕生成。"""
    print("\n=== 示例 1：字幕生成 ===")

    gen = SubtitleGenerator()

    # SubtitleGenerator 接受 str 路径 / ndarray / AudioData 作为 ``audio``
    result = gen.run(MosaicData(audio="speech.wav", language="zh"))
    subtitle = result.get("subtitle")
    segments = subtitle.segments

    print(f"字幕段数: {len(segments)}")
    print(f"前 3 段:")
    for seg in segments[:3]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")


def example_2_subtitle_translate():
    """示例 2：字幕翻译。"""
    print("\n=== 示例 2：字幕翻译（中文 → 英文）===")

    pipe = SubtitleGenerator() | SubtitleTranslator()

    result = pipe.run(MosaicData(
        audio="speech.wav",
        language="zh",
        target_lang="en",
    ))

    subtitle = result.get("subtitle")
    print("已生成中英双语字幕")
    for seg in subtitle.segments[:3]:
        print(f"  [{seg['start']:.1f}s] {seg['text']} / {seg['translation']}")


def example_3_subtitle_align():
    """示例 3：字幕时间轴对齐。"""
    print("\n=== 示例 3：字幕时间轴对齐 ===")

    pipe = (
        SubtitleGenerator()
        | SubtitleTranslator(target_lang="en")
        | SubtitleAligner()
    )

    result = pipe.run(MosaicData(
        audio="long_speech.wav",
        source_lang="zh",
        target_lang="en",
    ))
    print("已生成时间轴精确对齐的双语字幕")


# ============= RAG 示例 =============

def example_4_rag_basic():
    """示例 4：RAG 基础管道。"""
    print("\n=== 示例 4：RAG 基础管道（解析 → 索引 → 检索 → 生成）===")

    pipe = (
        DocumentParser()
        | VectorIndexer(embedding_model="BAAI/bge-m3", index_path="./index")
        | Retriever(top_k=3)
        | TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        | CitationGenerator()
    )

    result = pipe.run(MosaicData(
        file_path="manual.pdf",
        query="如何使用 Mosaic 的 Pipeline？",
        index_path="./index",
    ))

    print(f"答案：\n{result.get('answer')}")
    print(f"\n引用：")
    for cit in result.get("citations", []):
        print(f"  [{cit['source']}] {cit['content'][:80]}")


def example_5_rag_with_existing_index():
    """示例 5：使用已有索引的 RAG 检索。"""
    print("\n=== 示例 5：使用已有索引的 RAG 检索 ===")

    retriever = Retriever(index_path="./existing_index", top_k=5)
    result = retriever.run(MosaicData(query="Mosaic 的 TTS 后端有哪些？"))

    # Retriever 输出 ``results`` (list[dict])，每项含 content/score/source
    results = result.get("results", [])
    print(f"检索到 {len(results)} 个相关文档")
    for doc in results[:3]:
        print(f"  - {doc['content'][:100]}...")


# ============= 字幕 + RAG 组合示例 =============

def example_6_subtitle_rag_video_qa():
    """示例 6：视频内容问答（ASR → 字幕 → RAG 检索）。"""
    print("\n=== 示例 6：视频内容问答 ===")
    print("流程：视频 → ASR → 字幕生成 → 向量化索引 → 用户问题 → 检索 → 回答")

    # 第 1 步：建立视频字幕索引
    print("\n步骤 1: 处理视频并建立索引")
    index_pipe = (
        ASR(model="openai/whisper-large-v3")
        | SubtitleGenerator()
        | VectorIndexer(embedding_model="BAAI/bge-m3", index_path="./video_index")
    )
    index_pipe.run(MosaicData(audio="tutorial_video.wav"))
    print("  视频索引已建立")

    # 第 2 步：用户提问
    print("\n步骤 2: 用户提问")
    qa_pipe = (
        Retriever(index_path="./video_index", top_k=3)
        | TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        | CitationGenerator()
    )

    result = qa_pipe.run(MosaicData(
        query="视频中讲解的 WanVideo 节点支持哪些模型？",
    ))
    print(f"\n回答：\n{result.get('answer')}")
    print(f"\n引用：")
    for cit in result.get("citations", [])[:3]:
        print(f"  [{cit['source']}] {cit['content'][:80]}")


def main():
    print("=" * 60)
    print("Mosaic 字幕与 RAG 域示例")
    print("=" * 60)

    # 字幕示例
    example_1_subtitle_generation()
    example_2_subtitle_translate()
    example_3_subtitle_align()

    # RAG 示例
    example_4_rag_basic()
    example_5_rag_with_existing_index()

    # 组合示例
    example_6_subtitle_rag_video_qa()

    print("=" * 60)
    print("所有字幕与 RAG 示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
