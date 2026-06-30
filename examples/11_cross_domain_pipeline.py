"""
examples/11_cross_domain_pipeline.py
跨域综合管道示例 —— 演示 Mosaic 跨域组合的 5+ 个完整场景。

运行：
    python examples/11_cross_domain_pipeline.py
"""
from __future__ import annotations

import asyncio
import sys
import time
sys.path.insert(0, "/workspace/mosaic")

from mosaic import Pipeline, Branch, Merge
from mosaic.core.types import AudioData, ImageData, MosaicData
from mosaic.nodes.audio import TTS
from mosaic.nodes.digital_human import LipSyncer, AvatarDriver
from mosaic.nodes.export import VideoEncoder, MultiFormatExporter
from mosaic.nodes.image import TextToImage, Upscaler
from mosaic.nodes.rag import (
    DocumentParser, VectorIndexer, Retriever, CitationGenerator,
)
from mosaic.nodes.subtitle import SubtitleGenerator, SubtitleAligner
from mosaic.nodes.text import TextGenerator
from mosaic.nodes.video import WanVideo


# ============= 场景 1: 完整创作链 =============

def scenario_1_text_to_video():
    """场景 1：文本 → 图像 → 视频 → 导出（完整创作链）。"""
    print("\n" + "=" * 60)
    print("场景 1: 文本 → 图像 → 视频 → 导出")
    print("=" * 60)

    pipe = (
        TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        | TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
        | Upscaler(scale=2)
        | WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True)
        | MultiFormatExporter(formats=["mp4", "gif"], output_dir="./scenario_1")
    )

    print("管道：")
    print("  TextGenerator → TextToImage → Upscaler → WanVideo → MultiFormatExporter")

    result = pipe.run(
        prompt="A majestic dragon flying over a medieval castle at sunset",
        seed=42,
        num_frames=49,
    )

    print(f"\n✅ 输出：")
    print(f"  MP4: ./scenario_1/output.mp4")
    print(f"  GIF: ./scenario_1/output.gif")


# ============= 场景 2: 数字人链 =============

def scenario_2_tts_to_digital_human():
    """场景 2：TTS → 口型同步 → 数字人。"""
    print("\n" + "=" * 60)
    print("场景 2: TTS → 口型同步 → 数字人")
    print("=" * 60)

    pipe = (
        TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
        | AvatarDriver()
        | TTS(backend="chattts", language="zh")
        | LipSyncer()
    )

    print("管道：")
    print("  TextToImage → AvatarDriver → TTS → LipSyncer")

    result = pipe.run(
        prompt="a friendly female digital assistant, smile, professional",
        text="你好，我是数字人助手。",
    )

    print("\n✅ 已生成带口型同步的数字人视频")


# ============= 场景 3: RAG 知识链 =============

def scenario_3_document_qa():
    """场景 3：文档 → RAG → 回答。"""
    print("\n" + "=" * 60)
    print("场景 3: 文档 → RAG → 回答")
    print("=" * 60)

    pipe = (
        DocumentParser()
        | VectorIndexer(embedding_model="BAAI/bge-m3", index_path="./scenario_3_index")
        | Retriever(top_k=5)
        | TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
        | CitationGenerator()
    )

    print("管道：")
    print("  DocumentParser → VectorIndexer → Retriever → TextGenerator → CitationGenerator")

    result = pipe.run(
        file_path="mosaic_manual.pdf",
        query="Mosaic 的 TTS 后端有哪些？各自特点是什么？",
        index_path="./scenario_3_index",
    )

    print(f"\n✅ 答案：\n{result.get('answer')}")


# ============= 场景 4: 配音链 =============

def scenario_4_dubbing_pipeline():
    """场景 4：文本 → TTS → 字幕 → 视频编码。"""
    print("\n" + "=" * 60)
    print("场景 4: 文本 → TTS → 字幕 → 视频编码（配音链）")
    print("=" * 60)

    pipe = (
        WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True)
        | TTS(backend="chattts", language="zh")
        | SubtitleGenerator()
        | SubtitleAligner()
        | VideoEncoder(output_path="dubbed_video.mp4", fps=16)
    )

    print("管道：")
    print("  WanVideo → TTS → SubtitleGenerator → SubtitleAligner → VideoEncoder")

    result = pipe.run(
        prompt="A cat playing piano in a cozy room",
        text="这只小猫正在弹钢琴，旋律优美。",
        num_frames=49,
        fps=16,
    )

    print("\n✅ 已生成配音视频: dubbed_video.mp4")


# ============= 场景 5: 并行处理 =============

def scenario_5_parallel_image_and_audio():
    """场景 5：并行分支同时处理图像和音频。"""
    print("\n" + "=" * 60)
    print("场景 5: 并行分支（图像 + 音频）")
    print("=" * 60)

    pipe = Pipeline([
        Branch([
            TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
            | Upscaler(scale=2),
            TTS(backend="chattts", language="zh"),
            SubtitleGenerator(),
        ]),
        Merge(strategy="concat"),
    ])

    print("管道：")
    print("  Branch[")
    print("    TextToImage → Upscaler")
    print("    TTS")
    print("    SubtitleGenerator")
    print("  ] → Merge")

    result = pipe.run(
        prompt="A scenic mountain view at dawn",
        text="远处的山峰在云雾中若隐若现。",
    )

    print(f"\n✅ 同时获得：")
    print(f"  image:    {type(result.get('image')).__name__}")
    print(f"  audio:    {type(result.get('audio')).__name__}")
    print(f"  subtitle: {type(result.get('subtitle')).__name__}")


# ============= 场景 6: 异步执行 =============

async def scenario_6_async_execution():
    """场景 6：异步执行长任务。"""
    print("\n" + "=" * 60)
    print("场景 6: 异步执行长任务")
    print("=" * 60)

    pipe = WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True)

    # 启动后台任务
    task = pipe.run_async(
        prompt="A long video generation test",
        num_frames=81,
    )
    print(f"任务已启动: {task.task_id}")

    # 期间做其他事
    for i in range(3):
        print(f"  等待中... ({i*5}s)")
        await asyncio.sleep(5)

    # 等待完成
    result = await task.result_async()
    print(f"\n✅ 异步任务完成")


# ============= 场景 7: 四 TTS 后端对比 =============

def scenario_7_tts_backend_comparison():
    """场景 7：四个 TTS 后端对比合成同一文本。"""
    print("\n" + "=" * 60)
    print("场景 7: 四 TTS 后端对比合成")
    print("=" * 60)

    text = "这是同一段文本，用四个不同的 TTS 后端合成，对比效果。"

    backends = ["chattts", "fish", "sovits", "cosyvoice"]
    durations = {"chattts": 0, "fish": 0, "sovits": 0, "cosyvoice": 0}
    sample_rates = {"chattts": 0, "fish": 0, "sovits": 0, "cosyvoice": 0}

    for backend in backends:
        print(f"\n  后端: {backend}")
        try:
            tts = TTS(backend=backend, language="zh")
            start = time.time()
            result = tts.run(MosaicData(text=text, language="zh"))
            duration = time.time() - start
            audio = result.get("audio")
            audio.save(f"comparison_{backend}.wav")

            durations[backend] = audio.duration
            sample_rates[backend] = audio.sample_rate
            print(f"    采样率: {audio.sample_rate} Hz")
            print(f"    音频时长: {audio.duration:.2f}s")
            print(f"    合成耗时: {duration:.2f}s")
        except Exception as e:
            print(f"    跳过（未安装/未配置）: {e}")

    print(f"\n✅ 已保存到 comparison_{{chattts,fish,sovits,cosyvoice}}.wav")
    print(f"\n采样率对比：")
    for backend, sr in sample_rates.items():
        if sr:
            print(f"  {backend:12s}: {sr} Hz")


# ============= 主函数 =============

def main():
    print("=" * 70)
    print("Mosaic 跨域综合管道示例")
    print("=" * 70)
    print("\n7 个完整场景演示 Mosaic 的跨域组合能力")
    print("注：完整运行需要相应模型权重")

    # 场景 1
    scenario_1_text_to_video()

    # 场景 2
    scenario_2_tts_to_digital_human()

    # 场景 3
    scenario_3_document_qa()

    # 场景 4
    scenario_4_dubbing_pipeline()

    # 场景 5
    scenario_5_parallel_image_and_audio()

    # 场景 6
    asyncio.run(scenario_6_async_execution())

    # 场景 7
    scenario_7_tts_backend_comparison()

    print("\n" + "=" * 70)
    print("所有跨域综合示例运行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
