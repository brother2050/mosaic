"""
examples/11_cross_domain_pipeline.py
跨域综合管道示例 —— 演示 Mosaic 跨域组合的 7 个完整场景。

运行：
    python examples/11_cross_domain_pipeline.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from mosaic import Pipeline, Branch, Merge
from mosaic.core import MosaicData
from mosaic.nodes.audio import TTS
from mosaic.nodes.digital_human import LipSyncer
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
    """场景 1：文本 → 图像 → 视频 → 导出（完整创作链）。

    TextGenerator 输出 ``prompt`` 字段，可直接与 TextToImage 等
    diffusers 下游节点串联。此示例使用显式逐节点调用展示完整数据流转。
    """
    print("\n" + "=" * 60)
    print("场景 1: 文本 → 图像 → 视频 → 导出")
    print("=" * 60)

    gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
    t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
    upscaler = Upscaler()
    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True
    )

    print("流程：")
    print("  TextGenerator → TextToImage → Upscaler → WanVideo → MultiFormatExporter")

    # 1. 生成详细描述
    r1 = gen.run(MosaicData(
        prompt="A majestic dragon flying over a medieval castle at sunset",
        seed=42,
    ))
    description = r1.get("prompt")

    # 2. 根据描述生成图像
    r2 = t2i.run(MosaicData(prompt=description, seed=42))
    image = r2.get("image")

    # 3. 超分放大
    r3 = upscaler.run(MosaicData(image=image, scale_factor=2))

    # 4. 根据描述生成视频
    r4 = wan.run(MosaicData(
        prompt=description, num_frames=49, seed=42,
    ))
    video = r4.get("video")

    # 5. 多格式导出
    os.makedirs("./scenario_1", exist_ok=True)
    exporter = MultiFormatExporter()
    r5 = exporter.run(MosaicData(
        content_type="video",
        data=video,
        formats=["mp4", "gif"],
        output_dir="./scenario_1",
    ))

    print(f"\n  输出：{r5.get('total_files')} 个文件")
    print(f"  MP4: ./scenario_1/output.mp4")
    print(f"  GIF: ./scenario_1/output.gif")


# ============= 场景 2: 数字人链 =============

def scenario_2_tts_to_digital_human():
    """场景 2：生成形象 → TTS 合成 → 口型同步。

    LipSyncer 接受 ``face_image`` (PIL.Image) 与 ``audio`` (AudioData)，
    生成口型与音频同步的数字人说话视频。
    """
    print("\n" + "=" * 60)
    print("场景 2: 生成形象 → TTS → 口型同步")
    print("=" * 60)

    t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
    tts = TTS(backend="chattts", language="zh")
    lip_syncer = LipSyncer()

    print("流程：")
    print("  TextToImage → TTS → LipSyncer")

    # 1. 生成数字人形象
    r1 = t2i.run(MosaicData(
        prompt="a friendly female digital assistant, smile, professional",
    ))
    avatar = r1.get("image")

    # 2. TTS 合成语音
    r2 = tts.run(MosaicData(text="你好，我是数字人助手。"))
    audio = r2.get("audio")

    # 3. 口型同步生成视频
    r3 = lip_syncer.run(MosaicData(face_image=avatar, audio=audio))
    video = r3.get("video")

    # 4. 编码保存
    encoder = VideoEncoder(format="mp4")
    encoder.run(MosaicData(
        frames=video.frames,
        fps=video.fps,
        audio=audio,
        output_path="digital_human.mp4",
    ))

    print("\n  已生成带口型同步的数字人视频: digital_human.mp4")


# ============= 场景 3: RAG 知识链 =============

def scenario_3_document_qa():
    """场景 3：文档解析 → 向量索引 → 检索 → 引用生成。

    CitationGenerator 内部已集成 LLM 生成回答，无需额外 TextGenerator。
    """
    print("\n" + "=" * 60)
    print("场景 3: 文档 → RAG → 回答")
    print("=" * 60)

    parser = DocumentParser()
    indexer = VectorIndexer(
        embedding_model="BAAI/bge-m3", index_path="./scenario_3_index"
    )
    retriever = Retriever(top_k=5)
    cit_gen = CitationGenerator()

    print("流程：")
    print("  DocumentParser → VectorIndexer → Retriever → CitationGenerator")

    query = "Mosaic 的 TTS 后端有哪些？各自特点是什么？"

    # 1. 解析文档
    r1 = parser.run(MosaicData(file_path="mosaic_manual.pdf"))

    # 2. 向量索引
    r2 = indexer.run(MosaicData(document=r1.get("document")))

    # 3. 检索
    r3 = retriever.run(MosaicData(
        query=query,
        collection_name=r2.get("collection_name"),
    ))

    # 4. 生成带引用的回答
    r4 = cit_gen.run(MosaicData(
        query=query,
        results=r3.get("results"),
    ))

    print(f"\n  答案：\n{r4.get('answer')}")
    print(f"  引用数：{r4.get('sources_used')}")


# ============= 场景 4: 配音链 =============

def scenario_4_dubbing_pipeline():
    """场景 4：视频生成 → TTS → 字幕生成 → 字幕对齐 → 视频编码。

    VideoEncoder 的 ``output_path``、``fps`` 是运行时输入字段，
    不是构造参数。
    """
    print("\n" + "=" * 60)
    print("场景 4: 视频生成 → TTS → 字幕 → 视频编码（配音链）")
    print("=" * 60)

    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True
    )
    tts = TTS(backend="chattts", language="zh")
    sub_gen = SubtitleGenerator()
    sub_aligner = SubtitleAligner()
    encoder = VideoEncoder(format="mp4")

    print("流程：")
    print("  WanVideo → TTS → SubtitleGenerator → SubtitleAligner → VideoEncoder")

    # 1. 生成视频
    r1 = wan.run(MosaicData(
        prompt="A cat playing piano in a cozy room",
        num_frames=49,
        fps=16,
    ))
    video = r1.get("video")

    # 2. TTS 合成配音
    r2 = tts.run(MosaicData(text="这只小猫正在弹钢琴，旋律优美。"))
    audio = r2.get("audio")

    # 3. 从音频生成字幕
    r3 = sub_gen.run(MosaicData(audio=audio))
    subtitle = r3.get("subtitle")

    # 4. 字幕与音频对齐
    r4 = sub_aligner.run(MosaicData(subtitle=subtitle, audio=audio))
    aligned_subtitle = r4.get("subtitle")

    # 5. 编码带配音和字幕的视频
    encoder.run(MosaicData(
        frames=video.frames,
        fps=video.fps,
        audio=audio,
        output_path="dubbed_video.mp4",
        subtitle=aligned_subtitle,
    ))

    print("\n  已生成配音视频: dubbed_video.mp4")


# ============= 场景 5: 并行处理 =============

def scenario_5_parallel_image_and_audio():
    """场景 5：并行分支同时处理图像和音频。

    Branch 使用命名路径（``image=...``, ``audio=...``），各分支
    并行执行，Merge 以 ``flatten`` 策略将所有分支输出合并到一个
    MosaicData。
    """
    print("\n" + "=" * 60)
    print("场景 5: 并行分支（图像 + 音频）")
    print("=" * 60)

    pipe = Pipeline([
        Branch(
            image=TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
            | Upscaler(),
            audio=TTS(backend="chattts", language="zh"),
        ),
        Merge(strategy="flatten"),
    ])

    print("管道：")
    print("  Branch[")
    print("    image: TextToImage → Upscaler")
    print("    audio: TTS")
    print("  ] → Merge(flatten)")

    result = pipe.run(MosaicData(
        prompt="A scenic mountain view at dawn",
        text="远处的山峰在云雾中若隐若现。",
    ))

    print(f"\n  同时获得：")
    print(f"  image: {type(result.get('image')).__name__}")
    print(f"  audio: {type(result.get('audio')).__name__}")


# ============= 场景 6: 异步执行 =============

async def scenario_6_async_execution():
    """场景 6：异步执行长任务。

    ``run_async`` 是 :class:`Pipeline` 的方法，需将节点包装为 Pipeline。
    返回 :class:`AsyncTask`，可通过 ``task.wait()`` 阻塞等待结果。
    """
    print("\n" + "=" * 60)
    print("场景 6: 异步执行长任务")
    print("=" * 60)

    pipe = Pipeline([
        WanVideo(
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True
        )
    ])

    # 启动后台任务
    task = pipe.run_async(MosaicData(
        prompt="A long video generation test",
        num_frames=81,
    ))
    print(f"  任务已启动: {task.task_id}")

    # 期间做其他事
    for i in range(3):
        print(f"  等待中... ({i * 5}s)")
        await asyncio.sleep(5)

    # 等待完成
    result = task.wait()
    print(f"\n  异步任务完成")
    print(f"  生成视频：{result.get('num_frames')} 帧, {result.get('duration'):.2f} 秒")


# ============= 场景 7: 四 TTS 后端对比 =============

def scenario_7_tts_backend_comparison():
    """场景 7：四个 TTS 后端对比合成同一文本。

    使用 ``soundfile.write()`` 保存音频（AudioData 无 ``save()`` 方法）。
    音频时长从 TTS 输出的 ``duration`` 字段获取。
    """
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
            elapsed = time.time() - start
            audio = result.get("audio")

            # 使用 soundfile 保存音频
            waveform = audio.waveform
            if waveform is not None and hasattr(waveform, "ndim") and waveform.ndim > 1:
                # AudioData 波形为 (channels, samples)，soundfile 需要 (samples, channels)
                waveform = np.asarray(waveform).T
            sf.write(f"comparison_{backend}.wav", waveform, audio.sample_rate)

            audio_duration = result.get("duration", 0.0)
            durations[backend] = audio_duration
            sample_rates[backend] = audio.sample_rate
            print(f"    采样率: {audio.sample_rate} Hz")
            print(f"    音频时长: {audio_duration:.2f}s")
            print(f"    合成耗时: {elapsed:.2f}s")
        except Exception as e:  # noqa: BLE001
            print(f"    跳过（未安装/未配置）: {e}")

    print(f"\n  已保存到 comparison_{{chattts,fish,sovits,cosyvoice}}.wav")
    print(f"\n  采样率对比：")
    for backend, sr in sample_rates.items():
        if sr:
            print(f"    {backend:12s}: {sr} Hz")


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
