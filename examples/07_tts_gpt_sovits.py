"""
examples/07_tts_gpt_sovits.py
GPT-SoVITS 完整示例 —— 极少样本克隆、跨语言合成、特征缓存、流式输出。

依赖：
    pip install "GPT-SoVITS[cpu]"  # 或 [gpu]
    # 权重：参考 https://huggingface.co/lj1995/GPT-SoVITS

运行：
    python examples/07_tts_gpt_sovits.py
"""
from __future__ import annotations

import asyncio
import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio import TTS
from mosaic import MosaicData


def example_1_basic():
    """示例 1：基础合成（32kHz）。"""
    print("\n=== 示例 1：基础合成（32kHz）===")

    tts = TTS(backend="sovits", language="zh")
    result = tts.run(
        text="这是使用 GPT-SoVITS 合成的中文语音。",
        language="zh",
        speaker="default",  # 预训练说话人 ID
    )

    audio = result.get("audio")
    audio.save("output_sovits_basic.wav")
    print(f"采样率: {audio.sample_rate} Hz, 时长: {audio.duration:.2f}s")


def example_2_minimal_sample_clone():
    """示例 2：极少样本克隆（仅需 5-10 秒参考音频）。"""
    print("\n=== 示例 2：极少样本克隆（5-10 秒）===")

    tts = TTS(
        backend="sovits",
        ref_audio="short_ref.wav",   # 5-10 秒
    )

    result = tts.run(
        text="这是用 5 秒参考音频克隆的声音，合成任意新文本。",
        language="zh",
    )
    audio = result.get("audio")
    audio.save("output_sovits_cloned.wav")
    print(f"已克隆：{audio.duration:.2f}s")


def example_3_cross_language():
    """示例 3：跨语言合成。"""
    print("\n=== 示例 3：跨语言合成 ===")

    tts = TTS(
        backend="sovits",
        ref_audio="chinese_ref.wav",
    )

    # 用中文音色合成英文
    result = tts.run(
        text="This is a Chinese voice speaking English.",
        language="en",
    )
    audio = result.get("audio")
    audio.save("output_sovits_cross_lang.wav")
    print("已实现跨语言合成")


def example_4_precompute_speaker():
    """示例 4：预计算说话人特征（多次复用）。"""
    print("\n=== 示例 4：预计算说话人特征 ===")

    tts = TTS(backend="sovits")

    # 第一次：预计算并缓存
    print("预计算说话人特征...")
    features = tts.precompute_speaker(
        ref_audio="my_speaker.wav",
        ref_text="参考文本",
        cache_path="speaker_features.pt",
    )

    # 后续：直接复用，节省时间
    tts_cached = TTS(backend="sovits", speaker_features=features)
    result = tts_cached.run(MosaicData(text="第一次复用", language="zh"))
    result.get("audio").save("output_sovits_cached_1.wav")

    result = tts_cached.run(MosaicData(text="第二次复用", language="zh"))
    result.get("audio").save("output_sovits_cached_2.wav")

    print("已使用缓存特征合成两次，无需重新提取")


async def example_5_streaming():
    """示例 5：流式输出。"""
    print("\n=== 示例 5：流式输出 ===")

    tts = TTS(backend="sovits", streaming=True, language="zh")

    print("开始流式合成...")
    chunk_idx = 0
    start = asyncio.get_event_loop().time()

    async for chunk in tts.synthesize_stream(
        text="GPT-SoVITS 的流式延迟大约 100 毫秒。",
        language="zh",
    ):
        if chunk_idx == 0:
            first = (asyncio.get_event_loop().time() - start) * 1000
            print(f"首批延迟: {first:.0f}ms")
        chunk_idx += 1

    print(f"共流出 {chunk_idx} 个 chunk")


def example_6_batch_synthesis():
    """示例 6：批量合成（高吞吐）。"""
    print("\n=== 示例 6：批量合成 ===")

    tts = TTS(backend="sovits", language="zh")
    texts = ["第一句", "第二句", "第三句"] * 5  # 15 句

    audios = tts.batch_synthesize(texts, language="zh")
    print(f"已批量合成 {len(audios)} 段音频")

    for i, audio in enumerate(audios[:5]):
        audio.save(f"output_sovits_batch_{i}.wav")


def example_7_license_notice():
    """示例 7：许可证说明。"""
    print("\n=== 示例 7：许可证 ===")
    print("GPT-SoVITS 采用 MIT 许可证，可商用。")


def main():
    print("=" * 60)
    print("Mosaic GPT-SoVITS 完整示例")
    print("=" * 60)
    print("注：实际运行需要 4GB+ 显存和 GPT/SoVITS 权重")
    print("=" * 60)

    example_1_basic()
    example_2_minimal_sample_clone()
    example_3_cross_language()
    example_4_precompute_speaker()

    asyncio.run(example_5_streaming())

    example_6_batch_synthesis()
    example_7_license_notice()

    print("=" * 60)
    print("GPT-SoVITS 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
