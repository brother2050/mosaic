"""
examples/05_tts_chattts.py
ChatTTS 完整示例 —— 基础合成、韵律控制、随机音色、流式输出。

依赖：
    pip install "mosaic[audio]"
    pip install chattts
    # 权重：首次运行自动从 HF Hub 下载（约 1.2GB）

运行：
    python examples/05_tts_chattts.py
"""
from __future__ import annotations

import asyncio
import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic.core.types import MosaicData
from mosaic.nodes.audio import TTS


def example_1_basic_synthesis():
    """示例 1：基础合成。"""
    print("\n=== 示例 1：基础合成（24kHz）===")

    tts = TTS(backend="chattts", language="zh")
    result = tts.run(text="你好，欢迎使用 Mosaic 框架！", seed=42)

    audio = result.get("audio")
    audio.save("output_chattts_basic.wav")
    print(f"采样率: {audio.sample_rate} Hz, 时长: {audio.duration:.2f}s")


def example_2_prosody_control():
    """示例 2：韵律控制（oral / laugh / break / speed）。"""
    print("\n=== 示例 2：韵律控制 ===")

    tts = TTS(backend="chattts", language="zh")

    # oral - 口语化连接词
    audio = tts.run(
        text="那个[oral_嗯]东西[oral_啊]真是太好用了",
    ).get("audio")
    audio.save("output_chattts_oral.wav")
    print(f"[oral] 已保存")

    # laugh - 笑声
    audio = tts.run(
        text="哈哈哈，这个笑话太好笑了[laugh]",
    ).get("audio")
    audio.save("output_chattts_laugh.wav")
    print(f"[laugh] 已保存")

    # break - 停顿
    audio = tts.run(
        text="第一句话[break]第二句话[break_500]第三句话",
    ).get("audio")
    audio.save("output_chattts_break.wav")
    print(f"[break] 已保存")

    # speed - 局部语速
    audio = tts.run(
        text="[speed_0.8]这部分慢一点[ speed_1.5]这部分快一点",
    ).get("audio")
    audio.save("output_chattts_speed.wav")
    print(f"[speed] 已保存")


def example_3_random_voices():
    """示例 3：随机音色（不同 seed）。"""
    print("\n=== 示例 3：随机音色（seed 控制）===")

    tts = TTS(backend="chattts", language="zh")

    # 相同 seed 总是生成相同音色
    audio1 = tts.run(text="同一句话", seed=42).get("audio")
    audio2 = tts.run(text="同一句话", seed=42).get("audio")
    assert audio1.duration > 0 and audio2.duration > 0
    print("Seed 42 两次生成音色一致")

    # 不同 seed 音色不同
    for seed in [42, 123, 999, 2024, 8888]:
        audio = tts.run(text=f"这是第 {seed} 号声音", seed=seed).get("audio")
        audio.save(f"output_chattts_voice_{seed}.wav")
        print(f"Seed {seed} 已保存")


async def example_4_streaming():
    """示例 4：流式输出（首批延迟 ~50ms）。"""
    print("\n=== 示例 4：流式输出 ===")

    tts = TTS(backend="chattts", language="zh", streaming=True)

    text = "流式合成测试，第一批延迟应该很低。ChatTTS 的延迟是四个后端中最低的。"

    print(f"开始流式合成（文本长度 {len(text)} 字符）...")
    first_chunk_time = None
    chunk_idx = 0
    start = asyncio.get_event_loop().time()

    async for chunk in tts.synthesize_stream(text=text, language="zh"):
        if first_chunk_time is None:
            first_chunk_time = asyncio.get_event_loop().time() - start
            print(f"首批延迟: {first_chunk_time * 1000:.0f}ms")
        chunk_idx += 1
        # 实际播放：play(chunk)
        print(f"  chunk #{chunk_idx}: {chunk.duration * 1000:.0f}ms")

    print(f"共流出 {chunk_idx} 个 chunk")


def example_5_voice_with_video():
    """示例 5：与视频域组合的配音管道。"""
    print("\n=== 示例 5：视频配音管道 ===")
    print("说明：TTS 生成的音频可以与视频节点组合做配音")
    print("  pipe = WanVideo() | TTS(backend='chattts') | VideoEncoder()")


def main():
    print("=" * 60)
    print("Mosaic ChatTTS 完整示例")
    print("=" * 60)
    print("注：实际运行需要 2GB+ 显存和权重文件")
    print("=" * 60)

    example_1_basic_synthesis()
    example_2_prosody_control()
    example_3_random_voices()

    # 异步示例
    asyncio.run(example_4_streaming())

    example_5_voice_with_video()

    print("=" * 60)
    print("ChatTTS 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
