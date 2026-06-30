"""
examples/06_tts_fish_speech.py
Fish Speech 完整示例 —— 多语言合成（中/英/日/韩）、语音克隆、流式输出。

依赖：
    pip install fish-speech
    # 权重：HF Hub fishaudio/fish-speech-1.5

运行：
    python examples/06_tts_fish_speech.py
"""
from __future__ import annotations

import asyncio
import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio import TTS
from mosaic import MosaicData


def example_1_basic():
    """示例 1：基础合成（22.05kHz）。"""
    print("\n=== 示例 1：基础合成（22.05kHz）===")

    tts = TTS(backend="fish", language="zh")
    result = tts.run(MosaicData(text="你好世界，这是 Fish Speech 合成。"))

    audio = result.get("audio")
    audio.save("output_fish_basic.wav")
    print(f"采样率: {audio.sample_rate} Hz, 时长: {audio.duration:.2f}s")


def example_2_multilingual():
    """示例 2：多语言合成（中/英/日/韩）。"""
    print("\n=== 示例 2：多语言合成（Fish 多语言支持最佳）===")

    tts = TTS(backend="fish")

    texts = [
        ("zh", "你好，欢迎使用 Mosaic。"),
        ("en", "Hello, welcome to Mosaic."),
        ("ja", "こんにちは、Mosaicへようこそ。"),
        ("ko", "안녕하세요, Mosaic에 오신 것을 환영합니다."),
    ]

    for lang, text in texts:
        result = tts.run(MosaicData(text=text, language=lang))
        audio = result.get("audio")
        audio.save(f"output_fish_{lang}.wav")
        print(f"[{lang}] {text} → {audio.duration:.2f}s")


def example_3_voice_cloning():
    """示例 3：语音克隆（10-30 秒参考音频）。"""
    print("\n=== 示例 3：语音克隆 ===")

    tts = TTS(
        backend="fish",
        ref_audio="reference_voice.wav",   # 10-30 秒参考音频
        ref_text="参考音频的文字内容",
    )

    # 任意新文本都会用参考音频的音色
    result = tts.run(MosaicData(text="这是用参考音频克隆的声音，合成新文本。", language="zh"))
    audio = result.get("audio")
    audio.save("output_fish_cloned.wav")
    print(f"已克隆：{audio.duration:.2f}s")


def example_4_cross_lingual_clone():
    """示例 4：跨语种克隆。"""
    print("\n=== 示例 4：跨语种克隆 ===")

    # 用中文参考音频合成英文
    tts = TTS(
        backend="fish",
        ref_audio="chinese_ref.wav",
        ref_text="这是中文参考音频。",
    )

    result = tts.run(
        text="Cross-lingual voice cloning in English.",
        language="en",
    )
    audio = result.get("audio")
    audio.save("output_fish_cross_lingual.wav")
    print("已实现跨语种克隆")


async def example_5_streaming():
    """示例 5：流式输出。"""
    print("\n=== 示例 5：流式输出 ===")

    tts = TTS(backend="fish", streaming=True, language="zh")

    print("开始流式合成...")
    chunk_idx = 0
    start = asyncio.get_event_loop().time()

    async for chunk in tts.synthesize_stream(
        text="流式合成测试。Fish Speech 延迟约 80 毫秒。",
        language="zh",
    ):
        if chunk_idx == 0:
            first = (asyncio.get_event_loop().time() - start) * 1000
            print(f"首批延迟: {first:.0f}ms")
        chunk_idx += 1

    print(f"共流出 {chunk_idx} 个 chunk")


def example_6_comparison():
    """示例 6：与 ChatTTS 对比（同文本）。"""
    print("\n=== 示例 6：ChatTTS vs Fish Speech 对比 ===")

    text = "这是同一段文本，分别用 ChatTTS 和 Fish Speech 合成。"

    tts_chat = TTS(backend="chattts", language="zh")
    tts_fish = TTS(backend="fish", language="zh")

    audio_chat = tts_chat.run(MosaicData(text=text, seed=42)).get("audio")
    audio_fish = tts_fish.run(MosaicData(text=text)).get("audio")

    print(f"ChatTTS: {audio_chat.sample_rate}Hz, {audio_chat.duration:.2f}s")
    print(f"Fish:    {audio_fish.sample_rate}Hz, {audio_fish.duration:.2f}s")
    print("注意：两者的采样率和时长都不同")


def main():
    print("=" * 60)
    print("Mosaic Fish Speech 完整示例")
    print("=" * 60)
    print("注：实际运行需要 4GB+ 显存和权重文件")
    print("=" * 60)

    example_1_basic()
    example_2_multilingual()
    example_3_voice_cloning()
    example_4_cross_lingual_clone()

    asyncio.run(example_5_streaming())

    example_6_comparison()

    print("=" * 60)
    print("Fish Speech 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
