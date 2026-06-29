"""
examples/04_audio_domain.py
音频域示例 —— ASR、音乐生成、音效生成、语音克隆（不含 TTS，TTS 见 05-08）。

运行：
    python examples/04_audio_domain.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic import Pipeline
from mosaic.core.types import AudioData, MosaicData
from mosaic.nodes.audio import (
    ASR,
    MusicGenerator,
    SoundEffect,
    VoiceClone,
)


def example_1_asr():
    """示例 1：语音识别（ASR）。"""
    print("\n=== 示例 1：语音识别 ===")

    asr = ASR(model="openai/whisper-large-v3")
    audio = AudioData.from_file("speech.wav")

    result = asr.run(audio=audio, language="zh")

    print(f"识别文本：{result.get('text')}")
    print(f"检测语言：{result.get('language')}")
    print(f"分段数：{len(result.get('segments', []))}")


def example_2_music_generation():
    """示例 2：音乐生成。"""
    print("\n=== 示例 2：音乐生成 ===")

    music_gen = MusicGenerator()
    result = music_gen.run(
        prompt="upbeat lo-fi hip hop, jazzy piano, relaxing beat, 90 BPM",
        duration=30,
    )

    audio = result.get("audio")
    audio.save("output_music.wav")
    print(f"已生成背景音乐：{audio.duration:.1f} 秒")


def example_3_sound_effect():
    """示例 3：音效生成。"""
    print("\n=== 示例 3：音效生成 ===")

    se = SoundEffect()
    result = se.run(
        prompt="thunderstorm with heavy rain and distant thunder",
        duration=5.0,
    )

    audio = result.get("audio")
    audio.save("output_thunder.wav")
    print(f"已生成音效：{audio.duration:.1f} 秒")


def example_4_voice_clone():
    """示例 4：语音克隆。"""
    print("\n=== 示例 4：语音克隆（GPT-SoVITS）===")

    clone = VoiceClone(backend="sovits")
    result = clone.run(
        text="这是用 5 秒参考音频克隆的声音。",
        ref_audio="reference.wav",
        ref_text="参考音频的文字内容",
    )

    audio = result.get("audio")
    audio.save("output_clone.wav")
    print(f"已克隆声音：{audio.duration:.1f} 秒")


def example_5_pipeline_transcribe_to_summarize():
    """示例 5：ASR → 摘要管道。"""
    print("\n=== 示例 5：ASR → 文本摘要管道 ===")

    pipe = ASR(model="openai/whisper-large-v3") | __import__(
        "mosaic.nodes.text", fromlist=["TextSummarizer"]
    ).TextSummarizer()

    result = pipe.run(
        audio=AudioData.from_file("long_speech.wav"),
        mode="concise",
    )
    print(f"摘要：{result.get('text')}")


def main():
    print("=" * 60)
    print("Mosaic 音频域示例（不含 TTS）")
    print("=" * 60)
    print("TTS 详见 05-08 四个独立示例。")
    print("=" * 60)

    example_1_asr()
    example_2_music_generation()
    example_3_sound_effect()
    example_4_voice_clone()

    print("=" * 60)
    print("所有音频域示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
