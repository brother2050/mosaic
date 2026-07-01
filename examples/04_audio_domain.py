"""
examples/04_audio_domain.py
音频域示例 —— ASR、音乐生成、音效生成、语音克隆（不含 TTS，TTS 见 05-08）。

运行：
    python examples/04_audio_domain.py
"""
from __future__ import annotations

import soundfile as sf

from mosaic import Pipeline
from mosaic.core import MosaicData
from mosaic.nodes.audio import (
    ASR,
    MusicGenerator,
    SoundEffectGenerator,
    VoiceClone,
)
from mosaic.nodes.text import TextSummarizer


def example_1_asr():
    """示例 1：语音识别（ASR）。"""
    print("\n=== 示例 1：语音识别 ===")

    asr = ASR(model="openai/whisper-large-v3")

    # ASR 接受 str 路径 / ndarray / AudioData 作为 ``audio``
    result = asr.run(MosaicData(audio="speech.wav", language="zh"))

    print(f"识别文本：{result.get('text')}")
    print(f"检测语言：{result.get('language')}")
    print(f"分段数：{len(result.get('segments', []))}")


def example_2_music_generation():
    """示例 2：音乐生成。"""
    print("\n=== 示例 2：音乐生成 ===")

    music_gen = MusicGenerator()
    result = music_gen.run(
        MosaicData(prompt="upbeat lo-fi hip hop, jazzy piano, relaxing beat, 90 BPM", duration=30)
    )

    audio = result.get("audio")
    sf.write("output_music.wav", audio.waveform, audio.sample_rate)
    print(f"已生成背景音乐：{result.get('duration'):.1f} 秒")


def example_3_sound_effect():
    """示例 3：音效生成。"""
    print("\n=== 示例 3：音效生成 ===")

    se = SoundEffectGenerator()
    result = se.run(
        MosaicData(prompt="thunderstorm with heavy rain and distant thunder", duration=5.0)
    )

    audio = result.get("audio")
    sf.write("output_thunder.wav", audio.waveform, audio.sample_rate)
    print(f"已生成音效：{audio.metadata.get('duration', 0):.1f} 秒")


def example_4_voice_clone():
    """示例 4：语音克隆。"""
    print("\n=== 示例 4：语音克隆 ===")

    clone = VoiceClone(language="zh")
    # VoiceClone 用 ``reference_audio`` 提供参考音色，``text`` 为待合成文本
    result = clone.run(MosaicData(
        reference_audio="reference.wav",
        text="这是用 5 秒参考音频克隆的声音。",
    ))

    audio = result.get("audio")
    sf.write("output_clone.wav", audio.waveform, audio.sample_rate)
    print(f"已克隆声音：{audio.metadata.get('duration', 0):.1f} 秒")


def example_5_pipeline_transcribe_to_summarize():
    """示例 5：ASR → 摘要管道。"""
    print("\n=== 示例 5：ASR → 文本摘要管道 ===")

    pipe = ASR(model="openai/whisper-large-v3") | TextSummarizer()

    result = pipe.run(MosaicData(audio="long_speech.wav", mode="concise"))
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
