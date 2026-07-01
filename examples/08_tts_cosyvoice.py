"""
examples/08_tts_cosyvoice.py
CosyVoice 完整示例 —— 高质量多语言合成、克隆、ODE 步数对比、分块流式。

依赖：
    pip install cosyvoice
    # 权重：HF Hub FunAudioLLM/CosyVoice-300M

运行：
    python examples/08_tts_cosyvoice.py
"""
from __future__ import annotations

import time

import soundfile as sf

from mosaic.core import MosaicData
from mosaic.nodes.audio import TTS


def example_1_basic():
    """示例 1：基础合成（24kHz，高质量）。"""
    print("\n=== 示例 1：基础合成（24kHz）===")

    tts = TTS(backend="cosyvoice", language="zh")
    result = tts.run(MosaicData(text="CosyVoice 提供最高质量的语音合成。"))

    audio = result.get("audio")
    sf.write("output_cosyvoice_basic.wav", audio.waveform, audio.sample_rate)
    print(f"采样率: {audio.sample_rate} Hz, 时长: {result.get('duration'):.2f}s")


def example_2_instruct_control():
    """示例 2：SFT 指令控制情感。"""
    print("\n=== 示例 2：情感指令 ===")

    tts = TTS(backend="cosyvoice", language="zh")

    instructs = ["高兴地", "悲伤地", "愤怒地", "惊讶地", "平静地", "兴奋地"]

    for inst in instructs:
        result = tts.run(MosaicData(
            text="今天天气真好",
            language="zh",
            instruct=inst,
        ))
        audio = result.get("audio")
        sf.write(f"output_cosyvoice_{inst}.wav", audio.waveform, audio.sample_rate)
        print(f"  [{inst}] 已保存")


def example_3_ode_steps_benchmark():
    """示例 3：ODE 步数对比（5/10/20 步的质量差异）。"""
    print("\n=== 示例 3：ODE 步数对比 ===")

    text = "这是一段测试文本，用于比较不同 ODE 步数下的合成质量。"

    for steps in [5, 10, 20, 50]:
        tts = TTS(
            backend="cosyvoice",
            language="zh",
            ode_steps=steps,
        )
        result = tts.run(MosaicData(text=text, language="zh"))
        audio = result.get("audio")
        sf.write(f"output_cosyvoice_ode_{steps}.wav", audio.waveform, audio.sample_rate)
        print(f"  步数 {steps:2d}: {result.get('duration'):.2f}s")

    print("\n质量排序（通常）：5 步 < 10 步 < 20 步 ≤ 50 步")
    print("速度排序（通常）：5 步 > 10 步 > 20 步 > 50 步")


def example_4_voice_cloning():
    """示例 4：语音克隆（speech tokens + speaker embedding）。"""
    print("\n=== 示例 4：语音克隆 ===")

    tts = TTS(
        backend="cosyvoice",
        ref_audio="reference_zh.wav",
        ref_text="参考音频的文字内容",
    )

    result = tts.run(MosaicData(
        text="这是用 3-10 秒参考音频克隆的声音。",
        language="zh",
    ))
    audio = result.get("audio")
    sf.write("output_cosyvoice_cloned.wav", audio.waveform, audio.sample_rate)
    print("已克隆")


def example_5_cross_lingual():
    """示例 5：跨语言克隆。"""
    print("\n=== 示例 5：跨语言克隆 ===")

    # 用中文参考音频合成英文
    tts = TTS(
        backend="cosyvoice",
        ref_audio="chinese_ref.wav",
        ref_text="这是中文参考音频。",
    )

    result = tts.run(MosaicData(
        text="This is cross-lingual voice cloning.",
        language="en",
    ))
    audio = result.get("audio")
    sf.write("output_cosyvoice_cross_lingual.wav", audio.waveform, audio.sample_rate)
    print("已实现跨语言克隆")


def example_6_pretrained_speakers():
    """示例 6：使用预训练说话人。"""
    print("\n=== 示例 6：预训练说话人 ===")

    tts = TTS(backend="cosyvoice")

    speakers = ["中文女", "中文男", "英文女", "粤语女", "四川话女"]

    for spk in speakers:
        result = tts.run(MosaicData(
            text=f"我是{spk}音色。",
            language="zh",
            speaker=spk,
        ))
        audio = result.get("audio")
        sf.write(f"output_cosyvoice_speaker_{spk}.wav", audio.waveform, audio.sample_rate)
        print(f"  [{spk}] 已保存")


def example_7_chunk_streaming():
    """示例 7：分块流式输出。"""
    print("\n=== 示例 7：分块流式输出 ===")

    tts = TTS(backend="cosyvoice", language="zh", stream_chunk_size=2048)

    print("开始分块流式合成...")
    chunk_idx = 0
    start = time.time()

    # run_stream 返回同步生成器，每次 yield 一小段 AudioData
    for chunk in tts.run_stream(MosaicData(
        text="CosyVoice 是非自回归模型，需要一次性生成 mel 然后分块播放。",
        language="zh",
    )):
        if chunk_idx == 0:
            first = (time.time() - start) * 1000
            print(f"首批延迟: {first:.0f}ms")
        chunk_idx += 1
        print(f"  chunk #{chunk_idx}: {chunk.metadata.get('duration', 0) * 1000:.0f}ms")

    print(f"共流出 {chunk_idx} 个 chunk")


def example_8_chunk_size_choice():
    """示例 8：stream_chunk_size 选择。"""
    print("\n=== 示例 8：stream_chunk_size 选择 ===")

    # stream_chunk_size 控制每个流式 chunk 的样本数（默认 4096）
    for cs in [1024, 2048, 4096, 8192]:
        tts = TTS(backend="cosyvoice", stream_chunk_size=cs)
        # 不同 chunk_size 的影响：
        # 1024: 最低延迟，但每 chunk 短
        # 2048: 平衡
        # 4096: 默认（推荐）
        # 8192: 最高质量、最低开销
        print(f"  stream_chunk_size={cs}: 延迟与质量权衡")


def main():
    print("=" * 60)
    print("Mosaic CosyVoice 完整示例")
    print("=" * 60)
    print("注：实际运行需要 4GB+ 显存和 CosyVoice-300M 权重")
    print("=" * 60)

    example_1_basic()
    example_2_instruct_control()
    example_3_ode_steps_benchmark()
    example_4_voice_cloning()
    example_5_cross_lingual()
    example_6_pretrained_speakers()

    example_7_chunk_streaming()

    example_8_chunk_size_choice()

    print("=" * 60)
    print("CosyVoice 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
