"""
examples/07_tts_gpt_sovits.py
GPT-SoVITS 完整示例 —— 极少样本克隆、跨语言合成、模型复用、流式输出、批量合成。

依赖：
    pip install "GPT-SoVITS[cpu]"  # 或 [gpu]
    # 权重：参考 https://huggingface.co/lj1995/GPT-SoVITS

运行：
    python examples/07_tts_gpt_sovits.py
"""
from __future__ import annotations

import time

import soundfile as sf

from mosaic.core import MosaicData
from mosaic.nodes.audio import TTS


def example_1_basic():
    """示例 1：基础合成（32kHz）。"""
    print("\n=== 示例 1：基础合成（32kHz）===")

    tts = TTS(backend="sovits", language="zh")
    result = tts.run(MosaicData(
        text="这是使用 GPT-SoVITS 合成的中文语音。",
        language="zh",
        speaker="default",  # 预训练说话人 ID
    ))

    audio = result.get("audio")
    sf.write("output_sovits_basic.wav", audio.waveform, audio.sample_rate)
    print(f"采样率: {audio.sample_rate} Hz, 时长: {result.get('duration'):.2f}s")


def example_2_minimal_sample_clone():
    """示例 2：极少样本克隆（仅需 5-10 秒参考音频）。"""
    print("\n=== 示例 2：极少样本克隆（5-10 秒）===")

    # ref_audio 不是 TTS 的构造参数（传入会触发 TypeError）；
    # 参考音频路径通过 run 调用中的 speaker 参数传入
    tts = TTS(backend="sovits")

    result = tts.run(MosaicData(
        text="这是用 5 秒参考音频克隆的声音，合成任意新文本。",
        language="zh",
        speaker="short_ref.wav",
    ))
    audio = result.get("audio")
    sf.write("output_sovits_cloned.wav", audio.waveform, audio.sample_rate)
    print(f"已克隆：{result.get('duration'):.2f}s")


def example_3_cross_language():
    """示例 3：跨语言合成。"""
    print("\n=== 示例 3：跨语言合成 ===")

    # ref_audio 不是 TTS 的构造参数，参考音频路径通过 speaker 传入
    tts = TTS(backend="sovits")

    # 用中文音色合成英文
    result = tts.run(MosaicData(
        text="This is a Chinese voice speaking English.",
        language="en",
        speaker="chinese_ref.wav",
    ))
    audio = result.get("audio")
    sf.write("output_sovits_cross_lang.wav", audio.waveform, audio.sample_rate)
    print("已实现跨语言合成")


def example_4_reuse_instance():
    """示例 4：复用已加载模型（多次合成）。

    同一个 TTS 实例只需加载一次模型，后续 ``run`` 复用已加载权重，
    避免重复初始化的开销。
    """
    print("\n=== 示例 4：复用已加载模型 ===")

    tts = TTS(backend="sovits", language="zh")

    for i, text in enumerate(["第一次复用", "第二次复用"], 1):
        result = tts.run(MosaicData(text=text, language="zh"))
        audio = result.get("audio")
        sf.write(f"output_sovits_cached_{i}.wav", audio.waveform, audio.sample_rate)
        print(f"  第 {i} 次合成完成（复用已加载模型）")

    print("已复用同一实例合成两次，无需重新加载模型")


def example_5_streaming():
    """示例 5：流式输出。"""
    print("\n=== 示例 5：流式输出 ===")

    tts = TTS(backend="sovits", language="zh")

    print("开始流式合成...")
    chunk_idx = 0
    start = time.time()

    # run_stream 返回同步生成器，每次 yield 一小段 AudioData
    for chunk in tts.run_stream(MosaicData(text="GPT-SoVITS 的流式延迟大约 100 毫秒。", language="zh")):
        if chunk_idx == 0:
            first = (time.time() - start) * 1000
            print(f"首批延迟: {first:.0f}ms")
        chunk_idx += 1

    print(f"共流出 {chunk_idx} 个 chunk")


def example_6_batch_synthesis():
    """示例 6：批量合成（循环调用，复用已加载模型）。"""
    print("\n=== 示例 6：批量合成 ===")

    tts = TTS(backend="sovits", language="zh")
    texts = ["第一句", "第二句", "第三句"] * 5  # 15 句

    for i, text in enumerate(texts):
        result = tts.run(MosaicData(text=text, language="zh"))
        # 仅保存前 5 段作为演示
        if i < 5:
            audio = result.get("audio")
            sf.write(f"output_sovits_batch_{i}.wav", audio.waveform, audio.sample_rate)

    print(f"已批量合成 {len(texts)} 段音频（复用已加载模型）")


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
    example_4_reuse_instance()

    example_5_streaming()

    example_6_batch_synthesis()
    example_7_license_notice()

    print("=" * 60)
    print("GPT-SoVITS 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
