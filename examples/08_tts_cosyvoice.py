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
    """示例 2：SFT 指令控制情感。

    注意：``TTS.run`` 不读取 ``instruct`` 参数 —— 传入会被静默忽略，
    无法实现情感指令控制。CosyVoice 的指令控制（instruct）属于后端特有
    能力，需直接使用 ``CosyVoiceBackend``，而非 ``TTS`` 节点：

        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )
        backend = CosyVoiceBackend(model_path="model_path")
        backend.load(device="cuda", dtype="float16")
        for inst in ["高兴地", "悲伤地", "愤怒地"]:
            audio = backend.synthesize("今天天气真好", language="zh", instruct=inst)
    """
    print("\n=== 示例 2：情感指令 ===")
    print("说明：TTS 节点不支持 instruct，需直接使用 CosyVoiceBackend（见 docstring）。")


def example_3_ode_steps_benchmark():
    """示例 3：ODE 步数对比（5/10/20 步的质量差异）。

    注意：``ode_steps`` 不是 ``TTS`` 节点的构造参数（传入会触发
    ``TypeError``）。ODE 步数属于 CosyVoice 后端的推理参数，需直接使用
    ``CosyVoiceBackend`` 并通过 ``num_ode_steps`` 指定，且在 ``synthesize``
    前需调用 ``load``：

        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )
        text = "这是一段测试文本，用于比较不同 ODE 步数下的合成质量。"
        for steps in [5, 10, 20, 50]:
            backend = CosyVoiceBackend(model_path="model_path", num_ode_steps=steps)
            backend.load(device="cuda", dtype="float16")
            audio = backend.synthesize(text, language="zh")
    """
    print("\n=== 示例 3：ODE 步数对比 ===")
    print("说明：TTS 节点不支持 ode_steps，需直接使用 CosyVoiceBackend（见 docstring）。")
    print("质量排序（通常）：5 步 < 10 步 < 20 步 ≤ 50 步")
    print("速度排序（通常）：5 步 > 10 步 > 20 步 > 50 步")


def example_4_voice_cloning():
    """示例 4：语音克隆（speech tokens + speaker embedding）。"""
    print("\n=== 示例 4：语音克隆 ===")

    # ref_audio/ref_text 不是 TTS 的构造参数（传入会触发 TypeError）；
    # 参考音频路径通过 run 调用中的 speaker 参数传入
    tts = TTS(backend="cosyvoice")

    result = tts.run(MosaicData(
        text="这是用 3-10 秒参考音频克隆的声音。",
        language="zh",
        speaker="reference_zh.wav",
    ))
    audio = result.get("audio")
    sf.write("output_cosyvoice_cloned.wav", audio.waveform, audio.sample_rate)
    print("已克隆")


def example_5_cross_lingual():
    """示例 5：跨语言克隆。"""
    print("\n=== 示例 5：跨语言克隆 ===")

    # 用中文参考音频合成英文
    # ref_audio/ref_text 不是 TTS 的构造参数，参考音频路径通过 speaker 传入
    tts = TTS(backend="cosyvoice")

    result = tts.run(MosaicData(
        text="This is cross-lingual voice cloning.",
        language="en",
        speaker="chinese_ref.wav",
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
