"""
examples/10_digital_human.py
数字人域示例 —— 形象驱动、口型同步、动作生成、完整数字人管道。

运行：
    python examples/10_digital_human.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic import Pipeline
from mosaic.core.types import ImageData, AudioData, MosaicData
from mosaic.nodes.digital_human import (
    AvatarDriver,
    LipSyncer,
    MotionGenerator,
    RealtimeRenderer,
)
from mosaic.nodes.image import TextToImage
from mosaic.nodes.audio import TTS


def example_1_avatar_driver():
    """示例 1：形象驱动（姿态/表情驱动静态形象）。"""
    print("\n=== 示例 1：形象驱动 ===")

    driver = AvatarDriver()
    avatar = ImageData.from_file("portrait.jpg")
    # 假设已有驱动数据（姿态/表情）
    motion = {"pose": "smile", "head_rotation": 15}

    result = driver.run(avatar=avatar, motion=motion)
    result.get("output_image").save("output_avatar_driven.jpg")
    print("已生成驱动的数字人形象")


def example_2_lip_sync():
    """示例 2：口型同步。"""
    print("\n=== 示例 2：口型同步 ===")

    lip_sync = LipSyncer()
    video_frames = ImageData.load_video("talking_head_frames/")
    audio = AudioData.from_file("speech.wav")

    result = lip_sync.run(video=video_frames, audio=audio)
    result.get("video").save("output_lipsync.mp4")
    print("已对口型同步")


def example_3_motion_generation():
    """示例 3：动作生成。"""
    print("\n=== 示例 3：动作生成 ===")

    motion_gen = MotionGenerator()
    result = motion_gen.run(
        prompt="a person waving hand and nodding",
        duration=5.0,
    )

    motion_data = result.get("motion")
    print(f"已生成动作序列: {len(motion_data['frames'])} 帧")


def example_4_realtime_renderer():
    """示例 4：实时渲染。"""
    print("\n=== 示例 4：实时渲染（流式）===")

    renderer = RealtimeRenderer()

    # 模拟实时输入流
    print("启动实时渲染器...")
    renderer.start(session_id="demo_session")

    for i in range(10):
        renderer.push_frame(
            session_id="demo_session",
            frame=ImageData.random((512, 512)),
        )
        print(f"  推送第 {i+1} 帧")

    renderer.stop(session_id="demo_session")


def example_5_complete_digital_human_pipeline():
    """示例 5：完整数字人管道（文 → 形象 → TTS → 口型同步 → 视频）。"""
    print("\n=== 示例 5：完整数字人管道 ===")

    pipe = (
        TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
        | AvatarDriver()
        | TTS(backend="chattts", language="zh")
        | LipSyncer()
    )

    result = pipe.run(
        prompt="a friendly female digital assistant, smile, professional",
        text="你好，我是数字人助手，很高兴见到你。",
    )

    print("已生成完整数字人视频")
    if "video" in result.outputs:
        result.get("video").save("output_digital_human.mp4")


def example_6_tts_lipsync_shortcut():
    """示例 6：简化版 TTS + 口型同步。"""
    print("\n=== 示例 6：TTS → 口型同步（简化）===")

    pipe = TTS(backend="cosyvoice", language="zh") | LipSyncer()
    result = pipe.run(
        text="这是一段测试语音，数字人应该匹配口型。",
        avatar=ImageData.from_file("avatar.jpg"),
    )
    print("已生成带口型同步的视频")


def example_7_motion_to_video():
    """示例 7：动作 → 视频（MotionGenerator + AvatarDriver）。"""
    print("\n=== 示例 7：动作 → 视频 ===")

    pipe = MotionGenerator(duration=3.0) | AvatarDriver()
    result = pipe.run(prompt="a person walking forward")
    print("已根据动作描述生成数字人视频")


def main():
    print("=" * 60)
    print("Mosaic 数字人域示例")
    print("=" * 60)
    print("注：实际运行需要相应的视觉模型权重")
    print("=" * 60)

    example_1_avatar_driver()
    example_2_lip_sync()
    example_3_motion_generation()
    example_4_realtime_renderer()
    example_5_complete_digital_human_pipeline()
    example_6_tts_lipsync_shortcut()
    example_7_motion_to_video()

    print("=" * 60)
    print("所有数字人域示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
