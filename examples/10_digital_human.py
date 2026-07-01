"""
examples/10_digital_human.py
数字人域示例 —— 形象驱动、口型同步、动作生成、完整数字人管道。

运行：
    python examples/10_digital_human.py
"""
from __future__ import annotations

from PIL import Image

from mosaic import Pipeline
from mosaic.core import MosaicData, VideoData
from mosaic.nodes.digital_human import (
    AvatarDriver,
    LipSyncer,
    MotionGenerator,
    RealtimeRenderer,
)
from mosaic.nodes.image import TextToImage
from mosaic.nodes.audio import TTS
from mosaic.nodes.export import VideoEncoder


def save_video(video: VideoData, path: str) -> None:
    """使用 VideoEncoder 节点将 VideoData 保存为视频文件。"""
    encoder = VideoEncoder(format="mp4")
    encoder.run(MosaicData(
        frames=video.frames,
        fps=video.fps,
        output_path=path,
    ))


def example_1_avatar_driver():
    """示例 1：形象驱动（姿态/表情驱动静态形象）。"""
    print("\n=== 示例 1：形象驱动 ===")

    driver = AvatarDriver()
    avatar = Image.open("portrait.jpg")
    # 用表情参数驱动（smile / mouth_open / eye_openness）
    expression_params = [{"smile": 0.8, "mouth_open": 0.3, "eye_openness": 1.0}]

    result = driver.run(MosaicData(
        source_image=avatar,
        expression_params=expression_params,
        output_format="frames",
    ))

    frames = result.get("frames")
    frames[0].save("output_avatar_driven.jpg")
    print("已生成驱动的数字人形象")


def example_2_lip_sync():
    """示例 2：口型同步。"""
    print("\n=== 示例 2：口型同步 ===")

    lip_sync = LipSyncer()
    # face_image 接受 PIL.Image / 路径 / VideoData / 帧列表；audio 接受路径/ndarray/AudioData
    result = lip_sync.run(MosaicData(
        face_image="face.jpg",
        audio="speech.wav",
        output_format="video",
        fps=25,
    ))

    save_video(result.get("video"), "output_lipsync.mp4")
    print("已对口型同步")


def example_3_motion_generation():
    """示例 3：动作生成。"""
    print("\n=== 示例 3：动作生成 ===")

    motion_gen = MotionGenerator(method="text2motion")
    result = motion_gen.run(MosaicData(
        prompt="a person waving hand and nodding",
        duration=5.0,
    ))

    motion = result.get("motion")
    print(f"已生成动作序列: {motion.frame_count} 帧")


def example_4_realtime_renderer():
    """示例 4：实时渲染（回调式流式输入 → 流式输出帧）。"""
    print("\n=== 示例 4：实时渲染（流式）===")

    renderer = RealtimeRenderer(enable_tts=True, target_fps=25)
    print("启动实时渲染器...")

    # 模拟实时输入流：文本片段（enable_tts 时自动合成语音）
    sentences = ["你好，世界！", "今天天气真好。", "数字人渲染测试。"]
    queue = [("text", s) for s in sentences]

    def input_callback():
        # 返回 (类型, 数据) 元组；返回 None 表示输入结束
        return queue.pop(0) if queue else None

    def output_callback(frame):
        print(f"  收到渲染帧: {frame.size}")

    renderer.start_realtime(
        source_image="avatar.png",
        input_callback=input_callback,
        output_callback=output_callback,
    )
    renderer.stop_realtime()

    stats = renderer.get_stats()
    print(f"渲染统计: {stats.get('total_frames', 0)} 帧")


def example_5_complete_digital_human_pipeline():
    """示例 5：完整数字人管道（文 → 形象 → TTS → 口型同步 → 视频）。

    由于各节点字段名不完全匹配（TextToImage 输出 ``image``，
    AvatarDriver 需要 ``source_image``；TTS 输出 ``audio``，
    LipSyncer 需要 ``face_image`` + ``audio``），使用显式逐节点调用。
    """
    print("\n=== 示例 5：完整数字人管道 ===")

    t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
    avatar_driver = AvatarDriver()
    tts = TTS(backend="chattts", language="zh")
    lip_sync = LipSyncer()

    # 1. 文生形象
    r1 = t2i.run(MosaicData(
        prompt="a friendly female digital assistant, smile, professional",
    ))
    avatar = r1.get("image")
    print("已生成数字人形象")

    # 2. 形象驱动（用表情参数驱动）
    r2 = avatar_driver.run(MosaicData(
        source_image=avatar,
        expression_params=[{"smile": 0.8, "mouth_open": 0.2, "eye_openness": 1.0}],
        output_format="frames",
    ))
    driven_frames = r2.get("frames")
    print(f"已驱动形象：{len(driven_frames)} 帧")

    # 3. TTS 生成语音
    r3 = tts.run(MosaicData(text="你好，我是数字人助手，很高兴见到你。", language="zh"))
    audio = r3.get("audio")
    print("已生成语音")

    # 4. 口型同步（用驱动后的首帧 + 音频）
    r4 = lip_sync.run(MosaicData(
        face_image=driven_frames[0],
        audio=audio,
        output_format="video",
        fps=25,
    ))
    save_video(r4.get("video"), "output_digital_human.mp4")
    print("已生成完整数字人视频")


def example_6_tts_lipsync_shortcut():
    """示例 6：简化版 TTS + 口型同步。

    管道 ``TTS | LipSyncer``：TTS 读取 ``text``、输出 ``audio``，
    ``audio`` 自动传递给 LipSyncer；``face_image`` 由输入提供。
    """
    print("\n=== 示例 6：TTS → 口型同步（简化）===")

    pipe = TTS(backend="cosyvoice", language="zh") | LipSyncer()
    result = pipe.run(MosaicData(
        text="这是一段测试语音，数字人应该匹配口型。",
        face_image=Image.open("avatar.jpg"),
    ))
    save_video(result.get("video"), "output_lipsync_shortcut.mp4")
    print("已生成带口型同步的视频")


def example_7_motion_to_video():
    """示例 7：动作 → 视频（MotionGenerator 生成动作，AvatarDriver 驱动形象）。"""
    print("\n=== 示例 7：动作 → 视频 ===")

    # 1. 根据文本描述生成动作序列
    motion_gen = MotionGenerator(method="text2motion")
    m_result = motion_gen.run(MosaicData(prompt="a person walking forward", duration=3.0))
    motion = m_result.get("motion")
    print(f"已生成动作序列: {motion.frame_count} 帧")

    # 2. 用动作驱动静态形象生成视频
    driver = AvatarDriver()
    result = driver.run(MosaicData(
        source_image=Image.open("avatar.png"),
        expression_params=motion.keypoints,
        output_format="video",
        fps=25,
    ))
    save_video(result.get("video"), "output_motion_video.mp4")
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
