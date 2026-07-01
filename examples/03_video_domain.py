"""
examples/03_video_domain.py
视频域示例 —— 文生视频（Wan / HunyuanVideo / LTX-Video）+ 图生视频 + 视频增强管道。

运行：
    python examples/03_video_domain.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core import MosaicData, VideoData
from mosaic.nodes.video import (
    WanVideo,
    HunyuanVideo,
    LTXVideo,
    TextToVideo,    # CogVideoX
    ImageToVideo,   # SVD
    VideoContinuation,
    FrameInterpolator,
    FrameExtractor,
)
from mosaic.nodes.export import VideoEncoder


def save_video(video: VideoData, path: str) -> None:
    """使用 VideoEncoder 节点将 VideoData 保存为视频文件。

    Parameters
    ----------
    video:
        包含 ``frames`` 和 ``fps`` 的 :class:`VideoData`。
    path:
        输出文件路径（如 ``"output.mp4"``）。
    """
    encoder = VideoEncoder(format="mp4")
    encoder.run(MosaicData(
        frames=video.frames,
        fps=video.fps,
        output_path=path,
    ))


def example_1_wan_video():
    """示例 1：Wan2.1 文生视频（轻量版 1.3B）。"""
    print("\n=== 示例 1：Wan2.1-1.3B 文生视频（~8GB 显存）===")

    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
    )

    result = wan.run(
        MosaicData(
            prompt="A cat walking on the beach at sunset, slow motion",
            num_frames=81,  # 约 5 秒 @ 16fps
            fps=16,
            num_inference_steps=20,
            guidance_scale=5.0,
            seed=42,
        )
    )

    video = result.get("video")
    save_video(video, "output_wan.mp4")
    print(f"已生成：{result.get('num_frames')} 帧, {result.get('duration'):.2f} 秒")


def example_2_wan_high_quality():
    """示例 2：Wan2.1-14B 文生视频（高质量，需 30GB 显存）。"""
    print("\n=== 示例 2：Wan2.1-14B 高质量文生视频（需 30GB 显存）===")

    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
    )

    result = wan.run(
        MosaicData(
            prompt="A young woman walking through a field of sunflowers, golden hour, cinematic",
            num_frames=81,
            fps=16,
            num_inference_steps=30,
            guidance_scale=5.0,
            seed=42,
        )
    )

    save_video(result.get("video"), "output_wan_14b.mp4")
    print(f"已生成 14B 高质量视频：{result.get('duration'):.2f} 秒")


def example_3_hunyuan_video():
    """示例 3：HunyuanVideo 文生视频（需 60GB 显存或 40GB+offload）。"""
    print("\n=== 示例 3：HunyuanVideo 文生视频（需 60GB 显存）===")

    hv = HunyuanVideo(
        model="hunyuanvideo-community/HunyuanVideo",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
    )

    result = hv.run(
        MosaicData(
            prompt="A dancing robot in a neon city, cyberpunk style",
            num_frames=129,
            fps=24,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42,
        )
    )

    save_video(result.get("video"), "output_hunyuan.mp4")
    print(f"已生成 HunyuanVideo：{result.get('duration'):.2f} 秒")


def example_4_ltx_video():
    """示例 4：LTX-Video 轻量快速文生视频。"""
    print("\n=== 示例 4：LTX-Video 快速文生视频（~12GB 显存）===")

    ltx = LTXVideo(
        model="Lightricks/LTX-Video",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
    )

    result = ltx.run(
        MosaicData(
            prompt="A car driving on a mountain road at sunset",
            num_frames=97,  # 约 3 秒 @ 30fps
            fps=30,
            num_inference_steps=20,
            guidance_scale=3.0,
            seed=42,
        )
    )

    save_video(result.get("video"), "output_ltx.mp4")
    print(f"已生成 LTX-Video：{result.get('duration'):.2f} 秒")


def example_5_cogvideox():
    """示例 5：CogVideoX 文生视频（中等显存）。"""
    print("\n=== 示例 5：CogVideoX 文生视频（~18GB 显存）===")

    t2v = TextToVideo(
        model="THUDM/CogVideoX-5b",
        enable_vae_tiling=True,
    )

    result = t2v.run(
        MosaicData(
            prompt="阳光下的向日葵花田，电影感",
            num_frames=49,  # CogVideoX 必须 49 或 85
            fps=8,
        )
    )

    save_video(result.get("video"), "output_cogvideox.mp4")
    print(f"已生成 CogVideoX：{result.get('duration'):.2f} 秒")


def example_6_image_to_video():
    """示例 6：SVD 图生视频。"""
    print("\n=== 示例 6：SVD 图生视频 ===")

    i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid-xt")
    image = Image.open("cat.jpg")

    result = i2v.run(
        MosaicData(
            image=image,
            num_frames=25,
            fps=7,
            motion_bucket_id=127,
        )
    )

    save_video(result.get("video"), "output_i2v.mp4")
    print(f"已生成图生视频：{result.get('duration'):.2f} 秒")


def example_7_video_continuation():
    """示例 7：视频续写。

    使用 ``BaseVideoNode._load_video`` 从文件加载视频为 :class:`VideoData`。
    所有视频节点均继承此静态方法。
    """
    print("\n=== 示例 7：视频续写 ===")

    vc = VideoContinuation(model="THUDM/CogVideoX-5b")
    source_video = VideoContinuation._load_video("output_cogvideox.mp4")

    result = vc.run(
        MosaicData(
            video=source_video,
            overlap_frames=5,
        )
    )

    save_video(result.get("video"), "output_continued.mp4")
    print("已续写视频末尾")


def example_8_frame_interpolation():
    """示例 8：插帧（提高帧率）。"""
    print("\n=== 示例 8：帧插值（8fps → 24fps）===")

    interpolator = FrameInterpolator(method="rife")
    source_video = FrameInterpolator._load_video("output_wan.mp4")

    result = interpolator.run(
        MosaicData(
            video=source_video,
            target_fps=24,
        )
    )

    save_video(result.get("video"), "output_interpolated.mp4")
    print("已将帧率提升到 24")


def example_9_frame_extraction():
    """示例 9：拆帧。

    FrameExtractor 接受字符串路径或 :class:`VideoData` 作为 ``video``
    输入。返回 ``frames`` (list[PIL.Image])，可自行保存到目录。
    """
    print("\n=== 示例 9：拆帧 ===")

    extractor = FrameExtractor()

    # 直接传入视频文件路径字符串
    result = extractor.run(
        MosaicData(
            video="output_wan.mp4",
            mode="all",
        )
    )

    frames = result.get("frames")
    # 将提取的帧保存到目录
    os.makedirs("./frames", exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"./frames/frame_{i:04d}.png")

    print(f"已提取 {result.get('frame_count')} 帧到 ./frames/")


def example_10_complete_video_pipeline():
    """示例 10：完整视频生成流程（生成 → 插帧 → 编码）。

    由于 ``target_fps`` 需要传给 FrameInterpolator 而非 WanVideo，
    且 WanVideo 输出 MosaicData 不包含 ``target_fps``，因此使用
    显式逐节点调用。当节点间字段名匹配时，也可使用 ``|`` 管道。
    """
    print("\n=== 示例 10：完整视频生成流程 ===")

    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        enable_cpu_offload=True,
    )
    interpolator = FrameInterpolator(method="rife")

    # 1. 文生视频（16fps, 49 帧）
    r1 = wan.run(MosaicData(
        prompt="A scenic mountain landscape with moving clouds",
        num_frames=49,
        fps=16,
    ))
    video = r1.get("video")
    print(f"已生成视频：{len(video.frames)} 帧 @ {video.fps}fps")

    # 2. 插帧到 30fps
    r2 = interpolator.run(MosaicData(
        video=video,
        target_fps=30,
    ))
    interpolated = r2.get("video")
    print(f"插帧后：{len(interpolated.frames)} 帧 @ {interpolated.fps}fps")

    # 3. 编码保存
    save_video(interpolated, "output_complete.mp4")
    print("完整视频生成完成")


def main():
    print("=" * 60)
    print("Mosaic 视频域示例")
    print("=" * 60)
    print("注：实际运行需要相应显存的 GPU，未运行仅做 API 演示。")
    print("=" * 60)

    # 轻量级示例可运行；大模型示例需要相应硬件
    example_4_ltx_video()       # 12GB
    example_5_cogvideox()       # 18GB
    example_10_complete_video_pipeline()

    print("=" * 60)
    print("所有视频域示例代码已展示！")
    print("=" * 60)


if __name__ == "__main__":
    main()
