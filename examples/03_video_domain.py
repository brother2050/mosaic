"""
examples/03_video_domain.py
视频域示例 —— 文生视频（Wan / HunyuanVideo / LTX-Video）+ 图生视频 + 视频增强管道。

运行：
    python examples/03_video_domain.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/mosaic")

from mosaic import Pipeline
from mosaic.core.types import ImageData, VideoData
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


def example_1_wan_video():
    """示例 1：Wan2.1 文生视频（轻量版 1.3B）。"""
    print("\n=== 示例 1：Wan2.1-1.3B 文生视频（~8GB 显存）===")

    wan = WanVideo(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
    )

    result = wan.run(
        prompt="A cat walking on the beach at sunset, slow motion",
        num_frames=81,  # 约 5 秒 @ 16fps
        fps=16,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=42,
    )

    video = result.get("video")
    video.save("output_wan.mp4")
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
        prompt="A young woman walking through a field of sunflowers, golden hour, cinematic",
        num_frames=81,
        fps=16,
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )

    result.get("video").save("output_wan_14b.mp4")
    print(f"已生成 14B 高质量视频：{result.get('duration'):.2f} 秒")


def example_3_hunyuan_video():
    """示例 3：HunyuanVideo 文生视频（需 60GB 显存或 40GB+offload）。"""
    print("\n=== 示例 3：HunyuanVideo 文生视频（需 60GB 显存）===")

    hv = HunyuanVideo(
        model="tencent/HunyuanVideo",
        enable_cpu_offload=True,
        enable_vae_tiling=True,
        enable_chunking=True,
    )

    result = hv.run(
        prompt="A dancing robot in a neon city, cyberpunk style",
        num_frames=129,
        fps=24,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42,
    )

    result.get("video").save("output_hunyuan.mp4")
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
        prompt="A car driving on a mountain road at sunset",
        num_frames=97,  # 约 3 秒 @ 30fps
        fps=30,
        num_inference_steps=20,
        guidance_scale=3.0,
        seed=42,
    )

    result.get("video").save("output_ltx.mp4")
    print(f"已生成 LTX-Video：{result.get('duration'):.2f} 秒")


def example_5_cogvideox():
    """示例 5：CogVideoX 文生视频（中等显存）。"""
    print("\n=== 示例 5：CogVideoX 文生视频（~18GB 显存）===")

    t2v = TextToVideo(
        model="THUDM/CogVideoX-5b",
        enable_vae_tiling=True,
    )

    result = t2v.run(
        prompt="阳光下的向日葵花田，电影感",
        num_frames=49,  # CogVideoX 必须 49 或 85
        fps=8,
    )

    result.get("video").save("output_cogvideox.mp4")
    print(f"已生成 CogVideoX：{result.get('duration'):.2f} 秒")


def example_6_image_to_video():
    """示例 6：SVD 图生视频。"""
    print("\n=== 示例 6：SVD 图生视频 ===")

    i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid-xt")
    image = ImageData.from_file("cat.jpg")

    result = i2v.run(
        image=image,
        num_frames=25,
        fps=7,
        motion_bucket_id=127,
    )

    result.get("video").save("output_i2v.mp4")
    print(f"已生成图生视频：{result.get('duration'):.2f} 秒")


def example_7_video_continuation():
    """示例 7：视频续写。"""
    print("\n=== 示例 7：视频续写 ===")

    vc = VideoContinuation(model="THUDM/CogVideoX-5b")
    source_video = VideoData.from_file("output_cogvideox.mp4")

    result = vc.run(
        video=source_video,
        overlap_frames=5,
    )

    result.get("video").save("output_continued.mp4")
    print("已续写视频末尾")


def example_8_frame_interpolation():
    """示例 8：插帧（提高帧率）。"""
    print("\n=== 示例 8：帧插值（8fps → 24fps）===")

    interpolator = FrameInterpolator(method="rife")
    source_video = VideoData.from_file("output_wan.mp4")

    result = interpolator.run(
        video=source_video,
        target_fps=24,
    )

    result.get("video").save("output_interpolated.mp4")
    print("已将帧率从 16 提升到 24")


def example_9_frame_extraction():
    """示例 9：拆帧。"""
    print("\n=== 示例 9：拆帧 ===")

    extractor = FrameExtractor()
    source_video = VideoData.from_file("output_wan.mp4")

    result = extractor.run(
        video=source_video,
        output_dir="./frames",
    )

    print(f"已提取 {result.get('frame_count')} 帧到 ./frames/")


def example_10_complete_video_pipeline():
    """示例 10：完整视频生成管道（生成 → 插帧 → 编码）。"""
    print("\n=== 示例 10：完整视频生成管道 ===")

    pipe = (
        WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", enable_cpu_offload=True)
        | FrameInterpolator(method="rife")
    )

    result = pipe.run(
        prompt="A scenic mountain landscape with moving clouds",
        num_frames=49,
        fps=16,
        target_fps=30,  # 插帧到 30fps
    )

    result.get("video").save("output_complete.mp4")
    print("完整视频生成完成：49 帧@30fps")


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
