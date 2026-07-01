"""
examples/02_image_domain.py
图像域示例 —— 6 个图像节点 + 图像处理管道。

运行：
    python examples/02_image_domain.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core import MosaicData
from mosaic.nodes.image import (
    TextToImage,
    ImageToImage,
    Inpainting,
    Upscaler,
    BackgroundRemover,
    Stylizer,
)


def example_1_text_to_image():
    """示例 1：文生图。"""
    print("\n=== 示例 1：文生图 ===")

    t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
    result = t2i.run(MosaicData(
        prompt="A cute cat sitting on a windowsill, soft morning light, 4K",
        negative_prompt="blurry, low quality",
        seed=42,
    ))

    images = result.get("images")
    images[0].save("output_text_to_image.png")
    print(f"已生成：{images[0].size}, 保存到 output_text_to_image.png")
    return images[0]


def example_2_image_to_image():
    """示例 2：图生图（风格迁移）。"""
    print("\n=== 示例 2：图生图 ===")

    i2i = ImageToImage(model="stabilityai/stable-diffusion-xl-refiner-1.0")
    input_image = Image.open("input.jpg")  # 假设有 input.jpg

    result = i2i.run(
        MosaicData(
            image=input_image,
            prompt="the same scene in watercolor painting style",
            strength=0.6,
            seed=42,
        )
    )

    image = result.get("image")
    image.save("output_image_to_image.png")
    print(f"已生成风格化图像：{image.size}")


def example_3_inpainting():
    """示例 3：局部重绘。"""
    print("\n=== 示例 3：局部重绘 ===")

    inpaint = Inpainting(model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

    # 模拟输入图和蒙版（直接使用 PIL.Image）
    image = Image.open("room.jpg")
    mask = Image.open("mask.png")  # 白色区域为重绘区

    result = inpaint.run(MosaicData(
        image=image,
        mask_image=mask,
        prompt="a beautiful flower garden",
        seed=42,
    ))

    result.get("image").save("output_inpainting.png")
    print("已重绘指定区域")


def example_4_upscaler():
    """示例 4：超分辨率。

    注意：``scale_factor`` 是运行时输入字段，不是构造参数。
    """
    print("\n=== 示例 4：超分（4x）===")

    upscaler = Upscaler(model="stabilityai/stable-diffusion-x4-upscaler")
    low_res = Image.open("low_res.jpg")

    result = upscaler.run(MosaicData(image=low_res, scale_factor=4))
    image = result.get("image")
    image.save("output_upscaled.png")
    print(f"已放大：{low_res.size} → {image.size}")


def example_5_background_remover():
    """示例 5：去背景。"""
    print("\n=== 示例 5：去背景 ===")

    remover = BackgroundRemover(model="briaai/RMBG-2.0")
    image = Image.open("portrait.jpg")

    result = remover.run(MosaicData(image=image))
    result.get("image").save("output_no_bg.png")
    print("已移除背景（输出 RGBA 透明）")


def example_6_stylizer():
    """示例 6：艺术风格化。"""
    print("\n=== 示例 6：风格化 ===")

    stylizer = Stylizer()
    image = Image.open("photo.jpg")

    result = stylizer.run(
        MosaicData(
            image=image,
            style="oil painting, impressionist",
            strength=0.8,
        )
    )
    result.get("image").save("output_stylized.png")
    print("已应用艺术风格")


def example_7_combined_pipeline():
    """示例 7：组合管道（生成 → 去背景 → 4x 超分）。

    管道中各节点的输出字段与下一节点的输入字段名称匹配：
    - TextToImage 输出 ``image`` (PIL.Image) → BackgroundRemover 需要 ``image``
    - BackgroundRemover 输出 ``image`` (PIL.Image) → Upscaler 需要 ``image``

    注意：``scale_factor`` 无法通过管道透传到 Upscaler（中间节点会
    创建新的 MosaicData），Upscaler 默认 ``scale_factor=4``，因此
    最终仍为 4x 放大。
    """
    print("\n=== 示例 7：组合管道（生成 → 去背景 → 4x 超分）===")

    pipe = (
        TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
        | BackgroundRemover(model="briaai/RMBG-2.0")
        | Upscaler()
    )

    result = pipe.run(MosaicData(
        prompt="a beautiful flower, isolated on white background",
        seed=42,
    ))

    result.get("image").save("output_combined.png")
    print("已生成去背景并 4x 超分的高清图")


def example_8_video_model_note():
    """示例 8：视频模型接入说明。"""
    print("\n=== 示例 8：视频模型接入（说明）===")
    print(
        "Mosaic 已集成 Wan2.1/2.2、HunyuanVideo、LTX-Video 等视频模型。\n"
        "使用方法见 examples/03_video_domain.py。\n"
        "  - WanVideo(model='Wan-AI/Wan2.1-T2V-14B-Diffusers')"
    )
    print("  - HunyuanVideo(model='hunyuanvideo-community/HunyuanVideo')")
    print("  - LTXVideo(model='Lightricks/LTX-Video')")


def main():
    print("=" * 60)
    print("Mosaic 图像域示例")
    print("=" * 60)

    example_1_text_to_image()
    example_7_combined_pipeline()
    example_8_video_model_note()

    print("=" * 60)
    print("所有图像域示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
