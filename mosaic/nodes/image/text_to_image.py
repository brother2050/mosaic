# mosaic/nodes/image/text_to_image.py
"""TextToImage 节点 —— 根据文字 prompt 生成图片。

使用 ``diffusers.StableDiffusionXLPipeline`` 加载 SDXL 基础模型，根据正向/
反向提示词生成图像。支持控制输出分辨率、推理步数、引导系数与随机种子。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.image._base import BaseImageNode

__all__ = ["TextToImage"]


@registry.register
class TextToImage(BaseImageNode):
    """文生图节点。

    根据文字提示词生成图片，基于 Stable Diffusion XL。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
    >>> result = t2i(MosaicData(
    ...     prompt="a cat sitting on a windowsill, oil painting style",
    ...     negative_prompt="blurry, low quality",
    ...     width=1024, height=1024,
    ... ))
    >>> result["images"][0].save("cat.png")
    """

    name: str = "text-to-image"
    description: str = (
        "Generate images from text prompts using Stable Diffusion XL. "
        "Supports negative prompts, resolution, steps, guidance scale, and seed."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["image"]

    def _load_pipeline(self) -> None:
        """加载 StableDiffusionXLPipeline。"""
        from diffusers import AutoPipelineForText2Image  # type: ignore
        from mosaic.nodes._pipeline_utils import safe_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = safe_load_pipeline(
            AutoPipelineForText2Image,
            self._model_name,
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
            torch_dtype=torch_dtype,
        )

        # 迁移到目标设备
        if self._enable_model_cpu_offload:
            self._apply_optimizations()
            # cpu_offload 会自行管理设备迁移
        else:
            self._pipeline = self._pipeline.to(self._device)
            self._apply_optimizations()

        self._switch_scheduler()

        self._logger.info(
            "SDXL pipeline loaded (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文生图。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``negative_prompt`` (str)、
            ``width`` (int, 默认 1024)、``height`` (int, 默认 1024)、
            ``num_inference_steps`` (int, 默认 30)、``guidance_scale``
            (float, 默认 7.5)、``seed`` (int)、``num_images`` (int, 默认 1)。

        Returns
        -------
        MosaicData
            包含 ``images`` (list[PIL.Image])、``seed`` (int)、
            ``prompt`` (str)、``model_name`` (str)。

        Raises
        ------
        ValueError
            缺少 ``prompt`` 或 ``prompt`` 非字符串。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"TextToImage requires 'prompt' (str), "
                    f"got {type(prompt).__name__}."
                )

            # 提取参数
            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            width = int(input_data.get("width", 1024))
            height = int(input_data.get("height", 1024))
            # 确保尺寸是 8 的倍数
            width = max(8, (width // 8) * 8)
            height = max(8, (height // 8) * 8)

            num_inference_steps = int(input_data.get("num_inference_steps", 30))
            guidance_scale = float(input_data.get("guidance_scale", 7.5))
            num_images = int(input_data.get("num_images", 1))
            num_images = max(1, num_images)

            # 准备种子
            seed, generator = self._prepare_seed(input_data.get("seed"))

            # 构造 Pipeline 参数
            pipe_kwargs: dict = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt

            # 执行推理
            output = self._run_pipeline(**pipe_kwargs)
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 提取生成的图片
        images: list[Any] = output.images if hasattr(output, "images") else []

        result = MosaicData(
            images=images,
            seed=seed,
            prompt=prompt,
            model_name=self._model_name,
            num_images=len(images),
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "num_images": len(images),
                "width": width,
                "height": height,
                "seed": seed,
            },
        )
        return result
