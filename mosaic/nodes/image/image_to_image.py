# mosaic/nodes/image/image_to_image.py
"""ImageToImage 节点 —— 基于输入图片和 prompt 进行风格转换/修改。

通过 DiffusionPipeline 自动检测加载 SDXL Refiner，
根据输入图片与提示词生成修改后的图片。``strength`` 参数控制变换强度。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.coerce import safe_float, safe_int
from mosaic.nodes.image._base import BaseImageNode
from mosaic.nodes.image._image_utils import (
    validate_guidance_scale,
    validate_image_dimensions,
    validate_num_inference_steps,
    validate_strength,
)

__all__ = ["ImageToImage"]


@registry.register
class ImageToImage(BaseImageNode):
    """图生图节点。

    基于输入图片和文字提示词进行风格转换或内容修改。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-refiner-1.0"``。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> i2i = ImageToImage()
    >>> input_img = Image.open("photo.jpg")
    >>> result = i2i(MosaicData(
    ...     image=input_img,
    ...     prompt="turn this into a watercolor painting",
    ...     strength=0.75,
    ... ))
    >>> result["image"].save("watercolor.png")
    """

    name: str = "image-to-image"
    description: str = (
        "Transform an input image based on a text prompt using SDXL Img2Img. "
        "The 'strength' parameter controls the degree of modification."
    )
    version: str = "0.1.0"
    input_types = ("image", "mosaic")
    output_types = ("image",)

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)

    def _load_pipeline(self) -> None:
        """加载 diffusers Image2Image Pipeline（自动检测）。"""
        from mosaic.nodes._model_loader import auto_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = auto_load_pipeline(
            self._model_name,
            task="image-to-image",
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
            pipeline_class=self._pipeline_class,
            torch_dtype=torch_dtype,
        )

        if self._enable_model_cpu_offload:
            self._apply_optimizations()
        else:
            self._pipeline = self._pipeline.to(self._device)
            self._apply_optimizations()

        self._switch_scheduler()

        self._logger.info(
            "SDXL Img2Img pipeline loaded (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行图生图。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image) 和 ``prompt`` (str)；
            可选 ``negative_prompt`` (str)、``strength`` (float, 默认 0.75)、
            ``num_inference_steps`` (int, 默认 30)、
            ``guidance_scale`` (float, 默认 7.5)、``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``seed`` (int)。

        Raises
        ------
        ValueError
            缺少 ``image`` 或 ``prompt``。
        TypeError
            ``image`` 不是 PIL.Image。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            image = input_data.get("image")
            if image is None:
                raise ValueError(
                    f"ImageToImage requires 'image' (PIL.Image), "
                    f"got {type(image).__name__}."
                )
            image = self._ensure_pil_image(image)

            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{self.__class__.__name__} requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}: {prompt!r}"
                )

            # 提取参数
            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            strength = safe_float(input_data.get("strength", 0.75), "strength")
            validate_strength(strength)

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", 30), "num_inference_steps"
            )
            validate_num_inference_steps(num_inference_steps)
            guidance_scale = safe_float(
                input_data.get("guidance_scale", 7.5), "guidance_scale"
            )
            validate_guidance_scale(guidance_scale)

            seed, generator = self._prepare_seed(input_data.get("seed"))

            # 将输入图片尺寸对齐到 8 的倍数
            image = self._resize_to_multiple_of_8(image)
            # 校验尺寸上下限（A2/E3：防止过大导致显存溢出）
            validate_image_dimensions(image.size[0], image.size[1])

            # 构造 Pipeline 参数
            pipe_kwargs: dict = {
                "prompt": prompt,
                "image": image,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt

            # 执行推理
            output = self._run_pipeline(**pipe_kwargs)

            # 提取结果
            result_image = output.images[0] if hasattr(output, "images") and output.images else None
            if result_image is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} failed to generate output image. "
                    f"The model returned None. This may indicate an issue with "
                    f"the input parameters or model state."
                )
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        result = MosaicData(
            image=result_image,
            seed=seed,
            prompt=prompt,
            model_name=self._model_name,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "strength": strength,
                "image_size": result_image.size if result_image else None,
                "seed": seed,
            },
        )
        return result
