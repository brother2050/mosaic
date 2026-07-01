# mosaic/nodes/image/inpainting.py
"""Inpainting 节点 —— 根据 mask 遮罩区域重新绘制图片内容。

使用 ``diffusers.StableDiffusionXLInpaintPipeline`` 加载 SDXL Inpainting 模型，
根据原始图片、遮罩图片与提示词，仅重绘遮罩标记的区域。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.image._base import BaseImageNode
from mosaic.nodes.image._image_utils import (
    safe_float,
    safe_int,
    validate_guidance_scale,
    validate_image_dimensions,
    validate_num_inference_steps,
)

__all__ = ["Inpainting"]


@registry.register
class Inpainting(BaseImageNode):
    """局部重绘节点。

    根据遮罩区域重新绘制图片内容。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认
        ``"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"``。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> inpaint = Inpainting()
    >>> original = Image.open("photo.jpg")
    >>> mask = Image.open("mask.png")  # 白色区域为待重绘
    >>> result = inpaint(MosaicData(
    ...     image=original,
    ...     mask_image=mask,
    ...     prompt="a red car",
    ... ))
    >>> result["image"].save("result.png")
    """

    name: str = "inpainting"
    description: str = (
        "Inpaint specific regions of an image using a mask. "
        "Only the masked area is regenerated based on the prompt."
    )
    version: str = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(
        self,
        model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)

    def _load_pipeline(self) -> None:
        """加载 StableDiffusionXLInpaintPipeline。"""
        from diffusers import AutoPipelineForInpainting  # type: ignore
        from mosaic.nodes._pipeline_utils import safe_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = safe_load_pipeline(
            AutoPipelineForInpainting,
            self._model_name,
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
            torch_dtype=torch_dtype,
        )

        if self._enable_model_cpu_offload:
            self._apply_optimizations()
        else:
            self._pipeline = self._pipeline.to(self._device)
            self._apply_optimizations()

        self._switch_scheduler()

        self._logger.info(
            "SDXL Inpainting pipeline loaded (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行局部重绘。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image)、``mask_image`` (PIL.Image) 和
            ``prompt`` (str)；可选 ``negative_prompt`` (str)、
            ``num_inference_steps`` (int, 默认 30)、
            ``guidance_scale`` (float, 默认 7.5)、``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``seed`` (int)。

        Raises
        ------
        ValueError
            缺少 ``image`` / ``mask_image`` / ``prompt``。
        TypeError
            ``image`` 或 ``mask_image`` 不是 PIL.Image。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入：优先读取单数 image，回退到复数 images[0]
            image = input_data.get("image")
            if image is None:
                images_list = input_data.get("images")
                if images_list and isinstance(images_list, (list, tuple)):
                    image = images_list[0]
            if image is None:
                raise ValueError(
                    f"Inpainting requires 'image' (PIL.Image), "
                    f"got {type(image).__name__}."
                )
            image = self._ensure_pil_image(image)

            mask_image = input_data.get("mask_image")
            if mask_image is None:
                raise ValueError(
                    f"Inpainting requires 'mask_image' (PIL.Image), "
                    f"got {type(mask_image).__name__}."
                )
            mask_image = self._ensure_pil_image(mask_image)

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

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", 30), "num_inference_steps"
            )
            validate_num_inference_steps(num_inference_steps)
            guidance_scale = safe_float(
                input_data.get("guidance_scale", 7.5), "guidance_scale"
            )
            validate_guidance_scale(guidance_scale)

            seed, generator = self._prepare_seed(input_data.get("seed"))

            # 将 image 和 mask_image resize 到相同尺寸（8 的倍数）
            # 以 image 尺寸为准
            image = self._resize_to_multiple_of_8(image)
            # 校验尺寸上下限（A2/E3：防止过大导致显存溢出）
            validate_image_dimensions(image.size[0], image.size[1])
            if mask_image.size != image.size:
                self._logger.warning(
                    "Mask size %s does not match image size %s; "
                    "resizing mask to match image.",
                    mask_image.size, image.size,
                )
                mask_image = mask_image.resize(image.size)

            # 自动二值化 mask
            mask_image = self._binarize_mask(mask_image)

            # 构造 Pipeline 参数
            pipe_kwargs: dict = {
                "prompt": prompt,
                "image": image,
                "mask_image": mask_image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt

            # 执行推理
            output = self._run_pipeline(**pipe_kwargs)

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
                "image_size": result_image.size if result_image else None,
                "seed": seed,
            },
        )
        return result
