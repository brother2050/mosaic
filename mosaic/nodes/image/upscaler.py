# mosaic/nodes/image/upscaler.py
"""Upscaler 节点 —— 将低分辨率图片放大并增强画质。

使用 ``diffusers.StableDiffusionUpscalePipeline`` 加载 SD x4 Upscaler 模型，
将输入图片放大到更高分辨率，同时通过扩散过程增强细节。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.image._base import BaseImageNode
from mosaic.nodes.image._image_utils import (
    safe_int,
    validate_num_inference_steps,
)

__all__ = ["Upscaler"]


@registry.register
class Upscaler(BaseImageNode):
    """超分辨率节点。

    将低分辨率图片放大并增强画质。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认
        ``"stabilityai/stable-diffusion-x4-upscaler"``。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> upscaler = Upscaler()
    >>> low_res = Image.open("low_res.png")
    >>> result = upscaler(MosaicData(
    ...     image=low_res,
    ...     prompt="highly detailed, sharp focus",
    ... ))
    >>> result["image"].save("high_res.png")
    """

    name: str = "upscaler"
    description: str = (
        "Upscale a low-resolution image to higher resolution while enhancing "
        "details using Stable Diffusion x4 Upscaler."
    )
    version: str = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-x4-upscaler",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)

    def _load_pipeline(self) -> None:
        """加载 StableDiffusionUpscalePipeline。"""
        from diffusers import StableDiffusionUpscalePipeline  # type: ignore
        from mosaic.nodes._pipeline_utils import safe_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = safe_load_pipeline(
            StableDiffusionUpscalePipeline,
            self._model_name,
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
            "SD Upscale pipeline loaded (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行超分辨率放大。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image)；可选 ``prompt`` (str)、
            ``scale_factor`` (int, 默认 4)、``num_inference_steps`` (int, 默认 20)、
            ``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``original_size`` (tuple)、
            ``output_size`` (tuple)。

        Raises
        ------
        ValueError
            缺少 ``image``。
        TypeError
            ``image`` 不是 PIL.Image。
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
                    f"Upscaler requires 'image' (PIL.Image), "
                    f"got {type(image).__name__}."
                )
            image = self._ensure_pil_image(image)

            # 提取参数
            prompt = input_data.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = ""

            scale_factor = safe_int(input_data.get("scale_factor", 4), "scale_factor")
            scale_factor = max(2, min(8, scale_factor))  # 限制在 2-8 倍

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", 20), "num_inference_steps"
            )
            validate_num_inference_steps(num_inference_steps)

            seed, generator = self._prepare_seed(input_data.get("seed"))

            original_size: tuple[int, int] = image.size

            # 检查输入图片尺寸是否过小
            if min(original_size) < 64:
                self._logger.warning(
                    "Input image is very small (%dx%d). Results may be poor.",
                    original_size[0],
                    original_size[1],
                )

            # 限制输入图片最大边长，防止放大后超出显存
            # SD Upscaler 期望输入最大边 512 左右
            image = self._limit_image_size(image, max_side=512)

            # 对齐到 8 的倍数
            image = self._resize_to_multiple_of_8(image)

            # 构造 Pipeline 参数
            pipe_kwargs: dict = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }

            # 执行推理
            output = self._run_pipeline(**pipe_kwargs)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        result_image = output.images[0] if hasattr(output, "images") and output.images else None
        if result_image is None:
            raise RuntimeError(
                f"Upscaler pipeline returned no image for model {self._model_name!r}. "
                f"This may indicate a VAE decode failure."
            )
        output_size = result_image.size if result_image else None

        result = MosaicData(
            image=result_image,
            original_size=original_size,
            output_size=output_size,
            seed=seed,
            model_name=self._model_name,
            scale_factor=scale_factor,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "original_size": original_size,
                "output_size": output_size,
                "seed": seed,
            },
        )
        return result
