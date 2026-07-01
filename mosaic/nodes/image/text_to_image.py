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
from mosaic.nodes.image._image_utils import (
    ALIGNMENT_MULTIPLE,
    safe_float,
    safe_int,
    validate_guidance_scale,
    validate_image_dimensions,
    validate_num_inference_steps,
)

__all__ = ["TextToImage"]


@registry.register
class TextToImage(BaseImageNode):
    """文生图节点。

    根据文字提示词生成图片，支持 SDXL、SD 1.5、Z-Image 等 diffusers 模型。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
    pipeline_class:
        可选，手动指定 diffusers Pipeline 类。默认为 ``None``（使用
        ``AutoPipelineForText2Image`` 自动识别）。对于未被 AutoPipeline
        注册的模型（如 ``ZImagePipeline``），需显式传入：

        >>> from diffusers import ZImagePipeline
        >>> t2i = TextToImage(
        ...     model="Tongyi-MAI/Z-Image-Turbo",
        ...     dtype="bfloat16",
        ...     pipeline_class=ZImagePipeline,
        ... )
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

    # 默认输出尺寸（F2：避免魔法数字）
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 1024
    DEFAULT_NUM_INFERENCE_STEPS: int = 30
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_NUM_IMAGES: int = 1

    # 默认负面提示词（与 Stylizer 对齐，显著提升基础质量）
    DEFAULT_NEGATIVE_PROMPT: str = (
        "blurry, low quality, distorted, deformed, watermark, "
        "signature, text, ugly, extra limbs, bad anatomy"
    )

    def _load_pipeline(self) -> None:
        """加载 diffusers Pipeline。

        使用 :func:`auto_load_pipeline` 自动检测 Pipeline 类：
        1. 若用户指定了 ``pipeline_class``，优先使用
        2. 否则用 ``AutoPipelineForText2Image`` 自动匹配
        3. 若 AutoPipeline 不认识该模型，回退到 ``DiffusionPipeline`` 终极检测
        """
        from mosaic.nodes._pipeline_utils import auto_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = auto_load_pipeline(
            self._model_name,
            task="text-to-image",
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
            pipeline_class=self._pipeline_class,
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
            "Pipeline loaded (dtype=%s, device=%s, class=%s).",
            self._dtype_str,
            self._device,
            type(self._pipeline).__name__,
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
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{self.__class__.__name__} requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}: {prompt!r}"
                )

            # 提取参数
            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            width = safe_int(input_data.get("width", self.DEFAULT_WIDTH), "width")
            height = safe_int(input_data.get("height", self.DEFAULT_HEIGHT), "height")
            # 确保尺寸是 8 的倍数
            width = max(ALIGNMENT_MULTIPLE, (width // ALIGNMENT_MULTIPLE) * ALIGNMENT_MULTIPLE)
            height = max(ALIGNMENT_MULTIPLE, (height // ALIGNMENT_MULTIPLE) * ALIGNMENT_MULTIPLE)
            # 校验尺寸上下限（A2/E3：防止过大导致显存溢出）
            validate_image_dimensions(width, height)

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", self.DEFAULT_NUM_INFERENCE_STEPS),
                "num_inference_steps",
            )
            validate_num_inference_steps(num_inference_steps)
            guidance_scale = safe_float(
                input_data.get("guidance_scale", self.DEFAULT_GUIDANCE_SCALE),
                "guidance_scale",
            )
            validate_guidance_scale(guidance_scale)
            num_images = safe_int(input_data.get("num_images", self.DEFAULT_NUM_IMAGES), "num_images")
            if num_images < 1:
                raise ValueError(
                    f"num_images must be >= 1, got {num_images}."
                )

            # 准备种子
            seed, generator = self._prepare_seed(input_data.get("seed"))

            # 构造 Pipeline 参数
            # 未提供 negative_prompt 时使用默认值（显著减少伪影/模糊）
            effective_negative_prompt = negative_prompt
            if effective_negative_prompt is None:
                effective_negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

            pipe_kwargs: dict = {
                "prompt": prompt,
                "negative_prompt": effective_negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
            }

            # 执行推理
            output = self._run_pipeline(**pipe_kwargs)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 提取生成的图片
        images: list[Any] = output.images if hasattr(output, "images") else []
        if not images:
            raise RuntimeError(
                "Pipeline returned no images. This may indicate a model "
                "loading or inference error."
            )

        result = MosaicData(
            images=images,
            image=images[0],  # 兼容下游单数 image 字段
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
