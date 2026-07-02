# mosaic/nodes/image/stylizer.py
"""Stylizer 节点 —— 将图片转换为指定艺术风格。

底层复用 ImageToImage 的推理逻辑，自动根据目标风格构造提示词，
通过控制 ``strength`` 参数实现不同程度的风络化效果。

支持的预设风格包括：油画（oil painting）、水彩（watercolor）、动漫
（anime）、赛博朋克（cyberpunk）、铅笔素描（pencil sketch）等。
用户也可通过 ``prompt_extra`` 传入自定义补充提示词。
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

__all__ = ["Stylizer"]


# 预设风格提示词模板
_STYLE_PROMPTS: dict = {
    "oil painting": "oil painting style, thick brushstrokes, rich textures, masterpiece, highly detailed",
    "watercolor": "watercolor painting style, soft washes, delicate colors, artistic, masterpiece",
    "anime": "anime style, cel shading, vibrant colors, clean lines, high quality illustration",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dystopian, blade runner aesthetic, highly detailed",
    "pencil sketch": "pencil sketch style, graphite drawing, detailed shading, monochrome, artistic",
    "ink": "ink wash painting style, sumi-e, bold strokes, minimalist, traditional art",
    "pixel art": "pixel art style, 16-bit, retro game aesthetic, detailed pixel work",
    "3d render": "3D render style, octane render, cinematic lighting, photorealistic, highly detailed",
    "impressionist": "impressionist painting style, visible brush strokes, light and color focus, monet-like",
    "digital art": "digital art style, concept art, trending on artstation, highly detailed, vibrant",
}


@registry.register
class Stylizer(BaseImageNode):
    """风格化节点。

    将输入图片转换为指定的艺术风格。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
    reference_image:
        可选的风格参考图，启用 IP-Adapter 模式时使用（高级功能，需额外安装
        ``diffusers`` 的 IP-Adapter 支持）。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> stylizer = Stylizer()
    >>> photo = Image.open("photo.jpg")
    >>> result = stylizer(MosaicData(
    ...     image=photo,
    ...     style="oil painting",
    ...     strength=0.65,
    ... ))
    >>> result["image"].save("stylized.png")

    Notes
    -----
    ``strength`` 参数控制风格化强度，建议范围 0.5-0.7：
    * 0.3-0.5：轻微风格化，保留原图大部分结构
    * 0.5-0.7：最佳效果，风格明显但保留构图
    * 0.7-0.9：强风格化，原图结构可能被改变
    """

    name: str = "stylizer"
    description: str = (
        "Stylize an image into a specified artistic style (oil painting, "
        "watercolor, anime, etc.). Uses SDXL Img2Img under the hood."
    )
    version: str = "0.1.0"
    input_types = ("image", "mosaic")
    output_types = ("image",)

    # IP-Adapter 默认权重（F2：避免魔法数字）
    DEFAULT_IP_ADAPTER_SCALE: float = 0.6

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        reference_image: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._reference_image: Any = reference_image
        self._ip_adapter_loaded: bool = False

    def _load_pipeline(self) -> None:
        """加载 StableDiffusionXLImg2ImgPipeline（复用图生图流程）。"""
        from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore
        from mosaic.nodes._model_loader import safe_load_pipeline

        torch_dtype = self._resolve_dtype()

        self._pipeline = safe_load_pipeline(
            StableDiffusionXLImg2ImgPipeline,
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

        # 可选：加载 IP-Adapter（参考图驱动风格化）
        if self._reference_image is not None:
            self._load_ip_adapter()

        self._logger.info(
            "Stylizer pipeline loaded (dtype=%s, device=%s, ip_adapter=%s).",
            self._dtype_str,
            self._device,
            self._ip_adapter_loaded,
        )

    def _load_ip_adapter(self) -> None:
        """加载 IP-Adapter 用于参考图驱动的风格化（高级功能）。"""
        try:
            self._pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.safetensors",
            )
            self._ip_adapter_loaded = True
            self._logger.info("IP-Adapter loaded for reference-image stylization.")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load IP-Adapter: %s. Falling back to prompt-only stylization.",
                exc,
            )
            self._ip_adapter_loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行图片风格化。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image) 和 ``style`` (str)；
            可选 ``strength`` (float, 默认 0.65)、``prompt_extra`` (str)、
            ``num_inference_steps`` (int, 默认 30)、``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``style`` (str)、``seed`` (int)。

        Raises
        ------
        ValueError
            缺少 ``image`` 或 ``style``。
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
                raise ValueError("Stylizer requires 'image' (PIL.Image).")
            image = self._ensure_pil_image(image)

            style = input_data.get("style")
            if not isinstance(style, str) or not style.strip():
                raise ValueError(
                    f"Stylizer requires 'style' (str), got {type(style).__name__}."
                )

            # 提取参数
            strength = safe_float(input_data.get("strength", 0.65), "strength")
            validate_strength(strength)

            prompt_extra = input_data.get("prompt_extra", "")
            if not isinstance(prompt_extra, str):
                prompt_extra = ""

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", 30), "num_inference_steps"
            )
            validate_num_inference_steps(num_inference_steps)
            guidance_scale = safe_float(
                input_data.get("guidance_scale", 7.5), "guidance_scale"
            )
            validate_guidance_scale(guidance_scale)

            seed, generator = self._prepare_seed(input_data.get("seed"))

            # 构造风格化 prompt
            style_prompt = self._build_style_prompt(style, prompt_extra)
            negative_prompt = (
                "blurry, low quality, distorted, deformed, "
                "watermark, signature, text, ugly"
            )

            # 将输入图片尺寸对齐到 8 的倍数
            image = self._resize_to_multiple_of_8(image)
            # 校验尺寸上下限（A2/E3：防止过大导致显存溢出）
            validate_image_dimensions(image.size[0], image.size[1])

            # 构造 Pipeline 参数
            pipe_kwargs: dict = {
                "prompt": style_prompt,
                "image": image,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "generator": generator,
            }

            # IP-Adapter 模式（参考图驱动）
            if self._ip_adapter_loaded and self._reference_image is not None:
                ref_image = self._ensure_pil_image(self._reference_image)
                ref_image = self._resize_to_multiple_of_8(ref_image, (1024, 1024))
                pipe_kwargs["ip_adapter_image"] = ref_image
                # 设置 IP-Adapter scale
                try:
                    self._pipeline.set_ip_adapter_scale(self.DEFAULT_IP_ADAPTER_SCALE)
                except Exception:  # noqa: BLE001
                    pass

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
            style=style,
            seed=seed,
            prompt=style_prompt,
            model_name=self._model_name,
            ip_adapter_enabled=self._ip_adapter_loaded,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "style": style,
                "strength": strength,
                "image_size": result_image.size if result_image else None,
                "seed": seed,
            },
        )
        return result

    @staticmethod
    def _build_style_prompt(style: str, prompt_extra: str) -> str:
        """构造风格化提示词。

        如果 ``style`` 是预设风格之一，使用对应的模板；否则将 ``style``
        本身作为风格描述。

        Parameters
        ----------
        style:
            目标风格标识或自定义描述。
        prompt_extra:
            额外的提示词补充。

        Returns
        -------
        str
            完整的风格化提示词。
        """
        # 尝试匹配预设风格
        style_lower = style.lower().strip()
        base_prompt = _STYLE_PROMPTS.get(style_lower)
        if base_prompt is None:
            # 自定义风格：直接使用用户输入
            base_prompt = f"{style} style, masterpiece, highly detailed"

        if prompt_extra.strip():
            return f"{base_prompt}, {prompt_extra}"
        return base_prompt

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含支持的预设风格列表。"""
        spec = super().describe()
        spec.model_info["supported_styles"] = list(_STYLE_PROMPTS.keys())
        spec.model_info["ip_adapter"] = self._ip_adapter_loaded
        return spec
