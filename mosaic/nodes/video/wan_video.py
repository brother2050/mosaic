# mosaic/nodes/video/wan_video.py
"""WanVideo 节点 —— 基于 Wan2.1 / Wan2.2 的文生视频。

支持 Wan-AI 的 Wan2.1 和 Wan2.2 系列视频生成模型，通过 diffusers 的
``WanPipeline`` 加载。Wan 系列使用 DiT (Diffusion Transformer) 架构，
支持中英文提示词，输出高质量视频。

设计要点
--------
* 使用 ``diffusers.WanPipeline`` 加载模型，需 diffusers >= 0.33.0
  （Wan2.2 需 >= 0.35.0）。
* **重要**：HF 仓库需使用 ``-Diffusers`` 后缀的版本，例如
  ``Wan-AI/Wan2.1-T2V-14B-Diffusers``（非 ``Wan-AI/Wan2.1-T2V-14B``）。
  若用户传入不带后缀的名称，自动添加 ``-Diffusers``。
* 支持 ``enable_model_cpu_offload()`` 和 ``vae.enable_tiling()`` 显存优化。
* 默认输出 16fps，支持自定义帧数（推荐 81 帧约 5 秒视频）。
* 支持负向提示词和 guidance_scale 控制。

显存需求
--------
* ``Wan-AI/Wan2.1-T2V-14B-Diffusers``：约 30GB（fp16）
* ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``：约 8GB（fp16，轻量版）
* ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``：约 30GB（fp16）

许可证
------
* Wan2.1 / Wan2.2 系列：Apache 2.0
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.coerce import safe_float, safe_int
from mosaic.nodes.video._base import BaseVideoNode
from mosaic.nodes.video._video_utils import (
    extract_frames_from_output,
    prepare_seed,
    validate_common_video_params,
    validate_model_path,
)

__all__ = ["WanVideo"]


# Wan 系列的默认参数
_WAN_DEFAULT_FPS = 16
_WAN_DEFAULT_FRAMES = 81  # 约 5 秒 @ 16fps
_WAN_DEFAULT_STEPS = 30
_WAN_DEFAULT_GUIDANCE = 5.0
_WAN_DEFAULT_SIZE = (1280, 720)  # (width, height)


@registry.register
class WanVideo(BaseVideoNode):
    """Wan2.1 / Wan2.2 文生视频节点。

    基于 Wan-AI 的 DiT 视频生成模型，支持中英文提示词。

    Parameters
    ----------
    model:
        模型标识，默认 ``"Wan-AI/Wan2.1-T2V-14B-Diffusers"``。
        显存不足时可切换 ``"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"``。
        支持 Wan2.2：``"Wan-AI/Wan2.2-T2V-A14B-Diffusers"``。
        **注意**：必须使用带 ``-Diffusers`` 后缀的仓库名。
        若传入不带后缀的名称，会自动添加。
    device:
        推理设备，默认 ``"cuda"``。
    dtype:
        推理精度，默认 ``"float16"``。Wan2.2 推荐 ``"bfloat16"``。
    enable_cpu_offload:
        是否启用 ``enable_model_cpu_offload()``，默认 ``True``。
        将模型各组件按需从 CPU 移到 GPU，显著降低显存峰值。
    enable_vae_tiling:
        是否启用 VAE tiling，默认 ``True``。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    >>> wan = WanVideo(model="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    >>> result = wan(MosaicData(
    ...     prompt="一只猫在海滩上散步，夕阳西下",
    ...     num_frames=81,
    ...     fps=16,
    ... ))
    >>> video = result["video"]  # VideoData

    显存不足时使用 1.3B 版本：
    >>> wan = WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    使用 Wan2.2：
    >>> wan = WanVideo(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", dtype="bfloat16")
    """

    name: str = "wan-video"
    description: str = (
        "Generate video from text using Wan2.1/Wan2.2 DiT models. "
        "Supports Chinese & English prompts, negative prompts, "
        "and VAE tiling for memory efficiency."
    )
    version: str = "0.1.0"
    input_types = ("text", "mosaic")
    output_types = ("video",)

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        device: str = "cuda",
        dtype: str = "float16",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, dtype=dtype, **kwargs)
        self._enable_cpu_offload: bool = enable_cpu_offload
        self._enable_vae_tiling: bool = enable_vae_tiling

    def _resolve_model_name(self) -> str:
        """确保模型名带 ``-Diffusers`` 后缀。

        Wan 系列在 HF 上有两个仓库：
        - ``Wan-AI/Wan2.1-T2V-14B``：原始格式（不能直接用 from_pretrained）
        - ``Wan-AI/Wan2.1-T2V-14B-Diffusers``：diffusers 格式

        用户可能传入不带后缀的名称，此处自动补全。
        """
        name = self._model_name
        # 已带后缀则不处理
        if name.endswith("-Diffusers"):
            return name
        # 检查是否是 Wan 仓库（F3：去除冗余条件判断）
        if name.startswith("Wan-AI/"):
            return name + "-Diffusers"
        return name

    def _load_model(self) -> None:
        """加载 Wan Pipeline。"""
        import os
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
        from mosaic.nodes._model_loader import safe_load_pipeline

        # 校验模型路径：本地路径不存在时给出友好错误（B2）
        validate_model_path(self._model_name, self._logger)

        _device = self._resolve_device()
        if _device.startswith("cuda"):
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        torch_dtype = self._resolve_dtype()
        model_name = self._resolve_model_name()
        self._model_name = model_name  # 更新为带后缀的名称

        self._pipeline = safe_load_pipeline(
            DiffusionPipeline,
            model_name,
            torch_dtype=torch_dtype,
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
        )

        # 显存优化
        if self._enable_cpu_offload:
            try:
                self._pipeline.enable_model_cpu_offload()
                self._logger.info("Enabled model CPU offload for Wan.")
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to enable CPU offload: %s", exc)
                self._pipeline = self._pipeline.to(_device)
        else:
            self._pipeline = self._pipeline.to(_device)

        if self._enable_vae_tiling:
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                try:
                    vae.enable_tiling()
                    self._logger.debug("Enabled VAE tiling for Wan.")
                except Exception:  # noqa: BLE001
                    pass

        self._logger.info(
            "Wan pipeline loaded (model=%s, device=%s, dtype=%s, "
            "cpu_offload=%s, vae_tiling=%s).",
            model_name,
            _device,
            self._dtype_str,
            self._enable_cpu_offload,
            self._enable_vae_tiling,
        )

    def _prepare_seed(self, seed: int | None) -> tuple:
        """准备随机种子与 generator。"""
        return prepare_seed(seed, self._infer_device())

    def _extract_frames_from_output(self, output: Any) -> list:
        """从 Wan Pipeline 输出中提取帧列表。"""
        return extract_frames_from_output(output, self._logger)

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文生视频。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``negative_prompt`` (str)、
            ``num_frames`` (int, 默认 81)、``width`` (int, 默认 1280)、
            ``height`` (int, 默认 720)、``num_inference_steps`` (int, 默认 30)、
            ``guidance_scale`` (float, 默认 5.0)、``fps`` (int, 默认 16)、
            ``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``video`` (VideoData)、``prompt`` (str)、``seed`` (int)、
            ``num_frames`` (int)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``prompt`` 或 ``prompt`` 非字符串。
        RuntimeError
            显存不足时抛出，附带建议。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"WanVideo requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            num_frames = safe_int(
                input_data.get("num_frames", _WAN_DEFAULT_FRAMES), "num_frames"
            )
            # Wan 要求 num_frames + 1 为 4 的倍数（首帧）
            # 常见值：49(3s), 81(5s), 121(7s)
            if (num_frames - 1) % 4 != 0:
                # 调整到最近的合法值
                adjusted = round((num_frames - 1) / 4) * 4 + 1
                self._logger.warning(
                    "num_frames=%d adjusted to %d (must be 4k+1).",
                    num_frames, adjusted,
                )
                num_frames = adjusted

            width = safe_int(input_data.get("width", _WAN_DEFAULT_SIZE[0]), "width")
            height = safe_int(input_data.get("height", _WAN_DEFAULT_SIZE[1]), "height")
            width, height = self._ensure_even_dimensions(width, height)

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", _WAN_DEFAULT_STEPS),
                "num_inference_steps",
            )
            guidance_scale = safe_float(
                input_data.get("guidance_scale", _WAN_DEFAULT_GUIDANCE),
                "guidance_scale",
            )
            fps = safe_int(input_data.get("fps", _WAN_DEFAULT_FPS), "fps")
            seed = input_data.get("seed")

            # 参数范围校验（A2）
            validate_common_video_params(
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            actual_seed, generator = self._prepare_seed(seed)

            self._logger.info(
                "Wan generating: prompt=%r, frames=%d, size=%dx%d, "
                "steps=%d, guidance=%.1f, fps=%d, seed=%d",
                prompt[:50], num_frames, width, height,
                num_inference_steps, guidance_scale, fps, actual_seed,
            )

            self._emit_progress(0, num_inference_steps, "Starting Wan generation")

            pipe_kwargs: dict = {
                "prompt": prompt,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "height": height,
                "width": width,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt

            import torch  # type: ignore

            try:
                with torch.inference_mode():
                    output = self._pipeline(**pipe_kwargs)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    # 大小写不敏感匹配模型规模（B1）
                    model_name_lower = self._model_name.lower()
                    is_lite = "1.3b" in model_name_lower
                    min_vram = 8 if is_lite else 30
                    raise RuntimeError(
                        f"CUDA out of memory while running {self._model_name}. "
                        f"This model requires ~{min_vram}GB VRAM. "
                        f"Try: (1) reduce num_frames to 49, "
                        f"(2) use 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' (lighter), "
                        f"(3) reduce width/height, "
                        f"(4) ensure enable_cpu_offload=True, "
                        f"(5) ensure enable_vae_tiling=True."
                    ) from exc
                raise

            self._emit_progress(
                num_inference_steps, num_inference_steps,
                "Wan generation complete",
            )

            frames = self._extract_frames_from_output(output)
            if not frames:
                raise RuntimeError(
                    f"Wan pipeline returned no frames for prompt: {prompt[:50]}"
                )

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        video = self._ensure_video_data(
            frames, fps, prompt=prompt, seed=actual_seed,
        )
        duration = video.metadata.get("duration", 0.0)

        result = MosaicData(
            video=video,
            frames=frames,  # 暴露顶层 frames 供 VideoEncoder 直接读取
            fps=fps,  # 暴露顶层 fps 供 VideoEncoder 直接读取
            prompt=prompt,
            seed=actual_seed,
            num_frames=len(frames),
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "num_frames": len(frames),
                "duration": duration,
                "fps": fps,
                "seed": actual_seed,
                "model": self._model_name,
            },
        )
        return result
