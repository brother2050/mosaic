# mosaic/nodes/video/hunyuan_video.py
"""HunyuanVideo 节点 —— 基于腾讯混元视频生成模型。

使用 ``diffusers.HunyuanVideoPipeline`` 加载腾讯 HunyuanVideo 模型。
HunyuanVideo 是一个大规模视频生成模型，支持中英文提示词，输出高分辨率视频。

设计要点
--------
* 使用 ``diffusers.HunyuanVideoPipeline`` 加载，需 diffusers >= 0.32.0。
* HF 仓库 ``tencent/HunyuanVideo`` 可直接使用 ``from_pretrained``。
* 支持 ``enable_model_cpu_offload()`` 和 ``vae.enable_tiling()``。
* 模型较大（~13B 参数），建议使用 ``bfloat16`` 精度。
* 默认输出 24fps，支持自定义帧数。

显存需求
--------
* ``tencent/HunyuanVideo``：约 60GB（fp16/bf16）
  启用 CPU offload 后可在 40GB 显卡上运行。

许可证
------
* Tencent Hunyuan Video License
"""

from __future__ import annotations

import random
import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode

__all__ = ["HunyuanVideo"]

# HunyuanVideo 默认参数
_HY_DEFAULT_FPS = 24
_HY_DEFAULT_FRAMES = 129  # 约 5 秒 @ 24fps
_HY_DEFAULT_STEPS = 30
_HY_DEFAULT_GUIDANCE = 7.5
_HY_DEFAULT_SIZE = (1280, 720)


@registry.register
class HunyuanVideo(BaseVideoNode):
    """HunyuanVideo 文生视频节点。

    基于腾讯混元大规模视频生成模型。

    Parameters
    ----------
    model:
        模型标识，默认 ``"tencent/HunyuanVideo"``。
    device:
        推理设备，默认 ``"cuda"``。
    dtype:
        推理精度，默认 ``"bfloat16"``（HunyuanVideo 推荐 bf16）。
    enable_cpu_offload:
        是否启用 ``enable_model_cpu_offload()``，默认 ``True``。
    enable_vae_tiling:
        是否启用 VAE tiling，默认 ``True``。
    enable_chunking:
        是否启用 ``enable_chunking()``（HunyuanVideo 专属优化，
        将 VAE 解码分块处理），默认 ``True``。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    >>> hv = HunyuanVideo()
    >>> result = hv(MosaicData(
    ...     prompt="一只猫在草地上奔跑",
    ...     num_frames=129,
    ...     fps=24,
    ... ))
    >>> video = result["video"]
    """

    name: str = "hunyuan-video"
    description: str = (
        "Generate video from text using Tencent HunyuanVideo. "
        "Supports Chinese & English prompts, high-resolution output."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["video"]

    def __init__(
        self,
        model: str = "tencent/HunyuanVideo",
        device: str = "cuda",
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
        enable_chunking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, dtype=dtype, **kwargs)
        self._enable_cpu_offload: bool = enable_cpu_offload
        self._enable_vae_tiling: bool = enable_vae_tiling
        self._enable_chunking: bool = enable_chunking

    def _load_model(self) -> None:
        """加载 HunyuanVideo Pipeline。"""
        import os
        import torch  # type: ignore
        from diffusers import HunyuanVideoPipeline  # type: ignore
        from mosaic.nodes._pipeline_utils import safe_load_pipeline

        _device = self._resolve_device()
        if _device.startswith("cuda"):
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        torch_dtype = self._resolve_dtype()

        self._pipeline = safe_load_pipeline(
            HunyuanVideoPipeline,
            self._model_name,
            torch_dtype=torch_dtype,
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
        )

        # 显存优化
        if self._enable_cpu_offload:
            try:
                self._pipeline.enable_model_cpu_offload()
                self._logger.info("Enabled model CPU offload for HunyuanVideo.")
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
                    self._logger.debug("Enabled VAE tiling for HunyuanVideo.")
                except Exception:  # noqa: BLE001
                    pass

        # HunyuanVideo 专属：VAE chunking
        if self._enable_chunking:
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_chunking"):
                try:
                    vae.enable_chunking()
                    self._logger.debug("Enabled VAE chunking for HunyuanVideo.")
                except Exception:  # noqa: BLE001
                    pass

        self._logger.info(
            "HunyuanVideo pipeline loaded (model=%s, device=%s, dtype=%s, "
            "cpu_offload=%s, vae_tiling=%s, chunking=%s).",
            self._model_name, _device, self._dtype_str,
            self._enable_cpu_offload, self._enable_vae_tiling,
            self._enable_chunking,
        )

    def _prepare_seed(self, seed: int | None) -> tuple:
        """准备随机种子与 generator。"""
        import torch  # type: ignore

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed) % (2**32)

        device = self._infer_device()
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        except (RuntimeError, ValueError, TypeError):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        return seed, generator

    def _extract_frames_from_output(self, output: Any) -> list:
        """从 HunyuanVideo Pipeline 输出中提取帧列表。"""
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        frames: list = []

        if hasattr(output, "frames"):
            raw = output.frames
            if hasattr(raw, "cpu"):
                raw = raw.cpu()

            if hasattr(raw, "numpy"):
                arr = raw.numpy()
            else:
                arr = np.asarray(raw)

            if isinstance(arr, np.ndarray) and arr.dtype == np.float16:
                arr = arr.astype(np.float32)

            if arr.ndim == 5:
                arr = arr[0]

            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

            for i in range(arr.shape[0]):
                frames.append(Image.fromarray(arr[i]))

        elif hasattr(output, "images"):
            frames = list(output.images)

        return frames

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文生视频。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``negative_prompt`` (str)、
            ``num_frames`` (int, 默认 129)、``width`` (int, 默认 1280)、
            ``height`` (int, 默认 720)、``num_inference_steps`` (int, 默认 30)、
            ``guidance_scale`` (float, 默认 7.5)、``fps`` (int, 默认 24)、
            ``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``video`` (VideoData)、``prompt`` (str)、``seed`` (int)、
            ``num_frames`` (int)、``duration`` (float)。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"HunyuanVideo requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            num_frames = int(input_data.get("num_frames", _HY_DEFAULT_FRAMES))
            width = int(input_data.get("width", _HY_DEFAULT_SIZE[0]))
            height = int(input_data.get("height", _HY_DEFAULT_SIZE[1]))
            width, height = self._ensure_even_dimensions(width, height)

            num_inference_steps = int(
                input_data.get("num_inference_steps", _HY_DEFAULT_STEPS)
            )
            guidance_scale = float(
                input_data.get("guidance_scale", _HY_DEFAULT_GUIDANCE)
            )
            fps = int(input_data.get("fps", _HY_DEFAULT_FPS))
            seed = input_data.get("seed")

            actual_seed, generator = self._prepare_seed(seed)

            self._logger.info(
                "HunyuanVideo generating: prompt=%r, frames=%d, size=%dx%d, "
                "steps=%d, guidance=%.1f, fps=%d, seed=%d",
                prompt[:50], num_frames, width, height,
                num_inference_steps, guidance_scale, fps, actual_seed,
            )

            self._emit_progress(0, num_inference_steps, "Starting HunyuanVideo")

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
                    raise RuntimeError(
                        f"CUDA out of memory while running {self._model_name}. "
                        f"This model requires ~60GB VRAM. "
                        f"Try: (1) reduce num_frames, "
                        f"(2) reduce width/height, "
                        f"(3) ensure enable_cpu_offload=True, "
                        f"(4) ensure enable_vae_tiling=True, "
                        f"(5) ensure enable_chunking=True."
                    ) from exc
                raise

            self._emit_progress(
                num_inference_steps, num_inference_steps,
                "HunyuanVideo generation complete",
            )

            frames = self._extract_frames_from_output(output)
            if not frames:
                raise RuntimeError(
                    f"HunyuanVideo returned no frames for prompt: {prompt[:50]}"
                )

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        video = self._ensure_video_data(
            frames, fps, prompt=prompt, seed=actual_seed,
        )
        duration = video.metadata.get("duration", 0.0)

        result = MosaicData(
            video=video,
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
