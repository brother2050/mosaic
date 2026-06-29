# mosaic/nodes/video/text_to_video.py
"""TextToVideo 节点 —— 文生视频。

根据文字描述生成视频，基于 THUDM CogVideoX 模型。

设计要点
--------
* 使用 ``diffusers.CogVideoXPipeline`` 加载 CogVideoX-5b（约 18GB 显存）
  或 CogVideoX-2b（约 9GB 显存）。
* 支持 attention_slicing、vae_slicing 显存优化。
* CogVideoX 要求 ``num_frames`` 为特定值（49 或 85），非有效值时
  自动调整为最近的合法值。
* 长视频生成非常耗时，通过 EventBus 实时报告进度。
* 输出统一为 :class:`~mosaic.core.types.VideoData` 格式。

显存需求
--------
* ``THUDM/CogVideoX-5b``：约 16-20GB（fp16）
* ``THUDM/CogVideoX-2b``：约 8-10GB（fp16，推荐显存受限时使用）

许可证
------
* CogVideoX 系列：CogVideoX License (Apache 2.0)
"""

from __future__ import annotations

import random
import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode

__all__ = ["TextToVideo"]


# CogVideoX 支持的有效帧数
_VALID_NUM_FRAMES = [49, 85]


@registry.register
class TextToVideo(BaseVideoNode):
    """文生视频节点。

    根据文字描述生成视频，基于 CogVideoX。

    Parameters
    ----------
    model:
        模型标识，默认 ``"THUDM/CogVideoX-5b"``。
        显存不足时可切换 ``"THUDM/CogVideoX-2b"``。
    device:
        推理设备，默认 ``"cuda"``。
    dtype:
        推理精度，默认 ``"float16"``。
    enable_attention_slicing:
        是否启用 attention slicing 以节省显存，默认 ``True``。
    enable_vae_slicing:
        是否启用 VAE slicing 以节省显存，默认 ``True``。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    >>> t2v = TextToVideo(model="THUDM/CogVideoX-5b")
    >>> result = t2v(MosaicData(
    ...     prompt="一只猫在草地上奔跑，阳光明媚",
    ...     num_frames=49,
    ...     fps=8,
    ... ))
    >>> video = result["video"]  # VideoData

    显存不足时使用 2b 版本：
    >>> t2v = TextToVideo(model="THUDM/CogVideoX-2b")
    """

    name: str = "text-to-video"
    description: str = (
        "Generate video from text descriptions using CogVideoX. "
        "Supports negative prompts, duration control, and guidance scale."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["video"]

    def __init__(
        self,
        model: str = "THUDM/CogVideoX-5b",
        device: str = "cuda",
        dtype: str = "float16",
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, dtype=dtype, **kwargs)
        self._enable_attention_slicing: bool = enable_attention_slicing
        self._enable_vae_slicing: bool = enable_vae_slicing

    def _load_model(self) -> None:
        """加载 CogVideoX Pipeline。"""
        import torch  # type: ignore
        from diffusers import CogVideoXPipeline  # type: ignore

        device = self._resolve_device()
        torch_dtype = self._resolve_dtype()

        self._pipeline = CogVideoXPipeline.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
        )

        # 应用显存优化
        if self._enable_attention_slicing:
            try:
                self._pipeline.enable_attention_slicing()
                self._logger.debug("Enabled attention slicing.")
            except Exception:  # noqa: BLE001
                pass

        if self._enable_vae_slicing:
            vae = getattr(self._pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                try:
                    vae.enable_slicing()
                    self._logger.debug("Enabled VAE slicing (via pipe.vae).")
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    self._pipeline.enable_vae_slicing()
                    self._logger.debug("Enabled VAE slicing (via pipe).")
                except Exception:  # noqa: BLE001
                    pass

        self._pipeline = self._pipeline.to(device)

        self._logger.info(
            "CogVideoX pipeline loaded (model=%s, device=%s, dtype=%s).",
            self._model_name,
            device,
            self._dtype_str,
        )

    def _adjust_num_frames(self, num_frames: int) -> int:
        """调整 num_frames 为 CogVideoX 支持的有效值。

        CogVideoX 仅支持 49 或 85 帧。非有效值时取最近的合法值。

        Parameters
        ----------
        num_frames:
            用户请求的帧数。

        Returns
        -------
        int
            调整后的有效帧数。
        """
        if num_frames in _VALID_NUM_FRAMES:
            return num_frames

        # 找最近的合法值
        closest = min(_VALID_NUM_FRAMES, key=lambda v: abs(v - num_frames))
        self._logger.warning(
            "num_frames=%d is not a valid value for CogVideoX. "
            "Adjusted to %d (valid values: %s).",
            num_frames,
            closest,
            _VALID_NUM_FRAMES,
        )
        return closest

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
        except (RuntimeError, ValueError):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        return seed, generator

    def _extract_frames_from_output(self, output: Any) -> list:
        """从 CogVideoX Pipeline 输出中提取帧列表。

        CogVideoX 输出格式可能因版本而异：
        - ``output.frames``：形状 ``(batch, num_frames, H, W, C)`` 的 tensor
        - ``output.images``：图片列表

        Parameters
        ----------
        output:
            Pipeline 输出对象。

        Returns
        -------
        list[PIL.Image]
            帧列表。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        frames: list = []

        # 尝试从 frames 属性提取
        if hasattr(output, "frames"):
            raw = output.frames
            # 可能是 tensor 或 list
            if hasattr(raw, "cpu"):
                raw = raw.cpu()

            if hasattr(raw, "numpy"):
                arr = raw.numpy()
            else:
                arr = np.asarray(raw)

            # 显式转 float32 避免 float16 精度损失（仅对真实 ndarray）
            if isinstance(arr, np.ndarray) and arr.dtype == np.float16:
                arr = arr.astype(np.float32)

            # 形状 (batch, num_frames, H, W, C) -> 取第一个 batch
            if arr.ndim == 5:
                arr = arr[0]
            # 形状 (num_frames, H, W, C)

            # 归一化到 [0, 255]
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
            ``num_frames`` (int, 默认 49)、``width`` (int, 默认 720)、
            ``height`` (int, 默认 480)、``num_inference_steps`` (int, 默认 50)、
            ``guidance_scale`` (float, 默认 6.0)、``fps`` (int, 默认 8)、
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
            # 校验输入
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"TextToVideo requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            num_frames = int(input_data.get("num_frames", 49))
            num_frames = self._adjust_num_frames(num_frames)

            width = int(input_data.get("width", 720))
            height = int(input_data.get("height", 480))
            # 确保偶数
            width, height = self._ensure_even_dimensions(width, height)

            num_inference_steps = int(input_data.get("num_inference_steps", 50))
            guidance_scale = float(input_data.get("guidance_scale", 6.0))
            fps = int(input_data.get("fps", 8))
            seed = input_data.get("seed")

            actual_seed, generator = self._prepare_seed(seed)

            self._logger.info(
                "Generating video: prompt=%r, num_frames=%d, size=%dx%d, "
                "steps=%d, guidance=%.1f, fps=%d, seed=%d",
                prompt[:50],
                num_frames,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                fps,
                actual_seed,
            )

            self._emit_progress(0, num_inference_steps, "Starting generation")

            # 构造 pipeline 参数
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

            # 执行推理
            import torch  # type: ignore

            try:
                with torch.inference_mode():
                    output = self._pipeline(**pipe_kwargs)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"CUDA out of memory while running {self._model_name}. "
                        f"This model requires ~{_VALID_NUM_FRAMES and 18 or 9}GB VRAM. "
                        f"Try: (1) reduce num_frames to 49, "
                        f"(2) use 'THUDM/CogVideoX-2b' (lighter model), "
                        f"(3) reduce width/height."
                    ) from exc
                raise

            self._emit_progress(
                num_inference_steps, num_inference_steps, "Generation complete"
            )

            # 提取帧
            frames = self._extract_frames_from_output(output)
            if not frames:
                raise RuntimeError(
                    f"Pipeline returned no frames for prompt: {prompt[:50]}"
                )

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 VideoData
        video = self._ensure_video_data(
            frames,
            fps,
            prompt=prompt,
            seed=actual_seed,
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
            },
        )
        return result
