# mosaic/nodes/video/image_to_video.py
"""ImageToVideo 节点 —— 图生视频。

根据输入图片生成短视频，基于 Stability AI 的 Stable Video Diffusion (SVD)。

设计要点
--------
* 使用 ``diffusers.StableVideoDiffusionPipeline`` 加载 SVD-XT 模型。
* SVD 不接受文字 prompt，视频运动幅度由 ``motion_bucket_id``
  （范围 1-255，默认 127）控制：值越大运动越剧烈。
* 输入图片需统一 resize 到 1024x576（SVD 训练分辨率）。
* SVD-XT 固定输出 25 帧，超出请求帧数或模型上限时自动截断。
* ``noise_level``（对应 pipeline 的 ``noise_aug_level``）控制输入图片
  的噪声增强程度，影响生成稳定性，默认 0.02。
* 支持 VAE slicing 与 ``decode_chunk_size`` 显存优化。
* 输出统一为 :class:`~mosaic.core.types.VideoData` 格式。

显存需求
--------
* ``stabilityai/stable-video-diffusion-img2vid-xt``：约 10-12GB（fp16）
* ``stabilityai/stable-video-diffusion-img2vid``：约 8-10GB（fp16，
  14 帧，显存受限时推荐使用）

许可证
------
* Stable Video Diffusion 系列：Stability AI Community License
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode

__all__ = ["ImageToVideo"]


# SVD-XT 固定输出帧数（模型上限）
_SVD_XT_NUM_FRAMES = 25

# SVD 输入图片目标尺寸 (width, height)
_SVD_INPUT_SIZE = (1024, 576)

# motion_bucket_id 默认值（合法范围 1-255）
_MOTION_BUCKET_ID_DEFAULT = 127

# noise_level 默认值
_NOISE_LEVEL_DEFAULT = 0.02

# SVD-XT 粗略显存需求（fp16, GB），用于错误提示
_SVD_XT_VRAM_GB = 12.0


@registry.register
class ImageToVideo(BaseVideoNode):
    """图生视频节点。

    根据输入图片生成短视频，基于 Stable Video Diffusion (SVD)。
    SVD 不使用文字 prompt，运动幅度由 ``motion_bucket_id`` 控制。

    Parameters
    ----------
    model:
        模型标识，默认 ``"stabilityai/stable-video-diffusion-img2vid-xt"``。
        显存不足时可切换 ``"stabilityai/stable-video-diffusion-img2vid"``
        （14 帧，更轻量）。
    device:
        推理设备，默认 ``"cuda"``。
    dtype:
        推理精度，默认 ``"float16"``。
    enable_vae_slicing:
        是否启用 VAE slicing 以节省显存，默认 ``True``。
    decode_chunk_size:
        VAE 解码分块大小，``None`` 表示由 pipeline 决定（一次解码全部帧，
        显存占用较高）。显存不足时可设为较小值（如 8）以降低峰值占用。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid-xt")
    >>> result = i2v(MosaicData(
    ...     image=Image.open("photo.jpg"),
    ...     num_frames=25,
    ...     fps=7,
    ...     motion_bucket_id=127,
    ... ))
    >>> video = result["video"]  # VideoData

    显存不足时使用更轻量的 14 帧模型：
    >>> i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid")
    """

    name: str = "image-to-video"
    description: str = (
        "Generate video from an input image using Stable Video Diffusion. "
        "Motion intensity is controlled by motion_bucket_id; no text prompt is used."
    )
    version: str = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["video"]

    def __init__(
        self,
        model: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        dtype: str = "float16",
        enable_vae_slicing: bool = True,
        decode_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, dtype=dtype, **kwargs)
        self._enable_vae_slicing: bool = enable_vae_slicing
        self._decode_chunk_size: Optional[int] = decode_chunk_size

    def _load_model(self) -> None:
        """加载 StableVideoDiffusion Pipeline。"""
        import torch  # type: ignore
        from diffusers import StableVideoDiffusionPipeline  # type: ignore

        device = self._resolve_device()
        torch_dtype = self._resolve_dtype()

        self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
        )

        # 应用显存优化
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
            "StableVideoDiffusion pipeline loaded (model=%s, device=%s, dtype=%s).",
            self._model_name,
            device,
            self._dtype_str,
        )

    def _prepare_input_image(self, image: Any) -> Any:
        """将输入图片 resize 到 SVD 要求的 1024x576。

        SVD 模型在 1024x576 分辨率下训练，输入图片需统一 resize 到该
        尺寸以保证生成质量与显存可控。

        Parameters
        ----------
        image:
            输入 ``PIL.Image``（调用前已校验类型）。

        Returns
        -------
        PIL.Image
            resize 后的图片（1024x576）。
        """
        from PIL import Image  # type: ignore

        target_size = _SVD_INPUT_SIZE  # (width, height)
        if image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)
            self._logger.debug("Resized input image to %dx%d.", *target_size)
        return image

    def _prepare_seed(self, seed: Optional[int]) -> tuple:
        """准备随机种子与 generator。

        Parameters
        ----------
        seed:
            用户指定的种子，``None`` 时随机生成。

        Returns
        -------
        Tuple[int, torch.Generator]
            实际使用的种子与对应的 ``torch.Generator``。
        """
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
        """从 SVD Pipeline 输出中提取帧列表。

        SVD 默认以 ``output_type="pil"`` 返回，``output.frames`` 通常为
        嵌套列表 ``List[List[PIL.Image]]``（外层为 batch 维度）。本方法
        取第一个 batch 并展平为 ``List[PIL.Image]``；若返回 tensor，则
        借助 :meth:`BaseVideoNode._tensor_to_frames` 转换。

        Parameters
        ----------
        output:
            Pipeline 输出对象。

        Returns
        -------
        List[PIL.Image]
            帧列表。
        """
        from PIL import Image  # type: ignore

        raw = getattr(output, "frames", None)
        if raw is None:
            raw = getattr(output, "images", None)

        if raw is None:
            return []

        # tensor 路径
        if hasattr(raw, "cpu"):
            return self._tensor_to_frames(raw)

        # 列表路径：List[List[PIL.Image]] 或 List[PIL.Image]
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], list):
                raw = raw[0]
            return [f for f in raw if isinstance(f, Image.Image)]

        return []

    def _truncate_frames(self, frames: list, max_frames: int) -> list:
        """截断帧列表，超出上限时取前 ``max_frames`` 帧。

        SVD-XT 固定输出 25 帧；当请求帧数较少或超出模型上限时需截断。

        Parameters
        ----------
        frames:
            原始帧列表。
        max_frames:
            最大保留帧数。

        Returns
        -------
        List[PIL.Image]
            截断后的帧列表。
        """
        if len(frames) > max_frames:
            self._logger.debug(
                "Truncating output frames from %d to %d.",
                len(frames),
                max_frames,
            )
            frames = frames[:max_frames]
        return frames

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行图生视频。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image)；可选 ``num_frames`` (int,
            默认 25)、``fps`` (int, 默认 7)、``motion_bucket_id``
            (int, 默认 127, 范围 1-255)、``noise_level`` (float,
            默认 0.02)、``num_inference_steps`` (int, 默认 25)、
            ``decode_chunk_size`` (int)、``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``video`` (VideoData)、``seed`` (int)、``num_frames``
            (int)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``image`` 或 ``image`` 非 PIL.Image。
        RuntimeError
            显存不足时抛出，附带建议。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            from PIL import Image  # type: ignore

            image = input_data.get("image")
            if image is None or not isinstance(image, Image.Image):
                raise ValueError(
                    f"ImageToVideo requires 'image' (PIL.Image), "
                    f"got {type(image).__name__ if image is not None else 'None'}."
                )

            # resize 到 SVD 要求的 1024x576
            image = self._prepare_input_image(image)

            num_frames = int(input_data.get("num_frames", _SVD_XT_NUM_FRAMES))
            # SVD-XT 固定输出 25 帧，限制上限
            num_frames = max(1, min(num_frames, _SVD_XT_NUM_FRAMES))

            fps = int(input_data.get("fps", 7))

            motion_bucket_id = int(
                input_data.get("motion_bucket_id", _MOTION_BUCKET_ID_DEFAULT)
            )
            # 限制到合法范围 1-255
            motion_bucket_id = max(1, min(motion_bucket_id, 255))

            noise_level = float(input_data.get("noise_level", _NOISE_LEVEL_DEFAULT))
            num_inference_steps = int(input_data.get("num_inference_steps", 25))
            decode_chunk_size = input_data.get(
                "decode_chunk_size", self._decode_chunk_size
            )
            seed = input_data.get("seed")

            actual_seed, generator = self._prepare_seed(seed)

            self._logger.info(
                "Generating video from image: num_frames=%d, motion_bucket_id=%d, "
                "noise_level=%.4f, steps=%d, fps=%d, seed=%d",
                num_frames,
                motion_bucket_id,
                noise_level,
                num_inference_steps,
                fps,
                actual_seed,
            )

            self._emit_progress(0, num_inference_steps, "Starting generation")

            # 构造 pipeline 参数
            # 注意：SVD 不接受 prompt；noise_level 映射为 noise_aug_level
            pipe_kwargs: dict = {
                "image": image,
                "num_frames": num_frames,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_level": noise_level,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }
            if decode_chunk_size is not None:
                pipe_kwargs["decode_chunk_size"] = int(decode_chunk_size)

            # 执行推理
            import torch  # type: ignore

            try:
                with torch.inference_mode():
                    output = self._pipeline(**pipe_kwargs)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"CUDA out of memory while running {self._model_name}. "
                        f"This model requires ~{_SVD_XT_VRAM_GB}GB VRAM. "
                        f"Try: (1) reduce num_frames, "
                        f"(2) enable vae_slicing, "
                        f"(3) set a smaller decode_chunk_size (e.g. 8), "
                        f"(4) use 'stabilityai/stable-video-diffusion-img2vid' "
                        f"(lighter, 14-frame model)."
                    ) from exc
                raise

            self._emit_progress(
                num_inference_steps, num_inference_steps, "Generation complete"
            )

            # 提取帧
            frames = self._extract_frames_from_output(output)
            if not frames:
                raise RuntimeError(
                    "Pipeline returned no frames for the given image."
                )

            # SVD-XT 输出 25 帧，超出请求帧数时截断
            frames = self._truncate_frames(frames, num_frames)

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 VideoData
        video = self._ensure_video_data(
            frames,
            fps,
            seed=actual_seed,
            motion_bucket_id=motion_bucket_id,
            noise_level=noise_level,
        )
        duration = video.metadata.get("duration", 0.0)

        result = MosaicData(
            video=video,
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
