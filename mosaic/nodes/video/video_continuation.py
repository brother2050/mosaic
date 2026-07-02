# mosaic/nodes/video/video_continuation.py
"""VideoContinuation 节点 —— 视频续写。

在已有视频的末尾追加新生成的帧，实现视频“向后延长”。基于
THUDM CogVideoX 模型（与 :mod:`~mosaic.nodes.video.text_to_video` 相同的
``diffusers.CogVideoXPipeline``）。

设计要点
--------
* 复用 ``CogVideoXPipeline``（文生视频）加载方式：``_load_model`` 与
  :class:`~mosaic.nodes.video.text_to_video.TextToVideo` 一致，便于共享
  显存与缓存。
* 续写流程：取输入视频尾部 ``overlap_frames`` 帧作为“过渡锚点”，用
  ``prompt``（可选，缺省时使用平滑续写的默认提示）驱动 CogVideoX 生成
  ``num_frames`` 帧新内容；随后对重叠区域做交叉淡化（crossfade）实现
  平滑过渡，再与原始帧拼接。
* 生成分辨率对齐到原始视频尺寸（偶数化），生成后再次 resize 校正，确保
  续写帧与原始帧像素级一致，便于直接拼接。
* 帧率统一：合并产物一律采用原始视频的 ``fps``，避免原始段与续写段播放
  速度不一致。若原始 ``fps`` 缺失则回退到 8（CogVideoX 常用帧率）。
* CogVideoX 要求 ``num_frames`` 为 49 或 85，非有效值时自动调整为最近的
  合法值。
* 长视频生成非常耗时，通过 EventBus 实时报告进度。
* 输出统一为 :class:`~mosaic.core.types.VideoData` 格式，同时返回完整视频
  与“仅续写部分”两份产物。

显存需求
--------
* ``THUDM/CogVideoX-5b``：约 16-20GB（fp16）
* ``THUDM/CogVideoX-2b``：约 8-10GB（fp16，推荐显存受限时使用）

风格一致性提示
--------------
CogVideoX 为纯文生视频模型，续写片段主要受 ``prompt`` 与交叉淡化约束，
并不直接以原始帧作为像素级条件。若续写部分与原始视频在风格、色调、
运动节奏上差异较大，建议在管道中配合一致性 / 风格迁移节点（如
``video-to-video`` 或专门的时序一致性节点）做后处理，以获得更连贯的
视觉效果。

许可证
------
* CogVideoX 系列：CogVideoX License (Apache 2.0)
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode
from mosaic.nodes.video._video_utils import (
    adjust_num_frames_cogvideox,
    extract_frames_from_output,
    prepare_seed,
    safe_float,
    safe_int,
    validate_common_video_params,
)

__all__ = ["VideoContinuation"]


# CogVideoX 支持的有效帧数
_VALID_NUM_FRAMES = [49, 85]

# 默认重叠帧数（用于交叉淡化过渡）
_DEFAULT_OVERLAP_FRAMES = 4

# 缺省推理步数与引导尺度（与 TextToVideo 保持一致）
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 6.0

# 缺省 prompt（用户未提供时使用，强调平滑续写）
_DEFAULT_PROMPT = "Continue the video with smooth and coherent motion."

# CogVideoX-5b 粗略显存需求（fp16, GB），用于错误提示
_COGVIDEOX_5B_VRAM_GB = 18.0


@registry.register
class VideoContinuation(BaseVideoNode):
    """视频续写节点。

    在输入视频末尾追加 CogVideoX 生成的新帧，通过交叉淡化实现平滑过渡，
    输出“原始 + 续写”的完整视频以及“仅续写部分”。

    由于 CogVideoX 为文生视频模型，续写片段以 ``prompt`` 驱动、并以原始
    视频尾部帧的分辨率与重叠区交叉淡化作为视觉锚点。若续写风格与原始
    差异较大，建议配合一致性 / 风格迁移节点使用。

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
    >>> vc = VideoContinuation(model="THUDM/CogVideoX-5b")
    >>> result = vc(MosaicData(
    ...     video=input_video_data,          # VideoData
    ...     prompt="镜头继续向前推进，人物走向远方",
    ...     num_frames=49,
    ...     overlap_frames=4,
    ... ))
    >>> full = result["video"]              # VideoData: 原始 + 续写
    >>> cont = result["continuation_video"]  # VideoData: 仅续写部分
    >>> result["total_frames"], result["total_duration"]

    显存不足时使用 2b 版本：
    >>> vc = VideoContinuation(model="THUDM/CogVideoX-2b")
    """

    name: str = "video-continuation"
    description: str = (
        "Extend an existing video by generating new frames from its tail "
        "using CogVideoX. Overlap frames are crossfaded for a smooth "
        "transition; the original video's fps is preserved."
    )
    version: str = "0.1.0"
    input_types = ("video", "mosaic")
    output_types = ("video",)

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
        """加载 CogVideoXPipeline（与 TextToVideo 相同）。"""
        import torch  # type: ignore
        from diffusers import CogVideoXPipeline  # type: ignore
        from mosaic.nodes._model_loader import safe_load_pipeline

        device = self._resolve_device()
        torch_dtype = self._resolve_dtype()

        # CogVideoX 内部使用 T5 文本编码器，needs_t5=True 预导入 T5 组件
        self._pipeline = safe_load_pipeline(
            CogVideoXPipeline,
            self._model_name,
            needs_t5=True,
            torch_dtype=torch_dtype,
            variant_fp16=self._dtype_str in ("float16", "fp16"),
            dtype_str=self._dtype_str,
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
        return adjust_num_frames_cogvideox(
            num_frames, _VALID_NUM_FRAMES, self._logger
        )

    def _prepare_seed(self, seed: int | None) -> tuple[int, Any]:
        """准备随机种子与 generator。

        委托给 :func:`mosaic.nodes.video._video_utils.prepare_seed`：
        未指定种子时随机生成，并基于 ``torch.Generator`` 创建对应的
        随机数生成器。

        Parameters
        ----------
        seed:
            用户指定的种子，``None`` 时随机生成。

        Returns
        -------
        tuple[int, torch.Generator]
            实际使用的种子与对应的 ``torch.Generator``。
        """
        return prepare_seed(seed, self._infer_device())

    def _extract_frames_from_output(self, output: Any) -> list[Any]:
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
        return extract_frames_from_output(output, self._logger)

    def _crossfade_frames(
        self,
        tail_frames: list[Any],
        head_frames: list[Any],
        overlap: int,
    ) -> list[Any]:
        """对重叠区域的帧做交叉淡化（加权平均）。

        将 ``tail_frames``（原始视频尾部帧）与 ``head_frames``（生成片段
        头部帧）按位置一一加权混合：第 ``i`` 帧的混合权重
        ``alpha = i / (overlap - 1)``（``overlap == 1`` 时取 ``0.5``），
        使结果从“纯原始”平滑过渡到“纯生成”。

        Parameters
        ----------
        tail_frames:
            原始视频尾部 ``overlap`` 帧。
        head_frames:
            生成片段头部 ``overlap`` 帧。
        overlap:
            重叠帧数。

        Returns
        -------
        list[PIL.Image]
            交叉淡化后的 ``overlap`` 帧。

        Raises
        ------
        ValueError
            ``tail_frames`` 与 ``head_frames`` 长度均小于 ``overlap``。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if overlap <= 0:
            return []

        overlap = min(overlap, len(tail_frames), len(head_frames))
        if overlap <= 0:
            raise ValueError(
                "Cannot crossfade: tail/head frames shorter than overlap."
            )

        # 线性权重：tail 从 1 -> 0，head 从 0 -> 1
        if overlap == 1:
            alphas = [0.5]
        else:
            alphas = [i / (overlap - 1) for i in range(overlap)]

        # 统一尺寸到 tail 的首帧（防御性，正常情况下已对齐）
        target_size = tail_frames[0].size

        blended: list[Any] = []
        for i in range(overlap):
            alpha = alphas[i]
            tail = tail_frames[i]
            head = head_frames[i]
            if tail.size != target_size:
                tail = tail.resize(target_size, Image.Resampling.LANCZOS)
            if head.size != target_size:
                head = head.resize(target_size, Image.Resampling.LANCZOS)

            tail_arr = np.array(tail.convert("RGB"), dtype=np.float32)
            head_arr = np.array(head.convert("RGB"), dtype=np.float32)
            mixed = (1.0 - alpha) * tail_arr + alpha * head_arr
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)
            blended.append(Image.fromarray(mixed))

        return blended

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行视频续写。

        取输入视频尾部 ``overlap_frames`` 帧作为过渡锚点，结合 ``prompt``
        驱动 CogVideoX 生成 ``num_frames`` 帧新内容，对重叠区做交叉淡化后
        与原始帧拼接，统一到原始视频的 ``fps``。

        Parameters
        ----------
        input_data:
            必须包含 ``video`` (:class:`VideoData`)；可选 ``prompt``
            (str, 缺省时使用平滑续写默认提示)、``num_frames`` (int,
            默认 49)、``overlap_frames`` (int, 默认 4)、``seed`` (int)、
            ``num_inference_steps`` (int, 默认 50)、``guidance_scale``
            (float, 默认 6.0)、``negative_prompt`` (str)。

        Returns
        -------
        MosaicData
            包含 ``video`` (VideoData, 完整视频=原始+续写)、
            ``continuation_video`` (VideoData, 仅续写部分)、
            ``total_frames`` (int)、``total_duration`` (float)，
            以及 ``seed`` (int)、``overlap_frames`` (int)。

        Raises
        ------
        ValueError
            缺少 ``video`` 或 ``video`` 非 :class:`VideoData`，或视频无帧。
        RuntimeError
            显存不足时抛出，附带建议。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入视频
            video = input_data.get("video")
            if not isinstance(video, VideoData):
                raise ValueError(
                    f"VideoContinuation requires 'video' (VideoData), "
                    f"got {type(video).__name__ if video is not None else 'None'}."
                )

            original_frames: list[Any] = list(video.frames)
            if not original_frames:
                raise ValueError(
                    "VideoContinuation requires a non-empty 'video' "
                    "(video.frames is empty)."
                )

            # 帧率统一到原始视频的 fps（缺失时回退到 8）
            original_fps = video.fps
            if not isinstance(original_fps, (int, float)) or original_fps <= 0:
                original_fps = 8
            fps = int(original_fps)
            # fps 为派生值，越界时 clamp 到 [1, 60] 而非报错（A2）
            if fps > 60:
                self._logger.warning(
                    "Original video fps=%d exceeds 60; clamping to 60.", fps
                )
                fps = 60

            # prompt（可选）
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = _DEFAULT_PROMPT
                self._logger.warning(
                    "No 'prompt' provided for continuation; "
                    "using default: %r",
                    prompt,
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            # 生成帧数（对齐 CogVideoX 合法值）
            num_frames = safe_int(input_data.get("num_frames", 49), "num_frames")
            num_frames = self._adjust_num_frames(num_frames)

            # 重叠帧数（约束在原始帧数与生成帧数范围内）
            overlap_frames = safe_int(
                input_data.get("overlap_frames", _DEFAULT_OVERLAP_FRAMES),
                "overlap_frames",
            )
            overlap_frames = max(
                0, min(overlap_frames, len(original_frames), num_frames)
            )

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps", _DEFAULT_NUM_INFERENCE_STEPS),
                "num_inference_steps",
            )
            guidance_scale = safe_float(
                input_data.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE),
                "guidance_scale",
            )
            seed = input_data.get("seed")

            # 参数范围校验（A2）
            validate_common_video_params(
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            actual_seed, generator = self._prepare_seed(seed)

            # 生成分辨率对齐到原始视频尺寸（偶数化）
            ref_width, ref_height = original_frames[-1].size
            width, height = self._ensure_even_dimensions(ref_width, ref_height)

            self._logger.info(
                "Continuing video: original_frames=%d, num_frames=%d, "
                "overlap=%d, size=%dx%d, steps=%d, guidance=%.1f, fps=%d, "
                "seed=%d, prompt=%r",
                len(original_frames),
                num_frames,
                overlap_frames,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                fps,
                actual_seed,
                prompt[:50],
            )

            self._emit_progress(0, num_inference_steps, "Starting continuation")

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
                        f"This model requires ~{_COGVIDEOX_5B_VRAM_GB}GB VRAM. "
                        f"Try: (1) reduce num_frames to 49, "
                        f"(2) use 'THUDM/CogVideoX-2b' (lighter model), "
                        f"(3) reduce width/height, "
                        f"(4) reduce overlap_frames."
                    ) from exc
                raise

            self._emit_progress(
                num_inference_steps, num_inference_steps, "Generation complete"
            )

            # 提取生成帧
            gen_frames = self._extract_frames_from_output(output)
            if not gen_frames:
                raise RuntimeError(
                    "Pipeline returned no frames for the continuation request."
                )

            # 校正生成帧尺寸到原始分辨率（防御性，确保可拼接）
            if gen_frames[0].size != (width, height):
                gen_frames = self._resize_frames(gen_frames, (width, height))

            # ---- 拼接：原始(去掉尾部 overlap) + 交叉淡化区 + 生成(去掉头部 overlap)
            if overlap_frames > 0:
                tail_frames = original_frames[-overlap_frames:]
                head_frames = gen_frames[:overlap_frames]
                blended = self._crossfade_frames(
                    tail_frames, head_frames, overlap_frames
                )
                merged_frames = (
                    original_frames[: len(original_frames) - overlap_frames]
                    + blended
                    + gen_frames[overlap_frames:]
                )
                # 仅续写部分 = 交叉淡化区 + 生成(去掉头部 overlap)
                continuation_frames = blended + gen_frames[overlap_frames:]
            else:
                merged_frames = list(original_frames) + list(gen_frames)
                continuation_frames = list(gen_frames)

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 VideoData（统一到原始 fps，处理帧率不一致）
        full_video = self._ensure_video_data(
            merged_frames,
            fps,
            prompt=prompt,
            seed=actual_seed,
            overlap_frames=overlap_frames,
            source="video-continuation",
        )
        continuation_video = self._ensure_video_data(
            continuation_frames,
            fps,
            prompt=prompt,
            seed=actual_seed,
            overlap_frames=overlap_frames,
            source="video-continuation",
        )

        total_frames = len(merged_frames)
        total_duration = full_video.metadata.get("duration", 0.0)

        result = MosaicData(
            video=full_video,
            continuation_video=continuation_video,
            total_frames=total_frames,
            total_duration=total_duration,
            seed=actual_seed,
            overlap_frames=overlap_frames,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "total_frames": total_frames,
                "total_duration": total_duration,
                "continuation_frames": len(continuation_frames),
                "overlap_frames": overlap_frames,
                "fps": fps,
                "seed": actual_seed,
            },
        )
        return result
