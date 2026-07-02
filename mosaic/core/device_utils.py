# mosaic/core/device_utils.py
"""设备与 dtype 解析工具（跨域共享）。

本模块集中实现了原本在各个域基类（``image``、``video``、``consistency``、
``digital_human``、``audio``、``text``、``rag`` 等）中重复出现的设备/dtype 解析
以及 diffusers pipeline 推理辅助逻辑，避免相同实现散落在多处。

设计要点
--------
* ``torch`` 采用惰性导入（在函数内部 ``import torch``），使本模块在未安装
  ``torch`` 的环境中仍可被正常导入；仅在真正解析/推理时才要求依赖存在。
* :func:`resolve_dtype` 内部以 ``try/except ImportError`` 处理 ``torch`` 不可用的情况。
* 所有优化/推理辅助方法在失败时仅记录 ``debug`` 日志或回退，不抛出异常，
  以保证推理主流程不被可选的显存优化中断。
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "resolve_dtype",
    "resolve_device",
    "infer_device",
    "auto_resolve_device_dtype",
    "apply_optimizations",
    "run_diffusers_pipeline",
    "upcast_pipeline_components",
    "empty_device_cache",
]

logger = logging.getLogger("mosaic.core.device_utils")


def resolve_dtype(dtype: str) -> Any:
    """将 dtype 字符串解析为 torch dtype。

    支持的字符串：``float16``/``fp16``、``float32``/``fp32``、
    ``bfloat16``/``bf16``；未能识别的字符串回退为 ``torch.float16``。

    Parameters
    ----------
    dtype:
        dtype 字符串。

    Returns
    -------
    torch.dtype
        对应的 torch dtype。

    Raises
    ------
    ImportError
        ``torch`` 未安装时抛出。
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖环境
        raise ImportError(
            "torch is required to resolve dtype, but it is not installed."
        ) from exc

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype, torch.float16)


def resolve_device(device: str | None, scheduler: Any | None = None) -> str:
    """解析设备字符串。

    规则：

    * ``device`` 为 ``None`` 时，优先使用 ``scheduler.device``；若调度器也不可用，
      则通过 ``torch.cuda.is_available()`` 推断（有 CUDA 返回 ``"cuda"``，
      否则返回 ``"cpu"``）。
    * ``device`` 指定为 ``"cuda"`` 时：若调度器提供 ``is_gpu`` 属性则以其判断 GPU
      可用性，不可用时降级到 ``scheduler.device``；否则用 ``torch.cuda.is_available()``
      判断，不可用时降级到 ``"cpu"``。
    * ``torch`` 不可用时保持原 ``device`` 不变（无法判断 GPU 可用性）。
    * 非 CUDA 设备字符串原样返回。
    """
    if device is None:
        if scheduler is not None:
            return scheduler.device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    if not str(device).startswith("cuda"):
        return device

    # 优先使用调度器的 is_gpu 判断（与 consistency/digital_human 等域一致）
    if scheduler is not None and hasattr(scheduler, "is_gpu"):
        if scheduler.is_gpu:
            return device
        return scheduler.device

    # 退而使用 torch.cuda.is_available() 判断
    try:
        import torch  # type: ignore
    except ImportError:
        return device
    if torch.cuda.is_available():
        return device
    return "cpu"


def infer_device(model: Any, scheduler: Any | None = None) -> str:
    """推断模型所在设备。

    ``model`` 为 ``None`` 时返回 ``scheduler.device``（无调度器时返回 ``"cpu"``）；
    否则尝试 ``next(model.parameters()).device``，失败时回退到 ``scheduler.device``。

    Parameters
    ----------
    model:
        模型或 diffusers Pipeline 实例（支持 ``parameters()`` 方法）。
    scheduler:
        调度器实例，用于提供回退设备。

    Returns
    -------
    str | torch.device
        推断出的设备。
    """
    fallback = scheduler.device if scheduler is not None else "cpu"
    if model is None:
        return fallback
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, RuntimeError, TypeError):
        return fallback


def auto_resolve_device_dtype(
    device: str,
    dtype: str,
    scheduler: Any | None = None,
    logger: Any | None = None,
    model_name: str = "",
) -> tuple[str, str]:
    """自动解析设备与 dtype：CPU 或 SD 1.5 环境下将 float16 降级为 float32。

    1. **CPU + float16 → float32**：float16 在 CPU 上无法正确推理（PyTorch 限制）。
    2. **SD 1.5 + float16 → float32**：SD 1.5 的 text_encoder/UNet 对 float16 敏感，
       特定 prompt 触发 attention 溢出产生 NaN，导致黑图。整体 float32 加载避免
       组件间 dtype 不匹配。

    Parameters
    ----------
    device:
        用户指定的设备字符串（如 ``"cuda"``、``"cpu"``）。
    dtype:
        用户指定的精度字符串（如 ``"float16"``、``"float32"``）。
    scheduler:
        调度器实例，用于判断 GPU 可用性。
    logger:
        可选的日志器，用于输出降级告警。
    model_name:
        模型名称，用于判断是否为 SD 1.5 系列（float16 敏感）。

    Returns
    -------
    tuple[str, str]
        ``(resolved_device, resolved_dtype)``
    """
    resolved_device = resolve_device(device, scheduler)
    resolved_dtype = dtype

    if resolved_device in ("cpu", "mps") and dtype in ("float16", "fp16", "bfloat16", "bf16"):
        if resolved_device == "cpu" or _is_sd15_model(model_name):
            # CPU 下 float16 不稳定；MPS + SD 1.5 与 CPU 一致，均降级 float32
            resolved_dtype = "float32"
            if resolved_device == "cpu":
                msg = (
                    "Device downgraded to CPU but dtype is %s — auto-switching to "
                    "float32 to avoid black/garbage images (fp16 on CPU is not "
                    "supported by PyTorch)."
                ) % dtype
            else:
                msg = (
                    "Device is MPS and model %s is SD 1.5 series with dtype=%s — "
                    "auto-switching to float32 to avoid NaN/black images "
                    "(SD 1.5 is fp16-sensitive, same as CPU)."
                ) % (model_name, dtype)
            if logger is not None:
                logger.warning(msg)
            else:
                logging.getLogger("mosaic.core.device_utils").warning(msg)
        else:
            # MPS + SDXL/其它模型：保持 float16，但提示部分算子兼容性问题
            msg = (
                "MPS device with dtype=%s — keeping float16, but some operators "
                "may have compatibility issues on MPS (float16)."
            ) % dtype
            if logger is not None:
                logger.info(msg)
            else:
                logging.getLogger("mosaic.core.device_utils").info(msg)

    # SD 1.5 系列：text_encoder/UNet 对 float16 敏感，整体降为 float32
    if resolved_dtype in ("float16", "fp16") and _is_sd15_model(model_name):
        resolved_dtype = "float32"
        msg = (
            "Model %s is SD 1.5 series with dtype=float16 — auto-switching to "
            "float32 to avoid NaN/black images (SD 1.5 text_encoder is fp16-sensitive)."
        ) % model_name
        if logger is not None:
            logger.warning(msg)
        else:
            logging.getLogger("mosaic.core.device_utils").warning(msg)

    return resolved_device, resolved_dtype


def apply_optimizations(
    pipeline: Any,
    enable_cpu_offload: bool = False,
    enable_attention_slicing: bool = False,
    enable_vae_slicing: bool = False,
) -> None:
    """对已加载的 diffusers Pipeline 应用显存优化配置。

    各 ``enable_*`` 参数为 ``True`` 时尝试调用对应的优化方法；调用失败时仅记录
    ``debug`` 日志而不抛出异常，``pipeline`` 为 ``None`` 时直接返回。

    兼容 diffusers 0.40+ 的 API 变更：``pipe.enable_vae_slicing()`` 已废弃，
    优先使用 ``pipe.vae.enable_slicing()``，回退到旧 API。
    """
    if pipeline is None:
        return

    if enable_attention_slicing:
        try:
            pipeline.enable_attention_slicing()
        except Exception as exc:  # noqa: BLE001 - 优化失败不应中断推理
            logger.debug("enable_attention_slicing skipped: %s", exc)

    if enable_vae_slicing:
        try:
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            else:
                pipeline.enable_vae_slicing()
        except Exception as exc:  # noqa: BLE001
            logger.debug("enable_vae_slicing skipped: %s", exc)

    if enable_cpu_offload:
        try:
            pipeline.enable_model_cpu_offload()
        except Exception as exc:  # noqa: BLE001
            logger.debug("enable_model_cpu_offload skipped: %s", exc)


def run_diffusers_pipeline(pipeline: Any, **kwargs: Any) -> Any:
    """执行 diffusers pipeline 推理并返回结果。

    在 ``torch.inference_mode()`` 上下文中调用 ``pipeline(**kwargs)``。

    Parameters
    ----------
    pipeline:
        diffusers Pipeline 实例。
    **kwargs:
        透传给 pipeline 的参数。

    Returns
    -------
    Any
        Pipeline 输出。
    """
    import torch  # type: ignore

    with torch.inference_mode():
        return pipeline(**kwargs)


# SD 1.5 系列：text_encoder/UNet 对 float16 敏感，应整体加载为 float32
# SDXL 的 text_encoder 已兼容 float16
_SD15_MODEL_PATTERNS = (
    "stable-diffusion-v1", "stable-diffusion-2-1", "stable-diffusion-x4",
    "sd-v1", "sd15", "SD1.5",
)


def _is_sd15_model(model_name: str | None) -> bool:
    """判断模型是否为 SD 1.5 系列（text_encoder 对 float16 敏感）。"""
    if not model_name:
        return False
    return any(p in model_name for p in _SD15_MODEL_PATTERNS)


def upcast_pipeline_components(
    pipeline: Any,
    model_name: str = "",
    logger: Any | None = None,
) -> None:
    """上转 pipeline 的 VAE 为 float32，防止 float16 下的黑图/NaN 问题。

    **仅上转 VAE**。VAE 只在最后 decode 阶段调用一次，上转为 float32 可
    避免 float16 下的数值精度问题导致的黑图/NaN。

    **例外：CogVideoX 和 SVD 跳过 VAE 上转**。这两类 pipeline 的 decode_latents
    不做 ``latents.to(vae.dtype)`` 转换，VAE 上转会导致 dtype 不匹配报错。
    SVD 还依赖 diffusers 自身的 force_upcast→cast-back 机制，永久 float32 会
    破坏该机制。

    **SDXL 也跳过 VAE 上转**：SDXL 的 ``AutoencoderKL.decode()`` 在部分
    diffusers 版本中不做 ``latents.to(vae.dtype)`` 转换，VAE 上转后会导致
    ``RuntimeError: Input type (c10::Half) and bias type (float) should be
    the same``。改用 ``vae.enable_tiling()`` 在 fp16 下避免黑图，同时保持
    dtype 一致。

    Parameters
    ----------
    pipeline:
        diffusers Pipeline 实例。
    model_name:
        模型名称，用于判断是否为需要跳过 VAE 上转的模型。
    logger:
        可选的日志器。
    """
    if pipeline is None:
        return

    import torch  # type: ignore

    pipeline_cls_name = type(pipeline).__name__
    model_lower = model_name.lower()

    # CogVideoX 和 SVD 跳过 VAE 上转：diffusers 不做 latents→vae.dtype 转换
    skip_vae_upcast = (
        "CogVideoX" in pipeline_cls_name
        or "StableVideoDiffusion" in pipeline_cls_name
        or "cogvideo" in model_lower
        or "svd" in model_lower
        or "stable-video-diffusion" in model_lower
    )
    # SDXL 也跳过 VAE 上转：decode 不做 latents.to(vae.dtype) 导致 dtype 不匹配
    if not skip_vae_upcast:
        skip_vae_upcast = (
            "StableDiffusionXL" in pipeline_cls_name
            or "sdxl" in model_lower
            or "stable-diffusion-xl" in model_lower
        )

    if skip_vae_upcast:
        # SDXL：不手动上转 VAE，交给 pipeline 自己处理。
        #
        # diffusers SDXL pipeline 在 __call__ 中的逻辑（L1254-1258）：
        #   needs_upcasting = vae.dtype==float16 and vae.config.force_upcast
        #   if needs_upcasting:
        #       self.upcast_vae()                        # VAE.to(float32)
        #       latents = latents.to(vae.post_quant_conv.parameters().dtype)
        #
        # 如果我们手动 vae.to(float32)，needs_upcasting 变 False，pipeline
        # 不会转 latents → float16 latents 传入 float32 VAE → 崩溃。
        #
        # 如果手动 vae.to(float32) + force_upcast=False，needs_upcasting 仍
        # False，pipeline 仍不转 latents → 同样崩溃。
        #
        # 所以只能保持 force_upcast=True，让 pipeline 自己调 upcast_vae()
        # （会同步转 VAE 和 latents）。upcast_vae() 已弃用（FutureWarning），
        # 但 pipeline 内部的调用无法从外部消除该警告。
        #
        # 启用 VAE tiling 减少 fp16 下的显存占用和精度问题。
        if (
            "StableDiffusionXL" in pipeline_cls_name
            or "sdxl" in model_lower
            or "stable-diffusion-xl" in model_lower
        ):
            vae = getattr(pipeline, "vae", None)
            if vae is not None:
                try:
                    vae.enable_tiling()
                    if logger is not None:
                        logger.debug(
                            "VAE tiling enabled for %s.", pipeline_cls_name
                        )
                except Exception:
                    pass
        elif logger is not None:
            logger.debug(
                "Skipping VAE upcast for %s (diffusers doesn't convert "
                "latents dtype in decode_latents).",
                pipeline_cls_name,
            )
        return

    # 非 SDXL/CogVideoX/SVD：VAE → float32（decode 时做 latents.to(vae.dtype)）
    vae = getattr(pipeline, "vae", None)
    if vae is not None:
        try:
            vae.to(torch.float32)
            if logger is not None:
                logger.debug("VAE upcasted to float32 (black-image prevention).")
        except Exception as exc:
            if logger is not None:
                logger.debug("VAE upcast skipped: %s", exc)


def empty_device_cache(device: str = "") -> None:
    """统一的设备显存清理，支持 CUDA 和 MPS。

    依次尝试清理 CUDA 与 MPS 显存缓存；任一后端不可用或清理失败时静默忽略，
    不抛出异常，以保证推理主流程不被可选的显存优化中断。

    Parameters
    ----------
    device:
        预留参数，当前未使用。清理时会同时尝试 CUDA 与 MPS（仅清理可用的后端）。
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - 显存清理失败不应中断推理
        logging.getLogger(__name__).debug(
            "empty_device_cache failed", exc_info=True,
        )
