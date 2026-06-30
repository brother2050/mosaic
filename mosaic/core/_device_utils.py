# mosaic/core/_device_utils.py
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
    "apply_optimizations",
    "run_diffusers_pipeline",
]

logger = logging.getLogger("mosaic.core._device_utils")


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

            return "cuda" if torch.cuda.is_available() else "cpu"
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
