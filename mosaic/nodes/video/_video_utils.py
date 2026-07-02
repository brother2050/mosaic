# mosaic/nodes/video/_video_utils.py
"""视频节点共用工具函数。

提取各视频生成 / 处理节点中重复的辅助逻辑，集中维护以避免在多个节点
中复制粘贴相同实现。包括：

* 通用参数范围校验：:func:`validate_common_video_params`
* Pipeline 输出帧提取：:func:`extract_frames_from_output`
* 随机种子准备：:func:`prepare_seed`
* CogVideoX 帧数调整：:func:`adjust_num_frames_cogvideox`
* HunyuanVideo 帧数调整：:func:`adjust_num_frames_hunyuan`
* 模型路径校验：:func:`validate_model_path`

安全的类型转换 (:func:`safe_int` / :func:`safe_float`) 统一由
:mod:`mosaic.nodes.coerce` 提供，各节点直接从该模块导入。

这些函数由 :class:`~mosaic.nodes.video._base.BaseVideoNode` 的各子类按需
导入使用。``torch`` / ``numpy`` / ``PIL`` 均采用惰性导入，与视频域其它
模块保持一致。
"""

from __future__ import annotations

import os
import random
import re
from typing import Any

__all__ = [
    "validate_common_video_params",
    "extract_frames_from_output",
    "prepare_seed",
    "adjust_num_frames_cogvideox",
    "adjust_num_frames_hunyuan",
    "validate_model_path",
]


# HuggingFace repo ID 形如 "org/name"：仅含一次斜杠、无前导斜杠，
# 两段均以字母数字开头且仅含 ``[A-Za-z0-9_.-]``。
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


# ----------------------------------------------------------------------
# 参数范围校验（A2）
# ----------------------------------------------------------------------
def validate_common_video_params(
    *,
    num_frames: int | None = None,
    fps: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> None:
    """校验视频生成节点通用参数的范围。

    仅校验非 ``None`` 的参数，任一越界即抛出 :class:`ValueError`。

    Parameters
    ----------
    num_frames:
        帧数，需 ``>= 1``。
    fps:
        帧率，需在 ``[1, 60]``。
    num_inference_steps:
        推理步数，需在 ``[1, 100]``。
    guidance_scale:
        引导尺度，需在 ``[0, 20]``。

    Raises
    ------
    ValueError
        任一参数越界时抛出，信息中包含参数名与实际值。
    """
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if fps is not None and not 1 <= fps <= 60:
        raise ValueError(f"fps must be in [1, 60], got {fps}")
    if num_inference_steps is not None and not 1 <= num_inference_steps <= 100:
        raise ValueError(
            f"num_inference_steps must be in [1, 100], got {num_inference_steps}"
        )
    if guidance_scale is not None and not 0 <= guidance_scale <= 20:
        raise ValueError(
            f"guidance_scale must be in [0, 20], got {guidance_scale}"
        )


# ----------------------------------------------------------------------
# Pipeline 输出帧提取（F1 / E1）
# ----------------------------------------------------------------------
def extract_frames_from_output(output: Any, logger: Any = None) -> list:
    """从 Pipeline 输出中提取帧列表。

    兼容多种输出格式：

    - ``output.frames``：tensor / ndarray，形状 ``(batch, num_frames, H, W, C)``
      或 ``(num_frames, H, W, C)``。取第一个 batch，归一化到 ``[0, 255]`` 后
      转为 ``PIL.Image`` 列表。
    - ``output.images``：直接作为帧列表返回。

    对空数组做了保护：当 ``arr.size == 0`` 时跳过归一化并返回空列表，
    避免 ``arr.max()`` 对零大小数组抛出 ``ValueError``。

    Parameters
    ----------
    output:
        Pipeline 输出对象。
    logger:
        可选的 logger，用于在异常情况（如空数组）下记录告警。

    Returns
    -------
    list[PIL.Image]
        帧列表；无可用帧时返回空列表。
    """
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore

    frames: list = []

    if hasattr(output, "frames"):
        raw = output.frames
        # PIL 嵌套列表：List[List[PIL.Image]]（diffusers pipeline 默认 output_type="pil"）
        if isinstance(raw, list):
            from PIL import Image
            # 展平嵌套列表，取第一个 batch
            frames = []
            for item in raw:
                if isinstance(item, list):
                    for img in item:
                        if isinstance(img, Image.Image):
                            frames.append(img)
                elif isinstance(item, Image.Image):
                    frames.append(item)
            if frames:
                return frames
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

        # 空数组保护：arr.size == 0 时 arr.max() 会抛 ValueError
        if arr.size == 0:
            if logger is not None:
                logger.warning(
                    "Pipeline returned an empty frames array; "
                    "no frames extracted."
                )
        else:
            # 检测 NaN：若上游 VAE 产生 NaN，此处报错而非静默输出黑帧
            if np.isnan(arr).any():
                raise RuntimeError(
                    "NaN detected in video output tensor — likely VAE decode failure. "
                    "Consider upcasting VAE to float32 or reducing resolution."
                )
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


# ----------------------------------------------------------------------
# 随机种子准备（F1）
# ----------------------------------------------------------------------
def prepare_seed(seed: int | None, device: str) -> tuple:
    """准备随机种子与 ``torch.Generator``。

    ``seed`` 为 ``None`` 时随机生成一个 ``[0, 2**32)`` 的种子。随后基于
    ``device`` 创建 ``torch.Generator``；若该设备不支持则回退到 CPU。

    Parameters
    ----------
    seed:
        用户指定的种子，``None`` 时随机生成。
    device:
        推理设备字符串（如 ``"cuda"`` / ``"cpu"``）。

    Returns
    -------
    tuple[int, torch.Generator]
        实际使用的种子与对应的 ``torch.Generator``。
    """
    import torch  # type: ignore

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed) % (2**32)

    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    except (RuntimeError, ValueError, TypeError):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    return seed, generator


# ----------------------------------------------------------------------
# 帧数调整（F1 / E2）
# ----------------------------------------------------------------------
def adjust_num_frames_cogvideox(
    num_frames: int,
    valid_frames: list[int],
    logger: Any = None,
) -> int:
    """调整 ``num_frames`` 为 CogVideoX 支持的有效值。

    CogVideoX 仅支持 ``valid_frames`` 中的值（通常为 49 / 85）。非有效值
    时取最近的合法值并告警。

    Parameters
    ----------
    num_frames:
        用户请求的帧数。
    valid_frames:
        模型支持的有效帧数列表。
    logger:
        可选的 logger，用于记录调整告警。

    Returns
    -------
    int
        调整后的有效帧数。
    """
    if num_frames in valid_frames:
        return num_frames

    closest = min(valid_frames, key=lambda v: abs(v - num_frames))
    if logger is not None:
        logger.warning(
            "num_frames=%d is not a valid value for CogVideoX. "
            "Adjusted to %d (valid values: %s).",
            num_frames,
            closest,
            valid_frames,
        )
    return closest


def adjust_num_frames_hunyuan(
    num_frames: int,
    default_frames: int = 129,
    logger: Any = None,
) -> int:
    """调整 ``num_frames`` 为 HunyuanVideo 支持的有效值。

    HunyuanVideo 要求 ``(num_frames - 1) % 4 == 0``（即 ``4k+1``），
    常见值如 5 / 9 / ... / 129。非有效值时取最近的 ``4k+1`` 并告警；
    小于 1 时回退到 ``default_frames``。

    Parameters
    ----------
    num_frames:
        用户请求的帧数。
    default_frames:
        ``num_frames < 1`` 时使用的回退帧数，默认 129。
    logger:
        可选的 logger，用于记录调整告警。

    Returns
    -------
    int
        调整后的有效帧数。
    """
    if num_frames < 1:
        if logger is not None:
            logger.warning(
                "num_frames=%d is less than 1; using default %d.",
                num_frames,
                default_frames,
            )
        num_frames = default_frames

    if (num_frames - 1) % 4 == 0:
        return num_frames

    adjusted = round((num_frames - 1) / 4) * 4 + 1
    adjusted = max(1, adjusted)
    if logger is not None:
        logger.warning(
            "num_frames=%d is not a valid value for HunyuanVideo "
            "(must be 4k+1, e.g. 5/9/.../129). Adjusted to %d.",
            num_frames,
            adjusted,
        )
    return adjusted


# ----------------------------------------------------------------------
# 模型路径校验（B2）
# ----------------------------------------------------------------------
def _looks_like_local_path(name: str) -> bool:
    """判断 ``name`` 是否像本地路径（而非 HuggingFace repo ID 或 URL）。

    启发式规则：

    - 空字符串 -> ``False``
    - ``http://`` / ``https://`` 开头 -> ``False``（URL）
    - 形如 ``org/name`` 且仅含 ``[A-Za-z0-9_.-]`` 的单斜杠串 -> ``False``
      （HuggingFace repo ID）
    - 其余 -> ``True``（视为本地路径）
    """
    if not name:
        return False
    if name.startswith(("http://", "https://")):
        return False
    if _HF_REPO_ID_RE.match(name):
        return False
    return True


def validate_model_path(model_name: str, logger: Any = None) -> None:
    """校验模型路径：若像本地路径但不存在，抛出友好的 FileNotFoundError。

    仅对“看起来像本地路径”的名称做存在性检查；HuggingFace repo ID（如
    ``THUDM/CogVideoX-5b``）与 URL 不受影响，仍交给 ``safe_load_pipeline``
    处理。

    额外检查本地目录是否包含 ``model_index.json``（diffusers 格式的标志），
    缺失时提示用户下载 diffusers 格式的模型。

    Parameters
    ----------
    model_name:
        模型标识（HF repo ID / 本地路径 / URL）。
    logger:
        可选的 logger（当前未使用，保留以备扩展）。

    Raises
    ------
    FileNotFoundError
        ``model_name`` 像本地路径但文件/目录不存在时抛出。
    ValueError
        本地目录存在但缺少 ``model_index.json``（非 diffusers 格式）时抛出。
    """
    if not model_name:
        return
    if _looks_like_local_path(model_name) and not os.path.exists(model_name):
        raise FileNotFoundError(
            f"Model path not found: {model_name!r}. "
            f"Please check the path or use a valid HuggingFace model ID "
            f"(e.g. 'THUDM/CogVideoX-5b')."
        )

    # 检查本地目录是否为 diffusers 格式（包含 model_index.json）
    if _looks_like_local_path(model_name) and os.path.isdir(model_name):
        index_file = os.path.join(model_name, "model_index.json")
        if not os.path.exists(index_file):
            try:
                contents = os.listdir(model_name)
            except OSError:
                contents = []
            has_safetensors = any(
                f.endswith((".safetensors", ".pt", ".bin"))
                for f in contents
            )
            hint = (
                "This directory contains raw checkpoint files but not the "
                "diffusers pipeline structure (model_index.json).\n"
                "Please download the diffusers-format version instead:\n"
                "  hunyuanvideo-community/HunyuanVideo (diffusers format)\n"
                "  hunyuanvideo-community/HunyuanVideo (raw format, NOT compatible)"
            ) if has_safetensors else (
                "This directory does not contain model_index.json.\n"
                "Please ensure you downloaded the diffusers-format version."
            )
            raise ValueError(
                f"Directory {model_name!r} is not a valid diffusers model "
                f"directory (missing model_index.json).\n{hint}"
            )
