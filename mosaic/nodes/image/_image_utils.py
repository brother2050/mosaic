"""图像节点公共工具函数。

提供安全的类型转换与参数范围校验，统一图像生成节点的输入处理逻辑。

核心功能：
- :func:`safe_int` / :func:`safe_float`：将用户输入安全转换为 int/float，
  转换失败时抛出包含参数名与实际值的清晰 ``ValueError``。
- :func:`validate_image_dimensions`：校验图像尺寸上下限，防止过小导致
  崩溃或过大导致显存溢出（与 E3 大图像内存保护相关）。
- :func:`validate_guidance_scale` / :func:`validate_num_inference_steps` /
  :func:`validate_strength`：校验扩散模型推理参数的取值范围。
"""

from __future__ import annotations

from mosaic.nodes.coerce import safe_float, safe_int  # noqa: F401

__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MIN_IMAGE_DIMENSION",
    "MAX_GUIDANCE_SCALE",
    "MAX_INFERENCE_STEPS",
    "MIN_INFERENCE_STEPS",
    "ALIGNMENT_MULTIPLE",
    "safe_int",
    "safe_float",
    "validate_image_dimensions",
    "validate_guidance_scale",
    "validate_num_inference_steps",
    "validate_strength",
]


# ---------------------------------------------------------------------------
# 参数范围常量
# ---------------------------------------------------------------------------
MAX_IMAGE_DIMENSION = 4096  # 单边最大像素，防止显存溢出
MIN_IMAGE_DIMENSION = 64  # 单边最小像素，diffusers 下限
MAX_GUIDANCE_SCALE = 20.0  # guidance_scale 上限
MAX_INFERENCE_STEPS = 100  # 推理步数上限
MIN_INFERENCE_STEPS = 1  # 推理步数下限
ALIGNMENT_MULTIPLE = 8  # diffusers 要求图像尺寸为 8 的倍数


# ---------------------------------------------------------------------------
# 安全类型转换（re-export 自 mosaic.nodes.coerce，保持向后兼容）
# ---------------------------------------------------------------------------
# safe_int / safe_float 由 mosaic.nodes.coerce 统一实现，此处仅 re-export，
# 以兼容历史导入 ``from mosaic.nodes.image._image_utils import safe_int``。


# ---------------------------------------------------------------------------
# 参数范围校验
# ---------------------------------------------------------------------------
def validate_image_dimensions(width: int, height: int) -> None:
    """校验图像尺寸上下限。

    Parameters
    ----------
    width:
        图像宽度（像素）。
    height:
        图像高度（像素）。

    Raises
    ------
    ValueError
        尺寸过小（< ``MIN_IMAGE_DIMENSION``）或过大（>
        ``MAX_IMAGE_DIMENSION``）时抛出。
    """
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too small: {width}x{height}, "
            f"minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"
        )
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too large: {width}x{height}, "
            f"maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        )


def validate_guidance_scale(value: float) -> None:
    """校验 guidance_scale 取值范围 ``[0, MAX_GUIDANCE_SCALE]``。

    Parameters
    ----------
    value:
        guidance_scale 值。

    Raises
    ------
    ValueError
        值超出 ``[0, MAX_GUIDANCE_SCALE]`` 时抛出。
    """
    if not 0 <= value <= MAX_GUIDANCE_SCALE:
        raise ValueError(
            f"guidance_scale must be in [0, {MAX_GUIDANCE_SCALE}], got {value}"
        )


def validate_num_inference_steps(value: int) -> None:
    """校验 num_inference_steps 取值范围 ``[1, MAX_INFERENCE_STEPS]``。

    Parameters
    ----------
    value:
        推理步数。

    Raises
    ------
    ValueError
        值超出 ``[MIN_INFERENCE_STEPS, MAX_INFERENCE_STEPS]`` 时抛出。
    """
    if not MIN_INFERENCE_STEPS <= value <= MAX_INFERENCE_STEPS:
        raise ValueError(
            f"num_inference_steps must be in "
            f"[{MIN_INFERENCE_STEPS}, {MAX_INFERENCE_STEPS}], got {value}"
        )


def validate_strength(value: float) -> None:
    """校验 strength 取值范围。

    允许 ``0.0``（表示不对原图做任何修改，是 img2img 的合法退化情形），
    因此有效范围为 ``[0.0, 1.0]``。

    .. note::
       原始需求文档建议范围为 ``(0, 1.0]``，但现有节点实现与测试用例
       （如 ``strength=0``）均将 0 视为合法值，此处采用 ``[0.0, 1.0]``
       以保持向后兼容。

    Parameters
    ----------
    value:
        strength 值。

    Raises
    ------
    ValueError
        值超出 ``[0.0, 1.0]`` 时抛出。
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"strength must be in [0.0, 1.0], got {value}")
