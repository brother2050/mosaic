"""跨域节点通用的安全类型转换工具。

提供 :func:`safe_int` / :func:`safe_float`，统一处理来自
:meth:`mosaic.core.types.MosaicData.get` 的值在「键存在但值为 ``None``」时的
回退逻辑。

背景
----
``MosaicData.get(key, default)`` 的语义与 ``dict.get`` 一致：仅当 *键不存在*
时返回 ``default``；当键存在但值为 ``None`` 时返回 ``None``。若直接对返回值
调用 ``int(None)`` / ``float(None)`` 将抛出难以定位的 ``TypeError``。

本模块的转换函数在 ``value`` 为 ``None`` 时回退到调用方提供的 ``default``，
从而把「键缺失」与「键存在但值为 ``None``」两种情况统一为同一个默认值；当
``value`` 非 ``None`` 但无法转换时，抛出包含参数名与实际值的清晰
``ValueError``。

.. note::
   本模块是 ``safe_int`` / ``safe_float`` 的唯一定义处。
   :mod:`mosaic.nodes.image._image_utils` /
   :mod:`mosaic.nodes.video._video_utils` /
   :mod:`mosaic.nodes.text._text_utils` 中的同名函数已改为从此处 re-export，
   以统一语义：``value`` 为 ``None`` 且未提供 ``default`` 时抛 ``ValueError``，
   提供 ``default`` 时回退 ``default``。各域节点应统一从本模块（或各域
   ``_*_utils`` 的 re-export）导入，避免重复实现。
"""

from __future__ import annotations

from typing import Any

__all__ = ["safe_int", "safe_float"]


def safe_int(
    value: Any, name: str = "value", default: int | None = None
) -> int:
    """将 ``value`` 安全转换为 ``int``，``None`` 时回退 ``default``。

    与直接调用 ``int(value)`` 的区别：

    - ``value`` 为 ``None`` 且提供了 ``default`` 时返回 ``default``（而非抛
      ``TypeError``），从而把 ``MosaicData.get`` 在「键存在但值为 ``None``」
      时返回 ``None`` 的情形平滑回退到默认值。
    - ``value`` 非 ``None`` 但无法转换时，抛出包含参数名与实际值的
      ``ValueError``，便于定位。
    - ``value`` 为 ``None`` 且 ``default`` 也为 ``None`` 时抛出 ``ValueError``
      （视为必填参数缺失），避免将 ``None`` 静默传播到下游导致更难排查的崩溃。

    Parameters
    ----------
    value:
        待转换的值（通常来自 ``input_data.get(...)``）。
    name:
        参数名，用于构造错误消息。
    default:
        当 ``value`` 为 ``None`` 时的回退值；默认 ``None``（此时 ``None``
        输入将抛异常）。

    Returns
    -------
    int
        转换后的整数；若 ``value`` 为 ``None`` 则返回 ``default``。

    Raises
    ------
    ValueError
        ``value`` 非 ``None`` 但无法转换为 ``int``，或 ``value`` 与
        ``default`` 同时为 ``None`` 时抛出。
    """
    if value is None:
        if default is None:
            raise ValueError(
                f"Parameter {name!r} is required (got None) "
                f"and no default was provided."
            )
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {name!r} must be an integer, "
            f"got {type(value).__name__}: {value!r}"
        ) from exc


def safe_float(
    value: Any, name: str = "value", default: float | None = None
) -> float:
    """将 ``value`` 安全转换为 ``float``，``None`` 时回退 ``default``。

    语义与 :func:`safe_int` 对称：``value`` 为 ``None`` 且提供 ``default``
    时返回 ``default``；非 ``None`` 但无法转换时抛出清晰的 ``ValueError``；
    ``value`` 与 ``default`` 同时为 ``None`` 时抛出 ``ValueError``。

    Parameters
    ----------
    value:
        待转换的值（通常来自 ``input_data.get(...)``）。
    name:
        参数名，用于构造错误消息。
    default:
        当 ``value`` 为 ``None`` 时的回退值；默认 ``None``（此时 ``None``
        输入将抛异常）。

    Returns
    -------
    float
        转换后的浮点数；若 ``value`` 为 ``None`` 则返回 ``default``。

    Raises
    ------
    ValueError
        ``value`` 非 ``None`` 但无法转换为 ``float``，或 ``value`` 与
        ``default`` 同时为 ``None`` 时抛出。
    """
    if value is None:
        if default is None:
            raise ValueError(
                f"Parameter {name!r} is required (got None) "
                f"and no default was provided."
            )
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {name!r} must be a number, "
            f"got {type(value).__name__}: {value!r}"
        ) from exc
