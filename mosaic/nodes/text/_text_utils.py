"""文本节点公共工具函数。

提供安全的类型转换、生成参数范围校验与超长 prompt 的上下文长度保护，
统一文本生成节点（``TextGenerator``/``Chat``/``TextSummarizer``/
``Translator``/``TextRewriter``）的输入处理逻辑，与
:mod:`mosaic.nodes.image._image_utils` 保持一致的设计风格。

核心功能：
- :func:`safe_int` / :func:`safe_float`：将用户输入安全转换为 int/float，
  转换失败时抛出包含参数名与实际值的清晰 ``ValueError``。
- :func:`validate_max_new_tokens` / :func:`validate_temperature` /
  :func:`validate_top_p`：校验因果语言模型生成参数的取值范围。
- :func:`estimate_tokens` / :func:`check_prompt_length`：粗略估算 prompt
  的 token 数，并在接近模型上下文长度上限时发出警告。
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "MAX_CONTEXT_LENGTH",
    "MAX_TEMPERATURE",
    "MAX_NEW_TOKENS",
    "MIN_NEW_TOKENS",
    "safe_int",
    "safe_float",
    "validate_max_new_tokens",
    "validate_temperature",
    "validate_top_p",
    "estimate_tokens",
    "check_prompt_length",
]


# ---------------------------------------------------------------------------
# 参数范围常量
# ---------------------------------------------------------------------------
#: 默认模型上下文长度（token），用于超长 prompt 警告的粗略估算。
MAX_CONTEXT_LENGTH = 4096
#: max_new_tokens 上限，防止生成过长导致显存/时间开销过大。
MAX_NEW_TOKENS = 8192
#: max_new_tokens 下限。
MIN_NEW_TOKENS = 1
#: 采样温度上限（transformers 常用上限为 2.0）。
MAX_TEMPERATURE = 2.0


# ---------------------------------------------------------------------------
# 安全类型转换
# ---------------------------------------------------------------------------
def safe_int(value: Any, param_name: str) -> int:
    """安全的 int 转换。

    Parameters
    ----------
    value:
        待转换的值（可能来自 ``MosaicData.get``）。
    param_name:
        参数名，用于构造错误消息。

    Returns
    -------
    int
        转换后的整数值。

    Raises
    ------
    ValueError
        ``value`` 无法转换为 int 时抛出，消息包含参数名与实际值。
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {param_name!r} must be an integer, "
            f"got {type(value).__name__}: {value!r}"
        ) from exc


def safe_float(value: Any, param_name: str) -> float:
    """安全的 float 转换。

    Parameters
    ----------
    value:
        待转换的值（可能来自 ``MosaicData.get``）。
    param_name:
        参数名，用于构造错误消息。

    Returns
    -------
    float
        转换后的浮点数。

    Raises
    ------
    ValueError
        ``value`` 无法转换为 float 时抛出，消息包含参数名与实际值。
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {param_name!r} must be a number, "
            f"got {type(value).__name__}: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# 参数范围校验
# ---------------------------------------------------------------------------
def validate_max_new_tokens(value: int) -> None:
    """校验 max_new_tokens 取值范围 ``[1, MAX_NEW_TOKENS]``。

    Parameters
    ----------
    value:
        最大生成 token 数。

    Raises
    ------
    ValueError
        值小于 1 或大于 ``MAX_NEW_TOKENS`` 时抛出。
    """
    if value < MIN_NEW_TOKENS:
        raise ValueError(
            f"max_new_tokens must be >= {MIN_NEW_TOKENS}, got {value}"
        )
    if value > MAX_NEW_TOKENS:
        raise ValueError(
            f"max_new_tokens must be <= {MAX_NEW_TOKENS}, got {value}"
        )


def validate_temperature(value: float) -> None:
    """校验 temperature 取值范围 ``[0, MAX_TEMPERATURE]``。

    Parameters
    ----------
    value:
        采样温度。

    Raises
    ------
    ValueError
        值超出 ``[0, MAX_TEMPERATURE]`` 时抛出。
    """
    if not 0 <= value <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be in [0, {MAX_TEMPERATURE}], got {value}"
        )


def validate_top_p(value: float) -> None:
    """校验 top_p 取值范围 ``[0, 1.0]``。

    Parameters
    ----------
    value:
        nucleus sampling 概率阈值。

    Raises
    ------
    ValueError
        值超出 ``[0, 1.0]`` 时抛出。
    """
    if not 0 <= value <= 1.0:
        raise ValueError(f"top_p must be in [0, 1.0], got {value}")


# ---------------------------------------------------------------------------
# 上下文长度保护
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数。

    采用保守估计：英文约 1 word ≈ 1.3 tokens，这里按 ``len(text.split()) * 2``
    估算以预留余量；对中文等不含空格的语言，退化为按字符数近似（每个字符
    约等于 1~2 个 token），同样乘以保守系数。

    Parameters
    ----------
    text:
        待估算的文本。

    Returns
    -------
    int
        估算的 token 数（保守上界）。
    """
    if not text:
        return 0
    # 以空白切分的词数（英文友好）；中文等无空格语言词数会偏少
    word_count = len(text.split())
    char_count = len(text)
    # 取词数估计与字符数估计的较大者，乘以保守系数，避免严重低估
    return int(max(word_count * 2, char_count * 0.6))


def check_prompt_length(
    prompt: str,
    logger: logging.Logger,
    max_context: int | None = None,
) -> int:
    """检查 prompt 是否接近或超过模型上下文长度，超限时发出警告。

    仅做粗略估算与告警，不截断或抛出异常（避免误伤合法长输入）。

    Parameters
    ----------
    prompt:
        待检查的 prompt 文本。
    logger:
        节点 logger，用于输出警告。
    max_context:
        模型上下文长度上限（token）。``None`` 时取节点属性
        ``_max_context_length``，否则回退到 :data:`MAX_CONTEXT_LENGTH`。

    Returns
    -------
    int
        估算的 prompt token 数。
    """
    if not prompt:
        return 0
    if max_context is None:
        max_context = MAX_CONTEXT_LENGTH
    estimated_tokens = estimate_tokens(prompt)
    if estimated_tokens > max_context * 0.8:
        logger.warning(
            "Prompt length (~%d tokens) is close to or exceeds "
            "model context length (%d). Consider truncating.",
            estimated_tokens, max_context,
        )
    return estimated_tokens
