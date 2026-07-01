# mosaic/nodes/text/generator.py
"""TextGenerator 节点 —— 根据 prompt 生成文本。

将用户提供的 ``prompt`` 包装为单轮对话消息，调用底层因果语言模型生成文本。
适用于单次文本生成场景；多轮对话请使用 :class:`~mosaic.nodes.text.chat.Chat`。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode
from mosaic.nodes.text._text_utils import (
    check_prompt_length,
    safe_float,
    safe_int,
    validate_max_new_tokens,
    validate_temperature,
    validate_top_p,
)

__all__ = ["TextGenerator"]


@registry.register
class TextGenerator(BaseTextNode):
    """文本生成节点。

    根据 ``prompt`` 生成文本，支持温度、top-p、采样开关等生成参数。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数（``device_map``/``torch_dtype``
        /``scheduler``/``bus`` 等）。

    Examples
    --------
    >>> gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
    >>> result = gen(MosaicData(prompt="写一首关于春天的诗"))
    >>> print(result["generated_text"])
    """

    name: str = "text-generator"
    description: str = (
        "Generate text from a prompt using a causal language model. "
        "Supports temperature, top-p, and sampling controls."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本生成。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``max_new_tokens`` (int, 默认 512)、
            ``temperature`` (float, 默认 0.7)、``top_p`` (float, 默认 0.9)、
            ``do_sample`` (bool, 默认 True)。

        Returns
        -------
        MosaicData
            包含 ``generated_text`` (str)、``input_tokens`` (int)、
            ``output_tokens`` (int)。

        Raises
        ------
        ValueError
            缺少 ``prompt`` 或 ``prompt`` 非字符串。
        """
        # 通过调度器确保模型已加载（惰性加载 + LRU 淘汰）
        self._scheduler.ensure_loaded(self)

        # 发出开始事件
        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"TextGenerator requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            # 提取生成参数（安全类型转换）
            max_new_tokens = safe_int(
                input_data.get("max_new_tokens", 512), "max_new_tokens"
            )
            temperature = safe_float(
                input_data.get("temperature", 0.7), "temperature"
            )
            top_p = safe_float(input_data.get("top_p", 0.9), "top_p")
            do_sample = bool(input_data.get("do_sample", True))

            # 参数范围校验
            validate_max_new_tokens(max_new_tokens)
            validate_temperature(temperature)
            validate_top_p(top_p)

            # 超长上下文保护
            check_prompt_length(prompt, self._logger)

            # 构造单轮对话消息
            messages = [{"role": "user", "content": prompt}]

            # 执行生成
            generated_text, input_tokens, output_tokens = (
                self._generate_from_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        result = MosaicData(
            prompt=generated_text,
            generated_text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return result

    def stream(self, input_data: MosaicData) -> Any:
        """流式生成文本。

        与 :meth:`run` 使用相同的输入约定，但通过
        :meth:`BaseTextNode.stream_generate` 逐 token yield 生成的文本片段，
        适用于需要实时输出的场景。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``max_new_tokens`` (int, 默认 512)、
            ``temperature`` (float, 默认 0.7)、``top_p`` (float, 默认 0.9)、
            ``do_sample`` (bool, 默认 True)。

        Yields
        ------
        str
            生成的文本片段。

        Raises
        ------
        ValueError
            缺少 ``prompt`` 或 ``prompt`` 非字符串。
        """
        # 校验输入
        prompt = input_data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"TextGenerator requires 'prompt' (non-empty str), "
                f"got {type(prompt).__name__}."
            )

        # 提取生成参数（安全类型转换）
        max_new_tokens = safe_int(
            input_data.get("max_new_tokens", 512), "max_new_tokens"
        )
        temperature = safe_float(
            input_data.get("temperature", 0.7), "temperature"
        )
        top_p = safe_float(input_data.get("top_p", 0.9), "top_p")
        do_sample = bool(input_data.get("do_sample", True))

        # 参数范围校验
        validate_max_new_tokens(max_new_tokens)
        validate_temperature(temperature)
        validate_top_p(top_p)

        # 超长上下文保护
        check_prompt_length(prompt, self._logger)

        # 构造单轮对话消息
        messages = [{"role": "user", "content": prompt}]

        yield from self.stream_generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
