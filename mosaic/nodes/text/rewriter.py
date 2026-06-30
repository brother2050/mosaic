# mosaic/nodes/text/rewriter.py
"""TextRewriter 节点 —— 按指定风格/要求改写文本。

将输入文本按 ``instruction`` 描述的要求（如"改为正式语气"、"简化表达"）
进行改写，保持语义不变。底层复用 :class:`BaseTextNode` 的通用生成流程。
"""

from __future__ import annotations

import re
import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode

__all__ = ["TextRewriter"]

#: 未提供 instruction 时的默认改写指令。
_DEFAULT_INSTRUCTION = "请改写以下文本，保持语义不变，提升表达质量。只输出改写后的文本，不要任何解释。"


@registry.register
class TextRewriter(BaseTextNode):
    """文本改写节点。

    根据 ``instruction`` 改写输入文本，保持语义不变。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数。

    Examples
    --------
    >>> rewriter = TextRewriter(model="Qwen/Qwen2.5-7B-Instruct")
    >>> result = rewriter(MosaicData(
    ...     text="这个东西很好用",
    ...     instruction="改为正式语气",
    ... ))
    >>> print(result["rewritten_text"])
    """

    name: str = "text-rewriter"
    description: str = (
        "Rewrite text according to an instruction (e.g. tone, style) "
        "while preserving the original meaning."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本改写。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str)；可选 ``instruction`` (str, 默认提升表达质量)、
            ``max_new_tokens`` (int, 默认 512)、``temperature`` (float, 默认 0.7)。

        Returns
        -------
        MosaicData
            包含 ``rewritten_text`` (str) 与 ``original_text`` (str)。

        Raises
        ------
        ValueError
            缺少 ``text`` 或 ``text`` 非字符串。
        """
        # 通过调度器确保模型已加载（惰性加载 + LRU 淘汰）
        self._scheduler.ensure_loaded(self)

        # 发出开始事件
        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            text = input_data.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"TextRewriter requires 'text' (str), "
                    f"got {type(text).__name__}."
                )

            # 提取参数
            instruction = input_data.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                instruction = _DEFAULT_INSTRUCTION
            max_new_tokens = int(input_data.get("max_new_tokens", 512))
            temperature = float(input_data.get("temperature", 0.7))
            top_p = float(input_data.get("top_p", 0.9))
            do_sample = bool(input_data.get("do_sample", True))

            # 构造 prompt：指令 + 原文
            prompt = f"{instruction}\n\n原文：\n{text}"
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
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 清理模型可能附带的解释性文字，只保留改写结果
        rewritten_text = self._extract_rewritten(generated_text)

        result = MosaicData(
            rewritten_text=rewritten_text,
            original_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "original_length": len(text),
                "rewritten_length": len(rewritten_text),
            },
        )
        return result

    @staticmethod
    def _extract_rewritten(generated_text: str) -> str:
        """从模型输出中提取改写后的文本。

        去除常见的解释性前缀/后缀（如"改写后的文本："、"以下是..."等），
        保留主体内容。启发式规则，非精确解析。
        """
        text = generated_text.strip()
        if not text:
            return text

        # 去除常见前缀
        prefix_patterns = [
            r"^改写后的文本[：:]\s*",
            r"^改写结果[：:]\s*",
            r"^结果[：:]\s*",
            r"^以下是改写后的文本[：:]\s*",
            r"^以下是.*?[：:]\s*",
            r"^output[：:]\s*",
        ]
        for pat in prefix_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # 去除常见后缀解释（如"希望这个改写..."、"以上是..."等）
        suffix_markers = [
            "\n\n希望", "\n希望", "\n\n注：", "\n注：",
            "\n\n说明：", "\n说明：", "\n\n备注：", "\n备注：",
            "\n\n以上是", "\n以上是", "\n\n请注意", "\n请注意",
        ]
        for marker in suffix_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]

        return text.strip()
