# mosaic/nodes/text/chat.py
"""Chat 节点 —— 多轮对话。

接收对话历史 ``messages``，调用底层因果语言模型生成下一轮回复，并返回
包含新回复的完整对话历史。支持可选的系统提示词。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode

__all__ = ["Chat"]


@registry.register
class Chat(BaseTextNode):
    """多轮对话节点。

    接收对话历史，生成模型回复，并返回更新后的完整对话历史。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数。

    Examples
    --------
    >>> chat = Chat(model="Qwen/Qwen2.5-7B-Instruct")
    >>> result = chat(MosaicData(
    ...     messages=[{"role": "user", "content": "你好"}],
    ...     system_prompt="你是一个友好的助手。",
    ... ))
    >>> print(result["reply"])
    >>> # result["messages"] 包含完整的对话历史（含新回复）
    """

    name: str = "chat"
    description: str = (
        "Multi-turn chat: generate a reply from conversation history. "
        "Supports an optional system prompt and returns updated messages."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行多轮对话生成。

        Parameters
        ----------
        input_data:
            必须包含 ``messages`` (list[dict])，格式为
            ``[{"role": "user", "content": "..."}, ...]``；
            可选 ``system_prompt`` (str)、``max_new_tokens`` (int, 默认 1024)、
            ``temperature`` (float, 默认 0.7)、``top_p`` (float, 默认 0.9)、
            ``do_sample`` (bool, 默认 True)。

        Returns
        -------
        MosaicData
            包含 ``reply`` (str)、``messages`` (list[dict]，含新 assistant
            回复)、``input_tokens`` (int)、``output_tokens`` (int)。

        Raises
        ------
        ValueError
            缺少 ``messages`` 或格式不正确。
        """
        # 通过调度器确保模型已加载（惰性加载 + LRU 淘汰）
        self._scheduler.ensure_loaded(self)

        # 发出开始事件
        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            messages = input_data.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    "Chat requires 'messages' (list[dict]) with at least one "
                    f"message, got {type(messages).__name__}."
                )
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"Each message must be a dict with 'role' and 'content', got {msg!r}."
                    )

            # 提取参数
            system_prompt = input_data.get("system_prompt")
            max_new_tokens = int(input_data.get("max_new_tokens", 1024))
            temperature = float(input_data.get("temperature", 0.7))
            top_p = float(input_data.get("top_p", 0.9))
            do_sample = bool(input_data.get("do_sample", True))

            # 构造完整消息列表：可选系统提示词 + 对话历史
            full_messages: list[dict[str, str]] = []
            if isinstance(system_prompt, str) and system_prompt.strip():
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            # 执行生成
            reply, input_tokens, output_tokens = self._generate_from_messages(
                full_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 构造更新后的对话历史
        updated_messages = list(full_messages)
        updated_messages.append({"role": "assistant", "content": reply})

        result = MosaicData(
            reply=reply,
            messages=updated_messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reply_length": len(reply),
            },
        )
        return result
