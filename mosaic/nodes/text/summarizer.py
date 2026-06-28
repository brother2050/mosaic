# mosaic/nodes/text/summarizer.py
"""TextSummarizer 节点 —— 文本摘要。

支持两种模式：

a) **专用摘要模型**：构造函数传入 ``model="facebook/bart-large-cnn"`` 等
   seq2seq 摘要模型时，使用 ``transformers.pipeline("summarization")``。
b) **通用生成模式**：未指定专用模型时，复用 :class:`BaseTextNode` 的因果
   语言模型，通过 prompt 指令摘要，支持 concise/detailed/bullet_points 风格。

通过 ``model`` 标识判断：含 ``"bart"``/``"t5"``/``"pegasus"``/``"distilbart"``
视为专用摘要模型。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode

__all__ = ["TextSummarizer"]

#: 支持的摘要风格。
_VALID_STYLES = {"concise", "detailed", "bullet_points"}

#: 各风格对应的 prompt 指令片段。
_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "concise": "请用简洁的语言概括以下文本的核心要点，不超过{max_length}字。",
    "detailed": "请对以下文本进行详细摘要，保留关键信息和逻辑结构，不超过{max_length}字。",
    "bullet_points": (
        "请将以下文本提炼为要点列表（使用 - 开头），每个要点一行，"
        "不超过{max_length}字。只输出要点，不要解释。"
    ),
}


def _is_specialized_summarization_model(model: str) -> bool:
    """判断给定模型标识是否为专用摘要模型（BART/T5/Pegasus 等）。"""
    lowered = model.lower()
    return any(tag in lowered for tag in ("bart", "t5", "pegasus", "distilbart"))


@registry.register
class TextSummarizer(BaseTextNode):
    """文本摘要节点。

    Parameters
    ----------
    model:
        HuggingFace 模型标识。若为专用摘要模型（如
        ``"facebook/bart-large-cnn"``）则使用 summarization pipeline；
        否则使用通用生成模式。默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数。

    Examples
    --------
    >>> summarizer = TextSummarizer(model="Qwen/Qwen2.5-7B-Instruct")
    >>> result = summarizer(MosaicData(
    ...     text="长文本...",
    ...     style="bullet_points",
    ... ))
    >>> print(result["summary"])
    """

    name: str = "text-summarizer"
    description: str = (
        "Summarize text into concise, detailed, or bullet-point styles. "
        "Supports both specialized summarization models and generic LLMs."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._is_specialized: bool = _is_specialized_summarization_model(model)
        # 专用模型对应的 transformers pipeline（惰性创建）
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载模型：专用摘要模型走 pipeline，否则走基类流程。"""
        if self._is_specialized:
            self._load_summarization_pipeline()
        else:
            super()._load_model()

    def _load_summarization_pipeline(self) -> None:
        """加载专用摘要模型的 pipeline。"""
        from transformers import pipeline  # type: ignore

        self._pipeline = pipeline(
            "summarization", model=self._model_name, tokenizer=self._model_name
        )
        # 兼容：pipeline 内部持有 model/tokenizer
        self._model = getattr(self._pipeline, "model", None)
        self._tokenizer = getattr(self._pipeline, "tokenizer", None)
        self._logger.info("Summarization pipeline %s loaded.", self._model_name)

    def unload(self) -> None:
        """释放模型与 pipeline。"""
        super().unload()
        self._pipeline = None

    # ------------------------------------------------------------------
    # 摘要执行
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本摘要。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str)；可选 ``max_length`` (int, 默认 150，字数)、
            ``style`` (str, 默认 "concise"，可选 "concise"/"detailed"/"bullet_points")、
            ``max_new_tokens`` (int, 默认 512)、``temperature`` (float, 默认 0.3)。

        Returns
        -------
        MosaicData
            包含 ``summary`` (str)、``original_length`` (int)、
            ``summary_length`` (int)、``compression_ratio`` (float)。

        Raises
        ------
        ValueError
            缺少 ``text`` 或 ``style`` 非法。
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
                    f"TextSummarizer requires 'text' (str), got {type(text).__name__}."
                )
            max_length = int(input_data.get("max_length", 150))
            style = str(input_data.get("style", "concise"))
            if style not in _VALID_STYLES:
                raise ValueError(
                    f"Invalid style {style!r}, expected one of {sorted(_VALID_STYLES)}."
                )

            original_length = len(text)

            # 原文过短：直接返回并标注
            if original_length <= max_length:
                elapsed = time.perf_counter() - t0
                result = MosaicData(
                    summary=text,
                    original_length=original_length,
                    summary_length=original_length,
                    compression_ratio=1.0,
                    note="Original text is not longer than max_length; returned as-is.",
                )
                self._emit_complete(
                    duration=elapsed,
                    output_summary={"compression_ratio": 1.0, "skipped": True},
                )
                return result

            max_new_tokens = int(input_data.get("max_new_tokens", 512))
            temperature = float(input_data.get("temperature", 0.3))

            # 执行摘要
            if self._is_specialized:
                summary = self._summarize_specialized(text, max_length)
            else:
                summary = self._summarize_generic(
                    text, style, max_length, max_new_tokens, temperature
                )
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        summary_length = len(summary)
        compression_ratio = (
            round(original_length / summary_length, 4) if summary_length > 0 else 0.0
        )
        result = MosaicData(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": compression_ratio,
            },
        )
        return result

    def _summarize_generic(
        self,
        text: str,
        style: str,
        max_length: int,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """通用生成模式：通过 prompt 指令摘要。"""
        instruction = _STYLE_INSTRUCTIONS[style].format(max_length=max_length)
        prompt = f"{instruction}\n\n原文：\n{text}"
        messages = [{"role": "user", "content": prompt}]
        generated, _, _ = self._generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
        )
        return generated.strip()

    def _summarize_specialized(self, text: str, max_length: int) -> str:
        """专用摘要模型模式：调用 summarization pipeline。"""
        # pipeline 的 max_length/min_length 以 token 计，这里粗略映射
        result = self._pipeline(
            text,
            max_length=max_length,
            min_length=max(10, max_length // 4),
            truncation=True,
        )
        return result[0]["summary_text"].strip()

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，标注摘要模式与风格。"""
        spec = super().describe()
        spec.model_info["summarization_mode"] = (
            "specialized" if self._is_specialized else "generic"
        )
        spec.model_info["supported_styles"] = sorted(_VALID_STYLES)
        return spec
