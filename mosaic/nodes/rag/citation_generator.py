# mosaic/nodes/rag/citation_generator.py
"""CitationGenerator 节点 —— 引用生成。

基于检索结果和用户问题，生成带引用标注的回答。

设计要点
--------
* 复用文本域 :class:`BaseTextNode` 的 LLM 加载与生成逻辑。
* 构造 prompt 时将检索结果作为上下文注入，使用 [1]、[2] 等标注引用。
* 支持三种引用风格：inline（行内）、footnote（脚注）、academic（学术）。
* 解析 LLM 输出中的引用标注，生成结构化的 citations 列表。
* 使用低 temperature（0.3）保证回答的准确性和忠实度。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

from mosaic.nodes.rag._base import BaseRagNode

__all__ = ["CitationGenerator"]


# 常见 LLM 的粗略显存估算
_LLM_VRAM: Dict[str, float] = {
    "Qwen/Qwen2.5-7B-Instruct": 16.0,
    "Qwen/Qwen2.5-14B-Instruct": 30.0,
    "Qwen/Qwen2.5-72B-Instruct": 150.0,
    "meta-llama/Llama-3.1-8B-Instruct": 18.0,
}


# Prompt 模板
_PROMPT_TEMPLATE_ZH = (
    "基于以下参考资料回答用户问题。回答中请使用 [1]、[2] 等标注引用来源。"
    "只使用参考资料中的信息回答，如果资料不足以回答，请说明。\n\n"
    "参考资料：\n{context}\n\n"
    "用户问题：{query}"
)

_PROMPT_TEMPLATE_EN = (
    "Based on the following reference materials, answer the user's question. "
    "Please use [1], [2], etc. to cite sources in your answer. "
    "Only use information from the reference materials. "
    "If the materials are insufficient, please state so.\n\n"
    "Reference materials:\n{context}\n\n"
    "User question: {query}"
)


@registry.register
class CitationGenerator(BaseRagNode):
    """引用生成节点。

    基于 :class:`Retriever` 的检索结果和用户问题，生成带引用标注的回答。

    Parameters
    ----------
    llm_model:
        用于生成回答的大语言模型标识。
    citation_style:
        引用风格，``"inline"`` / ``"footnote"`` / ``"academic"``。
    include_sources:
        是否在回答中包含来源信息。
    max_tokens:
        最大生成 token 数。
    temperature:
        生成温度（低值更忠实）。
    device_map:
        传递给 ``from_pretrained`` 的 ``device_map``。
    torch_dtype:
        权重精度。
    trust_remote_code:
        是否信任远程代码。
    scheduler:
        显存调度器。
    bus:
        事件总线。

    Examples
    --------
    >>> gen = CitationGenerator(llm_model="Qwen/Qwen2.5-7B-Instruct")
    >>> gen.load()
    >>> result = gen(MosaicData(
    ...     query="什么是机器学习？",
    ...     results=[
    ...         {"content": "ML是AI的子领域...", "score": 0.9, "source": "doc1"},
    ...     ],
    ... ))
    >>> result["answer"]  # str, 包含 [1] 引用标注
    >>> result["citations"]  # list[dict]
    """

    name: str = "citation-generator"
    domain: str = "rag"
    description: str = (
        "Generate answers with citation annotations based on retrieved "
        "context and user query."
    )
    version: str = "0.1.0"
    input_types: List[str] = ["rag_query_result", "mosaic"]
    output_types: List[str] = ["text", "mosaic"]

    def __init__(
        self,
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        citation_style: str = "inline",
        include_sources: bool = True,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        device_map: str = "auto",
        torch_dtype: str = "fp16",
        trust_remote_code: bool = True,
        bus: Optional[EventBus] = None,
        scheduler: Optional[Scheduler] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._model_name: str = llm_model
        self._citation_style: str = citation_style
        self._include_sources: bool = include_sources
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._device_map: str = device_map
        self._torch_dtype: str = torch_dtype
        self._trust_remote_code: bool = trust_remote_code
        self._scheduler: Scheduler = scheduler or get_scheduler()

        # 运行时状态
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载 LLM 模型与 tokenizer。

        使用 ``transformers`` 加载因果语言模型，通过 :class:`Scheduler`
        管理显存。
        """
        self._scheduler.track(self)

        if self._model is not None and self._tokenizer is not None:
            self._loaded = True
            return

        self._logger.info("Loading LLM: %s", self._model_name)
        self._load_model()
        self._loaded = True

    def unload(self) -> None:
        """释放 LLM 模型。"""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._logger.info("CitationGenerator unloaded (model=%s).", self._model_name)

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行引用生成。

        Parameters
        ----------
        input_data:
            必须包含 ``query`` (str) 和 ``results`` (list[dict])。
            可选：``citation_style`` (str)、``language`` (str, 默认 "zh")。

        Returns
        -------
        MosaicData
            包含 ``answer`` (str)、``citations`` (list[dict])、
            ``query`` (str)、``sources_used`` (int)。

        Raises
        ------
        ValueError
            缺少 ``query`` 或 ``results``。
        """
        self._scheduler.ensure_loaded(self)
        self._emit_start()
        t0 = time.perf_counter()

        try:
            query = input_data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"CitationGenerator requires 'query' (str), "
                    f"got {type(query).__name__}."
                )

            results = input_data.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(
                    "CitationGenerator requires 'results' (non-empty list)."
                )

            # 覆盖 citation_style
            style = input_data.get("citation_style", self._citation_style)
            language = input_data.get("language", "zh")

            # 构造 prompt
            context, citation_map = self._build_context(results, style, language)
            prompt = self._build_prompt(query, context, language)

            # 生成回答
            answer, input_tokens, output_tokens = self._generate(prompt)

            # 解析引用
            citations = self._parse_citations(answer, results, citation_map)

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                answer=answer,
                citations=citations,
                query=query,
                sources_used=len(citations),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            self._emit_complete(
                duration=elapsed,
                output_summary={
                    "sources_used": len(citations),
                    "answer_length": len(answer),
                },
            )
            return result

        except Exception as exc:
            self._emit_error(exc)
            raise

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        vram = _LLM_VRAM.get(self._model_name, 16.0)
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info={
                "name": self._model_name,
                "source": "HuggingFace",
                "license": "See model card on HuggingFace",
                "vram_gb": vram,
                "dtype": self._torch_dtype,
                "device_map": self._device_map,
            },
        )

    # ------------------------------------------------------------------
    # LLM 加载与生成
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载 transformers 模型与 tokenizer。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(self._torch_dtype, torch.float16)

        load_kwargs: dict = {
            "device_map": self._device_map,
            "trust_remote_code": self._trust_remote_code,
        }
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                dtype=resolved_dtype,
                **load_kwargs,
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=resolved_dtype,
                **load_kwargs,
            )
        self._model.eval()

        self._logger.info(
            "LLM loaded (model=%s, dtype=%s, device_map=%s).",
            self._model_name,
            self._torch_dtype,
            self._device_map,
        )

    def _generate(self, prompt: str) -> Tuple[str, int, int]:
        """使用 LLM 生成回答。

        Returns
        -------
        Tuple[str, int, int]
            ``(generated_text, input_tokens, output_tokens)``。
        """
        import torch  # type: ignore

        messages = [{"role": "user", "content": prompt}]

        # 构造输入
        if hasattr(self._tokenizer, "apply_chat_template"):
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self._tokenizer(formatted, return_tensors="pt")
            input_ids = model_inputs["input_ids"]
        else:
            flat = "\n".join(m.get("content", "") for m in messages)
            encoded = self._tokenizer(flat, return_tensors="pt")
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded

        # 迁移到模型设备
        model_device = self._infer_device()
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(model_device)
        input_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else 0

        attention_mask = None
        try:
            attention_mask = torch.ones_like(input_ids)
        except Exception:  # noqa: BLE001
            pass

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._max_tokens,
            "do_sample": self._temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if self._temperature > 0:
            gen_kwargs["temperature"] = max(self._temperature, 1e-5)
            gen_kwargs["top_p"] = 0.9

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0, input_length:]
        generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_tokens = int(new_tokens.shape[-1])

        return generated_text, int(input_length), output_tokens

    def _infer_device(self) -> str:
        """推断模型所在设备。"""
        if self._model is None:
            return self._scheduler.device
        try:
            return next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            return self._scheduler.device

    # ------------------------------------------------------------------
    # Prompt 构造与引用解析
    # ------------------------------------------------------------------
    def _build_context(
        self,
        results: List[Dict[str, Any]],
        style: str,
        language: str,
    ) -> Tuple[str, Dict[int, int]]:
        """构造上下文文本和引用映射。

        Parameters
        ----------
        results:
            检索结果列表。
        style:
            引用风格。
        language:
            语言代码。

        Returns
        -------
        Tuple[str, Dict[int, int]]
            ``(context_text, citation_map)``，其中 citation_map 是
            引用编号 → results 索引的映射。
        """
        context_parts: List[str] = []
        citation_map: Dict[int, int] = {}

        for i, result in enumerate(results):
            citation_id = i + 1
            citation_map[citation_id] = i

            content = result.get("content", "")
            source = result.get("source", "unknown")
            score = result.get("score", 0.0)

            if self._include_sources:
                entry = f"[{citation_id}] {content} (来源: {source})"
            else:
                entry = f"[{citation_id}] {content}"

            context_parts.append(entry)

        return "\n".join(context_parts), citation_map

    def _build_prompt(self, query: str, context: str, language: str) -> str:
        """构造完整的 prompt。"""
        template = (
            _PROMPT_TEMPLATE_ZH if language == "zh" else _PROMPT_TEMPLATE_EN
        )
        return template.format(context=context, query=query)

    def _parse_citations(
        self,
        answer: str,
        results: List[Dict[str, Any]],
        citation_map: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        """解析 LLM 输出中的引用标注，生成结构化 citations 列表。

        Parameters
        ----------
        answer:
            LLM 生成的回答文本。
        results:
            原始检索结果列表。
        citation_map:
            引用编号 → results 索引的映射。

        Returns
        -------
        List[Dict[str, Any]]
            引用列表，每个 dict 包含 ``citation_id``、``source``、
            ``content``、``score``。
        """
        # 匹配 [1], [2], [1,2], [1-3] 等引用标注
        pattern = r"\[(\d+(?:\s*[-,]\s*\d+)*)\]"
        matches = re.findall(pattern, answer)

        used_ids: set = set()
        for match in matches:
            # 处理 [1-3] 和 [1,2] 格式
            parts = re.split(r"[-,]", match)
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    used_ids.add(int(part))

        citations: List[Dict[str, Any]] = []
        for citation_id in sorted(used_ids):
            result_idx = citation_map.get(citation_id)
            if result_idx is not None and result_idx < len(results):
                result = results[result_idx]
                citations.append({
                    "citation_id": citation_id,
                    "source": result.get("source", "unknown"),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })

        return citations
