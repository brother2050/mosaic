# mosaic/nodes/text/classifier.py
"""TextClassifier 节点 —— 文本分类。

支持三种模式，按构造参数与运行时 ``labels`` 自动选择：

a) **专用分类模型**：构造函数传入 ``model="bert-base-..."`` 等
   text-classification 模型时，直接用模型推理（labels 来自模型）。
b) **生成模式**：未指定专用模型且 ``len(labels) <= 10`` 时，用 LLM 通过
   prompt 从给定列表中选择类别。
c) **zero-shot 模式**：``len(labels) > 10`` 时，使用
   ``transformers.pipeline("zero-shot-classification")``。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode

__all__ = ["TextClassifier"]

#: 触发 zero-shot 模式的标签数量阈值。
_ZERO_SHOT_THRESHOLD = 10


def _is_specialized_classification_model(model: str) -> bool:
    """判断给定模型标识是否为专用文本分类模型。

    启发式：包含 ``"bert"``/``"roberta"``/``"distilbert"``/``"deberta"``
    且不含 ``"causal"``/``"gpt"``/``"qwen"``/``"llama"`` 等生成模型标识。
    """
    lowered = model.lower()
    gen_tags = ("causal", "gpt", "qwen", "llama", "mistral", "falcon", "chat")
    if any(tag in lowered for tag in gen_tags):
        return False
    cls_tags = ("bert", "roberta", "distilbert", "deberta", "xlnet", "albert")
    return any(tag in lowered for tag in cls_tags)


@registry.register
class TextClassifier(BaseTextNode):
    """文本分类节点。

    Parameters
    ----------
    model:
        HuggingFace 模型标识。若为专用分类模型（如 ``"bert-base-chinese"``
        微调分类头）则直接用模型推理；否则按 ``labels`` 数量选择生成模式
        或 zero-shot 模式。默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    zero_shot_model:
        zero-shot 分类使用的模型，默认 ``"facebook/bart-large-mnli"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数。

    Examples
    --------
    生成模式（少量标签）::

        >>> clf = TextClassifier(model="Qwen/Qwen2.5-7B-Instruct")
        >>> result = clf(MosaicData(
        ...     text="这部电影太棒了！",
        ...     labels=["正面", "负面", "中性"],
        ... ))
        >>> print(result["predicted_label"])

    多标签模式::

        >>> result = clf(MosaicData(
        ...     text="...",
        ...     labels=["科技", "财经", "体育"],
        ...     multi_label=True,
        ... ))
    """

    name: str = "text-classifier"
    description: str = (
        "Classify text into given labels. Supports specialized classification "
        "models, LLM-based selection (<=10 labels), and zero-shot (>10 labels)."
    )
    version: str = "0.1.0"
    input_types = ("text", "mosaic")
    output_types = ("text",)

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        zero_shot_model: str = "facebook/bart-large-mnli",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._is_specialized: bool = _is_specialized_classification_model(model)
        self._zero_shot_model_name: str = zero_shot_model
        # zero-shot pipeline（惰性创建）
        self._zero_shot_pipeline: Any = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载模型：专用分类模型走 pipeline，否则走基类流程。"""
        if self._is_specialized:
            self._load_classification_pipeline()
        else:
            super()._load_model()

    def _load_classification_pipeline(self) -> None:
        """加载专用文本分类 pipeline。"""
        from transformers import pipeline  # type: ignore

        self._pipeline = pipeline(
            "text-classification", model=self._model_name, tokenizer=self._model_name
        )
        self._model = getattr(self._pipeline, "model", None)
        self._tokenizer = getattr(self._pipeline, "tokenizer", None)
        self._logger.info("Classification pipeline %s loaded.", self._model_name)

    def _ensure_zero_shot_pipeline(self) -> None:
        """惰性加载 zero-shot 分类 pipeline（仅在需要时）。"""
        if self._zero_shot_pipeline is not None:
            return
        from transformers import pipeline  # type: ignore

        self._zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=self._zero_shot_model_name,
        )
        self._logger.info("Zero-shot pipeline %s loaded.", self._zero_shot_model_name)

    def unload(self) -> None:
        """释放模型与 pipeline。"""
        super().unload()
        self._pipeline = None
        self._zero_shot_pipeline = None

    # ------------------------------------------------------------------
    # 分类执行
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本分类。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str) 与 ``labels`` (list[str])；
            可选 ``multi_label`` (bool, 默认 False)。

        Returns
        -------
        MosaicData
            单标签模式含 ``predicted_label`` (str)、``scores`` (dict)、
            ``method`` (str)；多标签模式含 ``predicted_labels`` (list[str])。

        Raises
        ------
        ValueError
            缺少 ``text`` 或 ``labels``，或 ``labels`` 为空。
        """
        # 通过调度器确保主模型已加载（惰性加载 + LRU 淘汰）
        self._scheduler.ensure_loaded(self)

        # 发出开始事件
        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            text = input_data.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"TextClassifier requires 'text' (str), got {type(text).__name__}."
                )
            labels = input_data.get("labels")
            if not isinstance(labels, list) or not labels:
                raise ValueError(
                    "TextClassifier requires 'labels' (non-empty list[str])."
                )
            if not all(isinstance(lbl, str) for lbl in labels):
                raise ValueError("All labels must be strings.")
            multi_label = bool(input_data.get("multi_label", False))

            # 选择分类方法
            if self._is_specialized:
                method = "specialized"
                result_data = self._classify_specialized(text, labels, multi_label)
            elif len(labels) <= _ZERO_SHOT_THRESHOLD:
                method = "generation"
                result_data = self._classify_generation(text, labels, multi_label)
            else:
                method = "zero-shot"
                result_data = self._classify_zero_shot(text, labels, multi_label)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        result = MosaicData(method=method, **result_data)
        self._emit_complete(
            duration=elapsed,
            output_summary={"method": method, "labels_count": len(labels)},
        )
        return result

    # -- 专用分类模型 ------------------------------------------------------
    def _classify_specialized(
        self, text: str, labels: list[str], multi_label: bool
    ) -> dict[str, Any]:
        """专用分类模型模式：直接用模型推理。

        注意：专用分类模型的类别是预定义的，``labels`` 参数在此模式下仅用于
        过滤/映射输出。若模型输出不在 ``labels`` 中，仍原样返回。
        """
        preds = self._pipeline(text, top_k=None)
        # pipeline 返回 [{"label": str, "score": float}, ...]
        if isinstance(preds, dict):
            preds = [preds]
        scores = {p["label"]: float(p["score"]) for p in preds}
        if multi_label:
            # 多标签：返回所有 score > 0.5 的
            predicted = [lbl for lbl, sc in scores.items() if sc >= 0.5]
            return {"predicted_labels": predicted, "scores": scores}
        # 单标签：取最高分
        best = max(scores.items(), key=lambda kv: kv[1])
        return {"predicted_label": best[0], "scores": scores}

    # -- 生成模式 ----------------------------------------------------------
    def _classify_generation(
        self, text: str, labels: list[str], multi_label: bool
    ) -> dict[str, Any]:
        """生成模式：用 LLM 从给定标签列表中选择。"""
        labels_str = "、".join(labels)
        if multi_label:
            instruction = (
                f"请从以下类别中选择所有符合该文本的类别（可多选），"
                f"用顿号分隔，只输出类别名称，不要解释。\n"
                f"可选类别：{labels_str}"
            )
        else:
            instruction = (
                f"请从以下类别中选择最符合该文本的一个类别，"
                f"只输出类别名称，不要解释。\n"
                f"可选类别：{labels_str}"
            )
        prompt = f"{instruction}\n\n文本：\n{text}"
        messages = [{"role": "user", "content": prompt}]
        generated, _, _ = self._generate_from_messages(
            messages,
            max_new_tokens=64,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
        )

        # 解析模型输出：提取有效标签
        predicted = self._parse_labels(generated, labels)

        # 构造分数（生成模式无真实概率，用匹配标记做近似）
        if multi_label:
            scores = {lbl: (1.0 if lbl in predicted else 0.0) for lbl in labels}
            return {"predicted_labels": predicted, "scores": scores}
        # 单标签：取第一个匹配
        predicted_label = predicted[0] if predicted else labels[0]
        scores = {lbl: (1.0 if lbl == predicted_label else 0.0) for lbl in labels}
        return {"predicted_label": predicted_label, "scores": scores}

    @staticmethod
    def _parse_labels(generated: str, labels: list[str]) -> list[str]:
        """从模型生成文本中解析出有效标签。

        按顿号、逗号、换行等分隔符切分，再与候选标签做包含匹配。
        """
        text = generated.strip()
        # 去除常见前缀
        for prefix in ("类别：", "分类：", "答案：", "结果：", "Label:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        # 切分（含制表符等常见分隔符）
        import re

        parts = re.split(r"[、,，\n;；\t]+", text)
        matched: list[str] = []
        seen: set = set()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 精确匹配优先
            if part in labels and part not in seen:
                matched.append(part)
                seen.add(part)
                continue
            # 包含匹配（模型可能输出"类别：正面"）
            for lbl in labels:
                if lbl in part and lbl not in seen:
                    matched.append(lbl)
                    seen.add(lbl)
                    break
        return matched

    # -- zero-shot 模式 ---------------------------------------------------
    def _classify_zero_shot(
        self, text: str, labels: list[str], multi_label: bool
    ) -> dict[str, Any]:
        """zero-shot 模式：使用 zero-shot-classification pipeline。"""
        self._ensure_zero_shot_pipeline()
        result = self._zero_shot_pipeline(
            text, candidate_labels=labels, multi_label=multi_label
        )
        # result: {"labels": [...], "scores": [...]}
        scores = {
            lbl: float(sc)
            for lbl, sc in zip(result["labels"], result["scores"])
        }
        if multi_label:
            # 多标签：返回 score >= 0.5 的
            predicted = [lbl for lbl, sc in scores.items() if sc >= 0.5]
            return {"predicted_labels": predicted, "scores": scores}
        # 单标签：取最高分
        best_label = result["labels"][0]
        return {"predicted_label": best_label, "scores": scores}

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，标注分类模式。"""
        spec = super().describe()
        spec.model_info["classification_mode"] = (
            "specialized" if self._is_specialized else "adaptive"
        )
        spec.model_info["zero_shot_threshold"] = _ZERO_SHOT_THRESHOLD
        spec.model_info["zero_shot_model"] = self._zero_shot_model_name
        return spec
