# mosaic/nodes/text/translator.py
"""Translator 节点 —— 文本翻译。

支持两种模式：

a) **专用翻译模型**：构造函数传入 ``model="Helsinki-NLP/opus-mt-*"`` 时，
   使用 ``transformers.MarianMTModel`` 进行 seq2seq 翻译。
b) **通用生成模式**：未指定专用模型时，复用 :class:`BaseTextNode` 的因果
   语言模型，通过 prompt 指令完成翻译。

通过构造参数 ``model`` 的前缀判断模式：以 ``"Helsinki-NLP"`` 或包含
``"opus-mt"``/``"nllb"`` 视为专用翻译模型。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.text._base import BaseTextNode

__all__ = ["Translator"]

#: 语言代码 -> 语言名称映射表（用于构造 prompt）。
_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "fr": "Français",
    "de": "Deutsch",
    "ko": "한국어",
    "es": "Español",
    "ru": "Русский",
    "it": "Italiano",
    "pt": "Português",
    "ar": "العربية",
    "th": "ภาษาไทย",
    "vi": "Tiếng Việt",
    "auto": "自动检测",
}


def _is_specialized_translation_model(model: str) -> bool:
    """判断给定模型标识是否为专用翻译模型（MarianMT / NLLB / M2M100 等）。"""
    lowered = model.lower()
    return (
        lowered.startswith("helsinki-nlp")
        or "opus-mt" in lowered
        or "nllb" in lowered
        or "m2m100" in lowered
    )


@registry.register
class Translator(BaseTextNode):
    """文本翻译节点。

    Parameters
    ----------
    model:
        HuggingFace 模型标识。若为专用翻译模型（如
        ``"Helsinki-NLP/opus-mt-zh-en"``）则使用 MarianMT；否则使用通用
        生成模式。默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    **kwargs:
        透传给 :class:`BaseTextNode` 的参数。

    Examples
    --------
    通用模式::

        >>> tr = Translator(model="Qwen/Qwen2.5-7B-Instruct")
        >>> result = tr(MosaicData(text="你好", target_language="en"))
        >>> print(result["translated_text"])

    专用模型模式::

        >>> tr = Translator(model="Helsinki-NLP/opus-mt-zh-en")
        >>> result = tr(MosaicData(text="你好", target_language="en"))
    """

    name: str = "translator"
    description: str = (
        "Translate text between languages. Supports both specialized "
        "translation models (MarianMT/NLLB) and generic LLM-based translation."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._is_specialized: bool = _is_specialized_translation_model(model)

    # ------------------------------------------------------------------
    # 模型加载（专用翻译模型需要不同的加载逻辑）
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载模型：专用翻译模型用 MarianMT，否则走基类流程。"""
        if self._is_specialized:
            self._load_translation_model()
        else:
            super()._load_model()

    def _load_translation_model(self) -> None:
        """加载 MarianMT/NLLB 翻译模型与 tokenizer。"""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        self._model.eval()
        self._logger.info("Translation model %s loaded.", self._model_name)

    # ------------------------------------------------------------------
    # 翻译执行
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本翻译。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str) 与 ``target_language`` (str，如 "zh"/"en")；
            可选 ``source_language`` (str, 默认 "auto")、
            ``max_new_tokens`` (int, 默认 512)、``temperature`` (float, 默认 0.3)。

        Returns
        -------
        MosaicData
            包含 ``translated_text`` (str)、``source_language`` (str)、
            ``target_language`` (str)。

        Raises
        ------
        ValueError
            缺少 ``text`` 或 ``target_language``。
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
                    f"Translator requires 'text' (str), got {type(text).__name__}."
                )
            target_language = input_data.get("target_language")
            if not isinstance(target_language, str) or not target_language.strip():
                raise ValueError(
                    "Translator requires 'target_language' (str), e.g. 'zh', 'en'."
                )
            source_language = str(input_data.get("source_language", "auto"))

            max_new_tokens = int(input_data.get("max_new_tokens", 512))
            temperature = float(input_data.get("temperature", 0.3))

            # 执行翻译
            if self._is_specialized:
                translated = self._translate_specialized(text, max_new_tokens)
            else:
                translated = self._translate_generic(
                    text, source_language, target_language, max_new_tokens, temperature
                )
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        result = MosaicData(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "translated_length": len(translated),
                "mode": "specialized" if self._is_specialized else "generic",
            },
        )
        return result

    def _translate_generic(
        self,
        text: str,
        source_language: str,
        target_language: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """通用生成模式：通过 prompt 指令翻译。"""
        src_name = _LANGUAGE_NAMES.get(source_language, source_language)
        tgt_name = _LANGUAGE_NAMES.get(target_language, target_language)
        prompt = (
            f"将以下{src_name}文本翻译为{tgt_name}，只输出翻译结果，不要解释：\n\n{text}"
        )
        messages = [{"role": "user", "content": prompt}]
        generated, _, _ = self._generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
        )
        return generated.strip()

    def _translate_specialized(self, text: str, max_new_tokens: int) -> str:
        """专用翻译模型模式：直接调用 MarianMT generate。"""
        import torch  # type: ignore

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        model_device = self._infer_device()
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，标注翻译模式。"""
        spec = super().describe()
        spec.model_info["translation_mode"] = (
            "specialized" if self._is_specialized else "generic"
        )
        spec.model_info["supported_languages"] = sorted(_LANGUAGE_NAMES.keys())
        return spec
