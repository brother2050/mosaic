# mosaic/nodes/subtitle/translator.py
"""SubtitleTranslator 节点 —— 字幕翻译。

翻译字幕文本，保持时间轴不变。内部复用文本域 Translator 节点的翻译能力，
将所有片段批量翻译以减少模型调用次数。

设计要点
--------
* 时间轴完全保持不变，只翻译 ``text`` 字段。
* 批量翻译优化：将所有片段文本合并为一个大 prompt，一次调用完成。
* 翻译后重新检查字幕长度，必要时按标点拆分长行。
* 保留 ``speaker`` 等元信息。
* 如果未指定翻译模型，使用文本域 :class:`~mosaic.nodes.text.Translator`。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, SubtitleData

from mosaic.nodes.subtitle._base import BaseSubtitleNode

__all__ = ["SubtitleTranslator"]


@registry.register
class SubtitleTranslator(BaseSubtitleNode):
    """字幕翻译节点。

    翻译字幕文本，保持时间轴不变。

    Parameters
    ----------
    model:
        翻译模型标识，``None`` 时使用文本域 Translator 默认模型。
    source_language:
        源语言代码，默认 ``"auto"``（自动检测）。
    target_language:
        目标语言代码（必须指定），如 ``"zh"``、``"en"``。
    output_format:
        输出格式，``None`` 时保持原字幕格式。
    **kwargs:
        透传给 :class:`BaseSubtitleNode` 的参数。

    Examples
    --------
    >>> trans = SubtitleTranslator(target_language="en")
    >>> result = trans(MosaicData(
    ...     subtitle=my_subtitle_data,
    ... ))
    >>> print(result["subtitle"].segments[0]["text"])
    """

    name: str = "subtitle-translator"
    description: str = (
        "Translate subtitle text while preserving timestamps. "
        "Supports batch translation and automatic long-line splitting."
    )
    version: str = "0.1.0"
    input_types = ("subtitle", "mosaic")
    output_types = ("subtitle",)

    def __init__(
        self,
        model: str | None = None,
        source_language: str = "auto",
        target_language: str = "en",
        output_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name: str | None = model
        self._source_language: str = source_language
        self._target_language: str = target_language
        self._output_format: str | None = output_format
        # 内部翻译节点（延迟创建）
        self._translator_node: Any | None = None

    def _load_model(self) -> None:
        """加载翻译模型（复用文本域 Translator 节点）。"""
        from mosaic.nodes.text.translator import Translator

        translator_kwargs: dict[str, Any] = {
            "source_language": self._source_language,
            "target_language": self._target_language,
            "scheduler": self._scheduler,
            "bus": self._bus,
        }
        if self._model_name is not None:
            translator_kwargs["model"] = self._model_name

        self._translator_node = Translator(**translator_kwargs)
        self._translator_node.load()
        self._model = getattr(self._translator_node, "_model", None)
        self._logger.info(
            "SubtitleTranslator: translator model loaded (target=%s).",
            self._target_language,
        )

    def unload(self) -> None:
        """释放翻译模型。"""
        if self._translator_node is not None:
            self._translator_node.unload()
            self._translator_node = None
        self._model = None
        self._loaded = False

    def _batch_translate(self, texts: list[str]) -> list[str]:
        """批量翻译字幕文本。

        将所有文本合并为带序号的列表，一次调用翻译模型完成，
        然后拆分回各个片段。

        Parameters
        ----------
        texts:
            待翻译的文本列表。

        Returns
        -------
        list[str]
            翻译后的文本列表（与输入一一对应）。
        """
        if not texts:
            return []

        # 构造批量翻译 prompt
        numbered = "\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(texts)
        )

        prompt = (
            f"Translate the following subtitle lines to "
            f"{self._target_language}. Keep the [N] numbering format. "
            f"Translate each line independently:\n\n{numbered}"
        )

        # 使用翻译节点
        result = self._translator_node.run(
            MosaicData(text=prompt, target_language=self._target_language)
        )
        translated = result.get("translated_text", "")

        # 解析带序号的翻译结果
        import re

        translated_lines: dict[int, str] = {}
        for match in re.finditer(
            r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", translated, re.DOTALL
        ):
            idx = int(match.group(1))
            text = match.group(2).strip()
            translated_lines[idx] = text

        # 按原始顺序组装结果
        result_texts: list[str] = []
        for i in range(len(texts)):
            result_texts.append(translated_lines.get(i + 1, texts[i]))

        return result_texts

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行字幕翻译。

        Parameters
        ----------
        input_data:
            必须包含 ``subtitle`` (SubtitleData)；可选
            ``source_language`` (str)、``target_language`` (str)。

        Returns
        -------
        MosaicData
            包含 ``subtitle`` (SubtitleData)、``source_language`` (str)、
            ``target_language`` (str)、``translated_count`` (int)。

        Raises
        ------
        ValueError
            缺少 ``subtitle`` 输入。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            subtitle = input_data.get("subtitle")
            if subtitle is None:
                raise ValueError(
                    "SubtitleTranslator requires 'subtitle' (SubtitleData)."
                )

            # 提取 SubtitleData（如果是 dict 则转换）
            if isinstance(subtitle, dict):
                subtitle = SubtitleData(
                    segments=subtitle.get("segments", []),
                    format=subtitle.get("format", "srt"),
                )
            elif not isinstance(subtitle, SubtitleData):
                raise TypeError(
                    f"Expected SubtitleData, got {type(subtitle).__name__}."
                )

            source_lang = input_data.get(
                "source_language", self._source_language
            )
            target_lang = input_data.get(
                "target_language", self._target_language
            )

            segments = subtitle.segments
            if not segments:
                raise ValueError("Subtitle has no segments to translate.")

            # 提取所有待翻译文本
            texts = [seg.get("text", "") for seg in segments]

            # 批量翻译
            self._logger.info(
                "Translating %d subtitle segments (%s -> %s).",
                len(texts),
                source_lang,
                target_lang,
            )
            translated_texts = self._batch_translate(texts)

            # 构造翻译后的片段（保持时间轴不变）
            new_segments: list[dict[str, Any]] = []
            for i, (seg, trans_text) in enumerate(
                zip(segments, translated_texts), 1
            ):
                new_seg: dict[str, Any] = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": trans_text,
                    "index": i,
                }
                # 保留 speaker 等元信息
                if "speaker" in seg:
                    new_seg["speaker"] = seg["speaker"]
                new_segments.append(new_seg)

            # 翻译后检查并拆分过长片段
            new_segments = self._split_long_segments(
                new_segments, max_duration=10.0, max_chars=42
            )
            # 重新编号
            for i, seg in enumerate(new_segments, 1):
                seg["index"] = i

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 确定输出格式
        out_format = self._output_format or subtitle.subtitle_format

        # 构造输出 SubtitleData
        out_subtitle = self._make_subtitle_data(
            segments=new_segments,
            fmt=out_format,
            source_language=source_lang,
            target_language=target_lang,
        )

        result = MosaicData(
            subtitle=out_subtitle,
            source_language=source_lang,
            target_language=target_lang,
            translated_count=len(translated_texts),
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "segments": len(new_segments),
                "source_language": source_lang,
                "target_language": target_lang,
            },
        )
        return result
