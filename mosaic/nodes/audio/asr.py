# mosaic/nodes/audio/asr.py
"""ASR 节点 —— 语音识别（自动语音转文本）。

使用 OpenAI Whisper 模型将语音转换为文本，支持多语言自动检测、
翻译（转为英文）以及长音频分片处理。

设计要点
--------
* 基于 ``transformers.pipeline("automatic-speech-recognition")`` 加载
  Whisper 模型，支持 large-v3/medium/small/base/tiny 等不同规格。
* 超过 30 秒的长音频使用 ``chunk_length_s`` 分片策略自动处理。
* 输出包含分段信息（``segments``），每段含 ``start``/``end``/``text``，
  方便后续字幕生成。
* 如果输入是文件路径，自动加载为波形数据。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["ASR"]


@registry.register
class ASR(BaseAudioNode):
    """语音识别节点。

    将语音转换为文本，基于 OpenAI Whisper 模型。

    Parameters
    ----------
    model:
        Whisper 模型标识，默认 ``"openai/whisper-large-v3"``。
    language:
        指定源语言代码（如 ``"zh"``、``"en"``），``None`` 为自动检测。
    task:
        任务类型：``"transcribe"``（转写，默认）或 ``"translate"``
        （翻译为英文）。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> asr = ASR(model="openai/whisper-large-v3")
    >>> result = asr(MosaicData(audio="speech.wav"))
    >>> print(result["text"])

    自动检测语言并翻译为英文：
    >>> asr = ASR(task="translate")
    >>> result = asr(MosaicData(audio=audio_data))
    """

    name: str = "asr"
    description: str = (
        "Convert speech to text using OpenAI Whisper. "
        "Supports multi-language, auto-detection, translation, and "
        "long-audio chunking."
    )
    version: str = "0.1.0"
    input_types = ["audio", "mosaic"]
    output_types = ["text"]

    def __init__(
        self,
        model: str = "openai/whisper-large-v3",
        language: str | None = None,
        task: str = "transcribe",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._language: str | None = language
        self._task: str = task

    def _load_model(self) -> None:
        """加载 Whisper 模型。"""
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            AutoModelForSpeechSeq2Seq,
            pipeline,
        )

        device = self._resolve_device()
        try:
            resolved_dtype = torch.float16 if "cuda" in device else torch.float32
        except (AttributeError, RuntimeError):
            resolved_dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self._model_name)

        # 优先使用 dtype=（新版 transformers），回退 torch_dtype=（旧版兼容）
        try:
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_name,
                dtype=resolved_dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_name,
                torch_dtype=resolved_dtype,
                low_cpu_mem_usage=True,
            )
        self._model.to(device)

        # 构造 pipeline，支持长音频分片
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            device=device,
            torch_dtype=resolved_dtype,
        )

        self._logger.info(
            "Whisper model loaded (model=%s, device=%s).",
            self._model_name,
            device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行语音识别。

        Parameters
        ----------
        input_data:
            必须包含 ``audio`` (AudioData | str | numpy.ndarray)；
            可选 ``language`` (str)、``task`` (str)。

        Returns
        -------
        MosaicData
            包含 ``text`` (str)、``language`` (str)、``segments`` (list[dict])、
            ``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``audio`` 输入。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            audio_input = input_data.get("audio")
            if audio_input is None:
                raise ValueError(
                    "ASR requires 'audio' (AudioData, str path, or ndarray)."
                )

            language = input_data.get("language", self._language)
            task = input_data.get("task", self._task)

            # 加载音频
            waveform, sample_rate = self._load_audio(audio_input)

            # 转为单声道
            waveform = self._to_mono(waveform)

            # 计算时长
            duration = self._get_duration(waveform, sample_rate)

            # 构造 pipeline 参数
            pipe_kwargs: dict[str, Any] = {
                "return_timestamps": True,
            }

            # 长音频分片策略
            if duration > 30.0:
                pipe_kwargs["chunk_length_s"] = 30
                pipe_kwargs["stride_length_s"] = 5
                self._logger.info(
                    "Long audio (%.1fs), using chunking strategy.",
                    duration,
                )

            if language is not None:
                pipe_kwargs["language"] = language

            if task in ("transcribe", "translate"):
                pipe_kwargs["task"] = task

            # 执行识别
            import numpy as np  # type: ignore

            # pipeline 接受 numpy 数组
            audio_array = waveform if isinstance(waveform, np.ndarray) else None
            if audio_array is None:
                raise TypeError(
                    f"Expected numpy.ndarray waveform, got {type(waveform).__name__}."
                )

            result = self._pipeline(audio_array, **pipe_kwargs)
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 提取结果
        text = result.get("text", "").strip()
        detected_language = language or result.get("language", "unknown")

        # 提取分段信息
        segments: list[dict[str, Any]] = []
        chunks = result.get("chunks", [])
        if chunks:
            for chunk in chunks:
                seg: dict[str, Any] = {
                    "start": float(chunk.get("timestamp", [0, 0])[0] or 0),
                    "end": float(chunk.get("timestamp", [0, 0])[1] or 0),
                    "text": chunk.get("text", "").strip(),
                }
                segments.append(seg)
        else:
            # 无分段信息时，整体作为一个段
            segments = [
                {
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                }
            ]

        output = MosaicData(
            text=text,
            language=detected_language,
            segments=segments,
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "text_length": len(text),
                "language": detected_language,
                "num_segments": len(segments),
                "audio_duration": duration,
            },
        )
        return output
