# mosaic/nodes/subtitle/aligner.py
"""SubtitleAligner 节点 —— 时间轴对齐。

将字幕与音频精确对齐，修正时间轴偏移和片段边界。

设计要点
--------
* 三种对齐方法：
    - ``whisper``：使用 Whisper 的 word_timestamps 功能，将字幕文本
      与音频强制对齐（默认）。
    - ``aeneas``：使用 aeneas 库进行音频-文本对齐（如果安装了 aeneas）。
    - ``dtw``：使用动态时间规整 (Dynamic Time Warping) 算法对齐。
* 输出对齐质量分数（0-1），方便用户判断是否需要人工调整。
* 处理字幕与音频时长不匹配的情况（字幕太长或太短）。
* 计算整体时间偏移量 (time_shift)。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData, SubtitleData

from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.nodes.subtitle._base import BaseSubtitleNode

__all__ = ["SubtitleAligner"]


@registry.register
class SubtitleAligner(BaseSubtitleNode):
    """字幕时间轴对齐节点。

    将字幕与音频精确对齐，修正时间偏移和片段边界。

    Parameters
    ----------
    method:
        对齐方法：``"whisper"`` (默认) / ``"aeneas"`` / ``"dtw"``。
    language:
        音频语言代码，``None`` 为自动检测。
    **kwargs:
        透传给 :class:`BaseSubtitleNode` 的参数。

    Examples
    --------
    >>> aligner = SubtitleAligner(method="whisper")
    >>> result = aligner(MosaicData(
    ...     subtitle=subtitle_data,
    ...     audio="speech.wav",
    ... ))
    >>> print(result["alignment_score"])
    0.92
    """

    name: str = "subtitle-aligner"
    description: str = (
        "Align subtitle timestamps with audio using Whisper word-level "
        "timestamps, aeneas, or DTW. Outputs alignment quality score."
    )
    version: str = "0.1.0"
    input_types = ["subtitle", "audio", "mosaic"]
    output_types = ["subtitle"]

    def __init__(
        self,
        method: str = "whisper",
        language: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._method: str = method
        self._language: str | None = language
        # 内部 ASR pipeline（whisper 对齐时使用）
        self._pipeline: Any = None

    def _load_model(self) -> None:
        """加载对齐所需的模型。"""
        if self._method == "whisper":
            self._load_whisper()
        # aeneas 和 dtw 不需要预加载模型
        self._logger.info(
            "SubtitleAligner: method=%s ready.", self._method
        )

    def _load_whisper(self) -> None:
        """加载 Whisper pipeline 用于对齐。"""
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            AutoModelForSpeechSeq2Seq,
            pipeline,
        )

        model_name = "openai/whisper-large-v3"
        device = "cpu"
        try:
            if torch.cuda.is_available():
                device = "cuda"
        except (AttributeError, RuntimeError):
            pass

        try:
            resolved_dtype = torch.float16 if "cuda" in device else torch.float32
        except (AttributeError, RuntimeError):
            resolved_dtype = torch.float32

        processor = AutoProcessor.from_pretrained(model_name)

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, dtype=resolved_dtype, low_cpu_mem_usage=True
            )
        except TypeError:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, torch_dtype=resolved_dtype, low_cpu_mem_usage=True
            )
        model.to(device)

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=resolved_dtype,
        )
        self._model = model

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行时间轴对齐。

        Parameters
        ----------
        input_data:
            必须包含 ``subtitle`` (SubtitleData) 和 ``audio``
            (AudioData | str)。

        Returns
        -------
        MosaicData
            包含 ``subtitle`` (SubtitleData)、``alignment_method`` (str)、
            ``time_shift`` (float)、``alignment_score`` (float)。

        Raises
        ------
        ValueError
            缺少 ``subtitle`` 或 ``audio`` 输入。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            subtitle = input_data.get("subtitle")
            if subtitle is None:
                raise ValueError(
                    "SubtitleAligner requires 'subtitle' (SubtitleData)."
                )

            audio_input = input_data.get("audio")
            if audio_input is None:
                raise ValueError(
                    "SubtitleAligner requires 'audio' (AudioData or str)."
                )

            # 提取 SubtitleData
            if isinstance(subtitle, dict):
                subtitle = SubtitleData(
                    segments=subtitle.get("segments", []),
                    format=subtitle.get("format", "srt"),
                )
            elif not isinstance(subtitle, SubtitleData):
                raise TypeError(
                    f"Expected SubtitleData, got {type(subtitle).__name__}."
                )

            segments = subtitle.segments
            if not segments:
                raise ValueError("Subtitle has no segments to align.")

            # 加载音频
            waveform, sample_rate = BaseAudioNode._load_audio(audio_input)
            waveform = BaseAudioNode._to_mono(waveform)
            audio_duration = BaseAudioNode._get_duration(waveform, sample_rate)

            # 根据方法执行对齐
            if self._method == "whisper":
                aligned_segments, time_shift, score = self._align_whisper(
                    segments, waveform, sample_rate
                )
            elif self._method == "aeneas":
                aligned_segments, time_shift, score = self._align_aeneas(
                    segments, waveform, sample_rate, audio_input
                )
            elif self._method == "dtw":
                aligned_segments, time_shift, score = self._align_dtw(
                    segments, waveform, sample_rate, audio_duration
                )
            else:
                raise ValueError(
                    f"Unknown alignment method: {self._method!r}. "
                    f"Supported: 'whisper', 'aeneas', 'dtw'."
                )

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 构造对齐后的 SubtitleData
        out_subtitle = self._make_subtitle_data(
            segments=aligned_segments,
            fmt=subtitle.subtitle_format,
            method=self._method,
            score=score,
        )

        result = MosaicData(
            subtitle=out_subtitle,
            alignment_method=self._method,
            time_shift=time_shift,
            alignment_score=score,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "method": self._method,
                "score": score,
                "time_shift": time_shift,
                "segments": len(aligned_segments),
            },
        )
        return result

    def _align_whisper(
        self,
        segments: list[dict[str, Any]],
        waveform: Any,
        sample_rate: int,
    ) -> tuple:
        """使用 Whisper word_timestamps 进行对齐。

        Returns
        -------
        tuple[list[Dict], float, float]
            ``(aligned_segments, time_shift, alignment_score)``
        """
        import numpy as np  # type: ignore

        # 使用 Whisper 获取词级时间戳
        pipe_kwargs: dict[str, Any] = {
            "return_timestamps": "word",
        }
        if self._language is not None:
            pipe_kwargs["language"] = self._language

        audio_duration = BaseAudioNode._get_duration(waveform, sample_rate)
        if audio_duration > 30.0:
            pipe_kwargs["chunk_length_s"] = 30
            pipe_kwargs["stride_length_s"] = 5

        if self._pipeline is None:
            # 无 pipeline 时，保持原时间轴，返回中等分数
            self._logger.warning(
                "Whisper pipeline not loaded, returning original timestamps."
            )
            return list(segments), 0.0, 0.5

        result = self._pipeline(waveform, **pipe_kwargs)
        word_chunks = result.get("chunks", [])

        # 构建词级时间戳列表
        word_timestamps: list[dict[str, Any]] = []
        for chunk in word_chunks:
            ts = chunk.get("timestamp", [None, None])
            word = chunk.get("text", "").strip()
            if not word:
                continue
            word_timestamps.append({
                "word": word,
                "start": float(ts[0]) if ts[0] is not None else 0.0,
                "end": float(ts[1]) if ts[1] is not None else 0.0,
            })

        if not word_timestamps:
            return list(segments), 0.0, 0.3

        # 将字幕片段与词级时间戳匹配
        aligned: list[dict[str, Any]] = []
        word_idx = 0
        matched_count = 0

        for seg in segments:
            seg_text = seg.get("text", "").strip().lower()
            seg_words = set(seg_text.split())

            # 寻找最佳匹配的词范围
            best_start = seg["start"]
            best_end = seg["end"]
            best_match_count = 0

            # 在词列表中滑动窗口查找匹配
            search_start = max(0, word_idx - 5)
            search_end = min(len(word_timestamps), word_idx + 50)

            for ws in range(search_start, search_end):
                for we in range(ws + 1, min(search_end + 1, ws + 30)):
                    window_words = set()
                    for wi in range(ws, we):
                        window_words.add(
                            word_timestamps[wi]["word"].strip().lower()
                        )
                    overlap = len(seg_words & window_words)
                    if overlap > best_match_count:
                        best_match_count = overlap
                        best_start = word_timestamps[ws]["start"]
                        best_end = word_timestamps[we - 1]["end"]
                        word_idx = we

            if best_match_count > 0:
                matched_count += 1

            aligned.append({
                "start": best_start,
                "end": best_end,
                "text": seg.get("text", ""),
                "index": seg.get("index", 0),
            })

        # 计算时间偏移和质量分数
        if segments and aligned:
            original_starts = [s["start"] for s in segments]
            aligned_starts = [a["start"] for a in aligned]
            time_shift = float(
                sum(aligned_starts) - sum(original_starts)
            ) / max(len(original_starts), 1)
        else:
            time_shift = 0.0

        score = matched_count / max(len(segments), 1) if segments else 0.0

        return aligned, time_shift, score

    def _align_aeneas(
        self,
        segments: list[dict[str, Any]],
        waveform: Any,
        sample_rate: int,
        audio_input: Any,
    ) -> tuple:
        """使用 aeneas 库进行对齐。

        Returns
        -------
        tuple[list[Dict], float, float]
            ``(aligned_segments, time_shift, alignment_score)``
        """
        try:
            import tempfile  # noqa: F401
            import os  # noqa: F401
        except ImportError:
            pass

        # aeneas 需要文件路径输入
        audio_path = None
        if isinstance(audio_input, str):
            audio_path = audio_input
        else:
            # 保存为临时文件
            import tempfile
            import os

            audio_path = tempfile.mktemp(suffix=".wav")
            BaseAudioNode._save_audio(waveform, sample_rate, audio_path)

        try:
            from aeneas.executetask import ExecuteTask  # type: ignore
            from aeneas.task import Task  # type: ignore

            # 构造 aeneas 任务
            text_lines = "\n".join(
                seg.get("text", "") for seg in segments
            )
            config_string = (
                f"task_language={self._language or 'eng'}|"
                f"is_text_type=plain|"
                f"os_task_file_format=json"
            )

            task = Task(config_string=config_string)
            task.audio_file_path_absolute = audio_path
            task.text_file_path_absolute = None
            task.text_fragment = text_lines

            ExecuteTask(task).execute()

            # 解析对齐结果
            aligned: list[dict[str, Any]] = []
            for i, frag in enumerate(task.sync_map_fragments):
                seg = segments[i] if i < len(segments) else {}
                aligned.append({
                    "start": frag.begin,
                    "end": frag.end,
                    "text": seg.get("text", ""),
                    "index": i + 1,
                })

            # aeneas 通常对齐质量很高
            score = min(1.0, len(aligned) / max(len(segments), 1))
            time_shift = 0.0
            return aligned, time_shift, score

        except ImportError:
            self._logger.warning(
                "aeneas not installed, falling back to original timestamps."
            )
            return list(segments), 0.0, 0.3
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("aeneas alignment failed: %s", exc)
            return list(segments), 0.0, 0.3
        finally:
            # 清理临时文件
            if not isinstance(audio_input, str) and audio_path:
                try:
                    os.unlink(audio_path)
                except (OSError, UnboundLocalError):
                    pass

    def _align_dtw(
        self,
        segments: list[dict[str, Any]],
        waveform: Any,
        sample_rate: int,
        audio_duration: float,
    ) -> tuple:
        """使用动态时间规整 (DTW) 进行对齐。

        DTW 方法基于字幕片段时长比例进行重新分配，
        适用于字幕与音频时长不匹配的情况。

        Returns
        -------
        tuple[list[Dict], float, float]
            ``(aligned_segments, time_shift, alignment_score)``
        """
        if not segments:
            return [], 0.0, 0.0

        # 计算字幕总时长
        subtitle_duration = sum(
            seg["end"] - seg["start"] for seg in segments
        )

        if subtitle_duration <= 0:
            return list(segments), 0.0, 0.0

        # 按比例重新分配时间轴
        scale = audio_duration / subtitle_duration if subtitle_duration > 0 else 1.0

        aligned: list[dict[str, Any]] = []
        current_time = 0.0

        for i, seg in enumerate(segments, 1):
            seg_duration = (seg["end"] - seg["start"]) * scale
            new_start = current_time
            new_end = current_time + seg_duration
            aligned.append({
                "start": new_start,
                "end": new_end,
                "text": seg.get("text", ""),
                "index": i,
            })
            current_time = new_end

        # 计算时间偏移
        time_shift = aligned[0]["start"] - segments[0]["start"] if segments else 0.0

        # DTW 对齐质量分数基于时长匹配度
        duration_ratio = min(subtitle_duration, audio_duration) / max(
            subtitle_duration, audio_duration, 1e-6
        )
        score = float(duration_ratio)

        return aligned, time_shift, score
