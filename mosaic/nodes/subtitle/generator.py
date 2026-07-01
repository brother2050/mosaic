# mosaic/nodes/subtitle/generator.py
"""SubtitleGenerator 节点 —— 字幕生成。

从音频或视频文件生成带时间轴的字幕。内部复用音频域 ASR (Whisper) 的
模型加载与推理逻辑，并将识别结果转为标准字幕格式（SRT/WebVTT）。

设计要点
--------
* 复用 :class:`~mosaic.nodes.audio.asr.ASR` 节点进行语音识别。
* Whisper 的 ``return_timestamps=True`` 输出直接对应字幕片段。
* ``word_timestamps=True`` 时使用词级时间戳，精度更高。
* 自动处理超过 ``max_chars_per_line`` 的长句（按标点或字数拆分）。
* 支持从视频文件提取音频（使用 ffmpeg subprocess）。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData, SubtitleData

from mosaic.nodes._coerce import safe_int
from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.nodes.subtitle._base import BaseSubtitleNode

__all__ = ["SubtitleGenerator"]


@registry.register
class SubtitleGenerator(BaseSubtitleNode):
    """字幕生成节点。

    从音频或视频生成带时间轴的字幕，基于 Whisper ASR。

    Parameters
    ----------
    asr_model:
        Whisper 模型标识，默认 ``"openai/whisper-large-v3"``。
    output_format:
        输出字幕格式，``"srt"`` (默认) 或 ``"vtt"``。
    language:
        音频语言代码，``None`` 为自动检测。
    **kwargs:
        透传给 :class:`BaseSubtitleNode` 的参数。

    Examples
    --------
    >>> gen = SubtitleGenerator(output_format="srt")
    >>> result = gen(MosaicData(audio="speech.wav"))
    >>> print(result["subtitle"].segments[0])
    {'start': 0.0, 'end': 2.5, 'text': '你好世界'}
    """

    name: str = "subtitle-generator"
    description: str = (
        "Generate timed subtitles from audio or video using Whisper ASR. "
        "Supports SRT/VTT output, word-level timestamps, and long-sentence splitting."
    )
    version: str = "0.1.0"
    input_types = ["audio", "mosaic"]
    output_types = ["subtitle"]

    def __init__(
        self,
        asr_model: str = "openai/whisper-large-v3",
        output_format: str = "srt",
        language: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._asr_model_name: str = asr_model
        self._output_format: str = output_format
        self._language: str | None = language
        # 内部 ASR 节点（延迟创建）
        self._asr_node: Any | None = None

    def _load_model(self) -> None:
        """加载 Whisper ASR 模型（复用音频域 ASR 节点）。"""
        from mosaic.nodes.audio.asr import ASR

        self._asr_node = ASR(
            model=self._asr_model_name,
            language=self._language,
            scheduler=self._scheduler,
            bus=self._bus,
        )
        self._asr_node.load()
        self._model = self._asr_node._model
        self._logger.info(
            "SubtitleGenerator: ASR model loaded (model=%s).",
            self._asr_model_name,
        )

    def unload(self) -> None:
        """释放 ASR 模型。"""
        if self._asr_node is not None:
            self._asr_node.unload()
            self._asr_node = None
        self._model = None
        self._loaded = False

    @staticmethod
    def _extract_audio_from_video(video_path: str) -> str:
        """从视频文件提取音频为临时 wav 文件。

        Parameters
        ----------
        video_path:
            视频文件路径。

        Returns
        -------
        str
            临时 wav 文件路径。
        """
        tmp_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", tmp_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise ImportError(
                "FFmpeg is required to extract audio from video. "
                "Install it via `apt install ffmpeg` or `brew install ffmpeg`."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFmpeg failed to extract audio: {exc.stderr.decode('utf-8', errors='replace')}"
            ) from exc
        return tmp_path

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行字幕生成。

        Parameters
        ----------
        input_data:
            必须包含 ``audio`` (AudioData | str)；可选 ``video`` (str)、
            ``language`` (str)、``word_timestamps`` (bool, 默认 False)、
            ``max_chars_per_line`` (int, 默认 42)。

        Returns
        -------
        MosaicData
            包含 ``subtitle`` (SubtitleData)、``text`` (str)、
            ``language`` (str)、``segments_count`` (int)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``audio`` 或 ``video`` 输入。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        tmp_audio_path: str | None = None

        try:
            audio_input = input_data.get("audio")
            video_path = input_data.get("video")

            # 如果有视频但没有音频，从视频提取
            if audio_input is None and video_path is not None:
                if not isinstance(video_path, str):
                    raise ValueError("'video' must be a file path (str).")
                self._logger.info("Extracting audio from video: %s", video_path)
                tmp_audio_path = self._extract_audio_from_video(video_path)
                audio_input = tmp_audio_path

            if audio_input is None:
                raise ValueError(
                    "SubtitleGenerator requires 'audio' (AudioData or str) "
                    "or 'video' (str path)."
                )

            language = input_data.get("language", self._language)
            word_timestamps = bool(input_data.get("word_timestamps", False))
            max_chars = safe_int(
                input_data.get("max_chars_per_line"), "max_chars_per_line", default=42
            )

            # 使用 ASR 节点进行识别
            asr_input = MosaicData(audio=audio_input)
            if language is not None:
                asr_input["language"] = language

            # 构造 ASR pipeline 参数
            if self._asr_node is not None and self._asr_node._pipeline is not None:
                pipe_kwargs: dict[str, Any] = {"return_timestamps": True}

                # 长音频分片
                waveform, sr = BaseAudioNode._load_audio(audio_input)
                waveform = BaseAudioNode._to_mono(waveform)
                duration = BaseAudioNode._get_duration(waveform, sr)

                if duration > 30.0:
                    pipe_kwargs["chunk_length_s"] = 30
                    pipe_kwargs["stride_length_s"] = 5

                if language is not None:
                    pipe_kwargs["language"] = language

                if word_timestamps:
                    pipe_kwargs["return_timestamps"] = "word"

                import numpy as np  # type: ignore
                audio_array = waveform if isinstance(waveform, np.ndarray) else None
                if audio_array is None:
                    raise TypeError("Expected numpy.ndarray waveform.")

                asr_result = self._asr_node._pipeline(audio_array, **pipe_kwargs)
            else:
                # 回退：直接调用 ASR 节点的 run
                asr_result_data = self._asr_node.run(asr_input)
                asr_result = {
                    "text": asr_result_data.get("text", ""),
                    "chunks": [],
                }
                # 从 ASR 输出的 segments 构造 chunks
                for seg in asr_result_data.get("segments", []):
                    asr_result.setdefault("chunks", []).append({
                        "timestamp": [seg["start"], seg["end"]],
                        "text": seg["text"],
                    })
                waveform, sr = BaseAudioNode._load_audio(audio_input)
                duration = BaseAudioNode._get_duration(
                    BaseAudioNode._to_mono(waveform), sr
                )

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise
        finally:
            # 清理临时音频文件
            if tmp_audio_path is not None and os.path.exists(tmp_audio_path):
                try:
                    os.unlink(tmp_audio_path)
                except OSError:
                    pass

        elapsed = time.perf_counter() - t0

        # 提取文本与分段
        full_text = asr_result.get("text", "").strip()
        detected_language = language or asr_result.get("language", "unknown")

        # 构造字幕片段
        segments: list[dict[str, Any]] = []
        chunks = asr_result.get("chunks", [])

        if word_timestamps and chunks:
            # 词级时间戳：将词合并为行
            current_words: list[str] = []
            current_start: float | None = None
            current_end: float = 0.0

            for chunk in chunks:
                ts = chunk.get("timestamp", [None, None])
                word = chunk.get("text", "").strip()
                if not word:
                    continue

                w_start = float(ts[0]) if ts[0] is not None else current_end
                w_end = float(ts[1]) if ts[1] is not None else w_start + 0.5

                if current_start is None:
                    current_start = w_start

                current_words.append(word)
                current_end = w_end

                # 达到最大字符数时断行
                current_text = " ".join(current_words)
                if len(current_text) >= max_chars:
                    segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": current_text,
                    })
                    current_words = []
                    current_start = None

            # 剩余的词
            if current_words:
                segments.append({
                    "start": current_start or 0.0,
                    "end": current_end,
                    "text": " ".join(current_words),
                })
        elif chunks:
            # 句级时间戳
            for chunk in chunks:
                ts = chunk.get("timestamp", [0, 0])
                start = float(ts[0]) if ts[0] is not None else 0.0
                end = float(ts[1]) if ts[1] is not None else start + 1.0
                text = chunk.get("text", "").strip()
                if text:
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text,
                    })
        else:
            # 无分段信息
            segments = [{
                "start": 0.0,
                "end": duration,
                "text": full_text,
            }]

        # 后处理：合并过短片段 + 拆分过长片段
        segments = self._merge_short_segments(segments, min_duration=0.3)
        segments = self._split_long_segments(
            segments, max_duration=10.0, max_chars=max_chars
        )

        # 重新编号
        for i, seg in enumerate(segments, 1):
            seg["index"] = i

        # 构造 SubtitleData
        subtitle = self._make_subtitle_data(
            segments=segments,
            fmt=self._output_format,
            language=detected_language,
            source="whisper",
        )

        result = MosaicData(
            subtitle=subtitle,
            text=full_text,
            language=detected_language,
            segments_count=len(segments),
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "segments": len(segments),
                "language": detected_language,
                "format": self._output_format,
                "audio_duration": duration,
            },
        )
        return result
