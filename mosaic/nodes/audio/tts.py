# mosaic/nodes/audio/tts.py
"""TTS 节点 —— 文本转语音。

将文本转换为语音，支持多语言和语音克隆。主力方案为 Coqui XTTS-v2，
并提供 edge-tts 轻量备选方案（无需 GPU，适合快速测试）。

设计要点
--------
* 三种加载模式：
    - 模式 A (XTTS-v2)：使用 ``TTS`` 库加载 ``coqui/XTTS-v2``，支持
      多语言与语音克隆。
    - 模式 B (transformers)：使用 ``transformers.pipeline("text-to-speech")``
      加载如 ``facebook/mms-tts`` 系列。
    - 模式 C (edge-tts)：使用 ``edge-tts`` 库，无需 GPU，适合快速测试。
* 长文本自动分句处理，避免一次生成过长音频。
* 输出统一为 :class:`~mosaic.core.types.AudioData` 格式。
"""

from __future__ import annotations

import re
import time
from typing import Any, List

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["TTS"]


def _split_sentences(text: str, max_length: int = 200) -> List[str]:
    """将长文本按句子分割，避免一次生成过长音频。

    Parameters
    ----------
    text:
        待分割的文本。
    max_length:
        每段最大字符数，超出时在逗号/空格处进一步切分。

    Returns
    -------
    List[str]
        分割后的句子列表。
    """
    # 按中英文标点分句
    sentences = re.split(r"[。！？.!?；;\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 过长的句子进一步按逗号/空格切分
    result: List[str] = []
    for sent in sentences:
        if len(sent) <= max_length:
            result.append(sent)
            continue
        # 按逗号/空格切分
        parts = re.split(r"[，,、\s]+", sent)
        current = ""
        for part in parts:
            if len(current) + len(part) + 1 > max_length:
                if current:
                    result.append(current)
                current = part
            else:
                current = f"{current} {part}".strip() if current else part
        if current:
            result.append(current)

    return result if result else [text]


@registry.register
class TTS(BaseAudioNode):
    """文本转语音节点。

    将文本转换为语音，支持多语言和语音克隆。

    Parameters
    ----------
    model:
        模型标识，默认 ``"coqui/XTTS-v2"``。
    speaker:
        预设说话人名称（XTTS-v2 模式下使用），``None`` 使用默认说话人。
    language:
        语音语言代码，如 ``"zh"``、``"en"``，默认 ``"zh"``。
    use_edge_tts:
        是否强制使用 edge-tts（轻量、无需 GPU），默认 ``False``。
        设为 ``True`` 时忽略 ``model`` 参数。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> tts = TTS(language="zh")
    >>> result = tts(MosaicData(text="你好，世界！"))
    >>> result["audio"].waveform  # numpy.ndarray

    使用 edge-tts 轻量模式：
    >>> tts = TTS(use_edge_tts=True, language="zh")
    >>> result = tts(MosaicData(text="你好"))

    语音克隆模式：
    >>> tts = TTS()
    >>> result = tts(MosaicData(
    ...     text="你好",
    ...     speaker_wav="reference.wav",
    ... ))
    """

    name: str = "tts"
    description: str = (
        "Convert text to speech using XTTS-v2 or edge-tts. "
        "Supports multi-language, voice cloning, and speed control."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "coqui/XTTS-v2",
        speaker: str = None,
        language: str = "zh",
        use_edge_tts: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._speaker: str = speaker
        self._language: str = language
        self._use_edge_tts: bool = use_edge_tts
        self._backend: str = "edge_tts" if use_edge_tts else "xtts"

    def _load_model(self) -> None:
        """加载 TTS 模型。"""
        if self._use_edge_tts:
            self._backend = "edge_tts"
            self._logger.info("Using edge-tts backend (no GPU required).")
            return

        # 尝试使用 TTS 库加载 XTTS-v2
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore

            self._model = CoquiTTS(self._model_name)
            self._backend = "xtts"
            self._logger.info(
                "XTTS-v2 model loaded (device=%s).", self._resolve_device()
            )
            return
        except ImportError:
            self._logger.warning(
                "TTS library not available, trying transformers backend."
            )

        # 尝试使用 transformers pipeline
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            device = self._resolve_device()
            self._model = pipeline(
                "text-to-speech",
                model=self._model_name,
                device=device,
            )
            self._backend = "transformers"
            self._logger.info(
                "Transformers TTS pipeline loaded (model=%s, device=%s).",
                self._model_name,
                device,
            )
            return
        except ImportError:
            self._logger.warning(
                "transformers not available, falling back to edge-tts."
            )

        # 回退到 edge-tts
        self._backend = "edge_tts"
        self._logger.info("Falling back to edge-tts backend.")

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本转语音。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str)；可选 ``speaker_wav`` (str|ndarray)、
            ``language`` (str)、``speed`` (float, 默认 1.0)。

        Returns
        -------
        MosaicData
            包含 ``audio`` (AudioData)、``text`` (str)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``text`` 或 ``text`` 非字符串。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            text = input_data.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"TTS requires 'text' (non-empty str), "
                    f"got {type(text).__name__}."
                )

            language = input_data.get("language", self._language)
            speed = float(input_data.get("speed", 1.0))
            speaker_wav = input_data.get("speaker_wav")

            # 分句处理
            sentences = _split_sentences(text)
            self._logger.info(
                "TTS generating %d sentence(s) via %s backend.",
                len(sentences),
                self._backend,
            )

            # 根据后端生成语音
            if self._backend == "edge_tts":
                waveform, sr = self._generate_edge_tts(
                    sentences, language, speed
                )
            elif self._backend == "xtts":
                waveform, sr = self._generate_xtts(
                    sentences, language, speaker_wav, speed
                )
            else:  # transformers
                waveform, sr = self._generate_transformers(
                    sentences
                )
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 AudioData
        audio = self._ensure_audio_data(
            waveform, sr, backend=self._backend, language=language
        )
        duration = audio.metadata.get("duration", 0.0)

        result = MosaicData(
            audio=audio,
            text=text,
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "backend": self._backend,
                "duration": duration,
                "language": language,
            },
        )
        return result

    def _generate_edge_tts(
        self, sentences: List[str], language: str, speed: float
    ) -> tuple:
        """使用 edge-tts 生成语音。"""
        import asyncio
        import numpy as np  # type: ignore
        import edge_tts  # type: ignore

        # edge-tts 语言映射
        voice_map = {
            "zh": "zh-CN-XiaoxiaoNeural",
            "en": "en-US-JennyNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "es": "es-ES-ElviraNeural",
        }
        voice = voice_map.get(language, voice_map["en"])

        async def _synth(text: str) -> bytes:
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        # 逐句生成并合并
        all_audio: List[bytes] = []
        for sent in sentences:
            audio_bytes = asyncio.run(_synth(sent))
            all_audio.append(audio_bytes)

        # 将 mp3 字节流转为 numpy 波形
        import io

        try:
            import soundfile as sf  # type: ignore

            # soundfile 可以读取 mp3（需要 libsndfile >= 1.1.0）
            combined_bytes = b"".join(all_audio)
            waveform, sr = sf.read(io.BytesIO(combined_bytes), dtype="float32")
            return waveform, sr
        except Exception:
            # 回退：使用临时文件
            import tempfile
            import os

            tmp_files = []
            waveforms = []
            sr = 24000
            for i, audio_bytes in enumerate(all_audio):
                tmp_path = tempfile.mktemp(suffix=".mp3")
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                tmp_files.append(tmp_path)

                try:
                    import librosa  # type: ignore

                    wf, sr = librosa.load(tmp_path, sr=None)
                    waveforms.append(wf)
                except ImportError:
                    pass

            for tmp in tmp_files:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

            if waveforms:
                waveform = np.concatenate(waveforms)
                return waveform, sr

            raise ImportError(
                "edge-tts requires 'soundfile' or 'librosa' to decode audio. "
                "Install via `pip install soundfile` or `pip install librosa`."
            )

    def _generate_xtts(
        self,
        sentences: List[str],
        language: str,
        speaker_wav: Any,
        speed: float,
    ) -> tuple:
        """使用 XTTS-v2 生成语音。"""
        import numpy as np  # type: ignore

        device = self._resolve_device()
        waveforms: List[Any] = []
        sr = 24000  # XTTS-v2 默认输出采样率

        for sent in sentences:
            kwargs: dict = {
                "text": sent,
                "language": language,
                "speed": speed,
            }

            if speaker_wav is not None:
                # 语音克隆模式
                if isinstance(speaker_wav, str):
                    kwargs["speaker_wav"] = speaker_wav
                else:
                    # numpy 数组：保存为临时文件
                    import tempfile
                    import os

                    wf, sr_in = self._load_audio(speaker_wav)
                    tmp_path = tempfile.mktemp(suffix=".wav")
                    self._save_audio(wf, sr_in, tmp_path)
                    kwargs["speaker_wav"] = tmp_path

                result = self._model.tts(**kwargs)
                if isinstance(result, list):
                    waveforms.append(np.array(result, dtype="float32"))
                else:
                    waveforms.append(np.array(result, dtype="float32"))

                # 清理临时文件
                if not isinstance(speaker_wav, str):
                    try:
                        os.unlink(kwargs["speaker_wav"])
                    except (OSError, KeyError):
                        pass
            else:
                # 预设说话人模式
                if self._speaker:
                    kwargs["speaker"] = self._speaker
                result = self._model.tts(**kwargs)
                if isinstance(result, list):
                    waveforms.append(np.array(result, dtype="float32"))
                else:
                    waveforms.append(np.array(result, dtype="float32"))

        waveform = np.concatenate(waveforms) if waveforms else np.array([])
        return waveform, sr

    def _generate_transformers(self, sentences: List[str]) -> tuple:
        """使用 transformers pipeline 生成语音。"""
        import numpy as np  # type: ignore

        waveforms: List[Any] = []
        sr = 16000  # 默认采样率

        for sent in sentences:
            result = self._model(sent)
            waveform = np.array(result["audio"], dtype="float32")
            sr = result.get("sampling_rate", sr)
            waveforms.append(waveform)

        if waveforms:
            # 确保所有波形维度一致
            max_len = max(w.shape[-1] for w in waveforms)
            padded = []
            for w in waveforms:
                if w.shape[-1] < max_len:
                    pad_width = max_len - w.shape[-1]
                    if w.ndim == 1:
                        w = np.pad(w, (0, pad_width))
                    else:
                        w = np.pad(w, ((0, 0), (0, pad_width)))
                padded.append(w)
            waveform = np.concatenate(padded, axis=-1)
        else:
            waveform = np.array([])

        return waveform, sr
