# mosaic/nodes/audio/voice_clone.py
"""VoiceClone 节点 —— 语音克隆。

使用参考音频的音色，朗读目标文本。基于 Coqui XTTS-v2 模型，
该模型原生支持语音克隆功能，只需提供一段参考音频即可复刻音色。

设计要点
--------
* 与 TTS 节点共用 XTTS-v2 模型加载逻辑，但强制使用语音克隆模式。
* 参考音频建议 6-30 秒，太短效果差，太长浪费资源。
* 参考音频需要是清晰的人声，避免背景噪音。
* 如果输入是文件路径，自动加载为 AudioData。
* 与 TTS 节点的区别：TTS 使用预设说话人，VoiceClone 使用自定义音色。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["VoiceClone"]


@registry.register
class VoiceClone(BaseAudioNode):
    """语音克隆节点。

    使用参考音频的音色朗读目标文本，基于 XTTS-v2。

    Parameters
    ----------
    model:
        模型标识，默认 ``"coqui/XTTS-v2"``。
    language:
        语音语言代码，如 ``"zh"``、``"en"``，默认 ``"zh"``。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> cloner = VoiceClone()
    >>> result = cloner(MosaicData(
    ...     reference_audio="speaker.wav",
    ...     text="你好，这是我的克隆声音。",
    ...     language="zh",
    ... ))
    >>> audio = result["audio"]  # AudioData

    使用 AudioData 作为参考：
    >>> result = cloner(MosaicData(
    ...     reference_audio=ref_audio_data,
    ...     text="Hello, this is my cloned voice.",
    ...     language="en",
    ... ))
    """

    name: str = "voice-clone"
    description: str = (
        "Clone a voice from reference audio and synthesize speech. "
        "Based on XTTS-v2, supports multi-language voice cloning."
    )
    version: str = "0.1.0"
    input_types = ["audio", "text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "coqui/XTTS-v2",
        language: str = "zh",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._language: str = language

    def _load_model(self) -> None:
        """加载 XTTS-v2 模型用于语音克隆。"""
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore

            self._model = CoquiTTS(self._model_name)
            self._logger.info(
                "XTTS-v2 model loaded for voice cloning (device=%s).",
                self._resolve_device(),
            )
        except ImportError:
            raise ImportError(
                "VoiceClone requires the 'TTS' library (Coqui TTS). "
                "Install via `pip install TTS`."
            )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行语音克隆。

        Parameters
        ----------
        input_data:
            必须包含 ``reference_audio`` (AudioData | str | ndarray) 和
            ``text`` (str)；可选 ``language`` (str, 默认使用构造函数设置)。

        Returns
        -------
        MosaicData
            包含 ``audio`` (AudioData)、``reference_audio`` (AudioData)、
            ``text`` (str)。

        Raises
        ------
        ValueError
            缺少 ``reference_audio`` 或 ``text``。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            reference_input = input_data.get("reference_audio")
            if reference_input is None:
                raise ValueError(
                    "VoiceClone requires 'reference_audio' "
                    "(AudioData, str path, or ndarray)."
                )

            text = input_data.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"VoiceClone requires 'text' (non-empty str), "
                    f"got {type(text).__name__}."
                )

            language = input_data.get("language", self._language)

            # 加载参考音频
            ref_waveform, ref_sr = self._load_audio(reference_input)
            ref_duration = self._get_duration(ref_waveform, ref_sr)

            # 检查参考音频时长
            if ref_duration < 3.0:
                self._logger.warning(
                    "Reference audio is very short (%.1fs). "
                    "Recommend 6-30s for best cloning quality.",
                    ref_duration,
                )
            elif ref_duration > 30.0:
                self._logger.warning(
                    "Reference audio is long (%.1fs). "
                    "Recommend 6-30s to save resources.",
                    ref_duration,
                )

            # 将参考音频保存为临时 wav 文件（XTTS 需要）
            import tempfile
            import os
            import numpy as np  # type: ignore

            ref_path = tempfile.mktemp(suffix=".wav")
            self._save_audio(ref_waveform, ref_sr, ref_path)

            try:
                # 使用 XTTS 进行语音克隆
                result = self._model.tts(
                    text=text,
                    speaker_wav=ref_path,
                    language=language,
                )

                # 转为 numpy 数组
                waveform = np.array(result, dtype="float32")
                sample_rate = 24000  # XTTS-v2 默认输出采样率

            finally:
                # 清理临时文件
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装输出音频
        audio = self._ensure_audio_data(
            waveform, sample_rate, language=language
        )

        # 包装参考音频为 AudioData（方便对比）
        ref_audio = AudioData(
            waveform=ref_waveform,
            sample_rate=ref_sr,
            metadata={"duration": ref_duration, "format": "wav"},
        )

        result = MosaicData(
            audio=audio,
            reference_audio=ref_audio,
            text=text,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "duration": audio.metadata.get("duration", 0.0),
                "language": language,
                "reference_duration": ref_duration,
            },
        )
        return result
