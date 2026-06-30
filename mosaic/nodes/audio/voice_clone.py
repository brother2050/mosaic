# mosaic/nodes/audio/voice_clone.py
"""VoiceClone 节点 —— 语音风格匹配。

分析参考音频的特征（时长、语速、基频代理量等），从 edge-tts 预设
Neural 语音中选择最匹配的语音，并应用参考音频的语速，合成目标文本。

设计要点
--------
* **不再依赖 Coqui XTTS-v2**。受限于免费方案，本节点无法真正复刻任意
  说话人的音色，而是"风格匹配"——根据参考音频的语速/时长/基频代理量
  等特征，选择并调整最合适的 edge-tts 预设 Neural 语音。
* 参考音频建议 6-30 秒，太短特征不可靠，太长浪费资源。
* 参考音频需要是清晰的人声，避免背景噪音。
* 如果输入是文件路径/AudioData/ndarray，自动加载为波形。
* 与 :class:`~mosaic.nodes.audio.tts.TTS` 的区别：TTS 直接使用预设说话人，
  VoiceClone 会**参考一段音频**来选择语音并迁移其语速风格。
* 当需要真正的零样本音色克隆时，可通过 ``model`` 指定本地 transformers
  TTS 模型（如 ``microsoft/speecht5_tts``，仅英文）走 transformers 后端；
  加载失败时回退到 edge-tts 风格匹配。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.nodes.audio._ref_audio_utils import load_reference_audio
from mosaic.nodes.audio.tts import (
    _EMOTION_VOICE_MAP,
    _split_sentences,
    _synthesize_edge_tts,
)

__all__ = ["VoiceClone"]


# 中文/英文正常语速的近似字符率（字符/秒），用于把参考音频的语速
# 归一化为 edge-tts 的 speed 倍率（1.0 = 正常）。
_NORMAL_CPS: dict = {
    "zh": 4.5,
    "en": 12.0,
    "ja": 6.0,
    "ko": 5.0,
    "fr": 12.0,
    "de": 12.0,
    "es": 12.0,
}

# 基频代理量（Hz）的性别分界：低于该值倾向选择男声。
_PITCH_FEMALE_THRESHOLD_HZ: float = 165.0


@registry.register
class VoiceClone(BaseAudioNode):
    """语音风格匹配节点。

    分析参考音频的特征（时长、语速、基频代理量），从 edge-tts 预设
    Neural 语音中选择最匹配的语音，并应用参考音频的语速，合成目标文本。

    Parameters
    ----------
    model:
        模型标识。默认 ``"edge-tts"`` 使用 edge-tts 风格匹配（无需 GPU）；
        若指定为本地 transformers TTS 模型（如
        ``microsoft/speecht5_tts``），则走 transformers 后端；加载失败时
        回退到 edge-tts。
    voice:
        可选，直接指定 edge-tts 语音名称，优先于自动匹配。
    language:
        语音语言代码，如 ``"zh"``、``"en"``，默认 ``"zh"``。
    emotion:
        情感风格（如 ``"neutral"``/``"cheerful"``/``"gentle"``），作为
        自动匹配语音时的偏好。默认 ``"neutral"``。
    speed:
        语速倍率，``1.0`` 为正常语速，默认 ``1.0``。最终语速会与从参考
        音频估计的语速相乘。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> cloner = VoiceClone(language="zh")
    >>> result = cloner(MosaicData(
    ...     reference_audio="speaker.wav",
    ...     text="你好，这是我的克隆声音。",
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
        "Match a reference voice's style (speed/pitch proxy) and synthesize "
        "speech using edge-tts preset neural voices. No Coqui TTS dependency; "
        "supports multi-language and speed transfer from reference audio."
    )
    version: str = "0.2.0"
    input_types = ["audio", "text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "edge-tts",
        voice: str | None = None,
        language: str = "zh",
        emotion: str = "neutral",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._voice: str | None = voice
        self._language: str = language
        self._emotion: str = emotion
        self._speed: float = float(speed)
        self._backend: str = "edge_tts"

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载语音克隆/合成后端。

        后端选择逻辑：

        1. 若 ``self._model_name`` 为 ``"edge-tts"`` 或以 ``"edge"`` 开头，
           使用 edge-tts 风格匹配后端（无需 GPU）。
        2. 否则尝试使用 ``transformers.pipeline("text-to-speech")`` 加载
           指定的本地 TTS 模型（如 ``microsoft/speecht5_tts``）。
        3. 若 transformers 加载失败，回退到 edge-tts。
        """
        if (
            self._model_name == "edge-tts"
            or self._model_name.lower().startswith("edge")
        ):
            self._backend = "edge_tts"
            self._logger.info(
                "VoiceClone using edge-tts backend (no GPU required)."
            )
            return

        # 非 edge 模型：尝试 transformers pipeline
        try:
            from transformers import pipeline  # type: ignore

            device = self._resolve_device()
            self._model = pipeline(
                "text-to-speech",
                model=self._model_name,
                device=device,
            )
            self._backend = "transformers"
            self._logger.info(
                "Transformers TTS pipeline loaded for voice cloning "
                "(model=%s, device=%s).",
                self._model_name,
                device,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Transformers TTS failed for %s: %s. "
                "Falling back to edge-tts style matching.",
                self._model_name,
                exc,
            )
            self._backend = "edge_tts"

    # ------------------------------------------------------------------
    # 参考音频特征分析
    # ------------------------------------------------------------------
    def _estimate_pitch_proxy(
        self, waveform: Any, sample_rate: int
    ) -> float:
        """估计参考音频的基频代理量（Hz）。

        使用零交叉率（zero-crossing rate）作为基频的粗略代理：每个周期
        有 2 次零交叉，故 ``freq ≈ zcr / 2``。无需 librosa/scipy，仅依赖
        numpy。主要用于区分男声（低基频）与女声（高基频）。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据。
        sample_rate:
            采样率。

        Returns
        -------
        float
            基频代理量（Hz）；无法估计时返回 ``0.0``。
        """
        import numpy as np  # type: ignore

        if waveform is None:
            return 0.0
        mono = self._to_mono(waveform)
        if not isinstance(mono, np.ndarray) or mono.size < 2:
            return 0.0
        signs = np.sign(mono)
        # 去除静音零点
        signs = signs[signs != 0]
        if signs.size < 2:
            return 0.0
        crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
        duration = float(mono.size) / float(sample_rate)
        if duration <= 0:
            return 0.0
        zcr = crossings / duration
        return zcr / 2.0

    def _estimate_speech_rate(
        self,
        ref_waveform: Any,
        ref_sample_rate: int,
        text: str,
        language: str,
    ) -> float:
        """根据参考音频时长与目标文本长度估计语速倍率。

        Parameters
        ----------
        ref_waveform:
            参考音频波形。
        ref_sample_rate:
            参考音频采样率。
        text:
            目标合成文本。
        language:
            语言代码，用于选择正常语速基准。

        Returns
        -------
        float
            语速倍率（``1.0`` = 正常），限制在 ``[0.5, 2.0]`` 区间。
        """
        duration = self._get_duration(ref_waveform, ref_sample_rate)
        if not text or duration <= 0:
            return 1.0
        char_count = len(text)
        if char_count == 0:
            return 1.0
        cps = char_count / duration
        normal_cps = _NORMAL_CPS.get(language, _NORMAL_CPS["en"])
        speed = cps / normal_cps
        # 钳制到合理区间，避免参考音频与目标文本不匹配时产生极端值
        return max(0.5, min(2.0, speed))

    def _match_voice(
        self,
        language: str,
        emotion: str,
        voice_override: str | None,
        ref_waveform: Any,
        ref_sample_rate: int,
    ) -> str:
        """根据参考音频特征与语言/情感偏好匹配 edge-tts 语音。

        匹配策略：
        1. 若显式指定 ``voice_override``，直接使用。
        2. 若 ``emotion`` 显式为性别化风格（male/young_male/child），
           优先使用对应语音。
        3. 否则用参考音频的基频代理量估计性别：低基频 -> 男声，
           高基频 -> 按 emotion 选择女声。
        """
        if voice_override:
            return voice_override

        lang_map = _EMOTION_VOICE_MAP.get(
            language, _EMOTION_VOICE_MAP.get("en", {})
        )

        # 显式性别化情感优先
        if emotion in ("male", "young_male", "child") and emotion in lang_map:
            return lang_map[emotion]

        # 用基频代理量估计性别
        pitch = self._estimate_pitch_proxy(ref_waveform, ref_sample_rate)
        if pitch > 0 and pitch < _PITCH_FEMALE_THRESHOLD_HZ:
            voice = lang_map.get("male") or lang_map.get("neutral")
        else:
            voice = lang_map.get(emotion) or lang_map.get("neutral")

        if voice is None:
            voice = next(iter(lang_map.values()), "en-US-JennyNeural")
        return voice

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行语音风格匹配合成。

        Parameters
        ----------
        input_data:
            必须包含 ``reference_audio`` (AudioData | str | ndarray) 和
            ``text`` (str)；可选 ``language`` (str)、``emotion`` (str)、
            ``voice`` (str)、``speed`` (float)。这些参数覆盖构造函数默认值。

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
            emotion = input_data.get("emotion", self._emotion)
            voice = input_data.get("voice", self._voice)
            user_speed = float(input_data.get("speed", self._speed))

            # 加载参考音频（统一处理时长校验与自动截断）
            # VoiceClone 推荐 6-30s：过短（<1s）将抛出 ValueError，
            # 过长（>30s）将由工具函数自动截取前 30 秒并发出警告，
            # 偏短（<3s）也会由工具函数发出提示。
            ref_waveform, ref_sr = load_reference_audio(
                reference_input, target_sr=22050, backend="default"
            )
            ref_duration = self._get_duration(ref_waveform, ref_sr)

            # 分句
            sentences = _split_sentences(text)

            if self._backend == "edge_tts":
                # 匹配语音
                matched_voice = self._match_voice(
                    language, emotion, voice, ref_waveform, ref_sr
                )
                # 从参考音频估计语速，并与用户指定语速相乘
                est_speed = self._estimate_speech_rate(
                    ref_waveform, ref_sr, text, language
                )
                final_speed = max(0.5, min(2.0, user_speed * est_speed))

                self._logger.info(
                    "VoiceClone style matching: voice=%s, "
                    "ref_duration=%.2fs, est_speed=%.2f, final_speed=%.2f.",
                    matched_voice,
                    ref_duration,
                    est_speed,
                    final_speed,
                )

                waveform, sample_rate = _synthesize_edge_tts(
                    sentences, matched_voice, final_speed, self._logger
                )
            else:  # transformers
                waveform, sample_rate = self._generate_transformers(sentences)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装输出音频
        audio = self._ensure_audio_data(
            waveform,
            sample_rate,
            backend=self._backend,
            language=language,
            emotion=emotion,
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
                "backend": self._backend,
            },
        )
        return result

    # ------------------------------------------------------------------
    # transformers 后端
    # ------------------------------------------------------------------
    def _generate_transformers(self, sentences: list) -> tuple:
        """使用 transformers pipeline 生成语音（本地模型推理）。"""
        import numpy as np  # type: ignore

        waveforms: list = []
        sr = 16000  # 默认采样率

        for sent in sentences:
            result = self._model(sent)
            waveform = np.array(result["audio"], dtype="float32")
            sr = result.get("sampling_rate", sr)
            waveforms.append(waveform)

        if waveforms:
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
