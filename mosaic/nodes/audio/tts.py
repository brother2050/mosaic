# mosaic/nodes/audio/tts.py
"""TTS 节点 —— 文本转语音。

将文本转换为语音，默认使用 edge-tts（微软 Azure 神经网络语音）：
免费、支持中文与多语言、且具备情感风格（cheerful/sad/excited/gentle 等）。
同时保留 transformers pipeline 作为可选本地推理后端（如
``facebook/mms-tts-eng``）。

设计要点
--------
* 两种后端：
    - 模式 A (edge-tts)：**默认主力后端**，无需 GPU。通过选择不同的
      Neural 语音来表达不同情感风格，通过 ``rate`` 参数控制语速。
    - 模式 B (transformers)：当 ``model`` 指向 HuggingFace TTS 模型
      （如 ``facebook/mms-tts-eng``）时使用 ``transformers.pipeline``。
* 情感通过 ``emotion`` 参数选择，映射到不同的预设 Neural 语音。
* 长文本自动分句处理，避免一次生成过长音频。
* 输出统一为 :class:`~mosaic.core.types.AudioData` 格式。

注：Coqui XTTS-v2 已移除，edge-tts 不再是"回退方案"而是默认主力方案。
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["TTS"]


# ---------------------------------------------------------------------------
# 情感 -> 预设 Neural 语音映射
# ---------------------------------------------------------------------------
# edge-tts 库的 Communicate 不直接支持 SSML 的 mstts:express-as 情感标签，
# 因此情感风格通过选择不同的 Neural 语音来表达：
#   * zh-CN-XiaoxiaoNeural  —— 默认女声（neutral）
#   * zh-CN-XiaoyiNeural    —— 年轻女声，更有情感表现力（cheerful/excited）
#   * zh-CN-XiaomoNeural    —— 温柔女声（gentle/sad）
#   * zh-CN-XiaoruiNeural   —— 沉稳女声（calm）
#   * zh-CN-YunjianNeural   —— 男声，适合播报（male/angry）
#   * zh-CN-YunxiNeural     —— 年轻男声（young_male）
#   * zh-CN-YunxiaNeural    —— 儿童声（child）
_EMOTION_VOICE_MAP: Dict[str, Dict[str, str]] = {
    "zh": {
        "neutral": "zh-CN-XiaoxiaoNeural",
        "cheerful": "zh-CN-XiaoyiNeural",
        "sad": "zh-CN-XiaomoNeural",
        "excited": "zh-CN-XiaoyiNeural",
        "angry": "zh-CN-YunjianNeural",
        "gentle": "zh-CN-XiaomoNeural",
        "calm": "zh-CN-XiaoruiNeural",
        "male": "zh-CN-YunjianNeural",
        "young_male": "zh-CN-YunxiNeural",
        "child": "zh-CN-YunxiaNeural",
    },
    "en": {
        "neutral": "en-US-JennyNeural",
        "cheerful": "en-US-AriaNeural",
        "sad": "en-US-MichelleNeural",
        "excited": "en-US-AriaNeural",
        "angry": "en-US-GuyNeural",
        "gentle": "en-US-MichelleNeural",
        "calm": "en-US-DavisNeural",
        "male": "en-US-GuyNeural",
    },
    "ja": {
        "neutral": "ja-JP-NanamiNeural",
        "cheerful": "ja-JP-NanamiNeural",
        "male": "ja-JP-KeitaNeural",
    },
    "ko": {
        "neutral": "ko-KR-SunHiNeural",
        "male": "ko-KR-InJoonNeural",
    },
    "fr": {
        "neutral": "fr-FR-DeniseNeural",
        "male": "fr-FR-HenriNeural",
    },
    "de": {
        "neutral": "de-DE-KatjaNeural",
        "male": "de-DE-ConradNeural",
    },
    "es": {
        "neutral": "es-ES-ElviraNeural",
        "male": "es-ES-AlvaroNeural",
    },
}


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


def _resolve_voice(
    language: str, emotion: str, voice_override: Optional[str]
) -> str:
    """根据 language/emotion 解析 edge-tts 语音名称。

    Parameters
    ----------
    language:
        语言代码，如 ``"zh"``、``"en"``。
    emotion:
        情感风格，如 ``"neutral"``、``"cheerful"``。
    voice_override:
        显式指定的语音名称；非空时优先使用。

    Returns
    -------
    str
        edge-tts 语音名称，如 ``zh-CN-XiaoxiaoNeural``。
    """
    if voice_override:
        return voice_override
    lang_map = _EMOTION_VOICE_MAP.get(
        language, _EMOTION_VOICE_MAP.get("en", {})
    )
    voice = lang_map.get(emotion)
    if voice is None:
        # 未知情感回退到 neutral，再回退到该语言第一个语音
        voice = lang_map.get("neutral") or next(
            iter(lang_map.values()), "en-US-JennyNeural"
        )
    return voice


def _synthesize_edge_tts(
    sentences: List[str], voice: str, speed: float, logger: Any = None
) -> tuple:
    """使用 edge-tts 合成语音并解码为 numpy 波形。

    本函数为 :class:`TTS` 与 :class:`~mosaic.nodes.audio.voice_clone.VoiceClone`
    共用的底层合成工具，避免重复实现。

    Parameters
    ----------
    sentences:
        分句后的文本列表。
    voice:
        edge-tts 语音名称，如 ``zh-CN-XiaoxiaoNeural``。
    speed:
        语速倍率，``1.0`` 为正常语速。
    logger:
        可选的日志器，用于记录合成进度。

    Returns
    -------
    tuple
        ``(waveform, sample_rate)``。
    """
    import asyncio
    import io

    import numpy as np  # type: ignore
    import edge_tts  # type: ignore

    # 语速：edge-tts 接受 "+X%" / "-X%"
    rate_pct = int(round((speed - 1) * 100))
    rate_str = f"{rate_pct:+d}%"

    async def _synth(text: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    # 逐句合成并合并
    all_audio: List[bytes] = []
    for sent in sentences:
        audio_bytes = asyncio.run(_synth(sent))
        all_audio.append(audio_bytes)

    combined_bytes = b"".join(all_audio)

    # 解码 mp3 字节流为 numpy 波形
    try:
        import soundfile as sf  # type: ignore

        waveform, sr = sf.read(io.BytesIO(combined_bytes), dtype="float32")
        if logger is not None:
            logger.debug(
                "edge-tts synthesized %d sentence(s) -> %d samples @ %dHz.",
                len(sentences),
                int(waveform.shape[-1]) if waveform is not None else 0,
                sr,
            )
        return waveform, sr
    except Exception:
        # 回退：逐句保存临时文件用 librosa 解码
        import os
        import tempfile

        waveforms: List[Any] = []
        sr = 24000
        for audio_bytes in all_audio:
            tmp_path = tempfile.mktemp(suffix=".mp3")
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            try:
                import librosa  # type: ignore

                wf, sr = librosa.load(tmp_path, sr=None)
                waveforms.append(wf)
            except ImportError:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if waveforms:
            waveform = np.concatenate(waveforms)
            return waveform, sr

        raise ImportError(
            "edge-tts requires 'soundfile' or 'librosa' to decode audio. "
            "Install via `pip install soundfile` or `pip install librosa`."
        )


@registry.register
class TTS(BaseAudioNode):
    """文本转语音节点。

    将文本转换为语音，默认使用 edge-tts（免费、多语言、情感风格）。
    也可通过 ``model`` 参数指定 HuggingFace TTS 模型走 transformers
    pipeline 后端进行本地推理。

    Parameters
    ----------
    model:
        模型标识。默认 ``"edge-tts"`` 使用 edge-tts 云端 TTS（无需 GPU）；
        若指定为 HuggingFace TTS 模型（如 ``facebook/mms-tts-eng``），
        则走 transformers pipeline 后端；加载失败时回退到 edge-tts。
    voice:
        可选，直接指定 edge-tts 语音名称（如 ``zh-CN-XiaoxiaoNeural``），
        优先于 ``emotion``。
    language:
        语音语言代码，如 ``"zh"``、``"en"``，默认 ``"zh"``。
    emotion:
        情感风格，可选值：``"neutral"``/``"cheerful"``/``"sad"``/
        ``"excited"``/``"angry"``/``"gentle"``/``"calm"`` 等，默认
        ``"neutral"``。仅在 edge-tts 后端且未显式指定 ``voice`` 时生效。
    speed:
        语速倍率，``1.0`` 为正常语速，默认 ``1.0``。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> tts = TTS(language="zh")
    >>> result = tts(MosaicData(text="你好，世界！"))
    >>> result["audio"].waveform  # numpy.ndarray

    指定情感风格：
    >>> tts = TTS(language="zh", emotion="cheerful")
    >>> result = tts(MosaicData(text="今天天气真好！"))

    运行时覆盖情感与语速：
    >>> result = tts(MosaicData(text="慢点说", emotion="calm", speed=0.8))
    """

    name: str = "tts"
    description: str = (
        "Convert text to speech using edge-tts (default, multi-language, "
        "emotional neural voices) or a transformers TTS pipeline. "
        "Supports emotion styles and speed control."
    )
    version: str = "0.2.0"
    input_types = ["text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "edge-tts",
        voice: Optional[str] = None,
        language: str = "zh",
        emotion: str = "neutral",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._voice: Optional[str] = voice
        self._language: str = language
        self._emotion: str = emotion
        self._speed: float = float(speed)
        self._backend: str = "edge_tts"

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载 TTS 模型。

        后端选择逻辑：

        1. 若 ``self._model_name`` 为 ``"edge-tts"`` 或以 ``"edge"`` 开头，
           使用 edge-tts 云端后端（无需 GPU）。
        2. 否则尝试使用 ``transformers.pipeline("text-to-speech")`` 加载
           指定的 HuggingFace TTS 模型。
        3. 若 transformers 加载失败，回退到 edge-tts。
        """
        if (
            self._model_name == "edge-tts"
            or self._model_name.lower().startswith("edge")
        ):
            self._backend = "edge_tts"
            self._logger.info(
                "Using edge-tts backend (no GPU required, "
                "voice=%s, emotion=%s).",
                self._voice or "<auto>",
                self._emotion,
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
                "Transformers TTS pipeline loaded (model=%s, device=%s).",
                self._model_name,
                device,
            )
            return
        except Exception as exc:
            # 捕获所有异常（ImportError / ValueError / OSError 等），
            # 确保能回退到 edge-tts
            self._logger.warning(
                "Transformers TTS failed for %s: %s. "
                "Falling back to edge-tts.",
                self._model_name,
                exc,
            )
            self._backend = "edge_tts"

    # ------------------------------------------------------------------
    # 语音解析
    # ------------------------------------------------------------------
    def _resolve_voice(
        self, language: str, emotion: str, voice_override: Optional[str]
    ) -> str:
        """根据 language/emotion 解析 edge-tts 语音名称。"""
        return _resolve_voice(language, emotion, voice_override)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行文本转语音。

        Parameters
        ----------
        input_data:
            必须包含 ``text`` (str)；可选 ``voice`` (str)、
            ``language`` (str)、``emotion`` (str)、``speed`` (float, 默认 1.0)。
            这些参数会覆盖构造函数的默认设置。

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

            # 运行时参数（覆盖构造函数默认值）
            language = input_data.get("language", self._language)
            emotion = input_data.get("emotion", self._emotion)
            voice = input_data.get("voice", self._voice)
            speed = float(input_data.get("speed", self._speed))

            # 分句处理
            sentences = _split_sentences(text)
            self._logger.info(
                "TTS generating %d sentence(s) via %s backend "
                "(emotion=%s, speed=%.2f).",
                len(sentences),
                self._backend,
                emotion,
                speed,
            )

            # 根据后端生成语音
            if self._backend == "edge_tts":
                resolved_voice = self._resolve_voice(
                    language, emotion, voice
                )
                waveform, sr = self._generate_edge_tts(
                    sentences, resolved_voice, speed
                )
            else:  # transformers
                waveform, sr = self._generate_transformers(sentences)
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 AudioData
        audio = self._ensure_audio_data(
            waveform,
            sr,
            backend=self._backend,
            language=language,
            emotion=emotion,
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
                "emotion": emotion,
            },
        )
        return result

    # ------------------------------------------------------------------
    # edge-tts 后端
    # ------------------------------------------------------------------
    def _generate_edge_tts(
        self, sentences: List[str], voice: str, speed: float
    ) -> tuple:
        """使用 edge-tts 生成语音。

        Parameters
        ----------
        sentences:
            分句后的文本列表。
        voice:
            edge-tts 语音名称，如 ``zh-CN-XiaoxiaoNeural``。
        speed:
            语速倍率，``1.0`` 为正常语速。
        """
        return _synthesize_edge_tts(sentences, voice, speed, self._logger)

    # ------------------------------------------------------------------
    # transformers 后端
    # ------------------------------------------------------------------
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
