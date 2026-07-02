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
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.coerce import safe_float
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
_EMOTION_VOICE_MAP: dict[str, dict[str, str]] = {
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


# ---------------------------------------------------------------------------
# 句子分割与文本预处理
# ---------------------------------------------------------------------------
# edge-tts（Azure Neural TTS）输出固定为 24kHz，作为两条解码路径的统一
# 采样率来源，避免 soundfile 实际读取值与 librosa 回退默认值不一致。
EDGE_TTS_SAMPLE_RATE: int = 24000

# 占位符：用于保护小数/日期/缩写中的句点，使其不被当作句末标点。
# 使用 null 字节序列，TTS 输入文本中不会出现。
_PROTECTED_DOT: str = "\x00DOT\x00"

# 常见英文缩写（后接句点不应分句）。按长度降序排列，确保 "Sept" 优先于
# "Sep" 匹配，避免短缩写"吃掉"长缩写的尾部字符。
_ABBREVIATIONS: frozenset[str] = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Co", "Corp",
    "e.g", "i.e", "etc", "vs", "cf",
    "No", "Vol", "pp", "p",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
})

# 缩写 + 句点：\b(Mr|Dr|...)\. —— 长缩写在前，避免短缩写优先匹配。
_ABBREV_DOT_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(
        re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)
    ) + r")\.",
    re.IGNORECASE,
)

# 小数/日期中的句点：数字之间的 "."，如 3.14 / 2024.06.30
_DECIMAL_DOT_RE: re.Pattern[str] = re.compile(r"(?<=\d)\.(?=\d)")

# 句末标点（中英文）
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"[。！？.!?；;\n]+")

# CJK 汉字检测
_CJK_RE: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]")


def _is_chinese(text: str) -> bool:
    """检测文本是否包含 CJK 汉字。"""
    return bool(_CJK_RE.search(text))


def _split_sentences(text: str, max_length: int = 200) -> list[str]:
    """将长文本按句子分割，避免一次生成过长音频。

    采用"保护-分割-还原"三步策略，避免误切缩写（``Mr.``）、小数
    （``3.14``）、序号（``No.1``）与日期（``2024.06.30``）：

    1. **保护**：用占位符替换小数/日期中的句点（``\\d.\\d``）以及缩写
       后的句点（``Mr.`` / ``e.g.`` 等，含内部句点）。
    2. **分割**：按句末标点（``。！？.!?；;\\n``）切分。
    3. **还原**：将占位符还原为句点。

    超过 ``max_length`` 的句子进一步按逗号/空格切分；重组时若含 CJK
    字符则不加空格，避免污染中文。

    Parameters
    ----------
    text:
        待分割的文本。
    max_length:
        每段最大字符数，超出时在逗号/空格处进一步切分。

    Returns
    -------
    list[str]
        分割后的句子列表。
    """
    # 1. 保护小数/日期中的句点：3.14 -> 3<PH>14
    protected = _DECIMAL_DOT_RE.sub(_PROTECTED_DOT, text)

    # 2. 保护缩写后的句点：Mr. -> Mr<PH>，e.g. -> e<PH>g<PH>
    #    替换匹配到的全部句点（含 "e.g"/"i.e" 的内部句点），避免误切。
    protected = _ABBREV_DOT_RE.sub(
        lambda m: m.group(0).replace(".", _PROTECTED_DOT), protected
    )

    # 3. 按句末标点分割
    sentences = _SENTENCE_SPLIT_RE.split(protected)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 4. 还原占位符
    sentences = [s.replace(_PROTECTED_DOT, ".") for s in sentences]

    # 5. 超长句进一步按逗号/空格切分（B1-2：中文不加空格）
    result: list[str] = []
    for sent in sentences:
        if len(sent) <= max_length:
            result.append(sent)
            continue
        parts = re.split(r"[，,、\s]+", sent)
        current = ""
        for part in parts:
            if not part:
                continue
            if current and len(current) + len(part) + 1 > max_length:
                result.append(current)
                current = part
            elif current:
                if _is_chinese(current) or _is_chinese(part):
                    current = f"{current}{part}"
                else:
                    current = f"{current} {part}"
            else:
                current = part
        if current:
            result.append(current)

    return result if result else [text]


# ---------------------------------------------------------------------------
# SSML 处理
# ---------------------------------------------------------------------------
_SSML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")
_SSML_DETECT_RE: re.Pattern[str] = re.compile(
    r"<speak\b|<break\b|<emphasis\b|<prosody\b", re.IGNORECASE
)
# <break time="1.5s"/> / <break time="500ms"/>
_SSML_BREAK_RE: re.Pattern[str] = re.compile(
    r'<break\s+[^>]*?time="(\d+(?:\.\d+)?)(s|ms)"[^>]*/?\s*>',
    re.IGNORECASE,
)


def _has_ssml(text: str) -> bool:
    """检测文本是否包含 SSML 标签。"""
    return bool(_SSML_DETECT_RE.search(text))


def _strip_ssml(text: str) -> str:
    """剥离 SSML 标签，保留内部文本。

    ``<break time="Xs"/>`` 会被转换为对应时长的停顿标点（>=0.5s 用句号，
    否则用逗号），其余标签仅移除标签本身，保留内部文本。
    """
    def _break_repl(m: re.Match[str]) -> str:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "ms":
            val = val / 1000.0
        return "。" if val >= 0.5 else "，"

    text = _SSML_BREAK_RE.sub(_break_repl, text)
    # 移除所有剩余 SSML 标签，保留内部文本
    text = _SSML_TAG_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 音频拼接
# ---------------------------------------------------------------------------
def _concat_audios(audios: list[AudioData]) -> AudioData:
    """合并多个 :class:`AudioData` 为一个连续音频。

    用于扩展后端长文本分句合成后的多段音频拼接。同一后端的各段采样率
    理论上一致；若出现不一致则重采样至第一段的采样率（防御性处理）。

    Parameters
    ----------
    audios:
        待合并的 AudioData 列表，至少 1 个。

    Returns
    -------
    AudioData
        拼接后的音频，``metadata.duration`` 为各段时长之和。
    """
    if not audios:
        raise ValueError("Cannot concatenate an empty list of AudioData.")
    if len(audios) == 1:
        return audios[0]

    import numpy as np  # type: ignore

    ref_sr: int = audios[0].sample_rate
    waveforms: list[Any] = []
    for audio in audios:
        wf = audio.waveform
        sr = audio.sample_rate
        if sr != ref_sr:
            # 防御性重采样：同一后端理论上不会触发
            try:
                import librosa  # type: ignore

                wf = librosa.resample(wf, orig_sr=sr, target_sr=ref_sr)
            except Exception:  # noqa: BLE001
                pass
        if not isinstance(wf, np.ndarray):
            wf = np.array(wf, dtype=np.float32)
        elif wf.dtype != np.float32:
            wf = wf.astype(np.float32)
        waveforms.append(wf)

    # 维度对齐：若存在 2D 波形，将所有 1D 提升为 (1, samples)
    if any(w.ndim == 2 for w in waveforms):
        waveforms = [
            w[np.newaxis, :] if w.ndim == 1 else w for w in waveforms
        ]

    combined = np.concatenate(waveforms, axis=-1)

    # 合并 metadata：累加时长，保留首段元信息
    metadata: dict[str, Any] = dict(audios[0].metadata)
    metadata["duration"] = sum(
        float(a.metadata.get("duration", 0.0) or 0.0) for a in audios
    )
    metadata["segment_count"] = len(audios)

    return AudioData(
        waveform=combined, sample_rate=ref_sr, metadata=metadata
    )


def _resolve_voice(
    language: str, emotion: str, voice_override: str | None
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


def _decode_audio_bytes(audio_bytes: bytes, target_sr: int) -> Any:
    """将音频字节流解码为 numpy 波形数组。

    优先使用 soundfile，回退到 librosa。两种方式均失败时返回 None。
    """
    import io

    # 优先使用 soundfile（内存解码，无需临时文件）
    try:
        import soundfile as sf  # type: ignore

        waveform, _sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return waveform
    except Exception:  # noqa: BLE001
        pass

    # 回退：写临时文件用 librosa 解码
    import os
    import tempfile

    tmp_path = tempfile.mktemp(suffix=".mp3")
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        try:
            import librosa  # type: ignore

            wf, _sr = librosa.load(tmp_path, sr=None)
            return wf
        except ImportError:
            return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _synthesize_edge_tts(
    sentences: list[str], voice: str, speed: float, logger: Any = None
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
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)

    async def _synthesize_all() -> list[bytes]:
        """并发合成所有句子（单次事件循环，避免反复创建/销毁）。"""
        return await asyncio.gather(*[_synth(s) for s in sentences])

    # 单次事件循环并发合成全部句子
    all_audio: list[bytes] = asyncio.run(_synthesize_all())

    # 逐句解码为 numpy 波形后拼接（避免多 WAV/MP3 字节流拼接后无法解析）
    waveforms: list[Any] = []
    sr = EDGE_TTS_SAMPLE_RATE

    for audio_bytes in all_audio:
        decoded = _decode_audio_bytes(audio_bytes, sr)
        if decoded is not None:
            waveforms.append(decoded)

    if waveforms:
        waveform = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
        if logger is not None:
            logger.debug(
                "edge-tts synthesized %d sentence(s) -> %d samples @ %dHz.",
                len(sentences),
                int(waveform.shape[-1]) if waveform is not None else 0,
                sr,
            )
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
    max_sentence_length:
        单句最大字符数，超出时在逗号/空格处进一步切分，默认 ``200``。
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
        "emotional neural voices), a transformers TTS pipeline, or a "
        "pluggable TTS backend (ChatTTS / Fish Speech / CosyVoice / "
        "GPT-SoVITS via the tts_backends framework). "
        "Supports emotion styles, speed control, and streaming output."
    )
    version: str = "0.3.0"
    input_types = ("text", "mosaic")
    output_types = ("audio",)

    def __init__(
        self,
        backend: str = "auto",
        model: str = "edge-tts",
        voice: str | None = None,
        language: str = "zh",
        emotion: str = "neutral",
        speed: float = 1.0,
        speaker: str | None = None,
        stream_chunk_size: int = 4096,
        max_sentence_length: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._backend_name: str = backend
        self._voice: str | None = voice
        self._language: str = language
        self._emotion: str = emotion
        self._speed: float = float(speed)
        self._speaker: str | None = speaker
        self._stream_chunk_size: int = stream_chunk_size
        # B1-4: 可配置的最大句长，避免硬编码 200
        self._max_sentence_length: int = max_sentence_length
        self._backend_kwargs: dict[str, Any] = {}
        # 内置后端标识（edge_tts / transformers），与扩展后端区分
        self._backend: str = "edge_tts"
        # 扩展后端实例（使用 tts_backends 框架时填充）
        self._tts_backend: Any = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载 TTS 模型。

        后端路由逻辑（按优先级）：

        1. 若 ``backend`` 显式指定为内置后端（``"edge_tts"`` /
           ``"transformers"``），直接使用对应内置实现。
        2. 若 ``backend`` 指定了扩展后端名称（如 ``"chattts"`` /
           ``"fish"`` / ``"cosyvoice"`` / ``"sovits"``），从
           :class:`TTSBackendRegistry` 获取后端类并实例化。
        3. 若 ``backend="auto"``：
           a. ``model="edge-tts"`` → edge-tts 云端后端（无需 GPU）。
           b. ``model`` 指向 HuggingFace 模型 → transformers pipeline。
           c. 否则调用注册表 ``auto_select`` 自动选择最优扩展后端。
        4. 任何加载失败均回退到 edge-tts（保证可用性）。
        """
        backend_name = self._backend_name

        # ---- auto 模式：推断后端 ----
        if backend_name == "auto":
            if (
                self._model_name == "edge-tts"
                or self._model_name.lower().startswith("edge")
            ):
                backend_name = "edge_tts"
            else:
                try:
                    from mosaic.nodes.audio.tts_backends.registry import (
                        tts_backend_registry,
                    )
                    available = tts_backend_registry.list_backends()
                    if available:
                        backend_name = tts_backend_registry.auto_select(
                            {"language": self._language, "streaming": False, "gpu_memory_gb": 8.0}
                        )
                    else:
                        backend_name = "transformers"
                except Exception:  # noqa: BLE001
                    backend_name = "transformers"

        # ---- 内置后端 ----
        if backend_name in ("edge_tts", "edge-tts"):
            self._backend = "edge_tts"
            self._logger.info(
                "Using edge-tts backend (no GPU required, voice=%s, emotion=%s).",
                self._voice or "<auto>", self._emotion,
            )
            return

        if backend_name == "transformers":
            try:
                from transformers import pipeline
                device = self._resolve_device()
                self._model = pipeline("text-to-speech", model=self._model_name, device=device)
                self._backend = "transformers"
                self._logger.info("Transformers TTS pipeline loaded (model=%s, device=%s).", self._model_name, device)
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Transformers TTS failed for %s: %s. Falling back to edge-tts.", self._model_name, exc)
                self._backend = "edge_tts"
                return

        # ---- 扩展后端（tts_backends 框架） ----
        try:
            from mosaic.nodes.audio.tts_backends.registry import tts_backend_registry
            backend_class = tts_backend_registry.get(backend_name)
            if backend_class is None:
                self._logger.warning("TTS backend '%s' not registered. Falling back to edge-tts.", backend_name)
                self._backend = "edge_tts"
                return
            self._tts_backend = backend_class(model_path=self._model_name, **self._backend_kwargs)
            self._tts_backend.load(device=self._device, dtype=getattr(self, "_dtype", "float16"))
            self._backend = backend_name
            self._logger.info("TTS backend '%s' loaded successfully.", backend_name)
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to load TTS backend '%s': %s. Falling back to edge-tts.", backend_name, exc)
            self._backend = "edge_tts"

    # ------------------------------------------------------------------
    # 后端查询
    # ------------------------------------------------------------------
    @classmethod
    def list_backends(cls) -> list[str]:
        """列出所有可用的 TTS 后端名称。"""
        backends = ["edge_tts", "transformers"]
        try:
            from mosaic.nodes.audio.tts_backends.registry import tts_backend_registry
            for spec in tts_backend_registry.list_backends():
                if spec.name not in backends:
                    backends.append(spec.name)
        except Exception:  # noqa: BLE001
            pass
        return backends

    # ------------------------------------------------------------------
    # 语音解析
    # ------------------------------------------------------------------
    def _resolve_voice(
        self, language: str, emotion: str, voice_override: str | None
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
            speed = safe_float(input_data.get("speed"), "speed", default=self._speed)

            speaker = input_data.get("speaker", self._speaker)

            # ---- 扩展后端（tts_backends 框架） ----
            if self._tts_backend is not None:
                # B2-1: 扩展后端不支持 SSML，剥离标签保留纯文本
                synth_text = _strip_ssml(text) if _has_ssml(text) else text
                self._logger.info(
                    "TTS generating via %s backend (speaker=%s, language=%s, speed=%.2f).",
                    self._backend, speaker or "<default>", language, speed,
                )
                # B1-3: 对扩展后端也进行句子分割，避免超长文本导致越界报错
                sentences = _split_sentences(
                    synth_text, max_length=self._max_sentence_length
                )
                if len(sentences) > 1:
                    self._logger.info(
                        "TTS: splitting text into %d sentences for %s backend.",
                        len(sentences), self._backend,
                    )
                    audios: list[AudioData] = []
                    for sent in sentences:
                        seg_audio = self._tts_backend.synthesize(
                            text=sent, speaker=speaker,
                            language=language, speed=speed,
                        )
                        audios.append(seg_audio)
                    audio = _concat_audios(audios)
                else:
                    audio = self._tts_backend.synthesize(
                        text=synth_text, speaker=speaker,
                        language=language, speed=speed,
                    )
                elapsed = time.perf_counter() - t0
                duration = audio.metadata.get("duration", 0.0)
                result = MosaicData(audio=audio, text=text, duration=duration)
                self._emit_complete(
                    duration=elapsed,
                    output_summary={"backend": self._backend, "duration": duration, "language": language},
                )
                return result

            # ---- 内置后端 ----

            # B2-1: SSML 处理 + B1-4: 可配置最大句长
            if _has_ssml(text):
                if self._backend == "edge_tts":
                    # edge-tts 原生支持 SSML，直接透传，不进行分句
                    sentences = [text]
                else:
                    # transformers 等后端不支持 SSML，剥离标签后分句
                    sentences = _split_sentences(
                        _strip_ssml(text),
                        max_length=self._max_sentence_length,
                    )
            else:
                sentences = _split_sentences(
                    text, max_length=self._max_sentence_length
                )
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
        except Exception as exc:  # noqa: BLE001
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
        self, sentences: list[str], voice: str, speed: float
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
    def _generate_transformers(self, sentences: list[str]) -> tuple:
        """使用 transformers pipeline 生成语音。"""
        import numpy as np  # type: ignore

        waveforms: list[Any] = []
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

    # ------------------------------------------------------------------
    # 流式合成与资源释放
    # ------------------------------------------------------------------
    def run_stream(self, input_data: MosaicData) -> Any:
        """流式文本转语音。返回生成器，每次 yield 一小段 AudioData。"""
        self._scheduler.ensure_loaded(self)
        text = input_data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"TTS requires 'text' (non-empty str), got {type(text).__name__}.")
        language = input_data.get("language", self._language)
        speed = safe_float(input_data.get("speed"), "speed", default=self._speed)
        speaker = input_data.get("speaker", self._speaker)

        if self._tts_backend is not None:
            # B2-1: 扩展后端不支持 SSML，剥离标签保留纯文本
            stream_text = _strip_ssml(text) if _has_ssml(text) else text
            yield from self._tts_backend.synthesize_stream(
                text=stream_text, speaker=speaker, language=language, speed=speed,
                chunk_size=self._stream_chunk_size,
            )
            return

        self._logger.info("TTS streaming via %s backend (non-streaming fallback).", self._backend)
        result = self.run(input_data)
        yield result["audio"]

    def unload(self) -> None:
        """释放 TTS 模型资源。同时释放内置后端模型和扩展后端实例。"""
        if self._tts_backend is not None:
            try:
                self._tts_backend.unload()
            except Exception:  # noqa: BLE001
                pass
            self._tts_backend = None
        super().unload()
