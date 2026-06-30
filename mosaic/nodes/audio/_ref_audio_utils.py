"""参考音频预处理工具。

统一处理 TTS 后端的参考音频加载、时长校验和自动截断。
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

# 各后端的参考音频推荐时长（秒）
REF_AUDIO_LIMITS: dict[str, tuple[float, float]] = {
    "sovits": (3.0, 10.0),      # GPT-SoVITS: 3-10秒
    "cosyvoice": (3.0, 10.0),   # CosyVoice: 3-10秒
    "fish": (3.0, 30.0),        # Fish Speech: 3-30秒
    "default": (3.0, 30.0),     # 默认: 3-30秒
}

#: 扩展名 -> 音频格式标识映射（用于显式格式探测，C1-2）。
_AUDIO_FORMAT_MAP: dict[str, str] = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
    ".m4a": "m4a",
    ".aac": "aac",
    ".wma": "wma",
    ".opus": "opus",
    ".aiff": "aiff",
}


def _detect_audio_format(path: str) -> str:
    """基于扩展名检测音频格式。

    Parameters
    ----------
    path:
        音频文件路径。

    Returns
    -------
    str
        格式标识（如 ``"wav"``/``"mp3"``/``"flac"``）；无法识别时返回
        ``"unknown"``。
    """
    ext = Path(path).suffix.lower()
    return _AUDIO_FORMAT_MAP.get(ext, "unknown")


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """将多声道波形合并为单声道。

    ``soundfile`` 等库返回 ``[samples, channels]`` 布局，但部分库
    （如 ``torch`` / 旧版 ``librosa``）返回 ``[channels, samples]``。
    通过比较维度大小启发式判断布局并转置后再取均值，避免误合并
    （C3-1）。

    Parameters
    ----------
    waveform:
        输入波形，1D 或 2D。

    Returns
    -------
    np.ndarray
        1D 单声道波形。
    """
    if waveform.ndim <= 1:
        return waveform
    if waveform.ndim == 2:
        # 通常 samples >> channels；若首维小于次维，更可能是 [channels, samples]
        if 0 < waveform.shape[0] < waveform.shape[1]:
            logger.debug(
                "Waveform shape %s looks like [channels, samples]; transposing.",
                waveform.shape,
            )
            waveform = waveform.T
        waveform = waveform.mean(axis=-1)
    else:
        # 高维（罕见）退化为沿最后一维求均值
        waveform = waveform.mean(axis=-1)
    return waveform


def load_reference_audio(
    audio_path_or_data: str | Any,
    target_sr: int,
    backend: str = "default",
    max_duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """加载参考音频，自动截断到推荐时长。

    Parameters
    ----------
    audio_path_or_data:
        音频文件路径或 AudioData 对象。
    target_sr:
        目标采样率，会自动重采样。
    backend:
        后端名称，用于确定推荐时长上限。如 "sovits", "cosyvoice", "fish"。
    max_duration:
        手动指定最大时长（秒），覆盖后端默认值。

    Returns
    -------
    tuple[np.ndarray, int]
        (waveform, sample_rate) — 截断并重采样后的音频。

    Raises
    ------
    FileNotFoundError
        文件路径不存在。
    ValueError
        音频时长过短（< 1秒）。
    """
    # 确定时长上限
    min_dur, max_dur = REF_AUDIO_LIMITS.get(backend, REF_AUDIO_LIMITS["default"])
    if max_duration is not None:
        max_dur = max_duration

    # 加载音频（支持文件路径和 AudioData）
    if isinstance(audio_path_or_data, str):
        if not os.path.exists(audio_path_or_data):
            raise FileNotFoundError(f"参考音频文件不存在: {audio_path_or_data}")
        fmt = _detect_audio_format(audio_path_or_data)
        if fmt == "unknown":
            logger.warning(
                "无法识别参考音频格式（路径: %s）；将尝试通用加载流程。",
                audio_path_or_data,
            )
        else:
            logger.debug("检测到参考音频格式: %s (%s)", fmt, audio_path_or_data)
        waveform, sr = _load_from_file(audio_path_or_data, target_sr)
    elif hasattr(audio_path_or_data, "waveform") and hasattr(audio_path_or_data, "sample_rate"):
        # AudioData 对象
        waveform = np.asarray(audio_path_or_data.waveform, dtype=np.float32)
        sr = audio_path_or_data.sample_rate
        if sr != target_sr:
            waveform = _resample(waveform, sr, target_sr)
            sr = target_sr
    else:
        # 假设是 numpy array
        waveform = np.asarray(audio_path_or_data, dtype=np.float32)
        sr = target_sr

    # 确保是 1D 数组（C3-1：自动处理 [channels, samples] 布局）
    waveform = _to_mono(waveform)

    # 计算时长
    duration = len(waveform) / sr

    # 时长过短检查
    if duration < 1.0:
        raise ValueError(
            f"参考音频时长过短 ({duration:.1f}s)，至少需要 1.0 秒。"
            f" {backend} 后端推荐 {min_dur:.0f}-{max_dur:.0f} 秒。"
        )

    # 时长过长警告 + 自动截断
    if duration > max_dur:
        logger.warning(
            "参考音频时长 %.1fs 超过 %s 后端推荐上限 %.0fs，"
            "自动截取前 %.0f 秒。",
            duration, backend, max_dur, max_dur,
        )
        # 截取前 max_dur 秒（取中间部分效果更好，但取前部更简单可靠）
        max_samples = int(max_dur * sr)
        waveform = waveform[:max_samples]

    # 时长偏短提示
    if duration < min_dur:
        logger.warning(
            "参考音频时长 %.1fs 短于 %s 后端推荐下限 %.0fs，"
            "克隆效果可能不佳。",
            duration, backend, min_dur,
        )

    return waveform, sr


def _load_from_file(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """从文件加载音频并重采样到目标采样率。

    优先使用 ``soundfile``（快，但部分平台不支持 mp3/m4a 等）；
    失败时回退到 ``librosa``（支持更多格式但更慢），并记录明确告警
    （C1-1）。
    """
    try:
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32")
        waveform = _to_mono(waveform)
        if sr != target_sr:
            waveform = _resample(waveform, sr, target_sr)
        return waveform, target_sr
    except ImportError:
        # soundfile 未安装，静默回退到 librosa
        pass
    except Exception as exc:
        # soundfile 已安装但加载失败（如不支持 mp3 等格式），回退到 librosa
        logger.warning(
            "soundfile failed to load %s: %s. "
            "Trying librosa fallback (supports more formats but slower).",
            path, exc,
        )

    try:
        import librosa
        waveform, sr = librosa.load(path, sr=target_sr, mono=True)
        return waveform, target_sr
    except ImportError:
        raise ImportError(
            "加载音频需要 soundfile 或 librosa，请安装: pip install soundfile"
        )


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样音频。

    优先使用 ``librosa.resample``（高质量）；不可用时退化为线性插值
    （质量较差，大比率变化时尤其明显），并记录告警（C2-2）。
    """
    if orig_sr == target_sr:
        return waveform
    try:
        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        logger.warning(
            "librosa not available; using linear interpolation resampling "
            "(%d -> %d Hz). Quality may be degraded for large ratio changes.",
            orig_sr, target_sr,
        )
        # 简单线性插值重采样作为回退
        ratio = target_sr / orig_sr
        n_samples = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, n_samples)
        return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
