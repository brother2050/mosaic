# mosaic/nodes/audio/_base.py
"""音频域节点基类。

提取音频生成/识别节点共用的模型加载、推理与音频前后处理逻辑。
子类只需实现 :meth:`BaseAudioNode.run` 与 :meth:`_load_model`，底层
推理流程由本基类提供。

设计要点
--------
* ``transformers`` / ``torch`` / ``soundfile`` / ``librosa`` / ``edge-tts``
  均采用惰性导入，使本模块在未安装这些依赖时仍可被注册表发现与导入
  （仅在实际加载/推理时才报依赖缺失）。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
* 提供统一的音频前后处理工具：加载、保存、重采样、归一化、单声道转换。
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any

from mosaic.core._device_utils import infer_device, resolve_device
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import AudioData, MosaicData

__all__ = ["BaseAudioNode"]


# 常见音频模型的粗略显存估算（GB），用于 describe() 与调度器
_VRAM_ESTIMATES: dict[str, float] = {
    # edge-tts 为云端 TTS，不占用本地 GPU 显存
    "edge-tts": 0.0,
    "openai/whisper-large-v3": 5.0,
    "openai/whisper-medium": 3.0,
    "openai/whisper-small": 1.0,
    "openai/whisper-base": 0.5,
    "openai/whisper-tiny": 0.3,
    "facebook/mms-tts-eng": 2.0,
    "microsoft/speecht5_tts": 2.0,
    "facebook/musicgen-small": 3.0,
    "facebook/musicgen-medium": 6.0,
    "facebook/musicgen-large": 12.0,
    "cvssp/audioldm2": 8.0,
    "cvssp/audioldm2-large": 12.0,
    "cvssp/audioldm2-music": 8.0,
}

# 许可证信息
_LICENSE_INFO: dict[str, str] = {
    # edge-tts 为微软 Azure 神经网络语音的非官方 Python 客户端
    "edge-tts": "Microsoft Azure Neural TTS (unofficial client, MIT)",
    "openai/whisper-large-v3": "MIT License",
    "openai/whisper-medium": "MIT License",
    "openai/whisper-small": "MIT License",
    "openai/whisper-base": "MIT License",
    "openai/whisper-tiny": "MIT License",
    "facebook/mms-tts-eng": "CC-BY-NC 4.0",
    "microsoft/speecht5_tts": "MIT License",
    "facebook/musicgen-small": "CC-BY-NC 4.0",
    "facebook/musicgen-medium": "CC-BY-NC 4.0",
    "facebook/musicgen-large": "CC-BY-NC 4.0",
    "cvssp/audioldm2": "CC-BY-NC-SA 4.0",
    "cvssp/audioldm2-large": "CC-BY-NC-SA 4.0",
    "cvssp/audioldm2-music": "CC-BY-NC-SA 4.0",
}


class BaseAudioNode(Node):
    """音频域节点抽象基类。

    封装基于 ``transformers`` / ``diffusers`` / ``edge-tts`` 的音频模型加载
    与推理流程。子类需实现 :meth:`run` 与 :meth:`_load_model`。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    sample_rate:
        输出音频的目标采样率，``None`` 表示使用模型默认采样率。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "audio"
    description: str = "Base audio node."
    version: str = "0.1.0"
    input_types: list[str] = ["text", "audio", "mosaic"]
    output_types: list[str] = ["audio"]

    def __init__(
        self,
        model: str = "",
        device: str = "cuda",
        sample_rate: int | None = None,
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._model_name: str = model
        self._device: str = device
        self._target_sample_rate: int | None = sample_rate
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._logger = logging.getLogger(f"mosaic.nodes.audio.{self.name}")

        # 运行时持有的模型/管道（load 后填充）
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载音频模型到 GPU/CPU。

        通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
        ``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
        以免递归。
        """
        self._scheduler.track(self)

        if self._model is not None:
            self._loaded = True
            return

        self._logger.info("Loading audio model %s ...", self._model_name)
        self._load_model()
        self._loaded = True

    @abc.abstractmethod
    def _load_model(self) -> None:
        """子类实现：实际加载模型。

        子类应在此方法中：
        1. 惰性导入所需的库（transformers / diffusers / edge-tts 等）；
        2. 加载模型与 processor/processor；
        3. 迁移到目标设备；
        4. 将模型赋值给 ``self._model``。
        """

    def unload(self) -> None:
        """释放音频模型。

        本方法执行实际资源清理。它由 ``Scheduler.release`` /
        ``Scheduler._evict`` 回调，不应在其中调用
        ``scheduler.release(self)`` 以免递归。
        """
        self._model = None
        self._processor = None
        self._loaded = False
        self._logger.info("Audio model %s unloaded.", self._model_name)

    # ------------------------------------------------------------------
    # 设备与推理辅助
    # ------------------------------------------------------------------
    def _infer_device(self) -> str:
        """推断推理设备。"""
        return infer_device(self._model, self._scheduler)

    def _resolve_device(self) -> str:
        """解析实际设备字符串，无 GPU 时降级到 CPU。"""
        device = resolve_device(self._device)
        if device != self._device:
            self._logger.warning(
                "CUDA not available, falling back to CPU for %s.",
                self.name,
            )
        return device

    # ------------------------------------------------------------------
    # 音频前后处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio(path_or_array: Any) -> tuple[Any, int]:
        """从文件路径或数组加载音频。

        Parameters
        ----------
        path_or_array:
            文件路径（str）或 ``numpy.ndarray`` 波形数据。如果是
            ``AudioData`` 实例，直接提取 waveform 和 sample_rate。

        Returns
        -------
        tuple[numpy.ndarray, int]
            ``(waveform, sample_rate)``。
        """
        # AudioData 实例
        if isinstance(path_or_array, AudioData):
            return path_or_array.waveform, path_or_array.sample_rate

        # 已经是 numpy 数组
        try:
            import numpy as np  # type: ignore

            if isinstance(path_or_array, np.ndarray):
                return path_or_array, 22050
        except ImportError:
            pass

        # 文件路径
        if isinstance(path_or_array, str):
            try:
                import soundfile as sf  # type: ignore

                waveform, sr = sf.read(path_or_array, dtype="float32")
                # soundfile 返回 (samples, channels)，转为 (channels, samples)
                import numpy as np  # type: ignore

                if waveform.ndim == 1:
                    return waveform, sr
                return waveform.T, sr
            except ImportError:
                pass

            try:
                import librosa  # type: ignore

                waveform, sr = librosa.load(path_or_array, sr=None)
                return waveform, sr
            except ImportError:
                pass

            raise ImportError(
                "Loading audio from file requires 'soundfile' or 'librosa'. "
                "Install via `pip install soundfile` or `pip install librosa`."
            )

        raise TypeError(
            f"Expected file path (str), numpy.ndarray, or AudioData, "
            f"got {type(path_or_array).__name__}."
        )

    @staticmethod
    def _save_audio(waveform: Any, sample_rate: int, path: str) -> None:
        """保存音频为 wav 文件。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据，形状 ``(channels, samples)`` 或
            ``(samples,)``。
        sample_rate:
            采样率。
        path:
            输出文件路径。
        """
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # soundfile 期望 (samples, channels)
        if isinstance(waveform, np.ndarray):
            # 防御性 dtype 转换：soundfile 只支持 float32/float64/int16/int32
            if waveform.dtype not in (np.float32, np.float64, np.int16, np.int32):
                waveform = waveform.astype(np.float32)
            if waveform.ndim == 2:
                # (channels, samples) -> (samples, channels)
                waveform = waveform.T
        sf.write(path, waveform, sample_rate)

    @staticmethod
    def _resample(
        waveform: Any, orig_sr: int, target_sr: int
    ) -> Any:
        """重采样音频到目标采样率。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据。
        orig_sr:
            原始采样率。
        target_sr:
            目标采样率。

        Returns
        -------
        numpy.ndarray
            重采样后的波形数据。
        """
        if orig_sr == target_sr:
            return waveform
        if orig_sr <= 0 or target_sr <= 0:
            return waveform

        try:
            import librosa  # type: ignore

            return librosa.resample(
                waveform, orig_sr=orig_sr, target_sr=target_sr
            )
        except ImportError:
            # 无 librosa 时使用简单的线性插值
            import numpy as np  # type: ignore

            if waveform.ndim == 1:
                num_samples = int(len(waveform) * target_sr / orig_sr)
                indices = np.linspace(0, len(waveform) - 1, num_samples)
                return np.interp(indices, np.arange(len(waveform)), waveform)
            # 多声道：逐声道处理
            import numpy as np  # type: ignore

            num_samples = int(waveform.shape[-1] * target_sr / orig_sr)
            indices = np.linspace(0, waveform.shape[-1] - 1, num_samples)
            channels = []
            for ch in waveform:
                channels.append(np.interp(indices, np.arange(len(ch)), ch))
            return np.stack(channels)

    @staticmethod
    def _get_duration(waveform: Any, sample_rate: int) -> float:
        """获取音频时长（秒）。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据。
        sample_rate:
            采样率。

        Returns
        -------
        float
            时长（秒）。
        """
        import numpy as np  # type: ignore

        if isinstance(waveform, np.ndarray):
            num_samples = waveform.shape[-1]
            if sample_rate <= 0:
                return 0.0
            return float(num_samples) / float(sample_rate)
        return 0.0

    @staticmethod
    def _normalize(waveform: Any) -> Any:
        """归一化音频波形到 ``[-1, 1]`` 范围。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据。

        Returns
        -------
        numpy.ndarray
            归一化后的波形数据。
        """
        import numpy as np  # type: ignore

        if not isinstance(waveform, np.ndarray):
            return waveform

        max_val = float(np.max(np.abs(waveform)))
        if max_val > 0:
            return waveform / max_val
        return waveform

    @staticmethod
    def _to_mono(waveform: Any) -> Any:
        """将多声道音频转为单声道。

        Parameters
        ----------
        waveform:
            ``numpy.ndarray`` 波形数据，形状 ``(channels, samples)`` 或
            ``(samples,)``。

        Returns
        -------
        numpy.ndarray
            单声道波形数据，形状 ``(samples,)``。
        """
        import numpy as np  # type: ignore

        if not isinstance(waveform, np.ndarray):
            return waveform

        if waveform.ndim == 1:
            return waveform
        # 多声道取平均
        return np.mean(waveform, axis=0)

    def _ensure_audio_data(
        self, waveform: Any, sample_rate: int, **extra: Any
    ) -> AudioData:
        """将波形数据包装为 AudioData，应用目标采样率与归一化。

        Parameters
        ----------
        waveform:
            波形数据。
        sample_rate:
            原始采样率。
        **extra:
            额外 metadata 字段。

        Returns
        -------
        AudioData
            包装后的音频数据。
        """
        # 应用目标采样率
        if self._target_sample_rate is not None and sample_rate != self._target_sample_rate:
            waveform = self._resample(waveform, sample_rate, self._target_sample_rate)
            sample_rate = self._target_sample_rate

        # 归一化
        waveform = self._normalize(waveform)

        duration = self._get_duration(waveform, sample_rate)
        metadata = {"duration": duration, "format": "wav"}
        metadata.update(extra)

        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Node 抽象方法
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑（子类实现）。"""

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=self._build_model_info(),
        )

    def _build_model_info(self) -> dict[str, Any]:
        """构造模型信息字典。"""
        vram = _VRAM_ESTIMATES.get(self._model_name, 4.0)
        license_info = _LICENSE_INFO.get(
            self._model_name, "See model card on HuggingFace"
        )
        info: dict[str, Any] = {
            "name": self._model_name,
            "source": "HuggingFace",
            "license": license_info,
            "vram_gb": vram,
            "device": self._device,
        }
        if self._target_sample_rate is not None:
            info["sample_rate"] = self._target_sample_rate
        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"model={self._model_name!r} state={status}>"
        )
