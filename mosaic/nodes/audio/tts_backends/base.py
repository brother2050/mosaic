# mosaic/nodes/audio/tts_backends/base.py
"""TTS 后端抽象基类。

定义 TTS 后端的统一接口与四层管线编排逻辑：

  Layer 1: TextFrontend   —— 文本前端（清洗、韵律、分词、tokenize）
  Layer 2: AcousticModel  —— 声学模型（token → mel / VQ tokens）
  Layer 3: Vocoder        —— 声码器（mel / VQ → waveform）
  Layer 4: StreamAdapter  —— 流式适配（缓冲区、chunk、实时输出）

子类只需实现 :meth:`TTSBackend._build_pipeline` 组装上述四层，即可获得
阻塞合成 (:meth:`synthesize`) 与流式合成 (:meth:`synthesize_stream`) 能力。

设计要点
--------
* ``torch`` / ``numpy`` 采用惰性导入（在方法内部 import），避免硬依赖，
  使本模块在未安装这些依赖时仍可被导入与注册。
* 显存生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理：
  加载前解析设备并校验显存，不足时尝试释放其他已加载节点。
* :class:`TTSBackendSpec` 描述后端能力，供注册表做自动选择。
* 本基类对四层组件采用鸭子类型，不直接 import 各层基类，从而避免与
  其他子任务产生的文件产生循环依赖或硬耦合；同时对各层返回值做防御性
  归一化，兼容不同实现细节。

四层组件预期接口
----------------
* ``TextFrontend.tokenize(text, language="zh", **kwargs) -> Any``
* ``AcousticModel.generate(token_ids, **kwargs) -> Any``
  （``speaker``/``language``/``speed`` 等高层参数通过 ``**kwargs`` 透传）
* ``AcousticModel.generate_stream(token_ids, **kwargs) -> Iterator[Any]``
* ``Vocoder.decode(features) -> waveform`` —— 返回单一波形；
  采样率取自 ``Vocoder.sample_rate``（或 ``spec.sample_rate`` 兜底）。
  兼容返回 ``(waveform, sample_rate)`` 元组的实现。
* ``Vocoder.decode_chunk(features) -> waveform`` —— 同上。
* ``StreamAdapter.create_stream() -> StreamSession``；
  ``StreamSession.push(waveform)`` / ``pop() -> AudioData | None``
  / ``flush() -> AudioData | None``。亦兼容适配器自身直接提供
  ``push``/``pop``/``flush`` 的实现。
* 各层均可选实现 ``unload_weights()`` 用于释放权重。
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import AudioData

__all__ = ["TTSBackendSpec", "TTSBackend"]


# ---------------------------------------------------------------------------
# TTSBackendSpec — 后端能力描述
# ---------------------------------------------------------------------------
@dataclass
class TTSBackendSpec:
    """TTS 后端规格描述。

    用于注册表登记后端能力，并支持按需求自动选择最优后端。

    Attributes
    ----------
    name:
        后端唯一名称，如 ``"edge_tts"`` / ``"cosyvoice"``。
    supported_languages:
        支持的语言代码列表，如 ``["zh", "en"]``。
    supports_streaming:
        是否支持流式合成。
    supports_voice_clone:
        是否支持语音克隆（零样本/少样本）。
    vocoder_type:
        声码器类型，如 ``"vocos"`` / ``"hifi_gan"`` / ``"sovits_decoder"``。
    acoustic_type:
        声学模型类型，如 ``"ar"``（自回归）/ ``"flow_matching"``。
    min_gpu_memory_gb:
        最小 GPU 显存需求（GB）；``0`` 表示无需 GPU（云端/CPU 后端）。
    model_license:
        模型许可证说明。
    sample_rate:
        输出音频采样率（Hz）。
    default_params:
        默认推理参数。
    """

    name: str
    supported_languages: list[str] = field(default_factory=list)
    supports_streaming: bool = False
    supports_voice_clone: bool = False
    vocoder_type: str = "hifi_gan"
    acoustic_type: str = "ar"
    min_gpu_memory_gb: float = 0.0
    model_license: str = ""
    sample_rate: int = 22050
    default_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TTSBackend — 抽象基类
# ---------------------------------------------------------------------------
class TTSBackend(abc.ABC):
    """TTS 后端抽象基类。

    子类通过实现 :meth:`_build_pipeline` 组装四层管线（文本前端、声学
    模型、声码器、流式适配器），即可获得统一的合成接口。

    生命周期
    --------
    1. 构造后端实例（``is_loaded=False``）。
    2. 调用 :meth:`load` 加载模型权重并组装管线（``is_loaded=True``）。
    3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
    4. 调用 :meth:`unload` 释放资源（``is_loaded=False``）。

    Parameters
    ----------
    scheduler:
        显存调度器实例，``None`` 使用全局单例
        :func:`~mosaic.core.scheduler.get_scheduler`。
    """

    # -- 类属性：子类必须覆写 ----------------------------------------------
    name: str = "base"
    spec: TTSBackendSpec = TTSBackendSpec(name="base")

    def __init__(self, scheduler: Scheduler | None = None) -> None:
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._logger = logging.getLogger(f"mosaic.tts.backends.{self.name}")
        # 运行时配置（load 时填充）
        self._device: str = "cpu"
        self._dtype: str = "float32"
        # 四层管线（_build_pipeline 中实例化）
        self._text_frontend: Any = None
        self._acoustic_model: Any = None
        self._vocoder: Any = None
        self._stream_adapter: Any = None
        # 加载状态
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda", dtype: str = "float16") -> None:
        """加载模型权重并组装四层管线。

        通过 :class:`Scheduler` 解析设备与管理显存：显存不足时尝试释放
        其他已加载节点；仍不足则抛出 :class:`MemoryError`。

        Parameters
        ----------
        device:
            目标设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
        dtype:
            权重精度，如 ``"float16"`` / ``"float32"`` / ``"bfloat16"``。

        Raises
        ------
        MemoryError
            GPU 显存不足以加载本后端。
        ImportError
            缺少必要依赖。
        RuntimeError
            模型加载失败（附带友好提示）。
        """
        if self.is_loaded:
            self._logger.debug("Backend %r already loaded, skip.", self.name)
            return

        # 解析设备（无 GPU 时降级为 CPU）
        resolved_device = self._resolve_device(device)
        self._device = resolved_device
        self._dtype = dtype

        # GPU 模式下校验显存
        if resolved_device.startswith("cuda"):
            self._ensure_gpu_memory()

        self._logger.info(
            "Loading TTS backend %r on device=%s, dtype=%s ...",
            self.name,
            resolved_device,
            dtype,
        )
        try:
            self._build_pipeline()
        except ImportError as exc:
            self._destroy_pipeline()
            self.is_loaded = False
            raise ImportError(
                f"Failed to load TTS backend {self.name!r}: missing dependency. "
                f"{exc}. Please install the required packages and retry."
            ) from exc
        except MemoryError:
            self._destroy_pipeline()
            self.is_loaded = False
            raise
        except Exception as exc:  # noqa: BLE001
            self._destroy_pipeline()
            self.is_loaded = False
            raise RuntimeError(
                f"Failed to load TTS backend {self.name!r}: {exc}. "
                f"Check model paths, device availability, and dependencies."
            ) from exc

        self.is_loaded = True
        self._logger.info("TTS backend %r loaded successfully.", self.name)

    def unload(self) -> None:
        """释放模型权重与四层管线资源。"""
        if not self.is_loaded:
            self._logger.debug("Backend %r not loaded, skip unload.", self.name)
            return
        self._logger.info("Unloading TTS backend %r ...", self.name)
        try:
            self._destroy_pipeline()
        finally:
            self.is_loaded = False
            self._device = "cpu"
            self._logger.info("TTS backend %r unloaded.", self.name)

    # ------------------------------------------------------------------
    # 核心合成
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        language: str = "zh",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AudioData:
        """阻塞式合成完整语音。

        等待全部生成完毕后返回完整的 :class:`AudioData`。

        内部流程：``TextFrontend.tokenize`` → ``AcousticModel.generate``
        → ``Vocoder.decode``。

        Parameters
        ----------
        text:
            待合成文本。
        speaker:
            说话人标识。``None`` 使用默认说话人。

            .. note::

               同一个 ``speaker`` 参数在不同后端语义不同：

               * **ChatTTS**：speaker 嵌入字符串（由
                 :meth:`sample_random_speaker` / ``encode_speaker`` 生成的
                 Base16384 编码串）。
               * **Fish**：参考音频文件路径，或预编码的 codec token 张量。
               * **SoVITS / CosyVoice**：已缓存（``save_speaker``）的说话人
                 名称，或参考音频文件路径。

            基类 :meth:`_validate_speaker` 仅做统一类型校验（``str`` /
            张量 / ``AudioData`` / ``None``），不改变各后端实际语义。
        language:
            语言代码，默认 ``"zh"``。
        speed:
            语速倍率，``1.0`` 为正常语速。
        **kwargs:
            透传给文本前端与声学模型的额外参数。

        Returns
        -------
        AudioData
            合成结果，``metadata`` 含 ``backend``/``text``/``speaker``
            /``language``/``speed``/``duration`` 等字段。

        Raises
        ------
        RuntimeError
            后端未加载。
        ValueError
            文本为空。
        TypeError
            ``speaker`` 类型不被任何后端接受。
        """
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("synthesize requires a non-empty 'text' string.")
        # A3-1: 统一 speaker 类型校验（不改变各后端实际语义）
        self._validate_speaker(speaker)

        self._logger.info(
            "synthesize: backend=%s language=%s speaker=%s speed=%.2f text_len=%d",
            self.name,
            language,
            speaker,
            speed,
            len(text),
        )

        # Layer 1: 文本前端 —— tokenize
        tokens = self._text_frontend.tokenize(text, language=language, **kwargs)
        # Layer 2: 声学模型 —— 生成 mel / VQ
        acoustic = self._acoustic_model.generate(
            tokens, **self._acoustic_params(speaker, language, speed, kwargs)
        )
        # Layer 3: 声码器 —— 解码为波形
        waveform, sample_rate = self._decode_full(acoustic)

        duration = self._compute_duration(waveform, sample_rate)
        metadata: dict[str, Any] = {
            "backend": self.name,
            "text": text,
            "speaker": speaker,
            "language": language,
            "speed": speed,
            "duration": duration,
            "sample_rate": sample_rate,
            "streaming": False,
        }
        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    def synthesize_stream(
        self,
        text: str,
        speaker: str | None = None,
        language: str = "zh",
        speed: float = 1.0,
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> Iterator[AudioData]:
        """流式合成语音，逐块 yield :class:`AudioData`。

        若后端不支持流式（``spec.supports_streaming=False``），内部回退
        为 :meth:`synthesize` 一次性返回完整结果。

        流程：``TextFrontend.tokenize`` → ``AcousticModel.generate_stream``
        → ``Vocoder.decode_chunk`` → ``StreamAdapter.push/pop``。

        Parameters
        ----------
        text:
            待合成文本。
        speaker:
            说话人名称或 ID。
        language:
            语言代码，默认 ``"zh"``。
        speed:
            语速倍率。
        chunk_size:
            每个音频块的目标采样数，默认 ``4096``。当流式适配器支持重配
            时生效，否则使用适配器构造时设定的块大小。
        **kwargs:
            透传给文本前端与声学模型的额外参数。

        Yields
        ------
        AudioData
            逐块音频数据，``metadata`` 中 ``streaming=True``。
        """
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "synthesize_stream requires a non-empty 'text' string."
            )

        # 不支持流式：回退为一次性合成
        if not self.spec.supports_streaming:
            self._logger.info(
                "Backend %r does not support streaming; "
                "falling back to blocking synthesize.",
                self.name,
            )
            yield self.synthesize(
                text,
                speaker=speaker,
                language=language,
                speed=speed,
                **kwargs,
            )
            return

        self._logger.info(
            "synthesize_stream: backend=%s language=%s speaker=%s "
            "speed=%.2f chunk_size=%d text_len=%d",
            self.name,
            language,
            speaker,
            speed,
            chunk_size,
            len(text),
        )

        # Layer 4 预热：获取/创建流式会话并尝试配置块大小
        session = self._get_stream_session(chunk_size)

        # Layer 1: 文本前端 —— tokenize
        tokens = self._text_frontend.tokenize(text, language=language, **kwargs)

        try:
            # Layer 2 + 3 + 4: 流式生成 → 逐块解码 → 缓冲输出
            for acoustic_chunk in self._acoustic_model.generate_stream(
                tokens, **self._acoustic_params(speaker, language, speed, kwargs)
            ):
                # 流式取消：提前终止生成循环
                if session.is_cancelled is True:
                    self._logger.info(
                        "synthesize_stream cancelled for backend %s",
                        self.name,
                    )
                    break
                waveform_chunk, _sample_rate = self._decode_chunk(acoustic_chunk)
                # 推入缓冲区
                self._stream_push(session, waveform_chunk)
                # 弹出已凑齐的块
                yield from self._stream_drain(
                    session, text, speaker, language, speed
                )

            # 冲刷缓冲区中剩余数据
            yield from self._stream_finish(
                session, text, speaker, language, speed
            )
        finally:
            # 确保会话缓冲与声学模型 KV cache 被释放/重置（即使中途抛异常）
            self._cleanup_stream_state(session)

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------
    def list_speakers(self) -> list[str]:
        """返回可用说话人列表，默认返回空列表。

        子类可覆写以返回真实说话人清单。
        """
        return []

    def describe(self) -> TTSBackendSpec:
        """返回本后端的规格描述。"""
        return self.spec

    # ------------------------------------------------------------------
    # 内部抽象方法（子类实现）
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _build_pipeline(self) -> None:
        """组装四层管线。

        子类需在此方法中实例化并赋值：

        * ``self._text_frontend``   —— :class:`TextFrontend`
        * ``self._acoustic_model``  —— :class:`AcousticModel`
        * ``self._vocoder``         —— :class:`Vocoder`
        * ``self._stream_adapter``  —— :class:`StreamAdapter`

        可使用 ``self._device`` / ``self._dtype`` 指定加载设备与精度。
        """

    def _destroy_pipeline(self) -> None:
        """销毁四层管线并释放资源。

        默认实现：依次调用各层的 ``unload_weights()``（若存在），随后置
        ``None``。子类可覆写以执行更精细的清理。
        """
        layers = [
            ("text_frontend", self._text_frontend),
            ("acoustic_model", self._acoustic_model),
            ("vocoder", self._vocoder),
            ("stream_adapter", self._stream_adapter),
        ]
        for layer_name, layer in layers:
            if layer is None:
                continue
            unload = getattr(layer, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "Error while unloading %s weights: %s",
                        layer_name,
                        exc,
                    )
        self._text_frontend = None
        self._acoustic_model = None
        self._vocoder = None
        self._stream_adapter = None

    # ------------------------------------------------------------------
    # 内部辅助：状态与设备
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """检查后端是否已加载，未加载则抛出 RuntimeError。"""
        if not self.is_loaded:
            raise RuntimeError(
                f"TTS backend {self.name!r} is not loaded. "
                f"Call load() before synthesis."
            )

    def _validate_speaker(self, speaker: Any) -> None:
        """统一校验 ``speaker`` 参数类型（A3-1）。

        不同后端对 ``speaker`` 的语义不同（见 :meth:`synthesize` 文档），
        但公共接口统一要求其为 ``None`` / ``str`` / 张量
        （``numpy.ndarray`` 或 ``torch.Tensor``）/ 序列（``list``/``tuple``
        ，如 Fish 的 codec token ids）/ :class:`AudioData` 之一。

        此处仅做类型校验，给出清晰错误；不改变各后端的实际语义。子类的
        ``synthesize`` 在文本校验后应调用本方法。
        """
        if speaker is None or isinstance(speaker, str):
            return
        # 允许序列类型（Fish clone_voice 透传的 codec token id 列表）
        if isinstance(speaker, (list, tuple)):
            return
        # 允许张量类型（Fish 的 codec token ids / 预编码嵌入）
        try:
            import numpy as np  # type: ignore

            if isinstance(speaker, np.ndarray):
                return
        except ImportError:
            pass
        try:
            import torch  # type: ignore

            if isinstance(speaker, torch.Tensor):
                return
        except ImportError:
            pass
        # 兼容 AudioData（部分语音克隆路径透传参考音频对象）
        if hasattr(speaker, "waveform") and hasattr(speaker, "sample_rate"):
            return
        raise TypeError(
            f"speaker must be str, a sequence (list/tuple), a tensor "
            f"(numpy.ndarray/torch.Tensor), AudioData, or None; "
            f"got {type(speaker).__name__}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析目标设备，无 GPU 时降级为 CPU。"""
        if device.startswith("cuda"):
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _ensure_gpu_memory(self) -> None:
        """确保有足够 GPU 显存，不足时尝试释放其他已加载节点。"""
        required = self.spec.min_gpu_memory_gb
        if required <= 0:
            return
        if self._has_enough_gpu_memory(required):
            return
        # 显存不足：尝试释放其他已加载节点
        self._logger.warning(
            "GPU memory insufficient for %r (need %.2fGB); "
            "attempting to release other loaded nodes.",
            self.name,
            required,
        )
        try:
            self._scheduler.release_all()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("release_all() failed: %s", exc)
        if not self._has_enough_gpu_memory(required):
            status = self._scheduler.status()
            raise MemoryError(
                f"Cannot allocate {required:.2f}GB GPU memory for TTS backend "
                f"{self.name!r}. Available is insufficient "
                f"(used={status.get('memory_used_gb', 0.0):.2f}GB, "
                f"limit={status.get('memory_limit_gb', 0.0):.2f}GB). "
                f"Try unloading other models or using CPU."
            )

    def _has_enough_gpu_memory(self, required_gb: float) -> bool:
        """判断当前可用显存是否满足需求。"""
        status = self._scheduler.status()
        limit = float(status.get("memory_limit_gb", 0.0) or 0.0)
        used = float(status.get("memory_used_gb", 0.0) or 0.0)
        if limit <= 0:
            # 无法计量（CPU 模式或查询失败）：放宽限制，交由实际加载校验
            return True
        available = limit - used
        return available + 1e-9 >= required_gb

    # ------------------------------------------------------------------
    # 内部辅助：声学参数与声码器解码
    # ------------------------------------------------------------------
    @staticmethod
    def _acoustic_params(
        speaker: str | None,
        language: str,
        speed: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """构造声学模型调用参数。

        将 ``speaker``/``language``/``speed`` 等高层参数与用户额外参数合并，
        通过 ``**kwargs`` 透传给声学模型的 ``generate`` / ``generate_stream``。
        """
        params: dict[str, Any] = dict(extra)
        params.setdefault("speaker", speaker)
        params.setdefault("language", language)
        params.setdefault("speed", speed)
        return params

    def _decode_full(self, acoustic: Any) -> tuple[Any, int]:
        """阻塞解码：调用 ``Vocoder.decode`` 并归一化输出。"""
        result = self._vocoder.decode(acoustic)
        return self._coerce_vocoder_output(result)

    def _decode_chunk(self, acoustic_chunk: Any) -> tuple[Any, int]:
        """流式解码：优先调用 ``Vocoder.decode_chunk`` 并归一化输出。"""
        decode_chunk = getattr(self._vocoder, "decode_chunk", None)
        if callable(decode_chunk):
            result = decode_chunk(acoustic_chunk)
        else:
            result = self._vocoder.decode(acoustic_chunk)
        return self._coerce_vocoder_output(result)

    def _coerce_vocoder_output(self, result: Any) -> tuple[Any, int]:
        """归一化声码器输出为 ``(waveform, sample_rate)``。

        兼容两种返回形式：

        * ``(waveform, sample_rate)`` 元组/列表；
        * 单一波形（采样率取自 ``Vocoder.sample_rate`` 或 ``spec.sample_rate``）。
        """
        if isinstance(result, (tuple, list)) and len(result) == 2:
            waveform, sample_rate = result
            if isinstance(sample_rate, (int, float)) and sample_rate > 0:
                return waveform, int(sample_rate)
            return waveform, self._vocoder_sample_rate()
        return result, self._vocoder_sample_rate()

    def _vocoder_sample_rate(self) -> int:
        """获取声码器输出采样率。"""
        sr = getattr(self._vocoder, "sample_rate", None)
        if isinstance(sr, (int, float)) and sr > 0:
            return int(sr)
        return self.spec.sample_rate

    def _compute_duration(self, waveform: Any, sample_rate: int) -> float:
        """计算波形时长（秒），兼容 numpy / torch / list。"""
        if sample_rate <= 0:
            return 0.0
        # 优先用 shape[-1]（numpy.ndarray / torch.Tensor 均适用）
        shape = getattr(waveform, "shape", None)
        if shape:
            try:
                n = int(shape[-1])
                if n > 0:
                    return float(n) / float(sample_rate)
            except (TypeError, IndexError):
                pass
        try:
            n = len(waveform)  # type: ignore[arg-type]
            if n > 0:
                return float(n) / float(sample_rate)
        except TypeError:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # 内部辅助：流式适配
    # ------------------------------------------------------------------
    def _get_stream_session(self, chunk_size: int) -> Any:
        """获取或创建流式会话，并尝试配置块大小。

        兼容两种适配器形态：

        * 工厂形态：``StreamAdapter.create_stream() -> StreamSession``
        * 直连形态：适配器自身提供 ``push``/``pop``/``flush``
        """
        adapter = self._stream_adapter
        if adapter is None:
            return None
        create = getattr(adapter, "create_stream", None)
        if callable(create):
            try:
                session = create()
            except TypeError:
                session = create(0)
        else:
            session = adapter
        self._try_set_chunk_size(session, chunk_size)
        return session

    def _cleanup_stream_state(self, session: Any) -> None:
        """流式合成结束后清理会话状态与声学模型 KV cache。

        应在 ``synthesize_stream`` 的 ``finally`` 块中调用，确保即使
        流式生成中途抛异常，``StreamSession`` 缓冲与声学模型 KV cache
        也能被释放/重置，避免资源泄漏与状态污染。

        对 ``session`` 与 ``_acoustic_model`` 均采用鸭子类型：仅当对象
        提供相应方法时才调用，缺失时静默跳过。
        """
        # 1. 清理会话缓冲
        if session is not None:
            reset = getattr(session, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("session.reset() failed: %s", exc)
            else:
                # 退而求其次：尝试 flush 清空缓冲
                flush = getattr(session, "flush", None)
                if callable(flush):
                    try:
                        flush()
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug("session.flush() failed: %s", exc)
        # 2. 清理声学模型 KV cache（如果模型支持）
        model = self._acoustic_model
        if model is not None:
            reset_cache = getattr(model, "reset_cache", None)
            if callable(reset_cache):
                try:
                    reset_cache()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "acoustic_model.reset_cache() failed: %s", exc
                    )

    def _try_set_chunk_size(self, session: Any, chunk_size: int) -> None:
        """尝试为流式会话配置块大小（不支持时静默跳过）。"""
        if session is None:
            return
        set_size = getattr(session, "set_chunk_size", None)
        if callable(set_size):
            try:
                set_size(chunk_size)
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("set_chunk_size() failed: %s", exc)
        reset = getattr(session, "reset", None)
        if callable(reset):
            try:
                reset(chunk_size=chunk_size)
                return
            except TypeError:
                try:
                    reset()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("reset() failed: %s", exc)
        # 退而求其次：设置公开属性（若存在）
        if hasattr(session, "chunk_size"):
            try:
                session.chunk_size = chunk_size
            except (AttributeError, TypeError):
                pass

    def _stream_push(self, session: Any, waveform: Any) -> None:
        """向流式会话推入一段波形，兼容单参/带 sample_rate 的签名。"""
        if session is None:
            return
        push = getattr(session, "push", None)
        if not callable(push):
            return
        # 通过签名判断是否接受 sample_rate 关键字
        accepts_sample_rate = self._accepts_kwarg(push, "sample_rate")
        if accepts_sample_rate:
            try:
                push(waveform, sample_rate=self._vocoder_sample_rate())
                return
            except TypeError:
                pass
        try:
            push(waveform)
        except TypeError:
            push(waveform, sample_rate=self._vocoder_sample_rate())

    def _stream_drain(
        self,
        session: Any,
        text: str,
        speaker: str | None,
        language: str,
        speed: float,
    ) -> Iterator[AudioData]:
        """从流式会话弹出已凑齐的块。"""
        if session is None:
            return
        pop = getattr(session, "pop", None)
        if not callable(pop):
            return
        while True:
            piece = pop()
            if piece is None:
                break
            yield self._wrap_stream_chunk(
                piece, text, speaker, language, speed
            )

    def _stream_finish(
        self,
        session: Any,
        text: str,
        speaker: str | None,
        language: str,
        speed: float,
    ) -> Iterator[AudioData]:
        """冲刷流式会话中剩余的缓冲数据。"""
        if session is None:
            return
        flush = getattr(session, "flush", None)
        if callable(flush):
            result = flush()
            for piece in self._iter_flush_result(result):
                yield self._wrap_stream_chunk(
                    piece, text, speaker, language, speed
                )
            return
        # 无 flush：循环 pop 直到缓冲为空
        pop = getattr(session, "pop", None)
        if callable(pop):
            while True:
                piece = pop()
                if piece is None:
                    break
                yield self._wrap_stream_chunk(
                    piece, text, speaker, language, speed
                )

    @staticmethod
    def _iter_flush_result(result: Any) -> Iterator[Any]:
        """归一化 ``flush()`` 返回值为片段迭代器。

        兼容返回 ``None`` / 单个 :class:`AudioData` / 可迭代片段集合。
        """
        if result is None:
            return
        if isinstance(result, AudioData):
            yield result
            return
        try:
            for piece in result:
                yield piece
        except TypeError:
            # 非可迭代：视为单个片段
            yield result

    def _wrap_stream_chunk(
        self,
        piece: Any,
        text: str,
        speaker: str | None,
        language: str,
        speed: float,
    ) -> AudioData:
        """将流式会话返回的片段包装为 :class:`AudioData`。"""
        # 已经是 AudioData：补充元数据后返回
        if isinstance(piece, AudioData):
            metadata = dict(piece.metadata)
            metadata.setdefault("backend", self.name)
            metadata.setdefault("text", text)
            metadata.setdefault("speaker", speaker)
            metadata.setdefault("language", language)
            metadata.setdefault("streaming", True)
            metadata["speed"] = speed
            return AudioData(
                waveform=piece.waveform,
                sample_rate=piece.sample_rate,
                metadata=metadata,
            )
        # piece 为 (waveform, sample_rate) 元组或单纯 waveform
        sample_rate: int = self._vocoder_sample_rate()
        waveform: Any = piece
        if isinstance(piece, tuple) and len(piece) == 2:
            waveform, sr = piece
            if isinstance(sr, (int, float)) and sr > 0:
                sample_rate = int(sr)
        duration = self._compute_duration(waveform, sample_rate)
        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata={
                "backend": self.name,
                "text": text,
                "speaker": speaker,
                "language": language,
                "speed": speed,
                "duration": duration,
                "streaming": True,
            },
        )

    @staticmethod
    def _accepts_kwarg(func: Any, kwarg: str) -> bool:
        """判断函数是否接受指定关键字参数（或 ``**kwargs``）。"""
        import inspect

        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if kwarg in params:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        return f"<{self.__class__.__name__} name={self.name!r} state={state}>"
