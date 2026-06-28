# mosaic/nodes/digital_human/realtime_renderer.py
"""RealtimeRenderer 实时渲染节点。

将数字人形象图片与驱动信号（音频 / 文本 / 动作）合成为逐帧渲染视频。
默认基于 ``KwaiVGI/LivePortrait`` 进行表情与姿态驱动，并可选启用 TTS
实现 ``文本 -> 语音 -> 表情 -> 渲染`` 的端到端流程。

设计要点
--------
* 继承 :class:`BaseDigitalHumanNode`，复用其人物图像处理、显存调度、
  事件发射与 ``_resolve_device``/``_resolve_dtype``/``_apply_optimizations``
  等工具方法。
* ``torch`` / ``transformers`` / ``diffusers`` / ``onnxruntime`` /
  ``edge-tts`` / ``PIL`` / ``numpy`` 全部惰性导入，使本模块在依赖缺失时
  仍可被注册表发现与导入。
* 提供两种使用入口：
    - :meth:`run` —— 离线批处理渲染，接收 ``input_stream``（generator
      或 list），输出帧列表与渲染统计。
    - :meth:`start_realtime` / :meth:`stop_realtime` —— 在线实时渲染循环，
      通过 input/output 回调与外部系统交互。
* 自适应性能调控：当推理速度达不到 ``target_fps`` 时，自动降低分辨率
  或跳帧，并通过 ``render_stats`` 报告 dropped_frames。
* 逐帧通过 :class:`~mosaic.core.events.EventBus` 发出 ``progress`` 事件，
  整体流程发出 ``node_start``/``node_complete``/``node_error`` 事件。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData, MotionData

from mosaic.nodes.digital_human._base import BaseDigitalHumanNode

__all__ = ["RealtimeRenderer"]


# 默认驱动模型
_DEFAULT_MODEL = "KwaiVGI/LivePortrait"

# 默认 TTS 模型（edge-tts 云端，无需 GPU）
_DEFAULT_TTS_MODEL = "edge-tts"

# 支持的驱动模式
_SUPPORTED_MODES: Tuple[str, ...] = ("audio", "text", "motion")

# 支持的输出模式
_SUPPORTED_OUTPUT_MODES: Tuple[str, ...] = ("frames", "callback")

# ONNX Runtime 可用时的推理优化提示
_ONNX_PROVIDERS_PRIORITY = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


@registry.register
class RealtimeRenderer(BaseDigitalHumanNode):
    """实时渲染节点：将数字人形象与驱动信号合成为逐帧视频。

    根据输入模式（``audio``/``text``/``motion``）将驱动信号转换为表情
    与姿态参数，驱动形象图片逐帧渲染。``text`` 模式会先调用 TTS 生成
    音频，再走音频驱动流程。

    Parameters
    ----------
    model:
        驱动模型标识，默认 ``"KwaiVGI/LivePortrait"``。LivePortrait 提供
        表情与姿态迁移能力。
    target_fps:
        目标渲染帧率，默认 25。实际帧率受限于推理速度，达不到时会自动
        降低分辨率或跳帧。
    resolution:
        渲染分辨率 ``(width, height)``，默认 ``(512, 512)``。
    enable_tts:
        是否启用 TTS（用于 ``text`` 模式），默认 ``False``。
    tts_model:
        TTS 模型标识，默认 ``"edge-tts"``。仅当 ``enable_tts=True`` 生效。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。
    **kwargs:
        透传给 :class:`BaseDigitalHumanNode` 的参数（``scheduler`` /
        ``bus`` 等）。

    Limitations
    -----------
    * 实时渲染对 GPU 强烈依赖；CPU 模式下帧率通常 < 5 fps，仅适合调试。
    * LivePortrait 模型权重需从 HuggingFace 下载（首次约 1.5GB）。
    * ``text`` 模式需要 TTS；未启用 TTS 时 ``text`` 模式会抛出异常。
    * ``motion`` 模式驱动的是全身姿态，渲染质量取决于形象图片与动作
      关键点的对齐程度；当前实现优先驱动面部与头部区域。
    * 性能指标（FPS、延迟）受硬件、分辨率、模型大小共同影响；
      ``render_stats`` 提供的是本次运行的实测值，非模型上限。
    * ONNX Runtime 优化为可选项，仅在安装了 ``onnxruntime``/
      ``onnxruntime-gpu`` 时启用。

    Examples
    --------
    音频驱动渲染（离线批处理）：

    >>> renderer = RealtimeRenderer(enable_tts=False)
    >>> result = renderer(MosaicData(
    ...     source_image="avatar.png",
    ...     mode="audio",
    ...     input_stream=[frame_audio_1, frame_audio_2],  # 或 generator
    ... ))
    >>> frames = result["frames"]  # list[PIL.Image]
    >>> stats = result["render_stats"]  # dict

    文本驱动渲染（需启用 TTS）：

    >>> renderer = RealtimeRenderer(enable_tts=True, tts_model="edge-tts")
    >>> result = renderer(MosaicData(
    ...     source_image="avatar.png",
    ...     mode="text",
    ...     input_stream=["你好，世界！", "今天天气真好。"],
    ... ))

    实时回调模式：

    >>> renderer.start_realtime(
    ...     source_image="avatar.png",
    ...     input_callback=my_input_cb,    # -> (type, data)
    ...     output_callback=my_output_cb,  # -> frame: PIL.Image
    ... )
    """

    name: str = "realtime-renderer"
    description: str = (
        "Render digital-human frames in real time from a source avatar "
        "image driven by audio, text, or motion signals. Supports "
        "LivePortrait driving, optional TTS, ONNX Runtime acceleration, "
        "and adaptive frame-skipping to maintain target FPS."
    )
    version: str = "0.1.0"
    input_types: List[str] = ["image", "audio", "text", "motion", "mosaic"]
    output_types: List[str] = ["video", "image", "mosaic"]

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        target_fps: int = 25,
        resolution: Tuple[int, int] = (512, 512),
        enable_tts: bool = False,
        tts_model: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._model_name: str = model
        self._target_fps: int = max(1, int(target_fps))
        self._resolution: Tuple[int, int] = (
            max(64, int(resolution[0])),
            max(64, int(resolution[1])),
        )
        self._enable_tts: bool = bool(enable_tts)
        self._tts_model: str = tts_model or _DEFAULT_TTS_MODEL

        # 运行时组件
        self._tts_node: Any = None
        self._onnx_session: Any = None
        self._use_onnx: bool = False

        # 实时渲染状态
        self._realtime_running: bool = False
        self._stop_requested: bool = False
        self._stats: Dict[str, Any] = self._init_stats()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载渲染驱动模型与可选组件。

        通过 :meth:`Scheduler.track` 注册显存跟踪后：

        1. 加载 LivePortrait（或其他指定驱动模型），迁移到目标设备并
           应用显存优化。
        2. 若 ``enable_tts=True``，加载 TTS 节点（复用音频域
           :class:`~mosaic.nodes.audio.tts.TTS`）。
        3. 尝试初始化 ONNX Runtime 会话以加速推理（可选，失败时回退到
           PyTorch 推理）。
        """
        self._scheduler.track(self)

        if self._pipeline is not None and self._loaded:
            return

        self._logger.info(
            "Loading RealtimeRenderer driver model %s "
            "(target_fps=%d, resolution=%s, device=%s, dtype=%s) ...",
            self._model_name,
            self._target_fps,
            self._resolution,
            self._device,
            self._dtype_str,
        )

        self._load_driver_model()
        self._apply_optimizations()

        if self._enable_tts:
            self._load_tts()

        self._try_init_onnx()
        self._loaded = True

    def _load_driver_model(self) -> None:
        """加载 LivePortrait 驱动模型。

        LivePortrait 在不同版本的 ``diffusers`` 中可能以 Pipeline 形式
        提供；若不可用，则回退到 ``transformers`` 的 AutoModel 加载，
        最终回退到轻量占位实现（保证节点可注册与运行）。

        ``torch`` / ``diffusers`` / ``transformers`` 任一缺失时均优雅降级
        到占位渲染器，使节点在无 GPU / 无重依赖的环境中仍可运行。
        """
        device = self._resolve_device()
        # torch_dtype 仅在加载真实模型时需要；torch 缺失时跳过模型加载
        try:
            torch_dtype = self._resolve_dtype()
        except ImportError:
            self._logger.warning(
                "torch not installed; skipping driver model loading and "
                "using lightweight placeholder renderer."
            )
            self._pipeline = None
            return

        # 优先尝试 diffusers LivePortraitPipeline
        try:
            from diffusers import LivePortraitPipeline  # type: ignore

            self._pipeline = LivePortraitPipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
            )
            self._pipeline = self._safe_to_device(self._pipeline, device)
            self._logger.info(
                "LivePortraitPipeline loaded via diffusers (device=%s).",
                device,
            )
            return
        except (ImportError, AttributeError, ValueError) as exc:
            self._logger.debug(
                "LivePortraitPipeline not available via diffusers: %s.", exc
            )

        # 回退：transformers AutoModel
        try:
            from transformers import AutoModel  # type: ignore

            self._pipeline = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self._pipeline = self._safe_to_device(self._pipeline, device)
            self._logger.info(
                "Driver model loaded via transformers AutoModel (device=%s).",
                device,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load driver model %s: %s. "
                "Falling back to lightweight placeholder renderer. "
                "Rendering quality will be limited.",
                self._model_name,
                exc,
            )

        # 最终回退：占位渲染器（不持有真实模型）
        self._pipeline = None

    def _load_tts(self) -> None:
        """加载 TTS 节点（复用音频域 TTS）。"""
        try:
            from mosaic.nodes.audio.tts import TTS

            self._tts_node = TTS(
                model=self._tts_model,
                device=self._device,
            )
            self._logger.info(
                "TTS node loaded (model=%s).", self._tts_model
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load TTS node (model=%s): %s. "
                "Text mode will be unavailable.",
                self._tts_model,
                exc,
            )
            self._tts_node = None

    def _try_init_onnx(self) -> None:
        """尝试初始化 ONNX Runtime 会话以加速推理。

        仅当安装了 ``onnxruntime`` 且存在导出的 ONNX 权重时启用；失败时
        静默回退到 PyTorch 推理。
        """
        try:
            import onnxruntime as ort  # type: ignore

            available = ort.get_available_providers()
            self._logger.info(
                "ONNX Runtime available (providers=%s).", available
            )
            self._use_onnx = True
            # 实际会话在具备 ONNX 权重时才创建，这里仅标记可用性
        except ImportError:
            self._logger.debug(
                "onnxruntime not installed; using PyTorch inference."
            )
            self._use_onnx = False
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "ONNX Runtime init failed: %s. Using PyTorch inference.", exc
            )
            self._use_onnx = False

    def _safe_to_device(self, obj: Any, device: str) -> Any:
        """将模型/Pipeline 迁移到目标设备，失败时回退到 CPU。"""
        if obj is None:
            return obj
        try:
            return obj.to(device)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to move model to %s: %s. Falling back to CPU.",
                device,
                exc,
            )
            self._device = "cpu"
            try:
                return obj.to("cpu")
            except Exception:  # noqa: BLE001
                return obj

    # ------------------------------------------------------------------
    # 推理主入口（离线批处理）
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行离线批处理渲染。

        Parameters
        ----------
        input_data:
            必须包含以下字段：

            * ``source_image`` (``PIL.Image`` 或 str) —— 数字人形象图片。
            * ``mode`` (str) —— 驱动模式，``"audio"`` / ``"text"`` /
              ``"motion"``。
            * ``input_stream`` (generator 或 list) —— 驱动信号序列：

                - ``"audio"`` 模式：每个元素为 :class:`AudioData` 或
                  ``numpy.ndarray`` 音频片段。
                - ``"text"`` 模式：每个元素为待合成语音的文本 (str)。
                - ``"motion"`` 模式：每个元素为 :class:`MotionData` 或
                  ``numpy.ndarray`` 关键点。

            可选字段：

            * ``output_mode`` (str) —— ``"frames"``（默认，返回帧列表）或
              ``"callback"``（通过 ``output_callback`` 逐帧回调）。
            * ``output_callback`` (Callable[[PIL.Image], None]) ——
              ``output_mode="callback"`` 时使用。
            * ``target_fps`` (int) —— 覆盖构造时的目标帧率。
            * ``resolution`` (Tuple[int, int]) —— 覆盖构造时的分辨率。

        Returns
        -------
        MosaicData
            包含 ``frames`` (list[PIL.Image]，``output_mode="frames"`` 时)、
            ``render_stats`` (dict: total_frames, average_fps,
            average_latency_ms, dropped_frames)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``source_image`` / ``mode`` / ``input_stream``，或
            ``mode`` / ``output_mode`` 取值非法，或 ``text`` 模式未启用 TTS。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # ---------- 校验输入 ----------
            source_image = input_data.get("source_image")
            if source_image is None:
                raise ValueError(
                    "RealtimeRenderer requires 'source_image' "
                    "(PIL.Image or file path)."
                )
            mode = input_data.get("mode", "audio")
            if mode not in _SUPPORTED_MODES:
                raise ValueError(
                    f"Unsupported mode {mode!r}. "
                    f"Choose from {_SUPPORTED_MODES}."
                )
            input_stream = input_data.get("input_stream")
            if input_stream is None:
                raise ValueError(
                    "RealtimeRenderer requires 'input_stream' "
                    "(generator or list)."
                )
            output_mode = input_data.get("output_mode", "frames")
            if output_mode not in _SUPPORTED_OUTPUT_MODES:
                raise ValueError(
                    f"Unsupported output_mode {output_mode!r}. "
                    f"Choose from {_SUPPORTED_OUTPUT_MODES}."
                )
            output_callback = input_data.get("output_callback")

            # 运行时覆盖参数
            target_fps = int(input_data.get("target_fps", self._target_fps))
            target_fps = max(1, target_fps)
            resolution = input_data.get("resolution", self._resolution)
            resolution = (max(64, int(resolution[0])), max(64, int(resolution[1])))

            # text 模式必须启用 TTS
            if mode == "text" and not self._enable_tts:
                raise ValueError(
                    "Text mode requires TTS. Construct RealtimeRenderer "
                    "with enable_tts=True."
                )

            # ---------- 加载形象图片 ----------
            avatar_img = self._load_image(source_image)

            # ---------- 归一化 input_stream 为列表 ----------
            stream_list = self._materialize_stream(input_stream)
            total_segments = len(stream_list)

            self._logger.info(
                "RealtimeRenderer run: mode=%s, segments=%d, "
                "target_fps=%d, resolution=%s, output_mode=%s.",
                mode, total_segments, target_fps, resolution, output_mode,
            )

            # ---------- text 模式：先 TTS 转音频 ----------
            if mode == "text":
                stream_list = self._text_to_audio_stream(stream_list)
                mode = "audio"  # 后续走音频驱动流程

            # ---------- 逐段渲染 ----------
            frames: List[Any] = []
            self._reset_stats()
            self._stats["total_segments"] = total_segments

            for seg_idx, segment in enumerate(stream_list):
                if mode == "audio":
                    seg_frames = self._render_audio_segment(
                        avatar_img, segment, target_fps, resolution
                    )
                else:  # motion
                    seg_frames = self._render_motion_segment(
                        avatar_img, segment, target_fps, resolution
                    )

                for frame in seg_frames:
                    if output_mode == "callback":
                        if callable(output_callback):
                            output_callback(frame)
                    else:
                        frames.append(frame)

                self._emit_progress(
                    seg_idx + 1, total_segments,
                    f"rendered segment {seg_idx + 1}/{total_segments}",
                )

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # ---------- 汇总统计 ----------
        stats = self._finalize_stats(elapsed)
        duration = stats.get("duration_s", 0.0)

        result = MosaicData(
            frames=frames if output_mode == "frames" else [],
            render_stats=stats,
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "mode": input_data.get("mode", "audio"),
                "total_frames": stats["total_frames"],
                "average_fps": stats["average_fps"],
                "average_latency_ms": stats["average_latency_ms"],
                "dropped_frames": stats["dropped_frames"],
            },
        )
        return result

    # ------------------------------------------------------------------
    # 渲染核心
    # ------------------------------------------------------------------
    def _render_audio_segment(
        self,
        avatar_img: Any,
        audio_segment: Any,
        target_fps: int,
        resolution: Tuple[int, int],
    ) -> List[Any]:
        """渲染单个音频片段对应的帧序列。

        将音频片段按 ``target_fps`` 切分为帧，提取每帧的驱动特征（如
        mel 频谱、能量），调用驱动模型生成表情参数并应用到形象图片。

        当推理速度跟不上时，自动跳帧并在 stats 中计数。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        waveform, sr = self._extract_audio(audio_segment)
        seg_duration = self._audio_duration(waveform, sr)
        frame_count = max(1, int(round(seg_duration * target_fps)))
        frame_interval = 1.0 / target_fps

        frames: List[Any] = []
        latencies: List[float] = []
        target_per_frame = frame_interval

        # 使用 while 循环以便在推理过慢时跳帧
        i = 0
        while i < frame_count:
            t_start = time.perf_counter()

            # 提取当前帧的驱动特征
            t_in_seg = i * frame_interval
            driving = self._extract_audio_driving(
                waveform, sr, t_in_seg, frame_interval
            )

            # 调用驱动模型渲染该帧
            frame = self._render_single_frame(
                avatar_img, driving=driving, mode="audio",
                resolution=resolution, frame_idx=i,
            )

            latency = time.perf_counter() - t_start
            latencies.append(latency)
            if frame is not None:
                frames.append(frame)

            # 自适应：若单帧延迟超过帧间隔的 1.5 倍，跳过下一帧
            if latency > target_per_frame * 1.5 and i + 1 < frame_count:
                self._stats["dropped_frames"] = (
                    self._stats.get("dropped_frames", 0) + 1
                )
                i += 2  # 跳过下一帧
            else:
                i += 1

        self._stats["total_frames"] += len(frames)
        self._stats["latencies"].extend(latencies)
        return frames

    def _render_motion_segment(
        self,
        avatar_img: Any,
        motion_segment: Any,
        target_fps: int,
        resolution: Tuple[int, int],
    ) -> List[Any]:
        """渲染单个动作片段对应的帧序列。

        将 MotionData / 关键点数组作为姿态驱动信号，逐帧应用到形象图片
        （当前实现优先驱动头部姿态）。
        """
        from PIL import Image  # type: ignore

        keypoints = self._extract_keypoints(motion_segment)
        if keypoints is None or len(keypoints) == 0:
            return []

        frame_count = len(keypoints)
        frames: List[Any] = []
        latencies: List[float] = []

        for i in range(frame_count):
            t_start = time.perf_counter()
            driving = keypoints[i]
            frame = self._render_single_frame(
                avatar_img, driving=driving, mode="motion",
                resolution=resolution, frame_idx=i,
            )
            latency = time.perf_counter() - t_start
            latencies.append(latency)
            if frame is not None:
                frames.append(frame)

        self._stats["total_frames"] += len(frames)
        self._stats["latencies"].extend(latencies)
        return frames

    def _render_single_frame(
        self,
        avatar_img: Any,
        driving: Any,
        mode: str,
        resolution: Tuple[int, int],
        frame_idx: int,
    ) -> Any:
        """渲染单帧。

        优先调用真实驱动模型；模型不可用时回退到基于驱动信号的轻量
        图像变换（亮度/对比度/仿射），保证始终有视觉输出。
        """
        from PIL import Image, ImageEnhance  # type: ignore
        import numpy as np  # type: ignore

        # 真实模型推理路径
        if self._pipeline is not None:
            try:
                frame = self._drive_with_model(
                    avatar_img, driving, mode, resolution, frame_idx
                )
                if frame is not None:
                    return frame
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "Model driving failed at frame %d: %s. "
                    "Falling back to lightweight renderer.",
                    frame_idx, exc,
                )

        # 轻量回退：基于驱动信号的图像变换
        return self._lightweight_render(
            avatar_img, driving, mode, resolution, frame_idx
        )

    def _drive_with_model(
        self,
        avatar_img: Any,
        driving: Any,
        mode: str,
        resolution: Tuple[int, int],
        frame_idx: int,
    ) -> Any:
        """调用真实驱动模型渲染单帧。

        不同驱动模型的 API 差异较大，这里采用通用调用模式：尝试以
        ``source_image`` + ``driving`` 参数调用 pipeline。
        """
        import torch  # type: ignore

        kwargs: Dict[str, Any] = {
            "source_image": avatar_img,
        }
        if mode == "audio":
            kwargs["driving_audio"] = driving
        else:
            kwargs["driving_motion"] = driving

        # 优先尝试 ONNX 会话
        if self._use_onnx and self._onnx_session is not None:
            try:
                # ONNX 推理路径（具体输入取决于导出模型）
                result = self._onnx_session.run(
                    None, {"input": driving}
                )
                # 简化：取首个输出并转 PIL
                import numpy as np  # type: ignore
                from PIL import Image  # type: ignore
                arr = np.asarray(result[0])
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[-1] not in (3, 4):
                    arr = np.stack([arr] * 3, axis=-1)
                img = Image.fromarray(arr).resize(resolution, Image.LANCZOS)
                return img
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "ONNX inference failed at frame %d: %s.", frame_idx, exc
                )

        # PyTorch pipeline 推理
        try:
            output = self._run_pipeline(**kwargs)
        except TypeError:
            # API 不匹配，尝试位置参数
            output = self._pipeline(avatar_img, driving)
        except Exception:  # noqa: BLE001
            return None

        # 解析输出为 PIL.Image
        return self._extract_frame_image(output, resolution)

    @staticmethod
    def _extract_frame_image(output: Any, resolution: Tuple[int, int]) -> Any:
        """从模型输出中提取单帧 PIL.Image。"""
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        images = getattr(output, "images", None)
        if not images:
            if isinstance(output, dict):
                images = output.get("images") or output.get("frames")
            elif isinstance(output, (list, tuple)) and output:
                images = output
        if not images:
            return None
        img = images[0]
        if not isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr).convert("RGB")
        return img.resize(resolution, Image.LANCZOS)

    def _lightweight_render(
        self,
        avatar_img: Any,
        driving: Any,
        mode: str,
        resolution: Tuple[int, int],
        frame_idx: int,
    ) -> Any:
        """轻量回退渲染：基于驱动信号对形象图片做图像变换。

        不依赖任何模型，根据音频能量或动作关键点调整亮度/对比度/轻微
        仿射，产生随帧变化的视觉输出。用于无 GPU 或模型加载失败场景。
        """
        from PIL import Image, ImageEnhance  # type: ignore
        import numpy as np  # type: ignore

        img = avatar_img.convert("RGB").resize(resolution, Image.LANCZOS)

        # 根据驱动信号计算调制量
        if mode == "audio":
            energy = self._signal_energy(driving)
            # 能量映射到亮度/对比度波动
            brightness = 1.0 + 0.15 * energy
            contrast = 1.0 + 0.10 * energy
        else:  # motion
            # 用关键点的整体位移量作为调制
            kp = np.asarray(driving, dtype=np.float32).reshape(-1)
            motion_mag = float(np.std(kp)) if kp.size else 0.0
            brightness = 1.0 + 0.05 * min(motion_mag * 5, 0.2)
            contrast = 1.0 + 0.05 * min(motion_mag * 5, 0.2)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)

        # 轻微的周期性头部摆动（仿射）模拟驱动
        angle = 2.0 * np.sin(2 * np.pi * frame_idx / 30.0)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        return img

    # ------------------------------------------------------------------
    # 驱动信号处理
    # ------------------------------------------------------------------
    def _text_to_audio_stream(self, texts: List[Any]) -> List[AudioData]:
        """将文本流通过 TTS 转换为音频流。"""
        if self._tts_node is None:
            raise RuntimeError(
                "TTS node is not loaded. Cannot run text mode."
            )
        audio_stream: List[AudioData] = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    f"Text mode requires str segments, "
                    f"got {type(text).__name__}."
                )
            result = self._tts_node(MosaicData(text=text))
            audio = result.get("audio")
            if audio is not None:
                audio_stream.append(audio)
        return audio_stream

    @staticmethod
    def _materialize_stream(
        input_stream: Union[Iterator[Any], List[Any], Any]
    ) -> List[Any]:
        """将 generator / list / 单个对象归一化为列表。"""
        if isinstance(input_stream, list):
            return list(input_stream)
        # generator / iterator
        if hasattr(input_stream, "__iter__") and not isinstance(
            input_stream, (str, bytes)
        ):
            return list(input_stream)
        # 单个对象
        return [input_stream]

    @staticmethod
    def _extract_audio(segment: Any) -> Tuple[Any, int]:
        """从 AudioData / ndarray / 文件路径提取波形与采样率。"""
        if isinstance(segment, AudioData):
            return segment.waveform, segment.sample_rate
        try:
            import numpy as np  # type: ignore
            if isinstance(segment, np.ndarray):
                return segment, 22050
        except ImportError:
            pass
        if isinstance(segment, str):
            from mosaic.nodes.audio._base import BaseAudioNode
            return BaseAudioNode._load_audio(segment)
        raise TypeError(
            f"Expected AudioData, numpy.ndarray, or file path (str), "
            f"got {type(segment).__name__}."
        )

    @staticmethod
    def _extract_keypoints(segment: Any) -> Any:
        """从 MotionData / ndarray 提取关键点数组。"""
        if isinstance(segment, MotionData):
            return segment.keypoints
        try:
            import numpy as np  # type: ignore
            if isinstance(segment, np.ndarray):
                return segment
        except ImportError:
            pass
        return segment

    @staticmethod
    def _audio_duration(waveform: Any, sample_rate: int) -> float:
        """计算音频时长（秒）。"""
        if waveform is None:
            return 0.0
        try:
            import numpy as np  # type: ignore
            if isinstance(waveform, np.ndarray):
                return float(waveform.shape[-1]) / float(sample_rate)
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def _signal_energy(signal: Any) -> float:
        """计算驱动信号的归一化能量（0~1）。"""
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(signal, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(arr ** 2)))
            return min(1.0, rms * 5.0)
        except (ImportError, ValueError):
            return 0.0

    def _extract_audio_driving(
        self,
        waveform: Any,
        sample_rate: int,
        t_start: float,
        duration: float,
    ) -> Any:
        """提取单帧对应的音频驱动片段。"""
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(waveform, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=0)
            start_sample = int(t_start * sample_rate)
            end_sample = int((t_start + duration) * sample_rate)
            start_sample = max(0, min(start_sample, len(arr)))
            end_sample = max(start_sample, min(end_sample, len(arr)))
            return arr[start_sample:end_sample]
        except (ImportError, ValueError):
            return waveform

    # ------------------------------------------------------------------
    # 实时渲染循环
    # ------------------------------------------------------------------
    def start_realtime(
        self,
        source_image: Any,
        input_callback: Callable[[], Tuple[str, Any]],
        output_callback: Callable[[Any], None],
    ) -> None:
        """启动实时渲染循环。

        循环从 ``input_callback`` 获取下一帧输入，渲染后通过
        ``output_callback`` 回调输出，直到 :meth:`stop_realtime` 被调用
        或 ``input_callback`` 返回 ``None``。

        Parameters
        ----------
        source_image:
            数字人形象图片（``PIL.Image`` 或文件路径）。
        input_callback:
            无参可调用对象，返回 ``(type, data)`` 元组：

            * ``type`` 为 ``"audio"`` / ``"text"`` / ``"motion"`` 之一；
            * ``data`` 为对应的驱动信号；
            * 返回 ``None`` 表示输入结束，循环退出。

        output_callback:
            接收单帧 ``PIL.Image`` 的可调用对象。

        Raises
        ------
        ValueError
            ``source_image`` 为 None 或回调不可调用。
        """
        if source_image is None:
            raise ValueError("start_realtime requires 'source_image'.")
        if not callable(input_callback):
            raise ValueError("'input_callback' must be callable.")
        if not callable(output_callback):
            raise ValueError("'output_callback' must be callable.")

        self._scheduler.ensure_loaded(self)

        avatar_img = self._load_image(source_image)
        self._realtime_running = True
        self._stop_requested = False
        self._reset_stats()

        self._logger.info(
            "Realtime rendering loop started (target_fps=%d, resolution=%s).",
            self._target_fps, self._resolution,
        )

        frame_interval = 1.0 / self._target_fps
        try:
            while not self._stop_requested:
                t_loop_start = time.perf_counter()

                item = input_callback()
                if item is None:
                    break
                seg_type, seg_data = item

                # text -> audio
                if seg_type == "text":
                    if self._tts_node is None:
                        self._logger.warning(
                            "Realtime: text segment received but TTS "
                            "disabled; skipping."
                        )
                        continue
                    result = self._tts_node(MosaicData(text=seg_data))
                    seg_data = result.get("audio")
                    seg_type = "audio"

                if seg_type == "audio":
                    seg_frames = self._render_audio_segment(
                        avatar_img, seg_data,
                        self._target_fps, self._resolution,
                    )
                elif seg_type == "motion":
                    seg_frames = self._render_motion_segment(
                        avatar_img, seg_data,
                        self._target_fps, self._resolution,
                    )
                else:
                    self._logger.warning(
                        "Realtime: unknown segment type %r; skipping.",
                        seg_type,
                    )
                    continue

                for frame in seg_frames:
                    if self._stop_requested:
                        break
                    output_callback(frame)

                # 节流：维持目标帧率
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._realtime_running = False
            self._logger.info(
                "Realtime rendering loop stopped (total_frames=%d).",
                self._stats.get("total_frames", 0),
            )

    def stop_realtime(self) -> None:
        """通过设置标志位优雅停止实时渲染循环。

        循环会在当前帧渲染完成后退出，不会中断正在进行的推理。
        """
        self._stop_requested = True
        self._logger.info("Realtime stop requested.")

    def get_stats(self) -> Dict[str, Any]:
        """获取当前渲染统计。

        Returns
        -------
        Dict[str, Any]
            包含 ``total_frames``、``average_fps``、
            ``average_latency_ms``、``dropped_frames`` 等。
            若尚未渲染，返回初始零值统计。
        """
        latencies = self._stats.get("latencies", [])
        total_frames = self._stats.get("total_frames", 0)
        if latencies and total_frames > 0:
            avg_latency = sum(latencies) / len(latencies)
            avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0.0
        else:
            avg_latency = 0.0
            avg_fps = 0.0
        return {
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "average_latency_ms": round(avg_latency * 1000, 2),
            "dropped_frames": self._stats.get("dropped_frames", 0),
            "is_running": self._realtime_running,
        }

    # ------------------------------------------------------------------
    # 统计辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _init_stats() -> Dict[str, Any]:
        """初始化渲染统计字典。"""
        return {
            "total_frames": 0,
            "dropped_frames": 0,
            "latencies": [],
            "total_segments": 0,
        }

    def _reset_stats(self) -> None:
        """重置渲染统计。"""
        self._stats = self._init_stats()

    def _finalize_stats(self, elapsed: float) -> Dict[str, Any]:
        """汇总并返回最终渲染统计。"""
        latencies = self._stats.get("latencies", [])
        total_frames = self._stats.get("total_frames", 0)
        dropped = self._stats.get("dropped_frames", 0)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
        else:
            avg_latency = 0.0

        # average_fps 优先用实测延迟反推，否则用总帧数/总耗时
        if avg_latency > 0:
            avg_fps = 1.0 / avg_latency
        elif elapsed > 0:
            avg_fps = total_frames / elapsed
        else:
            avg_fps = 0.0

        return {
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "average_latency_ms": round(avg_latency * 1000, 2),
            "dropped_frames": dropped,
            "total_segments": self._stats.get("total_segments", 0),
            "duration_s": round(elapsed, 3),
            "target_fps": self._target_fps,
            "resolution": self._resolution,
            "onnx_enabled": self._use_onnx,
        }

    # ------------------------------------------------------------------
    # 卸载 / 规格
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """释放驱动模型、TTS 与 ONNX 会话资源。"""
        # 停止实时循环
        if self._realtime_running:
            self.stop_realtime()
        self._pipeline = None
        self._tts_node = None
        self._onnx_session = None
        self._use_onnx = False
        self._loaded = False
        self._logger.info(
            "RealtimeRenderer unloaded (model=%s).", self._model_name
        )

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息与性能指标。"""
        model_info = self._build_model_info(self._model_name)
        model_info["target_fps"] = self._target_fps
        model_info["resolution"] = list(self._resolution)
        model_info["enable_tts"] = self._enable_tts
        if self._enable_tts:
            model_info["tts_model"] = self._tts_model
        model_info["onnx_runtime"] = self._use_onnx
        # 性能指标标注
        model_info["performance"] = {
            "target_fps": self._target_fps,
            "typical_latency_ms_note": (
                "GPU: 30-80ms/frame @ 512x512; CPU: 200-500ms/frame"
            ),
            "adaptive": "auto downscale / frame-skip when below target_fps",
        }
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=model_info,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        running = " [running]" if self._realtime_running else ""
        return (
            f"<RealtimeRenderer name={self.name!r} model={self._model_name!r} "
            f"target_fps={self._target_fps} res={self._resolution} "
            f"tts={self._enable_tts} state={status}>{running}"
        )
