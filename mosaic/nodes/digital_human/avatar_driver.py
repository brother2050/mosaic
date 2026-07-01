# mosaic/nodes/digital_human/avatar_driver.py
"""AvatarDriver 节点 —— 数字人形象驱动。

根据一张源人物图片与驱动信号（驱动视频 / 驱动音频 / 表情参数序列），
生成数字人说话或表情动画视频。输出帧的背景与源图片保持一致，仅人脸
区域被驱动。

设计要点
--------
* 继承 :class:`BaseDigitalHumanNode`，复用人脸检测、对齐、裁剪、表情应用、
  面部融合等工具方法。
* 三种驱动方式（三选一）：
    - ``driving_video``：从驱动视频逐帧提取表情/运动，迁移到源图片；
    - ``driving_audio``：由音频生成表情序列（能量→口型）再驱动；
    - ``expression_params``：直接接收表情参数序列驱动。
* 三种后端模型（``method``）：
    - ``liveportrait``：外观编码 + 运动提取 + 扭曲 + SPADE 解码，显存约 4-6GB；
    - ``sadtalker``：3DMM 面部建模，显存约 4-6GB；
    - ``musetalk``：UNet 口型驱动，显存约 6-8GB。
* 所有第三方库（torch / diffusers / imageio / PIL / numpy）惰性导入，
  使本模块在依赖缺失时仍可被注册表发现与导入。
* 关键步骤通过事件总线发出 start / progress / complete / error 事件。
* 模型生命周期由 :class:`~mosaic.core.scheduler.Scheduler` 管理
  （按需加载 + LRU 淘汰）。

显存需求
--------
* ``KwaiVGI/LivePortrait``：约 4-6GB（fp16）
* ``cvitkwai/SadTalker``：约 4-6GB（fp16）
* ``KwaiVGI/MuseTalk``：约 6-8GB（fp16）

许可证
------
* LivePortrait：MIT License (model weights)
* SadTalker：Apache-2.0 (code), CC-BY-NC-4.0 (model weights)
* MuseTalk：CC-BY-NC 4.0

Limitations
-----------
* 真正的高质量驱动依赖对应模型库（liveportrait / sadtalker / musetalk），
  未安装时 ``load`` 会抛出带安装提示的 ``ImportError``。
* ``driving_audio`` + ``liveportrait`` 组合通过音频能量→口型的启发式桥接，
  表现力弱于原生音频驱动模型。
* ``driving_video`` / ``expression_params`` + ``sadtalker``/``musetalk`` 组合
  使用基类 :meth:`_apply_expression` 的简化表情应用（亮度/对比度调整），
  非模型原生逐帧驱动。
* 单张源图片需包含清晰可检测的人脸；检测不到人脸时抛出 ``ValueError``。
* 输出分辨率建议 ≤ 512x512 以控制显存与推理时间。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes._coerce import safe_float, safe_int
from mosaic.nodes.digital_human._base import BaseDigitalHumanNode

__all__ = ["AvatarDriver"]


# method -> 默认 HuggingFace 模型标识
_DEFAULT_MODELS: dict[str, str] = {
    "liveportrait": "KwaiVGI/LivePortrait",
    "sadtalker": "cvitkwai/SadTalker",
    "musetalk": "KwaiVGI/MuseTalk",
}

# method -> 粗略显存需求（GB, fp16），用于 describe()
_METHOD_VRAM: dict[str, float] = {
    "liveportrait": 5.0,
    "sadtalker": 5.0,
    "musetalk": 7.0,
}

# 输出默认参数
_DEFAULT_RESOLUTION: tuple[int, int] = (512, 512)
_DEFAULT_FPS: int = 25


@registry.register
class AvatarDriver(BaseDigitalHumanNode):
    """数字人形象驱动节点。

    根据源图片与驱动信号生成数字人表情/说话视频，背景与源图保持一致。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径，默认 ``"KwaiVGI/LivePortrait"``。
        当 ``method`` 改变而 ``model`` 仍为默认值时，自动按 ``method`` 解析
        对应默认模型；显式指定其他 ``model`` 时以用户指定为准。
    method:
        驱动方法，可选 ``"liveportrait"`` / ``"sadtalker"`` / ``"musetalk"``，
        默认 ``"liveportrait"``。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，默认 ``"float16"``。
    **kwargs:
        透传给 :class:`BaseDigitalHumanNode` 的参数（如 ``scheduler``/``bus``）。

    Examples
    --------
    >>> driver = AvatarDriver(method="liveportrait")
    >>> result = driver(MosaicData(
    ...     source_image="person.jpg",
    ...     driving_video="talk.mp4",
    ...     fps=25,
    ... ))
    >>> video = result["video"]  # VideoData
    """

    name: str = "avatar-driver"
    description: str = (
        "Drive a digital human avatar from a source image using a driving "
        "video, driving audio, or expression parameters. Supports LivePortrait, "
        "SadTalker and MuseTalk backends; output background matches the source."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["image", "video", "audio", "mosaic"]
    output_types: list[str] = ["video", "image", "mosaic"]

    # -- 表情参数推导的魔法数字（提取为类常量便于调参） -------------------
    #: 音频能量到嘴部张开度的缩放系数。
    ENERGY_SCALE: float = 3.0
    #: 嘴部张开代理值归一化的阈值（除数）。
    MOUTH_OPEN_THRESHOLD: float = 0.5
    #: 微笑判定时嘴宽 / 眼距的偏移量。
    MOUTH_WIDTH_OFFSET: float = 0.4

    def __init__(
        self,
        model: str = "KwaiVGI/LivePortrait",
        method: str = "liveportrait",
        device: str = "cuda",
        dtype: str = "float16",
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        onnx_model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype, model=model, **kwargs)
        self._method: str = (method or "liveportrait").lower().strip()
        # 以 method 为主导解析默认模型；用户显式指定其他 model 时保留之
        method_default = _DEFAULT_MODELS.get(self._method)
        if method_default is not None and (
            model is None or model == "KwaiVGI/LivePortrait"
        ):
            self._model_name: str = method_default
        else:
            self._model_name = model
        self._wav2vec2_model: str = wav2vec2_model
        self._onnx_model_path: str | None = onnx_model_path

        # 运行时子模块引用（load 后填充）
        self._components: dict[str, Any] | None = None
        self._audio_encoder: Any = None
        # ONNX Runtime 加速（可选，load 阶段惰性初始化）
        self._onnx_session: Any = None
        self._use_onnx: bool = False

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载驱动模型到 GPU/CPU。

        根据 :attr:`_method` 分发到对应的加载逻辑，加载完成后调用
        :meth:`_apply_optimizations` 应用显存优化。
        """
        self._scheduler.track(self)

        if self._pipeline is not None or self._model is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading avatar driver model %s (method=%s) ...",
            self._model_name,
            self._method,
        )

        if self._method == "liveportrait":
            self._load_liveportrait()
        elif self._method == "sadtalker":
            self._load_sadtalker()
        elif self._method == "musetalk":
            self._load_musetalk()
        else:
            raise ValueError(
                f"Unsupported method {self._method!r}. "
                f"Expected one of: {sorted(_DEFAULT_MODELS)}."
            )

        self._apply_optimizations()
        self._try_init_onnx()
        self._loaded = True
        self._logger.info(
            "Avatar driver loaded (method=%s, device=%s, dtype=%s).",
            self._method,
            self._resolve_device(),
            self._dtype_str,
        )

    def _load_liveportrait(self) -> None:
        """加载 LivePortrait 模型。

        LivePortrait 由四个子模块组成：``appearance_encoder``、
        ``motion_extractor``、``warping_module``、``spade_decoder``，通过
        ``diffusers.LivePortraitPipeline`` 统一加载。显存约 4-6GB（fp16）。
        """
        import torch  # type: ignore
        from diffusers import LivePortraitPipeline  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        self._pipeline = LivePortraitPipeline.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
        )
        self._pipeline = self._pipeline.to(device)

        # 记录子模块引用（便于 describe 与调试）
        self._components = {
            "appearance_encoder": getattr(
                self._pipeline, "appearance_encoder", None
            ),
            "motion_extractor": getattr(
                self._pipeline, "motion_extractor", None
            ),
            "warping_module": getattr(self._pipeline, "warping_module", None),
            "spade_decoder": getattr(self._pipeline, "spade_decoder", None),
        }

    def _load_sadtalker(self) -> None:
        """加载 SadTalker 模型（3DMM 面部建模）。

        SadTalker 通过 3DMM 系数预测表情/姿态并渲染，显存约 4-6GB（fp16）。
        优先使用 ``sadtalker`` Python 包；不可用时回退到本地权重目录的结构化
        加载（mapping / generator / kp_extractor / renderer）。
        """
        import os
        import torch  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        try:
            from sadtalker import SadTalker  # type: ignore

            self._model = SadTalker(
                checkpoint_path=self._model_name,
                device=device,
                dtype=dtype,
            )
        except ImportError:
            self._logger.warning(
                "'sadtalker' package not found. Attempting structured "
                "checkpoint loading from %s.",
                self._model_name,
            )

            if not os.path.isdir(self._model_name):
                raise ImportError(
                    "SadTalker backend requires the 'sadtalker' package or a "
                    "local checkpoint directory. Install via "
                    "`pip install sadtalker` or set `model` to a local path."
                )
            components: dict[str, Any] = {}
            for name in ("mapping", "generator", "kp_extractor", "renderer"):
                ckpt = os.path.join(self._model_name, f"{name}.pth")
                if os.path.exists(ckpt):
                    components[name] = torch.load(ckpt, map_location=device, weights_only=False)
            if not components:
                raise FileNotFoundError(
                    f"No SadTalker checkpoints found under {self._model_name!r}."
                )
            self._model = components

    def _load_musetalk(self) -> None:
        """加载 MuseTalk 模型（UNet 口型驱动）。

        MuseTalk 由 UNet 口型驱动网络与 wav2vec 音频特征提取器组成，
        显存约 6-8GB（fp16）。
        """
        import os
        import torch  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        try:
            from musetalk import MuseTalk  # type: ignore

            self._model = MuseTalk.from_pretrained(
                self._model_name, torch_dtype=dtype
            )
            self._model = self._model.to(device)
        except ImportError:
            self._logger.warning(
                "'musetalk' package not found. Attempting structured "
                "checkpoint loading from %s.",
                self._model_name,
            )
            if not os.path.isdir(self._model_name):
                raise ImportError(
                    "MuseTalk backend requires the 'musetalk' package or a "
                    "local checkpoint directory. Install via "
                    "`pip install musetalk` or set `model` to a local path."
                )
            unet_ckpt = os.path.join(self._model_name, "unet.pth")
            if not os.path.exists(unet_ckpt):
                raise FileNotFoundError(
                    f"MuseTalk UNet checkpoint not found at {unet_ckpt!r}."
                )
            self._model = {
                "unet": torch.load(unet_ckpt, map_location=device, weights_only=False),
            }

        # wav2vec 音频特征提取器（可选：加载失败时仅 debug 日志，不阻断）
        try:
            from transformers import (  # type: ignore
                Wav2Vec2Model,
                Wav2Vec2Processor,
            )

            self._processor = Wav2Vec2Processor.from_pretrained(
                self._wav2vec2_model
            )
            self._audio_encoder = Wav2Vec2Model.from_pretrained(
                self._wav2vec2_model,
                torch_dtype=self._resolve_dtype(),
            ).to(device)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "wav2vec audio encoder disabled: %s", exc,
            )

    def _try_init_onnx(self) -> None:
        """尝试初始化 ONNX Runtime 会话以加速推理。

        使用 ``mosaic.core.onnx_utils`` 验证 onnxruntime 是否真正可用
        （不仅检查 import，还验证 InferenceSession 属性存在）。当可用
        且提供了 ``onnx_model_path`` 时，实际创建推理会话并赋值给
        :attr:`_onnx_session`；否则仅记录原因并回退到 PyTorch 推理。
        """
        from mosaic.core.onnx_utils import (
            OnnxRuntimeStatus,
            create_inference_session,
            get_onnx_providers,
            is_onnxruntime_usable,
        )

        # 1) 没有提供 ONNX 模型路径：直接返回，使用 PyTorch 推理
        if not self._onnx_model_path:
            self._logger.info(
                "No onnx_model_path provided; ONNX acceleration disabled "
                "(using PyTorch inference)."
            )
            self._use_onnx = False
            self._onnx_session = None
            return

        # 2) 检测 onnxruntime 是否真正可用
        if not is_onnxruntime_usable():
            _, _version, _providers, error = OnnxRuntimeStatus.get()
            self._logger.warning(
                "ONNX runtime not available: %s; falling back to "
                "PyTorch inference.",
                error or "unknown reason",
            )
            self._use_onnx = False
            self._onnx_session = None
            return

        # 3) 检查模型文件是否存在
        import os

        if not os.path.exists(self._onnx_model_path):
            self._logger.warning(
                "ONNX model file not found: %s; falling back to "
                "PyTorch inference.",
                self._onnx_model_path,
            )
            self._use_onnx = False
            self._onnx_session = None
            return

        # 4) 创建 ONNX 推理会话
        try:
            providers = get_onnx_providers(self._device)
            self._onnx_session = create_inference_session(
                self._onnx_model_path, providers=providers
            )
            self._use_onnx = True
            self._logger.info(
                "ONNX session created for AvatarDriver "
                "(path=%s, providers=%s).",
                self._onnx_model_path,
                self._onnx_session.get_providers(),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "ONNX session creation failed: %s. Using PyTorch inference.",
                exc,
            )
            self._onnx_session = None
            self._use_onnx = False

    def unload(self) -> None:
        """释放驱动模型与显存。"""
        if self._pipeline is not None:
            # 移至 CPU 再置空，加速 GPU 显存回收
            try:
                self._pipeline.to("cpu")
            except Exception:
                pass
            self._pipeline = None
        self._model = None
        self._processor = None
        self._components = None
        self._audio_encoder = None
        self._onnx_session = None
        self._use_onnx = False
        self._loaded = False
        from mosaic.core._device_utils import empty_device_cache

        empty_device_cache()
        self._logger.info("Avatar driver unloaded (method=%s).", self._method)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行数字人形象驱动。

        Parameters
        ----------
        input_data:
            必须包含 ``source_image`` (PIL.Image | str)；驱动源三选一：
            ``driving_video`` (VideoData | str | list[PIL.Image])、
            ``driving_audio`` (AudioData | str | ndarray)、
            ``expression_params`` (list[dict] | dict)。
            可选：``output_format`` ("video"|"frames", 默认 "video")、
            ``fps`` (int, 默认 25)、``resolution`` (tuple[int,int],
            默认 (512,512))、``expression_scale`` (float, 默认 1.0)、
            ``motion_scale`` (float, 默认 1.0)。

        Returns
        -------
        MosaicData
            含 ``video`` (VideoData) 或 ``frames`` (list[PIL.Image])、
            ``source_image``、``driving_source_type``、``duration``、``fps``。

        Raises
        ------
        ValueError
            缺少 ``source_image`` 或驱动源、检测不到人脸、method 不支持。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 1. 加载源图片并 resize 到目标分辨率
            source_input = input_data.get("source_image")
            if source_input is None:
                raise ValueError(
                    "AvatarDriver requires 'source_image' "
                    "(PIL.Image or file path)."
                )
            from PIL import Image  # type: ignore

            source_image = self._load_image(source_input)

            output_format = str(
                input_data.get("output_format", "video")
            ).lower()
            fps = safe_int(input_data.get("fps"), "fps", default=_DEFAULT_FPS)
            resolution = input_data.get("resolution") or _DEFAULT_RESOLUTION
            resolution = (int(resolution[0]), int(resolution[1]))
            expression_scale = safe_float(
                input_data.get("expression_scale"), "expression_scale", default=1.0
            )
            motion_scale = safe_float(
                input_data.get("motion_scale"), "motion_scale", default=1.0
            )

            if source_image.size != resolution:
                source_image = source_image.resize(resolution, Image.Resampling.LANCZOS)

            # 2. 检测并对齐人脸
            _, bbox, _ = self._detect_face(source_image)
            self._logger.info(
                "Source face detected: bbox=%s, resolution=%s.",
                bbox,
                resolution,
            )

            # 3. 分发驱动源（三选一）
            driving_video = input_data.get("driving_video")
            driving_audio = input_data.get("driving_audio")
            expression_params = input_data.get("expression_params")

            if driving_video is not None:
                driving_source_type = "video"
                frames = self._drive_from_video(
                    source_image,
                    bbox,
                    driving_video,
                    fps,
                    expression_scale,
                    motion_scale,
                )
            elif driving_audio is not None:
                driving_source_type = "audio"
                frames = self._drive_from_audio(
                    source_image,
                    bbox,
                    driving_audio,
                    fps,
                    expression_scale,
                    motion_scale,
                )
            elif expression_params is not None:
                driving_source_type = "expression_params"
                frames = self._drive_from_params(
                    source_image,
                    bbox,
                    expression_params,
                    fps,
                    expression_scale,
                    motion_scale,
                )
            else:
                raise ValueError(
                    "AvatarDriver requires one of 'driving_video', "
                    "'driving_audio' or 'expression_params'."
                )
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        duration = len(frames) / fps if fps > 0 and frames else 0.0

        result = MosaicData(
            source_image=source_image,
            driving_source_type=driving_source_type,  # type: ignore[arg-type]
            duration=duration,
            fps=fps,
        )
        if output_format == "frames":
            result["frames"] = frames
        else:
            result["video"] = VideoData(
                frames=frames,
                fps=fps,
                metadata={
                    "duration": duration,
                    "width": resolution[0],
                    "height": resolution[1],
                    "frame_count": len(frames),
                    "method": self._method,
                    "driving_source_type": driving_source_type,
                },
            )

        self._emit_complete(
            duration=elapsed,
            output_summary={
                "method": self._method,
                "driving_source_type": driving_source_type,
                "frame_count": len(frames),
                "duration": duration,
                "fps": fps,
            },
        )
        return result

    # ------------------------------------------------------------------
    # 驱动分发
    # ------------------------------------------------------------------
    def _drive_from_video(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        driving_video: Any,
        fps: int,
        expression_scale: float,
        motion_scale: float,
    ) -> list[Any]:
        """从驱动视频逐帧迁移表情到源图片。"""
        from PIL import Image  # type: ignore

        video_data = self._load_driving_video(driving_video)
        driving_frames = video_data.frames
        if not driving_frames:
            raise ValueError("Driving video contains no frames.")

        total = len(driving_frames)
        self._emit_progress(0, total, "Driving from video")

        out_frames: list[Any] = []
        for i, drv_frame in enumerate(driving_frames):
            if not isinstance(drv_frame, Image.Image):
                drv_frame = Image.fromarray(drv_frame)
            # resize 驱动帧到源分辨率
            if drv_frame.size != source_image.size:
                drv_frame = drv_frame.resize(source_image.size, Image.Resampling.LANCZOS)

            if self._method == "liveportrait":
                driven = self._render_liveportrait(
                    source_image,
                    bbox,
                    drv_frame,
                    expression_scale,
                    motion_scale,
                )
            else:
                # sadtalker / musetalk 原生为音频驱动；视频驱动走表情参数回退
                try:
                    params = self._frame_to_expression_params(drv_frame)
                except ValueError as exc:
                    self._logger.warning(
                        "Skipping frame %d: %s", i, exc,
                    )
                    out_frames.append(source_image.copy())
                    continue
                driven = self._render_expression_fallback(
                    source_image, bbox, params
                )
            out_frames.append(driven)
            self._emit_progress(
                i + 1, total, f"Driven frame {i + 1}/{total}"
            )

        return out_frames

    def _drive_from_audio(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        driving_audio: Any,
        fps: int,
        expression_scale: float,
        motion_scale: float,
    ) -> list[Any]:
        """从驱动音频生成表情序列并驱动。

        * ``sadtalker`` / ``musetalk``：原生音频驱动，模型一次性生成全部帧。
        * ``liveportrait``：音频能量→口型参数序列→逐帧驱动（启发式桥接）。
        """
        waveform, sr = self._load_audio_signal(driving_audio)
        duration = self._get_audio_duration(waveform, sr)
        n_frames = max(1, int(round(duration * fps)))

        self._emit_progress(0, n_frames, "Driving from audio")

        if self._method in ("sadtalker", "musetalk"):
            frames = self._generate_audio_driven(
                source_image, bbox, waveform, sr, fps, n_frames
            )
            self._emit_progress(n_frames, n_frames, "Audio driving complete")
            return frames

        # liveportrait：音频 -> 表情参数序列 -> 逐帧驱动
        expr_seq = self._audio_to_expression_seq(waveform, sr, fps, n_frames)
        out_frames: list[Any] = []
        for i, params in enumerate(expr_seq):
            driven = self._render_liveportrait(
                source_image,
                bbox,
                params,
                expression_scale,
                motion_scale,
            )
            out_frames.append(driven)
            self._emit_progress(
                i + 1, n_frames, f"Driven frame {i + 1}/{n_frames}"
            )
        return out_frames

    def _drive_from_params(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        expression_params: Any,
        fps: int,
        expression_scale: float,
        motion_scale: float,
    ) -> list[Any]:
        """直接接收表情参数序列驱动。"""
        params_seq = self._normalize_expression_params(expression_params)
        if not params_seq:
            raise ValueError("'expression_params' is empty.")

        total = len(params_seq)
        self._emit_progress(0, total, "Driving from expression params")

        out_frames: list[Any] = []
        for i, params in enumerate(params_seq):
            if self._method == "liveportrait":
                driven = self._render_liveportrait(
                    source_image,
                    bbox,
                    params,
                    expression_scale,
                    motion_scale,
                )
            else:
                driven = self._render_expression_fallback(
                    source_image, bbox, params
                )
            out_frames.append(driven)
            self._emit_progress(i + 1, total, f"Driven frame {i + 1}/{total}")
        return out_frames

    # ------------------------------------------------------------------
    # 渲染（method 分发）
    # ------------------------------------------------------------------
    def _render_liveportrait(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        signal: Any,
        expression_scale: float,
        motion_scale: float,
    ) -> Any:
        """使用 LivePortrait 渲染单帧。

        ``signal`` 可为驱动帧（PIL.Image）或表情参数字典。
        ``expression_scale`` 对口型参数做缩放；``motion_scale`` 用于运动幅度
        调节（当模型支持时透传）。
        """
        from PIL import Image  # type: ignore

        pipe_kwargs: dict[str, Any] = {"source_image": source_image}
        if isinstance(signal, dict):
            params = dict(signal)
            mo = float(params.get("mouth_open", 0.0))
            params["mouth_open"] = max(0.0, min(1.0, mo * expression_scale))
            pipe_kwargs["driving_params"] = params
            pipe_kwargs["motion_scale"] = motion_scale
        elif isinstance(signal, Image.Image):
            pipe_kwargs["driving_image"] = signal
        else:
            pipe_kwargs["driving_image"] = self._load_image(signal)

        # 优先尝试 ONNX Runtime 加速推理，失败时回退到 PyTorch pipeline
        if self._use_onnx and self._onnx_session is not None:
            try:
                import numpy as np  # type: ignore

                # 根据会话声明的输入名匹配 driving / image 输入
                session_inputs = self._onnx_session.get_inputs()
                driving_kw = (
                    "driving", "driven", "motion", "expression", "pose",
                    "kp", "keypoint", "audio", "mel", "wave",
                )
                image_kw = (
                    "source", "image", "avatar", "reference", "src",
                )
                driving_input_name: str | None = None
                image_input_name: str | None = None
                for inp in session_inputs:
                    lname = inp.name.lower()
                    if driving_input_name is None and any(
                        kw in lname for kw in driving_kw
                    ):
                        driving_input_name = inp.name
                    elif image_input_name is None and any(
                        kw in lname for kw in image_kw
                    ):
                        image_input_name = inp.name
                # 兜底：未匹配到 driving 输入名时使用首个输入
                if driving_input_name is None and session_inputs:
                    driving_input_name = session_inputs[0].name

                feeds: dict[str, Any] = {}
                # 驱动信号 -> numpy 数组
                if isinstance(signal, dict):
                    # 表情参数 -> 有序 float32 向量（补 batch 维）
                    driving_arr = np.asarray(
                        [
                            float(params.get(k, 0.0))
                            for k in ("smile", "mouth_open", "eye_openness")
                        ],
                        dtype=np.float32,
                    )[np.newaxis, :]
                else:
                    drv_img = (
                        signal
                        if isinstance(signal, Image.Image)
                        else self._load_image(signal)
                    )
                    drv_img = drv_img.convert("RGB")
                    if drv_img.size != source_image.size:
                        drv_img = drv_img.resize(
                            source_image.size, Image.Resampling.LANCZOS
                        )
                    # HWC -> CHW -> NCHW
                    driving_arr = np.asarray(
                        drv_img, dtype=np.float32
                    ).transpose(2, 0, 1)[np.newaxis, ...]
                feeds[driving_input_name or "input"] = driving_arr

                # 形象图输入 -> NCHW float32
                if image_input_name is not None:
                    src_arr = np.asarray(
                        source_image.convert("RGB"), dtype=np.float32
                    )
                    # HWC -> CHW -> NCHW
                    src_arr = src_arr.transpose(2, 0, 1)[np.newaxis, ...]
                    feeds[image_input_name] = src_arr

                result = self._onnx_session.run(None, feeds)
                # 取首个输出并转 PIL.Image
                arr = np.asarray(result[0])
                if arr.ndim == 4:
                    arr = arr[0]
                # 检测 NaN：上游推理异常时报错而非静默输出黑帧
                if np.isnan(arr).any():
                    raise RuntimeError(
                        "NaN detected in ONNX output — model inference "
                        "may have failed."
                    )
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[-1] not in (3, 4):
                    arr = np.stack([arr] * 3, axis=-1)
                driven = Image.fromarray(arr)
                # 保证背景与源图一致
                return self._compose_frame(source_image, driven, bbox)
            except Exception as exc:  # noqa: BLE001
                self._onnx_fail_count = getattr(self, "_onnx_fail_count", 0) + 1
                if self._onnx_fail_count <= 3:
                    self._logger.debug(
                        "ONNX inference failed in _render_liveportrait: %s. "
                        "Falling back to PyTorch pipeline.",
                        exc,
                    )
                if self._onnx_fail_count >= 5:
                    self._logger.warning(
                        "ONNX inference failed %d times, disabling ONNX.",
                        self._onnx_fail_count,
                    )
                    self._use_onnx = False
                    self._onnx_session = None

        # PyTorch pipeline 推理（回退路径）
        output = self._run_pipeline(**pipe_kwargs)
        images = self._extract_images(output)
        if not images:
            raise RuntimeError(
                "LivePortrait returned no driven image — model inference may have failed."
            )
        driven = images[0]
        # 保证背景与源图一致
        return self._compose_frame(source_image, driven, bbox)

    def _generate_audio_driven(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        waveform: Any,
        sample_rate: int,
        fps: int,
        n_frames: int,
    ) -> list[Any]:
        """使用 sadtalker / musetalk 原生音频驱动批量生成帧。"""
        from PIL import Image  # type: ignore

        if self._method == "sadtalker":
            output = self._model.generate(
                source_image=source_image,
                audio=waveform,
                sample_rate=sample_rate,
                fps=fps,
            )
        else:  # musetalk
            output = self._model(
                source_image=source_image,
                audio=waveform,
                sample_rate=sample_rate,
                fps=fps,
                n_frames=n_frames,
            )

        frames = self._extract_images(output)
        if not frames:
            raise RuntimeError("Audio-driven model returned no frames.")

        # 逐帧保证背景与源图一致（模型仅修改人脸/口型区域）
        composed: list[Any] = []
        for f in frames:
            if not isinstance(f, Image.Image):
                f = Image.fromarray(f)
            composed.append(self._compose_frame(source_image, f, bbox))
        return composed

    def _render_expression_fallback(
        self,
        source_image: Any,
        bbox: tuple[int, int, int, int],
        params: dict[str, Any],
    ) -> Any:
        """sadtalker/musetalk 在非音频驱动模式下的简化表情应用回退。

        使用基类 :meth:`_apply_expression`（亮度/对比度调整）应用表情参数，
        再通过 :meth:`_blend_face` 融合回源图，保持背景一致。
        """
        face_crop, _, _ = self._detect_face(source_image)
        driven_face = self._apply_expression(face_crop, params)
        return self._blend_face(source_image, driven_face, bbox, blend_ratio=1.0)

    # ------------------------------------------------------------------
    # 信号转换工具
    # ------------------------------------------------------------------
    def _load_driving_video(self, driving_video: Any) -> VideoData:
        """加载驱动视频为 VideoData。"""
        from mosaic.nodes.video._base import BaseVideoNode

        if isinstance(driving_video, VideoData):
            return driving_video
        if isinstance(driving_video, str):
            return BaseVideoNode._load_video(driving_video)
        if isinstance(driving_video, list):
            return VideoData(
                frames=list(driving_video),
                fps=_DEFAULT_FPS,
                metadata={},
            )
        raise TypeError(
            f"driving_video must be VideoData, str path, or list of frames, "
            f"got {type(driving_video).__name__}."
        )

    def _load_audio_signal(self, driving_audio: Any) -> tuple[Any, int]:
        """加载驱动音频为 (waveform, sample_rate)。"""
        from mosaic.nodes.audio._base import BaseAudioNode

        return BaseAudioNode._load_audio(driving_audio)

    @staticmethod
    def _get_audio_duration(waveform: Any, sample_rate: int) -> float:
        """计算音频时长（秒）。"""
        from mosaic.nodes.audio._base import BaseAudioNode

        return BaseAudioNode._get_duration(waveform, sample_rate)

    def _audio_to_expression_seq(
        self,
        waveform: Any,
        sample_rate: int,
        fps: int,
        n_frames: int,
    ) -> list[dict[str, Any]]:
        """将音频能量包络转为逐帧表情参数序列（mouth_open 代理）。

        这是 ``driving_audio`` + ``liveportrait`` 组合下的启发式桥接：以每帧
        对应音频片段的 RMS 能量近似口型张开程度。
        """
        import numpy as np  # type: ignore

        mono = waveform
        if hasattr(mono, "ndim") and mono.ndim > 1:
            mono = mono.mean(axis=0)
        mono = np.asarray(mono, dtype=np.float32)
        total_samples = mono.shape[-1]

        hop = max(1, total_samples // max(1, n_frames))
        seq: list[dict[str, Any]] = []
        for i in range(n_frames):
            seg = mono[i * hop:(i + 1) * hop]
            energy = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            mouth_open = float(np.clip(energy * self.ENERGY_SCALE, 0.0, 1.0))
            seq.append(
                {
                    "smile": 0.0,
                    "mouth_open": mouth_open,
                    "eye_openness": 1.0,
                }
            )
        return seq

    def _frame_to_expression_params(self, frame: Any) -> dict[str, Any]:
        """从驱动帧的人脸关键点推导简化表情参数。"""
        import numpy as np  # type: ignore

        _, _, landmarks = self._detect_face(frame)
        if landmarks is None or len(landmarks) < 5:
            return {"smile": 0.0, "mouth_open": 0.0, "eye_openness": 1.0}

        lm = np.asarray(landmarks, dtype=np.float32)
        left_eye, right_eye, nose, left_mouth, right_mouth = lm[:5]
        eye_dist = float(np.linalg.norm(right_eye - left_eye)) or 1e-6
        mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
        mouth_mid = (left_mouth + right_mouth) / 2.0
        mouth_open_proxy = abs(float(mouth_mid[1] - nose[1])) / eye_dist
        mouth_open = float(np.clip(mouth_open_proxy / self.MOUTH_OPEN_THRESHOLD, 0.0, 1.0))
        smile = float(np.clip(mouth_width / eye_dist - self.MOUTH_WIDTH_OFFSET, 0.0, 1.0))
        return {
            "smile": smile,
            "mouth_open": mouth_open,
            "eye_openness": 1.0,
        }

    @staticmethod
    def _normalize_expression_params(params: Any) -> list[dict[str, Any]]:
        """将 expression_params 规整为字典列表。"""
        if isinstance(params, dict):
            return [params]
        if isinstance(params, list):
            return [
                p if isinstance(p, dict) else {"mouth_open": float(p)}
                for p in params
            ]
        raise TypeError(
            f"expression_params must be dict or list, "
            f"got {type(params).__name__}."
        )

    # ------------------------------------------------------------------
    # 帧合成与输出提取
    # ------------------------------------------------------------------
    def _compose_frame(
        self,
        source_image: Any,
        driven_frame: Any,
        bbox: tuple[int, int, int, int],
    ) -> Any:
        """将驱动帧与源图合成，保证背景与源图一致。

        取驱动帧在 ``bbox`` 区域的人脸，融合回源图（bbox 外保持源图背景）。
        """
        from PIL import Image  # type: ignore

        if not isinstance(driven_frame, Image.Image):
            driven_frame = Image.fromarray(driven_frame)
        if driven_frame.size != source_image.size:
            driven_frame = driven_frame.resize(
                source_image.size, Image.Resampling.LANCZOS
            )
        x1, y1, x2, y2 = bbox
        driven_face = driven_frame.crop((x1, y1, x2, y2))
        return self._blend_face(source_image, driven_face, bbox, blend_ratio=1.0)

    @staticmethod
    def _extract_images(output: Any) -> list[Any]:
        """从模型输出中提取 PIL.Image 列表（兼容多种返回格式）。

        支持 diffusers 风格的属性访问（``frames``/``images``/``image``）、
        字典风格、原始 tensor 与 numpy 数组。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        raw: Any = None
        if isinstance(output, dict):
            for key in ("frames", "images", "image", "video", "frame"):
                if key in output:
                    raw = output[key]
                    break
        else:
            for attr in ("frames", "images", "image", "video", "frame"):
                raw = getattr(output, attr, None)
                if raw is not None:
                    break
        if raw is None:
            raw = output

        # tensor 路径：借用视频域的 tensor->frames 工具
        if hasattr(raw, "cpu"):
            from mosaic.nodes.video._base import BaseVideoNode

            return BaseVideoNode._tensor_to_frames(raw)

        if isinstance(raw, list):
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            images: list[Any] = []
            for f in raw:
                if isinstance(f, Image.Image):
                    images.append(f)
                else:
                    images.append(Image.fromarray(np.asarray(f)))
            return images

        if isinstance(raw, Image.Image):
            return [raw]
        return [Image.fromarray(np.asarray(raw))]

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息（VRAM、许可证、method）。"""
        info = self._build_model_info(self._model_name)
        info["method"] = self._method
        # 用 method 粒度的显存估算覆盖（更精确）
        if self._method in _METHOD_VRAM:
            info["vram_gb"] = _METHOD_VRAM[self._method]
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=info,
        )
