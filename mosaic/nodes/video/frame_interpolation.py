# mosaic/nodes/video/frame_interpolation.py
"""FrameInterpolator 节点 —— 视频插帧。

在已有视频的相邻帧之间生成中间帧，提升帧率、平滑运动。支持三种插值
方法：

* ``rife``   —— 基于 RIFE 的 ONNX 模型推理（通过 ``onnxruntime``，无需
  PyTorch），运动估计较好，适合 GPU 推理。
* ``film``   —— 基于 Google FILM 模型推理（大运动场景表现优秀），通过
  ``tensorflow`` / ``tensorflow_hub`` 加载（独立实现），亦可替换为
  ``transformers`` 等价实现。
* ``linear`` —— 简单线性加权混合，无需任何模型，CPU 即可运行，速度最快
  但画质最低（适合预览 / 低算力场景）。

设计要点
--------
* 三种方法共用统一的外部接口（``run`` 输入输出一致），内部按 ``method``
  分派到 ``_interpolate_linear`` / ``_interpolate_rife`` / ``_interpolate_film``。
* 插帧本质为 “2 倍率” 操作（在每对相邻帧间插入 1 帧）。更高倍率通过
  **递归 2x** 实现：``scale_factor=4`` 等价于先 2x 再 2x。因此
  ``scale_factor`` 应为 2 的幂（1 / 2 / 4 / 8 / ...），非 2 的幂时自动
  取最近的 2 的幂并告警。
* ``target_fps`` 与 ``scale_factor`` 二选一：提供 ``target_fps`` 时按
  “原始 fps -> 目标 fps” 推算所需的 2x 次数；否则直接使用 ``scale_factor``。
  ``target_fps`` 优先级高于 ``scale_factor``。
* 大视频采用 **分段（chunk）处理**：按相邻帧对分块，每块处理
  ``chunk_size`` 对，避免一次性把全部帧送入显存导致 OOM。
* ``onnxruntime`` / ``tensorflow`` / ``PIL`` / ``numpy`` 均为惰性导入，
  未安装时模块仍可被注册表发现与导入（仅在实际加载 / 推理时报依赖缺失）。
* 模型生命周期由 :class:`~mosaic.core.scheduler.Scheduler` 管理；
  ``run`` 调用 ``scheduler.ensure_loaded(self)`` 触发按需加载 + LRU 淘汰。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start`` /
  ``node_complete`` / ``node_error`` / ``progress`` 事件。
* ``linear`` 方法无需模型，``_load_model`` 直接返回。
* 输出统一为 :class:`~mosaic.core.types.VideoData` 格式。

显存需求
--------
* ``rife``：约 1-3GB（取决于模型版本与分辨率）
* ``film``：约 2-4GB
* ``linear``：0（纯 CPU）

许可证
------
* RIFE：MIT License（模型权重参见对应 model card）
* FILM：Apache 2.0
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from mosaic.core.onnx_utils import (
    create_inference_session,
    get_onnx_providers,
    is_onnxruntime_usable,
)
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode
from mosaic.nodes.video._video_utils import safe_int

__all__ = ["FrameInterpolator"]


# 支持的插值方法
_VALID_METHODS: tuple[str, ...] = ("rife", "film", "linear")

# 分段大小：每块处理的相邻帧对数，用于控制显存峰值
_DEFAULT_CHUNK_SIZE: int = 64

# RIFE 默认 ONNX 模型路径（用户可按需替换为本地路径）
_DEFAULT_RIFE_MODEL: str = "rife_v4.onnx"

# FILM 默认模型来源（TFHub URL 或本地 SavedModel 目录）
_DEFAULT_FILM_MODEL: str = (
    "https://tfhub.dev/google/frame-interpolation/large/1"
)


@registry.register
class FrameInterpolator(BaseVideoNode):
    """视频插帧节点。

    在相邻帧之间生成中间帧以提升帧率，支持 ``rife`` / ``film`` / ``linear``
    三种方法，输出统一为 :class:`VideoData`。

    Parameters
    ----------
    model:
        模型路径或标识，``None`` 时使用方法对应的默认模型。
        ``linear`` 方法忽略此参数。
    method:
        插值方法，可选 ``"rife"`` / ``"film"`` / ``"linear"``，默认
        ``"rife"``。
    device:
        推理设备，默认 ``"cuda"``；``linear`` 方法忽略。
    dtype:
        推理精度，默认 ``"float16"``（仅部分后端生效）。
    chunk_size:
        分段处理的相邻帧对数，默认 64。增大可提升吞吐，但会增加显存占用。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    使用 RIFE 做 2 倍插帧：
    >>> fi = FrameInterpolator(method="rife", model="/path/to/rife_v4.onnx")
    >>> result = fi(MosaicData(video=input_video_data, scale_factor=2))
    >>> out = result["video"]  # VideoData
    >>> result["new_fps"], result["new_frame_count"]

    使用线性插值做 4 倍插帧（CPU 友好）：
    >>> fi = FrameInterpolator(method="linear")
    >>> result = fi(MosaicData(video=input_video_data, scale_factor=4))

    按目标帧率插帧：
    >>> result = fi(MosaicData(video=input_video_data, target_fps=60))
    """

    name: str = "frame-interpolation"
    description: str = (
        "Interpolate intermediate frames between existing video frames "
        "to increase fps. Supports RIFE (ONNX), FILM, and linear blending."
    )
    version: str = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["video"]

    def __init__(
        self,
        model: str | None = None,
        method: str = "rife",
        device: str = "cuda",
        dtype: str = "float16",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or "", device=device, dtype=dtype, **kwargs)
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unsupported interpolation method: {method!r}. "
                f"Supported methods: {_VALID_METHODS}."
            )
        self._method: str = method
        self._chunk_size: int = max(1, int(chunk_size))

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """按 ``method`` 加载对应模型。

        * ``linear``：无需模型，直接返回。
        * ``rife``：惰性导入 ``onnxruntime``，加载 RIFE ONNX 模型。
        * ``film``：惰性导入 ``tensorflow`` / ``tensorflow_hub``，加载 FILM 模型。
        """
        self._logger.info(
            "Loading interpolation model: method=%s, model=%r, chunk_size=%d.",
            self._method, self._model_name, self._chunk_size,
        )
        if self._method == "linear":
            self._logger.info(
                "Linear interpolation selected; no model required, "
                "skipping load."
            )
            return
        if self._method == "rife":
            self._load_rife_model()
        elif self._method == "film":
            self._load_film_model()
        else:  # pragma: no cover - 由构造函数兜底
            raise ValueError(
                f"Unsupported interpolation method: {self._method!r}. "
                f"Supported methods: {_VALID_METHODS}."
            )

    def _load_rife_model(self) -> None:
        """加载 RIFE ONNX 模型。

        使用 ``mosaic.core.onnx_utils`` 安全创建 InferenceSession，
        自动处理 CUDA/cuDNN 版本不匹配问题。当 onnxruntime 不可用时，
        自动回退到 linear 插值方法。
        """
        if not is_onnxruntime_usable():
            self._logger.warning(
                "onnxruntime 不可用（InferenceSession 加载失败），"
                "自动回退到 linear 插值方法。"
            )
            self._method = "linear"
            return

        model_path = self._model_name or _DEFAULT_RIFE_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RIFE ONNX model not found: {model_path!r}. "
                f"Please provide a valid 'model' path to a RIFE .onnx file."
            )

        device = self._resolve_device()
        providers = get_onnx_providers(device)

        self._pipeline = create_inference_session(
            model_path, providers=providers
        )
        self._logger.info(
            "RIFE ONNX model loaded (path=%s, providers=%s).",
            model_path,
            self._pipeline.get_providers(),
        )

    def _load_film_model(self) -> None:
        """加载 FILM 模型（惰性导入 ``tensorflow`` / ``tensorflow_hub``）。

        默认从 TFHub 加载官方 FILM 模型；若 ``model`` 指向本地 SavedModel
        目录，则从本地加载。亦可替换为 ``transformers`` 等价实现。
        """
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FILM interpolation requires TensorFlow. "
                "Install via `pip install tensorflow tensorflow_hub`."
            ) from exc

        model_src = self._model_name or _DEFAULT_FILM_MODEL

        loaded: Any = None
        try:
            import tensorflow_hub as hub  # type: ignore

            loaded = hub.load(model_src)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "FILM: TFHub load failed (%s); trying local SavedModel.", exc
            )
            if os.path.exists(model_src):
                loaded = tf.saved_model.load(model_src)
            else:
                raise FileNotFoundError(
                    f"FILM model could not be loaded from {model_src!r}. "
                    f"Provide a valid TFHub URL or local SavedModel directory."
                ) from exc

        self._pipeline = loaded
        self._logger.info("FILM model loaded from %s.", model_src)

    # ------------------------------------------------------------------
    # 倍率 / 帧率推算
    # ------------------------------------------------------------------
    def _passes_for_factor(self, factor: int) -> int:
        """将 ``scale_factor`` 转换为 2x 递归次数。

        ``scale_factor`` 应为 2 的幂（1/2/4/8/...）。非 2 的幂时取最近的
        2 的幂并告警。

        Parameters
        ----------
        factor:
            用户请求的倍率。

        Returns
        -------
        int
            2x 递归次数（0 表示不插帧）。
        """
        if factor <= 1:
            return 0
        if (factor & (factor - 1)) == 0:
            return int(round(math.log2(factor)))
        nearest = 2 ** int(round(math.log2(factor)))
        self._logger.warning(
            "scale_factor=%d is not a power of 2; using nearest %d "
            "(recursive 2x supports 1/2/4/8/...).",
            factor,
            nearest,
        )
        return int(round(math.log2(nearest)))

    def _resolve_num_passes(
        self,
        original_fps: int,
        scale_factor: int,
        target_fps: int | None,
    ) -> tuple[int, int | None]:
        """根据 ``target_fps`` 或 ``scale_factor`` 推算 2x 递归次数。

        ``target_fps`` 优先；未提供时使用 ``scale_factor``。

        Parameters
        ----------
        original_fps:
            原始视频帧率。
        scale_factor:
            用户请求的倍率。
        target_fps:
            用户请求的目标帧率，``None`` 表示按 ``scale_factor`` 推算。

        Returns
        -------
        tuple[int, int | None]
            ``(num_passes, target_fps)``。
        """
        if target_fps is not None:
            target_fps = int(target_fps)
            if original_fps <= 0:
                raise ValueError(
                    f"original_fps must be > 0 for interpolation, got {original_fps}."
                )
            if target_fps <= original_fps:
                self._logger.warning(
                    "target_fps=%d <= original_fps=%d; no interpolation applied.",
                    target_fps,
                    original_fps,
                )
                return 0, target_fps
            raw = target_fps / original_fps
            num_passes = max(0, int(round(math.log2(raw))))
            achieved = original_fps * (2 ** num_passes)
            if achieved != target_fps:
                self._logger.warning(
                    "target_fps=%d is not exactly reachable via 2x recursion "
                    "from %d fps; achieving %d fps (2^%d).",
                    target_fps,
                    original_fps,
                    achieved,
                    num_passes,
                )
            return num_passes, target_fps

        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}.")
        return self._passes_for_factor(scale_factor), None

    # ------------------------------------------------------------------
    # 帧预处理
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_frame_sizes(frames: list[Any]) -> list[Any]:
        """将所有帧统一到首帧尺寸（防御性，便于逐对插值）。"""
        if not frames:
            return frames
        from PIL import Image  # type: ignore

        base_size = frames[0].size
        out: list[Any] = [frames[0]]
        for f in frames[1:]:
            if f.size != base_size:
                f = f.resize(base_size, Image.Resampling.LANCZOS)
            out.append(f)
        return out

    # ------------------------------------------------------------------
    # 插值核心
    # ------------------------------------------------------------------
    def _interpolate_2x(
        self,
        frames: list[Any],
        progress_base: int = 0,
        progress_total: int | None = None,
        pass_label: str = "",
    ) -> list[Any]:
        """执行一次 2x 插帧（在每对相邻帧间插入 1 帧）。

        按 ``self._method`` 分派到 ``_interpolate_linear`` /
        ``_interpolate_rife`` / ``_interpolate_film``。更高倍率由调用方
        递归调用本方法实现（``scale_factor=4`` = 两次 2x）。

        Parameters
        ----------
        frames:
            输入帧列表。
        progress_base:
            全局进度偏移（已完成的工作量）。
        progress_total:
            全局总工作量，``None`` 时不发进度事件。
        pass_label:
            进度描述前缀，用于区分多趟递归。

        Returns
        -------
        list[PIL.Image]
            插帧后的帧列表，长度约为 ``2 * len(frames) - 1``。
        """
        label = pass_label or self._method
        n_in = len(frames)
        self._logger.debug(
            "interpolate_2x dispatch: method=%s, input_frames=%d, label=%r.",
            self._method, n_in, label,
        )
        if self._method == "linear":
            return self._interpolate_linear(
                frames, progress_base, progress_total, label
            )
        if self._method == "rife":
            return self._interpolate_rife(
                frames, progress_base, progress_total, label
            )
        if self._method == "film":
            return self._interpolate_film(
                frames, progress_base, progress_total, label
            )
        raise ValueError(  # pragma: no cover - 由构造函数兜底
            f"Unsupported interpolation method: {self._method!r}."
        )

    def _interpolate_pairs(
        self,
        frames: list[Any],
        pair_fn: Any,
        label: str,
        progress_base: int = 0,
        progress_total: int | None = None,
    ) -> list[Any]:
        """分段对相邻帧对执行插值并组装结果。

        对每对相邻帧 ``(frames[i], frames[i+1])`` 调用 ``pair_fn`` 生成
        中间帧，按 ``[f0, mid01, f1, mid12, f2, ...]`` 顺序组装。按
        ``self._chunk_size`` 分块处理，避免大视频一次性占用过多显存。

        Parameters
        ----------
        frames:
            输入帧列表。
        pair_fn:
            可调用对象 ``pair_fn(frame_a, frame_b) -> PIL.Image``。
        label:
            进度描述前缀。
        progress_base:
            全局进度偏移（已完成的工作量）。
        progress_total:
            全局总工作量，``None`` 时不发进度事件。

        Returns
        -------
        list[PIL.Image]
            插帧后的帧列表。
        """
        n = len(frames)
        if n < 2:
            return list(frames)

        total_pairs = n - 1
        chunk = self._chunk_size
        result: list[Any] = [frames[0]]
        done = 0

        for start in range(0, total_pairs, chunk):
            end = min(start + chunk, total_pairs)
            for i in range(start, end):
                mid = pair_fn(frames[i], frames[i + 1])
                result.append(mid)
                result.append(frames[i + 1])
                done += 1
            if progress_total is not None and progress_total > 0:
                self._emit_progress(
                    progress_base + done,
                    progress_total,
                    f"{label}: {done}/{total_pairs} pairs (chunk {start}-{end})",
                )

        return result

    def _interpolate_linear(
        self,
        frames: list[Any],
        progress_base: int = 0,
        progress_total: int | None = None,
        label: str = "linear",
    ) -> list[Any]:
        """线性插值：对相邻两帧做加权平均。

        不依赖任何模型，纯 CPU 即可运行。每对相邻帧的中间帧取
        ``0.5 * a + 0.5 * b``。

        Parameters
        ----------
        frames:
            输入帧列表。
        progress_base:
            全局进度偏移。
        progress_total:
            全局总工作量。
        label:
            进度描述前缀。

        Returns
        -------
        list[PIL.Image]
            2x 插帧后的帧列表。
        """
        return self._interpolate_pairs(
            frames,
            lambda a, b: self._linear_blend(a, b, 0.5),
            label,
            progress_base=progress_base,
            progress_total=progress_total,
        )

    def _interpolate_rife(
        self,
        frames: list[Any],
        progress_base: int = 0,
        progress_total: int | None = None,
        label: str = "rife",
    ) -> list[Any]:
        """RIFE 插值：使用 RIFE ONNX 模型推理生成中间帧。

        Parameters
        ----------
        frames:
            输入帧列表。
        progress_base:
            全局进度偏移。
        progress_total:
            全局总工作量。
        label:
            进度描述前缀。

        Returns
        -------
        list[PIL.Image]
            2x 插帧后的帧列表。
        """
        if self._pipeline is None:
            raise RuntimeError("RIFE model is not loaded.")
        return self._interpolate_pairs(
            frames,
            lambda a, b: self._rife_infer(a, b, 0.5),
            label,
            progress_base=progress_base,
            progress_total=progress_total,
        )

    def _interpolate_film(
        self,
        frames: list[Any],
        progress_base: int = 0,
        progress_total: int | None = None,
        label: str = "film",
    ) -> list[Any]:
        """FILM 插值：使用 FILM 模型推理生成中间帧。

        Parameters
        ----------
        frames:
            输入帧列表。
        progress_base:
            全局进度偏移。
        progress_total:
            全局总工作量。
        label:
            进度描述前缀。

        Returns
        -------
        list[PIL.Image]
            2x 插帧后的帧列表。
        """
        if self._pipeline is None:
            raise RuntimeError("FILM model is not loaded.")
        return self._interpolate_pairs(
            frames,
            lambda a, b: self._film_infer(a, b, 0.5),
            label,
            progress_base=progress_base,
            progress_total=progress_total,
        )

    # ------------------------------------------------------------------
    # 单对帧插值原语
    # ------------------------------------------------------------------
    @staticmethod
    def _linear_blend(
        frame_a: Any,
        frame_b: Any,
        t: float = 0.5,
    ) -> Any:
        """对两帧做线性加权平均。

        ``out = (1 - t) * a + t * b``，默认 ``t=0.5`` 即取中点。

        Parameters
        ----------
        frame_a, frame_b:
            两张 PIL 帧。
        t:
            混合权重，``[0, 1]``，默认 0.5。

        Returns
        -------
        PIL.Image
            混合后的帧。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        size = frame_a.size
        if frame_b.size != size:
            frame_b = frame_b.resize(size, Image.Resampling.LANCZOS)

        a_arr = np.array(frame_a.convert("RGB"), dtype=np.float32)
        b_arr = np.array(frame_b.convert("RGB"), dtype=np.float32)
        mixed = (1.0 - t) * a_arr + t * b_arr
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed)

    def _rife_infer(
        self,
        frame_a: Any,
        frame_b: Any,
        timestep: float = 0.5,
    ) -> Any:
        """使用 RIFE ONNX 模型推理一对帧的中间帧。

        自动按输入形状识别 NCHW / NHWC 与时间步张量，兼容 2 输入
        （仅 img0/img1）与 3 输入（img0/img1/time）两种导出。输入归一化
        为 ``[0, 1]``（除以 255），部分导出可能需要 ``[-1, 1]``，如有
        偏色可按需调整。

        Parameters
        ----------
        frame_a, frame_b:
            两张 PIL 帧。
        timestep:
            插值位置，``[0, 1]``，默认 0.5。

        Returns
        -------
        PIL.Image
            推理得到的中间帧。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        session = self._pipeline
        size = frame_a.size
        if frame_b.size != size:
            frame_b = frame_b.resize(size, Image.Resampling.LANCZOS)

        a_hwc = np.array(frame_a.convert("RGB"), dtype=np.float32) / 255.0
        b_hwc = np.array(frame_b.convert("RGB"), dtype=np.float32) / 255.0

        feeds: dict = {}
        image_inputs: list[tuple[str, list[int]]] = []
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) else -1 for d in (inp.shape or [])]
            if len(shape) == 4:
                image_inputs.append((inp.name, shape))
            elif len(shape) <= 2:
                feeds[inp.name] = (
                    np.array([[timestep]], dtype=np.float32)
                    if len(shape) == 2
                    else np.array([timestep], dtype=np.float32)
                )

        if len(image_inputs) >= 2:
            (n0, s0), (n1, s1) = image_inputs[0], image_inputs[1]
            feeds[n0] = self._to_model_layout(a_hwc, s0)
            feeds[n1] = self._to_model_layout(b_hwc, s1)
        else:
            ins = session.get_inputs()
            feeds[ins[0].name] = a_hwc.transpose(2, 0, 1)[None, ...]
            if len(ins) > 1:
                feeds[ins[1].name] = b_hwc.transpose(2, 0, 1)[None, ...]
            if len(ins) > 2:
                s = [d if isinstance(d, int) else -1 for d in (ins[2].shape or [])]
                feeds[ins[2].name] = (
                    np.array([[timestep]], dtype=np.float32)
                    if len(s) == 2
                    else np.array([timestep], dtype=np.float32)
                )

        outs = session.run(None, feeds)
        out = outs[0]
        if hasattr(out, "ndim") and out.ndim == 4:
            if out.shape[1] == 3:
                out = out[0].transpose(1, 2, 0)
            else:
                out = out[0]
        out = np.clip(np.asarray(out), 0.0, 1.0)
        out = (out * 255.0).astype(np.uint8)
        return Image.fromarray(out)

    @staticmethod
    def _to_model_layout(arr_hwc: Any, shape: list[int]) -> Any:
        """按目标形状把 ``(H, W, 3)`` 数组转为 ``(1, 3, H, W)`` 或 ``(1, H, W, 3)``。"""
        import numpy as np  # type: ignore

        if len(shape) >= 4 and shape[1] == 3:
            return arr_hwc.transpose(2, 0, 1)[None, ...]
        return arr_hwc[None, ...]

    def _film_infer(
        self,
        frame_a: Any,
        frame_b: Any,
        timestep: float = 0.5,
    ) -> Any:
        """使用 FILM 模型推理一对帧的中间帧。

        采用官方 TFHub FILM 调用约定：``model(x0, x1, time)``，输入为
        ``(1, H, W, 3)`` float32 ``[0, 1]``，``time`` 为 ``(1,)`` float32。

        Parameters
        ----------
        frame_a, frame_b:
            两张 PIL 帧。
        timestep:
            插值位置，``[0, 1]``，默认 0.5。

        Returns
        -------
        PIL.Image
            推理得到的中间帧。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        import tensorflow as tf  # type: ignore

        model = self._pipeline
        size = frame_a.size
        if frame_b.size != size:
            frame_b = frame_b.resize(size, Image.Resampling.LANCZOS)

        a = np.array(frame_a.convert("RGB"), dtype=np.float32) / 255.0
        b = np.array(frame_b.convert("RGB"), dtype=np.float32) / 255.0
        a_t = tf.constant(a[None, ...])
        b_t = tf.constant(b[None, ...])
        t_t = tf.constant([timestep], dtype=tf.float32)

        try:
            out = model(a_t, b_t, t_t)
        except (TypeError, ValueError):
            out = model(x0=a_t, x1=b_t, time=t_t)

        if isinstance(out, dict):
            out = out.get("image", next(iter(out.values())))
        arr = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # Node 执行
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行视频插帧。

        Parameters
        ----------
        input_data:
            必须包含 ``video`` (:class:`VideoData`)；可选 ``target_fps``
            (int, 与 ``scale_factor`` 二选一，优先级更高)、``scale_factor``
            (int, 默认 2)。

        Returns
        -------
        MosaicData
            包含 ``video`` (VideoData, 插帧后)、``original_fps`` (int)、
            ``new_fps`` (int)、``original_frame_count`` (int)、
            ``new_frame_count`` (int)，以及 ``method`` (str)、
            ``num_passes`` (int)、``duration`` (float)。

        Raises
        ------
        ValueError
            缺少 ``video``、``video`` 非 :class:`VideoData`、视频无帧，
            或 ``scale_factor`` 非法。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入视频
            video = input_data.get("video")
            if not isinstance(video, VideoData):
                raise ValueError(
                    f"FrameInterpolator requires 'video' (VideoData), "
                    f"got {type(video).__name__ if video is not None else 'None'}."
                )

            frames: list[Any] = list(video.frames)
            # E4：插帧至少需要 2 帧（单帧无法插值）
            if len(frames) < 2:
                raise ValueError(
                    f"Frame interpolation requires at least 2 input frames, "
                    f"got {len(frames)}."
                )

            original_fps = video.fps
            if not isinstance(original_fps, (int, float)) or original_fps <= 0:
                original_fps = 30
                self._logger.warning(
                    "Invalid video fps; falling back to %d.", original_fps
                )
            original_fps = int(original_fps)
            original_frame_count = len(frames)

            self._logger.debug(
                "Input video: %d frames @ %d fps, frame size=%s.",
                original_frame_count,
                original_fps,
                frames[0].size if hasattr(frames[0], "size") else "unknown",
            )

            # 参数解析（target_fps 与 scale_factor 二选一，target_fps 优先）
            target_fps = input_data.get("target_fps")
            if target_fps is not None:
                target_fps = safe_int(target_fps, "target_fps")
            scale_factor = safe_int(input_data.get("scale_factor", 2), "scale_factor")

            num_passes, target_fps_used = self._resolve_num_passes(
                original_fps, scale_factor, target_fps
            )

            # 防御性：统一帧尺寸
            before_norm_sizes = {
                getattr(f, "size", None) for f in frames
            }
            frames = self._normalize_frame_sizes(frames)
            if len(before_norm_sizes) > 1:
                self._logger.debug(
                    "Normalized %d frames to a uniform size %s.",
                    original_frame_count,
                    frames[0].size if hasattr(frames[0], "size") else "unknown",
                )

            self._logger.info(
                "Interpolating video: method=%s, original_frames=%d, fps=%d, "
                "scale_factor=%d, target_fps=%s, passes=%d",
                self._method,
                original_frame_count,
                original_fps,
                scale_factor,
                target_fps_used,
                num_passes,
            )

            # 预估各 pass 的工作量，用于全局进度
            work_per_pass: list[int] = []
            n = original_frame_count
            total_work = 0
            for _ in range(num_passes):
                pairs = max(0, n - 1)
                work_per_pass.append(pairs)
                total_work += pairs
                n = max(1, 2 * n - 1)

            if total_work <= 0:
                self._logger.info(
                    "No interpolation needed (num_passes=0); "
                    "returning input frames unchanged."
                )
                self._emit_progress(0, 1, "No interpolation needed")
            else:
                self._emit_progress(0, total_work, "Starting interpolation")

            # 递归 2x：scale_factor=4 等价于两次 2x
            new_frames = frames
            done_work = 0
            for p in range(num_passes):
                frames_before = len(new_frames)
                self._logger.debug(
                    "Starting pass %d/%d (method=%s): %d frames -> "
                    "expecting %d frames.",
                    p + 1,
                    num_passes,
                    self._method,
                    frames_before,
                    2 * frames_before - 1,
                )
                new_frames = self._interpolate_2x(
                    new_frames,
                    progress_base=done_work,
                    progress_total=total_work,
                    pass_label=f"pass {p + 1}/{num_passes}",
                )
                done_work += work_per_pass[p]
                self._logger.info(
                    "Pass %d/%d complete: %d frames -> %d frames.",
                    p + 1,
                    num_passes,
                    frames_before,
                    len(new_frames),
                )

            new_frame_count = len(new_frames)
            new_fps = original_fps * (2 ** num_passes)

            self._logger.info(
                "Interpolation complete: %d frames @ %d fps -> "
                "%d frames @ %d fps (method=%s, passes=%d).",
                original_frame_count,
                original_fps,
                new_frame_count,
                new_fps,
                self._method,
                num_passes,
            )

            if total_work <= 0:
                self._emit_progress(1, 1, "No interpolation needed")

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        out_video = self._ensure_video_data(
            new_frames,
            new_fps,
            method=self._method,
            scale_factor=scale_factor,
            target_fps=target_fps_used,
            num_passes=num_passes,
            source="frame-interpolation",
        )
        duration = out_video.metadata.get("duration", 0.0)

        result = MosaicData(
            video=out_video,
            original_fps=original_fps,
            new_fps=new_fps,
            original_frame_count=original_frame_count,
            new_frame_count=new_frame_count,
            method=self._method,
            num_passes=num_passes,
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "method": self._method,
                "original_frame_count": original_frame_count,
                "new_frame_count": new_frame_count,
                "original_fps": original_fps,
                "new_fps": new_fps,
                "scale_factor": scale_factor,
                "num_passes": num_passes,
                "duration": duration,
            },
        )
        return result