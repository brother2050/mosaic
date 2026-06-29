# mosaic/nodes/video/frame_extractor.py
"""FrameExtractor 节点 —— 视频拆帧。

从视频中提取帧，支持四种模式：

* ``all``        —— 提取全部帧
* ``interval``   —— 按帧间隔提取（每隔 ``interval`` 帧取一帧）
* ``keyframe``   —— 仅提取关键帧（基于相邻帧像素差异阈值判断）
* ``timestamps`` —— 按时间戳列表提取指定帧

设计要点
--------
* 拆帧无需任何模型，``_load_model`` 直接返回；``load`` 仍由调度器统一
  管理，以保持与其他视频节点一致的按需加载语义与生命周期事件。
* 输入既可是 :class:`~mosaic.core.types.VideoData`，也可是视频文件路径
  （``str``）。为路径时使用 :meth:`BaseVideoNode._load_video` 加载为
  VideoData。
* 大视频友好：``keyframe`` 模式在路径输入下采用逐帧流式读取
  （:meth:`_iter_frames_from_path`），仅保留关键帧，避免一次性载入全部
  帧导致内存峰值过高。``all`` / ``interval`` / ``timestamps`` 模式需要
  返回全部或大部分帧，故沿用整段加载。
* ``output_format`` 控制帧的返回形态：
  - ``"pil"``   —— ``PIL.Image`` 列表（默认）
  - ``"numpy"`` —— ``numpy.ndarray`` 列表
  - ``"path"``  —— 将帧落盘为临时图片文件，返回路径列表
* ``keyframe`` 模式使用简单的图像差异阈值判断：将相邻帧转为灰度数组，
  计算平均绝对像素差异，差异大于阈值则视为关键帧（首帧恒为关键帧）。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start``
  / ``node_complete`` / ``node_error`` / ``progress`` 事件。
* ``PIL`` / ``numpy`` / ``imageio`` 均为惰性导入，未安装时模块仍可被
  注册表发现与导入（仅在实际拆帧时报依赖缺失）。

许可证
------
* 无外部模型依赖。
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import MosaicData, VideoData

from mosaic.nodes.video._base import BaseVideoNode

__all__ = ["FrameExtractor"]


# 支持的拆帧模式
_VALID_MODES: tuple[str, ...] = ("all", "interval", "keyframe", "timestamps")

# keyframe 模式默认像素差异阈值（0-255 尺度，平均绝对差异）
_DEFAULT_KEYFRAME_THRESHOLD: float = 10.0

# 流式处理时的进度上报间隔（每处理多少帧上报一次）
_DEFAULT_PROGRESS_EVERY: int = 100


@registry.register
class FrameExtractor(BaseVideoNode):
    """视频拆帧节点。

    从视频中按模式提取帧，输出帧列表及其时间戳、帧率与时长。拆帧不依赖
    任何模型，``_load_model`` 为空实现；模型生命周期仍由调度器统一管理
    以保持节点语义一致。

    Parameters
    ----------
    model:
        模型标识，拆帧节点不使用模型，默认 ``""``。
    device:
        推理设备，默认 ``"cuda"``（拆帧实际不使用，仅为保持接口一致）。
    dtype:
        推理精度，默认 ``"float16"``（同上）。
    **kwargs:
        透传给 :class:`BaseVideoNode` 的参数。

    Examples
    --------
    提取全部帧：

    >>> extractor = FrameExtractor()
    >>> result = extractor(MosaicData(video="/path/to/video.mp4", mode="all"))
    >>> result["frames"], result["frame_count"], result["fps"]

    按间隔提取（每隔 5 帧取一帧）：

    >>> result = extractor(MosaicData(video=video_data, mode="interval", interval=5))

    提取关键帧（路径输入自动流式处理，内存友好）：

    >>> result = extractor(MosaicData(video="/path/to/video.mp4", mode="keyframe"))

    按时间戳提取并以 numpy 数组形式返回：

    >>> result = extractor(MosaicData(
    ...     video=video_data,
    ...     mode="timestamps",
    ...     timestamps=[1.0, 2.5, 4.0],
    ...     output_format="numpy",
    ... ))
    """

    name: str = "frame-extractor"
    description: str = (
        "Extract frames from a video. Supports 'all', 'interval', "
        "'keyframe' (pixel-diff based), and 'timestamps' modes, with "
        "PIL / numpy / file-path output formats."
    )
    version: str = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["image"]

    def __init__(
        self,
        model: str = "",
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, dtype=dtype, **kwargs)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """拆帧无需加载模型，直接返回。

        本方法为空实现以满足 :class:`BaseVideoNode` 的抽象接口；调度器
        仍会调用 ``load`` 完成状态登记与事件发布。
        """
        self._logger.debug("FrameExtractor requires no model; skipping load.")
        return

    def _build_model_info(self) -> dict[str, Any]:
        """构造模型信息字典。

        拆帧节点无模型，显存占用记为 0，避免调度器误判并淘汰其他节点。
        """
        return {
            "name": self._model_name or "frame-extractor",
            "source": "builtin",
            "license": "N/A",
            "vram_gb": 0.0,
            "dtype": self._dtype_str,
            "device": self._device,
        }

    # ------------------------------------------------------------------
    # 视频加载辅助
    # ------------------------------------------------------------------
    def _resolve_video(self, video: Any) -> VideoData:
        """将输入解析为 :class:`VideoData`。

        若 ``video`` 为字符串路径，调用 :meth:`BaseVideoNode._load_video`
        加载为 :class:`VideoData`；若已是 :class:`VideoData` 则原样返回。

        Parameters
        ----------
        video:
            ``VideoData`` 或视频文件路径。

        Returns
        -------
        VideoData
            解析后的视频数据。

        Raises
        ------
        ValueError
            ``video`` 缺失或类型不支持。
        """
        if video is None:
            raise ValueError(
                "FrameExtractor requires 'video' (VideoData or str path), "
                "got None."
            )
        if isinstance(video, VideoData):
            return video
        if isinstance(video, str):
            return self._load_video(video)
        raise ValueError(
            f"FrameExtractor requires 'video' (VideoData or str path), "
            f"got {type(video).__name__}."
        )

    @staticmethod
    def _iter_frames_from_path(path: str) -> Iterator[Any]:
        """逐帧流式读取视频文件（用于大视频，避免一次性载入全部帧）。

        使用 ``imageio`` 的 reader 迭代器逐帧 yield ``PIL.Image``，读取
        完毕自动关闭 reader。仅在 ``keyframe`` 模式 + 路径输入时使用，
        以在检测关键帧的同时仅保留关键帧、控制内存峰值。

        Parameters
        ----------
        path:
            视频文件路径。

        Yields
        ------
        PIL.Image
            单帧图像。

        Raises
        ------
        ImportError
            未安装 ``imageio``。
        FileNotFoundError
            文件不存在。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        import imageio.v2 as imageio  # type: ignore
        from PIL import Image  # type: ignore

        reader = imageio.get_reader(path)
        try:
            for frame in reader:
                # imageio 返回 numpy 数组 (H, W, C)
                yield Image.fromarray(frame)
        finally:
            reader.close()

    @staticmethod
    def _read_video_meta(path: str) -> tuple[int, int]:
        """读取视频文件的帧数与帧率（不载入帧数据）。

        优先取 ``imageio`` 元数据中的 ``nframes``；缺失时尝试
        ``count_frames()``；均不可得时帧数返回 0，帧率返回 30。

        Parameters
        ----------
        path:
            视频文件路径。

        Returns
        -------
        tuple[int, int]
            ``(frame_count, fps)``。
        """
        import imageio.v2 as imageio  # type: ignore

        reader = imageio.get_reader(path)
        try:
            meta = reader.get_meta_data()
            fps = int(meta.get("fps", 30) or 30)
            if fps <= 0:
                fps = 30

            count = 0
            nframes_meta = meta.get("nframes")
            if isinstance(nframes_meta, int) and nframes_meta > 0:
                count = nframes_meta
            else:
                try:
                    count = int(reader.count_frames())  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    count = 0
            return count, fps
        finally:
            reader.close()

    # ------------------------------------------------------------------
    # 各模式拆帧
    # ------------------------------------------------------------------
    def _extract_all(
        self,
        video_data: VideoData,
        total: int,
    ) -> tuple[list[Any], list[float]]:
        """提取全部帧。

        Parameters
        ----------
        video_data:
            视频数据。
        total:
            进度总量（仅用于事件，``<=0`` 时不发进度）。

        Returns
        -------
        tuple[list[Any], list[float]]
            ``(frames, timestamps)``，时间戳为 ``i / fps``。
        """
        frames = list(video_data.frames)
        fps = video_data.fps
        timestamps = [i / fps for i in range(len(frames))] if fps > 0 else []
        if total > 0:
            self._emit_progress(len(frames), total, "Extracted all frames")
        return frames, timestamps

    def _extract_interval(
        self,
        video_data: VideoData,
        interval: int,
        total: int,
    ) -> tuple[list[Any], list[float]]:
        """按间隔提取帧（每隔 ``interval`` 帧取一帧）。

        Parameters
        ----------
        video_data:
            视频数据。
        interval:
            帧间隔，``<=0`` 时按 1 处理。
        total:
            进度总量。

        Returns
        -------
        tuple[list[Any], list[float]]
            ``(frames, timestamps)``。
        """
        step = max(1, int(interval))
        frames = video_data.frames
        fps = video_data.fps
        indices = list(range(0, len(frames), step))
        out_frames = [frames[i] for i in indices]
        timestamps = (
            [i / fps for i in indices] if fps > 0 else [0.0] * len(indices)
        )
        if total > 0:
            self._emit_progress(
                len(indices), total, f"Extracted every {step}th frame"
            )
        return out_frames, timestamps

    def _extract_timestamps(
        self,
        video_data: VideoData,
        timestamps_in: list[float],
        total: int,
    ) -> tuple[list[Any], list[float]]:
        """按时间戳列表提取对应帧。

        每个时间戳 ``t`` 映射到帧索引 ``int(t * fps)``，越界时钳制到
        ``[0, len(frames)-1]``。返回的时间戳为实际命中帧的时间戳。

        Parameters
        ----------
        video_data:
            视频数据。
        timestamps_in:
            时间戳列表（秒）。
        total:
            进度总量。

        Returns
        -------
        tuple[list[Any], list[float]]
            ``(frames, timestamps)``。
        """
        frames = video_data.frames
        fps = video_data.fps
        n = len(frames)
        out_frames: list[Any] = []
        out_ts: list[float] = []
        for t in timestamps_in:
            if fps > 0:
                idx = int(float(t) * fps)
            else:
                idx = 0
            if idx < 0:
                idx = 0
            elif n > 0 and idx >= n:
                idx = n - 1
            if n > 0:
                out_frames.append(frames[idx])
            out_ts.append(idx / fps if fps > 0 else 0.0)
        if total > 0:
            self._emit_progress(
                len(out_frames), total, "Extracted frames at timestamps"
            )
        return out_frames, out_ts

    def _extract_keyframe_streaming(
        self,
        path: str,
        threshold: float,
        fps: int,
        total: int,
    ) -> tuple[list[Any], list[float], int]:
        """流式提取关键帧（路径输入，逐帧读取，内存友好）。

        逐帧读取视频，将每帧转为灰度数组，计算与上一帧的平均绝对像素
        差异，差异大于 ``threshold`` 则视为关键帧。首帧恒为关键帧。仅
        保留关键帧，避免大视频一次性载入全部帧。

        Parameters
        ----------
        path:
            视频文件路径。
        threshold:
            像素差异阈值（0-255 尺度）。
        fps:
            视频帧率，用于计算时间戳。
        total:
            进度总量（``<=0`` 时不发进度）。

        Returns
        -------
        tuple[list[Any], list[float], int]
            ``(keyframes, timestamps, total_scanned)``，其中
            ``total_scanned`` 为源视频总帧数。
        """
        import numpy as np  # type: ignore

        keyframes: list[Any] = []
        timestamps: list[float] = []
        prev_gray: Any | None = None
        idx = 0

        for frame in self._iter_frames_from_path(path):
            cur_gray = self._to_gray_array(frame)

            is_key = False
            if prev_gray is None:
                is_key = True
            else:
                diff = float(
                    np.mean(
                        np.abs(
                            cur_gray.astype(np.int32)
                            - prev_gray.astype(np.int32)
                        )
                    )
                )
                if diff > threshold:
                    is_key = True

            if is_key:
                keyframes.append(frame)
                timestamps.append(idx / fps if fps > 0 else 0.0)

            prev_gray = cur_gray
            idx += 1

            if total > 0 and idx % _DEFAULT_PROGRESS_EVERY == 0:
                self._emit_progress(
                    idx,
                    total,
                    f"Streamed {idx} frames, {len(keyframes)} keyframes",
                )

        if total > 0:
            self._emit_progress(
                max(idx, total), total, "Keyframe detection complete"
            )
        return keyframes, timestamps, idx

    def _extract_keyframe_inmemory(
        self,
        video_data: VideoData,
        threshold: float,
        total: int,
    ) -> tuple[list[Any], list[float]]:
        """在已载入的 :class:`VideoData` 上提取关键帧。

        与 :meth:`_extract_keyframe_streaming` 算法一致，但作用于内存中
        已有的帧列表，用于 ``VideoData`` 输入场景。

        Parameters
        ----------
        video_data:
            视频数据。
        threshold:
            像素差异阈值（0-255 尺度）。
        total:
            进度总量。

        Returns
        -------
        tuple[list[Any], list[float]]
            ``(keyframes, timestamps)``。
        """
        import numpy as np  # type: ignore

        frames = video_data.frames
        fps = video_data.fps
        n = len(frames)

        keyframes: list[Any] = []
        timestamps: list[float] = []
        prev_gray: Any | None = None

        for i, frame in enumerate(frames):
            cur_gray = self._to_gray_array(frame)

            is_key = False
            if prev_gray is None:
                is_key = True
            else:
                diff = float(
                    np.mean(
                        np.abs(
                            cur_gray.astype(np.int32)
                            - prev_gray.astype(np.int32)
                        )
                    )
                )
                if diff > threshold:
                    is_key = True

            if is_key:
                keyframes.append(frame)
                timestamps.append(i / fps if fps > 0 else 0.0)

            prev_gray = cur_gray

            if total > 0 and (i + 1) % _DEFAULT_PROGRESS_EVERY == 0:
                self._emit_progress(
                    i + 1,
                    total,
                    f"Scanned {i + 1}/{n}, {len(keyframes)} keyframes",
                )

        if total > 0:
            self._emit_progress(n, total, "Keyframe detection complete")
        return keyframes, timestamps

    @staticmethod
    def _to_gray_array(frame: Any) -> Any:
        """将一帧转为灰度 numpy 数组（``uint8``），用于差异计算。

        Parameters
        ----------
        frame:
            ``PIL.Image`` 或 numpy 数组。

        Returns
        -------
        numpy.ndarray
            二维灰度数组，``dtype=uint8``。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        if isinstance(frame, Image.Image):
            return np.array(frame.convert("L"), dtype=np.uint8)

        arr = np.asarray(frame)
        if arr.ndim == 3:
            # 加权灰度化（与 PIL "L" 一致）
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            arr = (arr[..., :3].astype(np.float32) @ weights).astype(np.uint8)
        return arr

    # ------------------------------------------------------------------
    # 输出格式转换
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_output_format(
        frames: list[Any],
        output_format: str,
    ) -> list[Any]:
        """按 ``output_format`` 转换帧列表的返回形态。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 帧列表。
        output_format:
            ``"pil"`` / ``"numpy"`` / ``"path"``。

        Returns
        -------
        list[Any]
            转换后的列表：``PIL.Image`` / ``numpy.ndarray`` / 路径字符串。

        Raises
        ------
        ValueError
            ``output_format`` 非法（正常由 ``run`` 提前校验，不会到达）。
        """
        if output_format == "pil":
            return list(frames)

        if output_format == "numpy":
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore

            out: list[Any] = []
            for f in frames:
                if isinstance(f, Image.Image):
                    out.append(np.array(f.convert("RGB")))
                else:
                    out.append(np.asarray(f))
            return out

        if output_format == "path":
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore

            tmp_dir = tempfile.mkdtemp(prefix="mosaic_frames_")
            paths: list[Any] = []
            for i, f in enumerate(frames):
                fname = os.path.join(tmp_dir, f"frame_{i:06d}.png")
                if isinstance(f, Image.Image):
                    f.convert("RGB").save(fname, format="PNG")
                else:
                    Image.fromarray(np.asarray(f)).save(fname, format="PNG")
                paths.append(fname)
            return paths

        raise ValueError(
            f"Unsupported output_format: {output_format!r}. "
            f"Supported: 'pil' / 'numpy' / 'path'."
        )

    # ------------------------------------------------------------------
    # Node 执行
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行视频拆帧。

        Parameters
        ----------
        input_data:
            必须包含 ``video`` (:class:`VideoData` 或 ``str`` 路径)；可选：

            * ``mode`` (str, 默认 ``"all"``) —— 拆帧模式，可选
              ``"all"`` / ``"interval"`` / ``"keyframe"`` / ``"timestamps"``。
            * ``interval`` (int, 默认 ``1``) —— ``mode="interval"`` 时生效，
              每隔 ``interval`` 帧取一帧。
            * ``timestamps`` (list[float]) —— ``mode="timestamps"`` 时必填，
              按时间戳（秒）提取对应帧。
            * ``output_format`` (str, 默认 ``"pil"``) —— 帧返回形态，可选
              ``"pil"`` / ``"numpy"`` / ``"path"``。
            * ``keyframe_threshold`` (float, 默认 ``10.0``) ——
              ``mode="keyframe"`` 时生效，像素差异阈值（0-255）。

        Returns
        -------
        MosaicData
            包含 ``frames`` (list)、``frame_count`` (int, 提取到的帧数)、
            ``timestamps`` (list[float], 各帧时间戳)、``fps`` (int)、
            ``duration`` (float, 源视频总时长，秒)。

        Raises
        ------
        ValueError
            缺少 ``video``、``video`` 类型不支持、``mode`` 非法、
            ``mode="timestamps"`` 时未提供非空 ``timestamps``，或
            ``output_format`` 非法。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验 mode
            mode = input_data.get("mode", "all")
            if not isinstance(mode, str) or mode not in _VALID_MODES:
                raise ValueError(
                    f"Unsupported mode: {mode!r}. "
                    f"Supported modes: {_VALID_MODES}."
                )

            # 校验 output_format
            output_format = input_data.get("output_format", "pil")
            if output_format not in ("pil", "numpy", "path"):
                raise ValueError(
                    f"Unsupported output_format: {output_format!r}. "
                    f"Supported: 'pil' / 'numpy' / 'path'."
                )

            video_input = input_data.get("video")

            # 输出默认值
            frames: list[Any] = []
            timestamps: list[float] = []
            fps: int = 30
            duration: float = 0.0

            # keyframe + 路径：逐帧流式读取，避免一次性载入全部帧（大视频友好）
            if mode == "keyframe" and isinstance(video_input, str):
                threshold = float(
                    input_data.get(
                        "keyframe_threshold", _DEFAULT_KEYFRAME_THRESHOLD
                    )
                )
                hint, meta_fps = self._read_video_meta(video_input)
                if meta_fps > 0:
                    fps = int(meta_fps)
                total = hint if hint > 0 else 0

                self._logger.info(
                    "Extracting keyframes (streaming): source=%s, "
                    "total_frames_hint=%d, fps=%d, threshold=%.2f",
                    video_input,
                    hint,
                    fps,
                    threshold,
                )
                if total > 0:
                    self._emit_progress(
                        0, total, "Starting keyframe streaming"
                    )

                keyframes, timestamps, total_scanned = (
                    self._extract_keyframe_streaming(
                        video_input, threshold, fps, total
                    )
                )
                frames = keyframes
                duration = (
                    (total_scanned / fps) if fps > 0 else 0.0
                )
            else:
                # 整段加载：路径用 _load_video，VideoData 原样使用
                video_data = self._resolve_video(video_input)
                fps = (
                    int(video_data.fps)
                    if video_data.fps and video_data.fps > 0
                    else 30
                )
                total_frames = len(video_data.frames)
                duration = (
                    (total_frames / fps) if fps > 0 else 0.0
                )
                total = total_frames

                self._logger.info(
                    "Extracting frames: mode=%s, total_frames=%d, fps=%d, "
                    "output_format=%s",
                    mode,
                    total_frames,
                    fps,
                    output_format,
                )
                if total > 0:
                    self._emit_progress(
                        0, total, f"Starting {mode} extraction"
                    )

                if mode == "all":
                    frames, timestamps = self._extract_all(
                        video_data, total
                    )
                elif mode == "interval":
                    interval = int(input_data.get("interval", 1))
                    frames, timestamps = self._extract_interval(
                        video_data, interval, total
                    )
                elif mode == "timestamps":
                    ts_input = input_data.get("timestamps")
                    if not isinstance(ts_input, list) or not ts_input:
                        raise ValueError(
                            "mode='timestamps' requires a non-empty "
                            "'timestamps' (list[float])."
                        )
                    ts_list = [float(t) for t in ts_input]
                    frames, timestamps = self._extract_timestamps(
                        video_data, ts_list, total
                    )
                else:  # mode == "keyframe"（VideoData 输入）
                    threshold = float(
                        input_data.get(
                            "keyframe_threshold",
                            _DEFAULT_KEYFRAME_THRESHOLD,
                        )
                    )
                    frames, timestamps = self._extract_keyframe_inmemory(
                        video_data, threshold, total
                    )

            # 输出格式转换
            frames = self._convert_output_format(frames, output_format)

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        result = MosaicData(
            frames=frames,
            frame_count=len(frames),
            timestamps=timestamps,
            fps=fps,
            duration=duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "mode": mode,
                "frame_count": len(frames),
                "fps": fps,
                "duration": duration,
                "output_format": output_format,
            },
        )
        return result
