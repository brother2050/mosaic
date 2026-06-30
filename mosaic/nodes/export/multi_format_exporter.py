# mosaic/nodes/export/multi_format_exporter.py
"""MultiFormatExporter 节点 —— 多格式导出。

将内容（视频/图像/音频/字幕）导出为多种目标格式。

设计要点
--------
* 纯工程节点，不涉及 AI 模型推理，不需要 GPU。
* 根据 ``content_type`` 分发到不同的导出逻辑：
    - 视频：复用 :class:`VideoEncoder`
    - 图像：使用 ``PIL.Image.save``
    - 音频：使用 ``soundfile.write``
    - 字幕：使用 ``BaseSubtitleNode`` 的格式转换
* 如果某格式不支持，记录警告并跳过，不中断整个流程。
* 文件命名规则：``{原文件名}_{format}.{ext}``
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。

支持格式
--------
- 视频：mp4, avi, webm, mov, mkv, gif
- 图像：png, jpg/jpeg, webp, bmp, tiff
- 音频：wav, mp3, flac, ogg
- 字幕：srt, vtt, ass, txt
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import (
    AudioData,
    ImageData,
    MosaicData,
    SubtitleData,
    VideoData,
)

__all__ = ["MultiFormatExporter"]


# 各内容类型支持的格式
_VIDEO_FORMATS = {"mp4", "avi", "webm", "mov", "mkv", "gif"}
_IMAGE_FORMATS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
_AUDIO_FORMATS = {"wav", "mp3", "flac", "ogg"}
_SUBTITLE_FORMATS = {"srt", "vtt", "ass", "txt"}

# PIL 格式名称映射
_PIL_FORMAT_MAP: dict[str, str] = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tiff": "TIFF",
}

# soundfile 格式映射
_SF_FORMAT_MAP: dict[str, str] = {
    "wav": "WAV",
    "mp3": "MP3",
    "flac": "FLAC",
    "ogg": "OGG",
}


@registry.register
class MultiFormatExporter(Node):
    """多格式导出节点。

    将内容导出为多种格式，支持视频、图像、音频和字幕。

    Parameters
    ----------
    bus:
        事件总线实例，``None`` 使用全局单例。

    Examples
    --------
    >>> exporter = MultiFormatExporter()
    >>> result = exporter(MosaicData(
    ...     content_type="video",
    ...     data=video_data,
    ...     formats=["mp4", "gif", "webm"],
    ...     output_dir="/tmp/outputs",
    ... ))
    >>> result["outputs"]  # {"mp4": "/tmp/outputs/output.mp4", ...}
    >>> result["total_files"]  # 3

    图像导出：
    >>> result = exporter(MosaicData(
    ...     content_type="image",
    ...     data=image_data,
    ...     formats=["png", "jpg", "webp"],
    ... ))
    """

    name: str = "multi-format-exporter"
    domain: str = "export"
    description: str = (
        "Export content to multiple formats (video/image/audio/subtitle). "
        "Supports batch conversion with graceful error handling."
    )
    version: str = "0.1.0"
    input_types: list[str] = [
        "video", "image", "audio", "subtitle", "mosaic",
    ]
    output_types: list[str] = ["file"]

    def __init__(
        self,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger(f"mosaic.nodes.export.{self.name}")

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载环境（无需加载模型，直接标记为已加载）。"""
        self._loaded = True
        self._logger.info("MultiFormatExporter ready.")

    def unload(self) -> None:
        """释放资源（无持久化资源需要释放）。"""
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行多格式导出。

        Parameters
        ----------
        input_data:
            必须包含：
            - ``content_type`` (str): "video"/"image"/"audio"/"subtitle"
            - ``data`` (VideoData/ImageData/AudioData/SubtitleData): 待导出数据
            - ``formats`` (list[str]): 目标格式列表
            可选：
            - ``output_dir`` (str): 输出目录，默认临时目录
            - ``quality`` (int): 质量参数

        Returns
        -------
        MosaicData
            包含 ``outputs`` (dict[str, str])、``total_files`` (int)、
            ``total_size`` (int)。

        Raises
        ------
        ValueError
            缺少必要字段或内容类型不支持。
        """
        self._emit_start()
        t0 = time.perf_counter()

        try:
            # 校验输入
            content_type = input_data.get("content_type")
            if not isinstance(content_type, str) or not content_type.strip():
                raise ValueError(
                    f"MultiFormatExporter requires 'content_type' (str), "
                    f"got {content_type!r}."
                )
            content_type = content_type.lower().strip()

            data = input_data.get("data")
            if data is None:
                raise ValueError(
                    "MultiFormatExporter requires 'data' (not None)."
                )

            formats = input_data.get("formats")
            if not isinstance(formats, list) or len(formats) == 0:
                raise ValueError(
                    f"MultiFormatExporter requires 'formats' (non-empty list), "
                    f"got {type(formats).__name__}."
                )

            output_dir = input_data.get("output_dir") or tempfile.gettempdir()
            output_dir = str(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            quality = input_data.get("quality", 23)

            self._logger.info(
                "Exporting %s to %d format(s): %s",
                content_type,
                len(formats),
                ", ".join(formats),
            )

            # 根据内容类型分发
            if content_type == "video":
                outputs = self._export_video(
                    data, formats, output_dir, quality
                )
            elif content_type == "image":
                outputs = self._export_image(
                    data, formats, output_dir, quality
                )
            elif content_type == "audio":
                outputs = self._export_audio(
                    data, formats, output_dir, quality
                )
            elif content_type == "subtitle":
                outputs = self._export_subtitle(
                    data, formats, output_dir
                )
            else:
                raise ValueError(
                    f"Unsupported content_type: {content_type!r}. "
                    f"Supported: video, image, audio, subtitle."
                )

            # 统计结果
            total_files = len(outputs)
            total_size = 0
            for path in outputs.values():
                if os.path.exists(path):
                    total_size += os.path.getsize(path)

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                outputs=outputs,
                total_files=total_files,
                total_size=total_size,
            )

            self._emit_complete(elapsed, {
                "total_files": total_files,
                "total_size": total_size,
                "content_type": content_type,
            })
            return result

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=None,  # 导出域无模型
        )

    # ------------------------------------------------------------------
    # 视频导出
    # ------------------------------------------------------------------
    def _export_video(
        self,
        data: Any,
        formats: list[str],
        output_dir: str,
        quality: int,
    ) -> dict[str, str]:
        """导出视频为多种格式。

        Parameters
        ----------
        data:
            ``VideoData`` 实例或包含 ``frames``/``fps`` 的 ``MosaicData``。
        formats:
            目标格式列表。
        output_dir:
            输出目录。
        quality:
            CRF 质量参数。

        Returns
        -------
        dict[str, str]
            格式 -> 文件路径 映射。
        """
        # 提取帧和帧率
        if isinstance(data, VideoData):
            frames = data.frames
            fps = data.fps
        elif isinstance(data, MosaicData):
            frames = data.get("frames", [])
            fps = data.get("fps", 30)
        else:
            raise TypeError(
                f"Video data must be VideoData or MosaicData, "
                f"got {type(data).__name__}."
            )

        if not frames:
            raise ValueError("Video data has no frames to export.")

        outputs: dict[str, str] = {}
        timestamp = int(time.time())

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt not in _VIDEO_FORMATS:
                self._logger.warning(
                    "Unsupported video format: %s, skipping.", fmt
                )
                continue

            output_path = os.path.join(
                output_dir, f"mosaic_export_{timestamp}.{fmt}"
            )

            try:
                if fmt == "gif":
                    self._export_gif(frames, fps, output_path)
                else:
                    # 复用 VideoEncoder
                    from mosaic.nodes.export.video_encoder import VideoEncoder

                    encoder = VideoEncoder(
                        format=fmt,
                        quality=quality,
                    )
                    encoder.load()
                    try:
                        result = encoder.run(MosaicData(
                            frames=frames,
                            fps=fps,
                            output_path=output_path,
                        ))
                        output_path = result["output_path"]
                    finally:
                        encoder.unload()

                outputs[fmt] = output_path
                self._logger.info("Exported video to %s: %s", fmt, output_path)

            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to export video as %s: %s. Skipping.",
                    fmt,
                    exc,
                )

        return outputs

    def _export_gif(
        self,
        frames: list[Any],
        fps: int,
        output_path: str,
    ) -> None:
        """将帧列表导出为 GIF 动画。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 帧列表。
        fps:
            帧率。
        output_path:
            输出文件路径。
        """
        from PIL import Image  # type: ignore

        if not frames:
            raise ValueError("No frames to export as GIF.")

        # 确保所有帧都是 PIL.Image
        pil_frames: list[Any] = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                pil_frames.append(frame.convert("P"))
            else:
                import numpy as np  # type: ignore

                if isinstance(frame, np.ndarray):
                    # 确保是 uint8，PIL Image.fromarray 对 float 会产生异常
                    if frame.dtype != np.uint8:
                        frame = np.clip(
                            frame * 255 if frame.max() <= 1.0 else frame,
                            0, 255,
                        ).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame).convert("P"))
                else:
                    raise TypeError(
                        f"Frame must be PIL.Image or numpy.ndarray, "
                        f"got {type(frame).__name__}."
                    )

        # GIF 帧间隔（毫秒）
        duration_ms = int(1000 / fps) if fps > 0 else 100

        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,  # 无限循环
            optimize=True,
        )

    # ------------------------------------------------------------------
    # 图像导出
    # ------------------------------------------------------------------
    def _export_image(
        self,
        data: Any,
        formats: list[str],
        output_dir: str,
        quality: int,
    ) -> dict[str, str]:
        """导出图像为多种格式。

        Parameters
        ----------
        data:
            ``ImageData`` 实例或 ``PIL.Image`` 或 ``MosaicData``。
        formats:
            目标格式列表。
        output_dir:
            输出目录。
        quality:
            JPEG/WebP 质量（1-100）。

        Returns
        -------
        dict[str, str]
            格式 -> 文件路径 映射。
        """
        from PIL import Image  # type: ignore

        # 提取 PIL.Image
        if isinstance(data, ImageData):
            img = data.image
        elif isinstance(data, Image.Image):
            img = data
        elif isinstance(data, MosaicData):
            img = data.get("image")
        else:
            raise TypeError(
                f"Image data must be ImageData, PIL.Image, or MosaicData, "
                f"got {type(data).__name__}."
            )

        if img is None:
            raise ValueError("Image data has no image to export.")

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Image must be PIL.Image.Image, got {type(img).__name__}."
            )

        outputs: dict[str, str] = {}
        timestamp = int(time.time())

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt not in _IMAGE_FORMATS:
                self._logger.warning(
                    "Unsupported image format: %s, skipping.", fmt
                )
                continue

            ext = "jpg" if fmt == "jpeg" else fmt
            output_path = os.path.join(
                output_dir, f"mosaic_export_{timestamp}.{ext}"
            )

            try:
                pil_format = _PIL_FORMAT_MAP.get(fmt, fmt.upper())
                save_kwargs: dict[str, Any] = {}

                # JPEG 不支持 RGBA
                if pil_format in ("JPEG", "BMP"):
                    save_img = img.convert("RGB")
                else:
                    save_img = img

                # 质量参数
                if pil_format in ("JPEG", "WEBP"):
                    save_kwargs["quality"] = max(1, min(100, quality * 2 + 30))
                if pil_format == "WEBP":
                    save_kwargs["method"] = 6  # 最高压缩

                save_img.save(
                    output_path,
                    format=pil_format,
                    **save_kwargs,
                )
                outputs[fmt] = output_path
                self._logger.info(
                    "Exported image to %s: %s", fmt, output_path
                )

            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to export image as %s: %s. Skipping.",
                    fmt,
                    exc,
                )

        return outputs

    # ------------------------------------------------------------------
    # 音频导出
    # ------------------------------------------------------------------
    def _export_audio(
        self,
        data: Any,
        formats: list[str],
        output_dir: str,
        quality: int,
    ) -> dict[str, str]:
        """导出音频为多种格式。

        Parameters
        ----------
        data:
            ``AudioData`` 实例或 ``MosaicData``。
        formats:
            目标格式列表。
        output_dir:
            输出目录。
        quality:
            质量参数（用于 MP3/OGG 比特率）。

        Returns
        -------
        dict[str, str]
            格式 -> 文件路径 映射。
        """
        import numpy as np  # type: ignore

        # 提取波形和采样率
        if isinstance(data, AudioData):
            waveform = data.waveform
            sample_rate = data.sample_rate
        elif isinstance(data, MosaicData):
            waveform = data.get("waveform")
            sample_rate = data.get("sample_rate", 22050)
        else:
            raise TypeError(
                f"Audio data must be AudioData or MosaicData, "
                f"got {type(data).__name__}."
            )

        if waveform is None:
            raise ValueError("Audio data has no waveform to export.")

        outputs: dict[str, str] = {}
        timestamp = int(time.time())

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt not in _AUDIO_FORMATS:
                self._logger.warning(
                    "Unsupported audio format: %s, skipping.", fmt
                )
                continue

            output_path = os.path.join(
                output_dir, f"mosaic_export_{timestamp}.{fmt}"
            )

            try:
                import soundfile as sf  # type: ignore

                # soundfile 期望 (samples, channels)
                wf = waveform
                if isinstance(wf, np.ndarray):
                    # 防御性 dtype 转换：soundfile 只支持 float32/float64/int16/int32
                    if wf.dtype not in (np.float32, np.float64, np.int16, np.int32):
                        wf = wf.astype(np.float32)
                    if wf.ndim == 2:
                        wf = wf.T  # (channels, samples) -> (samples, channels)

                sf_format = _SF_FORMAT_MAP.get(fmt, fmt.upper())

                # 质量参数：用于 MP3/OGG 比特率
                subtype = None
                if fmt == "mp3":
                    # MP3 比特率：quality 越低文件越大
                    bitrate = max(64, min(320, 320 - quality * 5))
                    subtype = f"MP3_{bitrate}K"  # type: ignore
                elif fmt == "flac":
                    subtype = "FLAC_24"  # 24-bit
                elif fmt == "wav":
                    subtype = "PCM_16"

                if subtype:
                    sf.write(output_path, wf, sample_rate, subtype=subtype)
                else:
                    sf.write(output_path, wf, sample_rate, format=sf_format)

                outputs[fmt] = output_path
                self._logger.info(
                    "Exported audio to %s: %s", fmt, output_path
                )

            except ImportError:
                self._logger.warning(
                    "soundfile not available, cannot export audio as %s.",
                    fmt,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to export audio as %s: %s. Skipping.",
                    fmt,
                    exc,
                )

        return outputs

    # ------------------------------------------------------------------
    # 字幕导出
    # ------------------------------------------------------------------
    def _export_subtitle(
        self,
        data: Any,
        formats: list[str],
        output_dir: str,
    ) -> dict[str, str]:
        """导出字幕为多种格式。

        Parameters
        ----------
        data:
            ``SubtitleData`` 实例或 ``MosaicData``。
        formats:
            目标格式列表。
        output_dir:
            输出目录。

        Returns
        -------
        dict[str, str]
            格式 -> 文件路径 映射。
        """
        from mosaic.nodes.subtitle._base import BaseSubtitleNode

        # 提取字幕片段
        if isinstance(data, SubtitleData):
            segments = data.segments
        elif isinstance(data, MosaicData):
            segments = data.get("segments", [])
        else:
            raise TypeError(
                f"Subtitle data must be SubtitleData or MosaicData, "
                f"got {type(data).__name__}."
            )

        if not segments:
            raise ValueError("Subtitle data has no segments to export.")

        outputs: dict[str, str] = {}
        timestamp = int(time.time())

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt not in _SUBTITLE_FORMATS:
                self._logger.warning(
                    "Unsupported subtitle format: %s, skipping.", fmt
                )
                continue

            output_path = os.path.join(
                output_dir, f"mosaic_export_{timestamp}.{fmt}"
            )

            try:
                content = self._subtitle_to_format(segments, fmt)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

                outputs[fmt] = output_path
                self._logger.info(
                    "Exported subtitle to %s: %s", fmt, output_path
                )

            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to export subtitle as %s: %s. Skipping.",
                    fmt,
                    exc,
                )

        return outputs

    @staticmethod
    def _subtitle_to_format(
        segments: list[dict[str, Any]],
        fmt: str,
    ) -> str:
        """将字幕片段转为指定格式字符串。

        Parameters
        ----------
        segments:
            字幕片段列表。
        fmt:
            目标格式：srt/vtt/ass/txt。

        Returns
        -------
        str
            格式化后的字幕内容。
        """
        from mosaic.nodes.subtitle._base import BaseSubtitleNode

        if fmt == "srt":
            return BaseSubtitleNode._to_srt(segments)
        elif fmt == "vtt":
            return BaseSubtitleNode._to_vtt(segments)
        elif fmt == "txt":
            # 纯文本：仅保留文本内容
            lines: list[str] = []
            for seg in segments:
                lines.append(seg.get("text", "").strip())
            return "\n".join(lines)
        elif fmt == "ass":
            # ASS 格式（简化版）
            header = (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "PlayResX: 1920\n"
                "PlayResY: 1080\n\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
                "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
                "MarginV, Encoding\n"
                "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
                "&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1\n\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, "
                "MarginL, MarginR, MarginV, Effect, Text\n"
            )
            lines = [header]
            for seg in segments:
                start = MultiFormatExporter._format_ass_time(seg["start"])
                end = MultiFormatExporter._format_ass_time(seg["end"])
                text = seg.get("text", "").replace("\n", "\\N")
                lines.append(
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported subtitle format: {fmt}")

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """格式化 ASS 时间戳 ``H:MM:SS.cc``。"""
        if seconds < 0:
            seconds = 0.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds * 100) % 100)
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"

    # ------------------------------------------------------------------
    # 事件发射辅助
    # ------------------------------------------------------------------
    def _emit_start(self) -> None:
        """发出 node_start 事件。"""
        self._bus.emit(
            EventType.NODE_START,
            node_name=self.name,
            node_domain=self.domain,
        )

    def _emit_complete(self, duration: float, output_summary: Any) -> None:
        """发出 node_complete 事件。"""
        self._bus.emit(
            EventType.NODE_COMPLETE,
            node_name=self.name,
            duration=duration,
            output_summary=output_summary,
        )

    def _emit_error(self, error: BaseException) -> None:
        """发出 node_error 事件。"""
        self._bus.emit(
            EventType.NODE_ERROR,
            node_name=self.name,
            error=error,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<MultiFormatExporter name={self.name!r} state={status}>"
        )
