# mosaic/nodes/export/video_encoder.py
"""VideoEncoder 节点 —— 视频编码封装。

将帧列表或图片序列编码为标准视频文件，支持音视频合并与字幕烧录。

设计要点
--------
* 纯工程节点，不涉及 AI 模型推理，不需要 GPU。
* 优先使用 ``imageio-ffmpeg`` 获取 FFmpeg 二进制路径，通过
  ``subprocess`` 调用 FFmpeg 进行编码、音频合并与字幕烧录。
* 通过 ``stdin pipe`` 传入帧数据（rawvideo 格式），避免中间文件。
* 所有临时文件使用 ``tempfile`` 管理，确保退出时清理。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。

支持格式
--------
- **mp4**：H.264/H.265 视频，最通用格式
- **avi**：传统格式，兼容性好但压缩率低
- **webm**：VP8/VP9 编码，适合网页
- **mov**：QuickTime 格式，Apple 生态
- **mkv**：Matroska 容器，支持多音轨/字幕

CRF 质量推荐值
--------------
- 18：视觉无损（大文件）
- 23：默认，质量与体积平衡
- 28：可接受质量（小文件）
- 32+：低质量（极小文件）

编码预设
--------
- ``ultrafast``：最快编码，压缩率最低
- ``fast``：快速编码
- ``medium``：默认平衡
- ``slow``：高压缩率
- ``veryslow``：最高压缩率，编码最慢
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData, SubtitleData

__all__ = ["VideoEncoder"]


# 支持的输出格式与对应的默认编码器
_FORMAT_CODEC_MAP: dict[str, str] = {
    "mp4": "libx264",
    "avi": "mpeg4",
    "webm": "libvpx-vp9",
    "mov": "libx264",
    "mkv": "libx264",
}

# 格式与文件扩展名映射
_FORMAT_EXTENSIONS: dict[str, str] = {
    "mp4": "mp4",
    "avi": "avi",
    "webm": "webm",
    "mov": "mov",
    "mkv": "mkv",
    "gif": "gif",
}

# 有效预设列表
_VALID_PRESETS = {"ultrafast", "fast", "medium", "slow", "veryslow"}


@registry.register
class VideoEncoder(Node):
    """视频编码封装节点。

    将 ``PIL.Image`` 帧列表编码为标准视频文件，支持音视频合并与
    字幕烧录。

    Parameters
    ----------
    format:
        输出格式，默认 ``"mp4"``。支持 mp4/avi/webm/mov/mkv。
    codec:
        视频编码器，默认 ``"libx264"``。``None`` 时按格式自动选择。
    quality:
        CRF 质量参数（0-51），越小质量越高，默认 ``23``。
    preset:
        编码预设，可选 ultrafast/fast/medium/slow/veryslow，默认
        ``"medium"``。
    audio_codec:
        音频编码器，默认 ``"aac"``。
    pixel_format:
        像素格式，默认 ``"yuv420p"``（兼容性最好）。
    bus:
        事件总线实例，``None`` 使用全局单例。

    Examples
    --------
    >>> encoder = VideoEncoder(format="mp4", quality=20, preset="slow")
    >>> result = encoder(MosaicData(
    ...     frames=[frame1, frame2, ...],
    ...     fps=30,
    ... ))
    >>> result["output_path"]  # /tmp/mosaic_xxx.mp4

    带音频合并：
    >>> result = encoder(MosaicData(
    ...     frames=frames,
    ...     fps=30,
    ...     audio=audio_data,
    ... ))

    带字幕烧录：
    >>> result = encoder(MosaicData(
    ...     frames=frames,
    ...     fps=30,
    ...     subtitle=subtitle_data,
    ... ))
    """

    name: str = "video-encoder"
    domain: str = "export"
    description: str = (
        "Encode frames into a video file (mp4/avi/webm/mov/mkv). "
        "Supports audio merging and subtitle burning via FFmpeg."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["video", "image", "mosaic"]
    output_types: list[str] = ["file"]

    def __init__(
        self,
        format: str = "mp4",
        codec: str | None = None,
        quality: int = 23,
        preset: str = "medium",
        audio_codec: str | None = "aac",
        pixel_format: str = "yuv420p",
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._format: str = format.lower()
        self._codec: str | None = codec
        self._quality: int = max(0, min(51, quality))
        self._preset: str = preset if preset in _VALID_PRESETS else "medium"
        self._audio_codec: str | None = audio_codec
        self._pixel_format: str = pixel_format
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger(f"mosaic.nodes.export.{self.name}")
        self._ffmpeg_path: str | None = None

    # ------------------------------------------------------------------
    # Node 接口实现
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载环境：检测 FFmpeg 可用性。

        导出域节点不需要加载 AI 模型，仅检测 FFmpeg 是否可用。
        """
        self._ffmpeg_path = self._find_ffmpeg()
        if self._ffmpeg_path is None:
            raise ImportError(
                "FFmpeg is required for video encoding. "
                "Install via `pip install imageio-ffmpeg` or install "
                "FFmpeg system-wide and ensure it is in PATH."
            )
        self._logger.info("FFmpeg found at: %s", self._ffmpeg_path)
        self._loaded = True

    def unload(self) -> None:
        """释放资源（无持久化资源需要释放）。"""
        self._ffmpeg_path = None
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行视频编码。

        Parameters
        ----------
        input_data:
            必须包含 ``frames`` (list[PIL.Image]) 和 ``fps`` (int)。
            可选：``audio`` (AudioData)、``output_path`` (str)、
            ``bitrate`` (str)、``subtitle`` (SubtitleData)。

        Returns
        -------
        MosaicData
            包含 ``output_path``/``format``/``codec``/``duration``/
            ``file_size``/``resolution``。

        Raises
        ------
        ValueError
            缺少 ``frames`` 或 ``fps``。
        """
        self._emit_start()
        t0 = time.perf_counter()

        try:
            # 校验输入
            frames = input_data.get("frames")
            if not isinstance(frames, list) or len(frames) == 0:
                raise ValueError(
                    f"VideoEncoder requires 'frames' (non-empty list), "
                    f"got {type(frames).__name__}."
                )

            fps = input_data.get("fps")
            if not isinstance(fps, (int, float)) or fps <= 0:
                raise ValueError(
                    f"VideoEncoder requires 'fps' (positive number), "
                    f"got {fps!r}."
                )
            fps = int(fps)

            audio = input_data.get("audio")
            subtitle = input_data.get("subtitle")
            output_path = input_data.get("output_path")
            bitrate = input_data.get("bitrate")

            # 确定输出路径
            if output_path is None:
                ext = _FORMAT_EXTENSIONS.get(self._format, "mp4")
                tmp_dir = tempfile.gettempdir()
                output_path = os.path.join(
                    tmp_dir, f"mosaic_export_{int(time.time())}.{ext}"
                )
            output_path = str(output_path)

            # 确保输出目录存在
            dir_path = os.path.dirname(output_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # 确定编码器
            codec = self._codec or _FORMAT_CODEC_MAP.get(self._format, "libx264")

            # 检查帧尺寸，确保偶数
            first_frame = frames[0]
            width, height = self._get_frame_size(first_frame)
            width, height = self._ensure_even(width, height)

            # 编码视频
            self._logger.info(
                "Encoding %d frames to %s (codec=%s, %dx%d, fps=%d)...",
                len(frames),
                output_path,
                codec,
                width,
                height,
                fps,
            )

            # 如果有音频或字幕，先生成无音视频再合并
            if audio is not None or subtitle is not None:
                # 先编码纯视频到临时文件
                tmp_video = tempfile.mktemp(suffix=f".{self._format}")
                self._encode_frames_to_file(
                    frames, fps, width, height, tmp_video, codec, bitrate
                )

                # 合并音频/烧录字幕
                try:
                    self._merge_av_subtitle(
                        tmp_video, audio, subtitle, output_path, codec, bitrate
                    )
                finally:
                    if os.path.exists(tmp_video):
                        os.unlink(tmp_video)
            else:
                # 直接编码
                self._encode_frames_to_file(
                    frames, fps, width, height, output_path, codec, bitrate
                )

            # 获取输出信息
            file_size = os.path.getsize(output_path)
            duration = len(frames) / fps if fps > 0 else 0.0

            elapsed = time.perf_counter() - t0
            result = MosaicData(
                output_path=output_path,
                format=self._format,
                codec=codec,
                duration=duration,
                file_size=file_size,
                resolution=(width, height),
            )

            self._emit_complete(elapsed, {
                "output_path": output_path,
                "format": self._format,
                "file_size": file_size,
                "duration": duration,
            })
            return result

        except Exception as exc:
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
    # FFmpeg 相关工具
    # ------------------------------------------------------------------
    @staticmethod
    def _find_ffmpeg() -> str | None:
        """查找 FFmpeg 可执行文件路径。

        优先使用 ``imageio-ffmpeg`` 提供的 FFmpeg 二进制；
        其次检查系统 PATH 中的 ``ffmpeg``。

        Returns
        -------
        str | None
            FFmpeg 可执行文件路径，未找到返回 ``None``。
        """
        # 尝试 imageio-ffmpeg
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        # 尝试系统 FFmpeg
        import shutil

        return shutil.which("ffmpeg")

    @staticmethod
    def _get_frame_size(frame: Any) -> tuple[int, int]:
        """获取帧的尺寸 ``(width, height)``。"""
        from PIL import Image  # type: ignore

        if isinstance(frame, Image.Image):
            return frame.size
        import numpy as np  # type: ignore

        if isinstance(frame, np.ndarray):
            h, w = frame.shape[:2]
            return w, h
        raise TypeError(
            f"Frame must be PIL.Image or numpy.ndarray, "
            f"got {type(frame).__name__}."
        )

    @staticmethod
    def _ensure_even(width: int, height: int) -> tuple[int, int]:
        """确保宽高为偶数（H.264/H.265 编码要求）。"""
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        return max(2, width), max(2, height)

    def _encode_frames_to_file(
        self,
        frames: list[Any],
        fps: int,
        width: int,
        height: int,
        output_path: str,
        codec: str,
        bitrate: str | None,
    ) -> None:
        """通过 FFmpeg stdin pipe 将帧编码为视频文件。

        使用 rawvideo 格式通过 stdin 传入帧数据，避免中间文件。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 帧列表。
        fps:
            帧率。
        width:
            输出宽度（已确保偶数）。
        height:
            输出高度（已确保偶数）。
        output_path:
            输出文件路径。
        codec:
            视频编码器。
        bitrate:
            可选比特率，如 ``"8M"``。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        ffmpeg = self._ffmpeg_path
        if ffmpeg is None:
            raise RuntimeError("FFmpeg not loaded. Call load() first.")

        # 构建 FFmpeg 命令
        cmd: list[str] = [
            ffmpeg,
            "-y",  # 覆盖输出
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",  # 从 stdin 读取
            "-c:v", codec,
            "-preset", self._preset,
            "-crf", str(self._quality),
            "-pix_fmt", self._pixel_format,
        ]

        if bitrate:
            cmd.extend(["-b:v", bitrate])

        # 添加格式特定参数
        if self._format == "mp4" or self._format == "mov":
            cmd.extend(["-movflags", "+faststart"])
        elif self._format == "webm":
            cmd.extend(["-b:v", bitrate or "1M"])

        cmd.append(output_path)

        self._logger.debug("FFmpeg command: %s", " ".join(cmd))

        # 启动 FFmpeg 进程
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # 逐帧写入 stdin
            for i, frame in enumerate(frames):
                if isinstance(frame, Image.Image):
                    img = frame.convert("RGB")
                    if img.size != (width, height):
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    arr = np.array(img)
                elif isinstance(frame, np.ndarray):
                    arr = frame
                    # 防御性 dtype 转换：FFmpeg rawvideo 期望 uint8
                    if arr.dtype != np.uint8:
                        arr = np.clip(
                            arr * 255 if arr.max() <= 1.0 else arr,
                            0, 255,
                        ).astype(np.uint8)
                    if arr.shape[:2] != (height, width):
                        # resize needed
                        from PIL import Image as PILImage  # type: ignore

                        img = PILImage.fromarray(frame).convert("RGB")
                        img = img.resize((width, height), PILImage.Resampling.LANCZOS)
                        arr = np.array(img)
                else:
                    raise TypeError(
                        f"Frame {i} must be PIL.Image or numpy.ndarray, "
                        f"got {type(frame).__name__}."
                    )

                # 确保是 RGB 连续内存
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]

                proc.stdin.write(arr.tobytes())

                # 发送进度事件
                if (i + 1) % 10 == 0 or i == len(frames) - 1:
                    self._emit_progress(i + 1, len(frames), "encoding")

            proc.stdin.close()
            stdout, stderr = proc.communicate(timeout=300)

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(
                    f"FFmpeg encoding failed (exit code {proc.returncode}): "
                    f"{err_msg}"
                )

        except Exception:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdin:
                proc.stdin.close()

    def _merge_av_subtitle(
        self,
        video_path: str,
        audio: Any,
        subtitle: Any,
        output_path: str,
        codec: str,
        bitrate: str | None,
    ) -> None:
        """合并音频与烧录字幕到视频。

        Parameters
        ----------
        video_path:
            已编码的无音频视频文件路径。
        audio:
            ``AudioData`` 实例或音频文件路径。
        subtitle:
            ``SubtitleData`` 实例。
        output_path:
            最终输出路径。
        codec:
            视频编码器。
        bitrate:
            可选比特率。
        """
        ffmpeg = self._ffmpeg_path
        if ffmpeg is None:
            raise RuntimeError("FFmpeg not loaded.")

        cmd: list[str] = [ffmpeg, "-y", "-i", video_path]

        # 音频输入
        audio_tmp_path: str | None = None
        if audio is not None:
            audio_tmp_path = self._prepare_audio_input(audio)
            if audio_tmp_path:
                cmd.extend(["-i", audio_tmp_path])

        # 字幕输入（通过临时文件）
        subtitle_filter: str | None = None
        subtitle_tmp_path: str | None = None
        if subtitle is not None:
            subtitle_tmp_path = self._prepare_subtitle_input(subtitle)
            if subtitle_tmp_path:
                # 使用 subtitles 滤镜烧录字幕
                sub_path_escaped = subtitle_tmp_path.replace("\\", "/").replace(":", "\\:")
                subtitle_filter = f"subtitles='{sub_path_escaped}'"

        # 构建 filter
        if subtitle_filter:
            cmd.extend(["-vf", subtitle_filter])

        # 视频编码参数
        cmd.extend([
            "-c:v", codec,
            "-preset", self._preset,
            "-crf", str(self._quality),
            "-pix_fmt", self._pixel_format,
        ])

        if bitrate:
            cmd.extend(["-b:v", bitrate])

        # 音频编码参数
        if audio_tmp_path:
            ac = self._audio_codec or "aac"
            cmd.extend(["-c:a", ac, "-b:a", "192k"])
            # 使用第二个输入的音频
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        else:
            cmd.extend(["-an"])  # 无音频

        if self._format in ("mp4", "mov"):
            cmd.extend(["-movflags", "+faststart"])

        cmd.append(output_path)

        self._logger.debug("FFmpeg merge command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                err_msg = result.stderr.decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(
                    f"FFmpeg merge failed (exit code {result.returncode}): "
                    f"{err_msg}"
                )
        finally:
            # 清理临时文件
            if audio_tmp_path and os.path.exists(audio_tmp_path):
                try:
                    os.unlink(audio_tmp_path)
                except OSError:
                    pass
            if subtitle_tmp_path and os.path.exists(subtitle_tmp_path):
                try:
                    os.unlink(subtitle_tmp_path)
                except OSError:
                    pass

    def _prepare_audio_input(self, audio: Any) -> str | None:
        """将音频数据准备为 FFmpeg 可读的文件路径。

        Parameters
        ----------
        audio:
            ``AudioData`` 实例或文件路径字符串。

        Returns
        -------
        str | None
            音频文件路径。如果是 ``AudioData``，保存为临时 WAV 文件。
        """
        if isinstance(audio, str):
            return audio

        if isinstance(audio, AudioData):
            tmp_path = tempfile.mktemp(suffix=".wav")
            waveform = audio.waveform
            sr = audio.sample_rate
            try:
                import numpy as np  # type: ignore
                import soundfile as sf  # type: ignore

                # soundfile 期望 (samples, channels)
                if isinstance(waveform, np.ndarray):
                    # 防御性 dtype 转换
                    if waveform.dtype not in (np.float32, np.float64, np.int16, np.int32):
                        waveform = waveform.astype(np.float32)
                    if waveform.ndim == 2:
                        waveform = waveform.T  # (channels, samples) -> (samples, channels)
                sf.write(tmp_path, waveform, sr)
                return tmp_path
            except ImportError:
                self._logger.warning(
                    "soundfile not available, cannot export AudioData to file."
                )
                return None

        self._logger.warning(
            "Unknown audio type: %s, skipping audio.",
            type(audio).__name__,
        )
        return None

    def _prepare_subtitle_input(self, subtitle: Any) -> str | None:
        """将字幕数据准备为 FFmpeg 可读的 SRT 文件。

        Parameters
        ----------
        subtitle:
            ``SubtitleData`` 实例或 SRT/VTT 文件路径字符串。

        Returns
        -------
        str | None
            SRT 文件路径。
        """
        if isinstance(subtitle, str):
            return subtitle

        if isinstance(subtitle, SubtitleData):
            from mosaic.nodes.subtitle._base import BaseSubtitleNode

            srt_content = BaseSubtitleNode._to_srt(subtitle.segments)
            tmp_path = tempfile.mktemp(suffix=".srt")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            return tmp_path

        self._logger.warning(
            "Unknown subtitle type: %s, skipping subtitles.",
            type(subtitle).__name__,
        )
        return None

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

    def _emit_progress(self, current: int, total: int, message: str = "") -> None:
        """发出 progress 事件。"""
        self._bus.emit(
            EventType.PROGRESS,
            node_name=self.name,
            current=current,
            total=total,
            message=message,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<VideoEncoder name={self.name!r} "
            f"format={self._format!r} state={status}>"
        )
