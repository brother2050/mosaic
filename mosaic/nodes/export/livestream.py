# mosaic/nodes/export/livestream.py
"""Livestreamer 节点 —— 直播推流。

将视频内容通过 RTMP/SRT 协议推流到直播平台。

设计要点
--------
* 纯工程节点，不涉及 AI 模型推理，不需要 GPU。
* 使用 FFmpeg subprocess 推流，通过 ``stdin pipe`` 传入帧数据。
* 支持 RTMP（常用直播平台）和 SRT（低延迟）两种协议。
* 推流是持续过程，支持通过 ``unload`` 或外部信号中断。
* 推流失败时给出明确错误（网络不通、地址无效、编码器不支持等）。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。

推流地址获取说明
----------------
获取推流地址的流程因平台而异：

**OBS / 通用直播平台：**
1. 在直播平台（如 B 站、YouTube Live、Twitch）创建直播间。
2. 在直播设置中找到"服务器地址"和"推流密钥"。
3. 拼接为 ``rtmp://服务器地址/推流密钥``。

**B 站直播：**
1. 直播中心 → 开播设置 → 获取推流地址。
2. 格式：``rtmp://live-push.bilivideo.com/live-bvc/?streamname=xxx``。

**YouTube Live：**
1. YouTube Studio → 直播 → 开始。
2. 在"直播设置"中找到"直播 URL"和"直播密钥"。
3. 拼接为 ``rtmp://a.rtmp.youtube.com/live2/密钥``。

**SRT 低延迟推流：**
1. 需要 SRT 服务器支持（如 SRT Live Server）。
2. 格式：``srt://服务器地址:端口?streamid=xxx``。
3. SRT 适合需要 < 1s 延迟的场景。
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
from mosaic.core.types import AudioData, MosaicData

__all__ = ["Livestreamer"]


# 支持的推流协议
_VALID_PROTOCOLS = {"rtmp", "srt"}

# 协议与 FFmpeg 输出格式映射
_PROTOCOL_FORMAT_MAP: dict[str, str] = {
    "rtmp": "flv",
    "srt": "mpegts",
}


@registry.register
class Livestreamer(Node):
    """直播推流节点。

    将帧列表通过 RTMP/SRT 协议推流到直播平台。

    Parameters
    ----------
    protocol:
        推流协议，``"rtmp"`` 或 ``"srt"``，默认 ``"rtmp"``。
    codec:
        视频编码器，默认 ``"libx264"``。
    bitrate:
        推流比特率，默认 ``"4M"``。
    fps:
        帧率，默认 ``24``。
    resolution:
        分辨率 ``(width, height)``，默认 ``(1920, 1080)``。
    bus:
        事件总线实例，``None`` 使用全局单例。

    Examples
    --------
    >>> streamer = Livestreamer(
    ...     protocol="rtmp",
    ...     bitrate="6M",
    ...     fps=30,
    ...     resolution=(1280, 720),
    ... )
    >>> result = streamer(MosaicData(
    ...     frames=frames,
    ...     stream_url="rtmp://live.example.com/stream/key",
    ... ))
    >>> result["status"]  # "completed" or "failed"

    SRT 低延迟推流：
    >>> streamer = Livestreamer(protocol="srt", bitrate="4M")
    >>> result = streamer(MosaicData(
    ...     frames=frames,
    ...     stream_url="srt://server:port?streamid=xxx",
    ... ))
    """

    name: str = "livestreamer"
    domain: str = "export"
    description: str = (
        "Stream video frames to live platforms via RTMP/SRT protocol. "
        "Supports audio merging and real-time encoding."
    )
    version: str = "0.1.0"
    input_types: tuple[str, ...] = ("video", "image", "mosaic")
    output_types: tuple[str, ...] = ("file",)

    def __init__(
        self,
        protocol: str = "rtmp",
        codec: str = "libx264",
        bitrate: str = "4M",
        fps: int = 24,
        resolution: tuple[int, int] = (1920, 1080),
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._protocol: str = protocol.lower() if protocol.lower() in _VALID_PROTOCOLS else "rtmp"
        self._codec: str = codec
        self._bitrate: str = bitrate
        self._fps: int = max(1, fps)
        self._resolution: tuple[int, int] = self._ensure_even(*resolution)
        self._logger = logging.getLogger(f"mosaic.nodes.export.{self.name}")
        self._ffmpeg_path: str | None = None
        self._stream_process: subprocess.Popen | None = None

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
                "FFmpeg is required for live streaming. "
                "Install via `pip install imageio-ffmpeg` or install "
                "FFmpeg system-wide and ensure it is in PATH."
            )
        self._logger.info("FFmpeg found at: %s", self._ffmpeg_path)
        self._loaded = True

    def unload(self) -> None:
        """释放资源：关闭推流进程。

        如果有正在运行的 FFmpeg 推流进程，会发送终止信号并等待退出。
        """
        self._stop_stream_process()
        self._ffmpeg_path = None
        self._loaded = False
        self._logger.info("Livestreamer unloaded.")

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行直播推流。

        Parameters
        ----------
        input_data:
            必须包含 ``frames`` (list[PIL.Image]) 和 ``stream_url`` (str)。
            可选：``fps`` (int，覆盖构造函数)、``audio`` (AudioData)。

        Returns
        -------
        MosaicData
            包含 ``status``/``stream_url``/``frames_sent``/``duration``。
            失败时包含 ``error``。

        Raises
        ------
        ValueError
            缺少 ``frames`` 或 ``stream_url``。
        """
        self._emit_start()
        t0 = time.perf_counter()

        try:
            # 校验输入
            frames = input_data.get("frames")
            if not isinstance(frames, list) or len(frames) == 0:
                raise ValueError(
                    f"Livestreamer requires 'frames' (non-empty list), "
                    f"got {type(frames).__name__}."
                )

            stream_url = input_data.get("stream_url")
            if not isinstance(stream_url, str) or not stream_url.strip():
                raise ValueError(
                    f"Livestreamer requires 'stream_url' (non-empty str), "
                    f"got {stream_url!r}."
                )

            # 覆盖帧率
            fps = input_data.get("fps", self._fps)
            if not isinstance(fps, (int, float)) or fps <= 0:
                fps = self._fps
            fps = int(fps)

            audio = input_data.get("audio")

            self._logger.info(
                "Starting livestream to %s (%d frames, %dx%d, %d fps, %s)...",
                stream_url,
                len(frames),
                self._resolution[0],
                self._resolution[1],
                fps,
                self._protocol,
            )

            # 执行推流
            frames_sent = self._stream_frames(
                frames, fps, stream_url, audio
            )

            elapsed = time.perf_counter() - t0
            status = "completed" if frames_sent == len(frames) else "failed"

            result = MosaicData(
                status=status,
                stream_url=stream_url,
                frames_sent=frames_sent,
                duration=elapsed,
            )

            if status == "failed":
                result["error"] = (
                    f"Only {frames_sent}/{len(frames)} frames were sent."
                )

            self._emit_complete(elapsed, {
                "status": status,
                "frames_sent": frames_sent,
                "stream_url": stream_url,
            })
            return result

        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)

            elapsed = time.perf_counter() - t0
            error_msg = str(exc)
            self._logger.error("Livestream failed: %s", error_msg)

            result = MosaicData(
                status="failed",
                stream_url=input_data.get("stream_url", ""),
                frames_sent=0,
                duration=elapsed,
                error=error_msg,
            )
            return result

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
    # FFmpeg 推流核心
    # ------------------------------------------------------------------
    def _stream_frames(
        self,
        frames: list[Any],
        fps: int,
        stream_url: str,
        audio: Any,
    ) -> int:
        """通过 FFmpeg stdin pipe 推流帧数据。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 帧列表。
        fps:
            帧率。
        stream_url:
            推流地址。
        audio:
            可选音频数据。

        Returns
        -------
        int
            成功发送的帧数。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        ffmpeg = self._ffmpeg_path
        if ffmpeg is None:
            raise RuntimeError("FFmpeg not loaded. Call load() first.")

        width, height = self._resolution

        # 准备音频临时文件
        audio_tmp_path: str | None = None
        if audio is not None:
            audio_tmp_path = self._prepare_audio_input(audio)

        # 构建 FFmpeg 推流命令
        cmd = self._build_stream_command(
            stream_url, fps, width, height,
            audio_tmp_path is not None, audio_tmp_path
        )

        self._logger.debug("FFmpeg stream command: %s", " ".join(cmd))

        # 启动 FFmpeg 进程
        self._stream_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frames_sent = 0
        try:
            for i, frame in enumerate(frames):
                # 检查进程是否还活着
                if self._stream_process.poll() is not None:
                    stderr = self._stream_process.stderr
                    if stderr:
                        err = stderr.read().decode("utf-8", errors="replace")[-500:]
                        self._logger.error(
                            "FFmpeg process exited early: %s", err
                        )
                    break

                # 转换帧为 raw RGB
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
                        img = Image.fromarray(frame).convert("RGB")
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        arr = np.array(img)
                else:
                    self._logger.warning(
                        "Skipping frame %d: unsupported type %s",
                        i,
                        type(frame).__name__,
                    )
                    continue

                # 确保 RGB 3通道
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]

                try:
                    self._stream_process.stdin.write(arr.tobytes())
                    frames_sent += 1
                except BrokenPipeError:
                    self._logger.error(
                        "Pipe broken at frame %d. Stream may have been rejected.",
                        i,
                    )
                    break

                # 发送进度事件
                if (i + 1) % 30 == 0 or i == len(frames) - 1:
                    self._emit_progress(i + 1, len(frames), "streaming")

            # 关闭 stdin，等待 FFmpeg 完成
            if self._stream_process.stdin:
                self._stream_process.stdin.close()

            stdout, stderr = self._stream_process.communicate(timeout=30)

            if self._stream_process.returncode not in (0, None):
                err_msg = stderr.decode("utf-8", errors="replace")[-500:]
                self._logger.warning(
                    "FFmpeg exited with code %d: %s",
                    self._stream_process.returncode,
                    err_msg,
                )

        except Exception as exc:  # noqa: BLE001
            self._logger.error("Streaming error: %s", exc)
            self._stop_stream_process()
        finally:
            # 清理音频临时文件
            if audio_tmp_path and os.path.exists(audio_tmp_path):
                try:
                    os.unlink(audio_tmp_path)
                except OSError:
                    pass
            self._stream_process = None

        return frames_sent

    def _build_stream_command(
        self,
        stream_url: str,
        fps: int,
        width: int,
        height: int,
        has_audio: bool,
        audio_path: str | None = None,
    ) -> list[str]:
        """构建 FFmpeg 推流命令。

        Parameters
        ----------
        stream_url:
            推流地址。
        fps:
            帧率。
        width:
            输出宽度。
        height:
            输出高度。
        has_audio:
            是否有音频输入。
        audio_path:
            音频文件路径（当 ``has_audio=True`` 时提供）。

        Returns
        -------
        list[str]
            FFmpeg 命令参数列表。
        """
        output_format = _PROTOCOL_FORMAT_MAP.get(self._protocol, "flv")

        cmd: list[str] = [
            self._ffmpeg_path,
            "-y",
            # 视频输入：从 stdin 读取 rawvideo
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]

        # 音频输入
        if has_audio and audio_path:
            cmd.extend(["-i", audio_path])

        # 视频编码
        cmd.extend([
            "-c:v", self._codec,
            "-preset", "ultrafast",  # 推流用最快预设
            "-tune", "zerolatency",  # 零延迟调优
            "-b:v", self._bitrate,
            "-pix_fmt", "yuv420p",
            "-g", str(fps * 2),  # GOP 大小 = 2秒
            "-keyint_min", str(fps),
        ])

        # 音频处理
        if has_audio:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-ac", "2",
            ])
        else:
            cmd.extend(["-an"])

        # 输出格式与地址
        cmd.extend([
            "-f", output_format,
            stream_url,
        ])

        return cmd

    def _stop_stream_process(self) -> None:
        """停止正在运行的推流进程。"""
        proc = self._stream_process
        if proc is None:
            return

        try:
            if proc.poll() is None:
                # 优雅终止：发送 'q' 到 stdin
                if proc.stdin:
                    try:
                        proc.stdin.write(b"q")
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass

                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Error stopping stream process: %s", exc)
        finally:
            self._stream_process = None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _find_ffmpeg() -> str | None:
        """查找 FFmpeg 可执行文件路径。"""
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        import shutil

        return shutil.which("ffmpeg")

    @staticmethod
    def _ensure_even(width: int, height: int) -> tuple[int, int]:
        """确保宽高为偶数。"""
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        return max(2, width), max(2, height)

    def _prepare_audio_input(self, audio: Any) -> str | None:
        """将音频数据准备为 FFmpeg 可读的文件路径。

        Parameters
        ----------
        audio:
            ``AudioData`` 实例或文件路径字符串。

        Returns
        -------
        str | None
            音频文件路径。
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

                if isinstance(waveform, np.ndarray):
                    if waveform.ndim == 2:
                        waveform = waveform.T
                sf.write(tmp_path, waveform, sr)
                return tmp_path
            except ImportError:
                self._logger.warning(
                    "soundfile not available, cannot export AudioData."
                )
                return None

        self._logger.warning(
            "Unknown audio type: %s, skipping audio.",
            type(audio).__name__,
        )
        return None

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<Livestreamer name={self.name!r} "
            f"protocol={self._protocol!r} state={status}>"
        )
