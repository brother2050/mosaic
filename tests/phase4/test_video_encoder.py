# tests/phase4/test_video_encoder.py
"""Phase 4 VideoEncoder 节点测试。

测试 VideoEncoder 的视频编码、音视频合并、字幕烧录、帧尺寸调整、
输出元数据等功能。全部使用 mock FFmpeg 与合成帧数据。
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.types import AudioData, MosaicData, SubtitleData
from mosaic.nodes.export.video_encoder import VideoEncoder


# ---------------------------------------------------------------------------
# Mock 辅助函数
# ---------------------------------------------------------------------------
def _make_mock_popen_side_effect(captured_commands, stdin_data, write_output=True):
    """创建 mock subprocess.Popen 的 side_effect 函数。

    Parameters
    ----------
    captured_commands : list
        用于存储捕获到的 FFmpeg 命令参数的列表。
    stdin_data : list
        用于存储写入 stdin 的帧数据的列表。
    write_output : bool
        是否写入输出文件（模拟 FFmpeg 生成输出文件）。
    """

    def _side_effect(args, **kwargs):
        captured_commands.append(list(args))
        # 写入输出文件（查找最后一个非选项参数中带扩展名的路径）
        if write_output:
            for arg in reversed(args):
                if isinstance(arg, str) and not arg.startswith("-") and "." in arg:
                    try:
                        dirname = os.path.dirname(arg)
                        if dirname:
                            os.makedirs(dirname, exist_ok=True)
                        with open(arg, "wb") as f:
                            f.write(b"mock-video-data-" * 50)
                    except (OSError, ValueError):
                        pass
                    break

        mock_stdin = MagicMock()
        mock_stdin.closed = False
        mock_stdin.write = MagicMock(side_effect=lambda data: stdin_data.append(data))

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = mock_stdin
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        return mock_proc

    return _side_effect


def _make_mock_run_side_effect(captured_commands, write_output=True):
    """创建 mock subprocess.run 的 side_effect 函数。

    Parameters
    ----------
    captured_commands : list
        用于存储捕获到的 FFmpeg 命令参数的列表。
    write_output : bool
        是否写入输出文件。
    """

    def _side_effect(args, **kwargs):
        captured_commands.append(list(args))
        if write_output:
            for arg in reversed(args):
                if isinstance(arg, str) and not arg.startswith("-") and "." in arg:
                    try:
                        dirname = os.path.dirname(arg)
                        if dirname:
                            os.makedirs(dirname, exist_ok=True)
                        with open(arg, "wb") as f:
                            f.write(b"mock-merged-video-data-" * 50)
                    except (OSError, ValueError):
                        pass
                    break

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""
        return mock_result

    return _side_effect


# ---------------------------------------------------------------------------
# 辅助 fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ffmpeg_env():
    """Mock FFmpeg 环境：_find_ffmpeg、subprocess.Popen、subprocess.run。"""
    captured_popen = []
    stdin_data = []
    captured_run = []

    with patch.object(
        VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"
    ) as mock_find:
        with patch(
            "subprocess.Popen",
            side_effect=_make_mock_popen_side_effect(captured_popen, stdin_data),
        ) as mock_popen:
            with patch(
                "subprocess.run",
                side_effect=_make_mock_run_side_effect(captured_run),
            ) as mock_run:
                yield {
                    "find_ffmpeg": mock_find,
                    "popen": mock_popen,
                    "run": mock_run,
                    "captured_popen": captured_popen,
                    "stdin_data": stdin_data,
                    "captured_run": captured_run,
                }


def _make_odd_frames():
    """创建奇数尺寸的帧列表（99x101）。"""
    return [Image.new("RGB", (99, 101), color=(100, 150, 200)) for _ in range(5)]


# ---------------------------------------------------------------------------
# TestVideoEncoder
# ---------------------------------------------------------------------------
class TestVideoEncoder:
    """VideoEncoder 编码功能测试。"""

    # T_ENC_01
    def test_basic_mp4_encoding(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_01：基本 mp4 编码，输出文件存在。"""
        encoder = VideoEncoder(format="mp4", quality=23)
        encoder.load()

        result = encoder.run(MosaicData(frames=list(sample_frames), fps=30))

        output_path = result["output_path"]
        assert output_path is not None, "output_path 不应为 None"
        assert os.path.exists(output_path), f"输出文件应存在: {output_path}"
        assert os.path.getsize(output_path) > 0, "输出文件不应为空"
        assert output_path.endswith(".mp4"), f"输出文件应以 .mp4 结尾: {output_path}"

        # 验证 stdin 有数据写入
        assert len(mock_ffmpeg_env["stdin_data"]) > 0, "应该有帧数据写入 stdin"

        encoder.unload()

    # T_ENC_02
    def test_different_formats(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_02：不同格式 (mp4, webm, avi) 各自编码成功。"""
        for fmt in ("mp4", "webm", "avi"):
            captured = []
            stdin_data = []

            with patch(
                "subprocess.Popen",
                side_effect=_make_mock_popen_side_effect(captured, stdin_data),
            ):
                encoder = VideoEncoder(format=fmt, quality=23)
                with patch.object(VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"):
                    encoder.load()

                result = encoder.run(
                    MosaicData(frames=list(sample_frames), fps=30)
                )

                output_path = result["output_path"]
                assert os.path.exists(output_path), f"[{fmt}] 输出文件应存在"
                assert output_path.endswith(f".{fmt}"), (
                    f"[{fmt}] 输出文件应以 .{fmt} 结尾: {output_path}"
                )
                assert result["format"] == fmt, f"[{fmt}] format 应为 {fmt}"

                encoder.unload()

    # T_ENC_03
    def test_quality_affects_command(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_03：quality 参数影响命令中的 -crf 值。"""
        for quality, expected_crf in [(18, "18"), (23, "23"), (32, "32")]:
            captured = []
            stdin_data = []

            with patch(
                "subprocess.Popen",
                side_effect=_make_mock_popen_side_effect(captured, stdin_data),
            ):
                encoder = VideoEncoder(format="mp4", quality=quality)
                with patch.object(VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"):
                    encoder.load()

                encoder.run(MosaicData(frames=list(sample_frames), fps=30))

                # 找出 -crf 参数及其值
                cmd = captured[0]
                crf_idx = cmd.index("-crf") if "-crf" in cmd else -1
                assert crf_idx >= 0, f"命令中应包含 -crf: {cmd}"
                assert cmd[crf_idx + 1] == expected_crf, (
                    f"quality={quality} 时 -crf 应为 {expected_crf}，"
                    f"实际为 {cmd[crf_idx + 1]}"
                )

                encoder.unload()

    # T_ENC_04
    def test_audio_video_merge(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_04：音视频合并，命令包含 -i 音频输入。"""
        audio = AudioData(
            waveform=np.zeros(22050, dtype=np.float32),
            sample_rate=22050,
        )

        encoder = VideoEncoder(format="mp4", quality=23)
        encoder.load()

        # Mock _prepare_audio_input 返回一个临时音频文件路径
        audio_tmp = tempfile.mktemp(suffix=".wav")
        with open(audio_tmp, "wb") as f:
            f.write(b"mock-audio-wav-data")

        with patch.object(
            encoder, "_prepare_audio_input", return_value=audio_tmp
        ):
            result = encoder.run(
                MosaicData(frames=list(sample_frames), fps=30, audio=audio)
            )

        output_path = result["output_path"]
        assert os.path.exists(output_path), "合并后输出文件应存在"

        # 验证 subprocess.run 被调用（merge 步骤使用 run）
        assert len(mock_ffmpeg_env["captured_run"]) >= 1, (
            "音频合并应调用 subprocess.run"
        )
        merge_cmd = mock_ffmpeg_env["captured_run"][0]
        assert "-i" in merge_cmd, f"合并命令应包含 -i: {merge_cmd}"
        # 验证有音频编码参数
        assert "-c:a" in merge_cmd or any(
            "aac" in str(arg) for arg in merge_cmd
        ), f"合并命令应包含音频编码参数: {merge_cmd}"

        # 清理临时文件
        try:
            if os.path.exists(audio_tmp):
                os.unlink(audio_tmp)
        except OSError:
            pass

        encoder.unload()

    # T_ENC_05
    def test_subtitle_burning(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_05：字幕烧录，命令包含 subtitles 滤镜。"""
        subtitle = SubtitleData(
            segments=[
                {"start": 0.0, "end": 2.0, "text": "测试字幕"},
                {"start": 2.0, "end": 4.0, "text": "第二行"},
            ],
            format="srt",
        )

        encoder = VideoEncoder(format="mp4", quality=23)
        encoder.load()

        result = encoder.run(
            MosaicData(frames=list(sample_frames), fps=30, subtitle=subtitle)
        )

        output_path = result["output_path"]
        assert os.path.exists(output_path), "字幕烧录后输出文件应存在"

        # 验证 subprocess.run 被调用
        assert len(mock_ffmpeg_env["captured_run"]) >= 1, (
            "字幕烧录应调用 subprocess.run"
        )
        merge_cmd = mock_ffmpeg_env["captured_run"][0]
        # 检查 subtitles 滤镜
        cmd_str = " ".join(str(arg) for arg in merge_cmd)
        assert "subtitles" in cmd_str.lower(), (
            f"合并命令应包含 subtitles 滤镜: {cmd_str}"
        )

        encoder.unload()

    # T_ENC_06
    def test_odd_dimensions_adjusted(self, mock_ffmpeg_env):
        """T_ENC_06：奇数尺寸帧自动调整为偶数。"""
        odd_frames = _make_odd_frames()

        encoder = VideoEncoder(format="mp4", quality=23)
        encoder.load()

        result = encoder.run(MosaicData(frames=odd_frames, fps=30))

        resolution = result["resolution"]
        width, height = resolution
        assert width % 2 == 0, f"宽度应为偶数，实际 {width}"
        assert height % 2 == 0, f"高度应为偶数，实际 {height}"
        assert width <= 99, "宽度不应超过原始值 99"
        assert height <= 101, "高度不应超过原始值 101"

        encoder.unload()

    # T_ENC_07
    def test_output_metadata_correct(self, sample_frames, mock_ffmpeg_env):
        """T_ENC_07：输出 MosaicData 包含正确的元数据。"""
        encoder = VideoEncoder(format="mp4", quality=23)
        encoder.load()

        result = encoder.run(MosaicData(frames=list(sample_frames), fps=30))

        # output_path
        assert "output_path" in result, "结果应包含 output_path"
        assert isinstance(result["output_path"], str), "output_path 应为字符串"
        assert os.path.exists(result["output_path"]), "output_path 文件应存在"

        # format
        assert result["format"] == "mp4", f"format 应为 mp4，实际 {result['format']}"

        # codec
        assert result["codec"] == "libx264", (
            f"codec 应为 libx264，实际 {result['codec']}"
        )

        # duration
        expected_duration = len(sample_frames) / 30
        assert result["duration"] == pytest.approx(expected_duration, rel=0.01), (
            f"duration 应为 {expected_duration}，实际 {result['duration']}"
        )

        # file_size
        assert result["file_size"] > 0, "file_size 应大于 0"
        assert isinstance(result["file_size"], int), "file_size 应为 int"

        # resolution
        resolution = result["resolution"]
        assert isinstance(resolution, tuple), "resolution 应为 tuple"
        assert len(resolution) == 2, "resolution 应有 2 个元素"
        w, h = resolution
        assert w % 2 == 0, "宽度应为偶数"
        assert h % 2 == 0, "高度应为偶数"

        encoder.unload()


class TestVideoEncoderEdgeCases:
    """VideoEncoder 边界情况测试。"""

    def test_missing_frames_raises(self):
        """缺少 frames 抛出 ValueError。"""
        encoder = VideoEncoder(format="mp4")
        with patch.object(VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"):
            encoder.load()

        with pytest.raises(ValueError, match="frames"):
            encoder.run(MosaicData(fps=30))

        encoder.unload()

    def test_empty_frames_raises(self):
        """空 frames 列表抛出 ValueError。"""
        encoder = VideoEncoder(format="mp4")
        with patch.object(VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"):
            encoder.load()

        with pytest.raises(ValueError, match="frames"):
            encoder.run(MosaicData(frames=[], fps=30))

        encoder.unload()

    def test_missing_fps_raises(self, sample_frames):
        """缺少 fps 抛出 ValueError。"""
        encoder = VideoEncoder(format="mp4")
        with patch.object(VideoEncoder, "_find_ffmpeg", return_value="ffmpeg"):
            encoder.load()

        with pytest.raises(ValueError, match="fps"):
            encoder.run(MosaicData(frames=list(sample_frames)))

        encoder.unload()

    def test_describe_returns_model_info_none(self):
        """describe() 返回 NodeSpec，model_info=None。"""
        encoder = VideoEncoder(format="mp4")
        spec = encoder.describe()
        assert spec.name == "video-encoder"
        assert spec.domain == "export"
        assert spec.model_info is None, "导出域节点 model_info 应为 None"