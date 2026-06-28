# tests/phase4/test_livestream.py
"""Phase 4 Livestreamer 节点测试。

测试 Livestreamer 的推流功能：describe 元信息、stream_url 校验、
推流错误处理、协议参数 (rtmp/srt)。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.nodes.export.livestream import Livestreamer


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_mock_popen_success(captured_commands):
    """创建成功的 mock Popen side_effect（捕获命令，返回成功）。"""

    def _side_effect(args, **kwargs):
        captured_commands.append(list(args))

        mock_stdin = MagicMock()
        mock_stdin.closed = False

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = mock_stdin
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        return mock_proc

    return _side_effect


def _make_mock_popen_error(error_to_raise):
    """创建会抛出异常的 mock Popen side_effect。"""

    def _side_effect(args, **kwargs):
        raise error_to_raise

    return _side_effect


def _make_sample_frames(count=5):
    """创建合成帧列表。"""
    return [Image.new("RGB", (64, 64), color=(100, 150, 200)) for _ in range(count)]


def _find_last_f_index(cmd):
    """查找 FFmpeg 命令中最后一个 -f 参数（输出格式）的索引。"""
    indices = [i for i, arg in enumerate(cmd) if arg == "-f"]
    if not indices:
        return -1
    return indices[-1]


# ---------------------------------------------------------------------------
# TestLivestreamer
# ---------------------------------------------------------------------------
class TestLivestreamer:
    """Livestreamer 推流功能测试。"""

    # T_LIVE_01
    def test_describe_returns_model_info_none(self):
        """T_LIVE_01：describe() 返回 NodeSpec，model_info=None。"""
        streamer = Livestreamer(protocol="rtmp")
        spec = streamer.describe()

        assert spec.name == "livestreamer", f"name 应为 livestreamer，实际 {spec.name}"
        assert spec.domain == "export", f"domain 应为 export，实际 {spec.domain}"
        assert spec.model_info is None, (
            "导出域节点 model_info 应为 None (不需要 GPU)"
        )

    # T_LIVE_02
    def test_missing_stream_url_raises(self):
        """T_LIVE_02：缺少 stream_url 返回错误结果（Livestreamer 内部捕获异常）。"""
        streamer = Livestreamer(protocol="rtmp")
        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        frames = _make_sample_frames()

        # 完全不提供 stream_url —— run() 内部捕获 ValueError 并返回 error
        result = streamer.run(MosaicData(frames=frames))
        assert result["status"] == "failed", (
            f"缺少 stream_url 时 status 应为 'failed'，实际 {result['status']}"
        )
        assert "error" in result, "结果应包含 error 字段"
        assert "stream_url" in str(result["error"]).lower(), (
            f"error 信息应提及 stream_url: {result['error']}"
        )

        # 提供空字符串
        result2 = streamer.run(MosaicData(frames=frames, stream_url=""))
        assert result2["status"] == "failed"
        assert "stream_url" in str(result2["error"]).lower()

        # 提供空白字符串
        result3 = streamer.run(MosaicData(frames=frames, stream_url="   "))
        assert result3["status"] == "failed"
        assert "stream_url" in str(result3["error"]).lower()

        streamer.unload()

    # T_LIVE_03
    def test_invalid_url_friendly_error(self):
        """T_LIVE_03：无效 URL 给出友好错误信息，返回 result["error"]。"""
        streamer = Livestreamer(protocol="rtmp")

        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        frames = _make_sample_frames()

        # Mock Popen 抛出异常，模拟推流失败
        with patch(
            "subprocess.Popen",
            side_effect=_make_mock_popen_error(OSError("Connection refused")),
        ):
            result = streamer.run(
                MosaicData(
                    frames=frames,
                    stream_url="rtmp://invalid-server:1935/live/key",
                )
            )

        # 验证错误被捕获并返回
        assert result["status"] == "failed", (
            f"推流失败时 status 应为 'failed'，实际 {result['status']}"
        )
        assert "error" in result, "结果应包含 error 字段"
        assert result["error"] is not None, "error 不应为 None"
        assert len(str(result["error"])) > 0, "error 消息不应为空"

        streamer.unload()

    # T_LIVE_04
    def test_protocol_rtmp_uses_flv(self):
        """T_LIVE_04：RTMP 协议使用 -f flv 输出格式。"""
        captured = []

        streamer = Livestreamer(protocol="rtmp")

        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        frames = _make_sample_frames()

        with patch(
            "subprocess.Popen",
            side_effect=_make_mock_popen_success(captured),
        ):
            streamer.run(
                MosaicData(
                    frames=frames,
                    stream_url="rtmp://live.example.com/live/key",
                )
            )

        assert len(captured) >= 1, "应至少有一次 Popen 调用"
        cmd = captured[0]
        # 查找最后一个 -f 参数（输出格式），第一个 -f 是 rawvideo（输入格式）
        f_idx = _find_last_f_index(cmd)
        assert f_idx >= 0, f"命令中应包含 -f 参数: {cmd}"
        assert cmd[f_idx + 1] == "flv", (
            f"RTMP 协议输出格式应为 flv，实际 {cmd[f_idx + 1]}"
        )

        streamer.unload()

    def test_protocol_srt_uses_mpegts(self):
        """T_LIVE_04 (续)：SRT 协议使用 -f mpegts 输出格式。"""
        captured = []

        streamer = Livestreamer(protocol="srt")

        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        frames = _make_sample_frames()

        with patch(
            "subprocess.Popen",
            side_effect=_make_mock_popen_success(captured),
        ):
            streamer.run(
                MosaicData(
                    frames=frames,
                    stream_url="srt://server:9000?streamid=test",
                )
            )

        assert len(captured) >= 1, "应至少有一次 Popen 调用"
        cmd = captured[0]
        # 查找最后一个 -f 参数（输出格式）
        f_idx = _find_last_f_index(cmd)
        assert f_idx >= 0, f"命令中应包含 -f 参数: {cmd}"
        assert cmd[f_idx + 1] == "mpegts", (
            f"SRT 协议输出格式应为 mpegts，实际 {cmd[f_idx + 1]}"
        )

        streamer.unload()


class TestLivestreamerEdgeCases:
    """Livestreamer 边界情况测试。"""

    def test_missing_frames_returns_error(self):
        """缺少 frames 返回错误结果（内部捕获异常）。"""
        streamer = Livestreamer(protocol="rtmp")
        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        result = streamer.run(
            MosaicData(stream_url="rtmp://live.example.com/live/key")
        )
        assert result["status"] == "failed", "缺少 frames 时 status 应为 failed"
        assert "error" in result, "结果应包含 error 字段"
        assert "frames" in str(result["error"]).lower(), (
            f"error 信息应提及 frames: {result['error']}"
        )

        streamer.unload()

    def test_empty_frames_returns_error(self):
        """空 frames 返回错误结果（内部捕获异常）。"""
        streamer = Livestreamer(protocol="rtmp")
        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        result = streamer.run(
            MosaicData(
                frames=[],
                stream_url="rtmp://live.example.com/live/key",
            )
        )
        assert result["status"] == "failed", "空 frames 时 status 应为 failed"
        assert "error" in result, "结果应包含 error 字段"
        assert "frames" in str(result["error"]).lower(), (
            f"error 信息应提及 frames: {result['error']}"
        )

        streamer.unload()

    def test_invalid_protocol_falls_back_to_rtmp(self):
        """无效协议参数自动回退到 rtmp。"""
        captured = []
        streamer = Livestreamer(protocol="invalid_proto")

        with patch.object(Livestreamer, "_find_ffmpeg", return_value="ffmpeg"):
            streamer.load()

        frames = _make_sample_frames()

        with patch(
            "subprocess.Popen",
            side_effect=_make_mock_popen_success(captured),
        ):
            result = streamer.run(
                MosaicData(
                    frames=frames,
                    stream_url="rtmp://live.example.com/live/key",
                )
            )

        assert result["status"] == "completed"
        # 无效协议应回退到 rtmp，输出格式为 flv
        cmd = captured[0]
        f_idx = _find_last_f_index(cmd)
        assert f_idx >= 0, f"命令中应包含 -f 参数: {cmd}"
        assert cmd[f_idx + 1] == "flv", "无效协议应回退到 flv"

        streamer.unload()