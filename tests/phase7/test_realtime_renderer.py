# tests/phase7/test_realtime_renderer.py
"""RealtimeRenderer 实时渲染节点测试。"""

# 测试策略：使用 device="cpu" dtype="float32"，mock pipeline
# 使用短输入流避免测试时间过长
# RealtimeRenderer 在模型不可用时自动回退到 lightweight renderer

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, AudioData, MotionData
from mosaic.core.node import NodeSpec


# ===========================================================================
# 辅助函数
# ===========================================================================
def _make_test_image(color=(128, 64, 200)):
    """创建测试图片。"""
    return Image.new("RGB", (512, 512), color=color)


def _make_short_audio():
    """创建短音频用于测试。"""
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return AudioData(waveform=waveform, sample_rate=sr)


def _make_motion_data():
    """创建 MotionData 用于测试。"""
    t = np.linspace(0, 0.5, 5, dtype=np.float32)
    rest = np.array([
        [0.50, 0.18], [0.48, 0.15], [0.52, 0.15], [0.45, 0.18], [0.55, 0.18],
        [0.42, 0.30], [0.58, 0.30], [0.38, 0.45], [0.62, 0.45], [0.36, 0.60],
        [0.64, 0.60], [0.45, 0.55], [0.55, 0.55], [0.44, 0.75], [0.56, 0.75],
        [0.44, 0.95], [0.56, 0.95],
    ], dtype=np.float32)
    kps = np.broadcast_to(rest, (5, 17, 2)).copy()
    return MotionData(keypoints=kps, frame_count=5, fps=10, skeleton_type="coco")


# ===========================================================================
# T_RT_01：基本实时渲染（audio 模式），输出帧列表
# ===========================================================================
class TestBasicAudioRender:
    """audio 模式基本渲染测试。"""

    def test_audio_mode_renders_frames(self, cpu_scheduler, sample_avatar_image):
        """T_RT_01：基本实时渲染（audio 模式），输出帧列表。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        short_audio = _make_short_audio()

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[short_audio],
        ))

        assert isinstance(result, MosaicData), "输出应为 MosaicData"
        assert "frames" in result, "输出应包含 'frames' 字段"
        assert isinstance(result["frames"], list), "frames 应为列表"
        assert len(result["frames"]) > 0, "frames 不应为空"

        # 每帧应为 PIL Image
        for frame in result["frames"]:
            assert isinstance(frame, Image.Image), (
                f"每帧应为 PIL Image，实际为 {type(frame).__name__}"
            )

        renderer.unload()


# ===========================================================================
# T_RT_02：text 模式可运行
# ===========================================================================
class TestTextModeRender:
    """text 模式渲染测试。"""

    def test_text_mode_with_tts_disabled_raises(self, cpu_scheduler, sample_avatar_image):
        """T_RT_02：text 模式未启用 TTS 时抛出 ValueError。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        with pytest.raises(ValueError, match="Text mode requires TTS"):
            renderer.run(MosaicData(
                source_image=sample_avatar_image,
                mode="text",
                input_stream=["hello"],
            ))

        renderer.unload()


# ===========================================================================
# T_RT_03：motion 模式可运行
# ===========================================================================
class TestMotionModeRender:
    """motion 模式渲染测试。"""

    def test_motion_mode_renders_frames(self, cpu_scheduler, sample_avatar_image):
        """T_RT_03：motion 模式可运行，输出帧列表。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        motion = _make_motion_data()

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="motion",
            input_stream=[motion],
        ))

        assert isinstance(result, MosaicData), "输出应为 MosaicData"
        assert "frames" in result, "输出应包含 'frames' 字段"
        assert len(result["frames"]) > 0, "frames 不应为空"

        renderer.unload()


# ===========================================================================
# T_RT_04：render_stats 包含正确字段
# ===========================================================================
class TestRenderStats:
    """render_stats 测试。"""

    def test_render_stats_contains_correct_fields(self, cpu_scheduler, sample_avatar_image):
        """T_RT_04：render_stats 包含正确字段。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        short_audio = _make_short_audio()

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[short_audio],
        ))

        assert "render_stats" in result, "输出应包含 'render_stats' 字段"
        stats = result["render_stats"]
        assert isinstance(stats, dict), "render_stats 应为字典"

        # 必需字段
        required_fields = ["total_frames", "average_fps", "average_latency_ms", "dropped_frames"]
        for field in required_fields:
            assert field in stats, f"render_stats 应包含 '{field}' 字段"

        assert stats["total_frames"] > 0, "total_frames 应大于 0"
        assert isinstance(stats["dropped_frames"], int), "dropped_frames 应为 int"
        assert isinstance(stats["average_fps"], (int, float)), "average_fps 应为数字"

        renderer.unload()


# ===========================================================================
# T_RT_05：target_fps 参数传递正确
# ===========================================================================
class TestTargetFps:
    """target_fps 参数测试。"""

    def test_target_fps_parameter(self, cpu_scheduler, sample_avatar_image):
        """T_RT_05：target_fps 参数传递正确。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        short_audio = _make_short_audio()
        custom_fps = 15

        renderer = RealtimeRenderer(
            target_fps=custom_fps,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[short_audio],
        ))

        stats = result["render_stats"]
        assert stats["target_fps"] == custom_fps, (
            f"target_fps 应为 {custom_fps}，实际为 {stats['target_fps']}"
        )

        renderer.unload()


# ===========================================================================
# T_RT_06：resolution 参数传递正确
# ===========================================================================
class TestResolution:
    """resolution 参数测试。"""

    def test_resolution_parameter(self, cpu_scheduler, sample_avatar_image):
        """T_RT_06：resolution 参数传递正确。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        short_audio = _make_short_audio()
        custom_res = (128, 128)

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=custom_res,
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[short_audio],
        ))

        stats = result["render_stats"]
        assert tuple(stats["resolution"]) == custom_res, (
            f"resolution 应为 {custom_res}，实际为 {stats['resolution']}"
        )

        renderer.unload()


# ===========================================================================
# T_RT_07：enable_tts=True 时加载 TTS 模型
# ===========================================================================
class TestTtsEnabled:
    """TTS 启用测试。"""

    def test_enable_tts_loads_tts_model(self, cpu_scheduler, sample_avatar_image):
        """T_RT_07：enable_tts=True 时加载 TTS 模型。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=True,
            tts_model="edge-tts",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        # 验证 describe 显示 TTS 已启用
        spec = renderer.describe()
        assert spec.model_info["enable_tts"] is True, "enable_tts 应为 True"
        assert "tts_model" in spec.model_info, "model_info 应包含 tts_model"

        renderer.unload()


# ===========================================================================
# T_RT_08：start_realtime 和 stop_realtime 生命周期正确
# ===========================================================================
class TestRealtimeLifecycle:
    """实时渲染生命周期测试。"""

    def test_start_stop_realtime(self, cpu_scheduler, sample_avatar_image):
        """T_RT_08：start_realtime 和 stop_realtime 生命周期正确。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        # 初始状态：未运行
        stats = renderer.get_stats()
        assert stats["is_running"] is False, "初始状态 is_running 应为 False"

        # 使用一个简单的 input_callback，返回有限数据后停止
        call_count = [0]

        def input_callback():
            call_count[0] += 1
            if call_count[0] > 2:
                return None  # 停止
            return ("audio", _make_short_audio())

        output_frames = []

        def output_callback(frame):
            output_frames.append(frame)

        # 启动实时渲染
        renderer.start_realtime(
            source_image=sample_avatar_image,
            input_callback=input_callback,
            output_callback=output_callback,
        )

        # 渲染完成后 is_running 应为 False
        stats = renderer.get_stats()
        assert stats["is_running"] is False, "渲染完成后 is_running 应为 False"

        renderer.unload()


# ===========================================================================
# T_RT_09：start_realtime 的 output_callback 被调用
# ===========================================================================
class TestOutputCallback:
    """output_callback 测试。"""

    def test_output_callback_called(self, cpu_scheduler, sample_avatar_image):
        """T_RT_09：start_realtime 的 output_callback 被调用。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        call_count = [0]
        output_frames = []

        def input_callback():
            call_count[0] += 1
            if call_count[0] > 1:
                return None
            return ("audio", _make_short_audio())

        def output_callback(frame):
            output_frames.append(frame)

        renderer.start_realtime(
            source_image=sample_avatar_image,
            input_callback=input_callback,
            output_callback=output_callback,
        )

        assert len(output_frames) > 0, (
            f"output_callback 应被调用至少一次，实际调用 {len(output_frames)} 次"
        )
        for frame in output_frames:
            assert isinstance(frame, Image.Image), (
                f"每帧应为 PIL Image，实际为 {type(frame).__name__}"
            )

        renderer.unload()


# ===========================================================================
# T_RT_10：stop_realtime 后渲染停止
# ===========================================================================
class TestStopRealtime:
    """stop_realtime 测试。"""

    def test_stop_realtime_flag(self, cpu_scheduler, sample_avatar_image):
        """T_RT_10：stop_realtime 后渲染停止。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        # 调用 stop_realtime 设置标志
        renderer.stop_realtime()
        assert renderer._stop_requested is True, "stop_realtime 后 _stop_requested 应为 True"

        # 实时渲染循环未运行，但 _realtime_running 应为 False
        assert renderer._realtime_running is False, "未启动时 _realtime_running 应为 False"

        renderer.unload()


# ===========================================================================
# T_RT_11：get_stats 返回当前统计
# ===========================================================================
class TestGetStats:
    """get_stats 测试。"""

    def test_get_stats_returns_current_stats(self, cpu_scheduler, sample_avatar_image):
        """T_RT_11：get_stats 返回当前统计。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        short_audio = _make_short_audio()

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        # 运行渲染
        renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[short_audio],
        ))

        stats = renderer.get_stats()
        assert isinstance(stats, dict), "get_stats 应返回字典"
        assert "total_frames" in stats, "应包含 total_frames"
        assert "average_fps" in stats, "应包含 average_fps"
        assert "average_latency_ms" in stats, "应包含 average_latency_ms"
        assert "dropped_frames" in stats, "应包含 dropped_frames"
        assert "is_running" in stats, "应包含 is_running"

        renderer.unload()


# ===========================================================================
# T_RT_12：describe 返回正确信息（标注性能指标）
# ===========================================================================
class TestDescribeRealtime:
    """describe 方法测试。"""

    def test_describe_returns_correct_info(self, cpu_scheduler):
        """T_RT_12：describe 返回正确信息（标注性能指标）。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=25,
            resolution=(512, 512),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = renderer.describe()
        assert isinstance(spec, NodeSpec), "describe 应返回 NodeSpec"
        assert spec.name == "realtime-renderer", (
            f"name 应为 'realtime-renderer'，实际为 {spec.name}"
        )
        assert spec.domain == "digital_human", (
            f"domain 应为 'digital_human'，实际为 {spec.domain}"
        )
        assert spec.version == "0.1.0", (
            f"version 应为 '0.1.0'，实际为 {spec.version}"
        )

        # 性能指标
        assert "performance" in spec.model_info, (
            "model_info 应包含 'performance' 性能指标"
        )
        perf = spec.model_info["performance"]
        assert "target_fps" in perf, "performance 应包含 target_fps"
        assert perf["target_fps"] == 25, f"target_fps 应为 25，实际为 {perf['target_fps']}"

        # 分辨率
        assert "resolution" in spec.model_info, "model_info 应包含 resolution"
        assert spec.model_info["resolution"] == [512, 512], (
            f"resolution 应为 [512, 512]，实际为 {spec.model_info['resolution']}"
        )


# ===========================================================================
# T_RT_13：load/unload 后 is_loaded 状态正确
# ===========================================================================
class TestLoadUnload:
    """load/unload 状态测试。"""

    def test_load_unload_state(self, cpu_scheduler):
        """T_RT_13：load/unload 后 is_loaded 状态正确。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        assert renderer.is_loaded() is False, "加载前 is_loaded 应为 False"

        renderer.load()
        assert renderer.is_loaded() is True, "加载后 is_loaded 应为 True"

        renderer.unload()
        assert renderer.is_loaded() is False, "卸载后 is_loaded 应为 False"


# ===========================================================================
# T_RT_14：输入为空流时优雅处理
# ===========================================================================
class TestEmptyStream:
    """空流处理测试。"""

    def test_empty_stream_handled_gracefully(self, cpu_scheduler, sample_avatar_image):
        """T_RT_14：输入为空流时优雅处理（空列表 -> 不崩溃）。"""
        from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

        renderer = RealtimeRenderer(
            target_fps=10,
            resolution=(256, 256),
            enable_tts=False,
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        renderer.load()

        # 空列表作为输入流
        result = renderer.run(MosaicData(
            source_image=sample_avatar_image,
            mode="audio",
            input_stream=[],
        ))

        assert isinstance(result, MosaicData), "输出应为 MosaicData"
        assert "frames" in result, "输出应包含 'frames' 字段"
        assert result["frames"] == [], "空输入流应产生空帧列表"

        renderer.unload()