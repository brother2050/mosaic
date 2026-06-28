# tests/phase7/test_digital_human_pipeline.py
# 数字人管道组合测试

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, AudioData, MotionData
from mosaic.core.node import Node, NodeSpec
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import PipelineResult


# ===========================================================================
# 辅助函数
# ===========================================================================
def _make_test_image(color=(128, 64, 200)):
    """创建测试图片。"""
    return Image.new("RGB", (512, 512), color=color)


def _make_short_audio():
    """创建短音频。"""
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return AudioData(waveform=waveform, sample_rate=sr)


def _make_motion_data(frames=30):
    """创建 MotionData。"""
    t = np.linspace(0, 1.0, frames, dtype=np.float32)
    rest = np.array([
        [0.50, 0.18], [0.48, 0.15], [0.52, 0.15], [0.45, 0.18], [0.55, 0.18],
        [0.42, 0.30], [0.58, 0.30], [0.38, 0.45], [0.62, 0.45], [0.36, 0.60],
        [0.64, 0.60], [0.45, 0.55], [0.55, 0.55], [0.44, 0.75], [0.56, 0.75],
        [0.44, 0.95], [0.56, 0.95],
    ], dtype=np.float32)
    kps = np.broadcast_to(rest, (frames, 17, 2)).copy()
    return MotionData(keypoints=kps, frame_count=frames, fps=30, skeleton_type="coco")


# ===========================================================================
# Mock 节点
# ===========================================================================
class _MockTextGenerator(Node):
    """Mock TextGenerator 节点。"""
    name = "mock-text-generator"
    domain = "text"
    description = "Mock TextGenerator for pipeline tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def __init__(self, name="mock-text-generator", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        return MosaicData(
            text="你好，我是数字人助手。",
            content="你好，我是数字人助手。",
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockTTS(Node):
    """Mock TTS 节点。"""
    name = "mock-tts"
    domain = "audio"
    description = "Mock TTS for pipeline tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["audio"]

    def __init__(self, name="mock-tts", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        return MosaicData(
            audio=_make_short_audio(),
            text=input_data.get("text", ""),
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockLipSyncer(Node):
    """Mock LipSyncer 节点。"""
    name = "mock-lip-syncer"
    domain = "digital_human"
    description = "Mock LipSyncer for pipeline tests."
    version = "0.1.0"
    input_types = ["image", "audio", "mosaic"]
    output_types = ["video", "image"]

    def __init__(self, name="mock-lip-syncer", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        frames = [_make_test_image((200, i * 40, 150)) for i in range(3)]
        return MosaicData(
            frames=frames,
            images=frames,
            frame_count=len(frames),
            lip_synced=True,
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockAvatarDriver(Node):
    """Mock AvatarDriver 节点。"""
    name = "mock-avatar-driver"
    domain = "digital_human"
    description = "Mock AvatarDriver for pipeline tests."
    version = "0.1.0"
    input_types = ["image", "motion", "mosaic"]
    output_types = ["video", "image"]

    def __init__(self, name="mock-avatar-driver", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        frames = [_make_test_image((100, i * 50, 200)) for i in range(5)]
        return MosaicData(
            frames=frames,
            images=frames,
            frame_count=len(frames),
            driven_by="mock-avatar-driver",
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockVideoEncoder(Node):
    """Mock VideoEncoder 节点。"""
    name = "mock-video-encoder"
    domain = "export"
    description = "Mock VideoEncoder for pipeline tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["video"]

    def __init__(self, name="mock-video-encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        images = input_data.get("images", input_data.get("frames", []))
        return MosaicData(
            video_path="mock_output.mp4",
            frame_count=len(images),
            encoded_by="mock-video-encoder",
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockFrameInterpolator(Node):
    """Mock FrameInterpolator 节点。"""
    name = "mock-frame-interpolator"
    domain = "video"
    description = "Mock FrameInterpolator for pipeline tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-frame-interpolator", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        images = input_data.get("images", input_data.get("frames", []))
        interpolated = []
        for img in images:
            interpolated.append(img)
            interpolated.append(_make_test_image((255, 255, 255)))
        return MosaicData(
            images=interpolated,
            frames=interpolated,
            frame_count=len(interpolated),
            interpolated=True,
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _MockMultiFormatExporter(Node):
    """Mock MultiFormatExporter 节点。"""
    name = "mock-multi-format-exporter"
    domain = "export"
    description = "Mock MultiFormatExporter for pipeline tests."
    version = "0.1.0"
    input_types = ["image", "video", "mosaic"]
    output_types = ["video"]

    def __init__(self, name="mock-multi-format-exporter", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        return MosaicData(
            output_path="exported_output.mp4",
            formats=["mp4", "gif", "webm"],
            exported_by="mock-multi-format-exporter",
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_DHPIPE_01：用 Pipeline 声明式组装完整数字人流程
# ===========================================================================
class TestPipelineDigitalHumanFull:
    """完整数字人管道测试。"""

    def test_motion_generator_avatar_driver_video_encoder(self, cpu_scheduler, sample_avatar_image):
        """T_DHPIPE_01：MotionGenerator -> AvatarDriver -> VideoEncoder。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        pipe = Pipeline("digital-human-pipe")

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
            name="motion-generator",
        )
        pipe.add(gen)

        driver = _MockAvatarDriver(name="avatar-driver")
        pipe.add(driver)

        encoder = _MockVideoEncoder(name="video-encoder")
        pipe.add(encoder)

        gen.load()
        gen.unload()

        result = pipe.execute_result(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=30,
            image=sample_avatar_image,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "video_path" in result.output, "输出应包含 video_path"
        assert len(result.intermediate) >= 3, (
            f"应有至少 3 个中间产物，实际 {len(result.intermediate)}"
        )


# ===========================================================================
# T_DHPIPE_02：数字人管道与文本域管道组合
# ===========================================================================
class TestPipelineDigitalHumanText:
    """数字人管道与文本域管道组合测试。"""

    def test_text_generator_tts_lip_syncer(self, sample_avatar_image):
        """T_DHPIPE_02：TextGenerator -> TTS -> LipSyncer。"""
        pipe = Pipeline("text-to-lip-sync-pipe")

        text_gen = _MockTextGenerator(name="text-generator")
        pipe.add(text_gen)

        tts = _MockTTS(name="tts")
        pipe.add(tts)

        lip = _MockLipSyncer(name="lip-syncer")
        pipe.add(lip)

        result = pipe.execute_result(MosaicData(
            prompt="打招呼",
            image=sample_avatar_image,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert result.output.get("lip_synced") is True, "输出应标记 lip_synced"
        assert len(result.intermediate) >= 3, (
            f"应有至少 3 个中间产物，实际 {len(result.intermediate)}"
        )


# ===========================================================================
# T_DHPIPE_03：数字人管道与音频域管道组合
# ===========================================================================
class TestPipelineDigitalHumanAudio:
    """数字人管道与音频域管道组合测试。"""

    def test_tts_lip_syncer_avatar_driver(self, sample_avatar_image):
        """T_DHPIPE_03：TTS -> LipSyncer -> AvatarDriver。"""
        pipe = Pipeline("audio-digital-human-pipe")

        tts = _MockTTS(name="tts")
        pipe.add(tts)

        lip = _MockLipSyncer(name="lip-syncer")
        pipe.add(lip)

        driver = _MockAvatarDriver(name="avatar-driver")
        pipe.add(driver)

        result = pipe.execute_result(MosaicData(
            text="你好",
            image=sample_avatar_image,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "driven_by" in result.output, "输出应包含 driven_by"
        assert len(result.intermediate) >= 3, (
            f"应有至少 3 个中间产物，实际 {len(result.intermediate)}"
        )


# ===========================================================================
# T_DHPIPE_04：数字人管道与视频域管道组合
# ===========================================================================
class TestPipelineDigitalHumanVideo:
    """数字人管道与视频域管道组合测试。"""

    def test_avatar_driver_frame_interpolator_video_encoder(self, sample_avatar_image):
        """T_DHPIPE_04：AvatarDriver -> FrameInterpolator -> VideoEncoder。"""
        pipe = Pipeline("dh-video-pipe")

        driver = _MockAvatarDriver(name="avatar-driver")
        pipe.add(driver)

        interpolator = _MockFrameInterpolator(name="frame-interpolator")
        pipe.add(interpolator)

        encoder = _MockVideoEncoder(name="video-encoder")
        pipe.add(encoder)

        result = pipe.execute_result(MosaicData(
            image=sample_avatar_image,
            motion=_make_motion_data(5),
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "video_path" in result.output, "输出应包含 video_path"
        assert len(result.intermediate) >= 3, (
            f"应有至少 3 个中间产物，实际 {len(result.intermediate)}"
        )


# ===========================================================================
# T_DHPIPE_05：数字人管道与导出域管道组合
# ===========================================================================
class TestPipelineDigitalHumanExport:
    """数字人管道与导出域管道组合测试。"""

    def test_lip_syncer_multi_format_exporter(self, sample_avatar_image):
        """T_DHPIPE_05：LipSyncer -> MultiFormatExporter。"""
        pipe = Pipeline("dh-export-pipe")

        lip = _MockLipSyncer(name="lip-syncer")
        pipe.add(lip)

        exporter = _MockMultiFormatExporter(name="multi-format-exporter")
        pipe.add(exporter)

        result = pipe.execute_result(MosaicData(
            image=sample_avatar_image,
            audio=_make_short_audio(),
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "output_path" in result.output, "输出应包含 output_path"
        assert "formats" in result.output, "输出应包含 formats"
        assert len(result.intermediate) >= 2, (
            f"应有至少 2 个中间产物，实际 {len(result.intermediate)}"
        )


# ===========================================================================
# T_DHPIPE_06：异步执行数字人长时间渲染任务
# ===========================================================================
class TestPipelineAsyncDigitalHuman:
    """异步执行测试。"""

    def test_async_execution_digital_human(self, cpu_scheduler):
        """T_DHPIPE_06：异步执行数字人长时间渲染任务。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        pipe = Pipeline("async-digital-human-pipe")

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
            name="motion-generator",
        )
        pipe.add(gen)

        gen.load()
        gen.unload()

        # 使用 run_async 异步执行
        task = pipe.run_async(MosaicData(
            preset_name="wave",
            duration=2.0,
            fps=30,
        ))

        # 初始状态（任务可能很快完成，状态为 pending/running/completed 均可接受）
        assert task.status in ("pending", "running", "completed"), (
            f"异步任务状态应为 pending/running/completed，实际为 {task.status}"
        )

        # 等待完成
        result = task.wait(timeout=30)

        assert result is not None, "异步任务应返回结果"
        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "motion" in result.output, "输出应包含 motion"