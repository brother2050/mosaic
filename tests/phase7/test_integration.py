# tests/phase7/test_integration.py
# 全部使用 @pytest.mark.integration 标记

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
from mosaic.core.events import EventBus, EventType

pytestmark = pytest.mark.integration


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
class _MockTTS(Node):
    """Mock TTS 节点。"""
    name = "mock-tts"
    domain = "audio"
    description = "Mock TTS for integration tests."
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
    description = "Mock LipSyncer for integration tests."
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


class _MockVideoEncoder(Node):
    """Mock VideoEncoder 节点。"""
    name = "mock-video-encoder"
    domain = "export"
    description = "Mock VideoEncoder for integration tests."
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


class _MockTextGenerator(Node):
    """Mock TextGenerator 节点。"""
    name = "mock-text-generator"
    domain = "text"
    description = "Mock TextGenerator for integration tests."
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
            text="你好，欢迎使用数字人系统。",
            content="你好，欢迎使用数字人系统。",
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
    description = "Mock AvatarDriver for integration tests."
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


class _MockFrameInterpolator(Node):
    """Mock FrameInterpolator 节点。"""
    name = "mock-frame-interpolator"
    domain = "video"
    description = "Mock FrameInterpolator for integration tests."
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
        # 插帧：每两帧之间插入一帧
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
    description = "Mock MultiFormatExporter for integration tests."
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


class _MockIdentityKeeper(Node):
    """Mock IdentityKeeper 节点。"""
    name = "mock-identity-keeper"
    domain = "consistency"
    description = "Mock IdentityKeeper for integration tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-identity-keeper", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        return MosaicData(
            image=_make_test_image(),
            identity_score=0.95,
            identity_kept=True,
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_E2E_P7_01：TTS -> LipSyncer -> VideoEncoder
# ===========================================================================
class TestE2ETTSLipSyncerVideoEncoder:
    """TTS -> LipSyncer -> VideoEncoder 端到端测试。"""

    def test_tts_lip_syncer_video_encoder(self, sample_avatar_image):
        """T_E2E_P7_01：TTS -> LipSyncer -> VideoEncoder 文本到说话数字人视频。"""
        # Step 1: TTS 生成音频
        tts = _MockTTS()
        tts.load()
        tts_result = tts.run(MosaicData(text="你好，我是数字人。"))
        assert "audio" in tts_result, "TTS 输出应包含 audio"
        tts.unload()

        # Step 2: LipSyncer 口型同步
        lip = _MockLipSyncer()
        lip.load()
        lip_result = lip.run(MosaicData(
            image=sample_avatar_image,
            audio=tts_result["audio"],
        ))
        assert "frames" in lip_result, "LipSyncer 输出应包含 frames"
        lip.unload()

        # Step 3: VideoEncoder 编码
        encoder = _MockVideoEncoder()
        encoder.load()
        enc_result = encoder.run(lip_result)
        assert "video_path" in enc_result, "VideoEncoder 输出应包含 video_path"
        assert enc_result["frame_count"] == 3, "frame_count 应为 3"
        encoder.unload()


# ===========================================================================
# T_E2E_P7_02：AvatarDriver -> VideoEncoder
# ===========================================================================
class TestE2EAvatarDriverVideoEncoder:
    """AvatarDriver -> VideoEncoder 端到端测试。"""

    def test_avatar_driver_video_encoder(self, sample_avatar_image):
        """T_E2E_P7_02：AvatarDriver -> VideoEncoder 形象驱动导出视频。"""
        # Step 1: AvatarDriver 驱动形象
        driver = _MockAvatarDriver()
        driver.load()
        motion = _make_motion_data(5)
        driver_result = driver.run(MosaicData(
            image=sample_avatar_image,
            motion=motion,
        ))
        assert "frames" in driver_result, "AvatarDriver 输出应包含 frames"
        driver.unload()

        # Step 2: VideoEncoder 编码
        encoder = _MockVideoEncoder()
        encoder.load()
        enc_result = encoder.run(driver_result)
        assert "video_path" in enc_result, "VideoEncoder 输出应包含 video_path"
        encoder.unload()


# ===========================================================================
# T_E2E_P7_03：MotionGenerator(preset) -> AvatarDriver
# ===========================================================================
class TestE2EMotionGeneratorAvatarDriver:
    """MotionGenerator -> AvatarDriver 端到端测试。"""

    def test_motion_generator_preset_to_avatar_driver(self, sample_avatar_image, cpu_scheduler):
        """T_E2E_P7_03：MotionGenerator(preset) -> AvatarDriver 预设动作驱动形象。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        # Step 1: MotionGenerator 生成动作
        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()
        motion_result = gen.run(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=30,
        ))
        assert "motion" in motion_result, "MotionGenerator 输出应包含 motion"
        gen.unload()

        # Step 2: AvatarDriver 驱动形象
        driver = _MockAvatarDriver()
        driver.load()
        driver_result = driver.run(MosaicData(
            image=sample_avatar_image,
            motion=motion_result["motion"],
        ))
        assert "frames" in driver_result, "AvatarDriver 输出应包含 frames"
        driver.unload()


# ===========================================================================
# T_E2E_P7_04：TextGenerator -> TTS -> LipSyncer
# ===========================================================================
class TestE2ETextGeneratorTTSLipSyncer:
    """TextGenerator -> TTS -> LipSyncer 端到端测试。"""

    def test_text_generator_tts_lip_syncer(self, sample_avatar_image):
        """T_E2E_P7_04：TextGenerator -> TTS -> LipSyncer 完整对话数字人流程。"""
        # Step 1: TextGenerator 生成文本
        text_gen = _MockTextGenerator()
        text_gen.load()
        text_result = text_gen.run(MosaicData(prompt="打招呼"))
        assert "text" in text_result, "TextGenerator 输出应包含 text"
        text_gen.unload()

        # Step 2: TTS 生成音频
        tts = _MockTTS()
        tts.load()
        tts_result = tts.run(text_result)
        assert "audio" in tts_result, "TTS 输出应包含 audio"
        tts.unload()

        # Step 3: LipSyncer 口型同步
        lip = _MockLipSyncer()
        lip.load()
        lip_result = lip.run(MosaicData(
            image=sample_avatar_image,
            audio=tts_result["audio"],
        ))
        assert "frames" in lip_result, "LipSyncer 输出应包含 frames"
        assert lip_result["lip_synced"] is True, "lip_synced 应为 True"
        lip.unload()


# ===========================================================================
# T_E2E_P7_05：AvatarDriver -> FrameInterpolator -> VideoEncoder
# ===========================================================================
class TestE2EAvatarDriverFrameInterpolatorVideoEncoder:
    """AvatarDriver -> FrameInterpolator -> VideoEncoder 端到端测试。"""

    def test_avatar_driver_frame_interpolator_video_encoder(self, sample_avatar_image):
        """T_E2E_P7_05：AvatarDriver -> FrameInterpolator -> VideoEncoder 驱动后插帧。"""
        # Step 1: AvatarDriver 驱动形象
        driver = _MockAvatarDriver()
        driver.load()
        motion = _make_motion_data(5)
        driver_result = driver.run(MosaicData(
            image=sample_avatar_image,
            motion=motion,
        ))
        assert len(driver_result["frames"]) == 5, "AvatarDriver 应输出 5 帧"
        driver.unload()

        # Step 2: FrameInterpolator 插帧
        interpolator = _MockFrameInterpolator()
        interpolator.load()
        interp_result = interpolator.run(driver_result)
        assert interp_result["interpolated"] is True, "应标记为插帧后"
        assert len(interp_result["frames"]) > 5, "插帧后帧数应增加"
        interpolator.unload()

        # Step 3: VideoEncoder 编码
        encoder = _MockVideoEncoder()
        encoder.load()
        enc_result = encoder.run(interp_result)
        assert "video_path" in enc_result, "VideoEncoder 输出应包含 video_path"
        encoder.unload()


# ===========================================================================
# T_E2E_P7_06：LipSyncer -> MultiFormatExporter
# ===========================================================================
class TestE2ELipSyncerMultiFormatExporter:
    """LipSyncer -> MultiFormatExporter 端到端测试。"""

    def test_lip_syncer_multi_format_exporter(self, sample_avatar_image):
        """T_E2E_P7_06：LipSyncer -> MultiFormatExporter 口型同步后多格式导出。"""
        # Step 1: LipSyncer 口型同步
        lip = _MockLipSyncer()
        lip.load()
        lip_result = lip.run(MosaicData(
            image=sample_avatar_image,
            audio=_make_short_audio(),
        ))
        assert "frames" in lip_result, "LipSyncer 输出应包含 frames"
        lip.unload()

        # Step 2: MultiFormatExporter 多格式导出
        exporter = _MockMultiFormatExporter()
        exporter.load()
        export_result = exporter.run(lip_result)
        assert "output_path" in export_result, "导出输出应包含 output_path"
        assert "formats" in export_result, "导出输出应包含 formats"
        assert len(export_result["formats"]) >= 1, "至少应有一种输出格式"
        exporter.unload()


# ===========================================================================
# T_E2E_P7_07：数字人管道与一致性域组合（IdentityKeeper + AvatarDriver）
# ===========================================================================
class TestE2EIdentityKeeperAvatarDriver:
    """数字人管道与一致性域组合测试。"""

    def test_identity_keeper_avatar_driver(self, sample_avatar_image):
        """T_E2E_P7_07：数字人管道与一致性域组合（IdentityKeeper + AvatarDriver）。"""
        # Step 1: IdentityKeeper 保持身份
        id_keeper = _MockIdentityKeeper()
        id_keeper.load()
        id_result = id_keeper.run(MosaicData(
            image=sample_avatar_image,
            reference_image=sample_avatar_image,
        ))
        assert "identity_kept" in id_result, "IdentityKeeper 输出应包含 identity_kept"
        id_keeper.unload()

        # Step 2: AvatarDriver 驱动形象
        driver = _MockAvatarDriver()
        driver.load()
        motion = _make_motion_data(5)
        driver_result = driver.run(MosaicData(
            image=id_result.get("image", sample_avatar_image),
            motion=motion,
        ))
        assert "frames" in driver_result, "AvatarDriver 输出应包含 frames"
        driver.unload()


# ===========================================================================
# T_E2E_P7_08：运行过程中事件被正确触发
# ===========================================================================
class TestE2EEventBus:
    """事件触发测试。"""

    def test_events_fired_during_run(self, cpu_scheduler, fresh_bus):
        """T_E2E_P7_08：运行过程中事件被正确触发。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        events = []

        def event_handler(event):
            events.append(event)

        fresh_bus.on(EventType.NODE_START, event_handler)
        fresh_bus.on(EventType.NODE_COMPLETE, event_handler)

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
            bus=fresh_bus,
        )
        gen.load()

        gen.run(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=30,
        ))

        assert len(events) >= 2, (
            f"应至少收到 2 个事件（NODE_START + NODE_COMPLETE），实际收到 {len(events)}"
        )

        event_types = [e.event_type for e in events]
        assert EventType.NODE_START in event_types, "应包含 NODE_START 事件"
        assert EventType.NODE_COMPLETE in event_types, "应包含 NODE_COMPLETE 事件"

        gen.unload()


# ===========================================================================
# T_E2E_P7_09：PipelineResult 包含正确信息
# ===========================================================================
class TestE2EPipelineResult:
    """PipelineResult 验证测试。"""

    def test_pipeline_result_contains_correct_info(self, cpu_scheduler):
        """T_E2E_P7_09：PipelineResult 包含正确信息。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        pipe = Pipeline("test-pipeline-p7")

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

        result = pipe.execute_result(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=30,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.pipeline_name == "test-pipeline-p7", "pipeline_name 应正确"
        assert result.duration > 0, "duration 应大于 0"
        assert result.output is not None, "output 不应为 None"
        assert len(result.intermediate) > 0, "intermediate 应至少包含 1 个产物"


# ===========================================================================
# T_E2E_P7_10：中间产物（驱动帧、口型帧）可单独取出
# ===========================================================================
class TestE2EIntermediate:
    """中间产物验证测试。"""

    def test_intermediate_artifacts_accessible(self, cpu_scheduler):
        """T_E2E_P7_10：中间产物（驱动帧、口型帧）可单独取出。"""
        pipe = Pipeline("intermediate-test-pipe")

        # 使用 mock 节点串联
        lip = _MockLipSyncer(name="lip-syncer")
        pipe.add(lip)

        encoder = _MockVideoEncoder(name="video-encoder")
        pipe.add(encoder)

        result = pipe.execute_result(MosaicData(
            image=_make_test_image(),
            audio=_make_short_audio(),
        ))

        intermediate = result.intermediate
        assert len(intermediate) > 0, "应有中间产物"

        # 至少有一个中间产物包含 frames
        found_frames = False
        for key, value in intermediate.items():
            if isinstance(value, MosaicData) and "frames" in value:
                found_frames = True
                break
        assert found_frames, "中间产物中应包含 frames 类型的数据"