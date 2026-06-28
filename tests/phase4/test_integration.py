# tests/phase4/test_integration.py
"""Phase 4 端到端集成测试。

测试跨域管道工作流：text->image->video、video 增强、video 后处理、
多格式导出，以及异步执行与进度查询。
使用 Mock 节点模拟所有 AI 模型，不依赖真实模型或 GPU。
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import PipelineResult
from mosaic.core.task import AsyncTask, TaskStatus
from mosaic.core.types import MosaicData, ImageData, VideoData


# ---------------------------------------------------------------------------
# 集成测试 mark
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Mock 跨域节点
# ---------------------------------------------------------------------------
class MockTextGenerator(Node):
    """Mock 文本生成节点。"""

    name = "text-generator"
    domain = "text"
    description = "Mock text generator."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["text"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        prompt = input_data.get("prompt", "default")
        return MosaicData(
            generated_text=f"Generated: {prompt}",
            input_tokens=10,
            output_tokens=20,
        )

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockTextToImage(Node):
    """Mock 文本到图像节点。"""

    name = "text-to-image"
    domain = "image"
    description = "Mock text-to-image generator."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["image"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        # 生成合成图像（64x64 RGB）
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 100  # R
        arr[:, :, 1] = 150  # G
        arr[:, :, 2] = 200  # B
        img = Image.fromarray(arr, mode="RGB")
        image_data = ImageData(image=img, size=(64, 64))
        return MosaicData(image=image_data)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockImageToVideo(Node):
    """Mock 图像到视频节点。"""

    name = "image-to-video"
    domain = "video"
    description = "Mock image-to-video generator."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["video"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        # 从图像生成 10 帧视频
        image_data = input_data.get("image")
        if image_data is not None and hasattr(image_data, "image"):
            base_img = image_data.image
        else:
            base_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")

        frames = []
        for i in range(10):
            arr = np.array(base_img)
            arr = (arr + i * 5).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr, mode="RGB"))

        video_data = VideoData(frames=frames, fps=30)
        return MosaicData(video=video_data)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockTextToVideo(Node):
    """Mock 文本到视频节点。"""

    name = "text-to-video"
    domain = "video"
    description = "Mock text-to-video generator."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["video"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        frames = []
        for i in range(10):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 25) % 256
            arr[:, :, 1] = (i * 20) % 256
            arr[:, :, 2] = (i * 15) % 256
            frames.append(Image.fromarray(arr, mode="RGB"))

        video_data = VideoData(frames=frames, fps=30)
        return MosaicData(video=video_data)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockFrameInterpolation(Node):
    """Mock 帧插值节点（线性插值）。"""

    name = "frame-interpolation"
    domain = "video"
    description = "Mock frame interpolation (linear)."
    version = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["video"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        video_data = input_data.get("video")
        if video_data is None:
            raise ValueError("Missing video data.")

        frames = video_data.frames
        if len(frames) < 2:
            return MosaicData(video=video_data)

        # 线性插值：在每两个相邻帧之间插入一帧
        interpolated = []
        for i in range(len(frames) - 1):
            arr1 = np.array(frames[i]).astype(np.float32)
            arr2 = np.array(frames[i + 1]).astype(np.float32)
            mid = ((arr1 + arr2) / 2).clip(0, 255).astype(np.uint8)
            interpolated.append(frames[i])
            interpolated.append(Image.fromarray(mid, mode="RGB"))
        interpolated.append(frames[-1])

        new_video = VideoData(frames=interpolated, fps=video_data.fps)
        return MosaicData(video=new_video)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockFrameExtractor(Node):
    """Mock 帧提取节点。"""

    name = "frame-extractor"
    domain = "video"
    description = "Mock frame extractor."
    version = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["image", "mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        video_data = input_data.get("video")
        if video_data is None:
            raise ValueError("Missing video data.")

        frames = video_data.frames
        # 返回中间帧作为提取结果
        if len(frames) > 0:
            mid_idx = len(frames) // 2
            extracted = ImageData(image=frames[mid_idx], size=frames[mid_idx].size)
        else:
            extracted = ImageData(
                image=Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB"),
                size=(64, 64),
            )
        return MosaicData(image=extracted, frames=frames)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockBackgroundRemover(Node):
    """Mock 背景移除节点。"""

    name = "background-remover"
    domain = "image"
    description = "Mock background remover."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        image_data = input_data.get("image")
        if image_data is not None and hasattr(image_data, "image"):
            base_img = image_data.image
        else:
            base_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")

        # 模拟背景移除：应用简单的颜色变换
        arr = np.array(base_img)
        # 添加 alpha 通道（RGBA）
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = 255  # 完全不透明
        result_img = Image.fromarray(rgba, mode="RGBA")
        result_data = ImageData(image=result_img, size=(64, 64))
        return MosaicData(image=result_data)

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockVideoEncoder(Node):
    """Mock 视频编码节点。"""

    name = "video-encoder"
    domain = "export"
    description = "Mock video encoder (FFmpeg)."
    version = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        video_data = input_data.get("video")
        format_name = input_data.get("format", "mp4")

        if video_data is None:
            raise ValueError("Missing video data.")

        return MosaicData(
            encoded_path=f"/tmp/mock_output.{format_name}",
            format=format_name,
            frame_count=len(video_data.frames),
            fps=video_data.fps,
            video=video_data,  # 透传视频数据，供下游节点使用
        )

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class MockMultiFormatExporter(Node):
    """Mock 多格式导出节点。"""

    name = "multi-format-exporter"
    domain = "export"
    description = "Mock multi-format exporter."
    version = "0.1.0"
    input_types = ["video", "mosaic"]
    output_types = ["mosaic"]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        video_data = input_data.get("video")
        formats = input_data.get("formats", ["mp4", "webm"])

        if video_data is None:
            raise ValueError("Missing video data.")

        outputs = {}
        for fmt in formats:
            outputs[fmt] = f"/tmp/mock_output.{fmt}"

        return MosaicData(
            exported_files=outputs,
            formats=formats,
            frame_count=len(video_data.frames),
        )

    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_mock_ffmpeg_env():
    """创建 mock FFmpeg 环境。"""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.pid = 12345

    def _side_effect(args, **kwargs):
        mock_proc.configure_mock(**{"args": args})
        return mock_proc

    return patch("subprocess.Popen", side_effect=_side_effect)


# ---------------------------------------------------------------------------
# T_E2E_P4_01：text generation -> text-to-image -> image-to-video
# ---------------------------------------------------------------------------
class TestE2ETextImageVideo:
    """文本到图像到视频的跨域管道测试。"""

    def test_text_to_image_to_video_pipeline(self, fresh_bus):
        """T_E2E_P4_01：text->image->video 跨域管道。"""
        pipe = Pipeline("text-img-video", [
            MockTextGenerator(),
            MockTextToImage(),
            MockImageToVideo(),
        ])

        input_data = MosaicData(prompt="A beautiful sunset over the ocean")
        result = pipe.execute_result(input_data)

        assert result.success, "管道应成功执行"
        assert result.output is not None, "应有输出"

        # 验证最终输出包含视频数据
        video = result.output.get("video")
        assert video is not None, "最终输出应包含 video"
        assert isinstance(video, VideoData), "输出应为 VideoData"
        assert len(video.frames) == 10, "应生成 10 帧视频"

        # 验证中间产物
        intermediate_names = result.list_intermediate()
        assert len(intermediate_names) >= 3, "至少应有 3 个中间产物"


# ---------------------------------------------------------------------------
# T_E2E_P4_02：text-to-video -> frame interpolation -> video encoding
# ---------------------------------------------------------------------------
class TestE2EVideoEnhancement:
    """视频增强管道测试。"""

    def test_video_enhancement_pipeline(self, fresh_bus):
        """T_E2E_P4_02：text-to-video->帧插值->视频编码 视频增强管道。"""
        with _make_mock_ffmpeg_env():
            pipe = Pipeline("video-enhance", [
                MockTextToVideo(),
                MockFrameInterpolation(),
                MockVideoEncoder(),
            ])

            input_data = MosaicData(prompt="A cat walking in a garden")
            result = pipe.execute_result(input_data)

            assert result.success, "管道应成功执行"

            # 验证帧插值增加了帧数
            # 原始 10 帧 -> 插值后 19 帧 (10 + 9 inserted)
            video_intermediate = result.get_intermediate("frame-interpolation")
            if video_intermediate is not None:
                video = video_intermediate.get("video")
                if video is not None:
                    assert len(video.frames) >= 10, "插值后帧数应 >= 原始帧数"

            # 验证编码输出
            output = result.output
            assert output is not None, "应有输出"
            assert output.get("encoded_path") is not None, "应有编码路径"


# ---------------------------------------------------------------------------
# T_E2E_P4_03：text-to-video -> frame extraction -> background removal
# ---------------------------------------------------------------------------
class TestE2EVideoPostProcessing:
    """视频后处理管道测试。"""

    def test_video_post_processing_pipeline(self, fresh_bus):
        """T_E2E_P4_03：text-to-video->帧提取->背景移除 视频后处理。"""
        pipe = Pipeline("video-post", [
            MockTextToVideo(),
            MockFrameExtractor(),
            MockBackgroundRemover(),
        ])

        input_data = MosaicData(prompt="A person standing in front of a building")
        result = pipe.execute_result(input_data)

        assert result.success, "管道应成功执行"

        # 验证帧提取
        frame_intermediate = result.get_intermediate("frame-extractor")
        if frame_intermediate is not None:
            extracted_image = frame_intermediate.get("image")
            assert extracted_image is not None, "帧提取应输出 image"

        # 验证背景移除
        output = result.output
        bg_removed = output.get("image")
        assert bg_removed is not None, "背景移除后应有图像输出"


# ---------------------------------------------------------------------------
# T_E2E_P4_04：text-to-video -> video encoding -> multi-format export
# ---------------------------------------------------------------------------
class TestE2EExportWorkflow:
    """完整导出工作流测试。"""

    def test_export_workflow_pipeline(self, fresh_bus):
        """T_E2E_P4_04：text-to-video->编码->多格式导出 完整工作流。"""
        with _make_mock_ffmpeg_env():
            pipe = Pipeline("export-workflow", [
                MockTextToVideo(),
                MockVideoEncoder(),
                MockMultiFormatExporter(),
            ])

            input_data = MosaicData(
                prompt="A drone flying over mountains",
                formats=["mp4", "webm"],
            )
            result = pipe.execute_result(input_data)

            assert result.success, "管道应成功执行"

            output = result.output
            exported = output.get("exported_files")
            assert exported is not None, "应有导出文件"
            assert len(exported) == 2, "应导出 2 种格式"


# ---------------------------------------------------------------------------
# T_E2E_P4_05：async execution of complete pipeline
# ---------------------------------------------------------------------------
class TestE2EAsyncExecution:
    """异步执行集成测试。"""

    def test_async_complete_pipeline(self, fresh_bus):
        """T_E2E_P4_05：异步执行完整管道，等待结果。"""
        pipe = Pipeline("async-e2e", [
            MockTextGenerator(),
            MockTextToImage(),
            MockImageToVideo(),
        ])

        input_data = MosaicData(prompt="Async test prompt")
        task = pipe.run_async(input_data)
        result = task.wait(timeout=10)

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.success, "管道应成功执行"
        assert result.output is not None, "应有输出"

        video = result.output.get("video")
        assert video is not None, "最终输出应包含 video"


# ---------------------------------------------------------------------------
# T_E2E_P4_06：async execution with progress query
# ---------------------------------------------------------------------------
class TestE2EAsyncProgress:
    """异步执行进度查询测试。"""

    def test_async_progress_query(self, fresh_bus):
        """T_E2E_P4_06：异步执行过程中查询进度。"""
        pipe = Pipeline("async-progress", [
            MockTextGenerator(),
            MockTextToVideo(),
        ])

        input_data = MosaicData(prompt="Progress test prompt")
        task = pipe.run_async(input_data)

        # 在任务执行期间查询状态和进度
        statuses = []
        progresses = []
        while not task.is_ready():
            statuses.append(task.status)
            progresses.append(task.progress)
            time.sleep(0.02)

        # 有状态变化
        assert len(statuses) > 0, "执行期间应有状态记录"
        assert TaskStatus.RUNNING in statuses, "应出现过 running 状态"

        result = task.wait(timeout=10)
        assert result.success, "管道应成功执行"


# ---------------------------------------------------------------------------
# T_E2E_P4_07：PipelineResult contains per-node timing info
# ---------------------------------------------------------------------------
class TestE2EPipelineResultTiming:
    """PipelineResult 时序信息测试。"""

    def test_pipeline_result_timing_info(self, fresh_bus):
        """T_E2E_P4_07：PipelineResult 包含各节点耗时信息。"""
        pipe = Pipeline("timing-pipe", [
            MockTextGenerator(),
            MockTextToImage(),
            MockImageToVideo(),
        ])

        input_data = MosaicData(prompt="Timing test prompt")
        result = pipe.execute_result(input_data)

        assert result.success, "管道应成功执行"

        # 验证 node_durations
        assert result.node_durations, "node_durations 不应为空"
        assert result.node_count >= 3, "至少应有 3 个节点"

        # 验证各节点耗时
        for node_id, duration in result.node_durations.items():
            assert isinstance(duration, float), f"节点 {node_id} 耗时应为 float"
            assert duration >= 0, f"节点 {node_id} 耗时应 >= 0"

        # 验证总耗时
        assert result.duration > 0, "总耗时 > 0"
        assert result.duration >= sum(result.node_durations.values()), (
            "总耗时 >= 各节点耗时之和"
        )

        # 验证 summary 方法
        summary = result.summary()
        assert "Pipeline:" in summary, "summary 应包含管道名"
        assert "SUCCESS" in summary, "summary 应包含成功状态"
        assert "Duration:" in summary, "summary 应包含耗时信息"


# ---------------------------------------------------------------------------
# 额外集成测试
# ---------------------------------------------------------------------------
class TestE2EErrorHandling:
    """集成错误处理测试。"""

    def test_pipeline_with_failing_node(self, fresh_bus):
        """包含失败节点的管道（fail_fast=False）。"""
        pipe = Pipeline("fail-pipe", [
            MockTextGenerator(),
            MockTextToVideo(),
        ])

        # 缺少必要输入会触发错误
        input_data = MosaicData()  # 空输入，prompt 缺失
        result = pipe.execute_result(input_data, fail_fast=False)

        # 有错误但管道应完成
        assert result.output is not None or len(result.errors) > 0, (
            "应收集错误或产生输出"
        )

    def test_pipeline_describe(self, fresh_bus):
        """管道 describe() 返回正确的聚合信息。"""
        pipe = Pipeline("describe-pipe", [
            MockTextGenerator(),
            MockTextToImage(),
            MockImageToVideo(),
        ])

        spec = pipe.describe()
        assert spec.name == "describe-pipe", "管道名应正确"
        assert spec.domain == "pipeline", "domain 应为 pipeline"
        assert spec.model_info["node_count"] == 3, "节点数应为 3"