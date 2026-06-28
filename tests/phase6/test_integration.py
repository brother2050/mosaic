# tests/phase6/test_integration.py
# 使用 @pytest.mark.integration 标记

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import PipelineResult
from mosaic.core.node import Node, NodeSpec
from mosaic.nodes.consistency.cross_frame_consistency import CrossFrameConsistency
from mosaic.nodes.consistency.identity_keeper import IdentityKeeper
from mosaic.nodes.consistency.style_keeper import StyleKeeper

# Helper function (also defined in conftest.py; imported here for explicit use)
def make_mock_pipeline_output(images):
    """创建模拟的 diffusers pipeline 输出（含 images 属性）。"""
    from unittest.mock import MagicMock
    output = MagicMock()
    output.images = images
    return output

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------
def _mock_run_pipeline(node, images):
    """Mock 节点的 _run_pipeline 方法。"""
    if isinstance(images, list):
        node._run_pipeline = MagicMock(
            return_value=make_mock_pipeline_output(images)
        )
    else:
        node._run_pipeline = MagicMock(
            return_value=make_mock_pipeline_output([images])
        )


def _make_test_image(color=(128, 64, 200)):
    """创建测试图片。"""
    return Image.new("RGB", (512, 512), color=color)


# ===========================================================================
# Mock ImageToImage 节点（用于跨域集成测试）
# ===========================================================================
class _MockImageToImage(Node):
    """Mock ImageToImage 节点，验证数据流但不实际执行图像变换。"""

    name = "mock-image-to-image"
    domain = "image"
    description = "Mock ImageToImage for integration tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-image-to-image", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0
        self._last_input = None

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        self._last_input = input_data
        # 透传输入 image，模拟 i2i 处理
        input_image = input_data.get("image")
        if input_image is None:
            input_image = _make_test_image()
        return MosaicData(
            image=input_image,
            processed_by="mock-image-to-image",
            call_count=self._run_calls,
        )

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# Mock VideoEncoder 节点（用于跨域集成测试）
# ===========================================================================
class _MockVideoEncoder(Node):
    """Mock VideoEncoder 节点，验证数据流但不实际编码视频。"""

    name = "mock-video-encoder"
    domain = "export"
    description = "Mock VideoEncoder for integration tests."
    version = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["video"]

    def __init__(self, name="mock-video-encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0
        self._last_input = None

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        self._last_input = input_data
        images = input_data.get("images", [])
        return MosaicData(
            video_path="mock_output.mp4",
            frame_count=len(images),
            encoded_by="mock-video-encoder",
        )

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# Mock TextToImage 节点
# ===========================================================================
class _MockTextToImage(Node):
    """Mock TextToImage 节点，用于生成参考图。"""

    name = "mock-text-to-image"
    domain = "image"
    description = "Mock TextToImage for integration tests."
    version = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["image"]

    def __init__(self, name="mock-text-to-image", **kwargs):
        super().__init__(name=name, **kwargs)
        self._run_calls = 0

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        self._run_calls += 1
        prompt = input_data.get("prompt", "")
        return MosaicData(
            image=_make_test_image(),
            images=[_make_test_image()],
            prompt=prompt,
            model_name="mock-t2i",
        )

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_E2E_P6_01：IdentityKeeper + ImageToImage 身份保持后风格化
# ===========================================================================
class TestE2EIdentityKeeperImageToImage:
    """IdentityKeeper + ImageToImage 端到端测试。"""

    def test_identity_keeper_then_image_to_image(self, sample_face_image, cpu_scheduler):
        """T_E2E_P6_01：IdentityKeeper 身份保持后，ImageToImage 风格化。"""
        # Step 1: IdentityKeeper 生成身份保持图
        keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())

        id_result = keeper.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person wearing a suit",
            seed=42,
        ))

        assert "image" in id_result, "IdentityKeeper 输出应包含 image"
        assert id_result["image"] is not None, "IdentityKeeper 应生成图片"
        assert "identity_score" in id_result, "输出应包含 identity_score"
        keeper.unload()

        # Step 2: ImageToImage (mock) 风格化
        i2i = _MockImageToImage()
        i2i.load()
        i2i_result = i2i.run(MosaicData(
            image=id_result["image"],
            prompt="transform to watercolor painting style",
        ))
        assert "image" in i2i_result, "ImageToImage 输出应包含 image"
        assert i2i_result["processed_by"] == "mock-image-to-image", "应经过 mock i2i 处理"
        i2i.unload()


# ===========================================================================
# T_E2E_P6_02：StyleKeeper + ImageToImage 风格保持后进一步处理
# ===========================================================================
class TestE2EStyleKeeperImageToImage:
    """StyleKeeper + ImageToImage 端到端测试。"""

    def test_style_keeper_then_image_to_image(self, sample_image, cpu_scheduler):
        """T_E2E_P6_02：StyleKeeper 风格保持后，ImageToImage 进一步处理。"""
        # Step 1: StyleKeeper 生成风格保持图
        keeper = StyleKeeper(
            method="ip-adapter", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())

        style_result = keeper.run(MosaicData(
            reference_image=sample_image,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        assert "image" in style_result, "StyleKeeper 输出应包含 image"
        assert style_result["image"] is not None, "StyleKeeper 应生成图片"
        keeper.unload()

        # Step 2: ImageToImage (mock) 进一步处理
        i2i = _MockImageToImage()
        i2i.load()
        i2i_result = i2i.run(MosaicData(
            image=style_result["image"],
            prompt="add dramatic lighting",
        ))
        assert "image" in i2i_result, "ImageToImage 输出应包含 image"
        i2i.unload()


# ===========================================================================
# T_E2E_P6_03：CrossFrameConsistency + VideoEncoder 跨帧一致后编码为视频
# ===========================================================================
class TestE2ECrossFrameVideoEncoder:
    """CrossFrameConsistency + VideoEncoder 端到端测试。"""

    def test_cross_frame_then_video_encoder(self, cpu_scheduler):
        """T_E2E_P6_03：CrossFrameConsistency 输出 → VideoEncoder 输入数据流。"""
        prompts = [
            "a girl reading a book",
            "the girl walking in a forest",
            "the girl sitting by a campfire",
        ]
        # Step 1: CrossFrameConsistency
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = [Image.new("RGB", (512, 512), color=(i * 80, i * 40, 200))
                       for i in range(3)]
        _mock_run_pipeline(node, test_images)

        cf_result = node.run(MosaicData(
            prompts=prompts,
            character_description="a young girl with red hair",
        ))

        assert "images" in cf_result, "CrossFrameConsistency 输出应包含 images"
        assert len(cf_result["images"]) == 3, "应有 3 帧"
        node.unload()

        # Step 2: VideoEncoder (mock) 接收 frames
        encoder = _MockVideoEncoder()
        encoder.load()
        enc_result = encoder.run(cf_result)
        assert "video_path" in enc_result, "VideoEncoder 输出应包含 video_path"
        assert enc_result["frame_count"] == 3, "frame_count 应为 3"
        encoder.unload()


# ===========================================================================
# T_E2E_P6_04：TextToImage + IdentityKeeper 先生成参考图，再保持身份
# ===========================================================================
class TestE2ETextToImageIdentityKeeper:
    """TextToImage + IdentityKeeper 端到端测试。"""

    def test_text_to_image_then_identity_keeper(self, sample_face_image, cpu_scheduler):
        """T_E2E_P6_04：先生成参考图，再保持身份生成新图。"""
        # Step 1: TextToImage (mock) 生成参考图
        t2i = _MockTextToImage()
        t2i.load()
        t2i_result = t2i.run(MosaicData(
            prompt="a reference photo of a person",
        ))
        assert "image" in t2i_result, "TextToImage 输出应包含 image"
        reference_image = t2i_result["image"]
        t2i.unload()

        # Step 2: IdentityKeeper 使用参考图保持身份
        keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())

        id_result = keeper.run(MosaicData(
            reference_image=reference_image,
            prompt="the same person wearing a hat",
            seed=42,
        ))

        assert "image" in id_result, "IdentityKeeper 输出应包含 image"
        assert "identity_score" in id_result, "输出应包含 identity_score"
        keeper.unload()


# ===========================================================================
# T_E2E_P6_05：一致性节点与 Pipeline 的 Branch 组合
# ===========================================================================
class TestE2EConsistencyBranch:
    """一致性节点与 Pipeline 的 Branch 组合测试。"""

    def test_identity_and_style_branch(self, sample_face_image, sample_image, cpu_scheduler):
        """T_E2E_P6_05：Pipeline 串联两个一致性节点（身份 + 风格）。"""
        # 使用 Pipeline 分别测试 IdentityKeeper 和 StyleKeeper
        # 由于 Pipeline 串行执行时只传递上一个节点的输出，
        # 这里分别测试两个节点通过 Pipeline 的执行

        # 测试 IdentityKeeper 通过 Pipeline
        pipe_id = Pipeline("identity-pipe")
        id_keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="identity-keeper",
        )
        pipe_id.add(id_keeper)
        id_keeper.load()
        _mock_run_pipeline(id_keeper, _make_test_image())
        id_keeper.unload()

        result_id = pipe_id.execute_result(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait in oil painting style",
        ))
        assert isinstance(result_id, PipelineResult), "应返回 PipelineResult"
        assert result_id.output is not None, "IdentityKeeper 管道输出不应为 None"

        # 测试 StyleKeeper 通过 Pipeline
        pipe_style = Pipeline("style-pipe")
        style_keeper = StyleKeeper(
            method="ip-adapter", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="style-keeper",
        )
        pipe_style.add(style_keeper)
        style_keeper.load()
        _mock_run_pipeline(style_keeper, _make_test_image((100, 200, 50)))
        style_keeper.unload()

        result_style = pipe_style.execute_result(MosaicData(
            reference_image=sample_image,
            prompt="a landscape painting",
        ))
        assert isinstance(result_style, PipelineResult), "应返回 PipelineResult"
        assert result_style.output is not None, "StyleKeeper 管道输出不应为 None"


# ===========================================================================
# T_E2E_P6_06：运行过程中事件被正确触发
# ===========================================================================
class TestE2EEventBus:
    """事件触发测试。"""

    def test_events_fired_during_run(self, sample_face_image, cpu_scheduler, fresh_bus):
        """T_E2E_P6_06：运行过程中事件被正确触发（验证 EventBus 收到事件）。"""
        events = []

        def event_handler(event):
            events.append(event)

        fresh_bus.on(EventType.NODE_START, event_handler)
        fresh_bus.on(EventType.NODE_COMPLETE, event_handler)

        keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler, bus=fresh_bus,
        )
        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())

        keeper.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo",
            seed=42,
        ))

        assert len(events) >= 2, (
            f"应至少收到 2 个事件（NODE_START + NODE_COMPLETE），实际收到 {len(events)}"
        )

        # 检查事件类型
        event_types = [e.event_type for e in events]
        assert EventType.NODE_START in event_types, "应包含 NODE_START 事件"
        assert EventType.NODE_COMPLETE in event_types, "应包含 NODE_COMPLETE 事件"

        keeper.unload()


# ===========================================================================
# T_E2E_P6_07：PipelineResult 包含正确信息
# ===========================================================================
class TestE2EPipelineResult:
    """PipelineResult 验证测试。"""

    def test_pipeline_result_contains_correct_info(self, sample_face_image, cpu_scheduler):
        """T_E2E_P6_07：PipelineResult 包含正确信息。"""
        pipe = Pipeline("test-pipeline-p6")

        keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="identity-keeper",
        )
        pipe.add(keeper)

        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())
        keeper.unload()

        result = pipe.execute_result(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo",
            seed=42,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.pipeline_name == "test-pipeline-p6", "pipeline_name 应正确"
        assert result.duration > 0, "duration 应大于 0"
        assert result.output is not None, "output 不应为 None"

        # 检查中间产物
        assert result.intermediate is not None, "intermediate 不应为 None"
        assert len(result.intermediate) > 0, "intermediate 应至少包含 1 个产物"


# ===========================================================================
# T_E2E_P6_08：中间产物可单独取出
# ===========================================================================
class TestE2EIntermediate:
    """中间产物验证测试。"""

    def test_intermediate_artifacts_accessible(self, sample_face_image, cpu_scheduler):
        """T_E2E_P6_08：中间产物（参考图、生成图）可单独取出。"""
        pipe = Pipeline("intermediate-test-pipe")

        keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="identity-keeper",
        )
        pipe.add(keeper)

        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())
        keeper.unload()

        result = pipe.execute_result(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo",
            seed=42,
        ))

        # 中间产物应可访问
        intermediate = result.intermediate
        assert len(intermediate) > 0, "应有中间产物"

        # 至少有一个中间产物包含 image
        found_image = False
        for key, value in intermediate.items():
            if isinstance(value, MosaicData) and "image" in value:
                found_image = True
                break
        assert found_image, "中间产物中应包含 image 类型的数据"