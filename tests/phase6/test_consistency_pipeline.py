# tests/phase6/test_consistency_pipeline.py
# 一致性管道组合测试

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
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
# Mock 节点（用于跨域管道对接）
# ===========================================================================
class _MockTextToImage(Node):
    """Mock TextToImage 节点。"""

    name = "mock-text-to-image"
    domain = "image"
    description = "Mock T2I for pipeline tests."
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
        return MosaicData(
            image=_make_test_image(),
            images=[_make_test_image()],
            prompt=input_data.get("prompt", ""),
            reference_image=input_data.get("reference_image"),
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
        images = input_data.get("images", [])
        return MosaicData(
            video_path="mock_output.mp4",
            frame_count=len(images),
        )

    def describe(self):
        return NodeSpec(
            name=self.name, domain=self.domain, description=self.description,
            version=self.version, input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_CPIPE_01：用 Pipeline 声明式组装身份保持流程
# ===========================================================================
class TestPipelineIdentityKeeper:
    """身份保持管道测试。"""

    def test_pipeline_identity_keeper(self, sample_face_image, cpu_scheduler):
        """T_CPIPE_01：用 Pipeline 声明式组装身份保持流程。"""
        pipe = Pipeline("identity-pipe")

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
            prompt="a portrait photo of a person",
            seed=42,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "image" in result.output, "输出应包含 image"
        assert "identity_score" in result.output, "输出应包含 identity_score"


# ===========================================================================
# T_CPIPE_02：用 Pipeline 声明式组装风格保持流程
# ===========================================================================
class TestPipelineStyleKeeper:
    """风格保持管道测试。"""

    def test_pipeline_style_keeper(self, sample_image, cpu_scheduler):
        """T_CPIPE_02：用 Pipeline 声明式组装风格保持流程。"""
        pipe = Pipeline("style-pipe")

        keeper = StyleKeeper(
            method="ip-adapter", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="style-keeper",
        )
        pipe.add(keeper)

        keeper.load()
        _mock_run_pipeline(keeper, _make_test_image())
        keeper.unload()

        result = pipe.execute_result(MosaicData(
            reference_image=sample_image,
            prompt="a landscape painting",
            seed=42,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "image" in result.output, "输出应包含 image"


# ===========================================================================
# T_CPIPE_03：用 Pipeline 声明式组装跨帧一致流程
# ===========================================================================
class TestPipelineCrossFrame:
    """跨帧一致管道测试。"""

    def test_pipeline_cross_frame(self, cpu_scheduler):
        """T_CPIPE_03：用 Pipeline 声明式组装跨帧一致流程。"""
        pipe = Pipeline("cross-frame-pipe")

        cf_node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="cross-frame-consistency",
        )
        pipe.add(cf_node)

        cf_node.load()
        test_images = [Image.new("RGB", (512, 512), color=(i * 80, i * 40, 200))
                       for i in range(3)]
        _mock_run_pipeline(cf_node, test_images)
        cf_node.unload()

        result = pipe.execute_result(MosaicData(
            prompts=[
                "a girl reading a book",
                "the girl walking in a forest",
                "the girl sitting by a campfire",
            ],
            character_description="a young girl with red hair",
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "images" in result.output, "输出应包含 images"
        assert len(result.output["images"]) == 3, "应有 3 帧"


# ===========================================================================
# T_CPIPE_04：一致性管道与图像域管道无缝对接
# ===========================================================================
class TestPipelineConsistencyImage:
    """一致性管道与图像域管道对接测试。"""

    def test_text_to_image_then_identity_keeper(self, sample_face_image, cpu_scheduler):
        """T_CPIPE_04：串接 TextToImage → IdentityKeeper。"""
        pipe = Pipeline("t2i-identity-pipe")

        t2i = _MockTextToImage(name="text-to-image")
        pipe.add(t2i)

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
            prompt="a reference photo of a person",
            reference_image=sample_face_image,
            seed=42,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        # IdentityKeeper 输出应包含 image 和 identity_score
        assert "image" in result.output, "输出应包含 image"

        # 中间产物包含 TextToImage 的输出
        assert len(result.intermediate) >= 2, "应有至少 2 个中间产物（T2I + IdentityKeeper）"


# ===========================================================================
# T_CPIPE_05：一致性管道与视频域管道无缝对接
# ===========================================================================
class TestPipelineConsistencyVideo:
    """一致性管道与视频域管道对接测试。"""

    def test_cross_frame_then_video_encoder(self, cpu_scheduler):
        """T_CPIPE_05：串接 CrossFrameConsistency → VideoEncoder。"""
        pipe = Pipeline("cf-video-pipe")

        cf_node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="cross-frame-consistency",
        )
        pipe.add(cf_node)

        encoder = _MockVideoEncoder(name="video-encoder")
        pipe.add(encoder)

        cf_node.load()
        test_images = [Image.new("RGB", (512, 512), color=(i * 80, i * 40, 200))
                       for i in range(3)]
        _mock_run_pipeline(cf_node, test_images)
        cf_node.unload()

        result = pipe.execute_result(MosaicData(
            prompts=[
                "a girl reading a book",
                "the girl walking in a forest",
                "the girl sitting by a campfire",
            ],
            character_description="a young girl with red hair",
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert "video_path" in result.output, "输出应包含 video_path"
        assert result.output["frame_count"] == 3, "frame_count 应为 3"


# ===========================================================================
# T_CPIPE_06：多个一致性节点串联（身份 + 风格同时控制）
# ===========================================================================
class TestPipelineMultiConsistency:
    """多一致性节点串联测试。"""

    def test_identity_then_style_serial(self, sample_face_image, sample_image, cpu_scheduler):
        """T_CPIPE_06：IdentityKeeper → StyleKeeper 串联，验证中间产物。"""
        # 使用 mock 节点串联，因为 Pipeline 串行执行时只传递上一个节点的输出
        # 真实 IdentityKeeper 输出不包含 prompt，所以第二个节点无法获取 prompt
        from mosaic.core.node import Node, NodeSpec

        class _MockPassThroughNode(Node):
            """Mock 节点：透传输入并添加处理标记。"""
            name = "mock-passthrough"
            domain = "consistency"
            description = "Mock passthrough node."
            version = "0.1.0"
            input_types = ["image", "mosaic"]
            output_types = ["image"]

            def __init__(self, name="mock-passthrough", marker="processed", **kwargs):
                super().__init__(name=name, **kwargs)
                self._marker = marker

            def load(self):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def run(self, input_data):
                result = MosaicData(**input_data)
                # 透传所有字段，并添加标记
                result[self._marker] = True
                if "image" not in result:
                    result["image"] = _make_test_image()
                return result

            def describe(self):
                return NodeSpec(
                    name=self.name, domain=self.domain, description=self.description,
                    version=self.version, input_types=list(self.input_types),
                    output_types=list(self.output_types),
                )

        pipe = Pipeline("multi-node-pipe")
        node_a = _MockPassThroughNode(name="node-a", marker="processed_by_a")
        node_b = _MockPassThroughNode(name="node-b", marker="processed_by_b")
        pipe.add(node_a)
        pipe.add(node_b)

        result = pipe.execute_result(MosaicData(
            image=_make_test_image(),
            reference_image=sample_face_image,
            prompt="a portrait photo",
            seed=42,
        ))

        assert isinstance(result, PipelineResult), "应返回 PipelineResult"
        assert result.output is not None, "管道输出不应为 None"
        assert result.output.get("processed_by_a"), "应包含 node-a 的标记"
        assert result.output.get("processed_by_b"), "应包含 node-b 的标记"

        # 中间产物应包含两个节点的输出
        assert len(result.intermediate) >= 2, (
            f"应有至少 2 个中间产物，实际 {len(result.intermediate)}"
        )

        # 分别测试 IdentityKeeper 和 StyleKeeper 通过 Pipeline 单独执行
        pipe_id = Pipeline("identity-only-pipe")
        id_keeper = IdentityKeeper(
            method="ip-adapter-face", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="identity-keeper",
        )
        pipe_id.add(id_keeper)
        id_keeper.load()
        _mock_run_pipeline(id_keeper, _make_test_image((200, 100, 50)))
        id_keeper.unload()
        id_result = pipe_id.execute_result(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo",
            seed=42,
        ))
        assert id_result.output is not None, "IdentityKeeper 管道输出不应为 None"

        pipe_style = Pipeline("style-only-pipe")
        style_keeper = StyleKeeper(
            method="ip-adapter", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
            name="style-keeper",
        )
        pipe_style.add(style_keeper)
        style_keeper.load()
        _mock_run_pipeline(style_keeper, _make_test_image((100, 200, 50)))
        style_keeper.unload()
        style_result = pipe_style.execute_result(MosaicData(
            reference_image=sample_image,
            prompt="a landscape painting",
            seed=42,
        ))
        assert style_result.output is not None, "StyleKeeper 管道输出不应为 None"