# tests/phase6/test_cross_frame_consistency.py
# 测试 CrossFrameConsistency 节点

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.core.types import MosaicData
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.nodes.consistency.cross_frame_consistency import CrossFrameConsistency

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
    """Mock 节点的 _run_pipeline 方法，返回含指定图片列表的输出。"""
    node._run_pipeline = MagicMock(return_value=make_mock_pipeline_output(images))


def _make_test_images(count=5):
    """创建指定数量的测试图片。"""
    return [Image.new("RGB", (512, 512), color=(i * 50, i * 30, 200 - i * 30))
            for i in range(count)]


# ---------------------------------------------------------------------------
# TestCrossFrameConsistencyBasic
# ---------------------------------------------------------------------------
class TestCrossFrameConsistencyBasic:
    """CrossFrameConsistency 节点基本功能测试。"""

    # T_CF_01 —————————————————————————————————————————————————————————————
    def test_basic_cross_frame_generation(self, sample_prompts, cpu_scheduler):
        """T_CF_01：基本跨帧生成，输出 images 列表。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
        ))

        assert "images" in result, "输出应包含 images 字段"
        assert isinstance(result["images"], list), "images 应为 list"
        assert len(result["images"]) == 5, "应生成 5 帧图片"
        for img in result["images"]:
            assert isinstance(img, Image.Image), "每帧应为 PIL Image"
        node.unload()

    # T_CF_02 —————————————————————————————————————————————————————————————
    def test_output_frame_count_matches_prompts(self, sample_prompts, cpu_scheduler):
        """T_CF_02：输出帧数与 prompts 列表长度一致。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
        ))

        assert len(result["images"]) == len(sample_prompts), (
            f"帧数 {len(result['images'])} 应与 prompts 数 {len(sample_prompts)} 一致"
        )
        node.unload()

    # T_CF_03 —————————————————————————————————————————————————————————————
    def test_consistency_scores_length(self, sample_prompts, cpu_scheduler):
        """T_CF_03：consistency_scores 列表长度正确（等于帧数）。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
        ))

        assert "consistency_scores" in result, "输出应包含 consistency_scores"
        assert isinstance(result["consistency_scores"], list), "consistency_scores 应为 list"
        assert len(result["consistency_scores"]) == len(result["images"]), (
            "consistency_scores 长度应与帧数一致"
        )
        node.unload()

    # T_CF_04 —————————————————————————————————————————————————————————————
    def test_average_consistency_in_range(self, sample_prompts, cpu_scheduler):
        """T_CF_04：average_consistency 在合理范围（0-1）。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
        ))

        assert "average_consistency" in result, "输出应包含 average_consistency"
        avg = result["average_consistency"]
        assert isinstance(avg, float), "average_consistency 应为 float"
        assert 0.0 <= avg <= 1.0, f"average_consistency ({avg}) 应在 0-1 范围内"
        node.unload()

    # T_CF_05 —————————————————————————————————————————————————————————————
    def test_reference_image_optional(self, sample_prompts, sample_face_image, cpu_scheduler):
        """T_CF_05：reference_image 可选参数生效（传入 sample_face_image）。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
            reference_image=sample_face_image,
        ))

        assert "reference_image" in result, "输出应包含 reference_image"
        assert result["reference_image"] is not None, "reference_image 不应为 None"
        node.unload()

    # T_CF_06 —————————————————————————————————————————————————————————————
    def test_character_description_in_output(self, sample_prompts, cpu_scheduler):
        """T_CF_06：character_description 在输出中正确返回。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        desc = "a young woman with black hair"
        result = node.run(MosaicData(
            prompts=sample_prompts,
            character_description=desc,
        ))

        assert "character_description" in result, "输出应包含 character_description"
        assert result["character_description"] == desc, (
            f"character_description 应为 '{desc}'，实际为 '{result['character_description']}'"
        )
        node.unload()

    # T_CF_07 —————————————————————————————————————————————————————————————
    def test_consistency_strength_parameter(self, sample_prompts, cpu_scheduler):
        """T_CF_07：consistency_strength 参数生效（构造不同 strength 的节点）。"""
        # 低 strength
        node_low = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node_low.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node_low, test_images)

        result_low = node_low.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
            consistency_strength=0.3,
        ))
        assert result_low["images"] is not None, "低 strength 应正常生成"
        node_low.unload()

        # 高 strength
        node_high = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node_high.load()
        _mock_run_pipeline(node_high, test_images)

        result_high = node_high.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
            consistency_strength=0.95,
        ))
        assert result_high["images"] is not None, "高 strength 应正常生成"
        node_high.unload()

    # T_CF_08 —————————————————————————————————————————————————————————————
    def test_seed_reproducibility(self, sample_prompts, cpu_scheduler):
        """T_CF_08：指定 seed 可复现（两次相同 seed 输出相同的 images 数量）。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        result1 = node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
            seed=42,
        ))
        node.unload()

        node2 = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node2.load()
        _mock_run_pipeline(node2, test_images)

        result2 = node2.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
            seed=42,
        ))
        node2.unload()

        # 相同 seed 应产生相同数量的帧
        assert len(result1["images"]) == len(result2["images"]), (
            "相同 seed 下帧数应一致"
        )
        assert result1["seed"] == result2["seed"], "seed 应相同"

    # T_CF_09 —————————————————————————————————————————————————————————————
    def test_single_frame_input(self, single_prompt, cpu_scheduler):
        """T_CF_09：单帧输入（prompts 只有 1 个元素）正常工作。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(1)
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=single_prompt,
            character_description="a young woman with black hair",
        ))

        assert len(result["images"]) == 1, "单帧输入应生成 1 张图片"
        assert len(result["consistency_scores"]) == 1, "consistency_scores 长度应为 1"
        assert result["average_consistency"] is not None, "average_consistency 不应为 None"
        node.unload()

    # T_CF_10 —————————————————————————————————————————————————————————————
    def test_multi_frame_input(self, cpu_scheduler):
        """T_CF_10：多帧输入（prompts 有 5+ 个元素）正常工作。"""
        many_prompts = [
            "a woman with red hair in a garden",
            "the woman walking on a beach",
            "the woman climbing a mountain",
            "the woman reading in a library",
            "the woman cooking in a kitchen",
            "the woman painting in a studio",
            "the woman playing piano",
        ]
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node.load()
        test_images = _make_test_images(len(many_prompts))
        _mock_run_pipeline(node, test_images)

        result = node.run(MosaicData(
            prompts=many_prompts,
            character_description="a woman with red hair",
        ))

        assert len(result["images"]) == len(many_prompts), (
            f"帧数 {len(result['images'])} 应与 prompts 数 {len(many_prompts)} 一致"
        )
        assert len(result["consistency_scores"]) == len(many_prompts), (
            "consistency_scores 长度应与 prompts 数一致"
        )
        node.unload()

    # T_CF_11 —————————————————————————————————————————————————————————————
    def test_method_parameter_switching(self, sample_prompts, cpu_scheduler):
        """T_CF_11：method 参数切换（consistory / story-diffusion / all-in-one）。"""
        for method in ["consistory", "story-diffusion", "all-in-one"]:
            node = CrossFrameConsistency(
                method=method, device="cpu", dtype="float32",
                scheduler=cpu_scheduler,
            )
            node.load()
            assert node._method == method, f"method 应为 {method}"
            node_spec = node.describe()
            assert node_spec is not None, f"{method} describe 应返回有效结果"
            assert node_spec.model_info is not None, f"{method} model_info 不应为 None"
            assert node_spec.model_info.get("method") == method, (
                f"{method} describe 中 method 应正确"
            )
            node.unload()

    # T_CF_12 —————————————————————————————————————————————————————————————
    def test_describe_returns_valid_info(self, cpu_scheduler):
        """T_CF_12：describe 返回正确信息。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )
        node_spec = node.describe()

        assert node_spec.name == "cross-frame-consistency", "name 应正确"
        assert node_spec.domain == "consistency", "domain 应正确"
        assert node_spec.version == "0.1.0", "version 应正确"
        assert "image" in node_spec.input_types, "input_types 应包含 image"
        assert "image" in node_spec.output_types, "output_types 应包含 image"
        assert node_spec.model_info is not None, "model_info 不应为 None"
        assert "method" in node_spec.model_info, "model_info 应包含 method"
        assert "supported_methods" in node_spec.model_info, (
            "model_info 应包含 supported_methods"
        )

    # T_CF_13 —————————————————————————————————————————————————————————————
    def test_load_unload_state(self, cpu_scheduler):
        """T_CF_13：load/unload 后 is_loaded 状态正确。"""
        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler,
        )

        assert not node.is_loaded(), "加载前应为未加载状态"
        node.load()
        assert node.is_loaded(), "加载后应为已加载状态"
        node.unload()
        assert not node.is_loaded(), "卸载后应为未加载状态"

    # T_CF_14 —————————————————————————————————————————————————————————————
    def test_progress_events_fired(self, sample_prompts, cpu_scheduler, fresh_bus):
        """T_CF_14：进度事件在多帧生成中被触发。"""
        # 收集事件
        events_received = []

        def handler(event):
            events_received.append(event)

        fresh_bus.on(EventType.NODE_COMPLETE, handler)

        node = CrossFrameConsistency(
            method="consistory", device="cpu", dtype="float32",
            scheduler=cpu_scheduler, bus=fresh_bus,
        )
        node.load()
        test_images = _make_test_images(5)
        _mock_run_pipeline(node, test_images)

        node.run(MosaicData(
            prompts=sample_prompts,
            character_description="a young woman with black hair",
        ))

        # 进度事件（每帧 1 个 + 最终 1 个 NODE_COMPLETE = 6 个）
        # 但 _emit_progress 也使用 NODE_COMPLETE 事件类型
        assert len(events_received) >= 5, (
            f"应至少收到 5 个进度事件，实际收到 {len(events_received)}"
        )

        # 至少有一个事件包含 progress 信息
        has_progress = False
        for event in events_received:
            summary = event.payload.get("output_summary", {})
            if isinstance(summary, dict) and "progress" in summary:
                has_progress = True
                break
        assert has_progress, "事件中应包含 progress 信息"

        node.unload()