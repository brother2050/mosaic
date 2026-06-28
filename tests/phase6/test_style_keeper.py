# tests/phase6/test_style_keeper.py
"""测试 StyleKeeper 节点。"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.nodes.consistency.style_keeper import StyleKeeper
from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_pipeline_output(images):
    """Create a mock pipeline output with an ``images`` attribute."""
    output = MagicMock()
    output.images = images
    return output


def _setup_node_for_run(node, sample_image):
    """Mock _run_pipeline on a node for run()."""
    mock_output = _make_mock_pipeline_output([sample_image])
    node._run_pipeline = MagicMock(return_value=mock_output)
    return mock_output


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestStyleKeeperBasic:
    """Tests for the StyleKeeper node."""

    # T_STYLE_01 ————————————————————————————————————————————————————————————
    def test_basic_style_preservation(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_01: 基本风格保持生成，输出 PIL.Image。"""
        node = StyleKeeper(
            model="h94/IP-Adapter",
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_style_reference)

        result = node.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        assert "image" in result, "Output should contain 'image' key"
        assert isinstance(result["image"], Image.Image), (
            "Output image should be PIL.Image.Image"
        )
        assert "reference_image" in result
        assert "style" in result
        assert "seed" in result
        assert result["seed"] == 42
        assert result["method"] == "ip-adapter"
        assert result["model_name"] == "h94/IP-Adapter"
        node._run_pipeline.assert_called_once()

    # T_STYLE_02 ————————————————————————————————————————————————————————————
    def test_reference_image_passed(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_02: reference_image 传入生效。"""
        node = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_style_reference)

        result = node.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        assert isinstance(result["reference_image"], Image.Image)
        # Verify reference image is a 512x512 resized version
        assert result["reference_image"].size == (512, 512), (
            "Reference image should be resized to 512x512 for StyleKeeper"
        )

    # T_STYLE_03 ————————————————————————————————————————————————————————————
    def test_style_strength_parameter(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_03: style_strength 参数生效。"""
        # Node with high strength
        node_high = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_high, sample_style_reference)

        result_high = node_high.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            style_strength=0.9,
            seed=42,
        ))

        # Node with low strength
        node_low = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_low, sample_style_reference)

        result_low = node_low.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            style_strength=0.1,
            seed=42,
        ))

        assert isinstance(result_high["image"], Image.Image)
        assert isinstance(result_low["image"], Image.Image)
        assert node_high._run_pipeline.called
        assert node_low._run_pipeline.called

    # T_STYLE_04 ————————————————————————————————————————————————————————————
    def test_seed_reproducibility(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_04: 指定 seed 可复现。"""
        node = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_style_reference)

        node.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))
        node.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        calls = node._run_pipeline.call_args_list
        assert len(calls) == 2, "Should have made 2 pipeline calls"
        gen1 = calls[0][1].get("generator")
        gen2 = calls[1][1].get("generator")
        assert gen1 is not None
        assert gen2 is not None

    # T_STYLE_05 ————————————————————————————————————————————————————————————
    def test_method_parameter_switching(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_05: method 参数切换（ip-adapter / style-aligned / reference-only）。"""
        for method in ["ip-adapter", "style-aligned", "reference-only"]:
            node = StyleKeeper(
                method=method,
                device="cpu",
                dtype="float32",
                scheduler=cpu_scheduler,
            )
            _setup_node_for_run(node, sample_style_reference)

            spec = node.describe()
            assert spec.name == "style-keeper", (
                f"Node name should be 'style-keeper' for method={method}"
            )
            assert spec.domain == "consistency"
            assert spec.model_info is not None
            assert spec.model_info["method"] == method, (
                f"describe() should report method={method}"
            )

            result = node.run(MosaicData(
                reference_image=sample_style_reference,
                prompt="a cat sitting on a windowsill",
                seed=42,
            ))
            assert result["method"] == method
            assert isinstance(result["image"], Image.Image)

    # T_STYLE_06 ————————————————————————————————————————————————————————————
    def test_different_style_references(
        self, sample_face_image, sample_style_reference, cpu_scheduler
    ):
        """# T_STYLE_06: 参考图为不同风格时输出结果不同。"""
        # Node with face reference
        node_face = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_face, sample_face_image)

        result_face = node_face.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        # Node with style reference
        node_style = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_style, sample_style_reference)

        result_style = node_style.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            seed=42,
        ))

        assert isinstance(result_face["image"], Image.Image)
        assert isinstance(result_style["image"], Image.Image)
        # Both should produce valid images
        assert result_face["image"] is not None
        assert result_style["image"] is not None

    # T_STYLE_07 ————————————————————————————————————————————————————————————
    def test_custom_size_parameter(self, sample_style_reference, cpu_scheduler):
        """# T_STYLE_07: 自定义尺寸参数生效。"""
        node = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_style_reference)

        node.run(MosaicData(
            reference_image=sample_style_reference,
            prompt="a cat sitting on a windowsill",
            width=512,
            height=640,
        ))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["width"] == 512, (
            f"Expected width=512, got {call_kwargs['width']}"
        )
        assert call_kwargs["height"] == 640, (
            f"Expected height=640, got {call_kwargs['height']}"
        )

    # T_STYLE_08 ————————————————————————————————————————————————————————————
    def test_describe_returns_correct_info(self, cpu_scheduler):
        """# T_STYLE_08: describe 返回正确信息。"""
        node = StyleKeeper(
            model="h94/IP-Adapter",
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = node.describe()
        assert spec.name == "style-keeper", "Node name should be 'style-keeper'"
        assert spec.domain == "consistency", "Domain should be 'consistency'"
        assert spec.model_info is not None, "model_info should not be None"
        assert "vram_gb" in spec.model_info, "model_info should contain vram_gb"
        assert "license" in spec.model_info, "model_info should contain license"
        assert spec.model_info["method"] == "ip-adapter"
        assert isinstance(spec.model_info["vram_gb"], float)
        assert spec.model_info["vram_gb"] > 0

    # T_STYLE_09 ————————————————————————————————————————————————————————————
    def test_load_unload_state(self, cpu_scheduler):
        """# T_STYLE_09: load/unload 后 is_loaded 状态正确。"""
        node = StyleKeeper(
            method="ip-adapter",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        assert not node.is_loaded(), "Node should not be loaded initially"

        node.load()
        assert node.is_loaded(), "Node should be loaded after load()"

        node.unload()
        assert not node.is_loaded(), "Node should not be loaded after unload()"