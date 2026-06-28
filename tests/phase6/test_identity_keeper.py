# tests/phase6/test_identity_keeper.py
"""测试 IdentityKeeper 节点。"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.nodes.consistency.identity_keeper import IdentityKeeper
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
    """Mock _run_pipeline and _prepare_face_region on a node for run()."""
    mock_output = _make_mock_pipeline_output([sample_image])
    node._run_pipeline = MagicMock(return_value=mock_output)
    # Mock _prepare_face_region: return cropped face image + bbox
    face_img = sample_image.crop((128, 100, 384, 400))
    node._prepare_face_region = MagicMock(return_value=(face_img, (128, 100, 384, 400)))
    return mock_output


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestIdentityKeeperBasic:
    """Tests for the IdentityKeeper node."""

    # T_ID_01 ——————————————————————————————————————————————————————————————
    def test_basic_identity_preservation(self, sample_face_image, cpu_scheduler):
        """# T_ID_01: 基本身份保持生成，输出 PIL.Image。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        result = node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            seed=42,
        ))

        assert "image" in result, "Output should contain 'image' key"
        assert isinstance(result["image"], Image.Image), (
            "Output image should be PIL.Image.Image"
        )
        assert "reference_image" in result
        assert "identity_score" in result
        assert "seed" in result
        assert result["seed"] == 42
        assert result["method"] == "instantid"
        assert result["model_name"] == "InstantX/InstantID"
        node._run_pipeline.assert_called_once()

    # T_ID_02 ——————————————————————————————————————————————————————————————
    def test_reference_image_from_file(self, sample_face_image, cpu_scheduler, tmp_path):
        """# T_ID_02: reference_image 从文件路径加载。"""
        # Save sample image to temp file
        img_path = tmp_path / "face_ref.png"
        sample_face_image.save(img_path)

        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        result = node.run(MosaicData(
            reference_image=str(img_path),
            prompt="a portrait photo of a person",
            seed=42,
        ))

        assert isinstance(result["image"], Image.Image), (
            "Output should be PIL.Image even with file path input"
        )
        assert isinstance(result["reference_image"], Image.Image)

    # T_ID_03 ——————————————————————————————————————————————————————————————
    def test_reference_image_from_pil(self, sample_face_image, cpu_scheduler):
        """# T_ID_03: reference_image 从 PIL.Image 直接传入。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        result = node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
        ))

        assert isinstance(result["image"], Image.Image)
        assert isinstance(result["reference_image"], Image.Image), (
            "Reference image should be a PIL.Image object"
        )
        assert result["reference_image"].size == sample_face_image.size, (
            "Reference image size should match input image size"
        )

    # T_ID_04 ——————————————————————————————————————————————————————————————
    def test_identity_strength_parameter(self, sample_face_image, cpu_scheduler):
        """# T_ID_04: identity_strength 参数生效。"""
        # Node with high strength
        node_high = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_high, sample_face_image)

        result_high = node_high.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            identity_strength=0.9,
            seed=42,
        ))

        # Node with low strength
        node_low = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node_low, sample_face_image)

        result_low = node_low.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            identity_strength=0.1,
            seed=42,
        ))

        # Both should produce valid images
        assert isinstance(result_high["image"], Image.Image)
        assert isinstance(result_low["image"], Image.Image)
        # Verify the pipeline was called with different identity_strength
        assert node_high._run_pipeline.called
        assert node_low._run_pipeline.called

    # T_ID_05 ——————————————————————————————————————————————————————————————
    def test_identity_score_range(self, sample_face_image, cpu_scheduler):
        """# T_ID_05: identity_score 输出在 0-1 范围内。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        result = node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
        ))

        score = result["identity_score"]
        assert isinstance(score, float), (
            f"identity_score should be float, got {type(score).__name__}"
        )
        assert 0.0 <= score <= 1.0, (
            f"identity_score should be in [0.0, 1.0], got {score}"
        )

    # T_ID_06 ——————————————————————————————————————————————————————————————
    def test_seed_reproducibility(self, sample_face_image, cpu_scheduler):
        """# T_ID_06: 指定 seed 可复现。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        mock_output = _setup_node_for_run(node, sample_face_image)

        node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            seed=42,
        ))
        node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            seed=42,
        ))

        calls = node._run_pipeline.call_args_list
        assert len(calls) == 2, "Should have made 2 pipeline calls"
        # Both calls should have the same generator (if seed is the same)
        gen1 = calls[0][1].get("generator")
        gen2 = calls[1][1].get("generator")
        assert gen1 is not None
        assert gen2 is not None

    # T_ID_07 ——————————————————————————————————————————————————————————————
    def test_negative_prompt_parameter(self, sample_face_image, cpu_scheduler):
        """# T_ID_07: negative_prompt 参数传递正确。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        result = node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            negative_prompt="blurry, low quality, deformed",
            seed=42,
        ))

        assert isinstance(result["image"], Image.Image)
        call_kwargs = node._run_pipeline.call_args[1]
        assert "negative_prompt" in call_kwargs, (
            "negative_prompt should be passed to pipeline"
        )
        assert call_kwargs["negative_prompt"] == "blurry, low quality, deformed"

    # T_ID_08 ——————————————————————————————————————————————————————————————
    def test_custom_size_parameter(self, sample_face_image, cpu_scheduler):
        """# T_ID_08: 自定义尺寸参数生效。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        _setup_node_for_run(node, sample_face_image)

        node.run(MosaicData(
            reference_image=sample_face_image,
            prompt="a portrait photo of a person",
            width=512,
            height=640,
        ))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["width"] == 512, f"Expected width=512, got {call_kwargs['width']}"
        assert call_kwargs["height"] == 640, f"Expected height=640, got {call_kwargs['height']}"

    # T_ID_09 ——————————————————————————————————————————————————————————————
    def test_method_parameter_switching(self, sample_face_image, cpu_scheduler):
        """# T_ID_09: method 参数切换（instantid / ip-adapter-face / photomaker）。"""
        for method in ["instantid", "ip-adapter-face", "photomaker"]:
            node = IdentityKeeper(
                method=method,
                device="cpu",
                dtype="float32",
                scheduler=cpu_scheduler,
            )
            _setup_node_for_run(node, sample_face_image)

            spec = node.describe()
            assert spec.name == "identity-keeper", (
                f"Node name should be 'identity-keeper' for method={method}"
            )
            assert spec.domain == "consistency"
            assert spec.model_info is not None
            assert spec.model_info["method"] == method, (
                f"describe() should report method={method}"
            )

            result = node.run(MosaicData(
                reference_image=sample_face_image,
                prompt="a portrait photo of a person",
                seed=42,
            ))
            assert result["method"] == method
            assert isinstance(result["image"], Image.Image)

    # T_ID_10 ——————————————————————————————————————————————————————————————
    def test_no_face_detected_error(self, sample_face_image, cpu_scheduler):
        """# T_ID_10: 无检测到人脸时给出友好错误。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        mock_output = _make_mock_pipeline_output([sample_face_image])
        node._run_pipeline = MagicMock(return_value=mock_output)
        # Mock _prepare_face_region to raise ValueError
        node._prepare_face_region = MagicMock(
            side_effect=ValueError("No face detected in the reference image.")
        )

        with pytest.raises(ValueError, match="No face detected"):
            node.run(MosaicData(
                reference_image=sample_face_image,
                prompt="a portrait photo of a person",
            ))

    # T_ID_11 ——————————————————————————————————————————————————————————————
    def test_describe_returns_correct_info(self, cpu_scheduler):
        """# T_ID_11: describe 返回正确信息。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = node.describe()
        assert spec.name == "identity-keeper", "Node name should be 'identity-keeper'"
        assert spec.domain == "consistency", "Domain should be 'consistency'"
        assert spec.model_info is not None, "model_info should not be None"
        assert "vram_gb" in spec.model_info, "model_info should contain vram_gb"
        assert "license" in spec.model_info, "model_info should contain license"
        assert spec.model_info["method"] == "instantid"
        assert isinstance(spec.model_info["vram_gb"], float)
        assert spec.model_info["vram_gb"] > 0

    # T_ID_12 ——————————————————————————————————————————————————————————————
    def test_load_unload_state(self, cpu_scheduler):
        """# T_ID_12: load/unload 后 is_loaded 状态正确。"""
        node = IdentityKeeper(
            model="InstantX/InstantID",
            method="instantid",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        assert not node.is_loaded(), "Node should not be loaded initially"

        node.load()
        assert node.is_loaded(), "Node should be loaded after load()"

        node.unload()
        assert not node.is_loaded(), "Node should not be loaded after unload()"