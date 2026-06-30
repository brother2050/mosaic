# /workspace/mosaic/tests/phase2/test_text_to_image.py
"""Tests for TextToImage node (phase2)."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.nodes.image.text_to_image import TextToImage
from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_output(images):
    """Create a mock pipeline output with an ``images`` attribute."""
    output = MagicMock()
    output.images = images
    return output


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestTextToImage:
    """Tests for the TextToImage node."""

    @pytest.fixture(autouse=True)
    def _mock_pipeline_from_pretrained(self):
        """Mock AutoPipelineForText2Image.from_pretrained to avoid downloading real models."""
        with patch("diffusers.AutoPipelineForText2Image.from_pretrained") as mock_fp:
            mock_fp.return_value = MagicMock()
            yield

    # T_T2I_01 —————————————————————————————————————————————————————————————
    def test_basic_text_to_image(self, sample_image, cpu_scheduler):
        """T_T2I_01: Basic text-to-image generation.

        Mock _run_pipeline to return a mock output containing a list with one
        sample image.  Verify that the result MosaicData contains the expected
        keys and values.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        result = node.run(MosaicData(prompt="a cat sitting on a sofa"))

        # Output structure
        assert "images" in result
        assert isinstance(result["images"], list)
        assert len(result["images"]) == 1
        assert isinstance(result["images"][0], Image.Image)
        assert result["prompt"] == "a cat sitting on a sofa"
        assert result["model_name"] == "stabilityai/stable-diffusion-xl-base-1.0"
        assert isinstance(result["seed"], int)
        assert result["num_images"] == 1

        node._run_pipeline.assert_called_once()

    # T_T2I_02 —————————————————————————————————————————————————————————————
    def test_custom_size(self, sample_image, cpu_scheduler):
        """T_T2I_02: Custom image size (512x512).

        Verify that ``width`` and ``height`` parameters are forwarded to the
        pipeline call.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        node.run(MosaicData(prompt="test", width=512, height=512))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 512

    # T_T2I_03 —————————————————————————————————————————————————————————————
    def test_seed_reproducibility(self, sample_image, cpu_scheduler):
        """T_T2I_03: Seed reproducibility.

        When the same seed is supplied twice the same generator seed should be
        forwarded to the pipeline both times.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        with patch.object(node, "_prepare_seed", return_value=(42, "fixed_gen")):
            node.run(MosaicData(prompt="test", seed=42))
            node.run(MosaicData(prompt="test", seed=42))

        calls = node._run_pipeline.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["generator"] == "fixed_gen"
        assert calls[1][1]["generator"] == "fixed_gen"

    # T_T2I_04 —————————————————————————————————————————————————————————————
    def test_negative_prompt(self, sample_image, cpu_scheduler):
        """T_T2I_04: negative_prompt parameter.

        Verify that ``negative_prompt`` is forwarded to the pipeline when
        provided.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        node.run(MosaicData(prompt="a cat", negative_prompt="blurry, low quality"))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "blurry, low quality"

    # T_T2I_05 —————————————————————————————————————————————————————————————
    def test_num_images(self, sample_image, cpu_scheduler):
        """T_T2I_05: num_images > 1.

        Verify that ``num_images`` is forwarded as ``num_images_per_prompt``
        to the pipeline.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image, sample_image, sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        node.run(MosaicData(prompt="test", num_images=3))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["num_images_per_prompt"] == 3

    # T_T2I_06 —————————————————————————————————————————————————————————————
    def test_describe(self, cpu_scheduler):
        """T_T2I_06: describe() returns correct info including license.

        The NodeSpec should contain the expected name, domain, and model_info
        including the license string.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        spec = node.describe()

        assert spec.name == "text-to-image"
        assert spec.domain == "image"
        assert spec.version == "0.1.0"
        assert "image" in spec.output_types
        assert "text" in spec.input_types or "mosaic" in spec.input_types

        # Model info
        assert "model_info" in spec.to_dict()
        model_info = spec.model_info
        assert model_info["name"] == "stabilityai/stable-diffusion-xl-base-1.0"
        assert "license" in model_info
        assert "OpenRAIL" in model_info["license"]
        assert "vram_gb" in model_info
        assert model_info["vram_gb"] > 0

    # T_T2I_07 —————————————————————————————————————————————————————————————
    def test_load_unload_state(self, cpu_scheduler):
        """T_T2I_07: load/unload state transitions.

        Verify that ``is_loaded()`` transitions correctly between
        False and True across load/unload calls.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)

        # Initially unloaded
        assert node.is_loaded() is False

        # Load
        node.load()
        assert node.is_loaded() is True
        assert node._pipeline is not None

        # Unload
        node.unload()
        assert node.is_loaded() is False
        assert node._pipeline is None

    # ——————————————————————————————————————————————————————————————————————
    def test_missing_prompt_raises_value_error(self, cpu_scheduler):
        """T_T2I_01b: Missing prompt raises ValueError.

        Verify that calling run() without a prompt (or with a non-string
        prompt) raises ValueError.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)

        # Missing prompt entirely
        with pytest.raises(ValueError, match="prompt"):
            node.run(MosaicData())

        # prompt is not a string
        with pytest.raises(ValueError, match="prompt"):
            node.run(MosaicData(prompt=123))

    # ——————————————————————————————————————————————————————————————————————
    def test_default_parameters(self, sample_image, cpu_scheduler):
        """T_T2I_01c: Default parameters are applied.

        Verify that when optional parameters are omitted the pipeline receives
        the documented defaults.
        """
        node = TextToImage(device="cpu", scheduler=cpu_scheduler)
        mock_output = _make_mock_output([sample_image])
        node._run_pipeline = MagicMock(return_value=mock_output)

        node.run(MosaicData(prompt="test"))

        call_kwargs = node._run_pipeline.call_args[1]
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1024
        assert call_kwargs["num_inference_steps"] == 30
        assert call_kwargs["guidance_scale"] == 7.5
        assert call_kwargs["num_images_per_prompt"] == 1
        assert "negative_prompt" not in call_kwargs
        assert "generator" in call_kwargs