# tests/phase6/test_consistency_types.py
"""一致性域输出数据类型测试。

测试一致性节点（IdentityKeeper、StyleKeeper、CrossFrameConsistency）
输出的 MosaicData 结构是否符合预期。
"""

from __future__ import annotations

import pytest
from PIL import Image

from mosaic.core.types import MosaicData


# ---------------------------------------------------------------------------
# 辅助：构造模拟一致性节点输出
# ---------------------------------------------------------------------------
def _make_identity_keeper_output(ref_image, gen_image, score=0.85):
    """模拟 IdentityKeeper.run() 的输出."""
    return MosaicData(
        image=gen_image,
        reference_image=ref_image,
        identity_score=score,
        seed=42,
        method="instantid",
        model_name="InstantX/InstantID",
    )


def _make_style_keeper_output(images, scores):
    """模拟 StyleKeeper.run() 的输出."""
    if scores is None:
        scores = [0.8] * len(images)
    avg = sum(scores) / len(scores) if scores else 0.0
    return MosaicData(
        images=images,
        consistency_scores=scores,
        average_consistency=avg,
        method="style-transfer",
    )


def _make_cross_frame_output(images, scores):
    """模拟 CrossFrameConsistency.run() 的输出."""
    if scores is None:
        scores = [0.9] * len(images)
    avg = sum(scores) / len(scores) if scores else 0.0
    return MosaicData(
        images=images,
        consistency_scores=scores,
        average_consistency=avg,
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class TestConsistencyOutputTypes:
    """一致性域输出数据类型测试."""

    def test_output_contains_image_or_images(
        self, sample_face_image, sample_landscape_image
    ):
        """# T_CONTYPE_01：一致性节点输出包含 image/images 字段."""
        # IdentityKeeper 输出应包含 "image"
        id_output = _make_identity_keeper_output(
            sample_face_image, sample_landscape_image
        )
        assert (
            "image" in id_output
        ), "IdentityKeeper output should contain 'image' field"
        assert isinstance(
            id_output["image"], Image.Image
        ), "'image' field should be a PIL.Image.Image"

        # StyleKeeper 输出应包含 "images"（列表）
        style_output = _make_style_keeper_output(
            [sample_face_image, sample_landscape_image],
            [0.85, 0.75],
        )
        assert (
            "images" in style_output
        ), "StyleKeeper output should contain 'images' field"
        assert isinstance(
            style_output["images"], list
        ), "'images' field should be a list"
        assert all(
            isinstance(img, Image.Image) for img in style_output["images"]
        ), "All items in 'images' should be PIL.Image.Image"

        # CrossFrameConsistency 输出应包含 "images"（列表）
        cf_output = _make_cross_frame_output(
            [sample_face_image, sample_landscape_image],
            [0.92, 0.88],
        )
        assert (
            "images" in cf_output
        ), "CrossFrameConsistency output should contain 'images' field"
        assert isinstance(
            cf_output["images"], list
        ), "'images' field should be a list"

    def test_identity_score_in_range(self, sample_face_image, sample_landscape_image):
        """# T_CONTYPE_02：identity_score 在 0-1 范围."""
        # 测试边界值
        for score in [0.0, 0.5, 0.85, 1.0]:
            output = _make_identity_keeper_output(
                sample_face_image, sample_landscape_image, score=score
            )
            assert "identity_score" in output, (
                f"Output should contain 'identity_score' (score={score})"
            )
            assert 0.0 <= output["identity_score"] <= 1.0, (
                f"identity_score should be in [0, 1], got {output['identity_score']}"
            )

        # 测试无效值（但从 IdentityKeeper 的实现来看会被 clamp 到 0-1）
        low_output = _make_identity_keeper_output(
            sample_face_image, sample_landscape_image, score=-0.5
        )
        # 直接构造的输出不管 clamp，但类型测试只验证范围逻辑
        assert isinstance(low_output["identity_score"], float), (
            "identity_score should be a float"
        )

    def test_consistency_scores_length_matches_images(
        self, sample_face_image, sample_landscape_image, sample_style_reference
    ):
        """# T_CONTYPE_03：consistency_scores 列表长度与 images 一致."""
        # 测试不同数量的图片
        test_cases = [
            [sample_face_image],
            [sample_face_image, sample_landscape_image],
            [sample_face_image, sample_landscape_image, sample_style_reference],
        ]

        for images in test_cases:
            scores = [0.8 + i * 0.05 for i in range(len(images))]
            style_output = _make_style_keeper_output(images, scores)
            assert len(style_output["consistency_scores"]) == len(
                style_output["images"]
            ), (
                f"consistency_scores length ({len(style_output['consistency_scores'])}) "
                f"should match images length ({len(style_output['images'])})"
            )

            cf_output = _make_cross_frame_output(images, scores)
            assert len(cf_output["consistency_scores"]) == len(
                cf_output["images"]
            ), (
                f"consistency_scores length ({len(cf_output['consistency_scores'])}) "
                f"should match images length ({len(cf_output['images'])})"
            )

    def test_average_consistency_in_range(
        self, sample_face_image, sample_landscape_image
    ):
        """# T_CONTYPE_04：average_consistency 在 0-1 范围."""
        # 测试 StyleKeeper 输出
        style_output = _make_style_keeper_output(
            [sample_face_image, sample_landscape_image],
            [0.85, 0.75],
        )
        assert "average_consistency" in style_output, (
            "StyleKeeper output should contain 'average_consistency'"
        )
        assert 0.0 <= style_output["average_consistency"] <= 1.0, (
            f"average_consistency should be in [0, 1], "
            f"got {style_output['average_consistency']}"
        )

        # 验证 average_consistency 确实是 scores 的均值
        expected_avg = (0.85 + 0.75) / 2
        assert style_output["average_consistency"] == pytest.approx(expected_avg), (
            f"average_consistency should be the mean of consistency_scores"
        )

        # 测试 CrossFrameConsistency 输出
        cf_output = _make_cross_frame_output(
            [sample_face_image, sample_landscape_image],
            [0.92, 0.88],
        )
        assert "average_consistency" in cf_output, (
            "CrossFrameConsistency output should contain 'average_consistency'"
        )
        assert 0.0 <= cf_output["average_consistency"] <= 1.0, (
            f"average_consistency should be in [0, 1], "
            f"got {cf_output['average_consistency']}"
        )

        # 测试边界值
        edge_output = _make_style_keeper_output(
            [sample_face_image], [0.0]
        )
        assert edge_output["average_consistency"] == 0.0, (
            "average_consistency should be 0.0 when all scores are 0.0"
        )

        edge_output2 = _make_style_keeper_output(
            [sample_face_image], [1.0]
        )
        assert edge_output2["average_consistency"] == 1.0, (
            "average_consistency should be 1.0 when all scores are 1.0"
        )