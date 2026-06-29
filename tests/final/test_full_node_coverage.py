# tests/final/test_full_node_coverage.py
"""全节点覆盖测试。

验证 Mosaic 框架中所有 39 个节点均已正确注册，覆盖 9 大域，
并验证每个节点的 NodeSpec 元信息和 TTS 后端注册表。

测试 ID 约定：
    T_COV_01 ~ T_COV_16 分别对应不同的覆盖检查项。
"""

from __future__ import annotations

import pytest

from mosaic.core import registry as _global_registry
from mosaic.core.node import NodeSpec


# ============================================================================
# 辅助函数
# ============================================================================
def _get_node_names_in_domain(domain: str) -> set[str]:
    """获取指定域中所有节点的名称集合（使用类名/PascalCase）。"""
    specs = _global_registry.list_nodes(domain=domain)
    # 尝试用类名（PascalCase）匹配，因为 registry 同时注册了 name 和 __name__
    names: set[str] = set()
    for spec in specs:
        try:
            node_class = _global_registry.get_class(spec.name)
            names.add(node_class.__name__)
        except KeyError:
            names.add(spec.name)
    return names


# ============================================================================
# T_COV_01：总节点数 == 39
# ============================================================================
class TestTotalNodeCount:
    """验证注册表中的总节点数。"""

    def test_total_nodes_equals_42(self, registry: object) -> None:
        """T_COV_01: 注册表中共有 42 个节点。"""
        total = len(registry)
        assert total == 42, (
            f"Expected 42 total nodes in registry, but got {total}. "
            f"Registered names: {registry.list_names()}"
        )


# ============================================================================
# T_COV_02 ~ T_COV_10：各域节点数及名称检查
# ============================================================================
class TestDomainNodeCounts:
    """验证每个域的节点数量与预期节点名称。"""

    # -- Text ---------------------------------------------------------------
    def test_text_domain_6_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_02: text 域有 6 个节点，且名称正确。"""
        domain = "text"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 6, (
            f"Expected 6 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Image --------------------------------------------------------------
    def test_image_domain_6_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_03: image 域有 6 个节点，且名称正确。"""
        domain = "image"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 6, (
            f"Expected 6 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Video --------------------------------------------------------------
    def test_video_domain_8_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_04: video 域有 8 个节点，且名称正确。"""
        domain = "video"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 8, (
            f"Expected 8 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Audio --------------------------------------------------------------
    def test_audio_domain_5_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_05: audio 域有 5 个节点，且名称正确。"""
        domain = "audio"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 5, (
            f"Expected 5 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Subtitle -----------------------------------------------------------
    def test_subtitle_domain_3_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_06: subtitle 域有 3 个节点，且名称正确。"""
        domain = "subtitle"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 3, (
            f"Expected 3 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Consistency --------------------------------------------------------
    def test_consistency_domain_3_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_07: consistency 域有 3 个节点，且名称正确。"""
        domain = "consistency"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 3, (
            f"Expected 3 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Digital Human ------------------------------------------------------
    def test_digital_human_domain_4_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_08: digital_human 域有 4 个节点，且名称正确。"""
        domain = "digital_human"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 4, (
            f"Expected 4 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- Export -------------------------------------------------------------
    def test_export_domain_3_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_09: export 域有 3 个节点，且名称正确。"""
        domain = "export"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 3, (
            f"Expected 3 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )

    # -- RAG ----------------------------------------------------------------
    def test_rag_domain_4_nodes(self, registry: object, expected_node_names: dict) -> None:
        """T_COV_10: rag 域有 4 个节点，且名称正确。"""
        domain = "rag"
        spec_count = len(registry.list_nodes(domain=domain))
        expected_names = expected_node_names[domain]

        assert spec_count == 4, (
            f"Expected 4 nodes in '{domain}' domain, but got {spec_count}. "
            f"Nodes: {[s.name for s in registry.list_nodes(domain=domain)]}"
        )
        for name in expected_names:
            assert name in registry, (
                f"Expected node '{name}' to be registered in '{domain}' domain, "
                f"but it was not found."
            )


# ============================================================================
# T_COV_11 ~ T_COV_14：NodeSpec 字段完整性
# ============================================================================
class TestNodeSpecIntegrity:
    """验证每个节点的 NodeSpec 元信息完整性。"""

    @pytest.mark.parametrize(
        "node_name",
        [
            # text
            "TextGenerator", "Chat", "TextRewriter", "Translator",
            "TextSummarizer", "TextClassifier",
            # image
            "TextToImage", "ImageToImage", "Inpainting", "Upscaler",
            "BackgroundRemover", "Stylizer",
            # video
            "TextToVideo", "ImageToVideo", "VideoContinuation",
            "FrameInterpolator", "FrameExtractor",
            # audio
            "TTS", "ASR", "MusicGenerator", "SoundEffectGenerator", "VoiceClone",
            # subtitle
            "SubtitleGenerator", "SubtitleTranslator", "SubtitleAligner",
            # consistency
            "IdentityKeeper", "StyleKeeper", "CrossFrameConsistency",
            # digital_human
            "AvatarDriver", "LipSyncer", "MotionGenerator", "RealtimeRenderer",
            # export
            "VideoEncoder", "Livestreamer", "MultiFormatExporter",
            # rag
            "DocumentParser", "VectorIndexer", "Retriever", "CitationGenerator",
        ],
    )
    def test_node_describe_returns_valid_spec(
        self, registry: object, node_name: str
    ) -> None:
        """T_COV_11: 每个节点的 describe() 返回不含空字段的 NodeSpec。"""
        node = registry.get(node_name)
        spec = node.describe()

        assert isinstance(spec, NodeSpec), (
            f"Node '{node_name}': describe() must return NodeSpec, "
            f"got {type(spec).__name__}"
        )
        assert spec.name, (
            f"Node '{node_name}': NodeSpec.name must not be empty."
        )
        assert spec.domain, (
            f"Node '{node_name}': NodeSpec.domain must not be empty."
        )
        assert spec.description, (
            f"Node '{node_name}': NodeSpec.description must not be empty."
        )
        assert spec.version, (
            f"Node '{node_name}': NodeSpec.version must not be empty."
        )

    @pytest.mark.parametrize(
        "node_name",
        [
            "TextGenerator", "Chat", "TextRewriter", "Translator",
            "TextSummarizer", "TextClassifier",
            "TextToImage", "ImageToImage", "Inpainting", "Upscaler",
            "BackgroundRemover", "Stylizer",
            "TextToVideo", "ImageToVideo", "VideoContinuation",
            "FrameInterpolator", "FrameExtractor",
            "TTS", "ASR", "MusicGenerator", "SoundEffectGenerator", "VoiceClone",
            "SubtitleGenerator", "SubtitleTranslator", "SubtitleAligner",
            "IdentityKeeper", "StyleKeeper", "CrossFrameConsistency",
            "AvatarDriver", "LipSyncer", "MotionGenerator", "RealtimeRenderer",
            "VideoEncoder", "Livestreamer", "MultiFormatExporter",
            "DocumentParser", "VectorIndexer", "Retriever", "CitationGenerator",
        ],
    )
    def test_node_spec_has_required_fields(
        self, registry: object, node_name: str
    ) -> None:
        """T_COV_12: 每个 NodeSpec 均有 name/domain/description/version 字段。"""
        node = registry.get(node_name)
        spec = node.describe()

        assert hasattr(spec, "name"), (
            f"Node '{node_name}': NodeSpec missing 'name' attribute."
        )
        assert hasattr(spec, "domain"), (
            f"Node '{node_name}': NodeSpec missing 'domain' attribute."
        )
        assert hasattr(spec, "description"), (
            f"Node '{node_name}': NodeSpec missing 'description' attribute."
        )
        assert hasattr(spec, "version"), (
            f"Node '{node_name}': NodeSpec missing 'version' attribute."
        )

    @pytest.mark.parametrize(
        "node_name",
        [
            "TextGenerator", "Chat", "TextRewriter", "Translator",
            "TextSummarizer", "TextClassifier",
            "TextToImage", "ImageToImage", "Inpainting", "Upscaler",
            "BackgroundRemover", "Stylizer",
            "TextToVideo", "ImageToVideo", "VideoContinuation",
            "FrameInterpolator", "FrameExtractor",
            "TTS", "ASR", "MusicGenerator", "SoundEffectGenerator", "VoiceClone",
            "SubtitleGenerator", "SubtitleTranslator", "SubtitleAligner",
            "IdentityKeeper", "StyleKeeper", "CrossFrameConsistency",
            "AvatarDriver", "LipSyncer", "MotionGenerator", "RealtimeRenderer",
            "VideoEncoder", "Livestreamer", "MultiFormatExporter",
            "DocumentParser", "VectorIndexer", "Retriever", "CitationGenerator",
        ],
    )
    def test_node_input_types_non_empty(
        self, registry: object, node_name: str
    ) -> None:
        """T_COV_13: 每个节点的 input_types 非空。"""
        node = registry.get(node_name)
        spec = node.describe()

        assert len(spec.input_types) > 0, (
            f"Node '{node_name}': input_types must be non-empty, "
            f"but got {spec.input_types}"
        )

    @pytest.mark.parametrize(
        "node_name",
        [
            "TextGenerator", "Chat", "TextRewriter", "Translator",
            "TextSummarizer", "TextClassifier",
            "TextToImage", "ImageToImage", "Inpainting", "Upscaler",
            "BackgroundRemover", "Stylizer",
            "TextToVideo", "ImageToVideo", "VideoContinuation",
            "FrameInterpolator", "FrameExtractor",
            "TTS", "ASR", "MusicGenerator", "SoundEffectGenerator", "VoiceClone",
            "SubtitleGenerator", "SubtitleTranslator", "SubtitleAligner",
            "IdentityKeeper", "StyleKeeper", "CrossFrameConsistency",
            "AvatarDriver", "LipSyncer", "MotionGenerator", "RealtimeRenderer",
            "VideoEncoder", "Livestreamer", "MultiFormatExporter",
            "DocumentParser", "VectorIndexer", "Retriever", "CitationGenerator",
        ],
    )
    def test_node_output_types_non_empty(
        self, registry: object, node_name: str
    ) -> None:
        """T_COV_14: 每个节点的 output_types 非空。"""
        node = registry.get(node_name)
        spec = node.describe()

        assert len(spec.output_types) > 0, (
            f"Node '{node_name}': output_types must be non-empty, "
            f"but got {spec.output_types}"
        )


# ============================================================================
# T_COV_15 ~ T_COV_16：TTS 后端注册表
# ============================================================================
class TestTTSBackendCoverage:
    """验证 TTS 后端注册表覆盖。"""

    def test_tts_registry_has_4_backends(self, tts_registry: object) -> None:
        """T_COV_15: TTS 后端注册表包含 4 个后端。"""
        backends = tts_registry.list_backends()
        count = len(backends)

        assert count == 4, (
            f"Expected 4 TTS backends, but got {count}. "
            f"Backends: {[b.name for b in backends]}"
        )

    def test_tts_backend_names_are_correct(self, tts_registry: object) -> None:
        """T_COV_16: 后端名称为 chattts、fish、sovits、cosyvoice。"""
        expected_names = {"chattts", "fish", "sovits", "cosyvoice"}
        backends = tts_registry.list_backends()
        actual_names = {b.name for b in backends}

        assert actual_names == expected_names, (
            f"Expected TTS backend names {expected_names}, "
            f"but got {actual_names}"
        )

        # 同时验证每个后端均可通过 get() 获取
        for name in expected_names:
            backend_cls = tts_registry.get(name)
            assert backend_cls is not None, (
                f"TTS backend '{name}' should be retrievable via tts_registry.get(), "
                f"but got None."
            )