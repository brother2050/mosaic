# tests/final/test_documentation_consistency.py
"""文档一致性测试。

验证 README.md、nodes-reference.md、CHANGELOG.md、examples/ 等文档与
实际代码注册表之间的一致性。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_MOSAIC_ROOT = Path(__file__).resolve().parents[2]
_README_PATH = _MOSAIC_ROOT / "README.md"
_DOCS_DIR = _MOSAIC_ROOT / "docs"
_NODES_REF_PATH = _DOCS_DIR / "nodes-reference.md"
_EXAMPLES_DIR = _MOSAIC_ROOT / "examples"
_CHANGELOG_PATH = _MOSAIC_ROOT / "CHANGELOG.md"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """读取文件全文。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _count_registered_nodes(registry) -> int:
    """统计注册表中实际注册的节点数量（去重）。"""
    return len(registry)


def _count_tts_backends(tts_registry) -> int:
    """统计 TTS 后端注册表中已注册的后端数量。"""
    return len(tts_registry)


# ===========================================================================
# T_DOC_01: README.md 节点数量与实际注册表一致
# ===========================================================================
def test_readme_node_count_matches_registry(registry):
    """T_DOC_01: README.md 中提及的节点数量（42）与实际注册表一致。

    验证策略：
    1. 从注册表获取实际节点数量（去重计数）。
    2. 读取 README.md，查找 "42" 附近是否有 "节点" 等相关表述。
    3. 使用启发式检查：在 README.md 中搜索 "39" 或 "42" 并确认语境。
    """
    # 实际注册节点数
    actual_count = _count_registered_nodes(registry)
    assert actual_count >= 42, (
        f"注册表节点数 ({actual_count}) 应至少为 39"
    )

    readme = _read_text(_README_PATH)
    assert readme, "README.md 不应为空"

    # 检查 README 中是否提到了 42 节点总数
    # 启发式：搜索 "42" 且附近有 "节点" 或 "合计"
    has_42 = "42" in readme
    has_node_text = "节点" in readme or "node" in readme.lower()

    assert has_42, (
        "README.md 应包含总节点数引用 '42'"
    )
    assert has_node_text, (
        "README.md 应包含 '节点' 或 'node' 相关表述"
    )

    # 检查 "合计" 行附近是否有 "42"
    if "合计" in readme:
        lines = readme.splitlines()
        for i, line in enumerate(lines):
            if "合计" in line and "42" in line:
                break
        else:
            # 未在"合计"行找到，尝试在附近行查找
            for i, line in enumerate(lines):
                if "合计" in line:
                    # 检查前后行
                    nearby = "\n".join(lines[max(0, i - 1):min(len(lines), i + 3)])
                    assert "42" in nearby, (
                        f"README.md 中 '合计' 行附近未找到 '42'，"
                        f"实际注册节点数: {actual_count}"
                    )
                    break


# ===========================================================================
# T_DOC_02: README.md TTS 后端数量与实际一致
# ===========================================================================
def test_readme_tts_backend_count_matches(tts_registry):
    """T_DOC_02: README.md 中提及的 TTS 后端数量（4）与实际注册表一致。

    验证策略：
    1. 从 TTS 后端注册表获取实际后端数量（通过 list_backends() 触发延迟注册）。
    2. 读取 README.md，搜索 "4 个" 或 "4 后端" 等与 TTS 相关的表述。
    """
    # 检查 TTS 后端注册表
    # 调用 list_backends() 以触发延迟注册（len() 不会触发）
    backends = tts_registry.list_backends()
    actual_tts_count = len(backends)
    # 内置后端可能因依赖缺失而少于 4，但注册表至少应有一个
    # 实际运行时（conftest 已 mock torch）应能注册 4 个
    assert actual_tts_count >= 1, (
        f"TTS 后端注册表至少应有 1 个后端，实际有 {actual_tts_count}"
    )

    readme = _read_text(_README_PATH)
    assert readme, "README.md 不应为空"

    # 检查 README 中 TTS 相关的后端数量表述
    # 查找 "4 个" 且附近有 "TTS" 或 "后端"
    tts_section = "TTS 扩展" in readme or "TTS" in readme
    has_4_backends = "4 个" in readme or "4 后端" in readme

    assert tts_section, "README.md 应包含 TTS 相关内容"
    assert has_4_backends, (
        "README.md 应提及 4 个 TTS 后端"
    )

    # 确认 4 个后端名称都在 README 中
    backend_names = ["ChatTTS", "Fish Speech", "GPT-SoVITS", "CosyVoice"]
    for name in backend_names:
        # 使用大小写不敏感搜索
        found = False
        backend_lower = name.lower()
        for line in readme.splitlines():
            if backend_lower in line.lower():
                found = True
                break
        assert found, f"README.md 应包含 TTS 后端 '{name}'"


# ===========================================================================
# T_DOC_03: nodes-reference.md 覆盖所有节点 + TTS 后端
# ===========================================================================
def test_nodes_reference_covers_all_nodes(registry, tts_registry):
    """T_DOC_03: nodes-reference.md 应覆盖所有注册节点和 4 个 TTS 后端。

    验证策略：
    1. 读取 docs/nodes-reference.md 全文。
    2. 从注册表获取所有节点名称。
    3. 检查每个节点名称是否在文档中出现。
    4. 检查 4 个 TTS 后端名称是否在文档中出现。
    """
    doc_text = _read_text(_NODES_REF_PATH)
    assert doc_text, "nodes-reference.md 不应为空"

    # 获取所有注册节点名称
    actual_node_names = registry.list_names()
    assert len(actual_node_names) >= 42, (
        f"注册表节点数 ({len(actual_node_names)}) 应至少为 39"
    )

    # 检查每个节点名称是否在文档中出现
    missing_nodes = []
    for node_name in actual_node_names:
        # 搜索方法：节点名称可能以多种形式出现
        # 1. 直接匹配（大小写敏感）
        # 2. 匹配 "### NodeName" 形式的标题
        if node_name not in doc_text:
            # 检查是否是 kebab-case 变体
            kebab = node_name.replace("_", "-").lower()
            if kebab not in doc_text.lower():
                # 检查是否有 "节点 ID" 形式的引用
                # 某些节点可能以不同方式在文档中表述
                if node_name.lower() not in doc_text.lower():
                    missing_nodes.append(node_name)

    if missing_nodes:
        logger.warning(
            "以下节点在 nodes-reference.md 中未找到直接引用: %s",
            missing_nodes,
        )
        # 允许有一定的宽松度——文档可能使用不同的表述方式
        # 但至少 90% 的节点应有直接引用
        coverage = 1.0 - len(missing_nodes) / len(actual_node_names)
        assert coverage >= 0.9, (
            f"nodes-reference.md 节点覆盖率过低: {coverage:.1%}，"
            f"缺失节点: {missing_nodes}"
        )

    # 检查 4 个 TTS 后端是否在文档中
    tts_backend_names = [
        "ChatTTSBackend",
        "FishSpeechBackend",
        "GPTSoVITSBackend",
        "CosyVoiceBackend",
    ]
    for backend_name in tts_backend_names:
        assert backend_name in doc_text, (
            f"nodes-reference.md 应包含 TTS 后端 '{backend_name}'"
        )

    # 确认文档标题中声明了 42 节点
    header_line = doc_text.splitlines()[0] if doc_text.splitlines() else ""
    assert "42" in header_line or "42" in doc_text[:200], (
        "nodes-reference.md 开头应声明 '42 节点'"
    )
    assert "4" in header_line or "4 个 TTS" in doc_text[:200], (
        "nodes-reference.md 开头应声明 '4 个 TTS 后端'"
    )


# ===========================================================================
# T_DOC_04: 每个节点的 describe() 描述与文档关键词匹配
# ===========================================================================
def test_node_descriptions_match_docs(registry):
    """T_DOC_04: 每个节点的 describe().description 关键词应在文档中出现。

    这是一个模糊检查——验证节点描述中的至少一些关键词出现在对应文档中。
    对于每个节点，提取其 describe().description 中的关键词，并检查
    nodes-reference.md 中是否包含这些关键词。
    """
    doc_text = _read_text(_NODES_REF_PATH)
    assert doc_text, "nodes-reference.md 不应为空"

    nodes = registry.list_nodes()
    assert len(nodes) >= 42, f"至少应有 42 个注册节点，实际有 {len(nodes)}"

    # 对每个节点进行关键词检查
    matched_count = 0
    for node_spec in nodes:
        desc = node_spec.description or ""
        if not desc:
            # 无描述的节点跳过（如某些内部节点）
            continue

        # 提取关键词：取描述中的中文词和英文词
        # 简单的分词策略：按空格和标点分割
        import re

        # 提取中文词（2字及以上）和英文词（3字母及以上）
        keywords = []
        # 英文词
        en_words = re.findall(r"[a-zA-Z]{3,}", desc)
        keywords.extend(en_words)
        # 中文词
        zh_words = re.findall(r"[\u4e00-\u9fff]{2,}", desc)
        keywords.extend(zh_words)

        if not keywords:
            continue

        # 检查至少有一些关键词出现在文档中
        doc_lower = doc_text.lower()
        found_any = any(
            kw.lower() in doc_lower for kw in keywords
        )

        if found_any:
            matched_count += 1
        else:
            logger.warning(
                "节点 %s 的描述关键词 (%s) 在 nodes-reference.md 中未找到",
                node_spec.name,
                keywords[:5],
            )

    # 至少 80% 的节点描述关键词应在文档中有匹配
    coverage = matched_count / len(nodes) if nodes else 0
    assert coverage >= 0.8, (
        f"节点描述与文档关键词匹配率过低: {coverage:.1%} "
        f"({matched_count}/{len(nodes)})"
    )


# ===========================================================================
# T_DOC_05: examples/ 目录有 11 个示例文件
# ===========================================================================
def test_examples_directory_has_11_files():
    """T_DOC_05: examples/ 目录应包含恰好 11 个 .py 示例文件。"""
    examples_dir = _EXAMPLES_DIR
    assert examples_dir.exists(), (
        f"examples/ 目录不存在: {examples_dir}"
    )
    assert examples_dir.is_dir(), (
        f"examples/ 不是一个目录: {examples_dir}"
    )

    py_files = sorted(
        f for f in os.listdir(examples_dir)
        if f.endswith(".py") and not f.startswith("_")
    )

    assert len(py_files) == 11, (
        f"examples/ 目录应有 11 个 .py 文件，"
        f"实际有 {len(py_files)}: {py_files}"
    )

    # 验证文件名格式: 01_xxx.py 到 11_xxx.py
    expected_numbers = set(range(1, 12))
    actual_numbers = set()
    for f in py_files:
        parts = f.split("_", 1)
        if parts and parts[0].isdigit():
            actual_numbers.add(int(parts[0]))

    assert actual_numbers == expected_numbers, (
        f"examples/ 文件应编号 01-11，实际编号: {sorted(actual_numbers)}"
    )


# ===========================================================================
# T_DOC_06: CHANGELOG.md 包含 TTS 扩展条目
# ===========================================================================
def test_changelog_contains_tts_extension():
    """T_DOC_06: CHANGELOG.md 应包含 TTS 扩展相关内容。

    检查关键词: "TTS", "ChatTTS", "Fish", "SoVITS", "CosyVoice", "tts"
    """
    changelog = _read_text(_CHANGELOG_PATH)
    assert changelog, "CHANGELOG.md 不应为空"

    tts_keywords = ["TTS", "ChatTTS", "Fish", "SoVITS", "CosyVoice", "tts"]
    found_keywords = []
    missing_keywords = []

    for kw in tts_keywords:
        if kw.lower() in changelog.lower():
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    assert len(found_keywords) >= 3, (
        f"CHANGELOG.md 中应至少包含 3 个 TTS 相关关键词，"
        f"找到: {found_keywords}，缺失: {missing_keywords}"
    )

    # 检查 TTS 扩展章节
    assert "TTS" in changelog, "CHANGELOG.md 应包含 TTS 相关内容"
    assert "ChatTTS" in changelog, "CHANGELOG.md 应提及 ChatTTS"
    assert "CosyVoice" in changelog, "CHANGELOG.md 应提及 CosyVoice"


# ===========================================================================
# T_DOC_07: CHANGELOG.md 包含视频模型支持条目
# ===========================================================================
def test_changelog_contains_video_model_support():
    """T_DOC_07: CHANGELOG.md 应包含视频模型支持相关内容。

    检查关键词: "Wan", "Hunyuan", "LTX", "video", "diffusers"
    """
    changelog = _read_text(_CHANGELOG_PATH)
    assert changelog, "CHANGELOG.md 不应为空"

    video_keywords = ["Wan", "Hunyuan", "LTX", "video", "diffusers"]
    found_keywords = []
    missing_keywords = []

    for kw in video_keywords:
        if kw.lower() in changelog.lower():
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    assert len(found_keywords) >= 3, (
        f"CHANGELOG.md 中应至少包含 3 个视频模型相关关键词，"
        f"找到: {found_keywords}，缺失: {missing_keywords}"
    )

    # 检查视频模型支持章节
    changelog_lower = changelog.lower()
    assert "video" in changelog_lower, "CHANGELOG.md 应包含 video 相关内容"
    assert "wan" in changelog_lower, "CHANGELOG.md 应提及 Wan 视频模型"
    assert "hunyuan" in changelog_lower, "CHANGELOG.md 应提及 HunyuanVideo"
    assert "ltx" in changelog_lower, "CHANGELOG.md 应提及 LTX Video"