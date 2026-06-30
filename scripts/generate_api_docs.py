#!/usr/bin/env python3
"""
scripts/generate_api_docs.py
自动从节点 docstring 生成 Markdown API 文档。

扫描 mosaic/nodes/ 下的所有节点、核心模块、TTS 后端，
输出到 docs/api/ 目录，按域分组，每个节点/后端一页。

用法：
    python scripts/generate_api_docs.py
    python scripts/generate_api_docs.py --output docs/api --project-root .
"""
from __future__ import annotations

import argparse
import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Any


def find_node_classes(module_path: Path) -> list[tuple[str, type]]:
    """从 Python 文件中找出所有继承自 Node 的类。"""
    if not module_path.exists():
        return []

    try:
        # 读取源码并解析
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 简化判断：类名以 Node / Backend 结尾或位于 nodes 目录
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in ("Node", "BaseVideoNode", "BaseImageNode", "TTSBackend"):
                    classes.append((node.name, ast.unparse(node) if hasattr(ast, "unparse") else ""))
                    break
    return classes


def get_class_info(class_name: str, module_name: str) -> dict[str, Any]:
    """尝试导入并获取类的完整信息。"""
    info: dict[str, Any] = {
        "name": class_name,
        "module": module_name,
        "docstring": "",
        "bases": [],
        "attributes": {},
        "methods": [],
    }

    try:
        # 动态导入
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name, None)
        if cls is None:
            return info

        info["docstring"] = inspect.getdoc(cls) or ""
        info["bases"] = [b.__name__ for b in cls.__bases__ if b is not object]

        # 类属性
        for name, value in vars(cls).items():
            if not name.startswith("_") and not callable(value):
                if isinstance(value, (str, int, float, bool, list, type(None))):
                    info["attributes"][name] = repr(value)

        # 方法
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_") or name in ("__init__", "__call__"):
                info["methods"].append({
                    "name": name,
                    "docstring": inspect.getdoc(method) or "",
                    "signature": str(inspect.signature(method)),
                })
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info


def render_node_doc(info: dict[str, Any]) -> str:
    """将类信息渲染为 Markdown。"""
    lines = []

    # 标题
    lines.append(f"# `{info['name']}`")
    lines.append("")

    # 基本信息
    lines.append(f"**模块**：`{info['module']}`")
    if info["bases"]:
        lines.append(f"**继承**：`{' -> '.join(info['bases'])}`")
    lines.append("")

    # 文档字符串
    if info["docstring"]:
        lines.append("## 描述")
        lines.append("")
        lines.append(info["docstring"])
        lines.append("")

    # 类属性
    if info["attributes"]:
        lines.append("## 类属性")
        lines.append("")
        lines.append("| 名称 | 值 |")
        lines.append("|---|---|")
        for name, value in info["attributes"].items():
            lines.append(f"| `{name}` | `{value}` |")
        lines.append("")

    # 方法
    if info["methods"]:
        lines.append("## 方法")
        lines.append("")
        for method in info["methods"]:
            lines.append(f"### `{method['name']}{method['signature']}`")
            lines.append("")
            if method["docstring"]:
                lines.append(method["docstring"])
                lines.append("")

    if "error" in info:
        lines.append(f"\n> ⚠️ 错误：{info['error']}\n")

    return "\n".join(lines)


def generate_domain_docs(
    domain: str,
    nodes_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> int:
    """为某个域生成文档。"""
    domain_path = nodes_dir / domain
    if not domain_path.exists():
        return 0

    output_domain_dir = output_dir / domain
    output_domain_dir.mkdir(parents=True, exist_ok=True)

    # 写域索引
    index_lines = [
        f"# {domain.upper()} 域 API 文档",
        "",
        "本目录包含 `mosaic.nodes.{domain}` 下所有节点的自动生成 API 文档。",
        "",
        "## 节点列表",
        "",
    ]

    count = 0
    for py_file in sorted(domain_path.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        module_name = f"mosaic.nodes.{domain}.{py_file.stem}"
        classes = find_node_classes(py_file)

        for class_name, _ in classes:
            info = get_class_info(class_name, module_name)
            doc = render_node_doc(info)
            doc_path = output_domain_dir / f"{class_name}.md"
            doc_path.write_text(doc, encoding="utf-8")

            index_lines.append(f"- [`{class_name}`](./{class_name}.md)")
            count += 1

    index_path = output_domain_dir / "README.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return count


def generate_core_docs(mosaic_dir: Path, output_dir: Path) -> int:
    """为核心模块生成文档。"""
    core_path = mosaic_dir / "core"
    if not core_path.exists():
        return 0

    output_core_dir = output_dir / "core"
    output_core_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for py_file in sorted(core_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"mosaic.core.{py_file.stem}"
        classes = find_node_classes(py_file)

        for class_name, _ in classes:
            info = get_class_info(class_name, module_name)
            doc = render_node_doc(info)
            doc_path = output_core_dir / f"{class_name}.md"
            doc_path.write_text(doc, encoding="utf-8")
            count += 1

    return count


def generate_tts_backend_docs(audio_dir: Path, output_dir: Path) -> int:
    """为 TTS 后端生成文档。"""
    backends_path = audio_dir / "tts_backends"
    if not backends_path.exists():
        return 0

    output_backends_dir = output_dir / "audio" / "tts_backends"
    output_backends_dir.mkdir(parents=True, exist_ok=True)

    impl_path = backends_path / "implementations"
    count = 0

    if impl_path.exists():
        for py_file in sorted(impl_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"mosaic.nodes.audio.tts_backends.implementations.{py_file.stem}"
            classes = find_node_classes(py_file)

            for class_name, _ in classes:
                info = get_class_info(class_name, module_name)
                doc = render_node_doc(info)
                doc_path = output_backends_dir / f"{class_name}.md"
                doc_path.write_text(doc, encoding="utf-8")
                count += 1

    return count


def generate_index(output_dir: Path, counts: dict[str, int]) -> None:
    """生成总索引页。"""
    lines = [
        "# Mosaic API 文档",
        "",
        "> 本目录由 `scripts/generate_api_docs.py` 自动生成。",
        "",
        "## 目录",
        "",
    ]

    for section, count in counts.items():
        lines.append(f"- **{section}**：{count} 个 API 页面")
    lines.append("")

    lines.append("## 各域概览")
    lines.append("")
    lines.append("| 域 | 节点数 | 链接 |")
    lines.append("|---|---|---|")
    domains = [
        ("text", "文本域"),
        ("image", "图像域"),
        ("video", "视频域"),
        ("audio", "音频域"),
        ("subtitle", "字幕域"),
        ("consistency", "一致性域"),
        ("digital-human", "数字人域"),
        ("export", "导出域"),
        ("rag", "RAG 域"),
    ]
    for domain, label in domains:
        index_file = output_dir / domain / "README.md"
        if index_file.exists():
            lines.append(f"| {label} | {counts.get(f'domain_{domain}', 0)} | [查看](./{domain}/README.md) |")
        else:
            lines.append(f"| {label} | - | - |")
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成 Mosaic API 文档")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/workspace/mosaic"),
        help="mosaic 项目根目录（包含 mosaic/ 目录）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录（默认：<project_root>/docs/api）",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if not project_root.exists():
        print(f"❌ 项目根目录不存在: {project_root}")
        sys.exit(1)

    output_dir = args.output or (project_root / "docs" / "api")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把项目根加入 sys.path 以便 import
    sys.path.insert(0, str(project_root))

    mosaic_dir = project_root / "mosaic"
    nodes_dir = mosaic_dir / "nodes"
    audio_dir = nodes_dir / "audio"

    print(f"📁 项目根: {project_root}")
    print(f"📁 输出: {output_dir}")
    print()

    counts = {}

    # 各域
    domains = [
        "text", "image", "video", "audio", "subtitle",
        "consistency", "digital_human", "export", "rag",
    ]
    for domain in domains:
        n = generate_domain_docs(domain, nodes_dir, output_dir, project_root)
        counts[f"domain_{domain}"] = n
        print(f"✅ {domain:15s}: {n} 个节点")

    # 核心模块
    n = generate_core_docs(mosaic_dir, output_dir)
    counts["core"] = n
    print(f"✅ core (核心模块): {n} 个类")

    # TTS 后端
    n = generate_tts_backend_docs(audio_dir, output_dir)
    counts["tts_backends"] = n
    print(f"✅ TTS 后端: {n} 个")

    # 总索引
    generate_index(output_dir, counts)
    print()
    print(f"🎉 完成！共生成 {sum(counts.values())} 个 API 页面")
    print(f"   索引：{output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
