# mosaic/cli/main.py
"""Mosaic CLI 命令行入口。

提供 ``mosaic`` 命令行工具，支持节点查看、管道执行、节点模板生成、
环境诊断等功能。

子命令
------
- ``list``         列出所有已注册节点或插件
- ``info``         查看节点详细信息
- ``create-node``  生成节点模板代码
- ``run``          从 YAML/JSON 文件运行管道
- ``version``      显示版本号
- ``doctor``       环境诊断

使用示例
--------
::

    mosaic version
    mosaic list
    mosaic list --domain text
    mosaic list --plugins
    mosaic info text-generator
    mosaic run pipeline.yaml
    mosaic create-node --domain text --name sentiment --output ./my_nodes/
    mosaic doctor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from mosaic import __version__
from mosaic.core.node import NodeSpec
from mosaic.core.pipeline import Pipeline
from mosaic.core.plugin import plugin_manager
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

__all__ = ["main"]


# ---------------------------------------------------------------------------
# 表格格式化辅助
# ---------------------------------------------------------------------------
def _format_table(
    rows: List[List[Any]],
    headers: List[str],
    max_widths: Optional[List[Optional[int]]] = None,
) -> str:
    """将行数据格式化为对齐的文本表格。

    Parameters
    ----------
    rows:
        数据行列表，每行为单元格值列表。
    headers:
        表头列表。
    max_widths:
        每列最大宽度；``None`` 表示不限制。超出宽度的单元格会被截断
        并以 ``...`` 结尾。

    Returns
    -------
    str
        格式化后的表格文本。
    """
    if not rows:
        return ""

    processed: List[List[str]] = []
    for row in rows:
        cells: List[str] = []
        for i, cell in enumerate(row):
            text = str(cell)
            if (
                max_widths
                and i < len(max_widths)
                and max_widths[i]
                and len(text) > max_widths[i]
            ):
                text = text[: max_widths[i] - 3] + "..."
            cells.append(text)
        processed.append(cells)

    col_widths = [len(h) for h in headers]
    for row in processed:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    separator = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(
        h.ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    lines = [header_line, separator]
    for row in processed:
        line = "  ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 节点发现辅助
# ---------------------------------------------------------------------------
def _ensure_nodes_discovered() -> None:
    """确保内置节点和插件已加载。

    依次执行：
    1. ``registry.discover()`` 扫描内置节点包
    2. ``plugin_manager.load_plugins()`` 加载 entry_points / 目录插件
    3. ``plugin_manager.mark_builtin()`` 将内置节点标记到插件管理器
    """
    registry.discover()
    plugin_manager.load_plugins()
    plugin_manager.mark_builtin()


def _resolve_node_spec(name: str) -> Optional[NodeSpec]:
    """根据名称解析节点规格说明。

    先从已注册节点列表中按 ``node.name`` 匹配；若未命中，尝试按类名
    别名查找。

    Parameters
    ----------
    name:
        节点名称或类名。

    Returns
    -------
    Optional[NodeSpec]
        节点规格说明；未找到时返回 ``None``。
    """
    for spec in registry.list_nodes():
        if spec.name == name:
            return spec

    # 按类名别名回退查找
    try:
        node_class = registry.get_class(name)
    except KeyError:
        return None

    try:
        return node_class().describe()
    except Exception:
        return NodeSpec(
            name=node_class.name,
            domain=node_class.domain,
            description=node_class.description,
            version=node_class.version,
            input_types=list(node_class.input_types),
            output_types=list(node_class.output_types),
        )


# ---------------------------------------------------------------------------
# 管道文件加载辅助
# ---------------------------------------------------------------------------
def _parse_yaml(content: str) -> Any:
    """解析 YAML 内容，PyYAML 缺失时给出友好提示。

    Raises
    ------
    RuntimeError
        PyYAML 未安装时抛出，提示安装命令。
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "解析 YAML 文件需要 PyYAML 库。请安装: pip install pyyaml"
        ) from exc
    return yaml.safe_load(content)


def _load_pipeline_file(path: str) -> Dict[str, Any]:
    """加载管道定义文件（YAML 或 JSON）。

    按扩展名选择解析器：``.yaml``/``.yml`` 使用 PyYAML，``.json`` 使用
    标准 ``json`` 模块。无法识别的扩展名先尝试 JSON，失败后回退到 YAML。

    Parameters
    ----------
    path:
        文件路径。

    Returns
    -------
    Dict[str, Any]
        解析后的管道定义字典。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    RuntimeError
        YAML 文件需要 PyYAML 但未安装。
    ValueError
        文件内容不是字典，或解析失败。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"管道文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    ext = os.path.splitext(path)[1].lower()
    data: Any

    if ext in (".yaml", ".yml"):
        data = _parse_yaml(content)
    elif ext == ".json":
        data = json.loads(content)
    else:
        # 自动检测：先尝试 JSON，失败则尝试 YAML
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = _parse_yaml(content)

    if not isinstance(data, dict):
        raise ValueError(
            "管道定义文件必须是一个字典（包含 'nodes' 和 'input' 键）。"
        )
    return data


# ---------------------------------------------------------------------------
# 参数解析器构建
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    Returns
    -------
    argparse.ArgumentParser
        配置好的参数解析器实例，包含所有子命令。
    """
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description="Mosaic — 多模态生成式 AI 编排框架",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list 命令
    p_list = subparsers.add_parser(
        "list", help="列出所有已注册节点"
    )
    p_list.add_argument(
        "--domain",
        default=None,
        help="按域过滤（如 text / image / audio）",
    )
    p_list.add_argument(
        "--plugins",
        action="store_true",
        help="只显示插件（非内置节点）",
    )

    # info 命令
    p_info = subparsers.add_parser(
        "info", help="查看节点详细信息"
    )
    p_info.add_argument(
        "node_name",
        help="节点名称或类名",
    )

    # create-node 命令
    p_create = subparsers.add_parser(
        "create-node", help="生成节点模板代码（交互式或参数式）"
    )
    p_create.add_argument("--domain", default=None, help="节点所属域")
    p_create.add_argument("--name", default=None, help="节点名称")
    p_create.add_argument("--description", default=None, help="节点描述")
    p_create.add_argument("--output", default=None, help="输出目录")
    p_create.add_argument("--model", default=None, help="默认模型标识")
    p_create.add_argument("--author", default=None, help="作者名称")

    # run 命令
    p_run = subparsers.add_parser(
        "run", help="从 YAML/JSON 文件运行管道"
    )
    p_run.add_argument(
        "pipeline_file",
        help="管道定义文件路径（.yaml / .yml / .json）",
    )

    # version 命令
    subparsers.add_parser("version", help="显示 Mosaic 版本号")

    # doctor 命令
    subparsers.add_parser("doctor", help="环境诊断")

    return parser


# ---------------------------------------------------------------------------
# 子命令实现：list
# ---------------------------------------------------------------------------
def _cmd_list(args: argparse.Namespace) -> int:
    """``list`` 命令：列出已注册节点或插件。

    - ``--domain`` 按域过滤
    - ``--plugins`` 只显示非内置插件

    输出为对齐的表格（name, domain, version, description）。
    """
    _ensure_nodes_discovered()

    if args.plugins:
        # 只显示非内置插件
        plugins = [
            p for p in plugin_manager.list_plugins() if p.source != "builtin"
        ]
        if not plugins:
            print("未发现任何插件。")
            print("提示: 使用 @mosaic.node 装饰器或安装第三方插件包来扩展节点。")
            return 0

        rows: List[List[Any]] = [
            [p.name, p.domain, p.version, p.source, p.description]
            for p in plugins
        ]
        headers = ["Name", "Domain", "Version", "Source", "Description"]
        print(_format_table(rows, headers, max_widths=[None, None, None, None, 60]))
        print(f"\n共 {len(plugins)} 个插件。")
        return 0

    # 列出所有节点（可按域过滤）
    specs = registry.list_nodes(domain=args.domain)
    if not specs:
        if args.domain:
            print(f"域 '{args.domain}' 下未发现任何节点。")
            domains = registry.list_domains()
            if domains:
                print(f"可用域: {', '.join(domains)}")
        else:
            print("未发现任何节点。")
        return 0

    rows = [
        [s.name, s.domain, s.version, s.description]
        for s in specs
    ]
    headers = ["Name", "Domain", "Version", "Description"]
    print(_format_table(rows, headers, max_widths=[None, None, None, 60]))
    print(f"\n共 {len(specs)} 个节点。")
    if not args.domain:
        domains = registry.list_domains()
        if domains:
            print(f"可用域: {', '.join(domains)}")
    return 0


# ---------------------------------------------------------------------------
# 子命令实现：info
# ---------------------------------------------------------------------------
def _cmd_info(args: argparse.Namespace) -> int:
    """``info`` 命令：查看节点详细信息。

    输出：name, domain, version, description, input_types,
    output_types, model_info。
    """
    _ensure_nodes_discovered()

    name = args.node_name
    spec = _resolve_node_spec(name)
    if spec is None:
        print(f"错误: 未找到节点 '{name}'。")
        available = registry.list_names()
        if available:
            print(f"可用节点: {', '.join(available)}")
        else:
            print("当前没有已注册的节点。")
        return 1

    input_types = ", ".join(spec.input_types) if spec.input_types else "(无)"
    output_types = ", ".join(spec.output_types) if spec.output_types else "(无)"

    print(f"名称:      {spec.name}")
    print(f"域:        {spec.domain}")
    print(f"版本:      {spec.version}")
    print(f"描述:      {spec.description or '(无)'}")
    print(f"输入类型:  {input_types}")
    print(f"输出类型:  {output_types}")
    if spec.model_info:
        print("模型信息:")
        for key, value in spec.model_info.items():
            print(f"  {key}: {value}")
    else:
        print("模型信息:  (无)")
    return 0


# ---------------------------------------------------------------------------
# 子命令实现：create-node
# ---------------------------------------------------------------------------
def _cmd_create_node(args: argparse.Namespace) -> int:
    """``create-node`` 命令：生成节点模板代码。

    有参数时直接生成，无参数时进入交互模式。
    支持 --domain, --name, --description, --output, --model, --author。
    """
    try:
        from mosaic.cli.create_node import NodeGenerator
    except ImportError:
        print("错误: 节点模板生成器尚未安装。")
        print("提示: 请确保 mosaic.cli.create_node 模块可用。")
        return 1

    # 判断是否有足够的参数直接生成
    has_args = args.domain is not None or args.name is not None

    if has_args:
        if not args.name:
            print("错误: 使用 --name 指定节点名称。")
            return 1
        generator = NodeGenerator(
            domain=args.domain or "custom",
            name=args.name,
            description=args.description,
            output=args.output or "./my_nodes/",
            model=args.model,
            author=args.author,
        )
    else:
        # 交互模式
        try:
            domain = (
                input("域 (如 text/image/custom) [custom]: ").strip()
                or "custom"
            )
            name = input("节点名称 (如 sentiment_analyzer): ").strip()
            if not name:
                print("错误: 节点名称不能为空。")
                return 1
            description = input("描述 (可选): ").strip() or None
            output = (
                input("输出目录 [./my_nodes/]: ").strip()
                or "./my_nodes/"
            )
            model = input("模型 (可选): ").strip() or None
            author = input("作者 (可选): ").strip() or None
        except (EOFError, KeyboardInterrupt):
            print("\n操作已取消。")
            return 130

        generator = NodeGenerator(
            domain=domain,
            name=name,
            description=description,
            output=output,
            model=model,
            author=author,
        )

    try:
        result = generator.generate()
        print(f"节点模板已生成: {result}")
    except Exception as exc:
        print(f"错误: 生成节点模板失败: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# 子命令实现：run
# ---------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    """``run`` 命令：从 YAML/JSON 文件运行管道。

    管道定义格式::

        nodes:
          - name: text-generator       # 可选，节点别名
            type: TextGenerator         # 节点类名或注册名
            params:                     # 构造参数
              model: Qwen/Qwen2.5-7B-Instruct
        input:
          prompt: "你好"
    """
    # 加载管道定义文件
    try:
        data = _load_pipeline_file(args.pipeline_file)
    except FileNotFoundError as exc:
        print(f"错误: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"错误: {exc}")
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"错误: 管道文件解析失败: {exc}")
        return 1

    # 校验结构
    if "nodes" not in data:
        print("错误: 管道定义中缺少 'nodes' 字段。")
        print(
            "示例格式:\n"
            "  nodes:\n"
            "    - name: text-generator\n"
            "      type: TextGenerator\n"
            "      params:\n"
            "        model: Qwen/Qwen2.5-7B-Instruct\n"
            "  input:\n"
            '    prompt: "你好"'
        )
        return 1

    node_defs = data.get("nodes", [])
    input_data = data.get("input", {})

    if not isinstance(node_defs, list) or not node_defs:
        print("错误: 管道中未定义任何节点（'nodes' 应为非空列表）。")
        return 1

    if not isinstance(input_data, dict):
        print("错误: 'input' 字段必须是一个字典。")
        return 1

    # 发现节点
    _ensure_nodes_discovered()

    # 构建节点实例列表
    elements: List[Any] = []
    for i, node_def in enumerate(node_defs):
        if not isinstance(node_def, dict):
            print(f"错误: 第 {i + 1} 个节点定义必须是字典。")
            return 1
        node_type = node_def.get("type")
        if not node_type:
            print(f"错误: 第 {i + 1} 个节点缺少 'type' 字段。")
            return 1
        params = dict(node_def.get("params", {}))
        node_alias = node_def.get("name")

        try:
            node_class = registry.get_class(node_type)
            node = node_class(**params)
            if node_alias:
                node.name = node_alias
        except KeyError:
            print(f"错误: 未找到节点类型 '{node_type}'。")
            available = registry.list_names()
            if available:
                print(f"可用节点: {', '.join(available)}")
            return 1
        except Exception as exc:
            print(f"错误: 实例化节点 '{node_type}' 失败: {exc}")
            return 1
        elements.append(node)

    # 构建并执行管道
    pipe = Pipeline("cli-pipeline", elements)
    pipeline_input = MosaicData(**input_data)

    try:
        result = pipe.execute_result(pipeline_input)
    except Exception as exc:
        print(f"错误: 管道执行失败: {exc}")
        return 1

    # 输出结果
    print(f"管道执行完成，耗时 {result.duration:.3f}s")
    if result.errors:
        print(f"警告: {len(result.errors)} 个节点执行失败:")
        for err in result.errors:
            print(f"  - {err.node_name}: {err.error}")

    if result.output is not None:
        print("输出:")
        for key, value in result.output.items():
            if isinstance(value, (str, int, float, bool)):
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"  {key}: {val_str}")
            elif value is None:
                print(f"  {key}: None")
            else:
                print(f"  {key}: <{type(value).__name__}>")

    return 0 if not result.errors else 1


# ---------------------------------------------------------------------------
# 子命令实现：version
# ---------------------------------------------------------------------------
def _cmd_version(args: argparse.Namespace) -> int:
    """``version`` 命令：显示版本号。"""
    print(f"mosaic {__version__}")
    return 0


# ---------------------------------------------------------------------------
# 子命令实现：doctor
# ---------------------------------------------------------------------------
def _cmd_doctor(args: argparse.Namespace) -> int:
    """``doctor`` 命令：环境诊断。"""
    try:
        from mosaic.cli.doctor import run_doctor
    except ImportError:
        print("错误: 环境诊断模块尚未安装。")
        print("提示: 请确保 mosaic.cli.doctor 模块可用。")
        return 1

    try:
        run_doctor()
        return 0
    except Exception as exc:
        print(f"错误: 环境诊断失败: {exc}")
        return 1


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口函数。

    Parameters
    ----------
    argv:
        命令行参数列表。``None`` 时使用 ``sys.argv[1:]``。

    Returns
    -------
    int
        退出码：``0`` 表示成功，非 ``0`` 表示失败。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "list": _cmd_list,
        "info": _cmd_info,
        "create-node": _cmd_create_node,
        "run": _cmd_run,
        "version": _cmd_version,
        "doctor": _cmd_doctor,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    try:
        result = handler(args)
        return result if result is not None else 0
    except KeyboardInterrupt:
        print("\n操作已取消。")
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
