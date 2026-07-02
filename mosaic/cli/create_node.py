# mosaic/cli/create_node.py
"""节点模板生成器。

根据参数自动生成 Mosaic 节点骨架代码，包括节点文件、包初始化文件、
测试骨架与说明文档。

使用 ``string.Template`` 的 ``$var`` 语法进行模板替换（不依赖 Jinja2）。

使用示例
--------
编程式调用::

    from mosaic.cli.create_node import NodeGenerator

    gen = NodeGenerator()
    paths = gen.generate(
        domain="text",
        node_name="SentimentAnalyzer",
        description="情感分析节点",
        input_types=("text",),
        output_types=("text",),
        model_name="uer/roberta-base-finetuned-jd-binary-chinese",
        author="Alice",
        output_dir="./my_nodes",
    )

交互式调用::

    gen = NodeGenerator()
    gen.interactive()
"""

from __future__ import annotations

import re
import string
from pathlib import Path
__all__ = ["NodeGenerator", "to_snake_case"]


# ---------------------------------------------------------------------------
# 命名转换工具
# ---------------------------------------------------------------------------
def to_snake_case(name: str) -> str:
    """将 CamelCase / PascalCase 名称转换为 snake_case。

    支持以下输入形式：

    - ``SentimentAnalyzer`` -> ``sentiment_analyzer``
    - ``HTMLParser``        -> ``html_parser``
    - ``sentiment-analyzer`` -> ``sentiment_analyzer``
    - ``My Cool Node``      -> ``my_cool_node``

    Parameters
    ----------
    name:
        待转换的名称。

    Returns
    -------
    str
        转换后的 snake_case 字符串。
    """
    if not name:
        return ""
    # 统一分隔符：连字符与空格转为下划线
    normalized = name.strip().replace("-", "_").replace(" ", "_")
    # 在 "小写/数字 + 大写" 边界插入下划线: Analyzer -> _Analyzer
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", normalized)
    # 在 "小写/数字 + 大写" 边界插入下划线: getHTTP -> get_HTTP
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # 折叠连续下划线并转小写
    collapsed = re.sub(r"_+", "_", s2)
    return collapsed.strip("_").lower()


# ---------------------------------------------------------------------------
# 节点代码生成器
# ---------------------------------------------------------------------------
class NodeGenerator:
    """节点代码生成器。

    读取 ``mosaic/templates/`` 目录下的模板文件，使用 ``string.Template``
    替换变量后写入指定输出目录，生成一套完整的节点骨架。

    Attributes
    ----------
    template_dir:
        模板文件所在目录（``mosaic/templates/``）。
    """

    #: 模板文件名 -> 生成文件名的映射
    _FILE_MAPPING: dict[str, str] = {
        "node.py.tpl": "{node_name_snake}.py",
        "init.py.tpl": "__init__.py",
        "test.py.tpl": "test_{node_name_snake}.py",
        "readme.md.tpl": "README.md",
    }

    def __init__(self) -> None:
        """初始化生成器，定位模板目录。"""
        # mosaic/cli/create_node.py -> mosaic/templates/
        self.template_dir: Path = (
            Path(__file__).resolve().parent.parent / "templates"
        )

    # -- 核心生成逻辑 ------------------------------------------------------
    def generate(
        self,
        domain: str,
        node_name: str,
        description: str = "",
        input_types: list[str] | None = None,
        output_types: list[str] | None = None,
        model_name: str = "",
        author: str = "",
        license: str = "Apache-2.0",
        output_dir: str = "./",
    ) -> list[str]:
        """生成节点文件，返回生成的文件路径列表。

        Parameters
        ----------
        domain:
            节点所属域，如 ``"text"`` / ``"image"`` / ``"custom"``。
        node_name:
            节点类名（CamelCase），如 ``"SentimentAnalyzer"``。
        description:
            节点功能描述。
        input_types:
            输入数据类型标识列表；``None`` 时默认 ``["text"]``。
        output_types:
            输出数据类型标识列表；``None`` 时默认 ``["text"]``。
        model_name:
            模型标识（HuggingFace repo id 等）。
        author:
            作者名称。
        license:
            许可证名称，默认 ``"Apache-2.0"``。
        output_dir:
            输出目录路径，默认当前目录。

        Returns
        -------
        list[str]
            生成的文件绝对路径列表。

        Raises
        ------
        FileNotFoundError
            模板文件不存在时抛出。
        ValueError
            ``node_name`` 为空时抛出。
        """
        if not node_name or not node_name.strip():
            raise ValueError("node_name 不能为空。")

        # 蛇形命名转换
        node_name = node_name.strip()
        node_name_snake = to_snake_case(node_name)

        # 默认类型
        if input_types is None:
            input_types = ("text",)
        if output_types is None:
            output_types = ("text",)

        # 构造模板变量（input_types / output_types 转为 Python 字面量）
        variables: dict[str, str] = {
            "domain": domain,
            "node_name": node_name,
            "node_name_snake": node_name_snake,
            "description": description,
            "input_types": repr(list(input_types)),
            "output_types": repr(list(output_types)),
            "model_name": model_name,
            "author": author,
            "license": license,
        }

        # 准备输出目录
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        generated: list[str] = []
        for tpl_name, out_name_tpl in self._FILE_MAPPING.items():
            out_name = out_name_tpl.format(node_name_snake=node_name_snake)
            content = self._render_template(tpl_name, variables)
            target = out_path / out_name
            target.write_text(content, encoding="utf-8")
            generated.append(str(target))

        return generated

    # -- 模板渲染 ----------------------------------------------------------
    def _render_template(
        self, template_name: str, variables: dict[str, str]
    ) -> str:
        """读取模板文件并使用 ``string.Template`` 替换变量。

        Parameters
        ----------
        template_name:
            模板文件名（位于 :attr:`template_dir` 下）。
        variables:
            变量名到值的映射。

        Returns
        -------
        str
            渲染后的文本内容。

        Raises
        ------
        FileNotFoundError
            模板文件不存在。
        """
        tpl_path = self.template_dir / template_name
        if not tpl_path.is_file():
            raise FileNotFoundError(
                f"模板文件不存在: {tpl_path}"
            )
        raw = tpl_path.read_text(encoding="utf-8")
        # safe_substitute: 缺失变量保留原样，不抛 KeyError
        return string.Template(raw).safe_substitute(variables)

    # -- 交互式模式 --------------------------------------------------------
    def interactive(self) -> list[str]:
        """交互式模式：通过 ``input()`` 收集参数并生成节点。

        依次提示用户输入域、节点类名、描述、输入/输出类型、模型名称、
        作者与输出目录，然后调用 :meth:`generate`。

        Returns
        -------
        list[str]
            生成的文件路径列表；用户取消时返回空列表。
        """
        print("=" * 50)
        print("Mosaic 节点生成器（交互模式）")
        print("=" * 50)
        print()

        # 域
        domain = self._prompt("域 (domain) [custom]", default="custom")

        # 节点类名（必填）
        node_name = input("节点类名 (CamelCase, 如 SentimentAnalyzer): ").strip()
        if not node_name:
            print("错误：节点类名不能为空，已取消生成。")
            return []

        # 描述
        description = input("描述 (description): ").strip()

        # 输入类型
        input_raw = input("输入类型 (逗号分隔) [text]: ").strip()
        input_types = self._parse_list(input_raw, default=["text"])

        # 输出类型
        output_raw = input("输出类型 (逗号分隔) [text]: ").strip()
        output_types = self._parse_list(output_raw, default=["text"])

        # 模型名称
        model_name = input("模型名称 (model_name) [留空跳过]: ").strip()

        # 作者
        author = input("作者 (author) [留空跳过]: ").strip()

        # 输出目录
        output_dir = self._prompt("输出目录 [./]", default="./")

        print()
        try:
            paths = self.generate(
                domain=domain,
                node_name=node_name,
                description=description,
                input_types=input_types,
                output_types=output_types,
                model_name=model_name,
                author=author,
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"生成失败: {exc}")
            return []

        print("已生成以下文件：")
        for p in paths:
            print(f"  - {p}")
        return paths

    # -- 交互辅助 ----------------------------------------------------------
    @staticmethod
    def _prompt(message: str, default: str) -> str:
        """提示用户输入，返回输入值或默认值。"""
        raw = input(f"{message}: ").strip()
        return raw if raw else default

    @staticmethod
    def _parse_list(raw: str, default: list[str]) -> list[str]:
        """将逗号分隔的字符串解析为列表。"""
        if not raw:
            return list(default)
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return items if items else list(default)


# ---------------------------------------------------------------------------
# 直接运行入口（便于 ``python -m mosaic.cli.create_node`` 调用）
# ---------------------------------------------------------------------------
def main() -> int:
    """命令行入口：启动交互式节点生成器。

    Returns
    -------
    int
        进程退出码，``0`` 表示成功。
    """
    try:
        gen = NodeGenerator()
        paths = gen.interactive()
        return 0 if paths else 1
    except KeyboardInterrupt:
        print("\n已取消。")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
