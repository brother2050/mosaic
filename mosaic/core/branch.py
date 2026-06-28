# mosaic/core/branch.py
"""分支与合并编排指令。

从 :mod:`mosaic.core.pipeline` 中拆分出来的独立模块，定义 :class:`Branch`
（fan-out / 条件分支）与 :class:`Merge`（fan-in 合并）。

设计要点
--------
* ``Branch`` 和 ``Merge`` 是 :class:`~mosaic.core.pipeline.Pipeline` 的编排指令，
  本身不是普通节点（``Branch`` 不是 ``Node``；``Merge`` 继承 ``Node`` 以便在
  DAG 中占位）。
* :class:`Branch` 增强：
    * 支持 ``input_strategy`` —— 控制分支输入是复制还是按名称分配。
    * 支持 ``fail_fast`` —— 某分支失败时是否立即中断其他分支。
* :class:`Merge` 增强：
    * 支持 ``keep`` —— 选择性保留某条分支的结果。
    * 支持 ``merge_fn`` —— 自定义合并函数。
    * 保留原有 ``dict`` / ``flatten`` 策略。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from mosaic.core.node import Node, NodeSpec
from mosaic.core.types import MosaicData

__all__ = ["Branch", "Merge", "PathLike"]

#: 路径类型：单个节点/管道，或由它们组成的线性列表。
PathLike = Union[Node, "Pipeline", List[Union[Node, "Pipeline"]]]  # noqa: F821


class Branch:
    """分支编排指令：fan-out 或条件分支。

    用作 :class:`~mosaic.core.pipeline.Pipeline` 元素列表中的一项，**本身不是节点**。

    * **fan-out**（不传 ``condition``）：输入同时喂给所有路径，各路径
      **并行**执行，输出在 :class:`Merge` 处汇合或在管道末端以命名键聚合。
    * **条件分支**（传 ``condition``）：运行时对输入数据求值 ``condition``，
      返回的键决定执行哪一条路径（其余路径跳过）。

    Parameters
    ----------
    paths:
        可选的 ``{路径名: 路径}`` 字典。路径可为单个 ``Node``/``Pipeline``
        或由它们组成的列表（线性子链）。
    condition:
        条件回调，签名 ``(MosaicData) -> str``，返回值须为某条路径名。
        为 ``None`` 时表示 fan-out。
    input_strategy:
        分支输入策略（仅 fan-out 模式生效）：

        * ``"copy"``（默认）：将同一份输入复制给所有分支。
        * ``"distribute"``：按分支名从输入中提取对应字段作为该分支输入。
          例如输入 ``{"en": ..., "ja": ...}``，分支名 ``"en"`` 的路径
          收到 ``input["en"]``。
    fail_fast:
        某分支失败时是否立即中断其他分支。``True``（默认）立即抛出异常；
        ``False`` 等待所有分支完成后统一报告。此值可被 ``Pipeline.execute()``
        的 ``fail_fast`` 参数覆盖。
    **named:
        以关键字形式指定路径，如 ``Branch(en=Translator(), ja=Translator())``。

    Examples
    --------
    fan-out（并行）::

        Pipeline("demo", [
            TextGenerator(),
            Branch(en=Translator(target="en"), ja=Translator(target="ja")),
            Merge(),
        ])

    条件分支::

        Pipeline("cond", [
            TextGenerator(),
            Branch(
                upper=UpperNode(),
                lower=LowerNode(),
                condition=lambda d: "upper" if d.get("flag") else "lower",
            ),
        ])
    """

    def __init__(
        self,
        paths: Optional[Dict[str, PathLike]] = None,
        *,
        condition: Optional[Callable[[MosaicData], str]] = None,
        input_strategy: str = "copy",
        fail_fast: bool = True,
        **named: PathLike,
    ) -> None:
        merged: Dict[str, PathLike] = {}
        if paths:
            merged.update(paths)
        merged.update(named)
        if not merged:
            raise ValueError("Branch requires at least one named path.")

        if input_strategy not in ("copy", "distribute"):
            raise ValueError(
                f"Invalid input_strategy {input_strategy!r}, "
                f"expected 'copy' or 'distribute'."
            )

        self.paths: Dict[str, PathLike] = merged
        self.condition: Optional[Callable[[MosaicData], str]] = condition
        self.input_strategy: str = input_strategy
        self.fail_fast: bool = fail_fast

    @property
    def is_conditional(self) -> bool:
        """是否为条件分支。"""
        return self.condition is not None

    @property
    def is_fanout(self) -> bool:
        """是否为 fan-out 分支（非条件分支）。"""
        return self.condition is None

    def __or__(self, other: Any) -> "Pipeline":  # noqa: F821
        """支持 ``Branch | node`` 语法。"""
        # 延迟导入避免循环引用
        from mosaic.core.pipeline import Pipeline

        if isinstance(other, (Node, Branch)):
            return Pipeline("anonymous", [self, other])
        if isinstance(other, Pipeline):
            return Pipeline("anonymous", [self, *other.elements])
        return NotImplemented

    def __repr__(self) -> str:
        kind = "conditional" if self.is_conditional else "fan-out"
        return (
            f"Branch({kind}, paths={list(self.paths.keys())}, "
            f"input_strategy={self.input_strategy!r})"
        )


class Merge(Node):
    """合并节点：将多条上游输出合并为单个 :class:`MosaicData`。

    在 DAG 中拥有多个前驱。运行时引擎会把各前驱输出按其路径名/节点名
    组装成一个 ``MosaicData``（键为标签，值为对应输出），再交给本节点。

    Parameters
    ----------
    strategy:
        合并策略：

        * ``"dict"``（默认）：原样返回组装好的 ``MosaicData``，
          下游可通过 ``result["路径名"]`` 访问各分支输出。
        * ``"flatten"``：将各分支 ``MosaicData`` 的键值平铺合并到一个
          ``MosaicData``（后出现的键覆盖先前的）。
    merge_fn:
        自定义合并函数，签名 ``(MosaicData) -> MosaicData``。提供时优先于
        ``strategy`` 生效。
    keep:
        选择性合并：只保留指定分支名的结果，忽略其他分支。
        例如 ``Merge(keep="path_a")`` 只返回 ``path_a`` 分支的输出。
        为 ``None`` 时合并所有分支。

    Note
    ----
    若 ``Merge`` 只有一个前驱（例如位于条件分支之后），其行为退化为
    透传：``"dict"`` 原样返回，``"flatten"`` 平铺该单一输入的键。
    """

    name = "merge"
    domain = "core"
    description = "Fan-in: merge multiple upstream outputs into one MosaicData."
    version = "0.1.0"
    input_types: List[str] = ["mosaic"]
    output_types: List[str] = ["mosaic"]

    def __init__(
        self,
        strategy: str = "dict",
        merge_fn: Optional[Callable[[MosaicData], MosaicData]] = None,
        keep: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if strategy not in {"dict", "flatten"}:
            raise ValueError(
                f"Invalid merge strategy {strategy!r}, expected 'dict' or 'flatten'."
            )
        self._strategy = strategy
        self._merge_fn = merge_fn
        self._keep = keep

    # -- Node 接口 ---------------------------------------------------------
    def load(self) -> None:
        """合并节点无需加载模型。"""
        self._loaded = True

    def unload(self) -> None:
        """释放资源。"""
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行合并。

        ``input_data`` 已由引擎组装为 ``{标签: 分支输出}`` 形式。
        """
        # 选择性合并：只保留指定分支
        if self._keep is not None:
            if self._keep in input_data:
                kept = input_data[self._keep]
                if isinstance(kept, MosaicData):
                    return kept
                return MosaicData(**{self._keep: kept})
            # keep 指定的分支不存在，返回空
            return MosaicData()

        # 自定义合并函数
        if self._merge_fn is not None:
            return self._merge_fn(input_data)

        # flatten 策略
        if self._strategy == "flatten":
            merged = MosaicData()
            for _label, value in input_data.items():
                if isinstance(value, MosaicData):
                    for inner_key, inner_val in value.items():
                        merged[inner_key] = inner_val
                else:
                    # 非数据容器值直接保留在原标签下
                    pass
            return merged

        # "dict": 原样返回
        return input_data

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        model_info: Dict[str, Any] = {"strategy": self._strategy}
        if self._keep is not None:
            model_info["keep"] = self._keep
        if self._merge_fn is not None:
            model_info["custom_fn"] = True
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=model_info,
        )

    def __repr__(self) -> str:
        parts = [f"strategy={self._strategy!r}"]
        if self._keep is not None:
            parts.append(f"keep={self._keep!r}")
        return f"<Merge({', '.join(parts)})>"
