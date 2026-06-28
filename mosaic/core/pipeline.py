# mosaic/core/pipeline.py
"""Mosaic 管道编排引擎。

本模块实现了框架最核心的用户 API —— :class:`Pipeline`，支持串行与并行执行、
分支 (:class:`Branch`) 与合并 (:class:`Merge`) 编排，以及完整的中间产物检查。

设计要点
--------
* ``Pipeline`` 本身继承 :class:`~mosaic.core.node.Node`，因此管道可以
  嵌套（管道也是一种节点），并支持 ``load``/``unload``/``run``/``describe``。
* 内部使用有向无环图（DAG）表示节点拓扑，由用户提供的元素列表
  （``Node`` / ``Branch`` / ``Merge``）编译而来。
* **并行执行**：DAG 中互相独立的节点（如 Branch 的多条路径）使用
  ``concurrent.futures.ThreadPoolExecutor`` 并行执行。
* 运行前执行 DAG 合法性检查（环检测、连通性、死端节点）。
* ``dry_run`` 模式只校验节点输入/输出类型是否匹配，不实际执行。
* 支持 ``|`` 运算符声明式串联：``node_a | node_b | node_c``。
* ``execute_result()`` 返回 :class:`PipelineResult`，含完整运行信息；
  ``execute()`` 保持返回 ``MosaicData`` 以兼容旧代码。

拓扑语义
--------
* **串行**：``[A, B, C]`` → A→B→C，前者输出作为后者输入。
* **fan-out** (:class:`Branch`)：一个输入同时喂给多条并行路径。
* **fan-in** (:class:`Merge`)：多条上游输出合并为一个 ``MosaicData``。
* **条件分支** (:class:`Branch` 带条件)：运行时按数据选择唯一路径执行。
"""

from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from mosaic.core.branch import Branch, Merge, PathLike
from mosaic.core.context import Context, Event, EventHandler, RunConfig
from mosaic.core.node import Node, NodeSpec
from mosaic.core.result import NodeError, PipelineResult
from mosaic.core.types import MosaicData

__all__ = [
    "Pipeline",
    "Branch",
    "Merge",
    "PipelineError",
    "DryRunResult",
    "PipelineResult",
    "NodeError",
]


# ---------------------------------------------------------------------------
# 异常与结果类型
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """管道结构或运行时错误。"""


@dataclass
class DryRunResult:
    """``dry_run`` 模式的校验结果。

    Attributes
    ----------
    ok:
        是否通过全部校验（无结构错误且无类型不匹配）。
    issues:
        发现的问题列表（结构错误或类型不匹配说明）。
    steps:
        按拓扑序排列的各节点规格说明。
    """

    ok: bool
    issues: List[str] = field(default_factory=list)
    steps: List[NodeSpec] = field(default_factory=list)

    def __bool__(self) -> bool:
        """``bool(result)`` 等价于 ``result.ok``。"""
        return self.ok

    def __repr__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return f"DryRunResult({status}, issues={len(self.issues)}, steps={len(self.steps)})"


# ---------------------------------------------------------------------------
# _ConditionalNode — 条件分支路由（内部节点）
# ---------------------------------------------------------------------------
class _ConditionalNode(Node):
    """条件分支的内部路由节点。

    持有各路径对应的子 :class:`Pipeline`，运行时按 ``condition`` 选择
    一条路径执行并返回其输出。在父 DAG 中表现为单个节点（单前驱、单后继）。
    """

    name = "conditional"
    domain = "core"
    description = "Conditional branch router: select one path at runtime."
    version = "0.1.0"
    input_types: List[str] = ["mosaic"]
    output_types: List[str] = ["mosaic"]

    def __init__(
        self,
        paths: Dict[str, "Pipeline"],
        condition: Callable[[MosaicData], str],
    ) -> None:
        super().__init__()
        self._paths: Dict[str, "Pipeline"] = paths
        self._condition: Callable[[MosaicData], str] = condition

    def load(self) -> None:
        """加载所有候选路径的子管道。"""
        for sub in self._paths.values():
            sub.load()
        self._loaded = True

    def unload(self) -> None:
        """卸载所有候选路径的子管道。"""
        for sub in self._paths.values():
            sub.unload()
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        """按条件选择路径并执行。"""
        key = self._condition(input_data)
        if key not in self._paths:
            raise PipelineError(
                f"Conditional branch returned {key!r}, "
                f"expected one of {sorted(self._paths.keys())}."
            )
        return self._paths[key].run(input_data)

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info={
                "paths": sorted(self._paths.keys()),
                "conditional": True,
            },
        )


# ---------------------------------------------------------------------------
# DAG 内部表示
# ---------------------------------------------------------------------------
@dataclass
class _DAGNode:
    """DAG 中的一个执行单元。

    Attributes
    ----------
    node_id:
        管道内唯一标识（重名节点自动加 ``#n`` 后缀）。
    node:
        被包装的 ``Node``/``Pipeline``/内部节点实例。
    predecessors:
        前驱节点 id 列表。
    input_labels:
        与 ``predecessors`` 一一对应的标签（路径名或节点名），用于
        多前驱时组装合并输入。
    successors:
        后继节点 id 列表。
    branch_name:
        所属分支名（如果是 Branch fan-out 路径中的节点），``None`` 表示
        不属于任何分支。用于错误报告。
    """

    node_id: str
    node: Node
    predecessors: List[str] = field(default_factory=list)
    input_labels: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    branch_name: Optional[str] = None


def _normalize_path(path: PathLike) -> List[Any]:
    """将单节点或列表统一为元素列表。"""
    if isinstance(path, (list, tuple)):
        return list(path)
    return [path]


# ---------------------------------------------------------------------------
# Pipeline — 管道编排引擎
# ---------------------------------------------------------------------------
class Pipeline(Node):
    """节点管道编排引擎。

    继承 :class:`~mosaic.core.node.Node`，因此管道本身也是一种节点，可被
    嵌套进更大的管道。

    Parameters
    ----------
    name:
        管道名称。
    elements:
        有序的编排元素列表，每项为 ``Node``/``Pipeline``/``Branch``/``Merge``。
    description:
        管道描述。

    Examples
    --------
    基本用法::

        pipe = Pipeline("my-pipe", [
            TextGenerator(model="qwen2.5"),
            TextToImage(model="sdxl"),
            VideoEncoder(format="mp4"),
        ])
        result = pipe.run(input_data)

    并行分支 + 合并::

        pipe = Pipeline("parallel", [
            ImageLoader(),
            Branch(
                bg=BackgroundRemover(),
                style=Stylizer(),
            ),
            Merge(),
        ])
        result = pipe.execute_result(input_data)
        # result.intermediate["bg"] → BackgroundRemover 输出
        # result.intermediate["style"] → Stylizer 输出
    """

    name = "pipeline"
    domain = "pipeline"
    description = "A composable pipeline of nodes."
    version = "0.1.0"
    input_types: List[str] = []
    output_types: List[str] = []

    def __init__(
        self,
        name: str = "pipeline",
        elements: Optional[List[Any]] = None,
        description: str = "A composable pipeline of nodes.",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, **kwargs)
        self._elements: List[Any] = list(elements) if elements else []
        self._dag: Optional[Dict[str, _DAGNode]] = None
        self._terminals: List[Tuple[str, str]] = []  # (label, node_id)
        self._sources: List[str] = []
        self._id_counter: Dict[str, int] = {}
        self._last_context: Optional[Context] = None
        self._last_result: Optional[PipelineResult] = None

    # -- 元素管理 ----------------------------------------------------------
    @property
    def elements(self) -> List[Any]:
        """返回编排元素列表的副本（用于 ``|`` 展开等）。"""
        return list(self._elements)

    def add(self, element: Any) -> "Pipeline":
        """追加一个编排元素，返回 ``self`` 以支持链式调用。

        追加后已编译的 DAG 会失效，下次访问时重新编译。
        """
        self._elements.append(element)
        self._dag = None
        return self

    def __len__(self) -> int:
        """返回编排元素数量。"""
        return len(self._elements)

    # -- Node 接口实现 -----------------------------------------------------
    def load(self) -> None:
        """加载 DAG 中所有节点（含子管道）的模型。"""
        self._build_dag_if_needed()
        for dn in self._dag.values():
            if not dn.node.is_loaded():
                dn.node.load()
        self._loaded = True

    def unload(self) -> None:
        """卸载 DAG 中所有节点的模型。"""
        if self._dag is None:
            return
        for dn in self._dag.values():
            if dn.node.is_loaded():
                dn.node.unload()
        self._loaded = False

    def run(
        self,
        input_data: MosaicData,
        *,
        config: Optional[RunConfig] = None,
        callbacks: Optional[List[EventHandler]] = None,
        context: Optional[Context] = None,
    ) -> MosaicData:
        """执行管道。

        等价于 :meth:`execute`，便于以 ``Node`` 接口调用（嵌套场景）。
        """
        return self.execute(input_data, config=config, callbacks=callbacks, context=context)

    def describe(self) -> NodeSpec:
        """返回管道的聚合规格说明。"""
        self._build_dag_if_needed()
        input_types: List[str] = []
        for sid in self._sources:
            input_types.extend(self._dag[sid].node.input_types)
        output_types: List[str] = []
        for _label, tid in self._terminals:
            output_types.extend(self._dag[tid].node.output_types)
        domains = sorted({dn.node.domain for dn in self._dag.values()})
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(dict.fromkeys(input_types)),
            output_types=list(dict.fromkeys(output_types)),
            model_info={
                "node_count": len(self._dag),
                "domains": domains,
            },
        )

    # -- 管道运算符 --------------------------------------------------------
    def __or__(self, other: Any) -> "Pipeline":
        """支持 ``pipeline | node`` / ``pipeline | pipeline`` 语法（扁平展开）。"""
        if isinstance(other, Pipeline):
            return Pipeline("anonymous", [*self._elements, *other._elements])
        if isinstance(other, (Node, Branch)):
            return Pipeline("anonymous", [*self._elements, other])
        return NotImplemented

    def __ror__(self, other: Any) -> "Pipeline":
        """支持左操作数为非 Node 类型（如 ``Branch``）时的 ``|``。"""
        if isinstance(other, Branch):
            return Pipeline("anonymous", [other, *self._elements])
        return NotImplemented

    # -- DAG 编译 ----------------------------------------------------------
    def _build_dag_if_needed(self) -> None:
        """若 DAG 未编译或已失效，则重新编译。"""
        if self._dag is None:
            self._build_dag()

    def _build_dag(self) -> None:
        """将元素列表编译为 DAG。"""
        self._dag = {}
        self._id_counter = {}
        self._terminals = []
        frontier: List[Tuple[str, str]] = []  # (label, node_id)
        for element in self._elements:
            frontier = self._add_element(element, frontier)
        self._terminals = frontier
        self._sources = [nid for nid, dn in self._dag.items() if not dn.predecessors]

    def _add_node(self, node: Node, branch_name: Optional[str] = None) -> str:
        """注册一个节点，返回唯一 id（重名加 ``#n`` 后缀）。"""
        base = getattr(node, "name", "node")
        n = self._id_counter.get(base, 0)
        self._id_counter[base] = n + 1
        nid = base if n == 0 else f"{base}#{n}"
        self._dag[nid] = _DAGNode(node_id=nid, node=node, branch_name=branch_name)
        return nid

    def _connect(self, pred_id: str, succ_id: str, label: str) -> None:
        """添加一条带标签的边 ``pred -> succ``。"""
        self._dag[succ_id].predecessors.append(pred_id)
        self._dag[succ_id].input_labels.append(label)
        self._dag[pred_id].successors.append(succ_id)

    def _add_element(
        self,
        element: Any,
        frontier: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """将一个编排元素加入 DAG，返回新的 frontier。"""
        if isinstance(element, Branch):
            return self._add_branch(element, frontier)
        if isinstance(element, Node):  # 含 Pipeline 与 Merge
            nid = self._add_node(element)
            for label, pid in frontier:
                self._connect(pid, nid, label)
            return [(element.name, nid)]
        raise TypeError(
            f"Unsupported pipeline element: {type(element).__name__}. "
            f"Expected Node, Pipeline, Branch or Merge."
        )

    def _add_branch(
        self,
        branch: Branch,
        frontier: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """处理 Branch 元素，返回新的 frontier。"""
        if branch.is_conditional:
            # 条件分支：每条路径编译为子管道，封装为单个路由节点
            sub_pipes: Dict[str, Pipeline] = {}
            for pname, path in branch.paths.items():
                sub_pipes[pname] = Pipeline(
                    f"{self.name}::{pname}", _normalize_path(path)
                )
            cnode = _ConditionalNode(sub_pipes, branch.condition)  # type: ignore[arg-type]
            nid = self._add_node(cnode)
            for label, pid in frontier:
                self._connect(pid, nid, label)
            return [(cnode.name, nid)]

        # fan-out：每条路径作为扁平子链加入父 DAG，标记 branch_name
        new_frontier: List[Tuple[str, str]] = []
        for pname, path in branch.paths.items():
            chain_ids = self._add_chain(path, frontier, branch_name=pname)
            new_frontier.append((pname, chain_ids[-1]))
        return new_frontier

    def _add_chain(
        self,
        path: PathLike,
        source_frontier: List[Tuple[str, str]],
        branch_name: Optional[str] = None,
    ) -> List[str]:
        """将一条路径（单节点或线性子链）加入 DAG，返回节点 id 列表。"""
        items = _normalize_path(path)
        ids: List[str] = []
        prev_id: Optional[str] = None
        for i, item in enumerate(items):
            if not isinstance(item, Node):
                raise TypeError(
                    f"Branch path item must be a Node/Pipeline, got {type(item).__name__}."
                )
            nid = self._add_node(item, branch_name=branch_name)
            ids.append(nid)
            if i == 0:
                for label, pid in source_frontier:
                    self._connect(pid, nid, label)
            else:
                # 子链内部连接：标签取前驱节点名
                assert prev_id is not None
                prev_name = self._dag[prev_id].node.name
                self._connect(prev_id, nid, prev_name)
            prev_id = nid
        return ids

    # -- DAG 校验 ----------------------------------------------------------
    def validate(self) -> None:
        """校验 DAG 结构合法性。

        Raises
        ------
        PipelineError
            存在环、不可达节点或死端节点时抛出。
        """
        self._build_dag_if_needed()
        if not self._dag:
            return  # 空管道合法

        # 环检测（拓扑排序长度不足即有环）
        order = self._topological_order()
        if len(order) != len(self._dag):
            raise PipelineError("Cycle detected in pipeline DAG.")

        # 连通性：所有节点须从某个源点可达
        if not self._sources:
            raise PipelineError("No source node found (every node has a predecessor).")
        reachable = self._reachable_from(self._sources)
        unreachable = set(self._dag) - reachable
        if unreachable:
            raise PipelineError(
                f"Unreachable nodes from input: {sorted(unreachable)}."
            )

        # 死端检测：所有节点须能到达某个终点
        terminal_ids = {tid for _label, tid in self._terminals}
        if not terminal_ids:
            raise PipelineError("No terminal node found.")
        rev_reachable = self._reachable_to(terminal_ids)
        dead = set(self._dag) - rev_reachable
        if dead:
            raise PipelineError(
                f"Dead-end nodes (cannot reach output): {sorted(dead)}."
            )

    def dry_run(self) -> DryRunResult:
        """干跑模式：只校验结构合法性与节点输入/输出类型匹配，不实际执行。

        类型匹配规则：若前驱 ``output_types`` 与后继 ``input_types`` 无交集，
        且二者均非空，则报告不匹配。空类型列表视为"接受任意类型"。
        """
        self._build_dag_if_needed()
        issues: List[str] = []

        # 结构校验
        try:
            self.validate()
        except PipelineError as exc:
            issues.append(f"Structure: {exc}")
            return DryRunResult(ok=False, issues=issues, steps=[])

        order = self._topological_order()
        steps: List[NodeSpec] = []
        for nid in order:
            dn = self._dag[nid]
            try:
                steps.append(dn.node.describe())
            except Exception as exc:  # pragma: no cover - 防御性
                issues.append(f"{nid}: describe() failed: {exc}")

        # 类型匹配校验
        for nid in order:
            dn = self._dag[nid]
            if not dn.predecessors:
                continue
            succ_in = set(dn.node.input_types)
            for pid in dn.predecessors:
                pred_out = set(self._dag[pid].node.output_types)
                if not succ_in or not pred_out:
                    continue  # 任一方为空视为宽松兼容
                if succ_in.isdisjoint(pred_out):
                    issues.append(
                        f"Type mismatch: '{self._dag[pid].node.name}' outputs "
                        f"{sorted(pred_out)} but '{dn.node.name}' expects "
                        f"{sorted(succ_in)}."
                    )

        return DryRunResult(ok=not issues, issues=issues, steps=steps)

    # -- 拓扑与可达性 ------------------------------------------------------
    def _topological_order(self) -> List[str]:
        """Kahn 拓扑排序，返回节点 id 列表（有环时长度小于节点总数）。"""
        indeg = {nid: len(dn.predecessors) for nid, dn in self._dag.items()}
        queue: deque = deque([nid for nid, d in indeg.items() if d == 0])
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._dag[nid].successors:
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    queue.append(succ)
        return order

    def _reachable_from(self, sources: List[str]) -> set:
        """从给定源点出发前向可达的节点集合。"""
        seen: set = set()
        queue: deque = deque(sources)
        while queue:
            nid = queue.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            for succ in self._dag[nid].successors:
                if succ not in seen:
                    queue.append(succ)
        return seen

    def _reachable_to(self, terminals: set) -> set:
        """从给定终点出发反向可达的节点集合。"""
        seen: set = set()
        queue: deque = deque(terminals)
        while queue:
            nid = queue.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            for pred in self._dag[nid].predecessors:
                if pred not in seen:
                    queue.append(pred)
        return seen

    # -- 执行（增强版：支持并行） -------------------------------------------
    def execute(
        self,
        input_data: MosaicData,
        *,
        config: Optional[RunConfig] = None,
        callbacks: Optional[List[EventHandler]] = None,
        context: Optional[Context] = None,
        fail_fast: bool = True,
        max_workers: int = 4,
    ) -> MosaicData:
        """执行管道，返回最终输出 :class:`MosaicData`。

        向后兼容的执行入口。如需获取完整运行信息（中间产物、错误列表、
        耗时统计），请使用 :meth:`execute_result`。

        Parameters
        ----------
        input_data:
            管道输入数据。
        config:
            运行配置（设备、精度、批大小等）。``None`` 使用默认配置。
        callbacks:
            事件回调列表，每个回调会在节点开始/结束及管道级事件时被触发。
        context:
            外部传入的运行上下文。``None`` 时创建新上下文。
        fail_fast:
            某节点失败时是否立即抛出异常。``True``（默认）立即抛出；
            ``False`` 收集所有错误后返回（最终输出可能为 ``None``）。
        max_workers:
            并行执行的最大线程数。Branch 的多条路径会并行执行。

        Returns
        -------
        MosaicData
            管道最终输出。若存在多个终点，则返回以各终点标签为键、对应
            输出为值的 ``MosaicData``；单终点时直接返回该输出。
            ``fail_fast=False`` 且有节点失败时，最终输出可能为 ``None``
            （转为空 ``MosaicData``）。
        """
        result = self.execute_result(
            input_data,
            config=config,
            callbacks=callbacks,
            context=context,
            fail_fast=fail_fast,
            max_workers=max_workers,
        )
        return result.output if result.output is not None else MosaicData()

    def execute_result(
        self,
        input_data: MosaicData,
        *,
        config: Optional[RunConfig] = None,
        callbacks: Optional[List[EventHandler]] = None,
        context: Optional[Context] = None,
        fail_fast: bool = True,
        max_workers: int = 4,
    ) -> PipelineResult:
        """执行管道，返回完整的 :class:`PipelineResult`。

        与 :meth:`execute` 相同的执行逻辑，但返回包含中间产物、错误列表、
        各节点耗时的完整结果对象。

        Parameters
        ----------
        input_data:
            管道输入数据。
        config:
            运行配置。``None`` 使用默认配置。
        callbacks:
            事件回调列表。
        context:
            外部传入的运行上下文。``None`` 时创建新上下文。
        fail_fast:
            某节点失败时是否立即抛出异常。``True``（默认）立即抛出；
            ``False`` 收集所有错误，其他分支继续执行。
        max_workers:
            并行执行的最大线程数。

        Returns
        -------
        PipelineResult
            包含最终输出、中间产物、错误列表和耗时统计的完整结果。
        """
        self._build_dag_if_needed()
        self.validate()

        ctx = context or Context(config=config, initial_data=input_data)
        if callbacks:
            for cb in callbacks:
                ctx.on_event(cb)

        t_start = time.perf_counter()

        with ctx:
            # 空管道：原样返回输入
            if not self._dag:
                ctx.store_artifact("__input__", input_data)
                self._last_context = ctx
                elapsed = time.perf_counter() - t_start
                result = PipelineResult(
                    output=input_data,
                    intermediate={"__input__": input_data},
                    duration=elapsed,
                    pipeline_name=self.name,
                )
                self._last_result = result
                return result

            # 并行执行 DAG
            outputs, errors, node_durations = self._execute_dag(
                ctx, input_data, fail_fast=fail_fast, max_workers=max_workers
            )

            # 收集最终输出
            final_output = self._collect_output(outputs)

        elapsed = time.perf_counter() - t_start
        self._last_context = ctx

        # 构建中间产物字典
        intermediate = {nid: out for nid, out in outputs.items()}

        result = PipelineResult(
            output=final_output,
            intermediate=intermediate,
            errors=errors,
            duration=elapsed,
            node_durations=node_durations,
            pipeline_name=self.name,
        )
        self._last_result = result
        return result

    def _execute_dag(
        self,
        ctx: Context,
        input_data: MosaicData,
        *,
        fail_fast: bool = True,
        max_workers: int = 4,
    ) -> Tuple[Dict[str, MosaicData], List[NodeError], Dict[str, float]]:
        """执行 DAG，支持并行。

        使用依赖驱动的就绪队列：当一个节点的所有前驱都完成时，它变为
        "就绪"状态。多个就绪节点使用 ``ThreadPoolExecutor`` 并行执行。

        Returns
        -------
        Tuple[outputs, errors, node_durations]
            ``(节点输出字典, 错误列表, 节点耗时字典)``。
        """
        outputs: Dict[str, MosaicData] = {}
        errors: List[NodeError] = []
        node_durations: Dict[str, float] = {}
        completed: set = set()
        pending: set = set(self._dag.keys())

        # 检测是否有可并行的节点（多前驱的 fan-out）
        has_parallel = self._has_parallel_paths()

        if not has_parallel or max_workers <= 1:
            # 串行执行（兼容模式）
            return self._execute_serial(ctx, input_data, fail_fast=fail_fast)

        # 并行执行
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures: Dict[concurrent.futures.Future, str] = {}

            while pending or futures:
                # 找出就绪节点（所有前驱已完成）
                ready: List[str] = []
                for nid in list(pending):
                    preds = self._dag[nid].predecessors
                    if all(p in completed for p in preds):
                        ready.append(nid)

                # 提交就绪节点
                for nid in ready:
                    pending.discard(nid)
                    dn = self._dag[nid]
                    # 在主线程组装输入（线程安全读取 outputs）
                    node_input = self._gather_input(dn, outputs, input_data)
                    # 提交节点执行到线程池
                    future = executor.submit(
                        self._run_single_node, dn, node_input, ctx
                    )
                    futures[future] = nid

                if not futures:
                    if pending:
                        # 死锁：有待执行节点但无就绪节点
                        raise PipelineError(
                            f"Deadlock in DAG execution. Pending: {sorted(pending)}, "
                            f"Completed: {sorted(completed)}"
                        )
                    break

                # 等待至少一个完成
                done, _not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    nid = futures.pop(future)
                    dn = self._dag[nid]
                    try:
                        out, duration = future.result()
                        outputs[nid] = out
                        node_durations[nid] = duration
                        completed.add(nid)
                    except Exception as exc:
                        if fail_fast:
                            # 取消所有未完成的 future
                            for f in futures:
                                f.cancel()
                            raise
                        # 收集错误，标记为已完成（避免死锁）
                        errors.append(NodeError(
                            node_id=nid,
                            node_name=dn.node.name,
                            error=exc,
                            branch_name=dn.branch_name,
                        ))
                        completed.add(nid)
        finally:
            executor.shutdown(wait=True)

        return outputs, errors, node_durations

    def _execute_serial(
        self,
        ctx: Context,
        input_data: MosaicData,
        *,
        fail_fast: bool = True,
    ) -> Tuple[Dict[str, MosaicData], List[NodeError], Dict[str, float]]:
        """串行执行 DAG（兼容旧逻辑）。

        当 ``max_workers <= 1`` 或无并行路径时使用。
        """
        outputs: Dict[str, MosaicData] = {}
        errors: List[NodeError] = []
        node_durations: Dict[str, float] = {}
        order = self._topological_order()

        for nid in order:
            dn = self._dag[nid]
            node_input = self._gather_input(dn, outputs, input_data)

            ctx.emit(
                Event(
                    event_type="node_start",
                    node_name=dn.node.name,
                    payload={"input_keys": list(node_input.keys())},
                )
            )
            t0 = time.perf_counter()
            try:
                if not dn.node.is_loaded():
                    dn.node.load()
                out = dn.node.run(node_input)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                node_durations[nid] = elapsed
                if fail_fast:
                    raise
                errors.append(NodeError(
                    node_id=nid,
                    node_name=dn.node.name,
                    error=exc,
                    branch_name=dn.branch_name,
                ))
                # 失败节点输出为空 MosaicData，不阻塞后续
                out = MosaicData()
            elapsed = time.perf_counter() - t0
            outputs[nid] = out
            node_durations[nid] = elapsed
            ctx.store_artifact(nid, out, duration=elapsed)
            ctx.emit(
                Event(
                    event_type="node_end",
                    node_name=dn.node.name,
                    payload={
                        "output_keys": list(out.keys()),
                        "elapsed_seconds": elapsed,
                    },
                )
            )

        return outputs, errors, node_durations

    def _run_single_node(
        self,
        dn: _DAGNode,
        node_input: MosaicData,
        ctx: Context,
    ) -> Tuple[MosaicData, float]:
        """在并行执行中运行单个节点（工作线程调用）。

        Returns
        -------
        Tuple[MosaicData, float]
            ``(节点输出, 耗时秒数)``。
        """
        ctx.emit(
            Event(
                event_type="node_start",
                node_name=dn.node.name,
                payload={"input_keys": list(node_input.keys())},
            )
        )
        t0 = time.perf_counter()
        if not dn.node.is_loaded():
            dn.node.load()
        out = dn.node.run(node_input)
        elapsed = time.perf_counter() - t0
        ctx.store_artifact(dn.node_id, out, duration=elapsed)
        ctx.emit(
            Event(
                event_type="node_end",
                node_name=dn.node.name,
                payload={
                    "output_keys": list(out.keys()),
                    "elapsed_seconds": elapsed,
                },
            )
        )
        return out, elapsed

    def _has_parallel_paths(self) -> bool:
        """检测 DAG 中是否存在可并行执行的路径。

        以下任一条件成立时返回 True：
        1. 某个节点有多个后继（fan-out）。
        2. 多个源节点（无前驱）同时存在（它们天然可并行）。
        3. 多个节点同时就绪（共享前驱集，且前驱已完成）。
        """
        if self._dag is None:
            return False
        # 条件 1：fan-out 节点
        for dn in self._dag.values():
            if len(dn.successors) > 1:
                return True
        # 条件 2：多个源节点
        sources = [nid for nid, dn in self._dag.items() if not dn.predecessors]
        if len(sources) > 1:
            return True
        # 条件 3：多个节点共享同一前驱集（同一层可并行）
        from collections import Counter
        pred_sig_count: Counter = Counter()
        for dn in self._dag.values():
            if dn.predecessors:
                sig = tuple(sorted(dn.predecessors))
                pred_sig_count[sig] += 1
        for count in pred_sig_count.values():
            if count > 1:
                return True
        return False

    def _gather_input(
        self,
        dn: _DAGNode,
        outputs: Dict[str, MosaicData],
        pipeline_input: MosaicData,
    ) -> MosaicData:
        """为某个 DAG 节点组装输入。"""
        if not dn.predecessors:
            # 源点：使用管道输入
            return pipeline_input
        if len(dn.predecessors) == 1:
            # 单前驱：直接透传
            pred_id = dn.predecessors[0]
            if pred_id in outputs:
                return outputs[pred_id]
            # 前驱失败（fail_fast=False），返回空
            return MosaicData()
        # 多前驱（fan-in）：按标签组装为 {标签: 前驱输出}
        combined = MosaicData()
        for label, pid in zip(dn.input_labels, dn.predecessors):
            if pid in outputs:
                combined[label] = outputs[pid]
        return combined

    def _collect_output(self, outputs: Dict[str, MosaicData]) -> Optional[MosaicData]:
        """从终点收集管道最终输出。"""
        if not self._terminals:
            return MosaicData()
        if len(self._terminals) == 1:
            tid = self._terminals[0][1]
            return outputs.get(tid, MosaicData())
        # 多终点：以标签聚合
        result = MosaicData()
        for label, tid in self._terminals:
            if tid in outputs:
                result[label] = outputs[tid]
        return result

    # -- 中间产物访问 ------------------------------------------------------
    @property
    def intermediate_names(self) -> List[str]:
        """上次运行后可用的中间产物（节点 id）列表。"""
        if self._last_context is None:
            return []
        return list(self._last_context.artifacts.keys())

    def get_intermediate(self, name: str) -> MosaicData:
        """获取某节点的中间输出。

        优先按节点 id 精确匹配；若未命中，则按节点 ``name`` 取首个匹配。

        Raises
        ------
        RuntimeError
            管道尚未运行。
        KeyError
            找不到对应产物。
        """
        if self._last_context is None:
            raise RuntimeError("Pipeline has not been run yet; no intermediates available.")
        artifacts = self._last_context.artifacts
        if name in artifacts:
            return artifacts[name]
        # 回退：按节点显示名匹配
        assert self._dag is not None
        for nid, dn in self._dag.items():
            if dn.node.name == name and nid in artifacts:
                return artifacts[nid]
        raise KeyError(
            f"No intermediate for {name!r}. "
            f"Available: {list(artifacts.keys())}"
        )

    @property
    def last_result(self) -> Optional[PipelineResult]:
        """上次运行的 :class:`PipelineResult`（``execute_result`` 后可用）。"""
        return self._last_result

    @property
    def node_specs(self) -> List[NodeSpec]:
        """按拓扑序返回各节点的规格说明。"""
        self._build_dag_if_needed()
        order = self._topological_order()
        specs: List[NodeSpec] = []
        for nid in order:
            try:
                specs.append(self._dag[nid].node.describe())
            except Exception:
                continue
        return specs

    def run_async(
        self,
        input_data: MosaicData,
        **kwargs: Any,
    ) -> "AsyncTask":
        """异步执行管道，返回 :class:`~mosaic.core.task.AsyncTask`。

        在新线程中调用 :meth:`execute_result`，不阻塞调用线程。
        适用于视频生成等长时间运行的任务。

        Parameters
        ----------
        input_data:
            管道输入数据。
        **kwargs:
            透传给 :meth:`execute_result` 的额外参数
            （如 ``config``、``fail_fast``、``max_workers``）。

        Returns
        -------
        AsyncTask
            异步任务实例，可用于查询状态、等待结果、注册回调或取消。

        Examples
        --------
        >>> task = pipe.run_async(input_data)
        >>> task.status      # "pending" / "running" / "completed" / "failed"
        >>> task.progress    # 0.0 - 1.0
        >>> result = task.wait(timeout=300)

        使用回调：
        >>> task = pipe.run_async(input_data)
        >>> task.on_complete(lambda r: print(f"Done: {r}"))
        >>> task.on_error(lambda e: print(f"Error: {e}"))
        """
        from mosaic.core.async_pipeline import create_async_task

        return create_async_task(
            pipeline=self,
            input_data=input_data,
            **kwargs,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        count = len(self._elements)
        return f"<Pipeline name={self.name!r} elements={count} state={status}>"
