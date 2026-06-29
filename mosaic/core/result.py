# mosaic/core/result.py
"""管道运行结果。

定义 :class:`PipelineResult`，封装管道执行后的完整信息：最终输出、中间产物、
错误列表、各节点耗时等。``Pipeline.execute_result()`` 返回此对象。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mosaic.core.types import MosaicData

__all__ = ["PipelineResult", "NodeError"]


@dataclass
class NodeError:
    """单个节点执行错误记录。

    Attributes
    ----------
    node_id:
        节点在 DAG 中的唯一标识。
    node_name:
        节点显示名。
    error:
        原始异常对象。
    branch_name:
        所属分支名（如果是分支中的节点），否则为 ``None``。
    """

    node_id: str
    node_name: str
    error: BaseException
    branch_name: str | None = None

    def __repr__(self) -> str:
        loc = f" branch={self.branch_name!r}" if self.branch_name else ""
        return (
            f"NodeError(node={self.node_name!r}, "
            f"error={type(self.error).__name__}: {self.error}){loc}"
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "branch_name": self.branch_name,
        }


@dataclass
class PipelineResult:
    """管道执行结果。

    封装管道运行后的所有信息，包括最终输出、中间产物、错误列表和耗时统计。

    Attributes
    ----------
    output:
        管道最终输出。全部成功时为最后一个节点（或 Merge）的输出；
        部分失败时可能为 ``None``。
    intermediate:
        中间产物字典 ``{node_id: MosaicData}``，包含每个已执行节点的输出。
    errors:
        执行过程中发生的错误列表。``fail_fast=True`` 时最多 1 个；
        ``fail_fast=False`` 时包含所有失败节点的错误。
    duration:
        管道总执行耗时（秒）。
    node_durations:
        各节点执行耗时字典 ``{node_id: float}``（秒）。
    pipeline_name:
        管道名称。

    Examples
    --------
    >>> result = pipe.execute_result(input_data)
    >>> if result.success:
    ...     print(result.output)
    ... else:
    ...     print(f"Failed nodes: {result.failed_nodes}")
    ...     for err in result.errors:
    ...         print(err)
    """

    output: MosaicData | None = None
    intermediate: dict[str, MosaicData] = field(default_factory=dict)
    errors: list[NodeError] = field(default_factory=list)
    duration: float = 0.0
    node_durations: dict[str, float] = field(default_factory=dict)
    pipeline_name: str = ""

    # -- 便捷属性 ----------------------------------------------------------
    @property
    def success(self) -> bool:
        """是否全部成功（无错误）。"""
        return len(self.errors) == 0

    @property
    def failed_nodes(self) -> list[str]:
        """失败的节点名列表。"""
        return [err.node_name for err in self.errors]

    @property
    def failed_node_ids(self) -> list[str]:
        """失败的节点 id 列表。"""
        return [err.node_id for err in self.errors]

    @property
    def node_count(self) -> int:
        """已执行的节点总数（含失败）。"""
        return len(self.node_durations)

    # -- 中间产物访问 ------------------------------------------------------
    def get_intermediate(self, node_name: str) -> MosaicData:
        """获取指定节点的中间产物。

        优先按节点 id 精确匹配；若未命中，按节点 name 模糊匹配首个。

        Raises
        ------
        KeyError
            找不到对应产物。
        """
        if node_name in self.intermediate:
            return self.intermediate[node_name]
        # 模糊匹配：以 node_name 开头或包含的 id
        for nid, data in self.intermediate.items():
            # 去除 #n 后缀后比较
            base = nid.split("#")[0]
            if base == node_name:
                return data
        raise KeyError(
            f"No intermediate for {node_name!r}. "
            f"Available: {list(self.intermediate.keys())}"
        )

    def list_intermediate(self) -> list[str]:
        """列出所有中间产物的节点 id。"""
        return list(self.intermediate.keys())

    # -- 序列化 ------------------------------------------------------------
    def summary(self) -> str:
        """返回格式化的运行摘要。"""
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"  Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"  Duration: {self.duration:.3f}s",
            f"  Nodes executed: {self.node_count}",
        ]
        if self.node_durations:
            lines.append("  Node durations:")
            for nid, dur in sorted(
                self.node_durations.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    {nid}: {dur:.3f}s")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    {err}")
        if self.intermediate:
            lines.append(f"  Intermediates: {len(self.intermediate)} artifacts")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。

        注意：``MosaicData`` 会通过其 ``to_dict()`` 序列化，
        图片等二进制数据会转为 base64。
        """
        return {
            "pipeline_name": self.pipeline_name,
            "success": self.success,
            "duration": self.duration,
            "node_count": self.node_count,
            "output": self.output.to_dict() if self.output else None,
            "intermediate": {
                nid: data.to_dict() if isinstance(data, MosaicData) else data
                for nid, data in self.intermediate.items()
            },
            "errors": [err.to_dict() for err in self.errors],
            "node_durations": dict(self.node_durations),
            "failed_nodes": self.failed_nodes,
        }

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"PipelineResult({status}, pipeline={self.pipeline_name!r}, "
            f"nodes={self.node_count}, duration={self.duration:.3f}s, "
            f"errors={len(self.errors)})"
        )
