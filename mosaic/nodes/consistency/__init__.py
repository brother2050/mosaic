# mosaic/nodes/consistency/__init__.py
"""一致性域节点。

导出该域所有节点类。当前包含 3 个节点与 1 个基类：

* :class:`BaseConsistencyNode`  —— 一致性域抽象基类
* :class:`IdentityKeeper`        —— 角色身份保持（IdentityKeeper）
* :class:`StyleKeeper`           —— 风格保持（StyleKeeper）
* :class:`CrossFrameConsistency` —— 跨帧角色 / 主体一致性生成

``IdentityKeeper`` 与 ``StyleKeeper`` 由独立任务并行提供，此处采用容错导入：
任一节点模块尚未就绪或处于编辑中的临时不可用状态（如 ``ImportError`` /
``SyntaxError``）时，本包仍可正常导入，仅缺失的名称不会被导出，并记录一条
警告日志。当对应模块就绪后即自动恢复导出。
"""

import logging

from mosaic.nodes.consistency._base import BaseConsistencyNode
from mosaic.nodes.consistency.cross_frame_consistency import CrossFrameConsistency

__all__ = [
    "BaseConsistencyNode",
    "CrossFrameConsistency",
]

_logger = logging.getLogger("mosaic.nodes.consistency")

# IdentityKeeper / StyleKeeper 由独立任务并行创建。
# 使用容错导入（捕获 Exception），确保任一节点处于缺失或临时不可用状态时
# 本包仍可被导入，从而保证 CrossFrameConsistency 等已就绪节点始终可用。
try:  # pragma: no cover - 依赖并行任务提供的模块
    from mosaic.nodes.consistency.identity_keeper import IdentityKeeper

    __all__.append("IdentityKeeper")
except Exception as exc:  # noqa: BLE001 - 容错：模块缺失或临时不可用
    _logger.warning(
        "IdentityKeeper 模块暂不可用，已跳过导出: %s: %s",
        type(exc).__name__,
        exc,
    )
    IdentityKeeper = None  # type: ignore[assignment]

try:  # pragma: no cover - 依赖并行任务提供的模块
    from mosaic.nodes.consistency.style_keeper import StyleKeeper

    __all__.append("StyleKeeper")
except Exception as exc:  # noqa: BLE001 - 容错：模块缺失或临时不可用
    _logger.warning(
        "StyleKeeper 模块暂不可用，已跳过导出: %s: %s",
        type(exc).__name__,
        exc,
    )
    StyleKeeper = None  # type: ignore[assignment]
