# mosaic/nodes/digital_human/__init__.py
"""数字人域节点包。

导出数字人域的全部节点与基类：

* :class:`BaseDigitalHumanNode` —— 数字人域节点抽象基类
* :class:`AvatarDriver`        —— 形象驱动节点
* :class:`LipSyncer`           —— 唇形同步节点
* :class:`MotionGenerator`     —— 动作生成节点
* :class:`RealtimeRenderer`    —— 实时渲染节点

设计要点
--------
* ``BaseDigitalHumanNode`` / ``MotionGenerator`` / ``RealtimeRenderer`` 由
  本包直接提供，始终可用。
* ``AvatarDriver`` / ``LipSyncer`` 由并行任务创建，可能在导入时尚未就绪
  或处于临时损坏状态。本模块对这两个导入做容错处理：若对应模块暂不可
  导入（缺失 / 语法错误 / 依赖未安装等），仅记录警告并跳过，不影响其余
  节点的导入与注册表发现。待并行任务完成后即可正常导入。
"""

from __future__ import annotations

import logging

from mosaic.nodes.digital_human._base import BaseDigitalHumanNode
from mosaic.nodes.digital_human.motion_generator import MotionGenerator
from mosaic.nodes.digital_human.realtime_renderer import RealtimeRenderer

__all__ = [
    "BaseDigitalHumanNode",
    "AvatarDriver",
    "LipSyncer",
    "MotionGenerator",
    "RealtimeRenderer",
]

_logger = logging.getLogger("mosaic.nodes.digital_human")

# AvatarDriver / LipSyncer 由并行任务创建，导入时可能尚未就绪或临时损坏。
# 此处做容错：捕获 ImportError（模块缺失/依赖未装）与 SyntaxError（文件
# 正在写入/损坏）等，仅记录警告并跳过，不影响其余节点的可用性。
# 待并行任务完成、文件就绪后即可正常导入。
try:
    from mosaic.nodes.digital_human.avatar_driver import AvatarDriver  # noqa: F401
except (ImportError, SyntaxError) as _exc:  # pragma: no cover - 依赖并行任务就绪
    _logger.warning(
        "avatar_driver module not yet available (%s: %s). "
        "AvatarDriver will be unavailable until it is created.",
        type(_exc).__name__,
        _exc,
    )
    AvatarDriver = None  # type: ignore[assignment, misc]

try:
    from mosaic.nodes.digital_human.lip_syncer import LipSyncer  # noqa: F401
except (ImportError, SyntaxError) as _exc:  # pragma: no cover - 依赖并行任务就绪
    _logger.warning(
        "lip_syncer module not yet available (%s: %s). "
        "LipSyncer will be unavailable until it is created.",
        type(_exc).__name__,
        _exc,
    )
    LipSyncer = None  # type: ignore[assignment, misc]
