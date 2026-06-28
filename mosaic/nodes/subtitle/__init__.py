# mosaic/nodes/subtitle/__init__.py
"""字幕域节点包。

导出字幕域全部 3 个节点：

- :class:`SubtitleGenerator` — 字幕生成（从音频/视频）
- :class:`SubtitleTranslator` — 字幕翻译（保持时间轴）
- :class:`SubtitleAligner` — 时间轴对齐
"""

from mosaic.nodes.subtitle._base import BaseSubtitleNode
from mosaic.nodes.subtitle.generator import SubtitleGenerator
from mosaic.nodes.subtitle.translator import SubtitleTranslator
from mosaic.nodes.subtitle.aligner import SubtitleAligner

__all__ = [
    "BaseSubtitleNode",
    "SubtitleGenerator",
    "SubtitleTranslator",
    "SubtitleAligner",
]
