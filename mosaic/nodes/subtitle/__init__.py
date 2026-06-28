# mosaic/nodes/subtitle/__init__.py
"""字幕域节点。

导出该域所有节点类。当前包含 3 个节点：

* :class:`SubtitleGenerator`  —— 字幕生成（从音频/视频，基于 Whisper）
* :class:`SubtitleTranslator` —— 字幕翻译（保持时间轴，批量翻译）
* :class:`SubtitleAligner`    —— 时间轴对齐（Whisper / aeneas / DTW）
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
