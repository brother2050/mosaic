# mosaic/nodes/export/__init__.py
"""导出域节点包。

导出域全部 3 个节点：

* VideoEncoder         —— 视频编码封装（FFmpeg）
* Livestreamer         —— 直播推流（RTMP/SRT）
* MultiFormatExporter  —— 多格式导出（视频/图像/音频/字幕）

——
导出域为纯工程域，不涉及 AI 模型推理，不需要 GPU。
所有节点继承 :class:`Node` 基类，通过 ``@registry.register``
自动注册到全局节点注册表。
"""

from mosaic.nodes.export.video_encoder import VideoEncoder
from mosaic.nodes.export.livestream import Livestreamer
from mosaic.nodes.export.multi_format_exporter import MultiFormatExporter

__all__ = [
    "VideoEncoder",
    "Livestreamer",
    "MultiFormatExporter",
]
