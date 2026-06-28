# mosaic/nodes/video/__init__.py
"""视频域节点包。

导出视频域全部节点与基类：

* BaseVideoNode       —— 视频域抽象基类
* TextToVideo         —— 文生视频（CogVideoX）
* ImageToVideo        —— 图生视频（SVD）
* VideoContinuation   —— 视频续写（CogVideoX）
* FrameInterpolator   —— 插帧（RIFE / FILM / 线性）
* FrameExtractor      —— 拆帧

——
所有节点均继承 :class:`BaseVideoNode`，通过 ``@registry.register``
自动注册到全局节点注册表。
"""

from mosaic.nodes.video._base import BaseVideoNode
from mosaic.nodes.video.text_to_video import TextToVideo
from mosaic.nodes.video.image_to_video import ImageToVideo
from mosaic.nodes.video.video_continuation import VideoContinuation
from mosaic.nodes.video.frame_interpolation import FrameInterpolator
from mosaic.nodes.video.frame_extractor import FrameExtractor

__all__ = [
    "BaseVideoNode",
    "TextToVideo",
    "ImageToVideo",
    "VideoContinuation",
    "FrameInterpolator",
    "FrameExtractor",
]
