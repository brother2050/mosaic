# mosaic/nodes/__init__.py
"""Mosaic 节点域集合。

本包按领域组织所有节点，每个子包对应一个模态域：

已实现的域
----------
- :mod:`mosaic.nodes.text`     — 文本域（6 节点：生成/对话/改写/翻译/摘要/分类）
- :mod:`mosaic.nodes.image`    — 图像域（6 节点：文生图/图生图/重绘/放大/去背景/风格化）
- :mod:`mosaic.nodes.audio`    — 音频域（5 节点：TTS/ASR/音乐/音效/克隆）
- :mod:`mosaic.nodes.subtitle` — 字幕域（3 节点：生成/翻译/对齐）
- :mod:`mosaic.nodes.video`    — 视频域（5 节点：文生视频/图生视频/续写/插帧/拆帧）

规划中的域
----------
- :mod:`mosaic.nodes.rag`            — RAG 检索增强生成
- :mod:`mosaic.nodes.export`         — 多格式导出
- :mod:`mosaic.nodes.digital_human`  — 数字人
- :mod:`mosaic.nodes.consistency`    — 一致性检查

便捷导入
--------
>>> from mosaic.nodes import TextGenerator, TextToImage, TTS, SubtitleGenerator
>>> from mosaic.nodes import TextToVideo, FrameExtractor
"""

# 导入已实现的域
from mosaic.nodes import text as text
from mosaic.nodes import image as image
from mosaic.nodes import audio as audio
from mosaic.nodes import subtitle as subtitle
from mosaic.nodes import video as video

# 便捷导出：基类
from mosaic.nodes.text import BaseTextNode
from mosaic.nodes.image import BaseImageNode
from mosaic.nodes.audio import BaseAudioNode
from mosaic.nodes.subtitle import BaseSubtitleNode
from mosaic.nodes.video import BaseVideoNode

# 便捷导出：text 域节点
from mosaic.nodes.text import (
    Chat,
    TextClassifier,
    TextGenerator,
    TextRewriter,
    TextSummarizer,
    Translator,
)
# 便捷导出：image 域节点
from mosaic.nodes.image import (
    BackgroundRemover,
    ImageToImage,
    Inpainting,
    Stylizer,
    TextToImage,
    Upscaler,
)
# 便捷导出：audio 域节点
from mosaic.nodes.audio import (
    ASR,
    MusicGenerator,
    SoundEffectGenerator,
    TTS,
    VoiceClone,
)
# 便捷导出：subtitle 域节点
from mosaic.nodes.subtitle import (
    SubtitleAligner,
    SubtitleGenerator,
    SubtitleTranslator,
)
# 便捷导出：video 域节点
from mosaic.nodes.video import (
    FrameExtractor,
    FrameInterpolator,
    ImageToVideo,
    TextToVideo,
    VideoContinuation,
)

__all__ = [
    # 子包
    "text",
    "image",
    "audio",
    "subtitle",
    "video",
    # base classes
    "BaseTextNode",
    "BaseImageNode",
    "BaseAudioNode",
    "BaseSubtitleNode",
    "BaseVideoNode",
    # text nodes
    "TextGenerator",
    "Chat",
    "TextRewriter",
    "Translator",
    "TextSummarizer",
    "TextClassifier",
    # image nodes
    "TextToImage",
    "ImageToImage",
    "Inpainting",
    "Upscaler",
    "BackgroundRemover",
    "Stylizer",
    # audio nodes
    "TTS",
    "ASR",
    "MusicGenerator",
    "SoundEffectGenerator",
    "VoiceClone",
    # subtitle nodes
    "SubtitleGenerator",
    "SubtitleTranslator",
    "SubtitleAligner",
    # video nodes
    "TextToVideo",
    "ImageToVideo",
    "VideoContinuation",
    "FrameInterpolator",
    "FrameExtractor",
]
