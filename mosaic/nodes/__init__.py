# mosaic/nodes/__init__.py
"""Mosaic 节点域集合。

本包按领域组织所有节点，每个子包对应一个模态域：

已实现的域
----------
- :mod:`mosaic.nodes.text`     — 文本域（6 节点：生成/对话/改写/翻译/摘要/分类）
- :mod:`mosaic.nodes.image`    — 图像域（6 节点：文生图/图生图/重绘/放大/去背景/风格化）
- :mod:`mosaic.nodes.audio`    — 音频域（5 节点：TTS/ASR/音乐/音效/克隆）
- :mod:`mosaic.nodes.subtitle` — 字幕域（3 节点：生成/翻译/对齐）

规划中的域
----------
- :mod:`mosaic.nodes.video`          — 视频域
- :mod:`mosaic.nodes.rag`            — RAG 检索增强生成
- :mod:`mosaic.nodes.export`         — 多格式导出
- :mod:`mosaic.nodes.digital_human`  — 数字人
- :mod:`mosaic.nodes.consistency`    — 一致性检查

便捷导入
--------
>>> from mosaic.nodes import TextGenerator, TextToImage, TTS, SubtitleGenerator
"""

# 导入已实现的域（惰性导入，避免未安装依赖时报错）
# 各域 __init__ 内部均使用惰性导入，此处仅导入包本身
from mosaic.nodes import text as text
from mosaic.nodes import image as image
from mosaic.nodes import audio as audio
from mosaic.nodes import subtitle as subtitle

# 便捷导出常用节点类
from mosaic.nodes.text import (
    Chat,
    TextClassifier,
    TextGenerator,
    TextRewriter,
    TextSummarizer,
    Translator,
)
from mosaic.nodes.image import (
    BackgroundRemover,
    ImageToImage,
    Inpainting,
    Stylizer,
    TextToImage,
    Upscaler,
)
from mosaic.nodes.audio import (
    ASR,
    MusicGenerator,
    SoundEffectGenerator,
    TTS,
    VoiceClone,
)
from mosaic.nodes.subtitle import (
    SubtitleAligner,
    SubtitleGenerator,
    SubtitleTranslator,
)

__all__ = [
    # 子包
    "text",
    "image",
    "audio",
    "subtitle",
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
]
