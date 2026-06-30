# mosaic/nodes/__init__.py
"""Mosaic 节点域集合。

本包按领域组织所有节点，每个子包对应一个模态域：

已实现的域
----------
- :mod:`mosaic.nodes.text`          — 文本域（6 节点：生成/对话/改写/翻译/摘要/分类）
- :mod:`mosaic.nodes.image`         — 图像域（6 节点：文生图/图生图/重绘/放大/去背景/风格化）
- :mod:`mosaic.nodes.audio`         — 音频域（5 节点：TTS/ASR/音乐/音效/克隆）
- :mod:`mosaic.nodes.subtitle`      — 字幕域（3 节点：生成/翻译/对齐）
- :mod:`mosaic.nodes.video`         — 视频域（8 节点：文生视频/图生视频/续写/插帧/拆帧/Wan/Hunyuan/LTX）
- :mod:`mosaic.nodes.export`        — 导出域（3 节点：编码/推流/多格式导出）
- :mod:`mosaic.nodes.consistency`   — 一致性域（3 节点：角色保持/风格保持/帧间一致）
- :mod:`mosaic.nodes.digital_human` — 数字人域（4 节点：驱动/唇形同步/动作生成/实时渲染）
- :mod:`mosaic.nodes.rag`           — RAG 检索增强生成域（4 节点：文档解析/向量索引/检索/引用生成）

便捷导入
--------
>>> from mosaic.nodes import TextGenerator, TextToImage, TTS, SubtitleGenerator
>>> from mosaic.nodes import TextToVideo, FrameExtractor, VideoEncoder
>>> from mosaic.nodes import WanVideo, IdentityKeeper, AvatarDriver, Retriever
"""

# 导入已实现的域
from mosaic.nodes import audio as audio
from mosaic.nodes import consistency as consistency
from mosaic.nodes import digital_human as digital_human
from mosaic.nodes import export as export
from mosaic.nodes import image as image
from mosaic.nodes import rag as rag
from mosaic.nodes import subtitle as subtitle
from mosaic.nodes import text as text
from mosaic.nodes import video as video

# 便捷导出：基类
from mosaic.nodes.audio import BaseAudioNode
from mosaic.nodes.consistency import BaseConsistencyNode
from mosaic.nodes.digital_human import BaseDigitalHumanNode
from mosaic.nodes.image import BaseImageNode
from mosaic.nodes.rag import BaseRagNode
from mosaic.nodes.subtitle import BaseSubtitleNode
from mosaic.nodes.text import BaseTextNode
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
    HunyuanVideo,
    ImageToVideo,
    LTXVideo,
    TextToVideo,
    VideoContinuation,
    WanVideo,
)
# 便捷导出：export 域节点
from mosaic.nodes.export import (
    Livestreamer,
    MultiFormatExporter,
    VideoEncoder,
)
# 便捷导出：consistency 域节点
from mosaic.nodes.consistency import (
    CrossFrameConsistency,
    IdentityKeeper,
    StyleKeeper,
)
# 便捷导出：digital_human 域节点
from mosaic.nodes.digital_human import (
    AvatarDriver,
    LipSyncer,
    MotionGenerator,
    RealtimeRenderer,
)
# 便捷导出：rag 域节点
from mosaic.nodes.rag import (
    CitationGenerator,
    DocumentParser,
    Retriever,
    VectorIndexer,
)

__all__ = [
    # 子包
    "text",
    "image",
    "audio",
    "subtitle",
    "video",
    "export",
    "consistency",
    "digital_human",
    "rag",
    # base classes
    "BaseTextNode",
    "BaseImageNode",
    "BaseAudioNode",
    "BaseSubtitleNode",
    "BaseVideoNode",
    "BaseConsistencyNode",
    "BaseDigitalHumanNode",
    "BaseRagNode",
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
    "WanVideo",
    "HunyuanVideo",
    "LTXVideo",
    # export nodes
    "VideoEncoder",
    "Livestreamer",
    "MultiFormatExporter",
    # consistency nodes
    "IdentityKeeper",
    "StyleKeeper",
    "CrossFrameConsistency",
    # digital_human nodes
    "AvatarDriver",
    "LipSyncer",
    "MotionGenerator",
    "RealtimeRenderer",
    # rag nodes
    "DocumentParser",
    "VectorIndexer",
    "Retriever",
    "CitationGenerator",
]
