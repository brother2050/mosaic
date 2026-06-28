# mosaic/nodes/text/__init__.py
"""文本域节点。

导出该域所有节点类。当前包含 6 个节点：

* :class:`TextGenerator`  —— 根据 prompt 生成文本
* :class:`Chat`           —— 多轮对话
* :class:`TextRewriter`   —— 文本改写（风格/要求）
* :class:`Translator`     —— 文本翻译（专用模型/通用生成）
* :class:`TextSummarizer` —— 文本摘要（concise/detailed/bullet_points）
* :class:`TextClassifier` —— 文本分类（专用/生成/zero-shot）
"""

from mosaic.nodes.text._base import BaseTextNode
from mosaic.nodes.text.chat import Chat
from mosaic.nodes.text.classifier import TextClassifier
from mosaic.nodes.text.generator import TextGenerator
from mosaic.nodes.text.rewriter import TextRewriter
from mosaic.nodes.text.summarizer import TextSummarizer
from mosaic.nodes.text.translator import Translator

__all__ = [
    "BaseTextNode",
    "TextGenerator",
    "Chat",
    "TextRewriter",
    "Translator",
    "TextSummarizer",
    "TextClassifier",
]
