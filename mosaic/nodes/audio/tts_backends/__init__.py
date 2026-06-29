# mosaic/nodes/audio/tts_backends/__init__.py
"""TTS 扩展后端框架。

四层架构：
  Layer 1: TextFrontend — 文本前端（清洗、韵律、分词、tokenize）
  Layer 2: AcousticModel — 声学模型（token → mel/VQ tokens）
  Layer 3: Vocoder — 声码器（mel/VQ → waveform）
  Layer 4: StreamAdapter — 流式适配（缓冲区、chunk、实时输出）
"""

from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec
from mosaic.nodes.audio.tts_backends.registry import TTSBackendRegistry, tts_backend_registry
from mosaic.nodes.audio.tts_backends.text_frontends.base import TextFrontend
from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel
from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder
from mosaic.nodes.audio.tts_backends.streaming.base import StreamAdapter, StreamSession

__all__ = [
    "TTSBackend", "TTSBackendSpec",
    "TTSBackendRegistry", "tts_backend_registry",
    "TextFrontend", "AcousticModel", "Vocoder",
    "StreamAdapter", "StreamSession",
]
