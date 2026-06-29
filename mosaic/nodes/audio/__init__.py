# mosaic/nodes/audio/__init__.py
"""音频域节点。

导出该域所有节点类。当前包含 5 个节点：

* :class:`TTS`                  —— 文本转语音（edge-tts 默认 / transformers / 扩展后端）
* :class:`ASR`                  —— 语音识别（Whisper）
* :class:`MusicGenerator`       —— 音乐生成（MusicGen）
* :class:`SoundEffectGenerator` —— 音效生成（AudioLDM2）
* :class:`VoiceClone`           —— 语音风格匹配（edge-tts 预设语音）

TTS 扩展后端框架（四层架构）：

* :class:`TTSBackend`           —— TTS 后端抽象基类
* :class:`TTSBackendSpec`       —— 后端规格信息
* :class:`TTSBackendRegistry`   —— 后端注册表
* :class:`TextFrontend`         —— 文本前端层
* :class:`AcousticModel`        —— 声学模型层
* :class:`Vocoder`              —— 声码器层
* :class:`StreamAdapter`        —— 流式适配层
"""

from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.nodes.audio.tts import TTS
from mosaic.nodes.audio.asr import ASR
from mosaic.nodes.audio.music_generator import MusicGenerator
from mosaic.nodes.audio.sound_effect import SoundEffectGenerator
from mosaic.nodes.audio.voice_clone import VoiceClone

from mosaic.nodes.audio.tts_backends import (
    TTSBackend,
    TTSBackendSpec,
    TTSBackendRegistry,
    tts_backend_registry,
    TextFrontend,
    AcousticModel,
    Vocoder,
    StreamAdapter,
    StreamSession,
)

__all__ = [
    "BaseAudioNode",
    "TTS",
    "ASR",
    "MusicGenerator",
    "SoundEffectGenerator",
    "VoiceClone",
    # TTS 扩展后端框架
    "TTSBackend",
    "TTSBackendSpec",
    "TTSBackendRegistry",
    "tts_backend_registry",
    "TextFrontend",
    "AcousticModel",
    "Vocoder",
    "StreamAdapter",
    "StreamSession",
]
