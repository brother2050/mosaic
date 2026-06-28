# mosaic/nodes/audio/__init__.py
"""音频域节点包。

导出音频域全部 5 个节点：

- :class:`TTS` — 文本转语音
- :class:`ASR` — 语音识别
- :class:`MusicGenerator` — 音乐生成
- :class:`SoundEffectGenerator` — 音效生成
- :class:`VoiceClone` — 语音克隆
"""

from mosaic.nodes.audio._base import BaseAudioNode
from mosaic.nodes.audio.tts import TTS
from mosaic.nodes.audio.asr import ASR
from mosaic.nodes.audio.music_generator import MusicGenerator
from mosaic.nodes.audio.sound_effect import SoundEffectGenerator
from mosaic.nodes.audio.voice_clone import VoiceClone

__all__ = [
    "BaseAudioNode",
    "TTS",
    "ASR",
    "MusicGenerator",
    "SoundEffectGenerator",
    "VoiceClone",
]
