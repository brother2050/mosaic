# mosaic/nodes/audio/__init__.py
"""音频域节点。

导出该域所有节点类。当前包含 5 个节点：

* :class:`TTS`                  —— 文本转语音（XTTS-v2 / edge-tts）
* :class:`ASR`                  —— 语音识别（Whisper）
* :class:`MusicGenerator`       —— 音乐生成（MusicGen）
* :class:`SoundEffectGenerator` —— 音效生成（AudioLDM2）
* :class:`VoiceClone`           —— 语音克隆（XTTS-v2）
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
