"""tests/tts 公共 fixtures。

提供 TTS 测试中复用的 mock 对象与样本张量。这些 fixtures 不依赖真实
预训练权重，也不强制要求可选依赖（torch 缺失时样本张量相关用例自动跳过）。
"""
from __future__ import annotations

import sys
from typing import Any

import pytest
from unittest.mock import MagicMock

# 确保能直接 import mosaic 包（与仓库内其它 conftest 行为一致）
sys.path.insert(0, "/workspace/mosaic")


# ----------------------------------------------------------------------
# Mock fixtures：模拟 TTS 三大组件，不依赖真实词表 / 模型权重
# ----------------------------------------------------------------------
@pytest.fixture
def mock_text_frontend() -> Any:
    """返回一个简单的 TextFrontend mock（不需要真实词表）。"""
    frontend = MagicMock()
    frontend.vocab_size = 120
    frontend.model_type = "ar"
    # tokenize 返回简单的 token id 列表，避免耦合 torch
    frontend.tokenize.return_value = [0, 1, 2, 3, 4, 5]
    frontend.detokenize.return_value = "mock 文本"
    frontend.encode_speaker.return_value = None
    frontend.preprocess.side_effect = lambda text: text
    frontend.insert_prosody_tokens.side_effect = lambda text, prosody: text
    frontend.unload_weights = MagicMock()
    return frontend


@pytest.fixture
def mock_acoustic_model() -> Any:
    """返回一个简单的 AcousticModel mock。"""
    model = MagicMock()
    model.model_type = "ar"
    model.vocab_size = 4216
    model.hidden_size = 512
    # generate 返回模拟的音频码序列
    model.generate.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]]
    model.generate_stream.return_value = iter([[[0, 1, 2, 3]], [[4, 5, 6, 7]]])
    model.get_input_embeddings.return_value = MagicMock()
    model.get_output_head.return_value = MagicMock()
    model.unload_weights = MagicMock()
    return model


@pytest.fixture
def mock_vocoder() -> Any:
    """返回一个简单的 Vocoder mock。"""
    vocoder = MagicMock()
    vocoder.vocoder_type = "vocos"
    vocoder.input_type = "mel"
    vocoder.sample_rate = 24000
    # decode 返回 (waveform, sample_rate)
    vocoder.decode.return_value = ([0.0] * 100, 24000)
    vocoder.decode_chunk.return_value = ([0.0] * 100, 24000)
    vocoder.unload_weights = MagicMock()
    return vocoder


# ----------------------------------------------------------------------
# 样本张量 fixtures（torch 可用时返回，否则触发跳过）
# ----------------------------------------------------------------------
@pytest.fixture
def sample_token_ids() -> Any:
    """返回随机 token ids tensor [1, 20]（torch 可用时）。"""
    torch = pytest.importorskip("torch")
    return torch.randint(0, 100, (1, 20), dtype=torch.long)


@pytest.fixture
def sample_mel() -> Any:
    """返回随机 mel spectrogram tensor [1, 80, 50]。"""
    torch = pytest.importorskip("torch")
    return torch.randn(1, 80, 50)


@pytest.fixture
def sample_audio_codes() -> Any:
    """返回随机音频码 tensor [4, 50]。"""
    torch = pytest.importorskip("torch")
    return torch.randint(0, 64, (4, 50), dtype=torch.long)


# ----------------------------------------------------------------------
# Fish Speech 相关 fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_fish_tokenizer() -> Any:
    """返回一个 FishTokenizer mock。"""
    frontend = MagicMock()
    frontend.vocab_size = 12000
    frontend.model_type = "ar"
    frontend.special_tokens = {
        "<s>": 0, "</s>": 1, "<text>": 2, "<audio>": 3,
        "<clone>": 4, "<pad>": 5, "<zh>": 6, "<en>": 7,
        "<ja>": 8, "<ko>": 9,
    }
    frontend.tokenize.return_value = [0, 6, 10, 11, 12, 3]
    frontend.detokenize.return_value = "mock 文本"
    frontend.encode_speaker.return_value = None
    frontend.preprocess.side_effect = lambda text: text
    frontend.insert_prosody_tokens.side_effect = lambda text, prosody: text
    frontend.unload_weights = MagicMock()
    return frontend


@pytest.fixture
def mock_fish_acoustic_model() -> Any:
    """返回一个 FishLlamaARModel mock。"""
    model = MagicMock()
    model.model_type = "ar"
    model.vocab_size = 12000
    model.hidden_size = 1024
    model._text_vocab_size = 10000
    model._audio_vocab_size = 2000
    model.generate.return_value = [[100, 200, 300, 400, 500]]
    model.generate_stream.return_value = iter([[[100, 200]], [[300, 400]], [[500]]])
    model.get_input_embeddings.return_value = MagicMock()
    model.get_output_head.return_value = MagicMock()
    model.encode_reference_audio.return_value = [1500, 1600, 1700]
    model.unload_weights = MagicMock()
    return model


@pytest.fixture
def mock_vq_decoder() -> Any:
    """返回一个 VQDecoder mock。"""
    decoder = MagicMock()
    decoder.forward.return_value = [[0.1] * 80 for _ in range(50)]
    decoder.forward_chunk.return_value = [[0.1] * 80 for _ in range(24)]
    decoder.unload_weights = MagicMock()
    return decoder


@pytest.fixture
def mock_hifi_gan() -> Any:
    """返回一个 HiFiGanVocoder mock。"""
    vocoder = MagicMock()
    vocoder.vocoder_type = "hifi_gan"
    vocoder.input_type = "mel"
    vocoder.sample_rate = 22050
    vocoder.decode.return_value = ([0.0] * 100, 22050)
    vocoder.decode_chunk.return_value = ([0.0] * 100, 22050)
    vocoder.unload_weights = MagicMock()
    return vocoder


@pytest.fixture
def sample_ref_audio_codec_tokens() -> Any:
    """返回模拟的参考音频 codec token ids [1, 30]。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randint(0, 2000, (1, 30), dtype=torch.long)
