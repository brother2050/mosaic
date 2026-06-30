"""tests/tts 公共 fixtures。

提供 TTS 测试中复用的 mock 对象与样本张量。这些 fixtures 不依赖真实
预训练权重，也不强制要求可选依赖（torch 缺失时样本张量相关用例自动跳过）。
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

import pytest
from unittest.mock import MagicMock

# 确保能直接 import mosaic 包（与仓库内其它 conftest 行为一致）
sys.path.insert(0, "/workspace/mosaic")


# ----------------------------------------------------------------------
# 恢复真实 transformers 模块（session 级别，autouse）
# ----------------------------------------------------------------------
# 其他 Phase 的 conftest 在模块加载时可能将 transformers 替换为 mock
# （types.ModuleType("transformers") 无 __spec__）。TTS 测试需要真实的
# LlamaForCausalLM / GPT2LMHeadModel 等类，因此在此恢复真实模块。
@pytest.fixture(scope="session", autouse=True)
def _restore_real_transformers():
    """恢复真实 transformers 模块，供 TTS 加载测试使用。"""
    mod = sys.modules.get("transformers")
    if mod is not None and getattr(mod, "__file__", None) is not None:
        # 已经是真实模块，无需恢复
        yield
        return

    # 检查真实 transformers 是否已安装
    # 临时移除 mock 以便 find_spec 能找到真实模块
    saved = {}
    for key in list(sys.modules.keys()):
        if key == "transformers" or key.startswith("transformers."):
            saved[key] = sys.modules.pop(key)

    try:
        spec = importlib.util.find_spec("transformers")
    except (ValueError, ModuleNotFoundError):
        spec = None

    if spec is not None:
        try:
            importlib.import_module("transformers")
        except Exception:  # noqa: BLE001
            # 恢复失败，还原 mock
            sys.modules.update(saved)
    else:
        # 真实模块未安装，还原 mock
        sys.modules.update(saved)

    yield


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


# ----------------------------------------------------------------------
# GPT-SoVITS 相关 fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_sovits_tokenizer() -> Any:
    """返回一个 SoVITSTokenizer mock。"""
    frontend = MagicMock()
    frontend.vocab_size = 256
    frontend.model_type = "ar"
    frontend.special_tokens = {
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
        "[SPK]": 4, "[SPLIT]": 5, "[ZH]": 6, "[EN]": 7,
        "[JA]": 8, "[KO]": 9, "[YUE]": 10, "_": 11,
    }
    frontend.tokenize.return_value = [2, 6, 12, 13, 14, 1]
    frontend.detokenize.return_value = "mock 音素"
    frontend.encode_speaker.return_value = None
    frontend.preprocess.side_effect = lambda text: text
    frontend.insert_prosody_tokens.side_effect = lambda text, prosody: text
    frontend._g2p_chinese.return_value = ["n3", "i3", "h3", "ao3"]
    frontend._g2p_english.return_value = ["HH", "AH0", "L", "OW1"]
    frontend._split_pinyin.side_effect = lambda p: [p] if len(p) <= 2 else [p[:1] + p[-1], p[1:-1] + p[-1]]
    frontend.unload_weights = MagicMock()
    return frontend


@pytest.fixture
def mock_gpt2_model() -> Any:
    """返回一个 GPT2ARModel mock。"""
    model = MagicMock()
    model.model_type = "ar"
    model.vocab_size = 256
    model.hidden_size = 768
    model._semantic_vocab_size = 768
    model._num_layers = 12
    model._num_heads = 12
    model._max_position_embeddings = 2048
    model.generate.return_value = [[10, 20, 30, 40, 50, 60, 70, 80]]
    model.generate_stream.return_value = iter([
        [[10, 20, 30, 40]],
        [[50, 60, 70, 80]],
    ])
    model.get_input_embeddings.return_value = MagicMock()
    model.get_output_head.return_value = MagicMock()
    model.unload_weights = MagicMock()
    return model


@pytest.fixture
def mock_sovits_decoder() -> Any:
    """返回一个 SoVITSDecoder mock。"""
    decoder = MagicMock()
    decoder.vocoder_type = "sovits_decoder"
    decoder.input_type = "vq_tokens"
    decoder.sample_rate = 32000
    decoder.ssl_vocab_size = 768
    decoder.hidden_size = 192
    decoder.decode.return_value = ([0.0] * 200, 32000)
    decoder.decode_chunk.return_value = ([0.0] * 200, 32000)
    decoder.set_reference = MagicMock()
    decoder.reset_stream = MagicMock()
    decoder.forward.return_value = {
        "waveform": [[0.0] * 200],
        "mu": [[0.0] * 192 for _ in range(10)],
        "log_var": [[0.0] * 192 for _ in range(10)],
        "z": [[0.0] * 192 for _ in range(10)],
        "z_p": [[0.0] * 192 for _ in range(10)],
        "log_det": [0.0],
    }
    decoder.unload_weights = MagicMock()
    decoder._is_loaded = True
    decoder._impl = MagicMock()
    decoder._impl.hop_length = 256
    return decoder


@pytest.fixture
def sample_phoneme_ids() -> Any:
    """返回模拟的音素 token ids [1, 16]。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randint(4, 256, (1, 16), dtype=torch.long)


@pytest.fixture
def sample_semantic_tokens() -> Any:
    """返回模拟的语义 token ids [1, 32]（SSL 码本 index）。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randint(0, 768, (1, 32), dtype=torch.long)


@pytest.fixture
def sample_speaker_info() -> Any:
    """返回包含 ref_semantic_tokens 和 speaker_embedding 的 dict。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return {
        "ref_semantic_tokens": torch.randint(0, 768, (1, 20), dtype=torch.long),
        "speaker_embedding": torch.randn(1, 768),
    }


@pytest.fixture
def sample_ref_features() -> Any:
    """返回模拟的参考音频 SSL 特征 [1, 50, 768]。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randn(1, 50, 768)


# ----------------------------------------------------------------------
# CosyVoice 相关 fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_cosyvoice_tokenizer() -> Any:
    """返回一个 CosyVoiceTokenizer mock。

    模拟基于 LLM tokenizer 的文本前端行为，包括特殊标记添加
    （[sos]、[flow_token]、[eos]）和文本预处理。
    """
    frontend = MagicMock()
    frontend.vocab_size = 151936  # Qwen2.5 词表大小
    frontend.model_type = "flow_matching"
    frontend.llm_vocab_size = 151936
    frontend.speech_token_size = 6561  # 81*81
    frontend.special_tokens = {
        "[sos]": 151936 + 6561,
        "[eos]": 151936 + 6561 + 1,
        "[flow_token]": 151936 + 6561 + 2,
    }
    # tokenize 返回 token id 列表（模拟 [sos] + text_tokens + [flow_token]）
    frontend.tokenize.return_value = [
        151936 + 6561,        # [sos]
        100, 200, 300, 400,   # text tokens
        151936 + 6561 + 2,    # [flow_token]
    ]
    frontend.detokenize.return_value = "mock CosyVoice 文本"
    frontend.encode_speaker.return_value = None
    frontend.preprocess.side_effect = lambda text: text.strip()
    frontend.unload_weights = MagicMock()
    return frontend


@pytest.fixture
def mock_flow_matching_model() -> Any:
    """返回一个 FlowMatchingModel mock。

    模拟 Flow Matching 声学模型行为：generate 返回 mel spectrogram，
    generate_stream 返回 mel chunk 迭代器。
    """
    model = MagicMock()
    model.model_type = "flow_matching"
    model.acoustic_type = "flow_matching"
    model.mel_bins = 80
    model.hidden_size = 512
    model.cond_dim = 512
    model.num_ode_steps = 10
    model.ode_solver = "euler"
    # generate 返回 mel spectrogram [1, 80, 100]
    model.generate.return_value = MagicMock(
        shape=torch.Size([1, 80, 100]) if (torch := _try_import_torch()) else None
    )
    if _try_import_torch():
        import torch
        model.generate.return_value = torch.randn(1, 80, 100)
        # generate_stream 返回 mel chunk 迭代器
        model.generate_stream.return_value = iter([
            torch.randn(1, 80, 30),
            torch.randn(1, 80, 30),
            torch.randn(1, 80, 30),
        ])
    else:
        model.generate_stream.return_value = iter([None, None, None])
    model.set_ode_params = MagicMock()
    model.unload_weights = MagicMock()
    return model


@pytest.fixture
def mock_speech_tokenizer() -> Any:
    """返回一个 SpeechTokenizer mock。

    模拟 2 层 RVQ 语音 Tokenizer：encode 返回 token ids [1, ref_len]。
    """
    tokenizer = MagicMock()
    tokenizer.codebook_size = 6561  # 81*81
    tokenizer.num_quantizers = 2
    tokenizer.sample_rate = 22050
    if _try_import_torch():
        import torch
        tokenizer.encode.return_value = torch.randint(
            0, 6561, (1, 30), dtype=torch.long
        )
        tokenizer.decode.return_value = torch.randn(1, 80, 50)
    else:
        tokenizer.encode.return_value = [100, 200, 300]
        tokenizer.decode.return_value = None
    tokenizer.unload_weights = MagicMock()
    return tokenizer


@pytest.fixture
def mock_speaker_encoder() -> Any:
    """返回一个 SpeakerEncoder mock。

    模拟 ECAPA-TDNN 说话人编码器：encode 返回嵌入向量 [1, embedding_dim]。
    """
    encoder = MagicMock()
    encoder.embedding_dim = 192
    encoder.sample_rate = 16000
    if _try_import_torch():
        import torch
        encoder.encode.return_value = torch.randn(1, 192)
    else:
        encoder.encode.return_value = [0.0] * 192
    encoder.unload_weights = MagicMock()
    return encoder


@pytest.fixture
def sample_text_features() -> Any:
    """返回模拟的 LLM 输出文本特征 [1, text_len, feat_dim]。

    模拟经过 LLM 编码和 text_projection 后的特征，用于 Flow Matching 的条件输入。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randn(1, 20, 512)


@pytest.fixture
def sample_condition() -> Any:
    """返回模拟的融合条件特征 [1, cond_len, cond_dim]。

    模拟 text_feats + ref_speech_feats + speaker_embedding 融合后的条件，
    用于 FlowEstimator 的交叉注意力输入。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randn(1, 30, 512)


@pytest.fixture
def sample_mel_from_flow() -> Any:
    """返回模拟的 Flow Matching 输出 mel spectrogram [1, 80, 100]。

    模拟 ODE 求解完成后输出的 mel，用于 HiFi-GAN 声码器解码。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch 未安装")
    import torch

    return torch.randn(1, 80, 100)


def _try_import_torch() -> Any:
    """安全地尝试导入 torch，失败返回 None。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    return torch
