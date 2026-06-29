"""测试 FishWeightConverter。

这些用例不依赖 transformers / 真实权重，仅验证转换器的类结构、映射表、
继承的格式检测，以及对不存在路径的错误处理。
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.weights.fish_convert import FishWeightConverter


def test_FCVT_01() -> None:
    """构造成功。"""
    converter = FishWeightConverter()
    assert converter is not None


def test_FCVT_02() -> None:
    """COMPONENTS 包含 text_frontend, acoustic_model, vocoder, vq_decoder, audio_encoder。"""
    components = FishWeightConverter.COMPONENTS
    assert "text_frontend" in components
    assert "acoustic_model" in components
    assert "vocoder" in components
    assert "vq_decoder" in components
    assert "audio_encoder" in components


def test_FCVT_03() -> None:
    """FISH_TO_LLAMA_MAP 包含关键映射。"""
    mapping = FishWeightConverter.FISH_TO_LLAMA_MAP
    assert "model.embed_tokens.weight" in mapping
    assert "model.norm.weight" in mapping
    assert "lm_head.weight" in mapping
    # 映射目标应为 LlamaForCausalLM 的标准命名（Fish Speech 使用恒等映射）
    assert mapping["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    assert mapping["lm_head.weight"] == "lm_head.weight"


def test_FCVT_04() -> None:
    """list_formats 继承正常工作。"""
    # /tmp 是一个真实目录，list_formats 应返回 list 类型
    formats = FishWeightConverter.list_formats("/tmp")
    assert isinstance(formats, list)


def test_FCVT_05() -> None:
    """dry_run 不存在的路径抛出 FileNotFoundError 或 OSError。"""
    converter = FishWeightConverter()
    with pytest.raises((FileNotFoundError, OSError)):
        converter.dry_run("/nonexistent/path/to/weights.pt")
