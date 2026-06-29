"""测试 ChatTTSWeightConverter。

这些用例不依赖 transformers / 真实权重，仅验证转换器的类结构、映射表、
继承的格式检测，以及对不存在路径的错误处理。
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.weights.chattts_convert import (
    ChatTTSWeightConverter,
)


def test_CVT_01() -> None:
    """构造成功。"""
    converter = ChatTTSWeightConverter()
    assert converter is not None


def test_CVT_02() -> None:
    """COMPONENTS 包含 text_frontend, acoustic_model, vocoder, dvae。"""
    components = ChatTTSWeightConverter.COMPONENTS
    assert "text_frontend" in components
    assert "acoustic_model" in components
    assert "vocoder" in components
    assert "dvae" in components


def test_CVT_03() -> None:
    """GPT_TO_LLAMA_MAP 包含关键映射。"""
    mapping = ChatTTSWeightConverter.GPT_TO_LLAMA_MAP
    assert "gpt.model.embed_tokens.weight" in mapping
    assert "gpt.model.norm.weight" in mapping
    assert "gpt.lm_head.weight" in mapping
    # 映射目标应为 LlamaForCausalLM 的标准命名
    assert mapping["gpt.model.embed_tokens.weight"] == "model.embed_tokens.weight"
    assert mapping["gpt.lm_head.weight"] == "lm_head.weight"


def test_CVT_04(tmp_path: Any) -> None:
    """list_formats 继承正常工作。"""
    # 在临时目录中放置一个 .safetensors 文件，验证继承的格式检测
    (tmp_path / "dummy.safetensors").write_bytes(b"\x00")
    formats = ChatTTSWeightConverter.list_formats(str(tmp_path))
    assert isinstance(formats, list)
    assert "safetensors_dir" in formats
    # 不存在的路径返回空列表
    assert ChatTTSWeightConverter.list_formats("/nonexistent/path") == []


def test_CVT_05() -> None:
    """dry_run 不存在的路径抛出 FileNotFoundError。"""
    converter = ChatTTSWeightConverter()
    with pytest.raises(FileNotFoundError):
        converter.dry_run("/nonexistent/path/to/weights.pt")
