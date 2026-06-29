# tests/final/test_weight_converter.py
"""TTS 权重转换框架测试。

验证各 TTS 后端的权重转换器（WeightConverter 及其子类）的基本功能：
- 实例化检查
- 类型继承关系检查
- 基类不可直接实例化
- list_formats 静态方法
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mosaic.nodes.audio.tts_backends.weights.converter import WeightConverter
from mosaic.nodes.audio.tts_backends.weights.chattts_convert import (
    ChatTTSWeightConverter,
)
from mosaic.nodes.audio.tts_backends.weights.fish_convert import (
    FishWeightConverter,
)
from mosaic.nodes.audio.tts_backends.weights.sovits_convert import (
    SoVITSWeightConverter,
)
from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
    CosyVoiceWeightConverter,
)


# ===========================================================================
# T_WCONV_01: ChatTTSWeightConverter 可实例化
# ===========================================================================
def test_chattts_weight_converter_instantiation():
    """T_WCONV_01: ChatTTSWeightConverter 可正常实例化，且是 WeightConverter 的子类。"""
    conv = ChatTTSWeightConverter()
    assert isinstance(conv, ChatTTSWeightConverter), (
        f"ChatTTSWeightConverter 实例的类型应为 ChatTTSWeightConverter，"
        f"实际为 {type(conv).__name__}"
    )
    assert isinstance(conv, WeightConverter), (
        f"ChatTTSWeightConverter 实例应是 WeightConverter 的子类"
    )


# ===========================================================================
# T_WCONV_02: FishWeightConverter 可实例化
# ===========================================================================
def test_fish_weight_converter_instantiation():
    """T_WCONV_02: FishWeightConverter 可正常实例化。"""
    conv = FishWeightConverter()
    assert isinstance(conv, FishWeightConverter), (
        f"FishWeightConverter 实例的类型应为 FishWeightConverter，"
        f"实际为 {type(conv).__name__}"
    )
    assert isinstance(conv, WeightConverter), (
        f"FishWeightConverter 实例应是 WeightConverter 的子类"
    )


# ===========================================================================
# T_WCONV_03: SoVITSWeightConverter 可实例化
# ===========================================================================
def test_sovits_weight_converter_instantiation():
    """T_WCONV_03: SoVITSWeightConverter 可正常实例化。"""
    conv = SoVITSWeightConverter()
    assert isinstance(conv, SoVITSWeightConverter), (
        f"SoVITSWeightConverter 实例的类型应为 SoVITSWeightConverter，"
        f"实际为 {type(conv).__name__}"
    )
    assert isinstance(conv, WeightConverter), (
        f"SoVITSWeightConverter 实例应是 WeightConverter 的子类"
    )


# ===========================================================================
# T_WCONV_04: CosyVoiceWeightConverter 可实例化
# ===========================================================================
def test_cosyvoice_weight_converter_instantiation():
    """T_WCONV_04: CosyVoiceWeightConverter 可正常实例化。"""
    conv = CosyVoiceWeightConverter()
    assert isinstance(conv, CosyVoiceWeightConverter), (
        f"CosyVoiceWeightConverter 实例的类型应为 CosyVoiceWeightConverter，"
        f"实际为 {type(conv).__name__}"
    )
    assert isinstance(conv, WeightConverter), (
        f"CosyVoiceWeightConverter 实例应是 WeightConverter 的子类"
    )


# ===========================================================================
# T_WCONV_05: WeightConverter 基类不可直接实例化
# ===========================================================================
def test_weight_converter_base_cannot_instantiate():
    """T_WCONV_05: WeightConverter 是 ABC，不应允许直接实例化。

    WeightConverter 有抽象方法 convert() 和 validate()，
    直接实例化应抛出 TypeError。
    """
    with pytest.raises(TypeError, match=r"(abstract|instantiate|Can't)"):
        WeightConverter()


# ===========================================================================
# T_WCONV_06: list_formats 静态方法可用
# ===========================================================================
def test_list_formats_static_method():
    """T_WCONV_06: WeightConverter.list_formats() 静态方法可检测权重格式。

    创建一个临时目录并在其中放置 .safetensors 文件，
    然后调用 list_formats() 验证能检测到格式。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 dummy .safetensors 文件
        safetensors_path = os.path.join(tmpdir, "model.safetensors")
        with open(safetensors_path, "w") as f:
            f.write("dummy")

        formats = WeightConverter.list_formats(tmpdir)
        assert isinstance(formats, list), (
            f"list_formats 应返回 list，实际返回 {type(formats).__name__}"
        )
        assert "safetensors_dir" in formats, (
            f"list_formats 应检测到 safetensors_dir 格式，"
            f"实际检测到: {formats}"
        )

    # 测试空目录
    with tempfile.TemporaryDirectory() as tmpdir:
        formats = WeightConverter.list_formats(tmpdir)
        assert isinstance(formats, list)
        # 空目录应返回空列表
        assert len(formats) == 0, (
            f"空目录应返回空列表，实际返回: {formats}"
        )

    # 测试文件路径（.safetensors 文件）
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors_path = os.path.join(tmpdir, "model.safetensors")
        with open(safetensors_path, "w") as f:
            f.write("dummy")

        formats = WeightConverter.list_formats(safetensors_path)
        assert isinstance(formats, list)
        assert "safetensors" in formats, (
            f"对 .safetensors 文件调用 list_formats 应检测到 safetensors 格式，"
            f"实际检测到: {formats}"
        )

    # 测试 .pt 文件路径
    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, "model.pt")
        with open(pt_path, "w") as f:
            f.write("dummy")

        formats = WeightConverter.list_formats(pt_path)
        assert isinstance(formats, list)
        assert "pytorch" in formats, (
            f"对 .pt 文件调用 list_formats 应检测到 pytorch 格式，"
            f"实际检测到: {formats}"
        )

    # 测试包含 .pt 文件的目录
    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, "model.pt")
        with open(pt_path, "w") as f:
            f.write("dummy")

        formats = WeightConverter.list_formats(tmpdir)
        assert isinstance(formats, list)
        assert "pytorch" in formats, (
            f"对包含 .pt 文件的目录调用 list_formats 应检测到 pytorch 格式，"
            f"实际检测到: {formats}"
        )