# tests/tts/test_cosyvoice_weight_converter.py
"""CosyVoice 权重转换器测试。

测试 CosyVoiceWeightConverter 将官方 CosyVoice 检查点拆分为
text_frontend / flow_matching / speech_tokenizer / speaker_encoder / vocoder
五个组件，并以 safetensors + JSON 配置形式落盘的能力。

CosyVoice 的权重拆分与 GPT-SoVITS 不同：
- text_frontend 组件只引用 LLM 路径（写入 JSON 配置，不复制权重）；
- flow_matching 组件保留 estimator.* / text_proj.* 等声学参数；
- speech_tokenizer / speaker_encoder / vocoder 各自按前缀抽取并重命名。

所有测试使用 ``torch`` 伪造检查点，``torch`` 在函数内部局部导入。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import pytest

sys.path.insert(0, "/workspace/mosaic")

_check = importlib.util.find_spec("torch")
_safetensors_check = importlib.util.find_spec("safetensors")
pytestmark = pytest.mark.skipif(
    _check is None or _safetensors_check is None,
    reason="torch/safetensors 未安装",
)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _make_fake_checkpoint(directory: str) -> str:
    """在 ``directory`` 下生成伪造 CosyVoice 检查点，返回检查点路径。

    伪造的 state_dict 覆盖全部五个组件所需的前缀：
    - flow_matching: estimator.* / text_proj.*
    - speech_tokenizer: quantizer.* / encoder.*
    - speaker_encoder: speaker_encoder.* / encoder.*
    - vocoder: generator.*

    ``torch`` 在本函数内部局部导入。``estimator.in_proj.weight`` 使用 3D 形状
    ``[64, 80, 1]`` 以匹配 ``nn.Conv1d``，确保 T_CWC_05 加载 FlowMatchingModel
    时不会因形状不匹配而报错。
    """
    import torch

    state_dict = {
        # flow_matching 组件
        "estimator.in_proj.weight": torch.randn(64, 80, 1),
        "estimator.in_proj.bias": torch.randn(64),
        "estimator.blocks.0.self_attn.in_proj_weight": torch.randn(192, 64),
        "text_proj.weight": torch.randn(64, 64),
        "text_proj.bias": torch.randn(64),
        # speech_tokenizer 组件
        "quantizer.codebook": torch.randn(81, 64),
        "encoder.conv.weight": torch.randn(64, 1, 3),
        # speaker_encoder 组件（encoder. 前缀同时被 speaker_encoder 抽取）
        "speaker_encoder.proj.weight": torch.randn(192, 384),
        # vocoder 组件
        "generator.conv_pre.weight": torch.randn(128, 80, 7),
    }
    path = os.path.join(directory, "cosyvoice.pt")
    torch.save(state_dict, path)
    return path


# ----------------------------------------------------------------------
# T_CWC_01 ~ T_CWC_06
# ----------------------------------------------------------------------
class TestCosyVoiceWeightConverter:
    """CosyVoice 权重转换器测试。"""

    def test_T_CWC_01(self, tmp_path: Any) -> None:
        """T_CWC_01：convert 基本流程，返回组件→文件路径的 dict。"""
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))
        out_dir = str(tmp_path / "converted")

        converter = CosyVoiceWeightConverter()
        result = converter.convert(src, out_dir)

        assert isinstance(result, dict)
        assert len(result) > 0
        # 应至少包含核心组件
        assert "flow_matching" in result

    def test_T_CWC_02(self, tmp_path: Any) -> None:
        """T_CWC_02：validate 验证转换后的权重（仅验证可运行、返回 bool）。"""
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))
        out_dir = str(tmp_path / "converted")

        converter = CosyVoiceWeightConverter()
        converter.convert(src, out_dir)

        result = converter.validate(out_dir)
        assert isinstance(result, bool)

    def test_T_CWC_03(self, tmp_path: Any) -> None:
        """T_CWC_03：flow_matching 组件权重正确映射，落盘 flow_matching.safetensors。"""
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))
        out_dir = str(tmp_path / "converted")

        converter = CosyVoiceWeightConverter()
        result = converter.convert(src, out_dir)

        fm_path = result["flow_matching"]
        assert fm_path.endswith("flow_matching.safetensors")
        assert os.path.isfile(fm_path)

    def test_T_CWC_04(self, tmp_path: Any) -> None:
        """T_CWC_04：text_frontend 组件只引用 LLM（JSON 配置，非 safetensors）。"""
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))
        out_dir = str(tmp_path / "converted")

        converter = CosyVoiceWeightConverter()
        result = converter.convert(src, out_dir)

        tf_path = result["text_frontend"]
        assert tf_path.endswith("text_frontend_config.json")
        assert os.path.isfile(tf_path)
        # 不应为 safetensors 权重文件
        assert not tf_path.endswith(".safetensors")

    def test_T_CWC_05(self, tmp_path: Any) -> None:
        """T_CWC_05：转换后的 flow_matching 权重可被 FlowMatchingModel 加载。

        使用与伪造检查点匹配的小尺寸（hidden_size=64, cond_dim=64,
        num_layers=1, num_heads=2）构造 FlowMatchingModel，调用 load_weights
        验证不崩溃且标记为已加载。
        """
        from mosaic.nodes.audio.tts_backends.acoustic_models.flow_matching import (
            FlowMatchingModel,
        )
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))
        out_dir = str(tmp_path / "converted")

        converter = CosyVoiceWeightConverter()
        converter.convert(src, out_dir)

        model = FlowMatchingModel(
            model_path=out_dir,
            in_channels=80,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            condition_dim=64,
        )
        model.load_weights(out_dir, device="cpu", dtype="float32")
        assert model._is_loaded is True

    def test_T_CWC_06(self, tmp_path: Any) -> None:
        """T_CWC_06：dry_run 模式返回预览映射 dict（不落盘）。"""
        from mosaic.nodes.audio.tts_backends.weights.cosyvoice_convert import (
            CosyVoiceWeightConverter,
        )

        src = _make_fake_checkpoint(str(tmp_path))

        converter = CosyVoiceWeightConverter()
        result = converter.dry_run(src)

        assert isinstance(result, dict)
        assert len(result) > 0
        # dry_run 返回 {组件: {源 key: 目标 key}} 映射
        assert "flow_matching" in result
        assert isinstance(result["flow_matching"], dict)
        # dry_run 不应创建 safetensors 文件
        safetensors_files = [
            f for f in os.listdir(str(tmp_path)) if f.endswith(".safetensors")
        ]
        assert len(safetensors_files) == 0
