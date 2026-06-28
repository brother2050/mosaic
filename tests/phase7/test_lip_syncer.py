# tests/phase7/test_lip_syncer.py
"""LipSyncer 口型同步节点测试。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import AudioData, MosaicData, VideoData
from mosaic.nodes.digital_human.lip_syncer import LipSyncer


# ============================================================================
# Session 级 mock 注入：musetalk / sadtalker / wav2lip
# ============================================================================
@pytest.fixture(scope="session", autouse=True)
def _mock_musetalk() -> None:
    """Mock musetalk 模块（session 级别）。"""
    mock = MagicMock()
    mock.MuseTalk.from_pretrained.return_value = MagicMock()
    mock.MuseTalk.from_pretrained.return_value.to.return_value = (
        mock.MuseTalk.from_pretrained.return_value
    )
    sys.modules["musetalk"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_sadtalker() -> None:
    """Mock sadtalker 模块（session 级别）。"""
    mock = MagicMock()
    mock.SadTalker.return_value = MagicMock()
    sys.modules["sadtalker"] = mock


@pytest.fixture(scope="session", autouse=True)
def _mock_wav2lip() -> None:
    """Mock wav2lip 模块（session 级别）。"""
    mock = MagicMock()
    mock.Wav2Lip.from_pretrained.return_value = MagicMock()
    mock.Wav2Lip.from_pretrained.return_value.to.return_value = (
        mock.Wav2Lip.from_pretrained.return_value
    )
    sys.modules["wav2lip"] = mock


# ============================================================================
# T_LIP_01: 基本口型同步
# ============================================================================
def test_lip_syncer_basic(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_01: 基本口型同步（单张图片 + 音频），输出 VideoData"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    # Mock _sync_mouth 返回与输入相同的 face_crop（绕过模型推理）
    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio=sample_short_audio,
        ))

    assert isinstance(result, MosaicData), "结果应为 MosaicData"
    assert "video" in result, "输出应包含 'video'"
    assert isinstance(result["video"], VideoData), "video 应为 VideoData"
    assert "audio" in result, "输出应包含 'audio'"
    assert isinstance(result["audio"], AudioData), "audio 应为 AudioData"
    syncer.unload()


# ============================================================================
# T_LIP_02: 输出时长与音频时长匹配
# ============================================================================
def test_lip_syncer_duration_matches_audio(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_02: 输出时长与音频时长匹配（2 秒音频 → 50 帧 @ 25fps）"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio=sample_short_audio,
            fps=25,
        ))

    video = result["video"]
    expected_frames = 50  # 2s * 25fps
    assert video.metadata["frame_count"] == expected_frames, (
        f"帧数应为 {expected_frames}，实际 {video.metadata['frame_count']}"
    )
    assert len(video.frames) == expected_frames, (
        f"frames 列表长度应为 {expected_frames}，实际 {len(video.frames)}"
    )
    # 时长约为 2.0 秒
    assert abs(result["duration"] - 2.0) < 0.1, f"时长应约为 2.0s，实际 {result['duration']}"
    syncer.unload()


# ============================================================================
# T_LIP_03: fps 参数生效
# ============================================================================
def test_lip_syncer_fps_parameter(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_03: fps 参数生效（fps=30 → 60 帧）"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio=sample_short_audio,
            fps=30,
        ))

    assert result["fps"] == 30, f"输出 fps 应为 30，实际 {result['fps']}"
    assert result["video"].fps == 30, f"VideoData fps 应为 30，实际 {result['video'].fps}"
    expected_frames = 60  # 2s * 30fps
    assert result["video"].metadata["frame_count"] == expected_frames, (
        f"帧数应为 {expected_frames}，实际 {result['video'].metadata['frame_count']}"
    )
    syncer.unload()


# ============================================================================
# T_LIP_04: output_format="frames" 输出帧列表
# ============================================================================
def test_lip_syncer_output_format_frames(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_04: output_format="frames" 输出帧列表"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio=sample_short_audio,
            output_format="frames",
        ))

    assert "frames" in result, "output_format='frames' 时输出应包含 'frames'"
    assert "video" not in result, "output_format='frames' 时不应包含 'video'"
    assert isinstance(result["frames"], list), "frames 应为列表"
    assert len(result["frames"]) > 0, "frames 不应为空"
    for f in result["frames"]:
        assert isinstance(f, Image.Image), f"每帧应为 PIL.Image，实际 {type(f)}"
    syncer.unload()


# ============================================================================
# T_LIP_05: padding 参数生效
# ============================================================================
def test_lip_syncer_padding(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_05: padding 参数生效（传入不同 padding 值）"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    # 测试不同的 padding 值
    for padding in [[10, 10, 10, 10], [0, 30, 0, 30], [5, 5, 5, 5]]:
        with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
            result = syncer.run(MosaicData(
                face_image=sample_avatar_image,
                audio=sample_short_audio,
                padding=padding,
            ))
        assert "video" in result, f"padding={padding} 应正常输出 video"
        assert isinstance(result["video"], VideoData)

    syncer.unload()


# ============================================================================
# T_LIP_06: 从文件路径输入图片
# ============================================================================
def test_lip_syncer_image_from_path(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler, tmp_path,
):
    """# T_LIP_06: 从文件路径输入图片"""
    image_path = tmp_path / "face.png"
    sample_avatar_image.save(str(image_path))

    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=str(image_path),
            audio=sample_short_audio,
        ))

    assert "video" in result, "从文件路径加载图片应正常输出"
    assert isinstance(result["video"], VideoData)
    syncer.unload()


# ============================================================================
# T_LIP_07: 从文件路径输入音频
# ============================================================================
def test_lip_syncer_audio_from_path(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_07: 从文件路径输入音频（传入文件路径字符串）"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    # Mock _load_audio_signal 以绕过 soundfile/librosa 依赖
    def _mock_load_audio(self, audio):
        return (sample_short_audio.waveform, sample_short_audio.sample_rate)

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync), \
         patch.object(LipSyncer, "_load_audio_signal", _mock_load_audio):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio="/fake/path/speech.wav",
        ))

    assert "video" in result, "从文件路径加载音频应正常输出"
    assert isinstance(result["video"], VideoData)
    syncer.unload()


# ============================================================================
# T_LIP_08: 输入多帧视频时逐帧处理
# ============================================================================
def test_lip_syncer_multi_frame_input(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_08: 输入多帧视频时逐帧处理（传入帧列表）"""
    # 创建 3 帧的图像列表
    frames = [
        sample_avatar_image.copy(),
        Image.new("RGB", (512, 512), (200, 200, 210)),
        sample_avatar_image.copy(),
    ]

    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=frames,
            audio=sample_short_audio,
            fps=25,
        ))

    assert "video" in result, "多帧输入应正常输出"
    video = result["video"]
    assert len(video.frames) == 50, f"应输出 50 帧（2s * 25fps），实际 {len(video.frames)}"
    syncer.unload()


# ============================================================================
# T_LIP_09: method 参数切换
# ============================================================================
def test_lip_syncer_method_switching(fresh_bus, cpu_scheduler):
    """# T_LIP_09: method 参数切换（musetalk / wav2lip / sadtalker）"""
    methods = ["musetalk", "wav2lip", "sadtalker"]
    specs = []

    for method in methods:
        syncer = LipSyncer(
            method=method, device="cpu", dtype="float32",
            bus=fresh_bus, scheduler=cpu_scheduler,
        )
        spec = syncer.describe()
        specs.append(spec)
        assert spec.name == "lip-syncer", f"method={method}: name 应为 'lip-syncer'"
        assert spec.domain == "digital_human", f"method={method}: domain 应为 'digital_human'"
        assert "model_info" in spec.to_dict(), f"method={method}: 应包含 model_info"

    assert len(specs) == 3, f"应有 3 个 spec，实际 {len(specs)}"


# ============================================================================
# T_LIP_10: 输出包含 audio 方便后续合并
# ============================================================================
def test_lip_syncer_output_includes_audio(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_LIP_10: 输出包含 audio 方便后续合并"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    syncer.load()

    def _mock_sync(self, face_crop, waveform, sample_rate, frame_idx, fps, parsing_mode):
        return face_crop

    with patch.object(LipSyncer, "_sync_mouth", _mock_sync):
        result = syncer.run(MosaicData(
            face_image=sample_avatar_image,
            audio=sample_short_audio,
        ))

    assert "audio" in result, "输出应包含 'audio'"
    out_audio = result["audio"]
    assert isinstance(out_audio, AudioData), "audio 应为 AudioData"
    assert out_audio.sample_rate == sample_short_audio.sample_rate, (
        f"采样率应为 {sample_short_audio.sample_rate}"
    )
    syncer.unload()


# ============================================================================
# T_LIP_11: describe 返回正确信息
# ============================================================================
def test_lip_syncer_describe(fresh_bus, cpu_scheduler):
    """# T_LIP_11: describe 返回正确信息"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    spec = syncer.describe()

    assert spec.name == "lip-syncer"
    assert spec.domain == "digital_human"
    assert "version" in spec.to_dict()

    model_info = spec.to_dict().get("model_info", {})
    assert "vram_gb" in model_info, "model_info 应包含 vram_gb"
    assert "license" in model_info, "model_info 应包含 license"
    assert "method" in model_info, "model_info 应包含 method"
    assert model_info["method"] == "musetalk"
    assert isinstance(model_info["vram_gb"], (int, float)), "vram_gb 应为数值"
    assert model_info["vram_gb"] > 0, "vram_gb 应大于 0"


# ============================================================================
# T_LIP_12: load/unload 后 is_loaded 状态正确
# ============================================================================
def test_lip_syncer_load_unload(fresh_bus, cpu_scheduler):
    """# T_LIP_12: load/unload 后 is_loaded 状态正确"""
    syncer = LipSyncer(
        method="musetalk", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )

    assert not syncer.is_loaded(), "初始状态应为未加载"
    syncer.load()
    assert syncer.is_loaded(), "load() 后应为已加载"
    syncer.unload()
    assert not syncer.is_loaded(), "unload() 后应为未加载"

    # 重复 load/unload 不应出错
    syncer.load()
    assert syncer.is_loaded()
    syncer.unload()
    assert not syncer.is_loaded()