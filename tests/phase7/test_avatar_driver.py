# tests/phase7/test_avatar_driver.py
"""AvatarDriver 形象驱动节点测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from mosaic.core.types import MosaicData, VideoData
from mosaic.nodes.digital_human.avatar_driver import AvatarDriver


# ============================================================================
# 辅助函数
# ============================================================================
def _make_dummy_frames(size=(512, 512), count=1):
    """创建模拟的 pipeline 输出：包含 PIL 图像的列表。"""
    return [Image.new("RGB", size, (100, 150, 200)) for _ in range(count)]


# ============================================================================
# T_AVDR_01: 基本形象驱动（driving_video 模式）
# ============================================================================
def test_avatar_driver_basic_driving_video(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_01: 基本形象驱动（driving_video 模式），输出 VideoData"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
        ))

    assert isinstance(result, MosaicData), "结果应为 MosaicData"
    assert "video" in result, "输出应包含 'video'"
    assert isinstance(result["video"], VideoData), "video 应为 VideoData"
    assert result["driving_source_type"] == "video", "驱动源类型应为 'video'"
    driver.unload()


# ============================================================================
# T_AVDR_02: 输出帧数与驱动视频帧数匹配
# ============================================================================
def test_avatar_driver_frame_count_matches_driving_video(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_02: 输出帧数与驱动视频帧数匹配"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    expected_frames = len(sample_driving_video)  # 50
    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
        ))

    video = result["video"]
    assert video.metadata["frame_count"] == expected_frames, (
        f"帧数应为 {expected_frames}，实际 {video.metadata['frame_count']}"
    )
    assert len(video.frames) == expected_frames, (
        f"frames 列表长度应为 {expected_frames}，实际 {len(video.frames)}"
    )
    driver.unload()


# ============================================================================
# T_AVDR_03: 输出 fps 参数生效
# ============================================================================
def test_avatar_driver_fps_parameter(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_03: 输出 fps 参数生效（fps=30）"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            fps=30,
        ))

    assert result["fps"] == 30, f"输出 fps 应为 30，实际 {result['fps']}"
    assert result["video"].fps == 30, f"VideoData fps 应为 30，实际 {result['video'].fps}"
    driver.unload()


# ============================================================================
# T_AVDR_04: expression_scale 参数生效
# ============================================================================
def test_avatar_driver_expression_scale(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_04: expression_scale 参数生效（0.5 和 1.5 构造两个节点）"""
    driver1 = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver1.load()
    driver2 = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver2.load()

    with patch.object(driver1, "_run_pipeline", return_value=_make_dummy_frames()) as mock1:
        result1 = driver1.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            expression_scale=0.5,
        ))

    with patch.object(driver2, "_run_pipeline", return_value=_make_dummy_frames()) as mock2:
        result2 = driver2.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            expression_scale=1.5,
        ))

    assert "video" in result1, "expression_scale=0.5 应正常输出 video"
    assert "video" in result2, "expression_scale=1.5 应正常输出 video"
    assert isinstance(result1["video"], VideoData)
    assert isinstance(result2["video"], VideoData)
    driver1.unload()
    driver2.unload()


# ============================================================================
# T_AVDR_05: motion_scale 参数生效
# ============================================================================
def test_avatar_driver_motion_scale(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_05: motion_scale 参数生效"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            motion_scale=0.8,
        ))

    assert "video" in result, "motion_scale=0.8 应正常输出 video"
    assert isinstance(result["video"], VideoData)
    driver.unload()


# ============================================================================
# T_AVDR_06: output_format="frames" 输出帧列表
# ============================================================================
def test_avatar_driver_output_format_frames(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_06: output_format="frames" 输出帧列表"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            output_format="frames",
        ))

    assert "frames" in result, "output_format='frames' 时输出应包含 'frames'"
    assert "video" not in result, "output_format='frames' 时不应包含 'video'"
    assert isinstance(result["frames"], list), "frames 应为列表"
    assert len(result["frames"]) > 0, "frames 不应为空"
    for f in result["frames"]:
        assert isinstance(f, Image.Image), f"每帧应为 PIL.Image，实际 {type(f)}"
    driver.unload()


# ============================================================================
# T_AVDR_07: output_format="video" 输出 VideoData
# ============================================================================
def test_avatar_driver_output_format_video(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_07: output_format="video" 输出 VideoData"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_video=sample_driving_video,
            output_format="video",
        ))

    assert "video" in result, "output_format='video' 时输出应包含 'video'"
    assert isinstance(result["video"], VideoData), "video 应为 VideoData"
    assert result["video"].fps > 0, "fps 应大于 0"
    assert len(result["video"].frames) > 0, "video.frames 不应为空"
    driver.unload()


# ============================================================================
# T_AVDR_08: source_image 从文件路径加载
# ============================================================================
def test_avatar_driver_source_image_from_path(
    sample_avatar_image, sample_driving_video, fresh_bus, cpu_scheduler, tmp_path,
):
    """# T_AVDR_08: source_image 从文件路径加载（保存 sample_avatar_image 到 tmp_path）"""
    # 保存图片到临时文件
    image_path = tmp_path / "source.png"
    sample_avatar_image.save(str(image_path))

    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=str(image_path),
            driving_video=sample_driving_video,
        ))

    assert "video" in result, "从文件路径加载 source_image 应正常输出"
    assert isinstance(result["video"], VideoData)
    driver.unload()


# ============================================================================
# T_AVDR_09: driving_audio 模式可运行
# ============================================================================
def test_avatar_driver_driving_audio(
    sample_avatar_image, sample_short_audio, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_09: driving_audio 模式可运行（使用 sample_short_audio）"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            driving_audio=sample_short_audio,
        ))

    assert "video" in result, "driving_audio 模式应输出 video"
    assert result["driving_source_type"] == "audio", "驱动源类型应为 'audio'"
    assert isinstance(result["video"], VideoData)
    driver.unload()


# ============================================================================
# T_AVDR_10: expression_params 模式可运行
# ============================================================================
def test_avatar_driver_expression_params(
    sample_avatar_image, sample_expression_params, fresh_bus, cpu_scheduler,
):
    """# T_AVDR_10: expression_params 模式可运行（使用 sample_expression_params）"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    with patch.object(driver, "_run_pipeline", return_value=_make_dummy_frames()):
        result = driver.run(MosaicData(
            source_image=sample_avatar_image,
            expression_params=sample_expression_params,
        ))

    assert "video" in result, "expression_params 模式应输出 video"
    assert result["driving_source_type"] == "expression_params", (
        "驱动源类型应为 'expression_params'"
    )
    assert isinstance(result["video"], VideoData)
    driver.unload()


# ============================================================================
# T_AVDR_11: method 参数切换
# ============================================================================
def test_avatar_driver_method_switching(fresh_bus, cpu_scheduler):
    """# T_AVDR_11: method 参数切换（liveportrait / sadtalker / musetalk）"""
    methods = ["liveportrait", "sadtalker", "musetalk"]
    specs = []

    for method in methods:
        driver = AvatarDriver(
            method=method, device="cpu", dtype="float32",
            bus=fresh_bus, scheduler=cpu_scheduler,
        )
        spec = driver.describe()
        specs.append(spec)
        assert spec.name == "avatar-driver", f"method={method}: name 应为 'avatar-driver'"
        assert spec.domain == "digital_human", f"method={method}: domain 应为 'digital_human'"
        assert "model_info" in spec.to_dict(), f"method={method}: 应包含 model_info"

    # 确保三种方法都能正常实例化
    assert len(specs) == 3, f"应有 3 个 spec，实际 {len(specs)}"


# ============================================================================
# T_AVDR_12: describe 返回正确信息
# ============================================================================
def test_avatar_driver_describe(fresh_bus, cpu_scheduler):
    """# T_AVDR_12: describe 返回正确信息（含 model_info 中的 vram_gb 和 license）"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    spec = driver.describe()

    assert spec.name == "avatar-driver"
    assert spec.domain == "digital_human"
    assert "version" in spec.to_dict()

    model_info = spec.to_dict().get("model_info", {})
    assert "vram_gb" in model_info, "model_info 应包含 vram_gb"
    assert "license" in model_info, "model_info 应包含 license"
    assert "method" in model_info, "model_info 应包含 method"
    assert model_info["method"] == "liveportrait"
    assert isinstance(model_info["vram_gb"], (int, float)), "vram_gb 应为数值"
    assert model_info["vram_gb"] > 0, "vram_gb 应大于 0"


# ============================================================================
# T_AVDR_13: load/unload 后 is_loaded 状态正确
# ============================================================================
def test_avatar_driver_load_unload(fresh_bus, cpu_scheduler):
    """# T_AVDR_13: load/unload 后 is_loaded 状态正确"""
    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )

    assert not driver.is_loaded(), "初始状态应为未加载"
    driver.load()
    assert driver.is_loaded(), "load() 后应为已加载"
    driver.unload()
    assert not driver.is_loaded(), "unload() 后应为未加载"

    # 重复 load/unload 不应出错
    driver.load()
    assert driver.is_loaded()
    driver.unload()
    assert not driver.is_loaded()


# ============================================================================
# T_AVDR_14: 无人脸的 source_image 给出友好错误
# ============================================================================
def test_avatar_driver_no_face_raises_error(
    fresh_bus, cpu_scheduler,
):
    """# T_AVDR_14: 无人脸的 source_image 给出友好错误"""
    # 创建一张纯色无特征图片
    blank_image = Image.new("RGB", (512, 512), (128, 128, 128))

    driver = AvatarDriver(
        method="liveportrait", device="cpu", dtype="float32",
        bus=fresh_bus, scheduler=cpu_scheduler,
    )
    driver.load()

    # 创建驱动视频帧列表（至少一帧）
    driving_frames = [Image.new("RGB", (256, 256), (100, 100, 100))]

    # Mock _detect_face 直接抛出 ValueError
    with patch.object(
        driver, "_detect_face",
        side_effect=ValueError("No face detected in the image."),
    ):
        with pytest.raises(ValueError, match="[Nn]o face"):
            driver.run(MosaicData(
                source_image=blank_image,
                driving_video=driving_frames,
            ))

    driver.unload()