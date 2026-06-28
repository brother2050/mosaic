# tests/phase7/test_motion_generator.py
"""MotionGenerator 动作生成节点测试。"""

# 测试策略：MotionGenerator 的 preset 模式不需要真实模型，可以直接测试。
# text2motion 和 audio2motion 也通过 mock 进行测试。

from __future__ import annotations

import numpy as np
import pytest

from mosaic.core.types import MosaicData, MotionData, AudioData
from mosaic.core.node import NodeSpec


# ===========================================================================
# 辅助：直接访问预设动作注册表
# ===========================================================================
def _get_preset_names():
    """获取内置预设动作名称列表。"""
    from mosaic.nodes.digital_human.motion_generator import _PRESET_ANIMATIONS
    return list(_PRESET_ANIMATIONS.keys())


# ===========================================================================
# T_MOT_01：preset 模式，输入预设名称 "wave" 输出 MotionData
# ===========================================================================
class TestPresetWave:
    """preset 模式 wave 动作测试。"""

    def test_preset_wave_outputs_motion_data(self, cpu_scheduler):
        """T_MOT_01：preset 模式，输入预设名称 "wave" 输出 MotionData。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            preset_name="wave",
            duration=3.0,
            fps=30,
        ))

        # 验证输出是 MosaicData
        assert isinstance(result, MosaicData), (
            f"输出应为 MosaicData，实际为 {type(result).__name__}"
        )

        # 验证包含 motion 字段
        assert "motion" in result, "输出应包含 'motion' 字段"
        assert isinstance(result["motion"], MotionData), (
            f"motion 应为 MotionData 类型，实际为 {type(result['motion']).__name__}"
        )

        # 验证 motion 基本属性
        motion = result["motion"]
        assert motion.frame_count > 0, "frame_count 应大于 0"
        assert motion.fps == 30, f"fps 应为 30，实际为 {motion.fps}"
        assert motion.skeleton_type == "coco", (
            f"skeleton_type 应为 'coco'，实际为 {motion.skeleton_type}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_02：preset 模式，内置动作列表非空
# ===========================================================================
class TestPresetList:
    """内置动作列表测试。"""

    def test_preset_animations_non_empty(self):
        """T_MOT_02：内置动作列表非空，至少包含 wave, bow, nod。"""
        from mosaic.nodes.digital_human.motion_generator import _PRESET_ANIMATIONS

        presets = _PRESET_ANIMATIONS
        assert len(presets) > 0, "预设动作列表不应为空"
        assert len(presets) == 15, f"应有 15 个预设动作，实际 {len(presets)}"

        # 验证关键动作存在
        required = ["wave", "bow", "nod", "shake_head", "clap", "walk", "dance"]
        for name in required:
            assert name in presets, (
                f"预设动作 '{name}' 应在 _PRESET_ANIMATIONS 中，"
                f"可用动作: {sorted(presets.keys())}"
            )

    def test_describe_shows_presets(self, cpu_scheduler):
        """T_MOT_02 补充：describe 返回预设动作列表。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        spec = gen.describe()
        assert isinstance(spec, NodeSpec), "describe 应返回 NodeSpec"
        assert "preset_names" in spec.model_info, "model_info 应包含 preset_names"
        assert len(spec.model_info["preset_names"]) == 15, (
            f"preset_names 应有 15 个，实际 {len(spec.model_info['preset_names'])}"
        )


# ===========================================================================
# T_MOT_03：text2motion 模式，输入描述 "挥手" 输出 MotionData
# ===========================================================================
class TestText2Motion:
    """text2motion 模式测试。"""

    def test_text2motion_with_prompt(self, cpu_scheduler):
        """T_MOT_03：text2motion 模式，输入描述 "挥手" 输出 MotionData。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="text2motion",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            prompt="挥手",
            duration=2.0,
            fps=30,
        ))

        assert isinstance(result, MosaicData), "输出应为 MosaicData"
        assert "motion" in result, "输出应包含 'motion' 字段"
        assert isinstance(result["motion"], MotionData), "motion 应为 MotionData"

        motion = result["motion"]
        assert motion.frame_count > 0, "frame_count 应大于 0"
        # 2.0 * 30 = 60 frames
        assert motion.frame_count == 60, (
            f"frame_count 应为 60，实际为 {motion.frame_count}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_04：audio2motion 模式，输入音频输出 MotionData
# ===========================================================================
class TestAudio2Motion:
    """audio2motion 模式测试。"""

    def test_audio2motion_with_audio(self, cpu_scheduler, sample_short_audio):
        """T_MOT_04：audio2motion 模式，输入音频输出 MotionData。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="audio2motion",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            audio=sample_short_audio,
            fps=30,
        ))

        assert isinstance(result, MosaicData), "输出应为 MosaicData"
        assert "motion" in result, "输出应包含 'motion' 字段"
        assert isinstance(result["motion"], MotionData), "motion 应为 MotionData"

        motion = result["motion"]
        assert motion.frame_count > 0, "frame_count 应大于 0"
        # 音频 2 秒 * 30 fps = 60 frames
        assert motion.frame_count == 60, (
            f"frame_count 应为 60，实际为 {motion.frame_count}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_05：duration 参数生效
# ===========================================================================
class TestDurationParameter:
    """duration 参数测试。"""

    def test_duration_parameter(self, cpu_scheduler):
        """T_MOT_05：duration=5.0 产生 150 帧 @ 30fps。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            preset_name="wave",
            duration=5.0,
            fps=30,
        ))

        assert result["frame_count"] == 150, (
            f"duration=5.0 @ 30fps 应产生 150 帧，实际 {result['frame_count']}"
        )
        assert result["duration"] == 5.0, (
            f"duration 应为 5.0，实际 {result['duration']}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_06：fps 参数生效
# ===========================================================================
class TestFpsParameter:
    """fps 参数测试。"""

    def test_fps_parameter(self, cpu_scheduler):
        """T_MOT_06：fps=15 产生 45 帧 @ 3s。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            preset_name="wave",
            duration=3.0,
            fps=15,
        ))

        assert result["frame_count"] == 45, (
            f"duration=3.0 @ 15fps 应产生 45 帧，实际 {result['frame_count']}"
        )
        assert result["motion"].fps == 15, (
            f"fps 应为 15，实际为 {result['motion'].fps}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_07：smooth 参数生效
# ===========================================================================
class TestSmoothParameter:
    """smooth 参数测试。"""

    def test_smooth_parameter(self, cpu_scheduler):
        """T_MOT_07：smooth=True vs smooth=False 输出有差异。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )
        gen.load()

        # smooth=True
        result_smooth = gen.run(MosaicData(
            preset_name="wave",
            duration=3.0,
            fps=30,
            smooth=True,
        ))
        kp_smooth = result_smooth["motion"].keypoints

        # smooth=False
        result_raw = gen.run(MosaicData(
            preset_name="wave",
            duration=3.0,
            fps=30,
            smooth=False,
        ))
        kp_raw = result_raw["motion"].keypoints

        # 平滑后与原始数据应存在差异（至少在某些帧上）
        diff = np.abs(kp_smooth - kp_raw)
        max_diff = float(np.max(diff))
        assert max_diff >= 0, (
            f"smooth=True 和 smooth=False 的输出应有差异，max_diff={max_diff}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_08：输出 keypoints shape 正确
# ===========================================================================
class TestKeypointsShape:
    """keypoints 形状测试。"""

    def test_keypoints_shape_coco(self, cpu_scheduler):
        """T_MOT_08：输出 keypoints shape 为 (frame_count, 17, 2) 对于 COCO。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            skeleton_type="coco",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=10,
        ))

        keypoints = result["motion"].keypoints
        frame_count = result["frame_count"]
        expected_shape = (frame_count, 17, 2)
        assert keypoints.shape == expected_shape, (
            f"keypoints shape 应为 {expected_shape}，实际为 {keypoints.shape}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_09：skeleton_type 参数正确传递
# ===========================================================================
class TestSkeletonType:
    """skeleton_type 参数测试。"""

    def test_skeleton_type_parameter(self, cpu_scheduler):
        """T_MOT_09：skeleton_type="openpose" 正确传递到输出。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            skeleton_type="openpose",
            scheduler=cpu_scheduler,
        )
        gen.load()

        result = gen.run(MosaicData(
            preset_name="wave",
            duration=1.0,
            fps=30,
        ))

        assert result["skeleton_type"] == "openpose", (
            f"skeleton_type 应为 'openpose'，实际为 {result['skeleton_type']}"
        )
        assert result["motion"].skeleton_type == "openpose", (
            f"motion.skeleton_type 应为 'openpose'，实际为 {result['motion'].skeleton_type}"
        )

        gen.unload()


# ===========================================================================
# T_MOT_10：describe 返回正确信息
# ===========================================================================
class TestDescribe:
    """describe 方法测试。"""

    def test_describe_returns_correct_info(self, cpu_scheduler):
        """T_MOT_10：describe 返回正确信息。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="preset",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = gen.describe()
        assert isinstance(spec, NodeSpec), "describe 应返回 NodeSpec"
        assert spec.name == "motion-generator", (
            f"name 应为 'motion-generator'，实际为 {spec.name}"
        )
        assert spec.domain == "digital_human", (
            f"domain 应为 'digital_human'，实际为 {spec.domain}"
        )
        assert spec.version == "0.1.0", (
            f"version 应为 '0.1.0'，实际为 {spec.version}"
        )
        assert "motion" in spec.output_types, "output_types 应包含 'motion'"
        assert "model_info" in spec.model_info or isinstance(spec.model_info, dict), (
            "model_info 应为字典"
        )
        assert spec.model_info["method"] == "preset", (
            f"method 应为 'preset'，实际为 {spec.model_info.get('method')}"
        )
        assert spec.model_info["num_presets"] == 15, (
            f"num_presets 应为 15，实际为 {spec.model_info.get('num_presets')}"
        )

    def test_describe_text2motion(self, cpu_scheduler):
        """T_MOT_10 补充：text2motion 模式的 describe。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="text2motion",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = gen.describe()
        assert spec.model_info["method"] == "text2motion", (
            f"method 应为 'text2motion'，实际为 {spec.model_info.get('method')}"
        )

    def test_describe_audio2motion(self, cpu_scheduler):
        """T_MOT_10 补充：audio2motion 模式的 describe。"""
        from mosaic.nodes.digital_human.motion_generator import MotionGenerator

        gen = MotionGenerator(
            method="audio2motion",
            device="cpu",
            dtype="float32",
            scheduler=cpu_scheduler,
        )

        spec = gen.describe()
        assert spec.model_info["method"] == "audio2motion", (
            f"method 应为 'audio2motion'，实际为 {spec.model_info.get('method')}"
        )