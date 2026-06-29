# mosaic/nodes/digital_human/motion_generator.py
"""MotionGenerator 动作生成节点。

为数字人生成骨骼关键点序列（:class:`~mosaic.core.types.MotionData`），
驱动后续的渲染与口型同步节点。支持三种生成方式：

* ``"preset"``       —— 预设动作库（挥手、鞠躬、点头等），纯 CPU、零依赖、
  零显存，基于 COCO 17 关键点的正弦波动画合成。默认方式。
* ``"audio2motion"`` —— 音频节拍分析 + 动作模板匹配，显存需求低，
  依据音频的节奏与能量从预设动作库中选择并拼接片段，使动作与音频对齐。
* ``"text2motion"``  —— 加载 MotionGPT 等文本驱动动作生成模型，
  根据 ``prompt`` 生成与语义匹配的动作序列，显存约 4-8GB。

设计要点
--------
* 继承 :class:`BaseDigitalHumanNode`，复用其显存调度、事件发射与
  ``_resolve_device``/``_resolve_dtype``/``_apply_optimizations`` 等工具。
* ``torch`` / ``transformers`` / ``librosa`` / ``numpy`` 全部惰性导入，
  使本模块在依赖缺失时仍可被注册表发现与导入。
* ``preset`` 方式不依赖任何外部库即可工作（仅 ``numpy``），适合在没有
  GPU 的环境中快速原型验证。
* 关键点平滑通过可选的滑动平均实现，避免逐帧抖动。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start``/
  ``node_complete``/``node_error``/``progress`` 事件。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MotionData, MosaicData

from mosaic.nodes.digital_human._base import BaseDigitalHumanNode

__all__ = ["MotionGenerator"]


# ---------------------------------------------------------------------------
# COCO 17 关键点定义
# ---------------------------------------------------------------------------
# 顺序与 COCO / CommonPose 标准一致：
#   0: nose            1: left_eye        2: right_eye
#   3: left_ear        4: right_ear
#   5: left_shoulder   6: right_shoulder
#   7: left_elbow      8: right_elbow
#   9: left_wrist     10: right_wrist
#  11: left_hip       12: right_hip
#  13: left_knee      14: right_knee
#  15: left_ankle     16: right_ankle
_COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_NUM_COCO_KEYPOINTS: int = len(_COCO_KEYPOINT_NAMES)  # 17

# 归一化静止姿态（坐标范围约 [0, 1]，人物居中、面向观察者）。
# 形状 (17, 2)，列顺序 (x, y)，y 向下为正。
_REST_POSE: tuple[tuple[float, float], ...] = (
    (0.50, 0.18),  # nose
    (0.48, 0.15),  # left_eye
    (0.52, 0.15),  # right_eye
    (0.45, 0.18),  # left_ear
    (0.55, 0.18),  # right_ear
    (0.42, 0.30),  # left_shoulder
    (0.58, 0.30),  # right_shoulder
    (0.38, 0.45),  # left_elbow
    (0.62, 0.45),  # right_elbow
    (0.36, 0.60),  # left_wrist
    (0.64, 0.60),  # right_wrist
    (0.45, 0.55),  # left_hip
    (0.55, 0.55),  # right_hip
    (0.44, 0.75),  # left_knee
    (0.56, 0.75),  # right_knee
    (0.44, 0.95),  # left_ankle
    (0.56, 0.95),  # right_ankle
)

# 关键点索引常量，便于动画函数引用
_KP = {name: idx for idx, name in enumerate(_COCO_KEYPOINT_NAMES)}


# ---------------------------------------------------------------------------
# 预设动作生成器
# ---------------------------------------------------------------------------
# 每个生成器签名: (t: np.ndarray) -> np.ndarray
#   t: 形状 (T,) 的时间数组（秒）
#   返回: 形状 (T, 17, 2) 的关键点偏移量（叠加在 _REST_POSE 上）
# 动作幅度均控制在归一化坐标的合理范围内。
PresetGenerator = Callable[[Any], Any]


def _wave(t: np.ndarray) -> np.ndarray:
    """挥手：右臂周期性上下摆动，右手腕幅度最大。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)  # (17, 2)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    # 右臂上举：肘与腕向斜上方抬起
    lift = 0.12 * (1.0 - np.cos(2 * np.pi * t / 1.2))  # 缓起
    swing = 0.10 * np.sin(2 * np.pi * t / 0.8)          # 快速摆动

    kps[:, _KP["right_elbow"], 1] -= lift
    kps[:, _KP["right_wrist"], 1] -= lift + 0.10
    kps[:, _KP["right_wrist"], 0] += 0.08 + swing
    kps[:, _KP["right_elbow"], 0] += 0.04
    return kps


def _bow(t: np.ndarray) -> np.ndarray:
    """鞠躬：上半身前倾后恢复（整体下压并回弹）。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    # 一个鞠躬周期 ~2.4s：前倾 -> 最低点 -> 回正
    phase = 2 * np.pi * t / 2.4
    bend = 0.5 * (1.0 - np.cos(phase))  # 0..1..0

    upper = [
        _KP["nose"], _KP["left_eye"], _KP["right_eye"],
        _KP["left_ear"], _KP["right_ear"],
        _KP["left_shoulder"], _KP["right_shoulder"],
        _KP["left_elbow"], _KP["right_elbow"],
        _KP["left_wrist"], _KP["right_wrist"],
    ]
    for idx in upper:
        kps[:, idx, 1] += 0.18 * bend      # 下压
        kps[:, idx, 0] += 0.02 * bend      # 微前倾
    # 手臂随身体下垂
    kps[:, _KP["left_wrist"], 1] += 0.05 * bend
    kps[:, _KP["right_wrist"], 1] += 0.05 * bend
    return kps


def _nod(t: np.ndarray) -> np.ndarray:
    """点头：头部前后摆动（y 方向小幅振荡）。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    osc = 0.025 * np.sin(2 * np.pi * t / 1.0)
    head = [_KP["nose"], _KP["left_eye"], _KP["right_eye"],
            _KP["left_ear"], _KP["right_ear"]]
    for idx in head:
        kps[:, idx, 1] += osc
    return kps


def _shake_head(t: np.ndarray) -> np.ndarray:
    """摇头：头部左右摆动（x 方向小幅振荡）。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    osc = 0.04 * np.sin(2 * np.pi * t / 0.9)
    head = [_KP["nose"], _KP["left_eye"], _KP["right_eye"],
            _KP["left_ear"], _KP["right_ear"]]
    for idx in head:
        kps[:, idx, 0] += osc
    return kps


def _clap(t: np.ndarray) -> np.ndarray:
    """鼓掌：双手腕向中心靠拢后分开。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    # 双手抬起至胸前
    lift = 0.12
    kps[:, _KP["left_wrist"], 1] -= lift
    kps[:, _KP["right_wrist"], 1] -= lift
    kps[:, _KP["left_elbow"], 1] -= 0.06
    kps[:, _KP["right_elbow"], 1] -= 0.06

    # 周期性合拢/分开
    close = 0.5 * (1.0 - np.cos(2 * np.pi * t / 0.6))  # 0..1..0
    kps[:, _KP["left_wrist"], 0] += 0.08 * close
    kps[:, _KP["right_wrist"], 0] -= 0.08 * close
    return kps


def _walk(t: np.ndarray) -> np.ndarray:
    """走路：双腿交替前后摆动，手臂反向摆动。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    stride = 0.06 * np.sin(2 * np.pi * t / 1.0)
    lift = 0.03 * (1.0 - np.cos(2 * np.pi * t / 1.0)) / 2.0

    # 左腿前摆、右腿后摆
    kps[:, _KP["left_knee"], 0] += stride
    kps[:, _KP["left_ankle"], 0] += stride * 1.5
    kps[:, _KP["left_ankle"], 1] -= lift
    kps[:, _KP["right_knee"], 0] -= stride
    kps[:, _KP["right_ankle"], 0] -= stride * 1.5
    kps[:, _KP["right_ankle"], 1] -= lift[::-1] if hasattr(lift, "__getitem__") else lift

    # 手臂反向摆动
    kps[:, _KP["left_wrist"], 0] -= stride * 0.8
    kps[:, _KP["right_wrist"], 0] += stride * 0.8
    # 整体轻微上下起伏
    body_bob = 0.01 * np.sin(2 * np.pi * t / 0.5)
    kps[:, :, 1] += body_bob[:, None]
    return kps


def _dance(t: np.ndarray) -> np.ndarray:
    """跳舞：手臂上举摆动 + 髋部左右摆动，节奏较快。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    beat = 2 * np.pi * t / 0.6
    # 双臂上举并交替摆动
    kps[:, _KP["left_wrist"], 1] -= 0.22
    kps[:, _KP["right_wrist"], 1] -= 0.22
    kps[:, _KP["left_elbow"], 1] -= 0.12
    kps[:, _KP["right_elbow"], 1] -= 0.12
    kps[:, _KP["left_wrist"], 0] += 0.06 * np.sin(beat)
    kps[:, _KP["right_wrist"], 0] -= 0.06 * np.sin(beat)

    # 髋部摆动
    sway = 0.04 * np.sin(beat)
    hips = [_KP["left_hip"], _KP["right_hip"]]
    for idx in hips:
        kps[:, idx, 0] += sway
    # 头部随节奏轻摆
    kps[:, _KP["nose"], 0] += 0.02 * np.sin(beat + 1.0)
    return kps


def _raise_hand(t: np.ndarray) -> np.ndarray:
    """举手：右臂高举过头顶并保持。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    # 缓举起
    raise_amt = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.8, 0, 1)))
    kps[:, _KP["right_elbow"], 1] -= 0.18 * raise_amt
    kps[:, _KP["right_wrist"], 1] -= 0.38 * raise_amt
    kps[:, _KP["right_wrist"], 0] += 0.02
    kps[:, _KP["right_elbow"], 0] += 0.01
    return kps


def _sit(t: np.ndarray) -> np.ndarray:
    """坐下：髋部下沉、膝盖弯曲。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    phase = 2 * np.pi * t / 2.0
    sit_amt = 0.5 * (1.0 - np.cos(phase))  # 0..1..0
    # 髋部下沉
    kps[:, _KP["left_hip"], 1] += 0.12 * sit_amt
    kps[:, _KP["right_hip"], 1] += 0.12 * sit_amt
    # 膝盖前移（弯曲）
    kps[:, _KP["left_knee"], 0] += 0.04 * sit_amt
    kps[:, _KP["right_knee"], 0] += 0.04 * sit_amt
    # 上半身微前倾
    upper = [_KP["nose"], _KP["left_shoulder"], _KP["right_shoulder"]]
    for idx in upper:
        kps[:, idx, 0] += 0.02 * sit_amt
    return kps


def _stand(t: np.ndarray) -> np.ndarray:
    """站立：基本静止，仅保留轻微呼吸起伏。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    breathe = 0.005 * np.sin(2 * np.pi * t / 3.0)
    chest = [_KP["left_shoulder"], _KP["right_shoulder"],
             _KP["left_hip"], _KP["right_hip"]]
    for idx in chest:
        kps[:, idx, 1] += breathe
    return kps


def _jump(t: np.ndarray) -> np.ndarray:
    """跳跃：整体上下抛物运动 + 屈膝蓄力。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    cycle = 2.0
    phase = (t % cycle) / cycle  # 0..1
    # 蓄力(0~0.25) -> 起跳(0.25~0.5) -> 落地(0.5~0.75) -> 恢复
    crouch = np.where(phase < 0.25, 0.06 * (phase / 0.25),
                      np.where(phase < 0.5, 0.06,
                               np.where(phase < 0.75, 0.06 * (1 - (phase - 0.5) / 0.25), 0.0)))
    air = np.where((phase > 0.25) & (phase < 0.5),
                   0.15 * np.sin(np.pi * (phase - 0.25) / 0.25), 0.0)

    kps[:, :, 1] += crouch[:, None] - air[:, None]
    # 蓄力时膝盖前弯
    kps[:, _KP["left_knee"], 0] += 0.03 * crouch
    kps[:, _KP["right_knee"], 0] += 0.03 * crouch
    # 手臂上摆
    kps[:, _KP["left_wrist"], 1] -= air * 1.5
    kps[:, _KP["right_wrist"], 1] -= air * 1.5
    return kps


def _point(t: np.ndarray) -> np.ndarray:
    """指引/指向：右臂前伸指向。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    extend = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.5, 0, 1)))
    kps[:, _KP["right_elbow"], 0] += 0.08 * extend
    kps[:, _KP["right_elbow"], 1] -= 0.10 * extend
    kps[:, _KP["right_wrist"], 0] += 0.18 * extend
    kps[:, _KP["right_wrist"], 1] -= 0.18 * extend
    return kps


def _thinking(t: np.ndarray) -> np.ndarray:
    """思考：右手托腮。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    reach = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.6, 0, 1)))
    # 右手抬至下巴位置
    target_y = rest[_KP["nose"]][1] - rest[_KP["right_wrist"]][1]
    target_x = rest[_KP["nose"]][0] - rest[_KP["right_wrist"]][0]
    kps[:, _KP["right_wrist"], 1] += target_y * reach
    kps[:, _KP["right_wrist"], 0] += target_x * reach
    kps[:, _KP["right_elbow"], 1] -= 0.12 * reach
    kps[:, _KP["right_elbow"], 0] += 0.04 * reach
    # 头部微倾
    kps[:, _KP["nose"], 0] += 0.01 * reach
    return kps


def _stretch(t: np.ndarray) -> np.ndarray:
    """伸展：双臂上举伸展。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    reach = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.8, 0, 1)))
    sway = 0.02 * np.sin(2 * np.pi * t / 2.0)
    kps[:, _KP["left_wrist"], 1] -= 0.30 * reach
    kps[:, _KP["right_wrist"], 1] -= 0.30 * reach
    kps[:, _KP["left_elbow"], 1] -= 0.16 * reach
    kps[:, _KP["right_elbow"], 1] -= 0.16 * reach
    kps[:, _KP["left_wrist"], 0] -= sway
    kps[:, _KP["right_wrist"], 0] += sway
    return kps


def _turn_around(t: np.ndarray) -> np.ndarray:
    """转身：髋部与肩部水平偏移，模拟侧身。"""
    import numpy as np  # type: ignore

    T = t.shape[0]
    rest = np.array(_REST_POSE, dtype=np.float32)
    kps = np.broadcast_to(rest, (T, _NUM_COCO_KEYPOINTS, 2)).copy()

    turn = 0.5 * (1.0 - np.cos(2 * np.pi * t / 2.5))  # 0..1..0
    # 右侧关键点向左收缩（模拟侧身透视）
    right_side = [_KP["right_shoulder"], _KP["right_elbow"], _KP["right_wrist"],
                  _KP["right_hip"], _KP["right_knee"], _KP["right_ankle"]]
    for idx in right_side:
        kps[:, idx, 0] -= 0.10 * turn
    # 头部跟随
    kps[:, _KP["nose"], 0] -= 0.03 * turn
    return kps


# 预设动作注册表：name -> (生成器, 默认周期秒数, 描述)
_PRESET_ANIMATIONS: dict[str, tuple[PresetGenerator, float, str]] = {
    "wave":        (_wave,        2.4, "挥手：右臂周期性上下摆动"),
    "bow":         (_bow,         2.4, "鞠躬：上半身前倾后恢复"),
    "nod":         (_nod,         2.0, "点头：头部前后摆动"),
    "shake_head":  (_shake_head,  1.8, "摇头：头部左右摆动"),
    "clap":        (_clap,        1.2, "鼓掌：双手腕向中心靠拢后分开"),
    "walk":        (_walk,        2.0, "走路：双腿交替前后摆动"),
    "dance":       (_dance,       2.4, "跳舞：手臂上举摆动 + 髋部摆动"),
    "raise_hand":  (_raise_hand,  2.0, "举手：右臂高举过头顶"),
    "sit":         (_sit,         2.0, "坐下：髋部下沉、膝盖弯曲"),
    "stand":       (_stand,       3.0, "站立：基本静止，轻微呼吸"),
    "jump":        (_jump,        2.0, "跳跃：整体上下抛物运动"),
    "point":       (_point,       2.0, "指引：右臂前伸指向"),
    "thinking":    (_thinking,    2.0, "思考：右手托腮"),
    "stretch":     (_stretch,     2.0, "伸展：双臂上举伸展"),
    "turn_around": (_turn_around, 2.5, "转身：髋部与肩部水平偏移"),
}


# 支持的生成方式
_SUPPORTED_METHODS: tuple[str, ...] = (
    "text2motion",
    "audio2motion",
    "preset",
)

# text2motion 默认模型
_DEFAULT_TEXT2MOTION_MODEL = "PrimeIntellect/MotionGPT"


@registry.register
class MotionGenerator(BaseDigitalHumanNode):
    """动作生成节点：为数字人生成骨骼关键点序列。

    支持三种生成方式（``method``）：

    * ``"preset"``（默认）—— 内置 15 个常用动作模板，基于 COCO 17 关键点
      的正弦波动画合成，纯 CPU、零显存、零模型下载。
    * ``"audio2motion"`` —— 分析音频节拍与能量，从预设动作库中选择并
      拼接片段，使动作与音频节奏对齐；显存需求低。
    * ``"text2motion"`` —— 加载 MotionGPT 等文本驱动动作生成模型，
      根据 ``prompt`` 生成语义匹配的动作序列；显存约 4-8GB。

    Parameters
    ----------
    model:
        模型标识，仅 ``"text2motion"`` 方式使用。``None`` 时使用默认模型
        ``"PrimeIntellect/MotionGPT"``。
    method:
        生成方式，可选 ``"text2motion"`` / ``"audio2motion"`` / ``"preset"``，
        默认 ``"preset"``。
    skeleton_type:
        骨骼关键点格式，默认 ``"coco"``（17 关键点）。当前内置预设动作
        仅支持 COCO 格式。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
        ``"preset"`` 方式始终在 CPU 上运行。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。仅 ``"text2motion"`` 方式使用。
    **kwargs:
        透传给 :class:`BaseDigitalHumanNode` 的参数（``scheduler`` /
        ``bus`` 等）。

    Limitations
    -----------
    * 内置预设动作库为基于正弦波的程序化动画，仅覆盖 COCO 17 关键点，
      不涉及真实动作捕捉数据，适合快速原型与轻量场景。
    * ``"text2motion"`` 方式需要 ``torch`` / ``transformers`` 及对应模型
      权重；GPU 强烈推荐，CPU 模式下推理极慢。
    * ``"audio2motion"`` 方式需要 ``numpy``；可选 ``librosa`` 用于更精确
      的节拍检测，缺失时回退到基于 RMS 能量的简化节拍估计。
    * ``skeleton_type`` 当前固定为 ``"coco"``，其他骨骼格式暂不支持。

    Examples
    --------
    预设动作（零依赖）：

    >>> gen = MotionGenerator(method="preset")
    >>> result = gen(MosaicData(preset_name="wave", duration=3.0, fps=30))
    >>> result["motion"].keypoints.shape  # (90, 17, 2)
    >>> result["frame_count"]
    90

    音频驱动动作：

    >>> gen = MotionGenerator(method="audio2motion")
    >>> result = gen(MosaicData(audio=audio_data, fps=30))

    文本驱动动作（需要 GPU 与模型）：

    >>> gen = MotionGenerator(method="text2motion", model="PrimeIntellect/MotionGPT")
    >>> result = gen(MosaicData(prompt="a person waving hello", duration=4.0))
    """

    name: str = "motion-generator"
    description: str = (
        "Generate skeleton keypoint sequences (MotionData) for digital "
        "humans via preset animations, audio-beat matching, or text-driven "
        "MotionGPT. Outputs COCO 17-keypoint trajectories with optional "
        "smoothing."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["text", "audio", "mosaic"]
    output_types: list[str] = ["motion", "mosaic"]

    def __init__(
        self,
        model: str | None = None,
        method: str = "preset",
        skeleton_type: str = "coco",
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {method!r}. "
                f"Choose from {_SUPPORTED_METHODS}."
            )
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._method: str = method
        self._skeleton_type: str = skeleton_type
        # model 仅对 text2motion 有意义；preset/audio2motion 不需要
        self._model_name: str = model or _DEFAULT_TEXT2MOTION_MODEL
        # 运行时状态
        self._tokenizer: Any = None
        self._last_fps: int = 30

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载动作生成所需的资源。

        通过 :meth:`Scheduler.track` 注册显存跟踪后，根据 ``method`` 分发：

        * ``"preset"`` —— 无需加载任何模型，直接置 ``_loaded=True``。
        * ``"audio2motion"`` —— 无需模型，仅记录就绪状态（节拍分析在
          ``run`` 时按需进行）。
        * ``"text2motion"`` —— 惰性导入 ``transformers``，加载 MotionGPT
          模型与 tokenizer，迁移到目标设备并应用显存优化。
        """
        self._scheduler.track(self)

        if self._method == "preset":
            self._logger.info(
                "MotionGenerator preset library ready (%d animations, CPU-only).",
                len(_PRESET_ANIMATIONS),
            )
            self._loaded = True
            return

        if self._method == "audio2motion":
            self._logger.info(
                "MotionGenerator audio2motion ready (beat analysis on-demand)."
            )
            self._loaded = True
            return

        # text2motion：加载模型。模型加载失败时降级到关键词匹配预设动作，
        # 保证节点在无 GPU / 无 transformers / 无网络时仍可运行。
        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading text2motion model %s (device=%s, dtype=%s) ...",
            self._model_name,
            self._device,
            self._dtype_str,
        )
        try:
            self._load_text2motion_model()
            self._apply_optimizations()
        except (ImportError, RuntimeError, OSError) as exc:
            self._logger.warning(
                "text2motion model load failed (%s: %s). "
                "Falling back to keyword-based preset selection. "
                "Install `transformers torch` and ensure network access "
                "for full text-driven motion generation.",
                type(exc).__name__,
                exc,
            )
            self._pipeline = None
            self._tokenizer = None
        self._loaded = True

    def _load_text2motion_model(self) -> None:
        """加载 MotionGPT 文本驱动动作生成模型。"""
        try:
            from transformers import (  # type: ignore
                AutoTokenizer,
                AutoModelForCausalLM,
            )
        except ImportError as exc:
            raise ImportError(
                "text2motion method requires 'transformers'. "
                "Install via `pip install transformers torch`."
            ) from exc

        import torch  # type: ignore

        torch_dtype = self._resolve_dtype()
        device = self._resolve_device()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            # 优先使用 dtype 参数（新版 transformers），回退 torch_dtype（旧版兼容）
            try:
                self._pipeline = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except TypeError:
                # 旧版 transformers 不支持 dtype 参数
                self._pipeline = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load text2motion model {self._model_name!r}: {exc}. "
                f"Ensure the model id is correct and network is available."
            ) from exc

        try:
            self._pipeline = self._pipeline.to(device)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to move model to %s: %s. Falling back to CPU.",
                device,
                exc,
            )
            self._device = "cpu"
            try:
                self._pipeline = self._pipeline.to("cpu")
            except Exception:  # noqa: BLE001
                pass

        self._logger.info(
            "text2motion model loaded (device=%s, dtype=%s).",
            self._device,
            self._dtype_str,
        )

    # ------------------------------------------------------------------
    # 推理主入口
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行动作生成。

        Parameters
        ----------
        input_data:
            可包含以下字段：

            * ``prompt`` (str, 可选) —— 动作描述文本，``"text2motion"`` 必须，
              ``"preset"``/``"audio2motion"`` 忽略。
            * ``audio`` (:class:`AudioData`, 可选) —— 驱动音频，
              ``"audio2motion"`` 必须，其他方式忽略。
            * ``preset_name`` (str, 可选) —— 预设动作名称，
              ``"preset"`` 方式使用；未指定时使用 ``"wave"``。
            * ``duration`` (float, 默认 3.0) —— 动作时长（秒）。
              ``"audio2motion"`` 方式若提供 ``audio`` 则以音频时长为准。
            * ``fps`` (int, 默认 30) —— 输出帧率。
            * ``smooth`` (bool, 默认 True) —— 是否对关键点做滑动平均平滑。

        Returns
        -------
        MosaicData
            包含 ``motion`` (:class:`MotionData`)、``keypoints``
            (``numpy.ndarray``, 形状 ``(frame_count, 17, 2)``)、
            ``frame_count`` (int)、``duration`` (float)、
            ``skeleton_type`` (str)。

        Raises
        ------
        ValueError
            ``"preset"`` 方式 ``preset_name`` 不在预设库中；或
            ``"text2motion"`` 缺少 ``prompt``；或 ``"audio2motion"``
            缺少 ``audio``。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # ---------- 提取公共参数 ----------
            duration = float(input_data.get("duration", 3.0))
            fps = int(input_data.get("fps", 30))
            fps = max(1, fps)
            smooth = bool(input_data.get("smooth", True))
            self._last_fps = fps

            # ---------- 分发 ----------
            if self._method == "preset":
                preset_name = input_data.get("preset_name", "wave")
                keypoints, frame_count, duration = self._generate_preset(
                    preset_name=preset_name,
                    duration=duration,
                    fps=fps,
                )
            elif self._method == "audio2motion":
                audio = input_data.get("audio")
                if audio is None:
                    raise ValueError(
                        "audio2motion method requires 'audio' (AudioData)."
                    )
                keypoints, frame_count, duration = self._generate_audio2motion(
                    audio=audio,
                    duration=duration,
                    fps=fps,
                )
            else:  # text2motion
                prompt = input_data.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        "text2motion method requires 'prompt' (non-empty str)."
                    )
                keypoints, frame_count, duration = self._generate_text2motion(
                    prompt=prompt,
                    duration=duration,
                    fps=fps,
                )

            # ---------- 平滑 ----------
            if smooth and frame_count > 3:
                keypoints = self._smooth_keypoints(keypoints, window=5)

        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # ---------- 封装输出 ----------
        motion = MotionData(
            keypoints=keypoints,
            frame_count=frame_count,
            fps=fps,
            skeleton_type=self._skeleton_type,
            metadata={
                "method": self._method,
                "model": self._model_name if self._method == "text2motion" else None,
                "smooth": smooth,
            },
        )

        result = MosaicData(
            motion=motion,
            keypoints=keypoints,
            frame_count=frame_count,
            duration=duration,
            skeleton_type=self._skeleton_type,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "method": self._method,
                "frame_count": frame_count,
                "duration": duration,
                "fps": fps,
                "skeleton_type": self._skeleton_type,
            },
        )
        return result

    # ------------------------------------------------------------------
    # preset 方式
    # ------------------------------------------------------------------
    def _generate_preset(
        self,
        preset_name: str,
        duration: float,
        fps: int,
    ) -> tuple[Any, int, float]:
        """从预设动作库生成关键点序列。

        Parameters
        ----------
        preset_name:
            预设动作名称，需在 :data:`_PRESET_ANIMATIONS` 中。
        duration:
            动作时长（秒）。
        fps:
            帧率。

        Returns
        -------
        tuple[numpy.ndarray, int, float]
            ``(keypoints, frame_count, duration)``，keypoints 形状
            ``(frame_count, 17, 2)``。
        """
        import numpy as np  # type: ignore

        if preset_name not in _PRESET_ANIMATIONS:
            available = ", ".join(sorted(_PRESET_ANIMATIONS.keys()))
            raise ValueError(
                f"Unknown preset {preset_name!r}. Available: {available}."
            )

        generator, cycle, desc = _PRESET_ANIMATIONS[preset_name]
        self._logger.info(
            "Generating preset motion %r (%s) duration=%.2fs fps=%d.",
            preset_name, desc, duration, fps,
        )

        frame_count = max(1, int(round(duration * fps)))
        t = np.linspace(0.0, duration, frame_count, dtype=np.float32)
        keypoints = generator(t)  # (T, 17, 2)
        keypoints = np.asarray(keypoints, dtype=np.float32)

        # 进度报告
        self._emit_progress(frame_count, frame_count, f"preset:{preset_name}")
        return keypoints, frame_count, duration

    # ------------------------------------------------------------------
    # audio2motion 方式
    # ------------------------------------------------------------------
    def _generate_audio2motion(
        self,
        audio: Any,
        duration: float,
        fps: int,
    ) -> tuple[Any, int, float]:
        """基于音频节拍分析从预设库选择并拼接动作片段。

        Parameters
        ----------
        audio:
            :class:`AudioData` 实例。
        duration:
            备用时长（秒）；若能从 audio 推断出时长则以音频为准。
        fps:
            帧率。

        Returns
        -------
        tuple[numpy.ndarray, int, float]
            ``(keypoints, frame_count, duration)``。
        """
        import numpy as np  # type: ignore

        waveform, sr = self._load_audio_waveform(audio)
        audio_duration = self._audio_duration(waveform, sr)
        if audio_duration > 0:
            duration = audio_duration

        frame_count = max(1, int(round(duration * fps)))
        t = np.linspace(0.0, duration, frame_count, dtype=np.float32)

        # 节拍检测
        beats = self._detect_beats(waveform, sr, fps, frame_count)
        energy = self._frame_energy(waveform, sr, frame_count)

        # 根据节拍密度选择动作模板
        tempo = len(beats) / max(duration, 1e-6)
        if tempo > 2.0:
            primary = "dance"
            secondary = "clap"
        elif tempo > 1.0:
            primary = "walk"
            secondary = "wave"
        else:
            primary = "nod"
            secondary = "stand"

        self._logger.info(
            "audio2motion: duration=%.2fs, tempo=%.2f beats/s, "
            "primary=%r, secondary=%r.",
            duration, tempo, primary, secondary,
        )

        # 合成：以 primary 为基底，在节拍时刻叠加 secondary 的脉冲
        gen_primary = _PRESET_ANIMATIONS[primary][0]
        gen_secondary = _PRESET_ANIMATIONS[secondary][0]

        base = np.asarray(gen_primary(t), dtype=np.float32)  # (T, 17, 2)
        accent = np.asarray(gen_secondary(t), dtype=np.float32)

        # 在每个节拍附近用 energy 加权混合 secondary
        beat_pulse = np.zeros(frame_count, dtype=np.float32)
        for b in beats:
            if 0 <= b < frame_count:
                # 高斯脉冲
                half_width = max(1, int(fps * 0.15))
                lo = max(0, b - half_width)
                hi = min(frame_count, b + half_width + 1)
                xs = np.arange(lo, hi)
                gauss = np.exp(-((xs - b) ** 2) / (2 * (half_width / 2.0) ** 2))
                beat_pulse[lo:hi] = np.maximum(beat_pulse[lo:hi], gauss)

        # 能量归一化到 [0, 1] 用于调制
        if energy.max() > 0:
            energy_norm = energy / energy.max()
        else:
            energy_norm = np.zeros_like(energy)

        mix = (0.6 + 0.4 * energy_norm) * (1.0 - 0.5 * beat_pulse)
        mix = mix[:, None, None]  # 广播到 (T, 17, 2)
        keypoints = base * mix + accent * (1.0 - mix)

        self._emit_progress(frame_count, frame_count, "audio2motion")
        return keypoints, frame_count, duration

    # ------------------------------------------------------------------
    # text2motion 方式
    # ------------------------------------------------------------------
    def _generate_text2motion(
        self,
        prompt: str,
        duration: float,
        fps: int,
    ) -> tuple[Any, int, float]:
        """使用文本驱动模型生成动作。

        当模型不可用或推理失败时，回退到根据 prompt 关键词选择预设动作
        的降级方案，保证 ``run`` 始终有输出。

        Parameters
        ----------
        prompt:
            动作描述文本。
        duration:
            动作时长（秒）。
        fps:
            帧率。

        Returns
        -------
        tuple[numpy.ndarray, int, float]
            ``(keypoints, frame_count, duration)``。
        """
        import numpy as np  # type: ignore

        frame_count = max(1, int(round(duration * fps)))
        t = np.linspace(0.0, duration, frame_count, dtype=np.float32)

        # 尝试调用真实模型
        if self._pipeline is not None and self._tokenizer is not None:
            try:
                keypoints = self._run_text2motion_model(prompt, duration, fps)
                if keypoints is not None:
                    self._logger.info(
                        "text2motion generated %d frames from prompt %r.",
                        keypoints.shape[0], prompt,
                    )
                    self._emit_progress(keypoints.shape[0], frame_count, "text2motion")
                    # 对齐到期望帧数
                    keypoints = self._resize_keypoints(keypoints, frame_count)
                    return keypoints, frame_count, duration
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "text2motion model inference failed: %s. "
                    "Falling back to keyword-based preset selection.", exc,
                )

        # 降级：根据 prompt 关键词匹配预设动作
        preset = self._match_preset_from_text(prompt)
        self._logger.info(
            "text2motion fallback: matched preset %r for prompt %r.",
            preset, prompt,
        )
        generator = _PRESET_ANIMATIONS[preset][0]
        keypoints = np.asarray(generator(t), dtype=np.float32)
        self._emit_progress(frame_count, frame_count, f"text2motion:fallback:{preset}")
        return keypoints, frame_count, duration

    def _run_text2motion_model(
        self, prompt: str, duration: float, fps: int
    ) -> Any | None:
        """调用 MotionGPT 模型生成关键点。

        不同 MotionGPT 实现的 API 差异较大，这里采用通用的 generate 调用
        并尽力解析输出。失败时返回 ``None`` 由调用方降级处理。
        """
        import numpy as np  # type: ignore
        import torch  # type: ignore

        device = self._resolve_device()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:  # noqa: BLE001
            pass

        with torch.inference_mode():
            output = self._pipeline.generate(**inputs, max_length=512)

        # 尝试从输出中解析关键点序列
        # MotionGPT 通常返回 token 序列，需通过模型的 motion decoder 解码。
        # 这里尽力兼容：优先调用模型的专用解码方法。
        decoder = getattr(self._pipeline, "decode_motion", None) or getattr(
            self._pipeline, "motion_decoder", None
        )
        if callable(decoder):
            try:
                motion = decoder(output)
                arr = np.asarray(motion, dtype=np.float32)
                # 期望形状 (T, K, D) 或 (1, T, K, D)
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim == 3:
                    # 若关键点数 != 17，截断或补零
                    arr = self._adjust_keypoints(arr, _NUM_COCO_KEYPOINTS)
                    return arr
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _adjust_keypoints(motion: np.ndarray, target_kp: int) -> np.ndarray:
        """将关键点数量调整到 ``target_kp``（截断或补零）。"""
        import numpy as np  # type: ignore

        if motion.ndim < 2:
            return motion
        kp = motion.shape[1]
        if kp == target_kp:
            return motion
        if kp > target_kp:
            return motion[:, :target_kp, ...]
        # 补零
        pad_shape = list(motion.shape)
        pad_shape[1] = target_kp - kp
        pad = np.zeros(pad_shape, dtype=motion.dtype)
        return np.concatenate([motion, pad], axis=1)

    @staticmethod
    def _resize_keypoints(keypoints: np.ndarray, target_frames: int) -> np.ndarray:
        """将关键点序列重采样到 ``target_frames`` 帧。"""
        import numpy as np  # type: ignore

        cur = keypoints.shape[0]
        if cur == target_frames:
            return keypoints
        if cur == 0:
            rest = np.array(_REST_POSE, dtype=np.float32)
            return np.broadcast_to(rest, (target_frames, _NUM_COCO_KEYPOINTS, 2)).copy()
        # 线性插值重采样
        xs_src = np.linspace(0, 1, cur)
        xs_dst = np.linspace(0, 1, target_frames)
        out = np.zeros((target_frames, keypoints.shape[1], keypoints.shape[2]),
                       dtype=keypoints.dtype)
        for k in range(keypoints.shape[1]):
            for d in range(keypoints.shape[2]):
                out[:, k, d] = np.interp(xs_dst, xs_src, keypoints[:, k, d])
        return out

    @staticmethod
    def _match_preset_from_text(prompt: str) -> str:
        """根据 prompt 中的关键词匹配预设动作名称。"""
        text = prompt.lower()
        # 关键词 -> 预设名（按优先级排序）
        rules = [
            (["wave", "hello", "hi", "挥手", "你好", "招呼"], "wave"),
            (["bow", "bend", "鞠躬", "鞠躬"], "bow"),
            (["nod", "yes", "点头", "同意"], "nod"),
            (["shake", "no", "摇头", "否定"], "shake_head"),
            (["clap", "applaud", "鼓掌", "拍手"], "clap"),
            (["walk", "step", "走", "走路", "行走"], "walk"),
            (["dance", "跳舞", "舞蹈"], "dance"),
            (["raise", "hand up", "举手", "举起"], "raise_hand"),
            (["sit", "seat", "坐", "坐下"], "sit"),
            (["stand", "still", "站", "站立"], "stand"),
            (["jump", "leap", "跳", "跳跃"], "jump"),
            (["point", "指引", "指向", "指出"], "point"),
            (["think", "consider", "思考", "想"], "thinking"),
            (["stretch", "extend", "伸展", "拉伸"], "stretch"),
            (["turn", "spin", "转身", "转身", "转圈"], "turn_around"),
        ]
        for keywords, name in rules:
            if any(kw in text for kw in keywords):
                return name
        return "wave"

    # ------------------------------------------------------------------
    # 音频处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio_waveform(audio: Any) -> tuple[Any, int]:
        """从 AudioData / ndarray / 文件路径加载波形与采样率。"""
        if isinstance(audio, AudioData):
            return audio.waveform, audio.sample_rate
        try:
            import numpy as np  # type: ignore
            if isinstance(audio, np.ndarray):
                return audio, 22050
        except ImportError:
            pass
        if isinstance(audio, str):
            # 复用音频域的加载逻辑
            from mosaic.nodes.audio._base import BaseAudioNode
            return BaseAudioNode._load_audio(audio)
        raise TypeError(
            f"Expected AudioData, numpy.ndarray, or file path (str), "
            f"got {type(audio).__name__}."
        )

    @staticmethod
    def _audio_duration(waveform: Any, sample_rate: int) -> float:
        """计算音频时长（秒）。"""
        if waveform is None:
            return 0.0
        try:
            import numpy as np  # type: ignore
            if isinstance(waveform, np.ndarray):
                return float(waveform.shape[-1]) / float(sample_rate)
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        """转单声道。"""
        try:
            import numpy as np  # type: ignore
            if isinstance(waveform, np.ndarray) and waveform.ndim == 2:
                return np.mean(waveform, axis=0)
        except ImportError:
            pass
        return waveform

    def _detect_beats(
        self,
        waveform: Any,
        sample_rate: int,
        fps: int,
        frame_count: int,
    ) -> list[int]:
        """检测音频节拍，返回节拍对应的帧索引列表。

        优先使用 ``librosa`` 的 onset/beat 检测；缺失时回退到基于 RMS
        能量峰值的简化估计。
        """
        import numpy as np  # type: ignore

        mono = self._to_mono(waveform)
        if mono is None or len(mono) == 0:
            return []

        try:
            import librosa  # type: ignore

            tempo, beat_frames = librosa.beat.beat_track(
                y=np.asarray(mono, dtype=np.float32), sr=sample_rate
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            return [int(round(bt * fps)) for bt in beat_times if 0 <= round(bt * fps) < frame_count]
        except ImportError:
            pass

        # 回退：RMS 能量峰值
        hop = max(1, sample_rate // fps)
        rms = []
        arr = np.asarray(mono, dtype=np.float32)
        for i in range(0, len(arr) - hop, hop):
            rms.append(float(np.sqrt(np.mean(arr[i:i + hop] ** 2))))
        rms = np.array(rms, dtype=np.float32)
        if len(rms) == 0:
            return []
        # 简单峰值检测
        threshold = np.mean(rms) + 0.5 * np.std(rms)
        beats = []
        for i in range(1, len(rms) - 1):
            if rms[i] > threshold and rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]:
                beats.append(i)
        # 限制最小间隔，避免过密
        min_gap = max(1, int(fps * 0.2))
        filtered: list[int] = []
        for b in beats:
            if not filtered or b - filtered[-1] >= min_gap:
                filtered.append(b)
        return filtered

    def _frame_energy(
        self, waveform: Any, sample_rate: int, frame_count: int
    ) -> Any:
        """计算每帧的 RMS 能量，返回形状 (frame_count,) 的数组。"""
        import numpy as np  # type: ignore

        mono = self._to_mono(waveform)
        if mono is None or len(mono) == 0:
            return np.zeros(frame_count, dtype=np.float32)
        arr = np.asarray(mono, dtype=np.float32)
        hop = max(1, len(arr) // frame_count)
        energy = []
        for i in range(frame_count):
            start = i * hop
            end = min(len(arr), start + hop)
            if start >= end:
                energy.append(0.0)
            else:
                energy.append(float(np.sqrt(np.mean(arr[start:end] ** 2))))
        return np.array(energy, dtype=np.float32)

    # ------------------------------------------------------------------
    # 平滑
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth_keypoints(keypoints: np.ndarray, window: int = 5) -> np.ndarray:
        """对关键点序列做滑动平均平滑。

        Parameters
        ----------
        keypoints:
            形状 ``(T, K, D)`` 的关键点数组。
        window:
            滑动窗口大小（奇数）。

        Returns
        -------
        numpy.ndarray
            平滑后的关键点，形状不变。
        """
        import numpy as np  # type: ignore

        if window < 2:
            return keypoints
        window = window if window % 2 == 1 else window + 1
        # 对时间维做边缘填充后，逐通道做一维卷积滑动平均
        pad = window // 2
        padded = np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window
        T, K, D = keypoints.shape
        out = np.empty_like(keypoints)
        for k in range(K):
            for d in range(D):
                out[:, k, d] = np.convolve(
                    padded[:, k, d], kernel, mode="valid"
                )[:T]
        return out

    # ------------------------------------------------------------------
    # 卸载 / 规格
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """释放模型资源。

        ``preset`` / ``audio2motion`` 方式无模型需释放；``text2motion``
        方式释放模型与 tokenizer。
        """
        self._pipeline = None
        self._tokenizer = None
        self._loaded = False
        self._logger.info(
            "MotionGenerator unloaded (method=%s).", self._method
        )

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息（VRAM、许可证、方式、预设数）。"""
        if self._method == "text2motion":
            model_info = self._build_model_info(self._model_name)
        else:
            # preset / audio2motion 无模型
            model_info = {
                "name": f"<{self._method}>",
                "source": "builtin",
                "license": "MIT License (builtin animations)",
                "vram_gb": 0.0 if self._method == "preset" else 1.0,
                "dtype": "float32",
                "device": "cpu",
            }
        model_info["method"] = self._method
        model_info["skeleton_type"] = self._skeleton_type
        model_info["num_presets"] = len(_PRESET_ANIMATIONS)
        model_info["preset_names"] = sorted(_PRESET_ANIMATIONS.keys())
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=model_info,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<MotionGenerator name={self.name!r} method={self._method!r} "
            f"skeleton={self._skeleton_type!r} state={status}>"
        )
