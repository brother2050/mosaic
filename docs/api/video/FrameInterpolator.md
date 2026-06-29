# `FrameInterpolator`

**模块**：`mosaic.nodes.video.frame_interpolation`
**继承**：`BaseVideoNode`

## 描述

视频插帧节点。

在相邻帧之间生成中间帧以提升帧率，支持 ``rife`` / ``film`` / ``linear``
三种方法，输出统一为 :class:`VideoData`。

Parameters
----------
model:
    模型路径或标识，``None`` 时使用方法对应的默认模型。
    ``linear`` 方法忽略此参数。
method:
    插值方法，可选 ``"rife"`` / ``"film"`` / ``"linear"``，默认
    ``"rife"``。
device:
    推理设备，默认 ``"cuda"``；``linear`` 方法忽略。
dtype:
    推理精度，默认 ``"float16"``（仅部分后端生效）。
chunk_size:
    分段处理的相邻帧对数，默认 64。增大可提升吞吐，但会增加显存占用。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
使用 RIFE 做 2 倍插帧：
>>> fi = FrameInterpolator(method="rife", model="/path/to/rife_v4.onnx")
>>> result = fi(MosaicData(video=input_video_data, scale_factor=2))
>>> out = result["video"]  # VideoData
>>> result["new_fps"], result["new_frame_count"]

使用线性插值做 4 倍插帧（CPU 友好）：
>>> fi = FrameInterpolator(method="linear")
>>> result = fi(MosaicData(video=input_video_data, scale_factor=4))

按目标帧率插帧：
>>> result = fi(MosaicData(video=input_video_data, target_fps=60))

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'frame-interpolation'` |
| `description` | `'Interpolate intermediate frames between existing video frames to increase fps. Supports RIFE (ONNX), FILM, and linear blending.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str | None' = None, method: 'str' = 'rife', device: 'str' = 'cuda', dtype: 'str' = 'float16', chunk_size: 'int' = 64, **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明，含模型信息。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载视频模型到 GPU/CPU。

通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
以免递归。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

执行视频插帧。

Parameters
----------
input_data:
    必须包含 ``video`` (:class:`VideoData`)；可选 ``target_fps``
    (int, 与 ``scale_factor`` 二选一，优先级更高)、``scale_factor``
    (int, 默认 2)。

Returns
-------
MosaicData
    包含 ``video`` (VideoData, 插帧后)、``original_fps`` (int)、
    ``new_fps`` (int)、``original_frame_count`` (int)、
    ``new_frame_count`` (int)，以及 ``method`` (str)、
    ``num_passes`` (int)、``duration`` (float)。

Raises
------
ValueError
    缺少 ``video``、``video`` 非 :class:`VideoData`、视频无帧，
    或 ``scale_factor`` 非法。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
