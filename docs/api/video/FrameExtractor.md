# `FrameExtractor`

**模块**：`mosaic.nodes.video.frame_extractor`
**继承**：`BaseVideoNode`

## 描述

视频拆帧节点。

从视频中按模式提取帧，输出帧列表及其时间戳、帧率与时长。拆帧不依赖
任何模型，``_load_model`` 为空实现；模型生命周期仍由调度器统一管理
以保持节点语义一致。

Parameters
----------
model:
    模型标识，拆帧节点不使用模型，默认 ``""``。
device:
    推理设备，默认 ``"cuda"``（拆帧实际不使用，仅为保持接口一致）。
dtype:
    推理精度，默认 ``"float16"``（同上）。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
提取全部帧：

>>> extractor = FrameExtractor()
>>> result = extractor(MosaicData(video="/path/to/video.mp4", mode="all"))
>>> result["frames"], result["frame_count"], result["fps"]

按间隔提取（每隔 5 帧取一帧）：

>>> result = extractor(MosaicData(video=video_data, mode="interval", interval=5))

提取关键帧（路径输入自动流式处理，内存友好）：

>>> result = extractor(MosaicData(video="/path/to/video.mp4", mode="keyframe"))

按时间戳提取并以 numpy 数组形式返回：

>>> result = extractor(MosaicData(
...     video=video_data,
...     mode="timestamps",
...     timestamps=[1.0, 2.5, 4.0],
...     output_format="numpy",
... ))

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'frame-extractor'` |
| `description` | `"Extract frames from a video. Supports 'all', 'interval', 'keyframe' (pixel-diff based), and 'timestamps' modes, with PIL / numpy / file-path output formats."` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = '', device: 'str' = 'cuda', dtype: 'str' = 'float16', **kwargs: 'Any') -> 'None'`

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

执行视频拆帧。

Parameters
----------
input_data:
    必须包含 ``video`` (:class:`VideoData` 或 ``str`` 路径)；可选：

    * ``mode`` (str, 默认 ``"all"``) —— 拆帧模式，可选
      ``"all"`` / ``"interval"`` / ``"keyframe"`` / ``"timestamps"``。
    * ``interval`` (int, 默认 ``1``) —— ``mode="interval"`` 时生效，
      每隔 ``interval`` 帧取一帧。
    * ``timestamps`` (list[float]) —— ``mode="timestamps"`` 时必填，
      按时间戳（秒）提取对应帧。
    * ``output_format`` (str, 默认 ``"pil"``) —— 帧返回形态，可选
      ``"pil"`` / ``"numpy"`` / ``"path"``。
    * ``keyframe_threshold`` (float, 默认 ``10.0``) ——
      ``mode="keyframe"`` 时生效，像素差异阈值（0-255）。

Returns
-------
MosaicData
    包含 ``frames`` (list)、``frame_count`` (int, 提取到的帧数)、
    ``timestamps`` (list[float], 各帧时间戳)、``fps`` (int)、
    ``duration`` (float, 源视频总时长，秒)。

Raises
------
ValueError
    缺少 ``video``、``video`` 类型不支持、``mode`` 非法、
    ``mode="timestamps"`` 时未提供非空 ``timestamps``，或
    ``output_format`` 非法。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
