# `VideoEncoder`

**模块**：`mosaic.nodes.export.video_encoder`
**继承**：`Node`

## 描述

视频编码封装节点。

将 ``PIL.Image`` 帧列表编码为标准视频文件，支持音视频合并与
字幕烧录。

Parameters
----------
format:
    输出格式，默认 ``"mp4"``。支持 mp4/avi/webm/mov/mkv。
codec:
    视频编码器，默认 ``"libx264"``。``None`` 时按格式自动选择。
quality:
    CRF 质量参数（0-51），越小质量越高，默认 ``23``。
preset:
    编码预设，可选 ultrafast/fast/medium/slow/veryslow，默认
    ``"medium"``。
audio_codec:
    音频编码器，默认 ``"aac"``。
pixel_format:
    像素格式，默认 ``"yuv420p"``（兼容性最好）。
bus:
    事件总线实例，``None`` 使用全局单例。

Examples
--------
>>> encoder = VideoEncoder(format="mp4", quality=20, preset="slow")
>>> result = encoder(MosaicData(
...     frames=[frame1, frame2, ...],
...     fps=30,
... ))
>>> result["output_path"]  # /tmp/mosaic_xxx.mp4

带音频合并：
>>> result = encoder(MosaicData(
...     frames=frames,
...     fps=30,
...     audio=audio_data,
... ))

带字幕烧录：
>>> result = encoder(MosaicData(
...     frames=frames,
...     fps=30,
...     subtitle=subtitle_data,
... ))

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'video-encoder'` |
| `domain` | `'export'` |
| `description` | `'Encode frames into a video file (mp4/avi/webm/mov/mkv). Supports audio merging and subtitle burning via FFmpeg.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'image', 'mosaic']` |
| `output_types` | `['file']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, format: 'str' = 'mp4', codec: 'str | None' = None, quality: 'int' = 23, preset: 'str' = 'medium', audio_codec: 'str | None' = 'aac', pixel_format: 'str' = 'yuv420p', bus: 'EventBus | None' = None, **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载环境：检测 FFmpeg 可用性。

导出域节点不需要加载 AI 模型，仅检测 FFmpeg 是否可用。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

执行视频编码。

Parameters
----------
input_data:
    必须包含 ``frames`` (list[PIL.Image]) 和 ``fps`` (int)。
    可选：``audio`` (AudioData)、``output_path`` (str)、
    ``bitrate`` (str)、``subtitle`` (SubtitleData)。

Returns
-------
MosaicData
    包含 ``output_path``/``format``/``codec``/``duration``/
    ``file_size``/``resolution``。

Raises
------
ValueError
    缺少 ``frames`` 或 ``fps``。

### `unload(self) -> 'None'`

释放资源（无持久化资源需要释放）。
