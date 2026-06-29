# `MultiFormatExporter`

**模块**：`mosaic.nodes.export.multi_format_exporter`
**继承**：`Node`

## 描述

多格式导出节点。

将内容导出为多种格式，支持视频、图像、音频和字幕。

Parameters
----------
bus:
    事件总线实例，``None`` 使用全局单例。

Examples
--------
>>> exporter = MultiFormatExporter()
>>> result = exporter(MosaicData(
...     content_type="video",
...     data=video_data,
...     formats=["mp4", "gif", "webm"],
...     output_dir="/tmp/outputs",
... ))
>>> result["outputs"]  # {"mp4": "/tmp/outputs/output.mp4", ...}
>>> result["total_files"]  # 3

图像导出：
>>> result = exporter(MosaicData(
...     content_type="image",
...     data=image_data,
...     formats=["png", "jpg", "webp"],
... ))

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'multi-format-exporter'` |
| `domain` | `'export'` |
| `description` | `'Export content to multiple formats (video/image/audio/subtitle). Supports batch conversion with graceful error handling.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'image', 'audio', 'subtitle', 'mosaic']` |
| `output_types` | `['file']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, bus: 'EventBus | None' = None, **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载环境（无需加载模型，直接标记为已加载）。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

执行多格式导出。

Parameters
----------
input_data:
    必须包含：
    - ``content_type`` (str): "video"/"image"/"audio"/"subtitle"
    - ``data`` (VideoData/ImageData/AudioData/SubtitleData): 待导出数据
    - ``formats`` (list[str]): 目标格式列表
    可选：
    - ``output_dir`` (str): 输出目录，默认临时目录
    - ``quality`` (int): 质量参数

Returns
-------
MosaicData
    包含 ``outputs`` (dict[str, str])、``total_files`` (int)、
    ``total_size`` (int)。

Raises
------
ValueError
    缺少必要字段或内容类型不支持。

### `unload(self) -> 'None'`

释放资源（无持久化资源需要释放）。
