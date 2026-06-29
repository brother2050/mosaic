# `Livestreamer`

**模块**：`mosaic.nodes.export.livestream`
**继承**：`Node`

## 描述

直播推流节点。

将帧列表通过 RTMP/SRT 协议推流到直播平台。

Parameters
----------
protocol:
    推流协议，``"rtmp"`` 或 ``"srt"``，默认 ``"rtmp"``。
codec:
    视频编码器，默认 ``"libx264"``。
bitrate:
    推流比特率，默认 ``"4M"``。
fps:
    帧率，默认 ``24``。
resolution:
    分辨率 ``(width, height)``，默认 ``(1920, 1080)``。
bus:
    事件总线实例，``None`` 使用全局单例。

Examples
--------
>>> streamer = Livestreamer(
...     protocol="rtmp",
...     bitrate="6M",
...     fps=30,
...     resolution=(1280, 720),
... )
>>> result = streamer(MosaicData(
...     frames=frames,
...     stream_url="rtmp://live.example.com/stream/key",
... ))
>>> result["status"]  # "completed" or "failed"

SRT 低延迟推流：
>>> streamer = Livestreamer(protocol="srt", bitrate="4M")
>>> result = streamer(MosaicData(
...     frames=frames,
...     stream_url="srt://server:port?streamid=xxx",
... ))

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'livestreamer'` |
| `domain` | `'export'` |
| `description` | `'Stream video frames to live platforms via RTMP/SRT protocol. Supports audio merging and real-time encoding.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'image', 'mosaic']` |
| `output_types` | `['file']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, protocol: 'str' = 'rtmp', codec: 'str' = 'libx264', bitrate: 'str' = '4M', fps: 'int' = 24, resolution: 'tuple[int, int]' = (1920, 1080), bus: 'EventBus | None' = None, **kwargs: 'Any') -> 'None'`

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

执行直播推流。

Parameters
----------
input_data:
    必须包含 ``frames`` (list[PIL.Image]) 和 ``stream_url`` (str)。
    可选：``fps`` (int，覆盖构造函数)、``audio`` (AudioData)。

Returns
-------
MosaicData
    包含 ``status``/``stream_url``/``frames_sent``/``duration``。
    失败时包含 ``error``。

Raises
------
ValueError
    缺少 ``frames`` 或 ``stream_url``。

### `unload(self) -> 'None'`

释放资源：关闭推流进程。

如果有正在运行的 FFmpeg 推流进程，会发送终止信号并等待退出。
