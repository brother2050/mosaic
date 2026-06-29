# `ImageToVideo`

**模块**：`mosaic.nodes.video.image_to_video`
**继承**：`BaseVideoNode`

## 描述

图生视频节点。

根据输入图片生成短视频，基于 Stable Video Diffusion (SVD)。
SVD 不使用文字 prompt，运动幅度由 ``motion_bucket_id`` 控制。

Parameters
----------
model:
    模型标识，默认 ``"stabilityai/stable-video-diffusion-img2vid-xt"``。
    显存不足时可切换 ``"stabilityai/stable-video-diffusion-img2vid"``
    （14 帧，更轻量）。
device:
    推理设备，默认 ``"cuda"``。
dtype:
    推理精度，默认 ``"float16"``。
enable_vae_slicing:
    是否启用 VAE slicing 以节省显存，默认 ``True``。
decode_chunk_size:
    VAE 解码分块大小，``None`` 表示由 pipeline 决定（一次解码全部帧，
    显存占用较高）。显存不足时可设为较小值（如 8）以降低峰值占用。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
>>> from PIL import Image
>>> i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid-xt")
>>> result = i2v(MosaicData(
...     image=Image.open("photo.jpg"),
...     num_frames=25,
...     fps=7,
...     motion_bucket_id=127,
... ))
>>> video = result["video"]  # VideoData

显存不足时使用更轻量的 14 帧模型：
>>> i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'image-to-video'` |
| `description` | `'Generate video from an input image using Stable Video Diffusion. Motion intensity is controlled by motion_bucket_id; no text prompt is used.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['image', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'stabilityai/stable-video-diffusion-img2vid-xt', device: 'str' = 'cuda', dtype: 'str' = 'float16', enable_vae_slicing: 'bool' = True, decode_chunk_size: 'int | None' = None, **kwargs: 'Any') -> 'None'`

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

执行图生视频。

Parameters
----------
input_data:
    必须包含 ``image`` (PIL.Image)；可选 ``num_frames`` (int,
    默认 25)、``fps`` (int, 默认 7)、``motion_bucket_id``
    (int, 默认 127, 范围 1-255)、``noise_level`` (float,
    默认 0.02)、``num_inference_steps`` (int, 默认 25)、
    ``decode_chunk_size`` (int)、``seed`` (int)。

Returns
-------
MosaicData
    包含 ``video`` (VideoData)、``seed`` (int)、``num_frames``
    (int)、``duration`` (float)。

Raises
------
ValueError
    缺少 ``image`` 或 ``image`` 非 PIL.Image。
RuntimeError
    显存不足时抛出，附带建议。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
