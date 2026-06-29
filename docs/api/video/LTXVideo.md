# `LTXVideo`

**模块**：`mosaic.nodes.video.ltx_video`
**继承**：`BaseVideoNode`

## 描述

LTX-Video 文生视频节点。

基于 Lightricks 的高效视频生成模型，支持快速生成。

Parameters
----------
model:
    模型标识，默认 ``"Lightricks/LTX-Video"``。
device:
    推理设备，默认 ``"cuda"``。
dtype:
    推理精度，默认 ``"bfloat16"``（LTX 推荐 bf16）。
enable_cpu_offload:
    是否启用 ``enable_model_cpu_offload()``，默认 ``True``。
enable_vae_tiling:
    是否启用 VAE tiling，默认 ``True``。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
>>> ltx = LTXVideo()
>>> result = ltx(MosaicData(
...     prompt="A cat walking on the beach at sunset",
...     num_frames=97,
...     fps=30,
... ))
>>> video = result["video"]

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'ltx-video'` |
| `description` | `'Generate video from text using Lightricks LTX-Video. Fast generation, supports real-time applications.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['text', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'Lightricks/LTX-Video', device: 'str' = 'cuda', dtype: 'str' = 'bfloat16', enable_cpu_offload: 'bool' = True, enable_vae_tiling: 'bool' = True, **kwargs: 'Any') -> 'None'`

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

执行文生视频。

Parameters
----------
input_data:
    必须包含 ``prompt`` (str)；可选 ``negative_prompt`` (str)、
    ``num_frames`` (int, 默认 97)、``width`` (int, 默认 768)、
    ``height`` (int, 默认 512)、``num_inference_steps`` (int, 默认 20)、
    ``guidance_scale`` (float, 默认 3.0)、``fps`` (int, 默认 30)、
    ``seed`` (int)。

Returns
-------
MosaicData
    包含 ``video`` (VideoData)、``prompt`` (str)、``seed`` (int)、
    ``num_frames`` (int)、``duration`` (float)。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
