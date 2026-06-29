# `Upscaler`

**模块**：`mosaic.nodes.image.upscaler`
**继承**：`BaseImageNode`

## 描述

超分辨率节点。

将低分辨率图片放大并增强画质。

Parameters
----------
model:
    HuggingFace 模型标识，默认
    ``"stabilityai/stable-diffusion-x4-upscaler"``。
**kwargs:
    透传给 :class:`BaseImageNode` 的参数。

Examples
--------
>>> from PIL import Image
>>> upscaler = Upscaler()
>>> low_res = Image.open("low_res.png")
>>> result = upscaler(MosaicData(
...     image=low_res,
...     prompt="highly detailed, sharp focus",
... ))
>>> result["image"].save("high_res.png")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'upscaler'` |
| `description` | `'Upscale a low-resolution image to higher resolution while enhancing details using Stable Diffusion x4 Upscaler.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['image', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'stabilityai/stable-diffusion-x4-upscaler', **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明，含模型信息。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载 diffusers Pipeline 到 GPU/CPU。

通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
以免递归。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

执行超分辨率放大。

Parameters
----------
input_data:
    必须包含 ``image`` (PIL.Image)；可选 ``prompt`` (str)、
    ``scale_factor`` (int, 默认 4)、``num_inference_steps`` (int, 默认 20)、
    ``seed`` (int)。

Returns
-------
MosaicData
    包含 ``image`` (PIL.Image)、``original_size`` (tuple)、
    ``output_size`` (tuple)。

Raises
------
ValueError
    缺少 ``image``。
TypeError
    ``image`` 不是 PIL.Image。

### `unload(self) -> 'None'`

释放 diffusers Pipeline。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
