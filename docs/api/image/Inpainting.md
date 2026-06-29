# `Inpainting`

**模块**：`mosaic.nodes.image.inpainting`
**继承**：`BaseImageNode`

## 描述

局部重绘节点。

根据遮罩区域重新绘制图片内容。

Parameters
----------
model:
    HuggingFace 模型标识，默认
    ``"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"``。
**kwargs:
    透传给 :class:`BaseImageNode` 的参数。

Examples
--------
>>> from PIL import Image
>>> inpaint = Inpainting()
>>> original = Image.open("photo.jpg")
>>> mask = Image.open("mask.png")  # 白色区域为待重绘
>>> result = inpaint(MosaicData(
...     image=original,
...     mask_image=mask,
...     prompt="a red car",
... ))
>>> result["image"].save("result.png")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'inpainting'` |
| `description` | `'Inpaint specific regions of an image using a mask. Only the masked area is regenerated based on the prompt.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['image', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', **kwargs: 'Any') -> 'None'`

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

执行局部重绘。

Parameters
----------
input_data:
    必须包含 ``image`` (PIL.Image)、``mask_image`` (PIL.Image) 和
    ``prompt`` (str)；可选 ``negative_prompt`` (str)、
    ``num_inference_steps`` (int, 默认 30)、
    ``guidance_scale`` (float, 默认 7.5)、``seed`` (int)。

Returns
-------
MosaicData
    包含 ``image`` (PIL.Image)、``seed`` (int)。

Raises
------
ValueError
    缺少 ``image`` / ``mask_image`` / ``prompt``。
TypeError
    ``image`` 或 ``mask_image`` 不是 PIL.Image。

### `unload(self) -> 'None'`

释放 diffusers Pipeline。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
