# `Stylizer`

**模块**：`mosaic.nodes.image.stylizer`
**继承**：`BaseImageNode`

## 描述

风格化节点。

将输入图片转换为指定的艺术风格。

Parameters
----------
model:
    HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
reference_image:
    可选的风格参考图，启用 IP-Adapter 模式时使用（高级功能，需额外安装
    ``diffusers`` 的 IP-Adapter 支持）。
**kwargs:
    透传给 :class:`BaseImageNode` 的参数。

Examples
--------
>>> from PIL import Image
>>> stylizer = Stylizer()
>>> photo = Image.open("photo.jpg")
>>> result = stylizer(MosaicData(
...     image=photo,
...     style="oil painting",
...     strength=0.65,
... ))
>>> result["image"].save("stylized.png")

Notes
-----
``strength`` 参数控制风格化强度，建议范围 0.5-0.7：
* 0.3-0.5：轻微风格化，保留原图大部分结构
* 0.5-0.7：最佳效果，风格明显但保留构图
* 0.7-0.9：强风格化，原图结构可能被改变

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'stylizer'` |
| `description` | `'Stylize an image into a specified artistic style (oil painting, watercolor, anime, etc.). Uses SDXL Img2Img under the hood.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['image', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'stabilityai/stable-diffusion-xl-base-1.0', reference_image: 'Any' = None, **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明，含支持的预设风格列表。

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

执行图片风格化。

Parameters
----------
input_data:
    必须包含 ``image`` (PIL.Image) 和 ``style`` (str)；
    可选 ``strength`` (float, 默认 0.65)、``prompt_extra`` (str)、
    ``num_inference_steps`` (int, 默认 30)、``seed`` (int)。

Returns
-------
MosaicData
    包含 ``image`` (PIL.Image)、``style`` (str)、``seed`` (int)。

Raises
------
ValueError
    缺少 ``image`` 或 ``style``。
TypeError
    ``image`` 不是 PIL.Image。

### `unload(self) -> 'None'`

释放 diffusers Pipeline。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
