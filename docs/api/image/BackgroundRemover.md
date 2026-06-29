# `BackgroundRemover`

**模块**：`mosaic.nodes.image.background_remover`
**继承**：`BaseImageNode`

## 描述

去背景节点。

去除图片背景，返回透明背景的主体图像和前景遮罩。

Parameters
----------
model:
    HuggingFace 模型标识，默认 ``"briaai/RMBG-2.0"``。
use_rembg:
    是否使用 ``rembg`` 库代替模型推理，默认 ``False``。
    ``rembg`` 是一个轻量的去背景库，无需下载大型模型。
**kwargs:
    透传给 :class:`BaseImageNode` 的参数。

Examples
--------
>>> from PIL import Image
>>> remover = BackgroundRemover()
>>> img = Image.open("photo.jpg")
>>> result = remover(MosaicData(image=img))
>>> result["image"].save("transparent.png")  # RGBA 透明背景
>>> result["mask"].save("mask.png")          # 灰度遮罩

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'background-remover'` |
| `description` | `'Remove the background from an image, returning a transparent RGBA image and a foreground mask. Supports both model-based and rembg backends.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['image', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'briaai/RMBG-2.0', use_rembg: 'bool' = False, **kwargs: 'Any') -> 'None'`

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

执行去背景。

Parameters
----------
input_data:
    必须包含 ``image`` (PIL.Image)。

Returns
-------
MosaicData
    包含 ``image`` (PIL.Image, RGBA 模式，透明背景)、
    ``mask`` (PIL.Image, 灰度遮罩)。

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
