# `TextToImage`

**模块**：`mosaic.nodes.image.text_to_image`
**继承**：`BaseImageNode`

## 描述

文生图节点。

根据文字提示词生成图片，基于 Stable Diffusion XL。

Parameters
----------
model:
    HuggingFace 模型标识，默认 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
**kwargs:
    透传给 :class:`BaseImageNode` 的参数。

Examples
--------
>>> t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
>>> result = t2i(MosaicData(
...     prompt="a cat sitting on a windowsill, oil painting style",
...     negative_prompt="blurry, low quality",
...     width=1024, height=1024,
... ))
>>> result["images"][0].save("cat.png")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'text-to-image'` |
| `description` | `'Generate images from text prompts using Stable Diffusion XL. Supports negative prompts, resolution, steps, guidance scale, and seed.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['text', 'mosaic']` |
| `output_types` | `['image']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'stabilityai/stable-diffusion-xl-base-1.0', device: 'str' = 'cuda', dtype: 'str' = 'float16', enable_attention_slicing: 'bool' = True, enable_vae_slicing: 'bool' = True, enable_model_cpu_offload: 'bool' = False, scheduler_name: 'str | None' = None, scheduler: 'Scheduler | None' = None, bus: 'EventBus | None' = None, **kwargs: 'Any') -> 'None'`

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

执行文生图。

Parameters
----------
input_data:
    必须包含 ``prompt`` (str)；可选 ``negative_prompt`` (str)、
    ``width`` (int, 默认 1024)、``height`` (int, 默认 1024)、
    ``num_inference_steps`` (int, 默认 30)、``guidance_scale``
    (float, 默认 7.5)、``seed`` (int)、``num_images`` (int, 默认 1)。

Returns
-------
MosaicData
    包含 ``images`` (list[PIL.Image])、``seed`` (int)、
    ``prompt`` (str)、``model_name`` (str)。

Raises
------
ValueError
    缺少 ``prompt`` 或 ``prompt`` 非字符串。

### `unload(self) -> 'None'`

释放 diffusers Pipeline。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
