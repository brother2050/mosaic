# `WanVideo`

**模块**：`mosaic.nodes.video.wan_video`
**继承**：`BaseVideoNode`

## 描述

Wan2.1 / Wan2.2 文生视频节点。

基于 Wan-AI 的 DiT 视频生成模型，支持中英文提示词。

Parameters
----------
model:
    模型标识，默认 ``"Wan-AI/Wan2.1-T2V-14B-Diffusers"``。
    显存不足时可切换 ``"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"``。
    支持 Wan2.2：``"Wan-AI/Wan2.2-T2V-A14B-Diffusers"``。
    **注意**：必须使用带 ``-Diffusers`` 后缀的仓库名。
    若传入不带后缀的名称，会自动添加。
device:
    推理设备，默认 ``"cuda"``。
dtype:
    推理精度，默认 ``"float16"``。Wan2.2 推荐 ``"bfloat16"``。
enable_cpu_offload:
    是否启用 ``enable_model_cpu_offload()``，默认 ``True``。
    将模型各组件按需从 CPU 移到 GPU，显著降低显存峰值。
enable_vae_tiling:
    是否启用 VAE tiling，默认 ``True``。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
>>> wan = WanVideo(model="Wan-AI/Wan2.1-T2V-14B-Diffusers")
>>> result = wan(MosaicData(
...     prompt="一只猫在海滩上散步，夕阳西下",
...     num_frames=81,
...     fps=16,
... ))
>>> video = result["video"]  # VideoData

显存不足时使用 1.3B 版本：
>>> wan = WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

使用 Wan2.2：
>>> wan = WanVideo(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", dtype="bfloat16")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'wan-video'` |
| `description` | `'Generate video from text using Wan2.1/Wan2.2 DiT models. Supports Chinese & English prompts, negative prompts, and VAE tiling for memory efficiency.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['text', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'Wan-AI/Wan2.1-T2V-14B-Diffusers', device: 'str' = 'cuda', dtype: 'str' = 'float16', enable_cpu_offload: 'bool' = True, enable_vae_tiling: 'bool' = True, **kwargs: 'Any') -> 'None'`

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
    ``num_frames`` (int, 默认 81)、``width`` (int, 默认 1280)、
    ``height`` (int, 默认 720)、``num_inference_steps`` (int, 默认 30)、
    ``guidance_scale`` (float, 默认 5.0)、``fps`` (int, 默认 16)、
    ``seed`` (int)。

Returns
-------
MosaicData
    包含 ``video`` (VideoData)、``prompt`` (str)、``seed`` (int)、
    ``num_frames`` (int)、``duration`` (float)。

Raises
------
ValueError
    缺少 ``prompt`` 或 ``prompt`` 非字符串。
RuntimeError
    显存不足时抛出，附带建议。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
