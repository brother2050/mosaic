# `TextToVideo`

**模块**：`mosaic.nodes.video.text_to_video`
**继承**：`BaseVideoNode`

## 描述

文生视频节点。

根据文字描述生成视频，基于 CogVideoX。

Parameters
----------
model:
    模型标识，默认 ``"THUDM/CogVideoX-5b"``。
    显存不足时可切换 ``"THUDM/CogVideoX-2b"``。
device:
    推理设备，默认 ``"cuda"``。
dtype:
    推理精度，默认 ``"float16"``。
enable_attention_slicing:
    是否启用 attention slicing 以节省显存，默认 ``True``。
enable_vae_slicing:
    是否启用 VAE slicing 以节省显存，默认 ``True``。
enable_vae_tiling:
    是否启用 VAE tiling 以进一步降低 VAE 解码显存峰值，默认 ``True``。
    对 22GB 显卡（如 A10）尤其重要，可将 VAE 解码显存从 ~5GB 降至 ~2GB。
enable_sequential_cpu_offload:
    是否启用顺序 CPU offload（逐层 GPU↔CPU 搬运），默认 ``False``。
    显存不足时开启可避免 OOM，但会增加推理时间约 2-3 倍。
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
>>> t2v = TextToVideo(model="THUDM/CogVideoX-5b")
>>> result = t2v(MosaicData(
...     prompt="一只猫在草地上奔跑，阳光明媚",
...     num_frames=49,
...     fps=8,
... ))
>>> video = result["video"]  # VideoData

显存不足时使用 2b 版本：
>>> t2v = TextToVideo(model="THUDM/CogVideoX-2b")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'text-to-video'` |
| `description` | `'Generate video from text descriptions using CogVideoX. Supports negative prompts, duration control, and guidance scale.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['text', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'THUDM/CogVideoX-5b', device: 'str' = 'cuda', dtype: 'str' = 'float16', enable_attention_slicing: 'bool' = True, enable_vae_slicing: 'bool' = True, enable_vae_tiling: 'bool' = True, enable_sequential_cpu_offload: 'bool' = False, **kwargs: 'Any') -> 'None'`

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
    ``num_frames`` (int, 默认 49)、``width`` (int, 默认 720)、
    ``height`` (int, 默认 480)、``num_inference_steps`` (int, 默认 50)、
    ``guidance_scale`` (float, 默认 6.0)、``fps`` (int, 默认 8)、
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
