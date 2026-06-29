# `VideoContinuation`

**模块**：`mosaic.nodes.video.video_continuation`
**继承**：`BaseVideoNode`

## 描述

视频续写节点。

在输入视频末尾追加 CogVideoX 生成的新帧，通过交叉淡化实现平滑过渡，
输出“原始 + 续写”的完整视频以及“仅续写部分”。

由于 CogVideoX 为文生视频模型，续写片段以 ``prompt`` 驱动、并以原始
视频尾部帧的分辨率与重叠区交叉淡化作为视觉锚点。若续写风格与原始
差异较大，建议配合一致性 / 风格迁移节点使用。

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
**kwargs:
    透传给 :class:`BaseVideoNode` 的参数。

Examples
--------
>>> vc = VideoContinuation(model="THUDM/CogVideoX-5b")
>>> result = vc(MosaicData(
...     video=input_video_data,          # VideoData
...     prompt="镜头继续向前推进，人物走向远方",
...     num_frames=49,
...     overlap_frames=4,
... ))
>>> full = result["video"]              # VideoData: 原始 + 续写
>>> cont = result["continuation_video"]  # VideoData: 仅续写部分
>>> result["total_frames"], result["total_duration"]

显存不足时使用 2b 版本：
>>> vc = VideoContinuation(model="THUDM/CogVideoX-2b")

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'video-continuation'` |
| `description` | `"Extend an existing video by generating new frames from its tail using CogVideoX. Overlap frames are crossfaded for a smooth transition; the original video's fps is preserved."` |
| `version` | `'0.1.0'` |
| `input_types` | `['video', 'mosaic']` |
| `output_types` | `['video']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, model: 'str' = 'THUDM/CogVideoX-5b', device: 'str' = 'cuda', dtype: 'str' = 'float16', enable_attention_slicing: 'bool' = True, enable_vae_slicing: 'bool' = True, **kwargs: 'Any') -> 'None'`

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

执行视频续写。

取输入视频尾部 ``overlap_frames`` 帧作为过渡锚点，结合 ``prompt``
驱动 CogVideoX 生成 ``num_frames`` 帧新内容，对重叠区做交叉淡化后
与原始帧拼接，统一到原始视频的 ``fps``。

Parameters
----------
input_data:
    必须包含 ``video`` (:class:`VideoData`)；可选 ``prompt``
    (str, 缺省时使用平滑续写默认提示)、``num_frames`` (int,
    默认 49)、``overlap_frames`` (int, 默认 4)、``seed`` (int)、
    ``num_inference_steps`` (int, 默认 50)、``guidance_scale``
    (float, 默认 6.0)、``negative_prompt`` (str)。

Returns
-------
MosaicData
    包含 ``video`` (VideoData, 完整视频=原始+续写)、
    ``continuation_video`` (VideoData, 仅续写部分)、
    ``total_frames`` (int)、``total_duration`` (float)，
    以及 ``seed`` (int)、``overlap_frames`` (int)。

Raises
------
ValueError
    缺少 ``video`` 或 ``video`` 非 :class:`VideoData`，或视频无帧。
RuntimeError
    显存不足时抛出，附带建议。

### `unload(self) -> 'None'`

释放视频模型。

本方法执行实际资源清理。它由 ``Scheduler.release`` /
``Scheduler._evict`` 回调，不应在其中调用
``scheduler.release(self)`` 以免递归。
