# Mosaic 节点参考手册

> 全部 42 节点 + 4 个 TTS 后端的完整 API 参考。

所有节点的 `run` 方法签名统一为 `run(self, input_data: MosaicData) -> MosaicData`，参数通过 `input_data.get(key)` 读取。调用示例统一使用 `result = node.run(MosaicData(prompt="..."))`，输出字段通过 `result.get("key")` 获取。

## 目录

- [文本域 (text, 6 节点)](#文本域-text-6-节点)
- [图像域 (image, 6 节点)](#图像域-image-6-节点)
- [视频域 (video, 8 节点)](#视频域-video-8-节点)
- [音频域 (audio, 5 节点)](#音频域-audio-5-节点)
- [字幕域 (subtitle, 3 节点)](#字幕域-subtitle-3-节点)
- [一致性域 (consistency, 3 节点)](#一致性域-consistency-3-节点)
- [数字人域 (digital-human, 4 节点)](#数字人域-digital-human-4-节点)
- [导出域 (export, 3 节点)](#导出域-export-3-节点)
- [RAG 域 (rag, 4 节点)](#rag-域-rag-4-节点)
- [TTS 后端 (4 后端)](#tts-后端-4-后端)

---

## 文本域 (text, 6 节点)

文本域提供 6 个核心文本处理节点，基于 `transformers` 的因果语言模型加载与生成流程。文本节点继承自 `BaseTextNode`，其构造函数参数如下（子类默认继承）：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace 模型 ID 或本地路径 |
| `device_map` | str | `auto` | 传递给 `from_pretrained` 的 `device_map` |
| `torch_dtype` | str | `fp16` | 权重精度（`fp32`/`fp16`/`bf16`） |
| `trust_remote_code` | bool | `True` | 是否信任远程代码 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例，`None` 使用全局单例 |
| `bus` | EventBus \| None | `None` | 事件总线实例，`None` 使用全局单例 |
| `**kwargs` | Any | — | 透传给 Node 基类 |

### TextGenerator — 文本生成

**所属域**：`text`
**节点 ID**：`text-generator`
**一句话描述**：根据 prompt 生成一段文本。

#### 构造函数

继承 `BaseTextNode`，默认参数见本域开头的基类表格。

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 输入 prompt |
| `max_new_tokens` | int | ❌ | 512 | 最大生成 token 数 |
| `temperature` | float | ❌ | 0.7 | 采样温度 |
| `top_p` | float | ❌ | 0.9 | nucleus 采样阈值 |
| `do_sample` | bool | ❌ | True | 是否采样；`False` 时贪心解码 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `prompt` | str | 生成的文本 |
| `input_tokens` | int | 输入 token 数 |
| `output_tokens` | int | 输出 token 数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import TextGenerator

gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
result = gen.run(MosaicData(prompt="写一首关于春天的诗", temperature=0.8))
print(result.get("prompt"))
```

#### 许可证

因模型而异，参考 HF 模型页面。

#### stream 方法

支持流式生成，逐 token yield 输出。参数同 `run` 输入，用法见下方"流式生成"小节。

---

### Chat — 对话

**所属域**：`text`
**节点 ID**：`chat`
**一句话描述**：多轮对话节点，接收消息列表并返回回复。

#### 构造函数

继承 `BaseTextNode`，默认参数见本域开头的基类表格。

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `messages` | list[dict] | ✅ | — | 对话消息列表，格式 `[{"role": ..., "content": ...}]` |
| `system_prompt` | str | ❌ | — | 系统提示 |
| `max_new_tokens` | int | ❌ | 1024 | 最大生成 token 数 |
| `temperature` | float | ❌ | 0.7 | 采样温度 |
| `top_p` | float | ❌ | 0.9 | nucleus 采样阈值 |
| `do_sample` | bool | ❌ | True | 是否采样 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `reply` | str | 模型回复 |
| `messages` | list[dict] | 更新后的对话历史 |
| `input_tokens` | int | 输入 token 数 |
| `output_tokens` | int | 输出 token 数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import Chat

chat = Chat()
result = chat.run(MosaicData(
    messages=[
        {"role": "system", "content": "你是一个友好的助手"},
        {"role": "user", "content": "你好"},
    ],
))
print(result.get("reply"))

# 多轮：把上一轮的 messages 传入继续对话
messages = result.get("messages")
messages.append({"role": "user", "content": "推荐一本 Python 入门书"})
result = chat.run(MosaicData(messages=messages))
print(result.get("reply"))
```

#### 流式生成

`Chat` 和 `TextGenerator` 均支持流式生成，逐 token yield 输出，适合实时显示场景：

```python
from mosaic.nodes.text import Chat

chat = Chat()

# 流式对话
for chunk in chat.stream(MosaicData(
    messages=[{"role": "user", "content": "写一首五言绝句"}],
    temperature=0.7,
)):
    print(chunk, end="", flush=True)  # 逐 token 实时打印
```

```python
from mosaic.nodes.text import TextGenerator

gen = TextGenerator()

# 流式文本生成
for chunk in gen.stream(MosaicData(
    prompt="写一篇关于 AI 的短文",
    max_new_tokens=512,
    temperature=0.8,
)):
    print(chunk, end="", flush=True)
```

> **原理**：流式生成通过 `transformers.TextIteratorStreamer` 实现，在后台线程中运行模型推理，主线程中逐 token yield 输出。首批延迟通常 < 100ms。

---

### TextRewriter — 文本改写

**所属域**：`text`
**节点 ID**：`text-rewriter`
**一句话描述**：按指令改写文本，保持语义不变并提升表达质量。

#### 构造函数

继承 `BaseTextNode`，默认参数见本域开头的基类表格。

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | str | ✅ | — | 原文 |
| `instruction` | str | ❌ | `请改写以下文本，保持语义不变，提升表达质量。只输出改写后的文本，不要任何解释。` | 改写指令 |
| `max_new_tokens` | int | ❌ | 512 | 最大生成 token 数 |
| `temperature` | float | ❌ | 0.7 | 采样温度 |
| `top_p` | float | ❌ | 0.9 | nucleus 采样阈值 |
| `do_sample` | bool | ❌ | True | 是否采样 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `rewritten_text` | str | 改写后的文本 |
| `original_text` | str | 原文 |
| `input_tokens` | int | 输入 token 数 |
| `output_tokens` | int | 输出 token 数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import TextRewriter

rewriter = TextRewriter()
result = rewriter.run(MosaicData(
    text="这个产品很好，我很喜欢",
    instruction="改写为更正式的商务语气",
))
print(result.get("rewritten_text"))
```

---

### Translator — 翻译

**所属域**：`text`
**节点 ID**：`translator`
**一句话描述**：文本翻译节点，支持专用翻译模型（MarianMT/NLLB）与通用生成模式，将文本翻译为目标语言。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | 模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseTextNode`（`device_map`/`torch_dtype`/`trust_remote_code`/`scheduler`/`bus` 等） |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | str | ✅ | — | 原文 |
| `target_language` | str | ✅ | — | 目标语言 |
| `source_language` | str | ❌ | `auto` | 源语言（`auto` 自动检测） |
| `max_new_tokens` | int | ❌ | 512 | 最大生成 token 数 |
| `temperature` | float | ❌ | 0.3 | 采样温度 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `translated_text` | str | 翻译结果 |
| `source_language` | str | 源语言 |
| `target_language` | str | 目标语言 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import Translator

translator = Translator()
result = translator.run(MosaicData(text="你好世界", target_language="en"))
print(result.get("translated_text"))  # Hello world
```

---

### TextSummarizer — 摘要

**所属域**：`text`
**节点 ID**：`text-summarizer`
**一句话描述**：生成文本摘要，支持简洁/详细/要点三种风格。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | 模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseTextNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | str | ✅ | — | 原文 |
| `max_length` | int | ❌ | 150 | 摘要最大长度（字数） |
| `style` | str | ❌ | `concise` | 摘要风格，可选 `concise` / `detailed` / `bullet_points` |
| `max_new_tokens` | int | ❌ | 512 | 最大生成 token 数 |
| `temperature` | float | ❌ | 0.3 | 采样温度 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `summary` | str | 摘要 |
| `original_length` | int | 原文长度 |
| `summary_length` | int | 摘要长度 |
| `compression_ratio` | float | 压缩比 |
| `note` | str | 可选，仅原文过短跳过摘要时出现 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import TextSummarizer

summarizer = TextSummarizer()
result = summarizer.run(MosaicData(
    text=long_article,
    style="bullet_points",
    max_length=200,
))
print(result.get("summary"))
print(result.get("compression_ratio"))
```

---

### TextClassifier — 分类

**所属域**：`text`
**节点 ID**：`text-classifier`
**一句话描述**：文本分类节点，支持专用分类模型、LLM 生成选择（≤10 标签）与零样本 NLI（>10 标签）三种模式。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | 生成模型 ID |
| `zero_shot_model` | str | `facebook/bart-large-mnli` | 零样本分类模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseTextNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | str | ✅ | — | 输入文本 |
| `labels` | list[str] | ✅ | — | 自定义标签列表 |
| `multi_label` | bool | ❌ | False | 是否多标签分类 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `method` | str | 使用的分类方法 |
| `predicted_label` | str | 单标签模式下的预测标签 |
| `predicted_labels` | list[str] | 多标签模式下的预测标签列表 |
| `scores` | dict | 各标签的概率得分 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.text import TextClassifier

classifier = TextClassifier()

# 单标签分类
result = classifier.run(MosaicData(
    text="这款手机拍照效果不错",
    labels=["正面", "负面", "中性"],
))
print(result.get("predicted_label"))  # 正面

# 多标签分类
result = classifier.run(MosaicData(
    text="今天的会议讨论了产品路线图和技术架构",
    labels=["技术", "商业", "运营", "人事"],
    multi_label=True,
))
print(result.get("predicted_labels"))
```

---

## 图像域 (image, 6 节点)

基于 `diffusers` 的图像生成/编辑能力。图像节点继承自 `BaseImageNode`，其构造函数参数如下（子类默认继承）：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `enable_attention_slicing` | bool | `True` | 启用注意力切片 |
| `enable_vae_slicing` | bool | `True` | 启用 VAE 切片 |
| `enable_model_cpu_offload` | bool | `False` | 启用模型 CPU offload |
| `scheduler_name` | str \| None | `None` | 调度器名称 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |

### TextToImage — 文生图

**所属域**：`image`
**节点 ID**：`text-to-image`
**一句话描述**：根据文本生成图像。

#### 构造函数

继承 `BaseImageNode`，默认参数见本域开头的基类表格。

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `width` | int | ❌ | 1024 | 输出宽度 |
| `height` | int | ❌ | 1024 | 输出高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `num_images` | int | ❌ | 1 | 生成图像数量 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `images` | list[PIL.Image] | 生成的图像列表 |
| `seed` | int | 实际使用的随机种子 |
| `prompt` | str | 输入 prompt |
| `model_name` | str | 模型名称 |
| `num_images` | int | 生成图像数量 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import TextToImage

t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
result = t2i.run(MosaicData(
    prompt="A cute cat sitting on a windowsill, sunlight",
    negative_prompt="blurry, low quality",
    seed=42,
))
result.get("images")[0].save("cat.png")
```

---

### ImageToImage — 图生图

**所属域**：`image`
**节点 ID**：`image-to-image`
**一句话描述**：根据输入图和 prompt 生成新图。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-xl-refiner-1.0` | 模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseImageNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `strength` | float | ❌ | 0.75 | 变化强度（0-1） |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 输出图像 |
| `seed` | int | 实际使用的随机种子 |
| `prompt` | str | 输入 prompt |
| `model_name` | str | 模型名称 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import ImageToImage
from PIL import Image

i2i = ImageToImage(model="stabilityai/stable-diffusion-xl-refiner-1.0")
result = i2i.run(MosaicData(
    image=Image.open("input.jpg"),
    prompt="the same image in watercolor style",
    strength=0.6,
))
result.get("image").save("output.jpg")
```

---

### Inpainting — 局部重绘

**所属域**：`image`
**节点 ID**：`inpainting`
**一句话描述**：在 mask 区域内重绘图像。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | 模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseImageNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |
| `mask_image` | PIL.Image | ✅ | — | 蒙版（白色=重绘区域） |
| `prompt` | str | ✅ | — | 重绘描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 输出图像 |
| `seed` | int | 实际使用的随机种子 |
| `prompt` | str | 输入 prompt |
| `model_name` | str | 模型名称 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import Inpainting
from PIL import Image

inpaint = Inpainting(model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
result = inpaint.run(MosaicData(
    image=Image.open("room.jpg"),
    mask_image=Image.open("mask.png"),  # 白色区域被替换
    prompt="a beautiful garden with flowers",
))
result.get("image").save("new_room.jpg")
```

---

### Upscaler — 超分

**所属域**：`image`
**节点 ID**：`upscaler`
**一句话描述**：图像超分辨率放大。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-x4-upscaler` | 模型 ID |
| `**kwargs` | Any | — | 透传给 `BaseImageNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |
| `prompt` | str | ❌ | `""` | 提示文本 |
| `scale_factor` | int | ❌ | 4 | 放大倍数（范围 2-8） |
| `num_inference_steps` | int | ❌ | 20 | 推理步数 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 放大后的图像 |
| `original_size` | tuple | 原始尺寸 |
| `output_size` | tuple | 输出尺寸 |
| `seed` | int | 实际使用的随机种子 |
| `model_name` | str | 模型名称 |
| `scale_factor` | int | 放大倍数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import Upscaler
from PIL import Image

upscaler = Upscaler()
result = upscaler.run(MosaicData(
    image=Image.open("low_res.png"),
    scale_factor=4,
))
result.get("image").save("high_res.png")
```

---

### BackgroundRemover — 去背景

**所属域**：`image`
**节点 ID**：`background-remover`
**一句话描述**：移除图像背景，输出透明背景图像与蒙版。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `briaai/RMBG-2.0` | 背景移除模型 ID |
| `use_rembg` | bool | `False` | 是否使用 rembg 后端；`use_rembg=True` 时使用 rembg 库的 u2net 模型，首次运行自动下载（~176MB） |
| `**kwargs` | Any | — | 透传给 `BaseImageNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 透明背景图像（RGBA） |
| `mask` | PIL.Image | 灰度蒙版 |
| `model_name` | str | 模型名称 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import BackgroundRemover
from PIL import Image

remover = BackgroundRemover()
result = remover.run(MosaicData(image=Image.open("person.jpg")))
result.get("image").save("person_no_bg.png")
result.get("mask").save("mask.png")
```

---

### Stylizer — 风格化

**所属域**：`image`
**节点 ID**：`stylizer`
**一句话描述**：基于 IP-Adapter 的艺术风格化。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID |
| `reference_image` | PIL.Image \| None | `None` | 参考风格图像 |
| `**kwargs` | Any | — | 透传给 `BaseImageNode` |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |
| `style` | str | ✅ | — | 风格描述 |
| `strength` | float | ❌ | 0.65 | 风格强度 |
| `prompt_extra` | str | ❌ | `""` | 额外提示文本 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 输出图像 |
| `style` | str | 风格描述 |
| `seed` | int | 实际使用的随机种子 |
| `prompt` | str | 实际使用的 prompt |
| `model_name` | str | 模型名称 |
| `ip_adapter_enabled` | bool | 是否启用了 IP-Adapter |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.image import Stylizer
from PIL import Image

stylizer = Stylizer()
result = stylizer.run(MosaicData(
    image=Image.open("photo.jpg"),
    style="oil painting, van gogh style",
    strength=0.7,
))
result.get("image").save("stylized.jpg")
```

---

## 视频域 (video, 8 节点)

基于 `diffusers` 的视频生成/处理能力。

### TextToVideo — 文生视频（CogVideoX）

**所属域**：`video`
**节点 ID**：`text-to-video`
**一句话描述**：基于 CogVideoX 的文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `THUDM/CogVideoX-5b` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `enable_attention_slicing` | bool | `True` | 启用注意力切片 |
| `enable_vae_slicing` | bool | `True` | 启用 VAE 切片 |
| `enable_vae_tiling` | bool | `True` | 启用 VAE 瓦片化 |
| `enable_sequential_cpu_offload` | bool | `False` | 启用顺序 CPU offload |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_frames` | int | ❌ | 49 | 帧数 |
| `width` | int | ❌ | 720 | 视频宽度 |
| `height` | int | ❌ | 480 | 视频高度 |
| `num_inference_steps` | int | ❌ | 50 | 推理步数 |
| `guidance_scale` | float | ❌ | 6.0 | CFG 引导强度 |
| `fps` | int | ❌ | 8 | 帧率 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `prompt` | str | 输入 prompt |
| `seed` | int | 实际使用的随机种子 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.video import TextToVideo

t2v = TextToVideo(model="THUDM/CogVideoX-5b")
result = t2v.run(MosaicData(
    prompt="一只猫在草地上奔跑",
    num_frames=49,
    fps=8,
))
result.get("video").save("cat.mp4")
```

---

### WanVideo — 文生视频（Wan2.1）

**所属域**：`video`
**节点 ID**：`wan-video`
**一句话描述**：基于阿里通义万相 Wan 系列的文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `enable_cpu_offload` | bool | `True` | 启用 CPU offload |
| `enable_vae_tiling` | bool | `True` | 启用 VAE 瓦片化 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_frames` | int | ❌ | 81 | 帧数（4k+1） |
| `width` | int | ❌ | 1280 | 视频宽度 |
| `height` | int | ❌ | 720 | 视频高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 5.0 | CFG 引导强度 |
| `fps` | int | ❌ | 16 | 帧率 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `prompt` | str | 输入 prompt |
| `seed` | int | 实际使用的随机种子 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

#### 许可证

Apache-2.0

---

### HunyuanVideo — 文生视频（腾讯混元）

**所属域**：`video`
**节点 ID**：`hunyuan-video`
**一句话描述**：基于腾讯 HunyuanVideo 的大规模文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `hunyuanvideo-community/HunyuanVideo` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `bfloat16` | 推理精度 |
| `enable_cpu_offload` | bool | `True` | 启用 CPU offload |
| `enable_vae_tiling` | bool | `True` | 启用 VAE 瓦片化 |
| `enable_chunking` | bool | `True` | 启用 VAE chunking（专属优化） |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_frames` | int | ❌ | 129 | 帧数 |
| `width` | int | ❌ | 1280 | 视频宽度 |
| `height` | int | ❌ | 720 | 视频高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `fps` | int | ❌ | 24 | 帧率 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `prompt` | str | 输入 prompt |
| `seed` | int | 实际使用的随机种子 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

---

### LTXVideo — 文生视频（Lightricks）

**所属域**：`video`
**节点 ID**：`ltx-video`
**一句话描述**：基于 Lightricks LTX-Video 的轻量快速文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Lightricks/LTX-Video` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `bfloat16` | 推理精度 |
| `enable_cpu_offload` | bool | `True` | 启用 CPU offload |
| `enable_vae_tiling` | bool | `True` | 启用 VAE 瓦片化 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_frames` | int | ❌ | 97 | 帧数 |
| `width` | int | ❌ | 768 | 视频宽度 |
| `height` | int | ❌ | 512 | 视频高度 |
| `num_inference_steps` | int | ❌ | 20 | 推理步数 |
| `guidance_scale` | float | ❌ | 3.0 | CFG 引导强度 |
| `fps` | int | ❌ | 30 | 帧率 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `prompt` | str | 输入 prompt |
| `seed` | int | 实际使用的随机种子 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

---

### ImageToVideo — 图生视频

**所属域**：`video`
**节点 ID**：`image-to-video`
**一句话描述**：基于 SVD 的图像到视频生成。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-video-diffusion-img2vid-xt` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `enable_vae_slicing` | bool | `True` | 启用 VAE 切片 |
| `decode_chunk_size` | int \| None | `None` | 解码 chunk 大小 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image` | PIL.Image | ✅ | — | 输入图像 |
| `num_frames` | int | ❌ | 25 | 帧数 |
| `fps` | int | ❌ | 7 | 帧率 |
| `motion_bucket_id` | int | ❌ | 127 | 运动强度（范围 1-255） |
| `noise_level` | float | ❌ | 0.02 | 噪声水平 |
| `num_inference_steps` | int | ❌ | 25 | 推理步数 |
| `decode_chunk_size` | int | ❌ | — | 解码 chunk 大小（覆盖构造函数） |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `seed` | int | 实际使用的随机种子 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.video import ImageToVideo
from PIL import Image

i2v = ImageToVideo()
result = i2v.run(MosaicData(
    image=Image.open("scene.jpg"),
    num_frames=25,
    fps=7,
))
result.get("video").save("scene.mp4")
```

---

### VideoContinuation — 视频续写

**所属域**：`video`
**节点 ID**：`video-continuation`
**一句话描述**：在视频末尾续写新帧。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `THUDM/CogVideoX-5b` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `enable_attention_slicing` | bool | `True` | 启用注意力切片 |
| `enable_vae_slicing` | bool | `True` | 启用 VAE 切片 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video` | VideoData | ✅ | — | 输入视频 |
| `prompt` | str | ❌ | — | 续写描述 |
| `num_frames` | int | ❌ | 49 | 续写帧数 |
| `overlap_frames` | int | ❌ | 4 | 与原视频重叠的帧数 |
| `num_inference_steps` | int | ❌ | 50 | 推理步数 |
| `guidance_scale` | float | ❌ | 6.0 | CFG 引导强度 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 完整视频（原视频 + 续写部分） |
| `continuation_video` | VideoData | 仅续写部分 |
| `total_frames` | int | 总帧数 |
| `total_duration` | float | 总时长（秒） |
| `seed` | int | 实际使用的随机种子 |
| `overlap_frames` | int | 重叠帧数 |

---

### FrameInterpolator — 插帧

**所属域**：`video`
**节点 ID**：`frame-interpolator`
**一句话描述**：在视频帧之间插入中间帧以提升帧率。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str \| None | `None` | 模型 ID |
| `method` | str | `rife` | 插值方法 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `chunk_size` | int | 64 | 处理 chunk 大小 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video` | VideoData | ✅ | — | 输入视频 |
| `target_fps` | int | ❌ | — | 目标帧率（与 `scale_factor` 二选一） |
| `scale_factor` | int | ❌ | 2 | 放大倍数 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 插帧后的视频 |
| `original_fps` | int | 原始帧率 |
| `new_fps` | int | 新帧率 |
| `original_frame_count` | int | 原始帧数 |
| `new_frame_count` | int | 新帧数 |
| `method` | str | 插值方法 |
| `num_passes` | int | 插值遍数 |
| `duration` | float | 时长（秒） |

---

### FrameExtractor — 拆帧

**所属域**：`video`
**节点 ID**：`frame-extractor`
**一句话描述**：从视频提取帧为图像列表，支持多种模式。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `""` | 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video` | VideoData \| str | ✅ | — | 输入视频或视频文件路径 |
| `mode` | str | ❌ | `all` | 提取模式，可选 `all` / `interval` / `keyframe` / `timestamps` |
| `interval` | int | ❌ | 1 | 间隔（`interval` 模式） |
| `timestamps` | list[float] | ❌ | — | 时间戳列表（`timestamps` 模式） |
| `output_format` | str | ❌ | `pil` | 输出格式，可选 `pil` / `numpy` / `path` |
| `keyframe_threshold` | float | ❌ | 10.0 | 关键帧阈值（`keyframe` 模式） |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `frames` | list | 帧列表 |
| `frame_count` | int | 帧数 |
| `timestamps` | list[float] | 时间戳列表 |
| `fps` | int | 帧率 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.video import FrameExtractor

extractor = FrameExtractor()
result = extractor.run(MosaicData(video="input.mp4", mode="all"))
print(result.get("frame_count"))
```

---

## 音频域 (audio, 5 节点)

### TTS — 文本转语音

**所属域**：`audio`
**节点 ID**：`tts`
**一句话描述**：将文本转为语音，支持多个后端（详见 [TTS 后端](#tts-后端-4-后端)）。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | str | `auto` | 后端选择（`auto` 自动选择） |
| `model` | str | `edge-tts` | 模型 ID |
| `voice` | str \| None | `None` | 语音音色 |
| `language` | str | `zh` | 语言 |
| `emotion` | str | `neutral` | 情感 |
| `speed` | float | 1.0 | 语速 |
| `speaker` | str \| None | `None` | 说话人 |
| `stream_chunk_size` | int | 4096 | 流式 chunk 大小 |
| `max_sentence_length` | int | 200 | 单句最大字符数（超长自动切分） |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | str | ✅ | — | 输入文本 |
| `voice` | str | ❌ | — | 语音音色 |
| `language` | str | ❌ | — | 语言 |
| `emotion` | str | ❌ | — | 情感 |
| `speed` | float | ❌ | 1.0 | 语速 |
| `speaker` | str | ❌ | — | 说话人 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | AudioData | 合成的音频 |
| `text` | str | 输入文本 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.audio import TTS

tts = TTS(backend="chattts", language="zh")
result = tts.run(MosaicData(text="你好世界"))
result.get("audio").save("hello.wav")
```

详见 [TTS 完整指南](tts-guide.md)。

---

### ASR — 语音识别

**所属域**：`audio`
**节点 ID**：`asr`
**一句话描述**：将语音转录为文本。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `openai/whisper-large-v3` | 模型 ID |
| `language` | str \| None | `None` | 源语言 |
| `task` | str | `transcribe` | 任务（`transcribe` / `translate`） |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `audio` | AudioData / str / ndarray | ✅ | — | 输入音频 |
| `language` | str | ❌ | — | 源语言 |
| `task` | str | ❌ | — | 任务 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 识别文本 |
| `language` | str | 检测到的语言 |
| `segments` | list[dict] | 时间戳分段 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.audio import ASR

asr = ASR(model="openai/whisper-large-v3")
result = asr.run(MosaicData(audio="speech.wav"))
print(result.get("text"))
```

---

### MusicGenerator — 音乐生成

**所属域**：`audio`
**节点 ID**：`music-generator`
**一句话描述**：根据文本 prompt 生成背景音乐。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `facebook/musicgen-small` | 模型 ID |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 音乐描述 |
| `duration` | float | ❌ | 8.0 | 时长（秒） |
| `guidance_scale` | float | ❌ | 3.0 | CFG 引导强度 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | AudioData | 生成的音乐 |
| `prompt` | str | 输入 prompt |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.audio import MusicGenerator

music = MusicGenerator()
result = music.run(MosaicData(prompt="轻快的钢琴背景音乐", duration=10.0))
result.get("audio").save("bgm.wav")
```

---

### SoundEffectGenerator — 音效生成

**所属域**：`audio`
**节点 ID**：`sound-effect-generator`
**一句话描述**：根据文本 prompt 生成环境音效。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `cvssp/audioldm2` | 模型 ID |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ✅ | — | 音效描述 |
| `duration` | float | ❌ | 5.0 | 时长（秒） |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `num_inference_steps` | int | ❌ | 10 | 推理步数 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | AudioData | 生成的音效 |
| `prompt` | str | 输入 prompt |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.audio import SoundEffectGenerator

sfx = SoundEffectGenerator()
result = sfx.run(MosaicData(prompt="雨声打在窗户上", duration=5.0))
result.get("audio").save("rain.wav")
```

---

### VoiceClone — 语音克隆

**所属域**：`audio`
**节点 ID**：`voice-clone`
**一句话描述**：基于参考音频特征匹配 edge-tts 预设语音风格（非真实音色克隆）。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `edge-tts` | 模型 ID |
| `voice` | str \| None | `None` | 语音音色 |
| `language` | str | `zh` | 语言 |
| `emotion` | str | `neutral` | 情感 |
| `speed` | float | 1.0 | 语速 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `reference_audio` | AudioData / str / ndarray | ✅ | — | 参考音频 |
| `text` | str | ✅ | — | 要合成的文本 |
| `language` | str | ❌ | — | 语言 |
| `emotion` | str | ❌ | — | 情感 |
| `voice` | str | ❌ | — | 语音音色 |
| `speed` | float | ❌ | — | 语速 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | AudioData | 合成的音频 |
| `reference_audio` | AudioData | 参考音频 |
| `text` | str | 输入文本 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.audio import VoiceClone

clone = VoiceClone(model="edge-tts")
result = clone.run(MosaicData(
    reference_audio="reference.wav",
    text="这是克隆的声音",
))
result.get("audio").save("cloned.wav")
# VoiceClone 仅支持 edge-tts 风格匹配；真实音色克隆请使用 TTS(backend="sovits") + speaker=参考音频路径
```

---

## 字幕域 (subtitle, 3 节点)

### SubtitleGenerator — 字幕生成

**所属域**：`subtitle`
**节点 ID**：`subtitle-generator`
**一句话描述**：从音频或视频生成带时间戳的字幕。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `asr_model` | str | `openai/whisper-large-v3` | ASR 模型 ID |
| `output_format` | str | `srt` | 输出格式 |
| `language` | str \| None | `None` | 源语言 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `audio` | AudioData / str | ❌（无 `video` 时必填） | — | 输入音频 |
| `video` | str | ❌ | — | 输入视频路径（可选） |
| `language` | str | ❌ | — | 源语言 |
| `word_timestamps` | bool | ❌ | False | 是否生成词级时间戳 |
| `max_chars_per_line` | int | ❌ | 42 | 每行最大字符数 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtitle` | SubtitleData | 字幕数据 |
| `text` | str | 完整文本 |
| `language` | str | 语言 |
| `segments_count` | int | 分段数量 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.subtitle import SubtitleGenerator

gen = SubtitleGenerator()
result = gen.run(MosaicData(audio="speech.wav", language="zh"))
print(result.get("text"))
```

---

### SubtitleTranslator — 字幕翻译

**所属域**：`subtitle`
**节点 ID**：`subtitle-translator`
**一句话描述**：将字幕翻译为其他语言。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str \| None | `None` | 翻译模型 ID |
| `source_language` | str | `auto` | 源语言 |
| `target_language` | str | `en` | 目标语言 |
| `output_format` | str \| None | `None` | 输出格式 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `subtitle` | SubtitleData | ✅ | — | 输入字幕 |
| `source_language` | str | ❌ | — | 源语言 |
| `target_language` | str | ❌ | — | 目标语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtitle` | SubtitleData | 翻译后的字幕 |
| `source_language` | str | 源语言 |
| `target_language` | str | 目标语言 |
| `translated_count` | int | 翻译的条目数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.subtitle import SubtitleTranslator

translator = SubtitleTranslator(target_language="en")
result = translator.run(MosaicData(subtitle=subtitle_data))
print(result.get("translated_count"))
```

---

### SubtitleAligner — 时间轴对齐

**所属域**：`subtitle`
**节点 ID**：`subtitle-aligner`
**一句话描述**：将翻译后的字幕与原始音频时间轴对齐。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `method` | str | `whisper` | 对齐方法 |
| `language` | str \| None | `None` | 语言 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `subtitle` | SubtitleData | ✅ | — | 字幕 |
| `audio` | AudioData / str | ✅ | — | 音频 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtitle` | SubtitleData | 对齐后的字幕 |
| `alignment_method` | str | 对齐方法 |
| `time_shift` | float | 时间偏移（秒） |
| `alignment_score` | float | 对齐得分 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.subtitle import SubtitleAligner

aligner = SubtitleAligner()
result = aligner.run(MosaicData(subtitle=subtitle_data, audio="speech.wav"))
print(result.get("alignment_score"))
```

---

## 一致性域 (consistency, 3 节点)

### IdentityKeeper — 人脸一致性

**所属域**：`consistency`
**节点 ID**：`identity-keeper`
**一句话描述**：在生成图像中保持人脸身份一致。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `InstantX/InstantID` | 模型 ID |
| `method` | str | `instantid` | 方法 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |

> **method 与模型对应关系**：
> - `method="instantid"`（默认）→ `InstantX/InstantID`
> - `method="photomaker"` → `TencentARC/PhotoMaker`
> - `method="ip_adapter"` → `h94/IP-Adapter`

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `reference_image` | PIL.Image / str | ✅ | — | 参考人脸图 |
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `width` | int | ❌ | 1024 | 输出宽度 |
| `height` | int | ❌ | 1024 | 输出高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 5.0 | CFG 引导强度 |
| `identity_strength` | float | ❌ | 0.8 | 身份保持强度 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 输出图像 |
| `reference_image` | PIL.Image | 参考人脸图 |
| `identity_score` | float | 身份一致性得分 |
| `seed` | int | 实际使用的随机种子 |
| `method` | str | 使用的方法 |
| `model_name` | str | 模型名称 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.consistency import IdentityKeeper

keeper = IdentityKeeper()
result = keeper.run(MosaicData(
    reference_image="person.jpg",
    prompt="a portrait of this person in a suit",
    identity_strength=0.85,
))
result.get("image").save("portrait.png")
```

---

### StyleKeeper — 风格一致性

**所属域**：`consistency`
**节点 ID**：`style-keeper`
**一句话描述**：在生成图像中保持风格一致。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `h94/IP-Adapter` | 模型 ID |
| `method` | str | `ip-adapter` | 方法 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `reference_image` | PIL.Image / str | ✅ | — | 参考风格图 |
| `prompt` | str | ✅ | — | 文本描述 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `width` | int | ❌ | 1024 | 输出宽度 |
| `height` | int | ❌ | 1024 | 输出高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `style_strength` | float | ❌ | 0.7 | 风格强度 |
| `seed` | int | ❌ | — | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | PIL.Image | 输出图像 |
| `reference_image` | PIL.Image | 参考风格图 |
| `style` | str \| None | 风格描述 |
| `seed` | int | 实际使用的随机种子 |
| `method` | str | 使用的方法 |
| `model_name` | str | 模型名称 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.consistency import StyleKeeper

keeper = StyleKeeper()
result = keeper.run(MosaicData(
    reference_image="style_ref.jpg",
    prompt="a city street at night",
    style_strength=0.7,
))
result.get("image").save("styled.png")
```

---

### CrossFrameConsistency — 跨帧一致性

**所属域**：`consistency`
**节点 ID**：`cross-frame-consistency`
**一句话描述**：在多帧生成中保持角色/风格跨帧一致。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID |
| `method` | str | `consistory` | 方法 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompts` | list[str] | ✅ | — | 每帧的 prompt 列表 |
| `character_description` | str | ✅ | — | 角色描述 |
| `reference_image` | PIL.Image / str | ❌ | — | 参考图像 |
| `negative_prompt` | str | ❌ | — | 负向描述 |
| `width` | int | ❌ | 1024 | 输出宽度 |
| `height` | int | ❌ | 1024 | 输出高度 |
| `num_inference_steps` | int | ❌ | 30 | 推理步数 |
| `guidance_scale` | float | ❌ | 7.5 | CFG 引导强度 |
| `seed` | int | ❌ | — | 随机种子 |
| `consistency_strength` | float | ❌ | 0.85 | 一致性强度 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `images` | list[PIL.Image] | 生成的图像列表 |
| `reference_image` | Any | 参考图像 |
| `character_description` | str | 角色描述 |
| `consistency_scores` | list[float] | 每帧一致性得分 |
| `average_consistency` | float | 平均一致性得分 |
| `seed` | int | 实际使用的随机种子 |
| `model_name` | str | 模型名称 |
| `method` | str | 使用的方法 |
| `num_frames` | int | 帧数 |
| `width` | int | 输出宽度 |
| `height` | int | 输出高度 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.consistency import CrossFrameConsistency

cc = CrossFrameConsistency()
result = cc.run(MosaicData(
    prompts=["a boy reading a book", "a boy running", "a boy sleeping"],
    character_description="a 10-year-old boy with curly hair",
    consistency_strength=0.85,
))
print(result.get("num_frames"))
```

---

## 数字人域 (digital-human, 4 节点)

### AvatarDriver — 形象驱动

**所属域**：`digital-human`
**节点 ID**：`avatar-driver`
**一句话描述**：根据驱动信号（视频/音频/表情参数）驱动数字人形象。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `KwaiVGI/LivePortrait` | 模型 ID |
| `method` | str | `liveportrait` | 驱动方法（`liveportrait` / `sadtalker` / `musetalk`） |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `wav2vec2_model` | str | `facebook/wav2vec2-base-960h` | wav2vec2 音频编码器模型（HF 仓库 ID 或本地路径） |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `source_image` | PIL.Image / str | ✅ | — | 源人物图片 |
| `driving_video` | VideoData / str / list[PIL.Image] | ❌（三选一） | — | 驱动视频 |
| `driving_audio` | AudioData / str / ndarray | ❌（三选一） | — | 驱动音频 |
| `expression_params` | list[dict] / dict | ❌（三选一） | — | 表情参数序列 |
| `output_format` | str | ❌ | `video` | 输出格式（`video` / `frames`） |
| `fps` | int | ❌ | 25 | 帧率 |
| `resolution` | tuple | ❌ | (512, 512) | 输出分辨率 |
| `expression_scale` | float | ❌ | 1.0 | 表情缩放 |
| `motion_scale` | float | ❌ | 1.0 | 运动幅度缩放 |

> 驱动源 `driving_video` / `driving_audio` / `expression_params` 三选一，必须提供其中之一。

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 输出视频（`output_format="video"` 时） |
| `frames` | list[PIL.Image] | 输出帧列表（`output_format="frames"` 时） |
| `source_image` | PIL.Image | 源图片 |
| `driving_source_type` | str | 驱动源类型（`video` / `audio` / `expression_params`） |
| `duration` | float | 时长（秒） |
| `fps` | int | 帧率 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.digital_human import AvatarDriver

driver = AvatarDriver(method="liveportrait")
result = driver.run(MosaicData(
    source_image="person.jpg",
    driving_video="talk.mp4",
    fps=25,
))
result.get("video").save("avatar.mp4")
```

---

### LipSyncer — 口型同步

**所属域**：`digital-human`
**节点 ID**：`lip-syncer`
**一句话描述**：将音频与人物口型对齐，生成口型同步视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `KwaiVGI/MuseTalk` | 模型 ID |
| `method` | str | `musetalk` | 方法 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |
| `wav2vec2_model` | str | `facebook/wav2vec2-base-960h` | wav2vec2 音频编码器模型（HF 仓库 ID 或本地路径） |

> **method 与模型对应关系**：
> - `method="musetalk"`（默认）→ `KwaiVGI/MuseTalk`
> - `method="wav2lip"` → 需手动下载 wav2lip 权重（GitHub: MapleVison/wav2lip-release）
> - `method="sadtalker"` → `cvitkwai/SadTalker`

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `face_image` | PIL.Image / str / VideoData / list[PIL.Image] | ✅ | — | 人脸图像/视频/帧列表 |
| `audio` | AudioData / str / ndarray | ✅ | — | 目标音频 |
| `fps` | int | ❌ | 25 | 帧率 |
| `output_format` | str | ❌ | `video` | 输出格式（`video` / `frames`） |
| `padding` | list[int] | ❌ | [0, 20, 0, 20] | 人脸 padding（上/下/左/右） |
| `parsing_mode` | str | ❌ | `jaw` | 解析模式 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 输出视频（`output_format="video"` 时） |
| `frames` | list[PIL.Image] | 输出帧列表（`output_format="frames"` 时） |
| `audio` | AudioData | 输入音频 |
| `duration` | float | 时长（秒） |
| `fps` | int | 帧率 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.digital_human import LipSyncer

lip = LipSyncer()
result = lip.run(MosaicData(
    face_image="face.jpg",
    audio="speech.wav",
))
result.get("video").save("lipsync.mp4")
```

---

### MotionGenerator — 动作生成

**所属域**：`digital-human`
**节点 ID**：`motion-generator`
**一句话描述**：生成人物动作序列（文本驱动或音频驱动）。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str \| None | `None` | 模型 ID |
| `method` | str | `preset` | 方法（`preset` / `text2motion` / `audio2motion`） |
| `skeleton_type` | str | `coco` | 骨架类型 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | str | ❌（`text2motion` 必填） | — | 动作描述 |
| `audio` | AudioData | ❌（`audio2motion` 必填） | — | 驱动音频 |
| `preset_name` | str | ❌ | `wave` | 预设动作名称（`preset` 模式） |
| `duration` | float | ❌ | 3.0 | 时长（秒） |
| `fps` | int | ❌ | 30 | 帧率 |
| `smooth` | bool | ❌ | True | 是否平滑 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `motion` | MotionData | 动作数据 |
| `keypoints` | ndarray | 关键点数组 |
| `frame_count` | int | 帧数 |
| `duration` | float | 时长（秒） |
| `skeleton_type` | str | 骨架类型 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.digital_human import MotionGenerator

mg = MotionGenerator(method="preset")
result = mg.run(MosaicData(preset_name="wave", duration=3.0))
print(result.get("frame_count"))
```

---

### RealtimeRenderer — 实时渲染

**所属域**：`digital-human`
**节点 ID**：`realtime-renderer`
**一句话描述**：实时渲染数字人输出流。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `KwaiVGI/LivePortrait` | 模型 ID |
| `target_fps` | int | 25 | 目标帧率 |
| `resolution` | tuple | (512, 512) | 输出分辨率 |
| `enable_tts` | bool | False | 是否启用内置 TTS |
| `tts_model` | str \| None | `None` | TTS 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 推理精度 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `source_image` | PIL.Image / str | ✅ | — | 源人物图片 |
| `mode` | str | ✅ | — | 驱动模式（`audio` / `text` / `motion`） |
| `input_stream` | generator / list | ✅ | — | 输入流 |
| `output_mode` | str | ❌ | `frames` | 输出模式 |
| `output_callback` | Callable | ❌ | — | 输出回调 |
| `target_fps` | int | ❌ | — | 目标帧率 |
| `resolution` | tuple | ❌ | — | 输出分辨率 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `frames` | list[PIL.Image] | 渲染帧列表 |
| `render_stats` | dict | 渲染统计信息 |
| `duration` | float | 时长（秒） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.digital_human import RealtimeRenderer

renderer = RealtimeRenderer()
result = renderer.run(MosaicData(
    source_image="person.jpg",
    mode="text",
    input_stream=["你好", "很高兴认识你"],
))
print(len(result.get("frames")))
```

---

## 导出域 (export, 3 节点)

### VideoEncoder — 视频编码

**所属域**：`export`
**节点 ID**：`video-encoder`
**一句话描述**：将帧列表编码为视频文件。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `format` | str | `mp4` | 输出格式 |
| `codec` | str \| None | `None` | 编码器 |
| `quality` | int | 23 | 质量（CRF） |
| `preset` | str | `medium` | 编码预设 |
| `audio_codec` | str \| None | `aac` | 音频编码器 |
| `pixel_format` | str | `yuv420p` | 像素格式 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `frames` | list[PIL.Image] | ✅ | — | 帧列表 |
| `fps` | int | ✅ | — | 帧率 |
| `audio` | AudioData | ❌ | — | 音频 |
| `output_path` | str | ❌ | — | 输出路径 |
| `bitrate` | str | ❌ | — | 码率 |
| `subtitle` | SubtitleData | ❌ | — | 字幕 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `output_path` | str | 输出路径 |
| `format` | str | 输出格式 |
| `codec` | str | 编码器 |
| `duration` | float | 时长（秒） |
| `file_size` | int | 文件大小（字节） |
| `resolution` | tuple | 分辨率 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.export import VideoEncoder

encoder = VideoEncoder(format="mp4")
result = encoder.run(MosaicData(
    frames=frame_list,
    fps=25,
    output_path="output.mp4",
))
print(result.get("output_path"))
```

---

### Livestreamer — 直播推流

**所属域**：`export`
**节点 ID**：`livestreamer`
**一句话描述**：将视频流推送到流媒体服务器。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `protocol` | str | `rtmp` | 推流协议 |
| `codec` | str | `libx264` | 编码器 |
| `bitrate` | str | `4M` | 码率 |
| `fps` | int | 24 | 帧率 |
| `resolution` | tuple | (1920, 1080) | 分辨率 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `frames` | list[PIL.Image] | ✅ | — | 帧列表 |
| `stream_url` | str | ✅ | — | 推流地址 |
| `fps` | int | ❌ | — | 帧率 |
| `audio` | AudioData | ❌ | — | 音频 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | str | 推流状态 |
| `stream_url` | str | 推流地址 |
| `frames_sent` | int | 已发送帧数 |
| `duration` | float | 时长（秒） |
| `error` | str | 失败时的错误信息 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.export import Livestreamer

streamer = Livestreamer(protocol="rtmp")
result = streamer.run(MosaicData(
    frames=frame_list,
    stream_url="rtmp://localhost/live/stream",
))
print(result.get("status"))
```

---

### MultiFormatExporter — 多格式导出

**所属域**：`export`
**节点 ID**：`multi-format-exporter`
**一句话描述**：将内容导出为多种格式。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bus` | EventBus \| None | `None` | 事件总线实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `content_type` | str | ✅ | — | 内容类型（`video` / `image` / `audio` / `subtitle`） |
| `data` | Any | ✅ | — | 输入数据 |
| `formats` | list[str] | ✅ | — | 输出格式列表 |
| `output_dir` | str | ❌ | — | 输出目录 |
| `quality` | int | ❌ | 23 | 质量 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `outputs` | dict[str, str] | 格式到文件路径的映射 |
| `total_files` | int | 总文件数 |
| `total_size` | int | 总大小（字节） |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.export import MultiFormatExporter

exporter = MultiFormatExporter()
result = exporter.run(MosaicData(
    content_type="video",
    data=video_data,
    formats=["mp4", "gif", "webm"],
    output_dir="./outputs",
))
print(result.get("outputs"))
```

---

## RAG 域 (rag, 4 节点)

### DocumentParser — 文档解析

**所属域**：`rag`
**节点 ID**：`document-parser`
**一句话描述**：解析文档为分块文本，支持多种格式。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `chunk_size` | int | 512 | 分块大小 |
| `chunk_overlap` | int | 50 | 分块重叠 |
| `preserve_structure` | bool | True | 是否保留文档结构 |
| `supported_formats` | list[str] \| None | `None` | 支持的格式列表 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `file_path` | str | ❌（无 `file_content` 时必填） | — | 文件路径 |
| `file_content` | str | ❌（无 `file_path` 时必填） | — | 文件内容 |
| `file_type` | str | ❌ | `txt` | 文件类型 |
| `filename` | str | ❌ | — | 文件名 |
| `metadata_filter` | dict | ❌ | — | 元数据过滤 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `document` | DocumentData | 文档数据 |
| `total_chunks` | int | 总分块数 |
| `total_chars` | int | 总字符数 |
| `file_type` | str | 文件类型 |
| `metadata` | dict | 元数据 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.rag import DocumentParser

parser = DocumentParser(chunk_size=512)
result = parser.run(MosaicData(file_path="doc.pdf", file_type="pdf"))
print(result.get("total_chunks"))
```

---

### VectorIndexer — 向量化索引

**所属域**：`rag`
**节点 ID**：`vector-indexer`
**一句话描述**：将文档块转为向量并建立索引。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | 嵌入模型 ID |
| `index_type` | str | `faiss` | 索引类型 |
| `index_path` | str \| None | `None` | 索引保存路径 |
| `batch_size` | int | 32 | 批大小 |
| `device` | str | `cuda` | 推理设备 |
| `metric` | str | `ip` | 相似度度量 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `document` | DocumentData | ❌（无 `chunks` 时必填） | — | 输入文档 |
| `chunks` | list[str] | ❌（备选） | — | 文本块列表 |
| `collection_name` | str | ❌ | `default` | 集合名称 |
| `metadata` | list[dict] | ❌ | — | 元数据列表 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `collection_name` | str | 集合名称 |
| `indexed_count` | int | 已索引数量 |
| `embedding_dim` | int | 嵌入维度 |
| `index_type` | str | 索引类型 |
| `index_path` | str \| None | 索引路径 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.rag import VectorIndexer

indexer = VectorIndexer()
result = indexer.run(MosaicData(document=doc_data, collection_name="my_docs"))
print(result.get("indexed_count"))
```

---

### Retriever — 检索

**所属域**：`rag`
**节点 ID**：`retriever`
**一句话描述**：根据查询从向量索引中检索相关文档。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `index_type` | str | `faiss` | 索引类型 |
| `index_path` | str \| None | `None` | 索引路径 |
| `embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | 嵌入模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `metric` | str | `ip` | 相似度度量 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例 |
| `indexer` | Any \| None | `None` | 已存在的索引器实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | str | ✅ | — | 查询 |
| `collection_name` | str | ❌ | `default` | 集合名称 |
| `top_k` | int | ❌ | 5 | 返回 Top-K |
| `score_threshold` | float | ❌ | 0.0 | 分数阈值 |
| `filter_metadata` | dict | ❌ | — | 元数据过滤 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | str | 查询 |
| `results` | list[dict] | 检索结果列表 |
| `result_count` | int | 结果数量 |
| `top_score` | float | 最高得分 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.rag import Retriever

retriever = Retriever(index_path="./index")
result = retriever.run(MosaicData(query="什么是 Mosaic？", top_k=5))
print(result.get("results"))
```

---

### CitationGenerator — 引用生成

**所属域**：`rag`
**节点 ID**：`citation-generator`
**一句话描述**：根据检索结果生成带引用标注的答案。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `llm_model` | str | `Qwen/Qwen2.5-7B-Instruct` | LLM 模型 ID |
| `citation_style` | str | `inline` | 引用样式 |
| `include_sources` | bool | True | 是否包含来源 |
| `max_tokens` | int | 1024 | 最大生成 token 数 |
| `temperature` | float | 0.3 | 采样温度 |
| `device_map` | str | `auto` | 设备映射 |
| `torch_dtype` | str | `fp16` | 权重精度 |
| `trust_remote_code` | bool | True | 是否信任远程代码 |
| `bus` | EventBus \| None | `None` | 事件总线实例 |
| `scheduler` | Scheduler \| None | `None` | 显存调度器实例 |

#### run 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | str | ✅ | — | 查询 |
| `results` | list[dict] | ✅ | — | 检索结果列表 |
| `citation_style` | str | ❌ | — | 引用样式 |
| `language` | str | ❌ | `zh` | 语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `answer` | str | 带引用的答案 |
| `citations` | list[dict] | 引用列表 |
| `query` | str | 查询 |
| `sources_used` | int | 使用的来源数量 |
| `input_tokens` | int | 输入 token 数 |
| `output_tokens` | int | 输出 token 数 |

#### 示例

```python
from mosaic.core.types import MosaicData
from mosaic.nodes.rag import CitationGenerator

gen = CitationGenerator()
result = gen.run(MosaicData(
    query="什么是 Mosaic？",
    results=retrieval_results,
    citation_style="inline",
))
print(result.get("answer"))
```

---

## TTS 后端 (4 后端)

Mosaic 音频域 TTS 是路由器节点，4 个后端均通过统一接口调用。详细文档见 [TTS 完整指南](tts-guide.md) 与 [TTS 后端指南](tts-backend-guide.md)。

### ChatTTSBackend — ChatTTS 后端

**特点**：AR 流式，延迟低，韵律控制丰富。

**采样率**：24000Hz

**许可证**：CC-BY-NC-4.0

**推理管线**：

```
text → ChatTokenizer → LlamaARModel → DVAE+Vocos → StreamAdapter → waveform (24kHz)
```

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | str | — | ChatTTS 模型目录路径 |
| `vocos_path` | str \| None | `None` | Vocos 权重路径 |
| `num_vq` | int | 4 | VQ 码本组数 |
| `language` | str | `zh` | 默认语言代码 |
| `use_flash_attention` | bool | `True` | 声学模型是否使用 Flash Attention |
| `streaming_enabled` | bool | `True` | 是否启用流式合成 |
| `scheduler` | Any \| None | `None` | 显存调度器实例 |
| `repo_id` | str \| None | `None` | HuggingFace 仓库 ID |

#### 用法

```python
from mosaic.nodes.audio.tts_backends.implementations import ChatTTSBackend

# 方式 1：传 HF 仓库 ID（自动下载到同名目录）
backend = ChatTTSBackend(model_path="2Noise/ChatTTS")

# 方式 2：传本地路径（直接使用已下载的模型）
# backend = ChatTTSBackend(model_path="/data/models/chattts")

backend.load()
audio = backend.synthesize(text="你好", language="zh")

# 流式：通过 tts.run_stream(MosaicData(...)) 调用（同步迭代器）
for chunk in tts.run_stream(MosaicData(text="...", language="zh")):
    play(chunk)
```

---

### FishSpeechBackend — Fish Speech 后端

**特点**：多语言支持最佳，跨语种克隆。

**采样率**：22050Hz

**许可证**：Apache-2.0

**推理管线**：

```
text → FishTokenizer → FishLlamaARModel → VQDecoder+HiFiGAN → StreamAdapter → waveform (22.05kHz)
```

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | str | — | Fish Speech 模型目录路径 |
| `hifi_gan_path` | str \| None | `None` | HiFiGAN 权重路径 |
| `audio_encoder_path` | str \| None | `None` | 音频编码器路径 |
| `codec_type` | str | `dac` | 编解码器类型 |
| `language` | str | `zh` | 默认语言代码 |
| `use_flash_attention` | bool | `True` | 声学模型是否使用 Flash Attention |
| `streaming_enabled` | bool | `True` | 是否启用流式合成 |
| `scheduler` | Any \| None | `None` | 显存调度器实例 |
| `repo_id` | str \| None | `None` | HuggingFace 仓库 ID |

#### 用法

```python
from mosaic.nodes.audio.tts_backends.implementations import FishSpeechBackend

# 方式 1：传 HF 仓库 ID（自动下载到同名目录）
backend = FishSpeechBackend(model_path="fishaudio/fish-speech-1.5")

# 方式 2：传本地路径（直接使用已下载的模型）
# backend = FishSpeechBackend(model_path="/data/models/fish-speech")

backend.load()
audio = backend.synthesize(text="你好", language="zh")
```

---

### GPTSoVITSBackend — GPT-SoVITS 后端

**特点**：极少样本克隆（5-10 秒），开源可商用。

**采样率**：32000Hz

**许可证**：MIT

**推理管线**：

```
text → SoVITSTokenizer → GPT2ARModel → SoVITSDecoder → StreamAdapter → waveform (32kHz)
```

> 注意：GPT-SoVITS 没有独立的 FlowDecoder 层，`SoVITSDecoder` 内含 SemanticEncoder + Flow + HiFiGAN。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | str | — | 模型目录路径 |
| `gpt_path` | str \| None | `None` | GPT 权重路径 |
| `sovits_path` | str \| None | `None` | SoVITS 权重路径 |
| `ssl_model` | str | `chinese-hubert-base` | SSL 模型 ID |
| `speaker_encoder_model` | str | `default` | 说话人编码器模型 |
| `language` | str | `zh` | 默认语言代码 |
| `streaming_enabled` | bool | `True` | 是否启用流式合成 |
| `scheduler` | Any \| None | `None` | 显存调度器实例 |
| `repo_id` | str \| None | `None` | HuggingFace 仓库 ID |

#### 用法

```python
from mosaic.nodes.audio.tts_backends.implementations import GPTSoVITSBackend

# 方式 1：传 HF 仓库 ID（自动下载到同名目录）
backend = GPTSoVITSBackend(model_path="lj1995/GPT-SoVITS")

# 方式 2：传本地路径（直接使用已下载的模型）
# backend = GPTSoVITSBackend(model_path="/data/models/gpt-sovits")

backend.load()
audio = backend.synthesize(text="你好", language="zh")
```

---

### CosyVoiceBackend — CosyVoice 后端

**特点**：质量最高，非自回归可并行，SFT 指令控制。

**采样率**：24000Hz

**许可证**：Apache-2.0

**推理管线**：

```
text → CosyVoiceTokenizer → FlowMatchingModel → HiFiGanVocoder → StreamAdapter → waveform (24kHz)
```

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | str | — | 模型目录路径 |
| `llm_model` | str | `Qwen/Qwen2.5-1.5B-Instruct` | LLM 模型 ID |
| `speech_tokenizer_model` | str \| None | `None` | 语音 tokenizer 模型 |
| `speaker_encoder_model` | str | `campp` | 说话人编码器模型 |
| `hifi_gan_path` | str \| None | `None` | HiFiGAN 权重路径 |
| `num_ode_steps` | int | 10 | ODE 求解步数 |
| `ode_solver` | str | `euler` | ODE 求解器 |
| `language` | str | `zh` | 默认语言代码 |
| `streaming_enabled` | bool | `True` | 是否启用流式合成 |
| `chunk_size_frames` | int | 150 | 流式 chunk 帧数 |
| `chunk_overlap_frames` | int | 16 | 流式 chunk 重叠帧数 |
| `scheduler` | Any \| None | `None` | 显存调度器实例 |
| `repo_id` | str \| None | `None` | HuggingFace 仓库 ID |

#### ODE 步数对比

| 步数 | 质量 | 速度 |
|---|---|---|
| 5 | 中 | 极快 |
| 10 | 良 | 快 |
| 20 | 优 | 中 |
| 50 | 最佳 | 慢 |

#### 用法

```python
from mosaic.nodes.audio.tts_backends.implementations import CosyVoiceBackend

# 方式 1：传 HF 仓库 ID（自动下载到同名目录）
backend = CosyVoiceBackend(model_path="FunAudioLLM/CosyVoice2-0.5B", num_ode_steps=10)

# 方式 2：传本地路径（直接使用已下载的模型）
# backend = CosyVoiceBackend(model_path="/data/models/cosyvoice2", num_ode_steps=10)

backend.load()
audio = backend.synthesize(text="你好", language="zh")
```

---

## 下一步

- [管道使用指南](pipeline-guide.md) — 节点组合
- [插件开发指南](plugin-development.md) — 自定义节点
- [TTS 完整指南](tts-guide.md) — TTS 节点与后端详解
- [TTS 后端指南](tts-backend-guide.md) — 4 个 TTS 后端深入文档
- [示例代码](../examples/) — 实际应用
