# Mosaic 节点参考手册

> 全部 42 节点 + 4 个 TTS 后端的完整 API 参考。

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

文本域提供 6 个核心文本处理节点，基于 `transformers` 的 `pipeline` 抽象。

### TextGenerator — 文本生成

**所属域**：`text`
**节点 ID**：`text-generator`
**一句话描述**：根据 prompt 生成一段文本。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace 模型 ID |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `bfloat16` | 推理精度 |
| `max_new_tokens` | int | 512 | 最大生成长度 |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 输入 prompt |
| `temperature` | float | ❌ | 采样温度（0-2） |
| `top_p` | float | ❌ | nucleus 采样阈值 |
| `seed` | int | ❌ | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 生成的文本 |

#### 示例

```python
from mosaic.nodes.text import TextGenerator

gen = TextGenerator(model="Qwen/Qwen2.5-7B-Instruct")
result = gen.run(prompt="写一首关于春天的诗", temperature=0.8, seed=42)
print(result.get("text"))
```

#### 推荐模型

| 模型 | 显存 | 适用 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 1GB | 快速测试 |
| Qwen2.5-7B-Instruct | 16GB | 通用 |
| Llama-3.1-70B-Instruct | 80GB×2 | 高质量 |

#### 许可证

因模型而异，参考 HF 模型页面。

---

### Chat — 对话

**所属域**：`text`
**节点 ID**：`chat`
**一句话描述**：多轮对话节点，维护对话历史。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Qwen/Qwen2.5-7B-Instruct` | 模型 ID |
| `system_prompt` | str | `""` | 系统提示 |
| `max_history` | int | 10 | 保留历史轮数 |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `message` | str | ✅ | 用户消息 |
| `clear_history` | bool | ❌ | 是否清空历史 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `reply` | str | 模型回复 |
| `history` | list | 更新后的对话历史 |

#### 示例

```python
from mosaic.nodes.text import Chat

chat = Chat(system_prompt="你是一个友好的助手")
result = chat.run(message="你好")
print(result.get("reply"))  # 你好！有什么可以帮你的吗？

# 多轮
result = chat.run(message="推荐一本 Python 入门书")
print(result.get("reply"))
```

---

### TextRewriter — 文本改写

**所属域**：`text`
**节点 ID**：`text-rewriter`
**一句话描述**：按风格或要求改写文本。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 原文 |
| `style` | str | ❌ | 目标风格 (`formal` / `casual` / `academic`) |
| `requirement` | str | ❌ | 改写要求 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 改写后的文本 |

#### 示例

```python
from mosaic.nodes.text import TextRewriter

rewriter = TextRewriter()
result = rewriter.run(
    text="这个产品很好，我很喜欢",
    style="formal",
    requirement="增加说服力",
)
```

---

### Translator — 翻译

**所属域**：`text`
**节点 ID**：`translator`
**一句话描述**：专用翻译节点。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 原文 |
| `source_lang` | str | ❌ | 源语言（自动检测如果省略） |
| `target_lang` | str | ✅ | 目标语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 翻译结果 |
| `source_lang` | str | 检测到的源语言 |

#### 示例

```python
from mosaic.nodes.text import Translator

translator = Translator()
result = translator.run(text="你好世界", target_lang="en")
print(result.get("text"))  # Hello world
```

#### 支持语言

中、英、日、韩、法、德、西、俄、阿、葡等 100+ 种。

---

### TextSummarizer — 摘要

**所属域**：`text`
**节点 ID**：`text-summarizer`
**一句话描述**：生成文本摘要。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 原文 |
| `mode` | str | ❌ | `concise` / `detailed` / `bullet_points` |
| `max_length` | int | ❌ | 摘要最大长度（token） |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 摘要 |

#### 示例

```python
from mosaic.nodes.text import TextSummarizer

summarizer = TextSummarizer()
result = summarizer.run(
    text=long_article,
    mode="bullet_points",
    max_length=200,
)
```

---

### TextClassifier — 分类

**所属域**：`text`
**节点 ID**：`text-classifier`
**一句话描述**：文本分类（情感、主题、zero-shot）。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 输入文本 |
| `labels` | list[str] | ❌ | 自定义标签（zero-shot 模式） |
| `mode` | str | ❌ | `sentiment` / `topic` / `zero_shot` |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `label` | str | 预测标签 |
| `scores` | dict | 各标签的概率 |
| `top_k` | list | Top-K 预测 |

#### 示例

```python
from mosaic.nodes.text import TextClassifier

classifier = TextClassifier()

# 情感分析
result = classifier.run(text="今天天气真好", mode="sentiment")
print(result.get("label"))  # positive

# 零样本分类
result = classifier.run(
    text="这款手机拍照效果不错",
    labels=["正面", "负面", "中性"],
    mode="zero_shot",
)
print(result.get("label"))  # 正面
```

---

## 图像域 (image, 6 节点)

基于 `diffusers` 的图像生成/编辑能力。

### TextToImage — 文生图

**所属域**：`image`
**节点 ID**：`text-to-image`
**一句话描述**：根据文本生成图像。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID |
| `num_inference_steps` | int | 30 | 推理步数 |
| `guidance_scale` | float | 7.5 | CFG 引导强度 |
| `width` | int | 1024 | 输出宽度 |
| `height` | int | 1024 | 输出高度 |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 文本描述 |
| `negative_prompt` | str | ❌ | 负向描述 |
| `seed` | int | ❌ | 随机种子 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | ImageData | 生成的图像 |

#### 示例

```python
from mosaic.nodes.image import TextToImage

t2i = TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0")
result = t2i.run(
    prompt="A cute cat sitting on a windowsill, sunlight",
    negative_prompt="blurry, low quality",
    seed=42,
)
result.get("image").save("cat.png")
```

#### 推荐模型

| 模型 | 显存 | 许可证 |
|---|---|---|
| stabilityai/stable-diffusion-xl-base-1.0 | 8GB | OpenRAIL++ |
| black-forest-labs/FLUX.1-schnell | 24GB | Apache-2.0 |
| stabilityai/sdxl-turbo | 6GB | OpenRAIL++ |

---

### ImageToImage — 图生图

**所属域**：`image`
**节点 ID**：`image-to-image`
**一句话描述**：根据输入图和 prompt 生成新图。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `prompt` | str | ✅ | 文本描述 |
| `strength` | float | 0.75 | 变化强度（0-1） |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | ImageData | 输出图像 |

#### 示例

```python
from mosaic.nodes.image import ImageToImage

i2i = ImageToImage(model="stabilityai/stable-diffusion-xl-refiner-1.0")
result = i2i.run(
    image=input_image,
    prompt="the same image in watercolor style",
    strength=0.6,
)
```

---

### Inpainting — 局部重绘

**所属域**：`image`
**节点 ID**：`inpainting`
**一句话描述**：在 mask 区域内重绘。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `mask` | ImageData | ✅ | 蒙版（白色=重绘） |
| `prompt` | str | ✅ | 重绘描述 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | ImageData | 输出图像 |

#### 示例

```python
from mosaic.nodes.image import Inpainting
from PIL import Image

inpaint = Inpainting(model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
result = inpaint.run(
    image=Image.open("room.jpg"),
    mask=Image.open("mask.png"),  # 白色区域被替换
    prompt="a beautiful garden with flowers",
)
result.get("image").save("new_room.jpg")
```

---

### Upscaler — 超分

**所属域**：`image`
**节点 ID**：`upscaler`
**一句话描述**：图像超分辨率放大。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `scale` | int | 4 | 放大倍数（2/4） |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | ImageData | 放大后的图像 |

#### 推荐模型

- `stabilityai/stable-diffusion-x4-upscaler`
- ` realesrgan-x4plus`（备选）

---

### BackgroundRemover — 去背景

**所属域**：`image`
**节点 ID**：`background-remover`
**一句话描述**：移除图像背景。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `model` | str | `briaai/RMBG-2.0` | 背景移除模型 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | ImageData | 透明背景图像（RGBA） |

---

### Stylizer — 风格化

**所属域**：`image`
**节点 ID**：`stylizer`
**一句话描述**：艺术风格化（IP-Adapter）。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `style` | str | ✅ | 风格描述或参考图 |
| `strength` | float | 0.8 | 风格强度 |

---

## 视频域 (video, 8 节点)

### TextToVideo — 文生视频（CogVideoX）

**所属域**：`video`
**节点 ID**：`text-to-video`
**一句话描述**：基于 CogVideoX 的文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `THUDM/CogVideoX-5b` | 模型 ID |
| `enable_vae_tiling` | bool | True | 启用 VAE 瓦片化 |
| `enable_sequential_cpu_offload` | bool | False | 顺序 CPU offload |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 文本描述 |
| `num_frames` | int | 49 | 帧数（49 或 85） |
| `fps` | int | 8 | 帧率 |
| `width` | int | 720 | 视频宽度 |
| `height` | int | 480 | 视频高度 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `video` | VideoData | 生成的视频 |
| `num_frames` | int | 实际帧数 |
| `duration` | float | 时长（秒） |

#### 显存需求

| 模型 | 显存 |
|---|---|
| CogVideoX-5b | 18GB |
| CogVideoX-2b | 9GB |

---

### WanVideo — 文生视频（Wan2.1/Wan2.2）

**所属域**：`video`
**节点 ID**：`wan-video`
**一句话描述**：基于阿里通义万相 Wan 系列的文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 模型 ID |
| `dtype` | str | `float16` | 推理精度 |
| `enable_cpu_offload` | bool | True | 启用 CPU offload |
| `enable_vae_tiling` | bool | True | 启用 VAE 瓦片化 |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 文本描述 |
| `num_frames` | int | 81 | 帧数（4k+1） |
| `fps` | int | 16 | 帧率 |
| `width` | int | 1280 | 视频宽度 |
| `height` | int | 720 | 视频高度 |
| `num_inference_steps` | int | 30 | 推理步数 |
| `guidance_scale` | float | 5.0 | CFG 引导强度 |

#### 显存需求

| 模型 | 显存 |
|---|---|
| Wan2.1-14B | 30GB |
| Wan2.1-1.3B | 8GB |
| Wan2.2-A14B | 30GB |

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
| `model` | str | `tencent/HunyuanVideo` | 模型 ID |
| `dtype` | str | `bfloat16` | 推理精度 |
| `enable_chunking` | bool | True | 启用 VAE chunking（专属优化） |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 文本描述 |
| `num_frames` | int | 129 | 帧数 |
| `fps` | int | 24 | 帧率 |

#### 显存需求

~60GB（启用 CPU offload 后 40GB 可运行）

#### 许可证

Tencent Hunyuan Video License

---

### LTXVideo — 文生视频（Lightricks）

**所属域**：`video`
**节点 ID**：`ltx-video`
**一句话描述**：基于 Lightricks LTX-Video 的轻量快速文生视频。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | str | `Lightricks/LTX-Video` | 模型 ID |
| `dtype` | str | `bfloat16` | 推理精度 |

#### 显存需求

~12GB

#### 许可证

OpenRAIL-M

---

### ImageToVideo — 图生视频

**所属域**：`video`
**节点 ID**：`image-to-video`
**一句话描述**：基于 SVD 的图像到视频生成。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | ImageData | ✅ | 输入图像 |
| `num_frames` | int | 14 | 帧数（14-25） |
| `fps` | int | 7 | 帧率 |
| `motion_bucket_id` | int | 127 | 运动强度（0-255） |

---

### VideoContinuation — 视频续写

**所属域**：`video`
**节点 ID**：`video-continuation`
**一句话描述**：在视频末尾续写新帧。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `overlap_frames` | int | 5 | 与原视频重叠的帧数 |

---

### FrameInterpolator — 插帧

**所属域**：`video`
**节点 ID**：`frame-interpolator`
**一句话描述**：在两帧之间插入中间帧。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `target_fps` | int | ✅ | 目标帧率 |
| `method` | str | `rife` | 插值方法（`rife` / `film` / `linear`） |

---

### FrameExtractor — 拆帧

**所属域**：`video`
**节点 ID**：`frame-extractor`
**一句话描述**：从视频提取所有帧为图像列表。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `output_dir` | str | ❌ | 保存目录 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `frames` | list[ImageData] | 帧列表 |
| `frame_count` | int | 帧数 |

---

## 音频域 (audio, 5 节点)

### TTS — 文本转语音

**所属域**：`audio`
**节点 ID**：`tts`
**一句话描述**：将文本转为语音，支持 4 个后端。

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | str | `edge_tts` | 后端：`chattts` / `fish` / `sovits` / `cosyvoice` / `edge_tts` |
| `voice` | str | `zh-CN-XiaoxiaoNeural` | 语音（edge_tts） |
| `language` | str | `zh` | 语言 |
| `streaming` | bool | False | 是否流式 |
| `chunk_size` | int | 30 | 流式 chunk 大小（CosyVoice） |

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 输入文本 |
| `language` | str | ❌ | 语言 |
| `ref_audio` | str | ❌ | 参考音频路径（克隆用） |
| `seed` | int | ❌ | 随机种子（ChatTTS） |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio` | AudioData | 合成的音频 |

#### 示例

```python
from mosaic.nodes.audio import TTS

tts = TTS(backend="chattts", language="zh")
result = tts.run(text="你好世界")
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

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | AudioData | ✅ | 输入音频 |
| `language` | str | ❌ | 源语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 识别文本 |
| `segments` | list | 时间戳分段 |
| `language` | str | 检测到的语言 |

---

### MusicGenerator — 音乐生成

**所属域**：`audio`
**节点 ID**：`music-generator`
**一句话描述**：根据文本 prompt 生成背景音乐。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 音乐描述 |
| `duration` | int | 30 | 时长（秒） |

---

### SoundEffect — 音效生成

**所属域**：`audio`
**节点 ID**：`sound-effect`
**一句话描述**：生成环境音效。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 音效描述 |
| `duration` | float | 5.0 | 时长（秒） |

---

### VoiceClone — 语音克隆

**所属域**：`audio`
**节点 ID**：`voice-clone`
**一句话描述**：基于参考音频克隆说话人声音。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | str | ✅ | 要合成的文本 |
| `ref_audio` | str | ✅ | 参考音频路径 |
| `ref_text` | str | ❌ | 参考音频的文字 |

#### 示例

```python
from mosaic.nodes.audio import VoiceClone

clone = VoiceClone(backend="sovits")
result = clone.run(
    text="这是克隆的声音",
    ref_audio="reference.wav",
    ref_text="参考音频的文字内容",
)
```

---

## 字幕域 (subtitle, 3 节点)

### SubtitleGenerator — 字幕生成

**所属域**：`subtitle`
**节点 ID**：`subtitle-generator`
**一句话描述**：从音频生成带时间戳的字幕。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | AudioData | ✅ | 输入音频 |
| `language` | str | ❌ | 源语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtitle` | SubtitleData | 字幕数据 |
| `segments` | list | 分段列表 |

---

### SubtitleTranslator — 字幕翻译

**所属域**：`subtitle`
**节点 ID**：`subtitle-translator`
**一句话描述**：将字幕翻译为其他语言。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `subtitle` | SubtitleData | ✅ | 输入字幕 |
| `target_lang` | str | ✅ | 目标语言 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `subtitle` | SubtitleData | 翻译后的字幕 |

---

### SubtitleAligner — 时间轴对齐

**所属域**：`subtitle`
**节点 ID**：`subtitle-aligner`
**一句话描述**：将翻译后的字幕与原始音频时间轴对齐。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `subtitle` | SubtitleData | ✅ | 字幕 |
| `audio` | AudioData | ✅ | 音频 |

---

## 一致性域 (consistency, 3 节点)

### IdentityKeeper — 人脸一致性

**所属域**：`consistency`
**节点 ID**：`identity-keeper`
**一句话描述**：在多张生成图像中保持人脸身份一致。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `images` | list[ImageData] | ✅ | 输入图像列表 |
| `reference_image` | ImageData | ✅ | 参考人脸图 |

---

### StyleKeeper — 风格一致性

**所属域**：`consistency`
**节点 ID**：`style-keeper`
**一句话描述**：保持多张图的风格一致。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `images` | list[ImageData] | ✅ | 输入图像列表 |
| `style_strength` | float | 0.7 | 风格强度 |

---

### CrossFrameConsistency — 跨帧一致性

**所属域**：`consistency`
**节点 ID**：`cross-frame-consistency`
**一句话描述**：视频跨帧一致性（颜色、风格）。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |

---

## 数字人域 (digital-human, 4 节点)

### AvatarDriver — 形象驱动

**所属域**：`digital-human`
**节点 ID**：`avatar-driver`
**一句话描述**：根据姿态/表情驱动数字人形象。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `avatar` | ImageData | ✅ | 形象图 |
| `motion` | Any | ✅ | 驱动数据（姿态/表情） |

---

### LipSyncer — 口型同步

**所属域**：`digital-human`
**节点 ID**：`lip-syncer`
**一句话描述**：将音频与视频口型对齐。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `audio` | AudioData | ✅ | 目标音频 |

---

### MotionGenerator — 动作生成

**所属域**：`digital-human`
**节点 ID**：`motion-generator`
**一句话描述**：生成人物动作序列。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 动作描述 |
| `duration` | float | 5.0 | 时长（秒） |

---

### RealtimeRenderer — 实时渲染

**所属域**：`digital-human`
**节点 ID**：`realtime-renderer`
**一句话描述**：实时渲染数字人输出流。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `stream` | Any | ✅ | 输入流（音频/姿态） |

---

## 导出域 (export, 3 节点)

### VideoEncoder — 视频编码

**所属域**：`export`
**节点 ID**：`video-encoder`
**一句话描述**：将帧列表编码为视频文件。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `output_path` | str | ✅ | 输出路径 |
| `codec` | str | `libx264` | 编码器 |
| `bitrate` | str | `5M` | 码率 |

---

### Livestreamer — 直播推流

**所属域**：`export`
**节点 ID**：`livestreamer`
**一句话描述**：将视频流推送到 RTMP 服务器。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 视频数据 |
| `rtmp_url` | str | ✅ | RTMP 服务器地址 |
| `loop` | bool | True | 是否循环推流 |

---

### MultiFormatExporter — 多格式导出

**所属域**：`export`
**节点 ID**：`multi-format-exporter`
**一句话描述**：导出为多种格式（MP4 / GIF / WebM / MOV）。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | VideoData | ✅ | 输入视频 |
| `formats` | list[str] | `[mp4]` | 输出格式列表 |
| `output_dir` | str | `./outputs` | 输出目录 |

---

## RAG 域 (rag, 4 节点)

### DocumentParser — 文档解析

**所属域**：`rag`
**节点 ID**：`document-parser`
**一句话描述**：解析 PDF / DOCX / HTML / Markdown 等文档。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file_path` | str | ✅ | 文件路径 |
| `chunk_size` | int | 500 | 分块大小 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `documents` | list[DocumentData] | 文档块列表 |

---

### VectorIndexer — 向量化索引

**所属域**：`rag`
**节点 ID**：`vector-indexer`
**一句话描述**：将文档块转为向量并建立索引（FAISS / ChromaDB）。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `documents` | list[DocumentData] | ✅ | 输入文档 |
| `embedding_model` | str | `BAAI/bge-m3` | 嵌入模型 |
| `index_path` | str | ✅ | 索引保存路径 |

---

### Retriever — 检索

**所属域**：`rag`
**节点 ID**：`retriever`
**一句话描述**：根据查询从向量索引中检索相关文档。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | str | ✅ | 查询 |
| `index_path` | str | ✅ | 索引路径 |
| `top_k` | int | 5 | 返回 Top-K |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `documents` | list[DocumentData] | 检索结果 |

---

### CitationGenerator — 引用生成

**所属域**：`rag`
**节点 ID**：`citation-generator`
**一句话描述**：为生成的答案添加引用标注。

#### run 输入

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `answer` | str | ✅ | 生成的答案 |
| `documents` | list[DocumentData] | ✅ | 来源文档 |

#### run 输出

| 字段 | 类型 | 说明 |
|---|---|---|
| `answer` | str | 带引用的答案 |
| `citations` | list | 引用列表 |

---

## TTS 后端 (4 后端)

Mosaic 音频域 TTS 是路由器节点，4 个后端均通过统一接口调用。详细文档见 [TTS 完整指南](tts-guide.md)。

### ChatTTSBackend — ChatTTS 后端

**特点**：AR 流式，延迟最低（~50ms），韵律控制丰富。

**推理管线**：

```
text → ChatTokenizer → LlamaARModel → DVAE → Vocos → waveform (24kHz)
```

**构造函数参数**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_dir` | str | (HF Hub) | 模型目录 |
| `device` | str | `cuda` | 推理设备 |
| `dtype` | str | `float16` | 精度 |
| `temperature` | float | 0.3 | 采样温度 |

**用法**

```python
from mosaic.nodes.audio.tts_backends.implementations import ChatTTSBackend

backend = ChatTTSBackend()
backend.load()
audio = backend.synthesize(text="你好", language="zh")

# 流式
async for chunk in backend.synthesize_stream(text="...", language="zh"):
    play(chunk)
```

**支持的韵律标记**：`[oral_xxx]` `[laugh]` `[break]` `[speed_x]`

**采样率**：24000Hz

**许可证**：CC-BY-NC-4.0

---

### FishSpeechBackend — Fish Speech 后端

**特点**：多语言支持最佳，跨语种克隆。

**推理管线**：

```
text → FishTokenizer → LlamaARModel → VQDecoder → HiFiGAN → waveform (22.05kHz)
```

**构造函数参数**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_dir` | str | (HF Hub) | 模型目录 |
| `ref_audio` | str | None | 参考音频 |
| `ref_text` | str | None | 参考音频文本 |

**支持的克隆**：10-30 秒参考音频

**支持语言**：中、英、日、韩、粤、法、德等

**采样率**：22050Hz

**许可证**：Apache-2.0

---

### GPTSoVITSBackend — GPT-SoVITS 后端

**特点**：极少样本克隆（5-10 秒），开源可商用。

**推理管线**：

```
text → SoVITSTokenizer → GPT2ARModel → FlowDecoder → SoVITSDecoder → waveform (32kHz)
```

**构造函数参数**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `gpt_weights` | str | (HF Hub) | GPT 权重路径 |
| `sovits_weights` | str | (HF Hub) | SoVITS 权重路径 |
| `ref_audio` | str | None | 参考音频 |

**采样率**：32000Hz

**许可证**：MIT

---

### CosyVoiceBackend — CosyVoice 后端

**特点**：质量最高，非自回归可并行，SFT 指令控制。

**推理管线**：

```
text → CosyVoiceTokenizer → FlowMatching → HiFiGAN → waveform (24kHz)
```

**构造函数参数**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_dir` | str | (HF Hub) | 模型目录 |
| `ref_audio` | str | None | 参考音频 |
| `ref_text` | str | None | 参考音频文本 |

**ODE 步数对比**

| 步数 | 质量 | 速度 |
|---|---|---|
| 5 | 中 | 极快 |
| 10 | 良 | 快 |
| 20 | 优 | 中 |
| 50 | 最佳 | 慢 |

**采样率**：24000Hz

**许可证**：Apache-2.0

---

## 下一步

- [管道使用指南](pipeline-guide.md) — 节点组合
- [插件开发指南](plugin-development.md) — 自定义节点
- [示例代码](../examples/) — 实际应用
