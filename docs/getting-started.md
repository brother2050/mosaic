# Mosaic 快速开始

> 5 分钟从零到生成第一张图片、第一段语音、第一段视频。

## 目录

- [Mosaic 是什么](#mosaic-是什么)
- [安装](#安装)
- [环境检查](#环境检查)
- [第一个示例：文字生成图片](#第一个示例文字生成图片)
- [第二个示例：文字生成语音](#第二个示例文字生成语音)
- [核心概念](#核心概念)
- [常见问题 FAQ](#常见问题-faq)

---

## Mosaic 是什么

Mosaic 是一个基于 Transformers + Diffusers 的全模态生成式 AI 框架，把文本、图像、视频、音频等能力抽象为可独立注册、自由组合的"节点"（Node）。你只需用 Python 就能像搭积木一样把它们串成任意复杂的生成式 AI 流水线。

**核心特点**

- **解耦**：节点之间通过 `MosaicData` 解耦，不直接依赖具体实现
- **可插拔**：TTS 已扩展为 4 个独立后端（ChatTTS / Fish / GPT-SoVITS / CosyVoice），可按场景切换
- **显存友好**：内置 LRU 调度器自动管理模型加载/卸载
- **跨域编排**：`Pipeline` 引擎支持串行、并行分支、条件路由
- **可扩展**：三种插件机制（entry_points / 装饰器 / 目录扫描）

---

## 安装

### 基础安装

```bash
pip install mosaic
```

### 按场景选择额外依赖

```bash
# 视频生成（Wan2.1 / HunyuanVideo / LTX-Video）
pip install "mosaic[video]"

# 音频（含 edge-tts 默认 TTS 后端）
pip install "mosaic[audio]"

# RAG 检索增强
pip install "mosaic[rag]"

# 数字人
pip install "mosaic[digital-human]"

# ONNX Runtime（CPU，RIFE 帧插值 / 部分数字人推理）
pip install "mosaic[onnx]"

# 全部安装
pip install "mosaic[all]"
```

> **GPU 用户注意**：若需要 ONNX GPU 推理，请额外安装 `mosaic[onnx-gpu]` 并先卸载 CPU 版本：
> ```bash
> pip uninstall onnxruntime && pip install "mosaic[onnx-gpu]"
> ```

### 从源码安装

```bash
git clone https://github.com/your-org/mosaic.git
cd mosaic
pip install -e ".[dev]"
```

---

## 环境检查

安装完成后，先用 `mosaic doctor` 跑一遍环境诊断：

```bash
mosaic doctor
```

预期输出（节选）：

```
Mosaic 环境诊断
==================================================

  ✓  Python 3.10.12
  ✓  torch 已安装 (v2.12.1)
  ✓  transformers 已安装 (v5.12.1)
  ✓  diffusers 已安装 (v0.32.0)
  ✓  GPU 可用: NVIDIA A100 80GB (80.0 GB 显存)
  ⚠  soundfile 未安装（可选依赖）
  ⚠  faiss-cpu 未安装（可选依赖）
  ✓  已注册 42 个节点
  ✓  已加载 0 个插件

诊断完成: 2 个警告, 0 个错误
```

任何 `✗` 项表示必需依赖缺失；`⚠` 项只是可选依赖缺失，不影响核心功能。

---

## 模型下载

节点运行时自动从 Hugging Face 下载模型权重到 `~/.cache/huggingface/hub/`。**首次运行每个模型都需要下载（几 GB），请确保网络通畅。**

### 模型仓库 ID 速查表

按域分类列出所有默认模型的 Hugging Face 仓库 ID：

**文本域**

| 节点 | 默认模型 | 仓库 ID | 大小 |
|------|---------|---------|------|
| TextGenerator / Chat / TextRewriter | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~15GB |
| Translator / TextSummarizer | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~15GB |
| TextClassifier | Qwen2.5-7B-Instruct + bart-large-mnli | `Qwen/Qwen2.5-7B-Instruct` / `facebook/bart-large-mnli` | ~15GB / ~1.6GB |

**图像域**

| 节点 | 默认模型 | 仓库 ID | 大小 |
|------|---------|---------|------|
| TextToImage / Stylizer | SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` | ~6.5GB |
| ImageToImage | SDXL Refiner | `stabilityai/stable-diffusion-xl-refiner-1.0` | ~6.5GB |
| Inpainting | SDXL Inpainting | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | ~6.5GB |
| Upscaler | SD x4 Upscaler | `stabilityai/stable-diffusion-x4-upscaler` | ~3.5GB |
| BackgroundRemover | RMBG-2.0 | `briaai/RMBG-2.0` | ~1.4GB |

**视频域**

| 节点 | 默认模型 | 仓库 ID | 大小 |
|------|---------|---------|------|
| TextToVideo / VideoContinuation | CogVideoX-5b | `THUDM/CogVideoX-5b` | ~10GB |
| WanVideo | Wan2.1-T2V-14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | ~28GB |
| HunyuanVideo | HunyuanVideo | `tencent/HunyuanVideo` | ~13GB |
| LTXVideo | LTX-Video | `Lightricks/LTX-Video` | ~2.5GB |
| ImageToVideo | SVD img2vid | `stabilityai/stable-video-diffusion-img2vid-xt` | ~5GB |

**音频域**

| 节点 | 默认模型 | 仓库 ID | 大小 |
|------|---------|---------|------|
| TTS (edge_tts) | Edge TTS | 云端服务，无需下载 | — |
| TTS (ChatTTS) | ChatTTS | `2Noise/ChatTTS` | ~1.2GB |
| TTS (Fish) | Fish Speech | `fishaudio/fish-speech-1.5` | ~3GB |
| TTS (SoVITS) | GPT-SoVITS | `lj1995/GPT-SoVITS` | ~1.5GB |
| TTS (CosyVoice) | CosyVoice2 | `FunAudioLLM/CosyVoice2-0.5B` + `Qwen/Qwen2.5-1.5B-Instruct` | ~2GB + ~3GB |
| ASR / SubtitleGenerator | Whisper | `openai/whisper-large-v3` | ~3GB |
| MusicGenerator | MusicGen | `facebook/musicgen-small` | ~2GB |
| SoundEffectGenerator | AudioLDM2 | `cvssp/audioldm2` | ~6GB |

**一致性域**

| 节点 | 默认模型 | 仓库 ID |
|------|---------|---------|
| IdentityKeeper | InstantID | `InstantX/InstantID` |
| StyleKeeper | IP-Adapter | `h94/IP-Adapter` |
| CrossFrameConsistency | SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` |

**数字人域**

| 节点 | 默认模型 | 仓库 ID |
|------|---------|---------|
| AvatarDriver / RealtimeRenderer | LivePortrait | `KwaiVGI/LivePortrait` |
| LipSyncer | MuseTalk | `KwaiVGI/MuseTalk` |
| MotionGenerator | MotionGPT | `PrimeIntellect/MotionGPT` |

**RAG 域**

| 节点 | 默认模型 | 仓库 ID |
|------|---------|---------|
| VectorIndexer / Retriever | all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` |
| CitationGenerator | Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` |

### 预下载命令

```bash
# 预下载常用模型（可选，加速首次运行）
python -c "
from huggingface_hub import snapshot_download

# 文本模型
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')  # 轻量版，适合快速测试

# 图像模型
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0')

# ASR 模型
snapshot_download('openai/whisper-large-v3')

# RAG 嵌入模型
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
"
```

> **离线环境**：先在有网机器上完成下载，将 `~/.cache/huggingface/` 复制到离线机器相同路径即可。TTS 后端模型下载详见 [TTS 完整指南](tts-guide.md)。

### 自定义模型路径

默认情况下，模型下载到 `~/.cache/huggingface/hub/`。如果需要将模型存放在其他位置，有以下三种方式：

**方式 1：节点构造函数传入本地路径（推荐）**

所有节点的 `model` 参数既接受 HuggingFace 仓库 ID，也接受本地目录路径。模型已在本地时，直接传路径即可：

```python
from mosaic import Pipeline, MosaicData
from mosaic.nodes.image import TextToImage

# 传入本地路径而非 HF 仓库 ID
pipeline = Pipeline()
pipeline.add(TextToImage(model="/data/models/sdxl-base"))

result = pipeline.run(MosaicData(prompt="a cat"))
```

TTS 后端同理，通过 `model_path` 指定：

```python
from mosaic.nodes.audio.tts import TTS

tts = TTS(backend="chattts", model="/data/models/chattts")
result = tts.run(MosaicData(text="你好", language="zh"))
```

**方式 2：设置 HF_HOME 环境变量**

修改 HuggingFace 默认缓存根目录，所有自动下载的模型都会存到该位置：

```bash
# 设置缓存目录（所有 from_pretrained 下载的模型都会存到这里）
export HF_HOME=/data/hf_cache

# 或在 Python 中设置
import os
os.environ["HF_HOME"] = "/data/hf_cache"
```

**方式 3：国内镜像加速下载**

设置镜像端点加速模型下载（不影响已下载的模型）：

```bash
# 使用 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 Mosaic 专用镜像变量
export MOSAIC_HF_MIRROR=https://hf-mirror.com
```

> TTS 后端模型会自动使用镜像加速下载。常规节点（文本/图像/视频等）依赖 HuggingFace 原生 `HF_ENDPOINT` 机制。

---

## 第一个示例：文字生成图片

```bash
pip install mosaic
```

`text_to_image.py`：

```python
from mosaic import Pipeline, MosaicData
from mosaic.nodes.image import TextToImage

# 1. 创建一个 Pipeline
pipeline = Pipeline()

# 2. 添加节点（按顺序）
pipeline.add(TextToImage(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    num_inference_steps=30,
    guidance_scale=7.5,
    width=1024,
    height=1024,
))

# 3. 运行（run 接收 MosaicData 对象，不是关键字参数）
result = pipeline.run(MosaicData(
    prompt="a cup of coffee on a wooden table, morning light",
))

# 4. 查看结果
image = result.get("image")
image.save("coffee.png")
print("Saved to coffee.png")
```

**预期输出**

```
Saved to coffee.png
```

执行后会在当前目录生成 `coffee.png`（约 1024×1024 的咖啡杯图片）。

---

## 第二个示例：文字生成语音

```bash
pip install "mosaic[audio]"  # edge-tts 后端
```

`text_to_speech.py`：

```python
from mosaic import Pipeline, MosaicData
from mosaic.nodes.audio import TTS

pipeline = Pipeline()
pipeline.add(TTS(
    backend="edge_tts",          # 默认云端后端，无需 GPU
    voice="zh-CN-XiaoxiaoNeural",
    language="zh",
))

result = pipeline.run(MosaicData(text="你好，欢迎使用 Mosaic 框架！"))
audio = result.get("audio")
audio.save("hello.wav")
print(f"已生成音频: {audio.duration:.2f} 秒, {audio.sample_rate} Hz")
```

**预期输出**

```
已生成音频: 3.42 秒, 24000 Hz
```

如果想用本地 TTS 后端（需要更多显存和权重文件），把 `backend` 改为：

```python
pipeline.add(TTS(backend="chattts"))    # 24000Hz, AR 流式
pipeline.add(TTS(backend="fish"))       # 22050Hz, 多语言
pipeline.add(TTS(backend="cosyvoice"))  # 24000Hz, 高质量
pipeline.add(TTS(backend="sovits"))     # 32000Hz, 极少样本克隆
```

详见 [TTS 完整指南](tts-guide.md)。

---

## 流式文本生成

`Chat` 和 `TextGenerator` 支持流式输出，逐 token 实时打印，延迟低至 100ms：

```python
from mosaic import MosaicData
from mosaic.nodes.text import Chat

chat = Chat(model="Qwen/Qwen2.5-1.5B-Instruct")

# 流式对话，逐 token 打印（在节点上调用 stream 方法）
for chunk in chat.stream(MosaicData(
    messages=[{"role": "user", "content": "用 Python 写一首诗"}],
    temperature=0.8,
)):
    print(chunk, end="", flush=True)
```

> 流式生成在**节点层面**调用 `chat.stream()` 或 `gen.stream()`，返回生成器逐个 yield 文本片段。非流式则调用 `run()` 返回完整 `MosaicData` 结果。

---

## 第三个示例：文字生成视频

```bash
pip install "mosaic[video]"
```

`text_to_video.py`：

```python
from mosaic import MosaicData
from mosaic.nodes.video import WanVideo

wan = WanVideo(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",  # 显存不足时选 1.3B 轻量版
    enable_cpu_offload=True,
    enable_vae_tiling=True,
)

result = wan.run(MosaicData(
    prompt="a cat walking on the beach, sunset",
    num_frames=81,   # 约 5 秒 @ 16fps
    fps=16,
))

video = result.get("video")  # VideoData 对象
print(f"已生成视频: {result.get('num_frames')} 帧, {result.get('duration'):.2f} 秒")
```

**预期输出**

```
已生成视频: 81 帧, 5.06 秒
```

更多模型和最佳实践见 [视频模型指南](video-models.md)。

---

## 核心概念

### Node（节点）

节点是 Mosaic 的最小能力单元。每个节点：

- 有明确的 `domain`（域，如 `text` / `image` / `video`）
- 暴露 `__call__(input_data) -> MosaicData` 接口（即可调用 `node(data)`）
- 由 `@registry.register` 装饰器自动注册到全局注册表
- 通过 `load()` / `unload()` 管理模型生命周期

```python
from mosaic.core.registry import registry
from mosaic.core.node import Node

@registry.register
class MyNode(Node):
    name = "my-node"
    domain = "custom"

    def run(self, input_data):
        return input_data.set("result", "hello")
```

### Domain（域）

域是节点的逻辑分组。Mosaic 内置 9 个域：

| 域 | 节点数 | 典型能力 |
|---|---|---|
| `text` | 6 | 文本生成、对话、改写、翻译、摘要、分类 |
| `image` | 6 | 文生图、图生图、局部重绘、超分、去背景、风格化 |
| `video` | 8 | 文生视频（Wan/Hunyuan/LTX/CogVideoX）、图生视频、续写、插帧、拆帧 |
| `audio` | 5 | TTS、ASR、音乐生成、音效、语音克隆 |
| `subtitle` | 3 | 字幕生成、字幕翻译、时间轴对齐 |
| `consistency` | 3 | 人脸一致性、风格一致性、跨帧一致性 |
| `digital-human` | 4 | 形象驱动、口型同步、动作生成、实时渲染 |
| `export` | 3 | 视频编码、直播推流、多格式导出 |
| `rag` | 4 | 文档解析、向量化、检索、引用生成 |

### Pipeline（管道）

管道是节点的编排器，支持：

- **串行**：`pipeline.add(A); pipeline.add(B)` — A 完成后 B 开始
- **`|` 运算符**：`A | B | C` — 声明式串联
- **并行分支**：`Branch([pipeline1, pipeline2])` — 同输入多路径并行
- **合并**：`Merge(strategy="concat")` — 多上游合并为单输入
- **异步执行**：`pipeline.run_async(...)` — 后台运行，返回 `Task` 句柄

```python
from mosaic import Pipeline
from mosaic.nodes.text import Chat
from mosaic.nodes.image import TextToImage

# 串行
pipe = Pipeline()
pipe.add(Chat(model="Qwen2.5-7B"))
pipe.add(TextToImage(model="SDXL"))

# | 运算符
pipe2 = Chat() | TextToImage()
```

### MosaicData（数据）

节点间传递的不可变数据容器。支持 `.get(key)` / `.set(key, value)`，类型化字段由各域定义（`ImageData` / `VideoData` / `AudioData` / `SubtitleData` / `DocumentData`）。

```python
data = MosaicData(prompt="hello")
data = data.set("result", "world")
data.get("prompt")   # "hello"
data.get("result")   # "world"
```

### TTSBackend（TTS 后端）

TTS 节点本身是一个**路由器**，真正执行合成的是后端实例。Mosaic 内置 4 个生产级后端，每个都是统一的四层架构：`TextFrontend → AcousticModel → Vocoder → StreamAdapter`。

```python
from mosaic.nodes.audio import TTS

tts = TTS(backend="chattts")   # 后端：chattts / fish / sovits / cosyvoice / edge_tts
result = tts.run(MosaicData(text="你好", language="zh"))
audio = result.get("audio")  # AudioData 对象
```

详见 [TTS 完整指南](tts-guide.md)。

---

## 常见问题 FAQ

### 1. 安装时报 `Microsoft Visual C++ 14.0 is required`

需要安装 C++ 编译器：

- **Windows**：安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选"使用 C++ 的桌面开发"
- **macOS**：`xcode-select --install`
- **Linux**：`sudo apt install build-essential`

### 2. 显存不足（CUDA OOM）

不同模型的显存需求（FP16 精度估算，量化可降低 40-70%）：

| 模型 | 显存需求 | 说明 |
|---|---|---|
| ChatTTS | 2GB | 最轻量 TTS |
| Whisper-large-v3 | 5GB | ASR 语音识别 |
| SDXL (fp16) | 8GB | 文生图 |
| Qwen2.5-7B-Instruct | 15GB | 文本生成/对话 |
| LTX-Video | 12GB | 快速视频生成 |
| CogVideoX-5B | 18GB | 中等视频生成 |
| Wan2.1-1.3B | 8GB | 轻量视频生成 |
| Wan2.1-14B | 30GB | 高质量视频生成 |
| HunyuanVideo | 60GB | 最高质量视频 |

> 以上为 FP16 精度下的模型权重显存估算。实际推理还需额外 KV cache / 激活内存（约 +10-20%）。使用 INT8 量化可减半，INT4 量化可降至 1/4。

启用 CPU offload 可将显存需求降低约 40%：

```python
WanVideo(model="Wan-AI/Wan2.1-T2V-14B-Diffusers", enable_cpu_offload=True)
```

### 3. diffusers 加载时 `cannot be loaded as it does not seem to have any loading methods`

通常是 `sentencepiece` 缺失（影响 T5 tokenizer）。修复：

```bash
pip install sentencepiece
```

Mosaic 已通过 `safe_load_pipeline()` 工具预检测并给出明确错误信息。

### 4. Wan2.1-14B 找不到权重

必须使用带 `-Diffusers` 后缀的仓库名（原始格式是 research 仓库，不含 diffusers 索引）：

```python
# 正确
WanVideo(model="Wan-AI/Wan2.1-T2V-14B-Diffusers")

# 错误（原始格式，不能直接 from_pretrained）
WanVideo(model="Wan-AI/Wan2.1-T2V-14B")
```

WanVideo 节点会自动给不带后缀的名称补全 `-Diffusers`。

### 5. 如何切换 TTS 后端？

通过 `backend` 参数：

```python
TTS(backend="chattts")   # 24kHz, AR 流式, 延迟最低
TTS(backend="fish")      # 22kHz, 多语言
TTS(backend="cosyvoice") # 24kHz, 高质量
TTS(backend="sovits")    # 32kHz, 极少样本克隆
TTS(backend="edge_tts")  # 云端 Azure, 无需 GPU
```

### 6. Pipeline 如何在节点间共享大模型？

通过 `Scheduler`：所有节点共享显存池，LRU 自动淘汰：

```python
from mosaic.core.scheduler import get_scheduler

sched = get_scheduler()  # 全局单例
# 节点 load() 时会自动注册到 scheduler
# 显存不足时按 LRU 自动卸载最近未使用的模型
```

### 7. 怎么监听节点进度？

订阅事件总线：

```python
from mosaic.core.events import get_event_bus, EventType

bus = get_event_bus()

@bus.on(EventType.PROGRESS)
def on_progress(event):
    print(f"[{event.node}] {event.current}/{event.total}: {event.message}")
```

### 8. 如何并行执行多个独立节点？

使用 `Branch`：

```python
from mosaic.core.pipeline import Branch, Merge

pipe = Pipeline()
pipe.add(Branch([
    TextToImage(model="SDXL"),
    TTS(backend="chattts"),
    SubtitleGenerator(),
]))
pipe.add(Merge())  # 合并三条路径的输出
```

### 9. 流式 TTS 怎么用？

```python
tts = TTS(backend="chattts")

async for chunk in tts.synthesize_stream(text="流式合成", language="zh"):
    # chunk 是 AudioData，可以立即播放
    play(chunk)
```

ChatTTS 流式延迟约 50ms，是 4 个后端中最低的。

### 10. 如何自定义节点？

三种方式：

```python
# 方式 1：继承 Node 基类
from mosaic.core.node import Node
from mosaic.core.registry import registry

@registry.register
class MyNode(Node):
    name = "my-node"
    domain = "custom"
    def run(self, data):
        return data.set("out", self._compute(data.get("in")))

# 方式 2：CLI 模板
# mosaic create-node --domain custom --name my-node

# 方式 3：发布 PyPI 插件包（通过 entry_points）
```

详见 [插件开发指南](plugin-development.md)。

### 11. 如何将管道保存为 YAML/JSON 并从 CLI 运行？

`mosaic run` 命令支持从 YAML 或 JSON 文件加载管道定义：

```yaml
# pipeline.yaml
nodes:
  - type: TextToImage
    params:
      model: stabilityai/stable-diffusion-xl-base-1.0
input:
  prompt: "a cup of coffee"
```

```bash
mosaic run pipeline.yaml
```

详见 [CLI 参考手册](cli-reference.md)。

### 12. 在没有 GPU 的环境能跑吗？

能，但有部分限制：

- ✅ 文本域（用小模型如 Qwen2.5-0.5B）
- ✅ edge-tts 后端（云端）
- ✅ RAG 域（CPU 检索）
- ❌ 视频/图像/数字人域（需要 GPU）
- ⚠️ 本地 TTS 后端（ChatTTS 可在 CPU 上跑，但慢）

---

## 下一步

- 阅读 [架构设计](architecture.md) 了解内部模块
- 阅读 [TTS 完整指南](tts-guide.md) 选择合适的后端
- 浏览 [示例代码](../examples/) 找灵感
- 阅读 [节点参考手册](nodes-reference.md) 了解全部 42 节点
