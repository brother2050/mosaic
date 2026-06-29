# Mosaic

> 一个基于 Transformers + Diffusers 的 Apache-2.0 全模态生成式 AI 框架 —— 像搭积木一样组合 AI 能力。

Mosaic 是一个模块化、可组合的全模态生成式 AI 框架。它将文本、图像、视频、音频、字幕、一致性、数字人、导出、RAG 九大领域的能力抽象为独立的"节点"（Node），用户只需用 Python 代码即可像搭积木一样自由编排这些节点，构建出任意复杂的生成式 AI 流水线。

核心理念：**解耦**。每个节点独立运行、独立测试、独立组合，后端推理引擎可插拔替换。

---

## 九大域 39 节点一览

| 域 (Domain) | 节点 (Nodes) | 数量 |
| :--- | :--- | :---: |
| **text** | `TextGenerator`, `Chat`, `TextRewriter`, `Translator`, `TextSummarizer`, `TextClassifier` | 6 |
| **image** | `TextToImage`, `ImageToImage`, `Inpainting`, `Upscaler`, `BackgroundRemover`, `Stylizer` | 6 |
| **video** | `TextToVideo`, `WanVideo`, `HunyuanVideo`, `LTXVideo`, `ImageToVideo`, `VideoContinuation`, `FrameInterpolator`, `FrameExtractor` | 8 |
| **audio** | `TTS`, `ASR`, `MusicGenerator`, `SoundEffect`, `VoiceClone` | 5 |
| **subtitle** | `SubtitleGenerator`, `SubtitleTranslator`, `SubtitleAligner` | 3 |
| **consistency** | `IdentityKeeper`, `StyleKeeper`, `CrossFrameConsistency` | 3 |
| **digital-human** | `AvatarDriver`, `LipSyncer`, `MotionGenerator`, `RealtimeRenderer` | 4 |
| **export** | `VideoEncoder`, `Livestreamer`, `MultiFormatExporter` | 3 |
| **rag** | `DocumentParser`, `VectorIndexer`, `Retriever`, `CitationGenerator` | 4 |
| **合计** | | **42** |

---

## TTS 扩展 — 4 个独立后端

音频域 TTS 已扩展为路由器，支持 4 个生产级 TTS 后端：

| 后端 | 声学模型 | 声码器 | 采样率 | 流式延迟 | 特点 | 许可证 |
|---|---|---|---|---|---|---|
| **ChatTTS** | LlamaForCausalLM | DVAE + Vocos | 24000Hz | ~50ms | 韵律控制丰富，延迟最低 | CC-BY-NC-4.0 |
| **Fish Speech** | LlamaForCausalLM | VQDec + HiFiGAN | 22050Hz | ~80ms | 多语言，统一词表 | Apache-2.0 |
| **GPT-SoVITS** | GPT2LMHeadModel | SoVITS(Flow+HiFiGAN) | 32000Hz | ~100ms | 极少样本克隆 | MIT |
| **CosyVoice** | FlowMatching | HiFiGAN | 24000Hz | ~300ms | 质量最高，非自回归 | Apache-2.0 |

每个后端均采用统一的四层架构：`TextFrontend → AcousticModel → Vocoder → StreamAdapter`。

---

## 视频模型支持

以下视频生成模型通过 diffusers 原生支持，已在 `WanVideo` / `HunyuanVideo` / `LTXVideo` 节点中集成：

| 模型 | 显存需求 | 默认参数 | 许可证 |
|---|---|---|---|
| **Wan2.1-14B** | ~30GB | 81 帧@16fps, 1280x720, fp16 | Apache-2.0 |
| **Wan2.1-1.3B** | ~8GB | 81 帧@16fps, 1280x720, fp16 | Apache-2.0 |
| **Wan2.2-A14B** | ~30GB | 81 帧@16fps, 1280x720, bf16 | Apache-2.0 |
| **HunyuanVideo** | ~60GB | 129 帧@24fps, 1280x720, bf16 | Tencent License |
| **LTX-Video** | ~12GB | 97 帧@30fps, 768x512, bf16 | OpenRAIL-M |

---

## 框架核心能力

- **Pipeline 编排引擎** —— 串行、并行分支、Merge，支持 `|` 运算符声明式串联
- **显存调度器** —— LRU 自动加载/卸载，跟踪每个节点的显存占用
- **事件总线** —— 进度、错误、中间结果实时上报
- **异步执行** —— `AsyncTask` + `TaskManager`，支持并发任务编排
- **插件系统** —— entry_points + 装饰器 + 目录扫描三种注册机制
- **CLI 工具** —— `list` / `info` / `create-node` / `run` / `doctor` / `version`

---

## 安装

### 基础安装

```bash
pip install mosaic
```

### 按领域安装可选依赖

```bash
# 视频处理（diffusers 视频模型 + imageio）
pip install mosaic[video]

# 音频处理（soundfile + librosa + edge-tts）
pip install mosaic[audio]

# RAG 检索增强（faiss + chromadb + sentence-transformers）
pip install mosaic[rag]

# 数字人（trimesh + insightface）
pip install mosaic[digital-human]

# 一致性域（insightface + scikit-image）
pip install mosaic[consistency]

# ONNX Runtime（CPU 版本，RIFE 帧插值）
pip install mosaic[onnx]

# 开发环境
pip install mosaic[dev]

# 全部安装
pip install mosaic[all]
```

### 从源码安装（开发模式）

```bash
git clone https://github.com/your-org/mosaic.git
cd mosaic
pip install -e ".[dev]"
```

### 验证安装

```bash
mosaic --version    # 或 mosaic -V / mosaic version
# mosaic 0.1.0

mosaic doctor       # 环境诊断
```

---

## 快速开始

### 示例 1：文字描述 → 图片

```python
from mosaic import Pipeline
from mosaic.nodes.text import Chat
from mosaic.nodes.image import TextToImage
from mosaic.nodes.export import MultiFormatExporter

# 构建流水线
pipeline = Pipeline()
pipeline.add(Chat(model="Qwen/Qwen2.5-7B-Instruct"))
pipeline.add(TextToImage(model="stabilityai/stable-diffusion-xl-base-1.0"))
pipeline.add(MultiFormatExporter())

# 运行
result = pipeline.run(prompt="画一只在月球上骑自行车的熊猫")
print(f"输出: {result}")
```

### 示例 2：文本 → 语音（ChatTTS 流式）

```python
import asyncio
from mosaic.nodes.audio import TTS

tts = TTS(backend="chattts")

# 阻塞合成
result = tts.run({"text": "你好，欢迎使用 Mosaic！", "language": "zh"})
audio = result.get("audio")  # AudioData 对象

# 流式合成（首批延迟 ~50ms）
async def stream_demo():
    async for chunk in tts.synthesize_stream(text="流式合成测试", language="zh"):
        play(chunk)  # 立即播放当前 chunk

asyncio.run(stream_demo())
```

### 示例 3：文字 → 视频（Wan2.1）

```python
from mosaic.nodes.video import WanVideo

wan = WanVideo(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    enable_cpu_offload=True,
    enable_vae_tiling=True,
)

result = wan.run({
    "prompt": "一只猫在海滩上散步，夕阳西下",
    "num_frames": 81,
    "fps": 16,
})
video = result.get("video")  # VideoData 对象
```

更多示例见 [`examples/`](examples/) 目录。

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Mosaic Framework                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Pipeline Orchestrator                   │    │
│  │  (串行 / 并行 / 分支 / Merge / 条件路由 / 异步执行)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │           │           │           │           │             │
│       ▼           ▼           ▼           ▼           ▼             │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │  Text  │  │ Image  │  │ Video  │  │ Audio  │  │  ...   │         │
│  │ Domain │  │ Domain │  │ Domain │  │ Domain │  │        │         │
│  │ (6节点)│  │ (6节点)│  │ (8节点)│  │ (5节点)│  │        │         │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘         │
│       │           │           │           │                         │
│       └────────┬──┴───────────┴───────────┘                         │
│                ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                       Core Framework                        │    │
│  │  Node | Registry | Pipeline | Scheduler | EventBus |       │    │
│  │  AsyncTask | TaskManager | PluginManager | Context         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                    │
│                ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    TTS Backend Layer (4 后端)               │    │
│  │  ChatTTS | Fish Speech | GPT-SoVITS | CosyVoice             │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │ TextFrontend → AcousticModel → Vocoder → StreamAdapter│   │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                    │
│                ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Engine Backends                         │    │
│  │  PyTorch (default) | ONNX Runtime | TensorRT (reserved)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 文档导航

- 📖 [快速开始](docs/getting-started.md) — 5 分钟上手
- 🏗️ [架构设计](docs/architecture.md) — 核心模块详解
- 📚 [节点参考手册](docs/nodes-reference.md) — 全部 42 节点文档
- 🔗 [管道使用指南](docs/pipeline-guide.md) — Pipeline 编排
- 🎙️ [TTS 完整指南](docs/tts-guide.md) — 4 后端详解
- 🎬 [视频模型指南](docs/video-models.md) — Wan/HunyuanVideo/LTX-Video
- 🧩 [插件开发指南](docs/plugin-development.md) — 自定义节点/后端
- 🖥️ [CLI 参考手册](docs/cli-reference.md) — 命令行工具
- 📂 [示例代码](examples/) — 11 个完整示例

---

## 项目结构

```
mosaic/
├── pyproject.toml          # 项目元数据与依赖
├── README.md               # 项目说明
├── LICENSE                 # Apache-2.0 许可证
├── CHANGELOG.md            # 变更日志
├── mosaic/                 # 源码目录
│   ├── core/               # 框架核心（Node、Pipeline、Registry 等）
│   ├── nodes/              # 九大域节点实现
│   │   ├── text/           # 文本域（6 节点）
│   │   ├── image/          # 图像域（6 节点）
│   │   ├── video/          # 视频域（8 节点）
│   │   ├── audio/          # 音频域（5 节点 + TTS 后端）
│   │   ├── subtitle/       # 字幕域（3 节点）
│   │   ├── consistency/    # 一致性域（3 节点）
│   │   ├── digital_human/  # 数字人域（4 节点）
│   │   ├── export/         # 导出域（3 节点）
│   │   └── rag/            # RAG 域（4 节点）
│   ├── backends/           # 推理后端抽象
│   ├── cli/                # 命令行工具
│   └── templates/          # create-node 模板
├── tests/                  # 测试目录（phase1-7 + tts）
├── examples/               # 示例代码
├── scripts/                # 工具脚本
└── docs/                   # 文档
```

---

## 许可证

本项目基于 Apache License 2.0 开源。详见 [LICENSE](LICENSE) 文件。

---

## 致谢

Mosaic 建立在以下优秀开源项目之上：

- [Transformers](https://github.com/huggingface/transformers) — 预训练模型生态
- [Diffusers](https://github.com/huggingface/diffusers) — 扩散模型推理
- [PyTorch](https://pytorch.org/) — 深度学习框架
- [ChatTTS](https://github.com/2noise/ChatTTS) / [Fish Speech](https://github.com/fishaudio/fish-speech) / [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) / [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — TTS 后端
- 以及所有贡献者
