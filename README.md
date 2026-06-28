# Mosaic

> 一个基于 Transformers + Diffusers 的 Apache-2.0 全模态生成式 AI 框架 —— 像搭积木一样组合 AI 能力。

Mosaic 是一个模块化、可组合的全模态生成式 AI 框架。它将文本、图像、视频、音频、字幕、一致性、数字人、导出、RAG 九大领域的能力抽象为独立的"节点"（Node），用户只需用 Python 代码即可像搭积木一样自由编排这些节点，构建出任意复杂的生成式 AI 流水线。

核心理念：**解耦**。每个节点独立运行、独立测试、独立组合，后端推理引擎可插拔替换。

---

## 九大域 39 节点一览

| 域 (Domain) | 节点 (Nodes) | 数量 |
| :--- | :--- | :---: |
| **text** | `text-gen`, `summarization`, `translation`, `embedding`, `chat` | 5 |
| **image** | `txt2img`, `img2img`, `inpainting`, `outpainting`, `super-resolution` | 5 |
| **video** | `txt2vid`, `img2vid`, `vid2vid`, `interpolation`, `frame-extract` | 5 |
| **audio** | `txt2audio`, `tts`, `stt`, `audio-classify`, `audio-separate` | 5 |
| **subtitle** | `asr`, `subtitle-gen`, `subtitle-translate`, `subtitle-embed` | 4 |
| **consistency** | `face-consistency`, `style-consistency`, `color-grading`, `scene-consistency` | 4 |
| **digital-human** | `face-detect`, `lip-sync`, `expression`, `body-animation` | 4 |
| **export** | `video-export`, `audio-export`, `image-export`, `subtitle-export` | 4 |
| **rag** | `document-loader`, `embedding-retriever`, `generation` | 3 |
| **合计** | | **39** |

---

## 安装

### 基础安装

```bash
pip install mosaic
```

### 按领域安装可选依赖

```bash
# 视频处理
pip install mosaic[video]

# 音频处理
pip install mosaic[audio]

# RAG 检索增强
pip install mosaic[rag]

# 数字人
pip install mosaic[digital-human]

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

---

## 快速开始

```python
from mosaic import Pipeline
from mosaic.nodes.text import chat
from mosaic.nodes.image import txt2img
from mosaic.nodes.export import image_export

# 构建流水线：对话 → 生成图片 → 导出
pipeline = Pipeline()

# 链接节点：像搭积木一样
pipeline.add(chat(model="Qwen2.5-7B"))
pipeline.add(txt2img(model="stable-diffusion-xl"))
pipeline.add(image_export(output_dir="./outputs"))

# 运行
result = pipeline.run(
    prompt="画一只在月球上骑自行车的熊猫"
)

# 检查结果
print(f"文本: {result.text}")
print(f"图片: {result.image_path}")
```

### 多模态组合示例

```python
# 视频 + 字幕 + 数字人
from mosaic.nodes.video import txt2vid
from mosaic.nodes.subtitle import asr, subtitle_embed
from mosaic.nodes.digital_human import lip_sync

pipeline = Pipeline()
pipeline.add(txt2vid(model="stable-video-diffusion"))
pipeline.add(asr(model="whisper-large-v3"))
pipeline.add(subtitle_embed())
pipeline.add(lip_sync(model="wav2lip"))

pipeline.run(audio="speech.wav", text="Hello, Mosaic!")
```

---

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│                        🧩 Pipeline                          │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────────┐  │
│  │ Node │──▶│ Node │──▶│ Node │──▶│ Node │──▶│  Export  │  │
│  └──────┘   └──────┘   └──────┘   └──────┘   └──────────┘  │
│      │           │           │           │                  │
│      ▼           ▼           ▼           ▼                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 🔌 Backend Abstraction               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │PyTorch   │ │ONNX     │ │TensorRT  │  ...         │   │
│  │  │(default) │ │Runtime  │ │          │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    📦 Core Layer                      │   │
│  │  Node | Pipeline | Config | Registry | DataModel     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      🌐 九大域 (Domains)                     │
│                                                              │
│  text ─── image ─── video ─── audio ─── subtitle             │
│    │         │         │         │         │                 │
│    └─────────┴─────────┴────┬────┴─────────┘                 │
│                             │                                │
│              consistency ───┼─── digital_human                │
│                             │                                │
│                     export ─┴─── rag                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
mosaic/
├── pyproject.toml          # 项目元数据与依赖
├── README.md               # 项目说明
├── LICENSE                 # Apache-2.0 许可证
├── CHANGELOG.md            # 变更日志
├── CONTRIBUTING.md         # 贡献指南
├── mosaic/                 # 源码目录
│   ├── __init__.py         # 包入口，导出版本号
│   ├── core/               # 框架核心（Node、Pipeline、Config 等）
│   ├── nodes/              # 九大域节点
│   │   ├── text/           # 文本域（5 节点）
│   │   ├── image/          # 图像域（5 节点）
│   │   ├── video/          # 视频域（5 节点）
│   │   ├── audio/          # 音频域（5 节点）
│   │   ├── subtitle/       # 字幕域（4 节点）
│   │   ├── consistency/    # 一致性域（4 节点）
│   │   ├── digital_human/  # 数字人域（4 节点）
│   │   ├── export/         # 导出域（4 节点）
│   │   └── rag/            # RAG 域（3 节点）
│   ├── backends/           # 推理后端抽象
│   ├── utils/              # 工具函数
│   └── cli/                # 命令行入口
├── tests/                  # 测试目录
├── examples/               # 示例代码
└── docs/                   # 文档
```

---

## 许可证

本项目基于 Apache License 2.0 开源。详见 [LICENSE](LICENSE) 文件。

Copyright 2025 [Your Name]

---

## 致谢

Mosaic 建立在以下优秀开源项目之上：

- [Transformers](https://github.com/huggingface/transformers) — 预训练模型生态
- [Diffusers](https://github.com/huggingface/diffusers) — 扩散模型推理
- [PyTorch](https://pytorch.org/) — 深度学习框架
- 以及所有贡献者 ❤️