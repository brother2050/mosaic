# Mosaic CLI 参考手册

> `mosaic` 命令行工具的完整参数说明。

## 目录

- [全局说明](#全局说明)
- [mosaic list](#mosaic-list)
- [mosaic info](#mosaic-info)
- [mosaic create-node](#mosaic-create-node)
- [mosaic run](#mosaic-run)
- [mosaic version](#mosaic-version)
- [mosaic doctor](#mosaic-doctor)

---

## 全局说明

### 安装 CLI

CLI 随 `mosaic` 包自动安装。验证：

```bash
mosaic --version
# mosaic 0.1.0
```

### 全局选项

| 选项 | 说明 |
|---|---|
| `--help`, `-h` | 显示帮助 |

### 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 一般错误 |
| 130 | 用户中断（Ctrl+C） |

---

## mosaic list

**作用**：列出所有已注册的节点或插件。

### 用法

```bash
mosaic list [OPTIONS]
```

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--domain` | str | (无) | 按域过滤：`text` / `image` / `video` / `audio` / `subtitle` / `consistency` / `digital_human` / `export` / `rag` |
| `--plugins` | flag | False | 仅列出插件（不显示内置节点） |

### 示例

#### 列出所有节点

```bash
mosaic list
```

输出（节选）：

```
Name                     Domain         Version  Description
-----------------------  -------------  -------  ------------------------------------------------------------
asr                      audio          0.1.0    Convert speech to text using OpenAI Whisper. Supports mul...
avatar-driver            digital_human  0.1.0    Drive a digital human avatar from a source image using a ...
background-remover       image          0.1.0    Remove the background from an image, returning a transpar...
chat                     text           0.1.0    Multi-turn chat: generate a reply from conversation histo...
...                      ...            ...      ...

共 42 个节点。
可用域: audio, consistency, digital_human, export, image, rag, subtitle, text, video
```

#### 列出某个域的节点

```bash
mosaic list --domain video
```

输出：

```
Name                   Domain  Version  Description
---------------------  ------  -------  ------------------------------------------------------------
frame-extractor        video   0.1.0    Extract frames from a video. Supports 'all', 'interval', ...
frame-interpolation    video   0.1.0    Interpolate intermediate frames between existing video fr...
hunyuan-video          video   0.1.0    Generate video from text using Tencent HunyuanVideo. Supp...
image-to-video         video   0.1.0    Generate video from an input image using Stable Video Dif...
ltx-video              video   0.1.0    Generate video from text using Lightricks LTX-Video. Fast...
text-to-video          video   0.1.0    Generate video from text using Stable Video Diffusion or ...
video-continuation     video   0.1.0    Continue or extend an existing video by generating addit...
wan-video              video   0.1.0    Generate video from text using Wan2.1/Wan2.2. Supports au...

共 8 个节点。
```

#### 列出插件

```bash
mosaic list --plugins
```

输出：

```
未发现任何插件。
提示: 使用 @mosaic.node 装饰器或安装第三方插件包来扩展节点。
```

---

## mosaic info

**作用**：显示节点的详细信息。

### 用法

```bash
mosaic info <NODE_NAME>
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `NODE_NAME` | 是 | 节点名称或类名（positional） |

### 示例

```bash
mosaic info text-to-image
```

输出：

```
名称:      text-to-image
域:        image
版本:      0.1.0
描述:      Generate images from text prompts using Stable Diffusion XL. Supports negative prompts, resolution, steps, guidance scale, and seed.
输入类型:  text, mosaic
输出类型:  image
模型信息:
  name: stabilityai/stable-diffusion-xl-base-1.0
  source: HuggingFace
  license: OpenRAIL++-M (CreativeML Open RAIL++-M License)
  vram_gb: 8.0
  dtype: float16
  device: cuda
  attention_slicing: True
  vae_slicing: True
  cpu_offload: False
```

也支持使用类名查找：

```bash
mosaic info TextToImage
```

---

## mosaic create-node

**作用**：使用模板创建新节点文件。

### 用法

```bash
mosaic create-node [OPTIONS]
```

无参数时进入交互模式，逐步提示输入。

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--domain` | str | (交互输入) | 节点所属域（如 `text` / `image` / `custom`） |
| `--name` | str | (交互输入) | 节点名称 |
| `--description` | str | (无) | 节点描述 |
| `--output` | path | `./my_nodes/` | 输出目录 |
| `--model` | str | (无) | 默认模型标识 |
| `--author` | str | (无) | 作者名称 |

### 示例

#### 参数式创建

```bash
mosaic create-node --domain text --name sentiment-analyzer
```

输出：

```
节点模板已生成: ['/path/to/my_nodes/sentiment_analyzer.py', '/path/to/my_nodes/__init__.py', '/path/to/my_nodes/test_sentiment_analyzer.py', '/path/to/my_nodes/README.md']
```

#### 自定义输出路径

```bash
mosaic create-node --domain text --name my-node --output ./my_nodes/
```

#### 交互模式

```bash
mosaic create-node
```

交互提示：

```
域 (domain) [custom]: text
节点类名 (CamelCase, 如 SentimentAnalyzer): SentimentAnalyzer
描述 (description): 情感分析节点
输入类型 (逗号分隔) [text]: text
输出类型 (逗号分隔) [text]: text
模型名称 (model_name) [留空跳过]:
作者 (author) [留空跳过]:
输出目录 [./]:
```

---

## mosaic run

**作用**：从 YAML/JSON 文件运行管道。

### 用法

```bash
mosaic run <PIPELINE_FILE>
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `PIPELINE_FILE` | 是 | 管道定义文件路径（`.yaml` / `.yml` / `.json`） |

### 管道文件格式

管道定义文件是一个 YAML 或 JSON 字典，包含 `nodes` 和 `input` 两个顶层键：

```yaml
# pipeline.yaml
nodes:
  - name: text-generator       # 可选，节点别名
    type: TextGenerator        # 节点类名或注册名
    params:                    # 构造参数
      model: Qwen/Qwen2.5-7B-Instruct
input:
  prompt: "你好"
```

JSON 格式：

```json
{
  "nodes": [
    {
      "type": "TextToImage",
      "params": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0"
      }
    }
  ],
  "input": {
    "prompt": "a cup of coffee on a wooden table"
  }
}
```

### 运行

```bash
mosaic run pipeline.yaml
```

输出：

```
管道执行完成，耗时 12.345s
输出:
  image: <PIL.Image.Image>
```

### 多节点管道

```yaml
nodes:
  - name: chat
    type: Chat
    params:
      model: Qwen/Qwen2.5-7B-Instruct
  - name: tts
    type: TTS
    params:
      backend: edge_tts
      voice: zh-CN-XiaoxiaoNeural
input:
  prompt: "你好"
```

---

## mosaic version

**作用**：显示版本信息。

### 用法

```bash
mosaic version
```

### 示例

```bash
mosaic version
```

输出：

```
mosaic 0.1.0
```

---

## mosaic doctor

**作用**：诊断环境配置问题，检查 Python 版本、核心依赖、GPU、可选依赖、节点注册和插件加载状态。

### 用法

```bash
mosaic doctor
```

### 示例

```bash
mosaic doctor
```

输出：

```
Mosaic 环境诊断
==================================================

  ✓  Python 3.10.12
  ✓  torch 已安装 (v2.12.1)
  ✓  transformers 已安装 (v5.12.1)
  ✗  diffusers 未安装（必需依赖）
  ⚠  GPU 不可用（将使用 CPU 推理）
  ✓  imageio 已安装 (v2.37.3)
  ⚠  soundfile 未安装（可选依赖）
  ⚠  librosa 未安装（可选依赖）
  ⚠  faiss-cpu 未安装（可选依赖）
  ⚠  chromadb 未安装（可选依赖）
  ⚠  sentence-transformers 未安装（可选依赖）
  ⚠  insightface 未安装（可选依赖）
  ✓  onnxruntime 已安装 (v1.23.2)
  ✓  已注册 42 个节点
  ✓  已加载 0 个插件
  ⚠  模型缓存目录不存在: ~/.cache/huggingface（首次下载模型时将自动创建）

诊断完成: 8 个警告, 1 个错误
```

符号含义：

| 符号 | 状态 |
|---|---|
| ✓ | 通过 |
| ⚠ | 警告（可选依赖缺失等） |
| ✗ | 错误（必需依赖缺失等） |

检查项包括：

- **Python 版本**：需要 >= 3.10
- **核心依赖**：torch / transformers / diffusers
- **GPU**：CUDA 可用性与显存
- **可选依赖**：imageio / imageio-ffmpeg / soundfile / librosa / faiss / chromadb / sentence-transformers / insightface / onnxruntime
- **节点注册**：已注册节点数量
- **插件加载**：已加载插件数量
- **模型缓存**：HuggingFace 缓存目录状态

---

## 下一步

- [快速开始](getting-started.md) — 5 分钟上手
- [节点参考手册](nodes-reference.md) — 全部节点
- [插件开发指南](plugin-development.md) — 自定义节点
- [示例代码](../examples/) — 实际应用
