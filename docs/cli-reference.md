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
| `--version` | 显示版本 |
| `--verbose`, `-v` | 详细输出 |
| `--quiet`, `-q` | 静默模式 |
| `--no-color` | 禁用彩色输出 |

### 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 参数错误 |
| 3 | 未找到（节点/文件） |
| 4 | 依赖缺失 |
| 5 | 配置错误 |

---

## mosaic list

**作用**：列出所有已注册的节点。

### 用法

```bash
mosaic list [OPTIONS]
```

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--domain`, `-d` | str | (无) | 限定域：`text` / `image` / `video` / `audio` / `subtitle` / `consistency` / `digital-human` / `export` / `rag` |
| `--plugins`, `-p` | flag | False | 仅列出插件（不显示内置节点） |
| `--format` | str | `table` | 输出格式：`table` / `json` / `yaml` |
| `--search`, `-s` | str | (无) | 按名称模糊搜索 |

### 示例

#### 列出所有节点

```bash
mosaic list
```

输出（节选）：

```
┌──────────────────────────┬─────────┬────────────┬────────┐
│ Name                     │ Domain  │ Version    │ Builtin│
├──────────────────────────┼─────────┼────────────┼────────┤
│ text-generator           │ text    │ 0.1.0      │ ✓      │
│ chat                     │ text    │ 0.1.0      │ ✓      │
│ text-rewriter            │ text    │ 0.1.0      │ ✓      │
│ translator               │ text    │ 0.1.0      │ ✓      │
│ text-summarizer          │ text    │ 0.1.0      │ ✓      │
│ text-classifier          │ text    │ 0.1.0      │ ✓      │
│ text-to-image            │ image   │ 0.1.0      │ ✓      │
│ image-to-image           │ image   │ 0.1.0      │ ✓      │
│ ...                      │ ...     │ ...        │ ...    │
└──────────────────────────┴─────────┴────────────┴────────┘
共 42 个节点
```

#### 列出某个域的节点

```bash
mosaic list --domain video
```

输出：

```
┌──────────────────────┬───────┬──────────┬────────┐
│ Name                 │Domain │ Version  │ Builtin│
├──────────────────────┼───────┼──────────┼────────┤
│ text-to-video        │ video │ 0.1.0    │ ✓      │
│ wan-video            │ video │ 0.1.0    │ ✓      │
│ hunyuan-video        │ video │ 0.1.0    │ ✓      │
│ ltx-video            │ video │ 0.1.0    │ ✓      │
│ image-to-video       │ video │ 0.1.0    │ ✓      │
│ video-continuation   │ video │ 0.1.0    │ ✓      │
│ frame-interpolator   │ video │ 0.1.0    │ ✓      │
│ frame-extractor      │ video │ 0.1.0    │ ✓      │
└──────────────────────┴───────┴──────────┴────────┘
共 8 个节点
```

#### JSON 格式

```bash
mosaic list --format json
```

输出：

```json
[
  {
    "name": "text-to-image",
    "domain": "image",
    "version": "0.1.0",
    "description": "Generate image from text descriptions using SDXL.",
    "input_types": ["text", "mosaic"],
    "output_types": ["image"]
  },
  ...
]
```

#### 搜索节点

```bash
mosaic list --search tts
```

输出：

```
┌──────────────────────┬───────┬──────────┬────────┐
│ Name                 │Domain │ Version  │ Builtin│
├──────────────────────┼───────┼──────────┼────────┤
│ tts                  │ audio │ 0.1.0    │ ✓      │
│ voice-clone          │ audio │ 0.1.0    │ ✓      │
└──────────────────────┴───────┴──────────┴────────┘
```

---

## mosaic info

**作用**：显示节点的详细信息。

### 用法

```bash
mosaic info <NODE_NAME> [OPTIONS]
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `NODE_NAME` | ✅ | 节点名（positional） |

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--format` | str | `text` | 输出格式：`text` / `json` |
| `--show-params` | flag | True | 显示参数详情 |

### 示例

```bash
mosaic info text-to-image
```

输出：

```
节点: text-to-image
域:   image
版本: 0.1.0

描述: Generate image from text descriptions using SDXL.

输入类型:  text, mosaic
输出类型:  image

构造函数参数:
┌──────────────────────┬─────────┬────────────────────────┐
│ Name                 │ Default │ Type                    │
├──────────────────────┼─────────┼────────────────────────┤
│ model                │ SDXL    │ str                     │
│ num_inference_steps  │ 30      │ int                     │
│ guidance_scale       │ 7.5     │ float                   │
│ width                │ 1024    │ int                     │
│ height               │ 1024    │ int                     │
│ enable_cpu_offload   │ False   │ bool                    │
└──────────────────────┴─────────┴────────────────────────┘

run 输入字段:
┌──────────────────┬──────┬──────┐
│ Name             │ Type │ Req  │
├──────────────────┼──────┼──────┤
│ prompt           │ str  │ ✓    │
│ negative_prompt  │ str  │      │
│ seed             │ int  │      │
└──────────────────┴──────┴──────┘

run 输出字段:
┌───────┬───────┐
│ Name  │ Type  │
├───────┼───────┤
│ image │ Image │
└───────┴───────┘

模型信息:
  默认模型: stabilityai/stable-diffusion-xl-base-1.0
  显存需求: 8GB
  许可证:   OpenRAIL++

相关节点:
  - image-to-image
  - inpainting
  - upscaler
```

#### JSON 格式

```bash
mosaic info wan-video --format json
```

输出：

```json
{
  "name": "wan-video",
  "domain": "video",
  "version": "0.1.0",
  "description": "Generate video from text using Wan2.1/Wan2.2.",
  "input_types": ["text", "mosaic"],
  "output_types": ["video"],
  "constructor_params": {
    "model": {"type": "str", "default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
    "dtype": {"type": "str", "default": "float16"},
    "enable_cpu_offload": {"type": "bool", "default": true},
    "enable_vae_tiling": {"type": "bool", "default": true}
  },
  "model_info": {
    "default_model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "vram_gb": 30.0,
    "license": "Apache-2.0"
  }
}
```

---

## mosaic create-node

**作用**：使用模板创建新节点文件。

### 用法

```bash
mosaic create-node [OPTIONS]
```

### 选项

| 选项 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--domain`, `-d` | str | — | ✅ | 域：`text` / `image` / ... |
| `--name`, `-n` | str | — | ✅ | 节点名（kebab-case） |
| `--output`, `-o` | path | `mosaic/nodes/<domain>/<name>.py` | ❌ | 输出路径 |
| `--template`, `-t` | str | `default` | ❌ | 模板名：`default` / `ml` / `video` |
| `--no-test` | flag | False | ❌ | 不生成测试文件 |
| `--author` | str | (git config) | ❌ | 作者名 |

### 示例

#### 创建文本节点

```bash
mosaic create-node --domain text --name sentiment-analyzer
```

输出：

```
✅ Created: mosaic/nodes/text/sentiment_analyzer.py
✅ Created: tests/phase3/test_sentiment_analyzer.py
```

#### 创建视频节点（使用 video 模板）

```bash
mosaic create-node --domain video --name my-video --template video
```

输出：

```
✅ Created: mosaic/nodes/video/my_video.py
✅ Created: tests/phase4/test_my_video.py
```

#### 自定义输出路径

```bash
mosaic create-node --domain text --name my-node --output ./my_nodes/
```

#### 不生成测试

```bash
mosaic create-node --domain text --name my-node --no-test
```

---

## mosaic run

**作用**：运行 Python 管道脚本。

### 用法

```bash
mosaic run <PIPELINE_FILE> [OPTIONS]
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `PIPELINE_FILE` | ✅ | Python 脚本路径（包含 `if __name__ == "__main__"`） |

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--input`, `-i` | str | (无) | 输入数据 JSON 文件 |
| `--output`, `-o` | str | (无) | 输出结果保存路径 |
| `--async` | flag | False | 异步执行 |
| `--timeout` | int | 600 | 超时（秒） |
| `--verbose`, `-v` | flag | False | 详细输出 |

### 管道文件示例

`my_pipeline.py`：

```python
from mosaic import Pipeline
from mosaic.nodes.image import TextToImage

pipeline = Pipeline()
pipeline.add(TextToImage(model="SDXL"))

if __name__ == "__main__":
    result = pipeline.run(prompt="A sunset over mountains")
    result.get("image").save("output.png")
```

### 运行

```bash
mosaic run my_pipeline.py
```

输出：

```
[INFO] 加载 my_pipeline.py ...
[INFO] 运行管道 ...
[INFO] Pipeline 包含 1 步: text-to-image
[INFO] 步骤 1/1: text-to-image ...
[INFO] 完成，耗时 12.3s
[INFO] 输出已保存到 output.png
```

### 异步运行

```bash
mosaic run my_pipeline.py --async
```

输出：

```
[INFO] 任务已启动，ID: abc-123-def
[INFO] 等待完成（超时 600 秒）...
[INFO] 完成
```

### 输入数据

```json
// input.json
{
  "prompt": "A cat in a hat",
  "num_inference_steps": 20
}
```

```bash
mosaic run my_pipeline.py --input input.json
```

---

## mosaic version

**作用**：显示版本信息。

### 用法

```bash
mosaic version [OPTIONS]
```

### 选项

| 选项 | 类型 | 说明 |
|---|---|---|
| `--json` | flag | JSON 格式输出 |
| `--check-update` | flag | 检查是否有新版本 |

### 示例

```bash
mosaic version
```

输出：

```
mosaic 0.1.0
Python 3.10.12
Platform: linux-x86_64
PyTorch: 2.1.0+cu121
```

#### JSON 格式

```bash
mosaic version --json
```

输出：

```json
{
  "mosaic": "0.1.0",
  "python": "3.10.12",
  "platform": "linux-x86_64",
  "torch": "2.1.0+cu121",
  "cuda_available": true
}
```

#### 检查更新

```bash
mosaic version --check-update
```

输出：

```
当前版本: 0.1.0
最新版本: 0.1.5
可通过 pip install --upgrade mosaic 升级
```

---

## mosaic doctor

**作用**：诊断环境配置问题。

### 用法

```bash
mosaic doctor [OPTIONS]
```

### 选项

| 选项 | 类型 | 说明 |
|---|---|---|
| `--fix` | flag | 自动尝试修复（仅可修复的项） |
| `--json` | flag | JSON 格式输出 |
| `--check-deps` | flag | 仅检查依赖 |

### 示例

```bash
mosaic doctor
```

输出：

```
mosaic doctor
================

[系统]
[OK]    Python:        3.10.12
[OK]    OS:            Linux 5.15.0
[OK]    Platform:      linux-x86_64

[PyTorch]
[OK]    PyTorch:       2.1.0+cu121
[OK]    CUDA available: True
[OK]    GPU:           NVIDIA A100 80GB
[OK]    VRAM total:    81920 MB

[核心依赖]
[OK]    mosaic:        0.1.0
[OK]    diffusers:     0.32.0
[OK]    transformers:  4.45.0
[OK]    torch:         2.1.0
[OK]    pillow:        10.0.0
[OK]    numpy:         1.24.0
[OK]    sentencepiece: 0.2.0
[OK]    protobuf:      4.25.0

[可选依赖]
[WARN]  audio: 未安装（mosaic[audio]）
[WARN]  video: 未安装（mosaic[video]）
[WARN]  rag: 未安装（mosaic[rag]）
[WARN]  digital-human: 未安装（mosaic[digital-human]）

[TTS 后端]
[OK]    edge-tts:      6.1.9（云端，默认）
[WARN]  chattts:       未安装
[WARN]  fish-speech:   未安装
[WARN]  cosyvoice:     未安装
[WARN]  gpt-sovits:    未安装

[节点]
[OK]    42 个节点已注册
        - 6 文本域
        - 6 图像域
        - 8 视频域
        - 5 音频域
        - 3 字幕域
        - 3 一致性域
        - 4 数字人域
        - 3 导出域
        - 4 RAG 域

================
总结
[OK]    1 项
[WARN]  9 项（可选依赖）
[FAIL]  0 项

✅ 环境就绪，可以开始使用 Mosaic！
```

#### 自动修复

```bash
mosaic doctor --fix
```

会自动尝试安装缺失的可选依赖。

#### JSON 格式

```bash
mosaic doctor --json
```

输出（节选）：

```json
{
  "python": {"version": "3.10.12", "ok": true},
  "torch": {"version": "2.1.0+cu121", "ok": true, "cuda": true},
  "gpu": {"name": "NVIDIA A100 80GB", "vram_mb": 81920, "ok": true},
  "deps": {
    "mosaic": "0.1.0",
    "diffusers": "0.32.0",
    "transformers": "4.45.0"
  },
  "optional": {
    "audio": false,
    "video": false,
    "rag": false,
    "digital_human": false
  },
  "summary": {"ok": 1, "warn": 9, "fail": 0}
}
```

---

## 下一步

- [快速开始](getting-started.md) — 5 分钟上手
- [节点参考手册](nodes-reference.md) — 全部节点
- [插件开发指南](plugin-development.md) — 自定义节点
- [示例代码](../examples/) — 实际应用
