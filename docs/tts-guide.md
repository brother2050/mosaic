# Mosaic TTS 完整指南

> 4 个生产级 TTS 后端的对比、选型、用法与进阶技巧。

## 目录

- [TTS 子系统介绍](#tts-子系统介绍)
- [四个后端一览](#四个后端一览)
- [后端选择指南](#后端选择指南)
- [安装与配置](#安装与配置)
- [基础用法](#基础用法)
- [流式输出详解](#流式输出详解)
- [语音克隆详解](#语音克隆详解)
- [韵律控制详解](#韵律控制详解)
- [多语言合成](#多语言合成)
- [权重转换指南](#权重转换指南)
- [性能优化](#性能优化)
- [技术参数对比表](#技术参数对比表)

---

## TTS 子系统介绍

Mosaic 的 TTS（Text-to-Speech）是一个**路由器节点** + **多后端实现**的架构：

- **TTS 节点**（`mosaic.nodes.audio.TTS`）负责路由和生命周期管理
- **4 个后端**各自实现统一接口：ChatTTS / Fish Speech / GPT-SoVITS / CosyVoice
- 此外还有**edge_tts**云端默认后端（无需 GPU）

所有后端都采用统一的**四层架构**：

```
TextFrontend → AcousticModel → Vocoder → StreamAdapter
(文本前端)    (声学模型)     (声码器)    (流式适配)
```

这意味着：

- 切换后端只改 `backend="..."` 参数
- 自定义后端实现这 4 个接口即可接入
- 流式合成接口在所有后端上一致

---

## 四个后端一览

| 后端 | 声学模型 | 声码器 | 采样率 | 流式延迟 | 适用场景 | 许可证 |
|---|---|---|---|---|---|---|
| **ChatTTS** | LlamaForCausalLM | DVAE + Vocos | 24000Hz | ~50ms | 实时对话、韵律控制 | CC-BY-NC-4.0 |
| **Fish Speech** | LlamaForCausalLM | VQDec + HiFiGAN | 22050Hz | ~80ms | 多语言、跨语种 | Apache-2.0 |
| **GPT-SoVITS** | GPT2LMHeadModel | SoVITS (Flow+HiFiGAN) | 32000Hz | ~100ms | 极少样本克隆 | MIT |
| **CosyVoice** | FlowMatching | HiFiGAN | 24000Hz | ~300ms | 高质量、非自回归 | Apache-2.0 |
| **edge_tts** | 云端 Azure | 云端 | 24000Hz | 不可流式 | 默认无 GPU | — |

### 各后端推理管线

#### ChatTTS

```
文本 "你好世界"
  │
  ▼ ChatTokenizer
  ├─ 文本清洗、韵律标记注入
  ├─ Speaker Embedding（随机生成或参考音频）
  └─ token_ids
  │
  ▼ LlamaARModel (自回归)
  └─ 逐 token 生成 VQ 音频码
     ├─ 第 1 个 token 即可解码得到第 1 帧
     └─ 边生成边合成
  │
  ▼ DVAE (解码 VQ → mel)
  ▼ Vocos (mel → waveform)
  │
  ▼ 24000Hz 单声道波形
```

#### Fish Speech

```
文本 "你好世界"
  │
  ▼ FishTokenizer (BPE, 统一词表支持中英日韩)
  └─ token_ids
  │
  ▼ LlamaARModel (统一词表自回归)
  └─ VQ codes (来自参考音频的 codec tokens)
  │
  ▼ VQDecoder (codes → mel)
  ▼ HiFiGAN (mel → waveform)
  │
  ▼ 22050Hz 单声道波形
```

#### GPT-SoVITS

```
文本 "你好世界" + 参考音频 (1分钟)
  │
  ▼ SoVITSTokenizer
  ├─ 音素级 G2P 切分、韵律预测
  └─ phoneme_ids
  │
  ▼ GPT2ARModel (基于参考音频 SSL 特征自回归)
  └─ 语义 token 序列
  │
  ▼ SoVITSDecoder (SemanticEncoder + Flow + 条件HiFiGAN)
  │  ├─ SemanticEncoder: SSL 语义编码
  │  ├─ Normalizing Flow: 隐空间变换
  │  └─ ConditionalHiFiGAN: 隐空间→波形
  │
  ▼ 32000Hz 单声道波形
```

#### CosyVoice

```
文本 "你好世界" + 指令
  │
  ▼ CosyVoiceTokenizer
  ├─ SFT (Supervised Fine-Tuning) 指令处理
  └─ token_ids
  │
  ▼ FlowMatching (非自回归 ODE 求解)
  └─ mel 频谱
  │
  ▼ HiFiGAN (mel → waveform)
  │
  ▼ 24000Hz 单声道波形
```

---

## 后端选择指南

按使用场景选后端：

| 使用场景 | 推荐后端 | 原因 |
|---|---|---|
| 快速中文对话（低延迟） | **ChatTTS** | AR 流式延迟最低 (~50ms) |
| 中文高质量朗读 | **CosyVoice** | Flow Matching 质量最高 |
| 多语言（中英日韩） | **Fish** / **CosyVoice** | 多语言支持好 |
| 语音克隆（有参考音频） | **GPT-SoVITS** / **Fish** | 克隆效果好 |
| 实时流式（<100ms） | **ChatTTS** | 逐 token 延迟最低 |
| 批量合成（高吞吐） | **CosyVoice** | 非自回归可并行 |
| 开源商用 | **Fish** / **CosyVoice** / **SoVITS** | Apache-2.0 / MIT |
| 最低显存（<2GB） | **ChatTTS** | 模型最小 |
| 无 GPU / 快速试用 | **edge_tts** | 云端，无需权重 |

---

## 安装与配置

### 通用依赖

```bash
# 基础音频
pip install "mosaic[audio]"
```

### ChatTTS

```bash
# Mosaic 已包含代码，只需要安装基础依赖
pip install "mosaic[audio]" vocos

# 权重：首次运行自动从 Hugging Face 下载（约 1.2GB）到 ~/.cache/huggingface/hub/
# 也可手动指定模型目录：
#   export CHATTTS_MODEL_DIR=/path/to/chattts-200m
```

> 官方仓库：[2noise/ChatTTS](https://github.com/2noise/ChatTTS) — 代码已集成到 Mosaic，无需单独克隆。
>
> **子模型说明**：ChatTTS 使用 Vocos 声码器，已作为 pip 依赖安装（见上方 `pip install vocos`），无需单独下载模型权重。

### Fish Speech

```bash
# Mosaic 已包含代码，只需要安装基础依赖
pip install "mosaic[audio]"

# 权重：从 Hugging Face 下载
# from huggingface_hub import snapshot_download
# snapshot_download("fishaudio/fish-speech-1.5", local_dir="weights/fish-speech")
```

> 官方仓库：[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — Mosaic 复用了 Fish Speech 的模型架构代码，权重遵循 Apache-2.0 许可证。
>
> **子模型说明**：Fish Speech 使用 DAC codec，权重已包含在仓库内，无需单独下载。

### GPT-SoVITS

```bash
# Mosaic 已包含代码，需要额外安装 GPT-SoVITS 依赖
pip install "mosaic[audio]" chinese-text-splitter pypinyin

# 预训练权重：从 https://huggingface.co/lj1995/GPT-SoVITS 下载
# 将下载的文件放到你指定的 model_path 目录即可使用
# 自训练说话人模型参考 GPT-SoVITS 官方 README
# from huggingface_hub import snapshot_download
# snapshot_download("lj1995/GPT-SoVITS", local_dir="weights/gpt-sovits")
```

> 官方仓库：[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 代码架构已集成，遵循 MIT 许可证。
>
> **子模型说明**：GPT-SoVITS 还需要 `chinese-hubert-base` 作为 SSL 模型，首次运行自动下载。

### CosyVoice

```bash
# Mosaic 已包含代码，只需要安装基础依赖
pip install "mosaic[audio]"

# 权重：从 Hugging Face 下载
# from huggingface_hub import snapshot_download
# snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir="weights/cosyvoice")
```

> 官方仓库：[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 代码架构已集成，遵循 Apache-2.0 许可证。

> CosyVoice 内部使用 Qwen/Qwen2.5-1.5B-Instruct 做文本前端，首次运行自动下载。
> 说话人编码器 (CAM++) 权重已包含在 CosyVoice 仓库内，无需单独下载。
> 如需预下载 LLM：
> ```python
> from huggingface_hub import snapshot_download
> snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
> ```

### 离线 / 内网环境

通过 `HFModelManager` 统一管理权重下载与路径解析：

```python
from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager

# 下载到自定义路径（路径不存在时自动创建）
HFModelManager.ensure_model(
    model_path="/data/weights/chattts",
    repo_id="2Noise/ChatTTS",
)
HFModelManager.ensure_model(
    model_path="/data/weights/fish-speech",
    repo_id="fishaudio/fish-speech-1.5",
)
HFModelManager.ensure_model(
    model_path="/data/weights/gpt-sovits",
    repo_id="lj1995/GPT-SoVITS",
)
HFModelManager.ensure_model(
    model_path="/data/weights/cosyvoice2",
    repo_id="FunAudioLLM/CosyVoice2-0.5B",
)
# CosyVoice 的文本 LLM 单独下载
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
```

---

## 基础用法

### 1. ChatTTS 基础合成

```python
from mosaic.nodes.audio import TTS
from mosaic import MosaicData

tts = TTS(backend="chattts")
result = tts.run(MosaicData(text="你好，欢迎使用 Mosaic！", language="zh"))
audio = result.get("audio")  # AudioData 对象
print(f"采样率: {audio.sample_rate} Hz, 时长: {audio.duration:.2f}s")
# 采样率: 24000 Hz, 时长: 2.45s
```

### 2. Fish Speech 基础合成

```python
tts = TTS(backend="fish")
result = tts.run(MosaicData(text="Hello world, this is Fish Speech.", language="en"))
audio = result.get("audio")  # 采样率 22050 Hz
```

### 3. GPT-SoVITS 基础合成

```python
tts = TTS(backend="sovits")
result = tts.run(MosaicData(text="这是使用 GPT-SoVITS 合成的中文语音。", language="zh"))
audio = result.get("audio")  # 采样率 32000 Hz
```

### 4. CosyVoice 基础合成

```python
tts = TTS(backend="cosyvoice")
result = tts.run(MosaicData(text="CosyVoice 支持高质量中文合成。", language="zh"))
audio = result.get("audio")  # 采样率 24000 Hz
```

### 5. edge_tts（云端默认）

```python
tts = TTS(backend="edge_tts", voice="zh-CN-XiaoxiaoNeural")
result = tts.run(MosaicData(text="无需 GPU 即可使用。", language="zh"))
audio = result.get("audio")
```

---

## 流式输出详解

### ChatTTS 逐 token 流式

**原理**：Llama 自回归声学模型每生成一个 token，DVAE 就能解码出对应帧。因此理论上第一个 token 解码后即可听到声音，延迟约 50ms（首帧预热）。

```python
import time
from mosaic.nodes.audio import TTS
from mosaic import MosaicData

tts = TTS(backend="chattts")

print("开始流式合成...")
chunk_idx = 0
first_chunk_time = None
start = time.time()
for chunk in tts.run_stream(
    MosaicData(text="流式合成测试，第一批延迟应该很低。", language="zh")
):
    if first_chunk_time is None:
        first_chunk_time = time.time() - start
        print(f"首批延迟: {first_chunk_time * 1000:.0f}ms")
    chunk_idx += 1
    print(f"chunk #{chunk_idx}: {chunk.duration:.2f}s")
    # 实际播放：play(chunk)
```

### Fish Speech / GPT-SoVITS 流式

这两个后端与 ChatTTS 同样采用 AR 架构，流式接口一致：

```python
tts = TTS(backend="fish")  # 或 "sovits"
for chunk in tts.run_stream(MosaicData(text="...", language="zh")):
    play(chunk)
```

### CosyVoice 分块流式

**原理**：Flow Matching 是非自回归的，需要一次性 ODE 求解才能得到 mel 频谱。但为支持流式播放，CosyVoice 把 mel 频谱切成多个 chunk（默认 150 帧，约 1.74 秒），每生成一个 chunk 就送入声码器播放。

```python
tts = TTS(backend="cosyvoice")

for chunk in tts.run_stream(
    MosaicData(text="长文本，CosyVoice 分块流式输出", language="zh")
):
    # 首批延迟约 300ms（需要生成前 150 帧 mel）
    play(chunk)
```

### chunk_size 选择

CosyVoice 后端的 `chunk_size_frames` 构造参数控制分块大小（默认 150 帧，mel_fps=86.13）：

| 场景 | 推荐 chunk_size_frames | 理由 |
|---|---|---|
| 实时对话 | 30-50 | 最低延迟 |
| 语音助手 | 80-100 | 平衡延迟和质量 |
| 长篇朗读 | 150（默认） | 高吞吐 |
| 离线合成 | 完整 | 最高质量 |

### 流式延迟对比

```
时间轴 →
0ms     50ms     100ms    300ms    500ms    1s
├────────┼────────┼────────┼────────┼────────┤
ChatTTS: ▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         ↑ 首批 ~50ms
Fish:    ░░░░░░░░░▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░
                ↑ 首批 ~80ms
SoVITS:  ░░░░░░░░░░░▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░
                  ↑ 首批 ~100ms
CosyV:   ░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓
                            ↑ 首批 ~300ms
```

---

## 语音克隆详解

### ChatTTS：seed 随机音色

ChatTTS 不支持真正的 zero-shot 克隆，但可以通过 `speaker` 参数生成**稳定**的随机音色：

```python
tts = TTS(backend="chattts")

# 通过 seed 生成稳定的随机音色
backend = tts._tts_backend  # 获取后端实例
backend.set_seed(222)
spk = backend.sample_random_speaker()
result = tts.run(MosaicData(text="第一句话", language="zh", speaker=spk))
audio1 = result.get("audio")

# 复用同一个 spk 可保证音色一致
result = tts.run(MosaicData(text="第二句话", language="zh", speaker=spk))
audio2 = result.get("audio")

# 试听不同 seed 找喜欢的声音
for seed in [2, 222, 786, 2024]:
    backend.set_seed(seed)
    spk = backend.sample_random_speaker()
    result = tts.run(MosaicData(text=f"这是{seed}号声音", language="zh", speaker=spk))
    audio = result.get("audio")
```

常用 speaker 种子：`2`, `222`, `786`, `2024`, `6653`, `7114`

### Fish Speech：参考音频 codec tokens

Fish 通过参考音频提取 codec tokens 实现声音克隆。需要先准备 10-30 秒清晰的参考音频。

```python
tts = TTS(backend="fish")

# 在后端加载时设置参考音频（通过 backend 参数传递）
result = tts.run(MosaicData(
    text="新的文本内容",
    language="zh",
    speaker="reference_audio_path",  # 参考音频路径
))
audio = result.get("audio")
```

**参考音频要求**：

- 时长：10-30 秒最佳
- 内容：清晰单人朗读，无背景音乐
- 采样率：≥ 22050Hz
- 噪声：尽量低

### GPT-SoVITS：SSL 特征 + 极少样本

GPT-SoVITS 只需 **5-10 秒**参考音频即可克隆，是少样本克隆的代表：

```python
tts = TTS(backend="sovits")

# 通过 speaker 参数传递参考音频
result = tts.run(MosaicData(
    text="这是用 5 秒参考音频克隆的声音",
    language="zh",
    speaker="short_ref.wav",  # 参考音频路径
))
audio = result.get("audio")
```

**进阶用法**：训练自己的说话人模型：

```bash
# 1. 准备 1 分钟参考音频和文字转写
# 2. 运行训练
python GPT-SoVITS/svtrain.py -g reference_audio/ -sv reference_text/
# 3. 在 Mosaic 中使用
```

```python
tts = TTS(backend="sovits")
```

### CosyVoice：speech tokens + speaker embedding

CosyVoice 通过 SFT（Supervised Fine-Tuning）指令实现跨语言克隆：

```python
tts = TTS(backend="cosyvoice")

# 通过 speaker 参数传递参考音频
result = tts.run(MosaicData(
    text="Cross-lingual voice cloning in English",
    language="en",
    speaker="ref.wav",  # 参考音频路径
))
audio = result.get("audio")
```

**预训练说话人**：CosyVoice 内置多个预训练说话人，可直接使用：

```python
audio = tts.run(MosaicData(
    text="使用预训练说话人",
    language="zh",
    speaker="中文女",  # 或 "英文男", "粤语女" 等
))
```

---

## 韵律控制详解

### ChatTTS 的韵律标记

ChatTTS 支持特殊标记控制韵律：

```python
tts = TTS(backend="chattts")

# oral - 口语化连接词
result = tts.run(MosaicData(text="那个[oral_嗯]东西[oral_啊]特别好", language="zh"))
audio = result.get("audio")

# laugh - 笑声
result = tts.run(MosaicData(text="这个笑话太好笑了[laugh]", language="zh"))
audio = result.get("audio")

# break - 停顿
result = tts.run(MosaicData(text="第一句[break]第二句[break_500]第三句", language="zh"))
audio = result.get("audio")

# speed - 语速控制（按段）
result = tts.run(MosaicData(text="[speed_0.8]慢一点[speed_1.2]快一点", language="zh"))
audio = result.get("audio")
```

**支持的韵律标记**

| 标记 | 作用 | 示例 |
|---|---|---|
| `[oral_xxx]` | 口语化填充 | `[oral_嗯]`, `[oral_啊]` |
| `[laugh]` | 笑声 | `[laugh]`, `[laugh_2]` |
| `[break]` | 短停顿 (~200ms) | `[break]` |
| `[break_500]` | 长停顿 (ms) | `[break_500]` |
| `[speed_x]` | 局部语速 | `[speed_0.8]` 慢, `[speed_1.5]` 快 |

### CosyVoice 的指令控制

TTS 节点当前**不支持**对 CosyVoice 传入 `emotion` / `instruct` 参数，相关字段会被忽略。CosyVoice 的 SFT 情感指令需通过后端 API 直接调用：

```python
from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import CosyVoiceBackend

backend = CosyVoiceBackend(model_path="/data/cosyvoice")
backend.load(device="cuda", dtype="float16")
# 通过后端 API 直接传递 SFT 情感指令
audio = backend.synthesize("今天天气真好", language="zh", instruct="高兴地")
```

支持的情感指令：`高兴地` / `悲伤地` / `愤怒地` / `惊讶地` / `平静地` / `兴奋地`

### 数字读法

```python
# ChatTTS 自动处理
result = tts.run(MosaicData(text="我的电话号码是 138-0013-8000", language="zh"))
audio = result.get("audio")

# CosyVoice 需要在 SFT 指令中显式说明
audio = tts.run(MosaicData(
    text="我的电话号码是一三八零零一三八零零零",
    language="zh",
))
```

---

## 多语言合成

```python
# ChatTTS
tts = TTS(backend="chattts")
result = tts.run(MosaicData(text="你好世界", language="zh"))
result = tts.run(MosaicData(text="Hello world", language="en"))
result = tts.run(MosaicData(text="こんにちは", language="ja"))

# Fish Speech（多语言最佳）
tts = TTS(backend="fish")
for lang, text in [("zh", "你好"), ("en", "Hello"), ("ja", "こんにちは"), ("ko", "안녕하세요")]:
    result = tts.run(MosaicData(text=text, language=lang))
    audio = result.get("audio")

# CosyVoice
tts = TTS(backend="cosyvoice")
result = tts.run(MosaicData(text="Hello world in English", language="en"))
```

**多语言能力对比**

| 后端 | 中文 | 英文 | 日文 | 韩文 | 粤语 | 德语 | 法语 | 跨语种 |
|---|---|---|---|---|---|---|---|---|
| ChatTTS | 优秀 | 良好 | 一般 | 一般 | 不支持 | 不支持 | 不支持 | 不支持 |
| Fish Speech | 优秀 | 优秀 | 优秀 | 优秀 | 不支持 | 不支持 | 不支持 | 优秀 |
| GPT-SoVITS | 优秀 | 良好 | 一般 | 一般 | 良好 | 不支持 | 不支持 | 训练支持 |
| CosyVoice | 优秀 | 优秀 | 良好 | 良好 | 优秀 | 良好 | 良好 | 优秀 |

---

## 权重转换指南

将原版模型权重转换为 Mosaic 内部格式：

```python
from mosaic.nodes.audio.tts_backends.weights import (
    chattts_convert,
    fish_convert,
    sovits_convert,
    cosyvoice_convert,
)

# ChatTTS 原版 → Mosaic 格式
chattts_convert.convert(
    src_dir="original_chatts",
    dst_dir="weights/chatts",
)

# Fish Speech 原版 → Mosaic 格式
fish_convert.convert(
    src_dir="original_fish_speech",
    dst_dir="weights/fish",
)

# GPT-SoVITS 原版 → Mosaic 格式
sovits_convert.convert(
    gpt_weights="GPT-SoVITS/GPT_weights/",
    sovits_weights="GPT-SoVITS/SoVITS_weights/",
    dst_dir="weights/sovits",
)

# CosyVoice 原版 → Mosaic 格式
cosyvoice_convert.convert(
    src_dir="original_cosyvoice",
    dst_dir="weights/cosyvoice",
)
```

转换器会自动处理：

- 张量重命名（`transformer.h.0` → `model.layers.0`）
- 精度转换（fp32 → bf16）
- 缺失张量填充
- 配置文件生成

---

## 性能优化

### GPU 推理

确保 PyTorch 检测到 GPU：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### 混合精度

```python
# TTS 节点不直接暴露 dtype 参数，在后端加载时设置
# 通过 CosyVoiceBackend 构造函数或在 TTS 节点 kwargs 中传递
tts = TTS(backend="cosyvoice")
# 后端加载时使用 float16/bfloat16 减半显存
```

### 批处理

同一文本批量合成可提高吞吐：

```python
# 通过管道批量调用 TTS 节点
texts = ["第一句", "第二句", "第三句"]
for text in texts:
    result = tts.run(MosaicData(text=text, language="zh"))
    audio = result.get("audio")
```

### 未来优化（规划中）

- **ONNX 加速**：声码器（HiFiGAN、Vocos）导出为 ONNX，CPU 推理加速 2-3 倍
- **TensorRT 优化**：声学模型 INT8 量化，GPU 推理加速 1.5-2 倍
- **Speculative Decoding**：AR 后端并行生成多 token

---

## 技术参数对比表

| 参数 | ChatTTS | Fish Speech | GPT-SoVITS | CosyVoice |
|---|---|---|---|---|
| 声学模型类型 | AR (Llama) | AR (Llama) | AR (GPT2) | Flow Matching |
| 声码器 | DVAE + Vocos | VQDec + HiFiGAN | Flow + HiFiGAN | HiFiGAN |
| 模型大小 | ~200M | ~1B | ~300M | ~0.5B |
| 显存 (推理) | ~2GB | ~3GB | ~4GB | ~4GB |
| 显存 (训练) | ~16GB | ~24GB | ~16GB | ~16GB |
| 采样率 | 24000Hz | 22050Hz | 32000Hz | 24000Hz |
| 比特深度 | 16-bit | 16-bit | 16-bit | 16-bit |
| 流式延迟 (首帧) | ~50ms | ~80ms | ~100ms | ~300ms |
| 流式延迟 (稳定) | ~20ms/token | ~30ms/token | ~30ms/token | 一次性 |
| 长文本处理 | 优 | 优 | 中 | 优 |
| 多语言 | 中 | 优 | 中 | 优 |
| 零样本克隆 | 不支持 (seed) | 支持 (10-30s) | 支持 (5-10s) | 支持 (3-10s) |
| 韵律控制 | 丰富 (标记) | 一般 | 中等 | 指令式 |
| 商用许可 | CC-BY-NC-4.0 | Apache-2.0 | MIT | Apache-2.0 |

---

## 下一步

- [节点参考手册](nodes-reference.md) — TTS 节点的完整 API
- [示例代码](../examples/05_tts_chattts.py) — ChatTTS 完整示例
- [示例代码](../examples/06_tts_fish_speech.py) — Fish Speech 完整示例
- [示例代码](../examples/07_tts_gpt_sovits.py) — GPT-SoVITS 完整示例
- [示例代码](../examples/08_tts_cosyvoice.py) — CosyVoice 完整示例
