# TTS 后端选择指南

Mosaic TTS 子系统提供四个本地后端和一个云端回退后端，覆盖从低延迟对话到高质量朗读的各类场景。本指南帮助开发者根据使用场景、语言需求、许可证约束和硬件条件选择最合适的后端。

## 场景推荐表

| 使用场景 | 推荐后端 | 原因 |
|---|---|---|
| 快速中文对话（低延迟） | ChatTTS | AR 流式延迟最低（~50ms/chunk），韵律控制丰富 |
| 中文高质量朗读 | CosyVoice | Flow Matching 质量高，ODE 只需 10 步即可获得优秀效果 |
| 多语言（中英日韩） | Fish Speech / CosyVoice | 多语言支持好，统一词表 BPE 或 LLM tokenizer |
| 语音克隆（有参考音频） | GPT-SoVITS / Fish | 极少样本克隆效果好，SSL 语义 token 或 codec token |
| 实时流式（<100ms 延迟） | ChatTTS | AR 逐 token 生成，首字延迟最低 |
| 批量合成（高吞吐） | CosyVoice | 非自回归，可并行处理，无逐 token 串行瓶颈 |
| 开源商用（许可证友好） | Fish / CosyVoice / SoVITS | Apache-2.0 / MIT，可商用 |
| 最低显存需求（<2GB） | ChatTTS | 模型最小，~2GB 即可运行 |
| 无 GPU 环境 | edge-tts | 云端服务，无需本地推理 |

## 各后端技术参数对比表

| 参数 | ChatTTS | Fish | GPT-SoVITS | CosyVoice |
|---|---|---|---|---|
| 声学模型 | LLaMA | LLaMA | GPT-2 | Flow Matching |
| 生成方式 | 自回归 | 自回归 | 自回归 | ODE 求解 |
| 采样率 | 24000 | 22050 | 32000 | 22050 |
| 流式延迟 | ~50ms | ~80ms | ~100ms | ~300ms |
| 显存需求 | ~2GB | ~3GB | ~4GB | ~4GB |
| 语音克隆 | seed 随机 | ref tokens | ref + SSL | ref + speech tokens |
| 许可证 | CC BY-NC | Apache-2.0 | MIT | Apache-2.0 |
| 中文质量 | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| 英文质量 | ★★★ | ★★★★ | ★★★ | ★★★★ |
| 韵律控制 | ★★★★★ | ★★★ | ★★★ | ★★★ |
| 支持语言 | zh, en | zh, en, ja, ko | zh, en, ja, ko, yue | zh, en, ja, ko, yue, de, fr |

## 后端架构总览

```
TTS Node (路由器)
│
├── backend="chattts"  → ChatTTSBackend (24000Hz, AR, 流式延迟最低)
│   ├── ChatTokenizer        [BPE + 韵律标记]
│   ├── LlamaARModel         [LlamaForCausalLM + 双路Embed + spk_emb]
│   ├── DVAEDecoder          [离散码→mel, ConvNeXt]
│   └── VocosVocoder         [mel→waveform]
│
├── backend="fish"     → FishSpeechBackend (22050Hz, AR, 多语言)
│   ├── FishTokenizer        [统一词表 BPE]
│   ├── FishLlamaARModel     [LlamaForCausalLM + 统一Embed + ref tokens]
│   ├── VQDecoder            [VQ tokens→mel, 残差卷积]
│   └── HiFiGanVocoder       [mel→waveform]
│
├── backend="sovits"   → GPTSoVITSBackend (32000Hz, AR, 极少样本克隆)
│   ├── SoVITSTokenizer      [音素级 G2P]
│   ├── GPT2ARModel          [GPT2LMHeadModel + ref tokens + spk_emb]
│   └── SoVITSDecoder        [SemanticEnc + Flow + 条件化HiFiGAN]
│
└── backend="cosyvoice"→ CosyVoiceBackend (22050Hz, FlowMatching, 高质量)
    ├── CosyVoiceTokenizer   [LLM tokenizer + LLM 编码]
    ├── FlowMatchingModel    [FlowEstimator Transformer + ODE Solver]
    ├── SpeechTokenizer      [参考音频→语音tokens]
    ├── SpeakerEncoder       [参考音频→speaker embedding]
    └── HiFiGanVocoder       [mel→waveform, 与Fish共享代码]
```

## 共享基础设施

| 组件 | 共享者 | 说明 |
|---|---|---|
| LlamaARModelBase | ChatTTS + Fish | 共享 LLaMA 自回归框架 |
| HiFiGanVocoder | Fish + CosyVoice | 共享声码器代码，权重独立 |
| StreamAdapter | 所有后端 | 共享流式缓冲与 chunk 切分 |
| TTSBackendRegistry | 所有后端 | 统一注册和路由 |
| WeightConverter | 所有后端 | 统一权重转换框架 |
| Scheduler | 所有后端 | 统一显存管理 |

## 自动选择逻辑

`TTSBackendRegistry.auto_select(requirements)` 根据需求字典自动选择最优后端：

```python
from mosaic.nodes.audio.tts_backends.registry import tts_backend_registry

# 高质量合成
backend = tts_backend_registry.auto_select({"language": "zh", "quality": True})
# → "cosyvoice"

# 低延迟对话
backend = tts_backend_registry.auto_select({"language": "zh", "low_latency": True})
# → "chattts"

# 语音克隆 + 开源许可证
backend = tts_backend_registry.auto_select({"voice_clone": True, "open_license": True})
# → "sovits" 或 "fish"

# 批量合成
backend = tts_backend_registry.auto_select({"batch": True})
# → "cosyvoice"
```

### 选择优先级

1. **语言过滤**：后端必须支持目标语言
2. **功能过滤**：流式、语音克隆等功能必须满足
3. **显存过滤**：后端最低显存需求不得超过可用显存
4. **场景评分**：根据 `quality`、`low_latency`、`batch`、`open_license` 等标志为每个后端打分
5. **回退**：无候选时返回 `"edge_tts"`

## CosyVoice ODE 步数调优

CosyVoice 的核心优势是 ODE 步数可调，在质量和速度之间灵活权衡：

| num_ode_steps | 延迟 | 质量 | 适用场景 |
|---|---|---|---|
| 5 | ~50ms | 中等 | 实时对话、低延迟场景 |
| 10 | ~100ms | 好 | 推荐默认值，质量/速度均衡 |
| 20 | ~200ms | 最高 | 离线合成、高质量要求 |
| 50 | ~500ms | 极高 | 研究对比、极限质量 |

```python
backend = CosyVoiceBackend(model_path="/data/cosyvoice", num_ode_steps=10)
backend.load(device="cuda", dtype="float16")

# 运行时切换到高质量模式
backend.set_ode_params(num_steps=20, solver="midpoint")

# 基准测试不同步数
results = backend.benchmark_ode_steps("你好世界", steps_list=[5, 10, 20, 50])
# → {5: {"time": 0.05, "mel_std": 0.82}, 10: {"time": 0.10, "mel_std": 0.85}, ...}
```

## 快速开始

### 安装依赖

```bash
pip install torch transformers safetensors
# 可选：特定后端的额外依赖
pip install vocos  # ChatTTS 的 Vocos 声码器
```

### 使用 ChatTTS（低延迟对话）

```python
from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import ChatTTSBackend

backend = ChatTTSBackend(model_path="/data/chattts")
backend.load(device="cuda", dtype="float16")
audio = backend.synthesize("你好，世界", speaker="seed_42", language="zh")
# sample_rate = 24000
```

### 使用 CosyVoice（高质量朗读）

```python
from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import CosyVoiceBackend

backend = CosyVoiceBackend(
    model_path="/data/cosyvoice",
    num_ode_steps=10,
    ode_solver="euler",
)
backend.load(device="cuda", dtype="float16")
audio = backend.synthesize("你好，世界", language="zh")
# sample_rate = 22050

# 流式合成
for chunk in backend.synthesize_stream("这是一段流式合成的文本。", language="zh"):
    play(chunk)
```

### 使用 GPT-SoVITS（语音克隆）

```python
from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import GPTSoVITSBackend

backend = GPTSoVITSBackend(model_path="/data/gpt_sovits")
backend.load(device="cuda", dtype="float16")
audio = backend.clone_voice(
    audio="/data/reference.wav",
    text="用参考音色说这句话",
    language="zh",
)
# sample_rate = 32000
```

### 使用 Fish Speech（多语言）

```python
from mosaic.nodes.audio.tts_backends.implementations.fish_backend import FishSpeechBackend

backend = FishSpeechBackend(model_path="/data/fish_speech")
backend.load(device="cuda", dtype="float16")
audio = backend.synthesize("Hello, 世界", language="en")
# sample_rate = 22050
```

### 通过 TTS 节点统一调用

```python
from mosaic.nodes.audio.tts import TTS

# 显式指定后端
tts = TTS(backend="cosyvoice", model="/data/cosyvoice")
audio = tts.run({"text": "你好，世界", "language": "zh"})

# 自动选择
tts = TTS(backend="auto", model="auto")
audio = tts.run({"text": "你好", "language": "zh", "quality": True})
```

## 注意事项

1. **采样率差异**：四个后端的采样率不同（24000/22050/32000/22050Hz），混用时需注意重采样。
2. **显存管理**：同时加载多个后端时，使用 Scheduler 统一管理 GPU 显存，支持 LRU 淘汰。
3. **许可证**：ChatTTS 为 CC BY-NC 4.0（不可商用），其余三个后端均可商用。
4. **依赖惰性加载**：所有后端的 torch/transformers/safetensors 均为惰性导入，模块可在无这些依赖时正常 import。
5. **流式策略差异**：AR 后端（ChatTTS/Fish/SoVITS）为逐 token 流式，CosyVoice 为 Chunk-aware ODE 求解流式。
