# Mosaic 架构设计

> 理解 Mosaic 框架的内部模块划分、数据流、调度策略与扩展机制。

## 目录

- [整体架构](#整体架构)
- [核心模块](#核心模块)
- [TTS 扩展架构](#tts-扩展架构)
- [数据流向](#数据流向)
- [显存调度策略](#显存调度策略)
- [流式输出架构](#流式输出架构)
- [设计决策](#设计决策)

---

## 整体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                          Mosaic Framework                          │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    User-Facing API                         │   │
│  │                                                            │   │
│  │  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐    │   │
│  │  │ Pipeline       │  │ AsyncTask /  │  │ CLI          │    │   │
│  │  │ Orchestrator   │  │ TaskManager  │  │ (list/info)  │    │   │
│  │  └────────────────┘  └──────────────┘  └──────────────┘    │   │
│  └────────────────────────────────────────────────────────────┘   │
│       │              │                │                            │
│       ▼              ▼                ▼                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                       Core Framework                       │   │
│  │                                                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐    │   │
│  │  │ Node    │  │Registry │  │Scheduler│  │  EventBus   │    │   │
│  │  │(节点)   │  │(注册表) │  │(调度器) │  │  (事件总线) │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │   │
│  │  │ Context │  │ Result  │  │ Plugin  │                      │   │
│  │  │(上下文) │  │(结果)   │  │(插件)   │                      │   │
│  │  └─────────┘  └─────────┘  └─────────┘                      │   │
│  └────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                     Domain Nodes (9 域)                    │   │
│  │                                                            │   │
│  │   text ─ image ─ video ─ audio ─ subtitle                  │   │
│  │                            │                               │   │
│  │              consistency ──┼── digital_human                │   │
│  │                            │                               │   │
│  │                      export ─── rag                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  TTS Backend Layer (4 后端)                 │   │
│  │                                                            │   │
│  │   ChatTTS ─ Fish Speech ─ GPT-SoVITS ─ CosyVoice           │   │
│  │   ┌─────────────────────────────────────────────────────┐  │   │
│  │   │ TextFrontend → AcousticModel → Vocoder → Stream     │  │   │
│  │   └─────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   Engine Backends                           │   │
│  │   PyTorch (default) | ONNX Runtime | TensorRT (reserved)   │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

四层职责：

1. **User-Facing API** — `Pipeline` 编排器、`AsyncTask` 异步句柄、CLI 工具
2. **Core Framework** — `Node` / `Registry` / `Scheduler` / `EventBus` 等基础机制
3. **Domain Nodes** — 9 域 42 节点的具体实现
4. **TTS Backend** — 4 个生产级 TTS 后端 + 统一四层架构

---

## 核心模块

### Node（节点基类）

**位置**：`mosaic/core/node.py`

所有节点继承自 `Node` 基类。`Node` 是一个抽象类，定义了节点的最小契约：

```python
from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import registry

@registry.register
class MyNode(Node):
    name: str = "my-node"           # 全局唯一名
    domain: str = "custom"          # 所属域
    description: str = "..."        # 描述
    version: str = "0.1.0"          # 版本
    input_types: list[str] = [...]  # 输入数据类型
    output_types: list[str] = [...] # 输出数据类型

    def load(self) -> None: ...     # 加载模型（可选重写）
    def unload(self) -> None: ...   # 卸载模型（可选重写）
    def run(self, input_data) -> MosaicData: ...  # 必须实现
    def describe(self) -> NodeSpec: ...           # 节点规格
```

**关键点**：

- `load()` / `unload()` 由 `Scheduler` 自动管理
- `run()` 接收 `MosaicData`，返回 `MosaicData`
- `@registry.register` 装饰器在模块导入时自动注册

### Pipeline（编排引擎）

**位置**：`mosaic/core/pipeline.py`

`Pipeline` 本身继承 `Node`，因此可嵌套。它把用户给定的元素列表编译为 DAG（Directed Acyclic Graph）并按拓扑序执行。

```python
from mosaic import Pipeline, Branch, Merge
from mosaic.nodes.text import Chat
from mosaic.nodes.image import TextToImage

# 串行
pipe = Pipeline()
pipe.add(Chat(model="Qwen2.5-7B"))
pipe.add(TextToImage(model="SDXL"))

# | 运算符
pipe2 = Chat() | TextToImage()

# 并行分支
pipe.add(Branch([
    TextToImage(model="SDXL"),
    TTS(backend="chattts"),
]))
pipe.add(Merge())
```

**核心特性**：

- **DAG 合法性检查**：环检测、连通性、死端节点
- **并行执行**：`concurrent.futures.ThreadPoolExecutor`
- **dry_run 模式**：只校验类型不实际执行
- **条件路由**：基于数据动态选择唯一路径

### Registry（注册表）

**位置**：`mosaic/core/registry.py`

全局单例节点注册表。三种注册机制：

```python
# 1. 装饰器（推荐）
@registry.register
class MyNode(Node): ...

# 2. 显式注册
registry.register(MyNode, name="alias", domain="custom")

# 3. 目录扫描（启动时自动）
# mosaic/nodes/ 下的所有节点在 import 时自动注册
```

注册表的核心操作：

```python
registry.list()                  # 全部节点
registry.list(domain="text")     # 某域
registry.get("text-generator")   # 按名查询
registry.unregister("name")      # 注销
```

### Scheduler（显存调度器）

**位置**：`mosaic/core/scheduler.py`

`Scheduler` 是全局单例，职责：

- **跟踪**：每个节点 `load()` 时注册到 `Scheduler`
- **显存估算**：通过 `Node.model_info["vram_gb"]` 估算
- **按需加载**：`ensure_loaded(node)` 检查并触发加载
- **LRU 淘汰**：显存不足时卸载最近未使用的节点

```python
from mosaic.core.scheduler import get_scheduler

sched = get_scheduler()
# 节点 load() 时自动 sched.track(self)
# 节点 run() 时自动 sched.ensure_loaded(self)
```

**显存不足时的回退**：

1. 卸载最近未使用的节点
2. 若仍不足，尝试启用 `enable_sequential_cpu_offload`
3. 仍不足则抛 `RuntimeError` 并附建议

### EventBus（事件总线）

**位置**：`mosaic/core/events.py`

观察者模式的全局事件通道。节点执行期间发出：

| 事件类型 | 触发时机 | 关键字段 |
|---|---|---|
| `NODE_START` | 节点 `run()` 开始 | `node`, `timestamp` |
| `NODE_COMPLETE` | 节点 `run()` 成功完成 | `node`, `duration`, `output_summary` |
| `NODE_ERROR` | 节点 `run()` 抛出异常 | `node`, `error` |
| `PROGRESS` | 长任务的进度更新 | `node`, `current`, `total`, `message` |
| `INTERMEDIATE` | 中间产物 | `node`, `key`, `value` |

```python
from mosaic.core.events import get_event_bus, EventType

bus = get_event_bus()

@bus.on(EventType.PROGRESS)
def on_progress(event):
    print(f"[{event.node}] {event.current}/{event.total}: {event.message}")
```

### AsyncTask（异步任务）

**位置**：`mosaic/core/async_pipeline.py`, `mosaic/core/task_manager.py`

将 `Pipeline` 转为后台任务。`run_async` 返回 `AsyncTask` 句柄：

```python
task = pipeline.run_async(prompt="hello")
# ... 做其他事情
result = task.result(timeout=60)  # 阻塞等待
```

`TaskManager` 管理多个并发任务：

```python
from mosaic.core.task_manager import TaskManager

tm = TaskManager()
task1 = tm.submit(pipe1, prompt="...")
task2 = tm.submit(pipe2, prompt="...")
tm.wait_all(timeout=300)
```

### PluginManager（插件管理器）

**位置**：`mosaic/core/plugin.py`

三种插件加载机制：

```python
# 1. entry_points（PyPI 插件）
# 在第三方包 pyproject.toml:
# [project.entry-points."mosaic.nodes"]
# my-node = "my_pkg.nodes:MyNode"

# 2. 装饰器（应用代码内）
@registry.register
class MyNode(Node): ...

# 3. 目录扫描（运行时）
plugin_manager.discover_directory("/path/to/plugins")
```

---

## TTS 扩展架构

### 四层架构总览

```
┌────────────────────────────────────────────────────────────────┐
│                      TTS 节点（路由器）                         │
│  TTS(backend="chattts")  TTS(backend="fish")  TTS(backend=...) │
└────────────────────────────┬───────────────────────────────────┘
                             │ 委托给具体后端
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                       TTSBackend (基类)                        │
│  synthesize() / synthesize_stream() / clone()                  │
└────────────┬───────────────┬───────────────┬───────────────────┘
             │               │               │
   ┌─────────┴─────┐  ┌──────┴─────┐  ┌──────┴─────┐  ┌─────────┴──────┐
   │  Layer 1      │  │  Layer 2   │  │  Layer 3   │  │  Layer 4       │
   │  TextFrontend │─▶│  Acoustic  │─▶│  Vocoder   │─▶│  StreamAdapter │
   │  (文本前端)   │  │  (声学模型) │  │  (声码器)   │  │  (流式适配)    │
   └───────────────┘  └────────────┘  └────────────┘  └────────────────┘
```

### 四个后端的具体管线

#### ChatTTS

```
文本 → ChatTokenizer → LlamaARModel → DVAE → Vocos → StreamAdapter
       (清洗+韵律)    (自回归生成)   (VQ→mel) (mel→wav)
       
采样率: 24000Hz    流式延迟: ~50ms
```

#### Fish Speech

```
文本 → FishTokenizer → LlamaARModel → VQDecoder → HiFiGAN → StreamAdapter
       (BPE+多语言)   (统一词表)    (code→mel)   (mel→wav)
       
采样率: 22050Hz    流式延迟: ~80ms
```

#### GPT-SoVITS

```
文本 → SoVITSTokenizer → GPT2ARModel → SoVITSDecoder → StreamAdapter
       (音素G2P)      (语义token)  (SemanticEnc+Flow+HiFiGAN)
       
采样率: 32000Hz    流式延迟: ~100ms
```

#### CosyVoice

```
文本 → CosyVoiceTokenizer → FlowMatching → HiFiGAN → StreamAdapter
       (SFT+指令)         (非自回归)     (mel→wav)
       
采样率: 24000Hz    流式延迟: ~300ms
```

### 四层组件的统一接口

```python
class TextFrontend(Protocol):
    def tokenize(self, text: str, language: str = "zh", **kwargs) -> Any: ...

class AcousticModel(Protocol):
    def generate(self, token_ids, **kwargs) -> Any: ...
    def generate_stream(self, token_ids, **kwargs) -> Iterator[Any]: ...

class Vocoder(Protocol):
    sample_rate: int
    def decode(self, features) -> np.ndarray: ...
    def decode_chunk(self, features) -> np.ndarray: ...

class StreamAdapter(Protocol):
    def create_stream(self) -> StreamSession: ...
    # StreamSession.push() / pop() / flush()
```

子类实现这些接口后，`TTSBackend` 基类自动提供 `synthesize` / `synthesize_stream` 等统一方法。

---

## 数据流向

### 完整路径（以 TTS 为例）

```
用户输入
   │
   │ text="你好"
   ▼
Pipeline.run(MosaicData(text="你好"))
   │
   │ MosaicData(text="你好")
   ▼
TTS.run(MosaicData)
   │
   │ 路由到具体后端
   ▼
ChatTTSBackend.synthesize(text, language="zh")
   │
   ├─▶ Layer 1: ChatTokenizer.tokenize("你好", language="zh")
   │     → token_ids
   │
   ├─▶ Layer 2: LlamaARModel.generate(token_ids)
   │     → vq_tokens
   │
   ├─▶ Layer 3: _CompositeVocoder.decode(vq_tokens)
   │     → waveform (numpy array, 24000Hz)
   │
   └─▶ Layer 4: StreamAdapter.flush()
         → AudioData(waveform, sample_rate=24000)
   │
   ▼
MosaicData(audio=AudioData(...), text="你好", duration=2.3)
   │
   ▼
PipelineResult (返回给用户)
```

### 跨域数据流

```
text " 写一个关于猫的故事 "
  │
  ▼ [TextGenerator]
  │
  ├─ "从前有一只猫..." (text)
  │
  ▼ [TextSummarizer]
  │
  ├─ "关于猫的冒险故事" (text)
  │
  ▼ [Translator]（中文→英文）
  │
  ├─ "An adventure story of a cat" (text)
  │
  ▼ [TextToImage]（并行执行 [TTS]）
  │                       │
  │  ▼ image              │  ▼ audio
  │  cat.png              │  cat.wav
  ▼                       ▼
```

---

## 显存调度策略

### Scheduler 的工作流程

```
节点 run() 被调用
  │
  ▼
Scheduler.ensure_loaded(node)
  │
  ├─ 检查 node.is_loaded()
  │     ├─ True: 直接返回
  │     └─ False: 继续
  │
  ├─ 估算 node.model_info["vram_gb"]
  │
  ├─ 检查当前显存使用 + 新模型需求 vs 显存上限
  │     ├─ 充足: 加载
  │     └─ 不足: 触发 LRU 淘汰
  │             │
  │             ├─ 找最久未使用的节点
  │             ├─ 调用 node.unload()
  │             ├─ 释放显存
  │             └─ 重新检查
  │
  └─ 加载新节点
```

### LRU 淘汰策略

```
节点 A (5GB)        ← 最早未使用
节点 B (10GB)
节点 C (3GB)         ← 最近使用
[空闲: 2GB]
  │
  ▼ 加载新节点 D (12GB)，需要 10GB
  │
  ├─ 卸载 A (5GB) → 空闲 7GB，仍不足
  ├─ 卸载 B (10GB) → 空闲 17GB，足够
  └─ 加载 D
```

### 显存优化选项

每个视频/图像节点支持：

| 选项 | 效果 | 性能影响 |
|---|---|---|
| `enable_attention_slicing` | attention 计算分块 | 推理时间 +20% |
| `enable_vae_slicing` | VAE 分块 | 推理时间 +10% |
| `enable_vae_tiling` | VAE 瓦片化 | 推理时间 +15%，显存峰值 -50% |
| `enable_sequential_cpu_offload` | 模型层按需搬运 | 推理时间 +200%，显存峰值 -40% |
| `enable_model_cpu_offload` | 模型子模块按需搬运 | 推理时间 +30%，显存峰值 -30% |

---

## 流式输出架构

### AR 流式（ChatTTS / Fish / GPT-SoVITS）

```
时间轴 →
token:  [t1] [t2] [t3] [t4] [t5] ...
          │    │    │    │    │
          ▼    ▼    ▼    ▼    ▼
vocoder: [chunk1: 1-3] [chunk2: 4-5] ...
          │              │
          ▼              ▼
audio:   ▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░░░░░
         ↑              ↑
         第一批延迟      持续流出
         ~50ms          每 token ~20ms
```

**特点**：

- 第一批延迟极低（一个 token + 一次 vocoder 即可）
- 边生成边合成
- 适合实时对话、语音助手
- 总延迟 = 第一批延迟 + 后续 token 生成时间

### Flow Matching 分块流式（CosyVoice）

```
时间轴 →
mel frame: [f1] [f2] [f3] ... [fN]
           │    │    │       │
           ▼    ▼    ▼       ▼
flow decoder:  [chunk1: 1-150 frames]  [chunk2: 151-300 frames] ...
               │                       │
               ▼                       ▼
HiFiGAN:     [chunk1: ~1.74s]         [chunk2: ~1.74s] ...
             │                       │
             ▼                       ▼
audio:       ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░
             ↑
             第一批延迟
             ~300ms（需要生成 ~150 frames）
```

**特点**：

- 非自回归，可一次性生成整段 mel
- 分块流式：每 N 帧（典型 150，约 1.74 秒）通过声码器播放
- 总延迟 = ODE 步数 × step_size + vocoder 时间
- 适合长篇朗读、高质量场景

### chunk_size 选择

| 场景 | 推荐 chunk_size | 理由 |
|---|---|---|
| 实时对话 | 1-3 frames | 最低延迟 |
| 语音助手 | 5-10 frames | 平衡延迟和质量 |
| 长篇朗读 | 20-30 frames | 高吞吐，延迟可接受 |
| 离线合成 | 完整 | 最高质量 |

### 文本流式生成（Chat / TextGenerator）

```
时间轴 →
token:  [t1] [t2] [t3] [t4] [t5] ...
          │    │    │    │    │
          ▼    ▼    ▼    ▼    ▼
用户:    "人" "工" "智" "能" "是" ...
          ↑
         首批延迟
         ~100ms
```

**实现**：基于 `transformers.TextIteratorStreamer`，在后台线程中运行模型推理，主线程中逐个 yield token。`Chat` 和 `TextGenerator` 节点均提供 `stream()` 方法（与 `run()` 并列），在节点层面调用即可逐 token 获取输出。

**使用方式**：
```python
chat = Chat(model="Qwen/Qwen2.5-1.5B-Instruct")
for chunk in chat.stream(MosaicData(messages=[...], temperature=0.8)):
    print(chunk, end="", flush=True)
```

---

## 设计决策

### 1. 为什么用 MosaicData 而不是直接传 dict？

**dict 方案的问题**：
- 类型不明确
- 字段拼写错误只在运行时发现
- IDE 无法补全

**MosaicData 方案**：
- 不可变容器，类 Pydantic BaseModel
- `.get()` / `.set()` 链式调用
- 各域有专门类型（`ImageData` / `VideoData` / `AudioData` / `SubtitleData` / `DocumentData`）提供字段校验

### 2. 为什么 Pipeline 本身也是 Node？

**核心收益**：

- **可嵌套**：`big_pipe.add(small_pipe)` 即可把一组节点当一个节点用
- **可序列化**：`pipe.describe()` 完整描述拓扑结构
- **统一接口**：管道和节点都用 `load()` / `unload()` / `run()`

### 3. 为什么 TTS 要做四层架构而不是一个端到端类？

**单类方案的问题**：

- 不同后端（AR vs Flow）共享代码少
- 难以单独替换某一层（如想换声码器）
- 测试粒度粗

**四层架构收益**：

- **可插拔**：换声码器只需要替换 Vocoder 层
- **可复用**：ChatTTS 的 Vocos 也可被 Fish 复用
- **可测试**：每层可独立 mock 测试

### 4. 为什么 Scheduler 用 LRU 而非 FIFO？

- LRU 反映真实访问模式：用户最近用过的最可能再用
- 短视频/小模型不会驱逐大模型
- 实现简单，O(1) 的 `OrderedDict.move_to_end`

### 5. 为什么允许 entry_points / 装饰器 / 目录扫描三种插件机制？

| 场景 | 推荐机制 | 原因 |
|---|---|---|
| 第三方包发布 | entry_points | PyPI 友好，无侵入 |
| 应用内部 | 装饰器 | 最简单，import 时自动注册 |
| 运行时动态加载 | 目录扫描 | 无需重启 |

### 6. 为什么 lazy import 重依赖？

每个域节点在 `import` 时不实际加载 `torch` / `diffusers` / `transformers`，仅在 `load()` 时导入。好处：

- **导入快**：核心包可在无 ML 库环境运行
- **可选依赖友好**：未安装 `mosaic[video]` 也能用文本/音频域
- **测试友好**：单元测试无需 ML 环境

### 7. 为什么 PipelineResult 同时提供 get() 和 dict-like 访问？

- `result.get(key, default)`：安全查找，节点未运行返回 default
- `result[key]` / `result.outputs[key]`：直接访问，键不存在抛 `KeyError`

这两种语义并存可满足不同使用习惯。

---

## 下一步

- [TTS 完整指南](tts-guide.md) — 4 个 TTS 后端的详细对比
- [视频模型指南](video-models.md) — Wan / HunyuanVideo / LTX-Video
- [节点参考手册](nodes-reference.md) — 全部 42 节点
- [管道使用指南](pipeline-guide.md) — Pipeline 编排进阶
