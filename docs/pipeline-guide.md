# Mosaic 管道使用指南

> 用 `Pipeline` 引擎把多个节点自由编排为复杂工作流。

## 目录

- [基础管道（串行）](#基础管道串行)
- [管道运算符 |](#管道运算符)
- [并行分支（Branch / Merge）](#并行分支-branch--merge)
- [中间产物检查](#中间产物检查)
- [异步执行](#异步执行)
- [PipelineResult 解读](#pipelineresult-解读)
- [跨域管道示例](#跨域管道示例)
- [性能优化建议](#性能优化建议)

---

## 基础管道（串行）

最简单的管道就是按顺序添加节点：

```python
from mosaic import Pipeline
from mosaic.nodes.text import TextGenerator
from mosaic.nodes.image import TextToImage

# 方式 1：add() 链式调用
pipeline = Pipeline()
pipeline.add(TextGenerator(model="Qwen2.5-7B"))
pipeline.add(TextToImage(model="SDXL"))

# 方式 2：构造时直接传
pipeline = Pipeline([
    TextGenerator(model="Qwen2.5-7B"),
    TextToImage(model="SDXL"),
])

# 运行
result = pipeline.run({"prompt": "画一只在月球上骑车的熊猫"})
print(result.get("image"))  # ImageData
```

**执行流程**：

```
MosaicData(prompt="...")
    │
    ▼ TextGenerator.run()
MosaicData(prompt="...", text="...")
    │
    ▼ TextToImage.run()
MosaicData(prompt="...", text="...", image=ImageData(...))
```

每个节点的输出会成为下一个节点的输入。

---

## 管道运算符 |

Pythonic 的 `|` 运算符让管道声明更优雅：

```python
from mosaic.nodes.text import TextGenerator
from mosaic.nodes.image import TextToImage
from mosaic.nodes.export import MultiFormatExporter

# 基础用法
pipe = TextGenerator() | TextToImage() | MultiFormatExporter()

# 混合 add() 和 |
pipe = Pipeline()
pipe.add(TextGenerator())
pipe.add(TextToImage() | MultiFormatExporter())  # 嵌套

# 等价展开
pipe2 = Pipeline([TextGenerator(), TextToImage(), MultiFormatExporter()])
```

`|` 返回一个新的 `Pipeline`，因此可以链式组合：

```python
# 复杂链路
pipe = (
    TextGenerator()
    | TextRewriter(style="formal")
    | TextToImage()
    | Upscaler(scale=2)
    | MultiFormatExporter()
)
```

---

## 并行分支（Branch / Merge）

`Branch` 让一个输入同时被多个节点处理，`Merge` 把多个输出合并：

```python
from mosaic import Pipeline, Branch, Merge
from mosaic.nodes.image import TextToImage
from mosaic.nodes.audio import TTS
from mosaic.nodes.subtitle import SubtitleGenerator

pipeline = Pipeline()

# 同一段文本 → 同时生成图、音频、字幕
pipeline.add(Branch([
    TextToImage(model="SDXL"),
    TTS(backend="chattts", language="zh"),
    SubtitleGenerator(),
]))

# 三路输出合并为一个 MosaicData
pipeline.add(Merge())

result = pipeline.run({"text": "一段文本"})
print(result.get("image"))    # ImageData
print(result.get("audio"))    # AudioData
print(result.get("subtitle")) # SubtitleData
```

**执行流程**：

```
MosaicData(text="...")
    │
    ▼ Branch
    ├─ TextToImage ─▶ image
    ├─ TTS ─────────▶ audio
    └─ SubtitleGen ─▶ subtitle
    │
    ▼ Merge
MosaicData(text, image, audio, subtitle)
```

### 合并策略

```python
Merge(strategy="concat")    # 默认，字段拼接
Merge(strategy="override")  # 后执行的覆盖先执行的
Merge(strategy="first")     # 保留第一个非空
```

### 条件分支

基于数据动态选择唯一路径：

```python
from mosaic import Pipeline, Branch, Merge

pipeline = Pipeline()
pipeline.add(Branch([
    TextToImage(model="SDXL").when(lambda d: d.get("mode") == "image"),
    TTS(backend="chattts").when(lambda d: d.get("mode") == "audio"),
], condition_field="mode"))
pipeline.add(Merge())

# 根据 input.mode 选择不同路径
result = pipeline.run({"text": "hello", "mode": "image"})
# 仅执行 TextToImage
```

---

## 中间产物检查

### 方式 1：边构造边运行（推荐）

把长管道拆成多段执行，逐步检查：

```python
from mosaic import Pipeline

# 段 1：生成文本
text_step = TextGenerator(model="Qwen2.5-7B")
text_result = text_step(MosaicData(prompt="写一段关于猫的故事"))
print("生成的文本:", text_result.get("text"))

# 段 2：基于该文本生成图
img_step = TextToImage(model="SDXL")
img_result = img_step(text_result)
img_result.get("image").save("cat.png")
```

### 方式 2：使用 on_intermediate 回调

```python
from mosaic import Pipeline

def log_intermediate(event):
    print(f"中间产物: {event.node} → {event.key}={event.value}")

pipeline = Pipeline([
    TextGenerator(),
    TextToImage(),
    MultiFormatExporter(),
])

pipeline.on(EventType.INTERMEDIATE, log_intermediate)
result = pipeline.run({"prompt": "..."})
```

### 方式 3：dry_run 验证

```python
from mosaic import Pipeline

pipeline = Pipeline([TextGenerator(), TextToImage()])

# 不实际执行，只校验类型和拓扑
dry = pipeline.dry_run(input_data=MosaicData(prompt="test"))
if dry.ok:
    print(f"管道结构有效，包含 {len(dry.steps)} 步")
else:
    print("错误:", dry.issues)
```

---

## 异步执行

对于长任务（视频生成、批量 TTS），用 `run_async` 后台运行：

```python
import asyncio
from mosaic import Pipeline
from mosaic.nodes.video import WanVideo

pipeline = Pipeline([WanVideo(model="Wan2.1-14B")])

# 启动后台任务
task = pipeline.run_async(prompt="一段长视频", num_frames=81)
print(f"任务已启动，ID: {task.task_id}")

# 期间做其他事
print("正在生成视频...")
for i in range(10):
    print(f"{i*30}秒过去了...")
    asyncio.sleep(30)

# 等待完成
result = task.result(timeout=600)
print(f"完成: {result.get('video')}")
```

### 任务管理器

并发管理多个任务：

```python
from mosaic.core.task_manager import TaskManager

tm = TaskManager()

# 提交多个独立任务
task1 = tm.submit(pipeline1, prompt="A")
task2 = tm.submit(pipeline2, prompt="B")
task3 = tm.submit(pipeline3, prompt="C")

# 等待全部完成
results = tm.wait_all(timeout=600)
for r in results:
    print(r)
```

### 取消任务

```python
task = pipeline.run_async(prompt="...")
# 任务在 30 秒内未完成
import time
time.sleep(30)
task.cancel()
```

---

## PipelineResult 解读

`Pipeline.run()` 返回 `PipelineResult`：

```python
result = pipeline.run({"prompt": "..."})

# 节点输出（按节点名）
result.get("text")           # MosaicData
result.get("image")          # MosaicData
result.outputs               # dict[str, MosaicData]

# 整链路状态
result.steps                 # list[NodeSpec], 执行过的节点
result.duration              # float, 总耗时（秒）
result.ok                    # bool, 是否全部成功
result.errors                # list[NodeError], 失败的节点错误

# 类型安全访问
text_data: MosaicData = result.get("text", default=MosaicData())
```

### 错误处理

```python
result = pipeline.run({"prompt": "..."})

if not result.ok:
    for err in result.errors:
        print(f"节点 {err.node} 失败: {err.error}")
        # 可以单独重试
        retry_result = err.node.run(err.input_data)
```

### 链式 get

```python
# 嵌套查找
image = result.get("image").get("image_data")  # MosaicData -> ImageData
```

---

## 跨域管道示例

### 示例 1：文字 → 图像 → 视频

```python
from mosaic import Pipeline
from mosaic.nodes.text import TextGenerator
from mosaic.nodes.image import TextToImage, Upscaler
from mosaic.nodes.video import ImageToVideo
from mosaic.nodes.export import VideoEncoder

pipe = (
    TextGenerator(model="Qwen2.5-7B")
    | TextToImage(model="SDXL")
    | Upscaler(scale=2)
    | ImageToVideo(model="SVD-XT")
    | VideoEncoder(output_path="output.mp4", fps=8)
)

result = pipe.run({"prompt": "A panda riding a bicycle on the moon"})
print(f"视频已保存: output.mp4")
```

### 示例 2：TTS → 口型同步 → 数字人

```python
from mosaic.nodes.audio import TTS
from mosaic.nodes.digital_human import LipSyncer, AvatarDriver
from mosaic.nodes.image import TextToImage

pipe = (
    TextToImage(model="SDXL")  # 生成数字人形象
    | AvatarDriver()            # 初始化形象
    | TTS(backend="cosyvoice")  # 生成语音
    | LipSyncer()               # 口型同步
)

result = pipe.run({
    "prompt": "a friendly digital assistant",
    "text": "你好，我是数字人助手。",
})
```

### 示例 3：文档 → RAG → 回答

```python
from mosaic.nodes.rag import DocumentParser, VectorIndexer, Retriever, CitationGenerator
from mosaic.nodes.text import TextGenerator

pipe = (
    DocumentParser()
    | VectorIndexer(embedding_model="BAAI/bge-m3", index_path="./index")
    | Retriever(top_k=5)
    | TextGenerator(model="Qwen2.5-7B")
    | CitationGenerator()
)

result = pipe.run({
    "file_path": "manual.pdf",
    "query": "如何使用 Mosaic 的 Pipeline？",
    "index_path": "./index",
})
print(result.get("answer"))  # 带引用的答案
```

### 示例 4：文本 → TTS → 字幕 → 视频编码

```python
from mosaic.nodes.audio import TTS
from mosaic.nodes.subtitle import SubtitleGenerator, SubtitleAligner
from mosaic.nodes.export import VideoEncoder
from mosaic.nodes.image import TextToImage

pipe = (
    TextToImage(model="SDXL")
    | TTS(backend="chattts")
    | SubtitleGenerator()
    | SubtitleAligner()
    | VideoEncoder(output_path="video.mp4", fps=24)
)

result = pipe.run({
    "prompt": "A scenic mountain view",
    "text": "远处的山峰在云雾中若隐若现",
})
```

### 示例 5：并行处理图像和音频

```python
from mosaic import Pipeline, Branch, Merge
from mosaic.nodes.image import TextToImage, Upscaler
from mosaic.nodes.audio import TTS
from mosaic.nodes.subtitle import SubtitleGenerator

pipe = Pipeline([
    Branch([
        TextToImage(model="SDXL") | Upscaler(scale=4),
        TTS(backend="cosyvoice"),
        SubtitleGenerator(),
    ]),
    Merge(strategy="concat"),
])

result = pipe.run({"text": "A poetic description of autumn"})
# 同时得到 image, audio, subtitle
```

---

## 性能优化建议

### 1. 显存管理

#### 共享调度器

所有节点共享全局 `Scheduler`，自动 LRU：

```python
from mosaic.core.scheduler import get_scheduler

sched = get_scheduler()  # 全局单例
# 节点 load() 时自动 sched.track(self)
# 显存不足时按 LRU 自动卸载
```

#### 强制卸载

```python
# 单节点卸载
text_node.unload()

# 卸载所有
for node in pipeline.nodes:
    node.unload()
```

### 2. 并行策略

#### 何时用 Branch

适合：

- 多个独立的下游任务
- 不同模态（图像、音频、字幕）
- 计算密集且互不依赖

不适合：

- 节点 B 依赖节点 A 的输出
- 总显存需求超过单卡容量

```python
# 适合：图文音三路独立
Branch([TextToImage(), TTS(), SubtitleGenerator()])

# 不适合：图像增强依赖前置
TextToImage() | Upscaler()  # 必须串行
```

#### 限制并发数

```python
from mosaic import Pipeline
from concurrent.futures import ThreadPoolExecutor

pipeline = Pipeline([...])
pipeline.executor = ThreadPoolExecutor(max_workers=2)  # 限制为 2 路
```

### 3. 模型复用

#### 共享同一模型实例

```python
from mosaic.nodes.image import TextToImage, ImageToImage

# 两个节点共享同一 SDXL 实例（通过 scheduler 复用）
shared_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = (
    TextToImage(model=shared_model)
    | ImageToImage(model=shared_model)  # 不会重复加载
)
```

#### 预热常用模型

```python
# 启动时预热
text_node = TextGenerator(model="Qwen2.5-7B")
text_node.load()  # 立即加载

# 后续管道运行更快
pipe = Pipeline([text_node, TextToImage()])
result = pipe.run({"prompt": "..."})
```

### 4. 缓存与持久化

#### 缓存 embedding 结果

```python
from mosaic.nodes.rag import VectorIndexer

# 索引一次，多次查询
VectorIndexer(index_path="./my_index").run(documents)
# 后续 Retriever 直接读 ./my_index
```

#### 缓存 TTS 结果

```python
import hashlib

def tts_with_cache(text):
    key = hashlib.md5(text.encode()).hexdigest()
    cache_path = f"./tts_cache/{key}.wav"
    if os.path.exists(cache_path):
        return AudioData.load(cache_path)
    audio = TTS(backend="chattts").run({"text": text}).get("audio")
    audio.save(cache_path)
    return audio
```

### 5. 流式与异步结合

```python
import asyncio

async def stream_video_pipeline():
    pipe = Pipeline([
        TTS(backend="chattts"),
        SubtitleGenerator(),
    ])

    # 启动管道
    task = pipe.run_async(text="一段长文本...")

    # 同时流式处理 TTS
    async for chunk in pipe.nodes[0].synthesize_stream(text="..."):
        play(chunk)

    # 等待完整结果
    result = await task
    return result

asyncio.run(stream_video_pipeline())
```

### 6. 监控与日志

订阅事件总线做实时监控：

```python
from mosaic.core.events import get_event_bus, EventType

bus = get_event_bus()

@bus.on(EventType.NODE_START)
def on_start(event):
    print(f"[{event.timestamp:.2f}] 节点 {event.node} 开始")

@bus.on(EventType.NODE_COMPLETE)
def on_complete(event):
    print(f"[{event.timestamp:.2f}] 节点 {event.node} 完成, 耗时 {event.duration:.2f}s")

@bus.on(EventType.PROGRESS)
def on_progress(event):
    if event.total > 1:
        pct = event.current / event.total * 100
        print(f"[{event.node}] 进度: {pct:.0f}% {event.message}")

@bus.on(EventType.NODE_ERROR)
def on_error(event):
    print(f"[ERROR] {event.node}: {event.error}")
```

---

## 下一步

- [节点参考手册](nodes-reference.md) — 全部节点 API
- [插件开发指南](plugin-development.md) — 自定义节点
- [示例代码](../examples/11_cross_domain_pipeline.py) — 跨域管道完整示例
