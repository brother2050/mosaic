# Mosaic 视频模型指南

> Wan2.1 / Wan2.2 / HunyuanVideo / LTX-Video 在 Mosaic 中的使用方法。

## 目录

- [支持的视频模型](#支持的视频模型)
- [模型对比](#模型对比)
- [快速上手](#快速上手)
- [diffusers Pipeline 参数](#diffusers-pipeline-参数)
- [视频生成最佳实践](#视频生成最佳实践)
- [常见问题排查](#常见问题排查)

---

## 支持的视频模型

Mosaic 通过 diffusers 的 Pipeline 接口集成了 4 个主流视频生成模型：

| 模型 | 节点 | diffusers Pipeline | diffusers 版本 | 显存需求 |
|---|---|---|---|---|
| **Wan2.1-14B** | `WanVideo` | `WanPipeline` | >= 0.33.0 | ~30GB |
| **Wan2.1-1.3B** | `WanVideo` | `WanPipeline` | >= 0.33.0 | ~8GB |
| **Wan2.2-A14B** | `WanVideo` | `WanPipeline` | >= 0.35.0 | ~30GB |
| **HunyuanVideo** | `HunyuanVideo` | `HunyuanVideoPipeline` | >= 0.32.0 | ~60GB |
| **LTX-Video** | `LTXVideo` | `LTXPipeline` | >= 0.32.0 | ~12GB |
| **CogVideoX-5B** | `TextToVideo` | `CogVideoXPipeline` | >= 0.27.0 | ~18GB |
| **CogVideoX-2B** | `TextToVideo` | `CogVideoXPipeline` | >= 0.27.0 | ~9GB |
| **SVD** | `ImageToVideo` | `StableVideoDiffusionPipeline` | >= 0.27.0 | ~10GB |

---

## 模型对比

### Wan 系列（阿里通义万相）

**架构**：DiT (Diffusion Transformer)
**特点**：

- 中英文双语支持优秀
- 视频时长可达 5-10 秒
- 14B 旗舰版与 1.3B 轻量版可按显存切换
- Wan2.2 引入 MoE 架构，效果更好

**适用场景**：

- 中文短视频创作
- 显存受限的轻量部署（1.3B）
- 高质量商用视频

### HunyuanVideo（腾讯混元）

**架构**：DiT
**特点**：

- 大规模参数（13B）
- 支持中英文
- 视频质量最高，但显存需求大
- VAE chunking 优化降低峰值

**适用场景**：

- 顶级质量要求的视频
- 数据中心级 GPU（A100/H100）

### LTX-Video（Lightricks）

**架构**：Transformer
**特点**：

- 仅 2B 参数，模型最小
- 推理速度快（实时生成）
- 默认 30fps，30 帧约 1 秒视频
- 显存友好（~12GB）

**适用场景**：

- 实时应用（游戏、互动）
- 显存受限的设备
- 快速原型开发

### CogVideoX（智谱）

**架构**：3D VAE + Expert Transformer
**特点**：

- 较早的视频生成模型
- 显存需求中等
- 49 或 85 帧（特定值）

**适用场景**：

- 已有项目集成
- 显存中等（18GB 级别）

### SVD（Stable Video Diffusion）

**架构**：2D UNet + 3D UNet
**特点**：

- 图像到视频专用
- 输入单图，输出短动画
- 14-25 帧，约 2-4 秒

**适用场景**：

- 图生视频
- 静态图片"活化"

---

## 快速上手

### 1. Wan2.1 文生视频

```python
from mosaic.nodes.video import WanVideo
from mosaic import MosaicData

wan = WanVideo(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",  # 14B 高质量版
    enable_cpu_offload=True,                  # 显存不足时开启
    enable_vae_tiling=True,
)

result = wan.run(MosaicData(
    prompt="一只猫在海滩上散步，夕阳西下，海浪轻拍",
    num_frames=81,         # 约 5 秒 @ 16fps
    fps=16,
    num_inference_steps=30,
    guidance_scale=5.0,
))

result.get("video").save("cat_walk.mp4")
print(f"已生成 {result.get('num_frames')} 帧视频，时长 {result.get('duration'):.2f}s")
```

**轻量版（8GB 显存）**：

```python
wan = WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
```

### 2. HunyuanVideo

```python
from mosaic.nodes.video import HunyuanVideo
from mosaic import MosaicData

hv = HunyuanVideo(
    model="tencent/HunyuanVideo",
    enable_cpu_offload=True,    # 必须开启（默认 60GB 显存）
    enable_vae_tiling=True,
    enable_chunking=True,       # HunyuanVideo 专属 VAE 分块
)

result = hv.run(MosaicData(
    prompt="A dancing robot in the neon city",
    num_frames=129,    # 约 5 秒 @ 24fps
    fps=24,
    num_inference_steps=30,
    guidance_scale=7.5,
))
result.get("video").save("robot_dance.mp4")
```

### 3. LTX-Video（轻量快速）

```python
from mosaic.nodes.video import LTXVideo
from mosaic import MosaicData

ltx = LTXVideo(
    model="Lightricks/LTX-Video",
    enable_cpu_offload=True,
    enable_vae_tiling=True,
)

result = ltx.run(MosaicData(
    prompt="A car driving on a mountain road at sunset",
    num_frames=97,     # 约 3 秒 @ 30fps
    fps=30,
    num_inference_steps=20,
    guidance_scale=3.0,
))
result.get("video").save("car_drive.mp4")
```

### 4. CogVideoX（中等显存）

```python
from mosaic.nodes.video import TextToVideo
from mosaic import MosaicData

t2v = TextToVideo(
    model="THUDM/CogVideoX-5b",
    enable_vae_tiling=True,
)

# CogVideoX 必须 num_frames=49 或 85
result = t2v.run(MosaicData(
    prompt="阳光下的向日葵花田",
    num_frames=49,     # 自动调整
    fps=8,
))
result.get("video").save("sunflower.mp4")
```

### 5. SVD 图生视频

```python
from mosaic.nodes.video import ImageToVideo
from PIL import Image
from mosaic import MosaicData

i2v = ImageToVideo(model="stabilityai/stable-video-diffusion-img2vid-xt")
input_image = Image.open("input.jpg")

result = i2v.run(MosaicData(
    image=input_image,
    num_frames=25,         # SVD 支持 14-25 帧
    fps=7,
    motion_bucket_id=127,  # 0-255，值越大运动越剧烈
))
result.get("video").save("animated.mp4")
```

---

## diffusers Pipeline 参数

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | str | — | 文本提示词 |
| `negative_prompt` | str | None | 负向提示词 |
| `num_frames` | int | 模型特定 | 生成帧数 |
| `num_inference_steps` | int | 30 | 推理步数 |
| `guidance_scale` | float | 5.0-7.5 | 引导强度（越高越贴近 prompt） |
| `height` | int | 720 | 视频高度 |
| `width` | int | 1280 | 视频宽度 |
| `fps` | int | 16-30 | 输出帧率 |
| `seed` | int | None | 随机种子 |

### 各模型的帧数约束

| 模型 | num_frames 范围 | 特殊值 |
|---|---|---|
| **Wan2.1/2.2** | 任意 (自动调整) | 必须为 `4k+1` (49, 81, 121) |
| **HunyuanVideo** | 任意 | 必须为 4k+1（5/9/.../129），非有效值自动调整 |
| **LTX-Video** | 任意 | 建议 8 的倍数 + 1 |
| **CogVideoX** | 49 或 85 | 非有效值时自动调整 |
| **SVD** | 14-25 | 默认 14 (xt: 25) |

### 显存的参数影响

| 参数 | 显存影响 |
|---|---|
| `width × height` | 二次方关系 |
| `num_frames` | 线性关系 |
| `num_inference_steps` | 不影响峰值（影响时长） |

**显存公式**（约）：

```
峰值显存 ≈ 7GB (基础) + (W × H × num_frames × dtype_size × 1.5)
```

对于 fp16 的 1280×720×81 帧：~28GB。

---

## 视频生成最佳实践

### 1. Prompt 编写技巧

**好的 prompt**：

```
A young woman walking through a cherry blossom park, 
soft sunlight filtering through pink petals, slow motion, 
cinematic, 4K, shallow depth of field
```

**结构建议**：

1. **主体**：人物/物体（"A young woman"）
2. **动作**：核心动作（"walking through"）
3. **环境**：场景细节（"cherry blossom park"）
4. **氛围**：光线、风格（"soft sunlight, cinematic"）
5. **技术词**：画质、运镜（"4K, shallow depth of field"）

**避免**：

- 过于抽象（"a beautiful scene"）
- 矛盾描述（"dark bright room"）
- 复杂多主体（"three people, two dogs, one cat playing"）

### 2. 分辨率选择

| 显存 | 推荐分辨率 | 模型 |
|---|---|---|
| < 10GB | 480×720 | LTX-Video, Wan2.1-1.3B |
| 10-20GB | 720×1280 | Wan2.1-1.3B, LTX-Video |
| 20-40GB | 720×1280 | Wan2.1-14B, CogVideoX-5B |
| 40-80GB | 1280×1280 | HunyuanVideo, Wan2.1-14B |

### 3. 步数建议

| 场景 | 推荐步数 | 质量/速度权衡 |
|---|---|---|
| 快速预览 | 10-15 步 | 速度优先 |
| 一般用途 | 25-30 步 | 平衡 |
| 高质量 | 40-50 步 | 质量优先 |
| 离线生成 | 50+ 步 | 最佳质量 |

**经验法则**：步数从 30 提到 50 质量提升约 5-10%，但时间翻倍。

### 4. guidance_scale 选择

| 值 | 效果 |
|---|---|
| 1.0-3.0 | 创造性高，可能偏离 prompt |
| 5.0-7.5 | 推荐范围，平衡 |
| 10.0-15.0 | 严格遵循 prompt，可能过饱和 |
| > 15.0 | 容易过拟合、出现 artifacts |

**模型默认**：
- Wan: 5.0
- HunyuanVideo: 7.5
- LTX-Video: 3.0
- CogVideoX: 6.0

### 5. 显存优化策略

```python
# 策略 1: enable_model_cpu_offload（推荐）
WanVideo(enable_cpu_offload=True)
# 效果: 显存 -30%, 速度 -20%

# 策略 2: enable_vae_tiling
WanVideo(enable_vae_tiling=True)
# 效果: 显存峰值 -50%（仅影响 VAE 解码阶段）

# 策略 3: WanVideo 不支持 enable_sequential_cpu_offload
# WanVideo 仅支持 enable_cpu_offload（见策略 1）；
# 如需更激进的显存优化，请使用 TextToVideo 节点

# 策略 4: 减少 num_frames
WanVideo()
# run({"num_frames": 49})  # 49 帧比 81 帧省 40% 显存
```

### 6. 长视频生成

当前模型单次最多生成 5-10 秒。如需更长：

```python
from mosaic import MosaicData

# 方法 1: 多段拼接
chunks = []
for i in range(4):
    result = wan.run(MosaicData(
        prompt=f"一段连续动作的第 {i+1} 段",
        num_frames=81,
        seed=42 + i,  # 不同 seed 但保持风格
    ))
    chunks.append(result.get("video"))

# 方法 2: 视频续写
from mosaic.nodes.video import VideoContinuation
vc = VideoContinuation(model="THUDM/CogVideoX-5b")
extended = vc.run(MosaicData(video=chunks[0], num_frames=49))
```

### 7. 与其他节点组合

```python
from mosaic import MosaicData
from mosaic.nodes.export import VideoEncoder

# VideoEncoder 构造时仅指定输出格式；
# output_path 与 fps 在 run() 中通过 MosaicData 传入
encoder = VideoEncoder(format="mp4")
result = encoder.run(MosaicData(
    frames=frames,
    fps=16,
    output_path="final.mp4",
))
```

---

## 常见问题排查

### Q1: 加载时 `WanPipeline not found in diffusers`

需要 `diffusers >= 0.33.0`：

```bash
pip install --upgrade diffusers>=0.33.0
```

### Q2: Wan2.1-14B 找不到权重

必须使用带 `-Diffusers` 后缀的仓库名：

```python
# 正确
WanVideo(model="Wan-AI/Wan2.1-T2V-14B-Diffusers")

# 错误（原始格式）
WanVideo(model="Wan-AI/Wan2.1-T2V-14B")
```

WanVideo 节点会自动给不带后缀的名称补全。

### Q3: `CUDA out of memory`

按以下顺序尝试：

```python
from mosaic import MosaicData

# 1. 启用 CPU offload（最常用）
WanVideo(enable_cpu_offload=True, enable_vae_tiling=True)

# 2. 切换到更小的模型
WanVideo(model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")  # 8GB vs 30GB

# 3. 减少帧数
result = wan.run(MosaicData(prompt="...", num_frames=49))  # 81 → 49 省 40%

# 4. 减少分辨率
result = wan.run(MosaicData(prompt="...", width=832, height=480))  # 1280x720 → 832x480

# 5. 使用 enable_cpu_offload（WanVideo 不支持 sequential_cpu_offload）
WanVideo(enable_cpu_offload=True)
```

### Q4: 生成视频模糊或质量差

```python
from mosaic import MosaicData

# 1. 增加步数
result = wan.run(MosaicData(prompt="...", num_inference_steps=50))

# 2. 调整 guidance_scale
result = wan.run(MosaicData(prompt="...", guidance_scale=7.5))  # 适当提高

# 3. 使用更详细的 prompt
prompt = """
A young woman with long black hair, walking through 
a field of sunflowers, golden hour lighting, 
photorealistic, 8K, cinematic depth of field
"""
result = wan.run(MosaicData(prompt=prompt))
```

### Q5: T5 tokenizer 错误

```
ValueError: cannot be loaded as it does not seem to have any loading methods
```

安装缺失依赖：

```bash
pip install sentencepiece
```

Mosaic 已通过 `safe_load_pipeline()` 预检测并提示此问题。

### Q6: HunyuanVideo 加载慢

HunyuanVideo 模型较大（13B），首次加载需要：

- 下载：~30GB
- 加载到显存：~60GB 显存或开启 CPU offload

建议在 80GB 显存的 A100/H100 上运行。

### Q7: 视频生成时间太长

| 模型 | 步数 | 帧数 | 大致时间（A100） |
|---|---|---|---|
| Wan2.1-1.3B | 30 | 81 | ~3 分钟 |
| Wan2.1-14B | 30 | 81 | ~15 分钟 |
| HunyuanVideo | 30 | 129 | ~30 分钟 |
| LTX-Video | 20 | 97 | ~1 分钟 |
| CogVideoX-5B | 50 | 49 | ~5 分钟 |

减少时间的策略：

```python
from mosaic import MosaicData

# 减少步数（质量略降）
result = wan.run(MosaicData(prompt="...", num_inference_steps=15))  # 减半时间

# 减少帧数
result = wan.run(MosaicData(prompt="...", num_frames=49))  # 省 40% 时间
```

---

## 下一步

- [示例代码](../examples/03_video_domain.py) — 视频域完整示例
- [架构设计](architecture.md) — 调度器与显存管理
- [节点参考手册](nodes-reference.md) — 全部视频节点 API
