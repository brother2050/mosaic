# Changelog

Mosaic 项目的所有重要变更记录。

格式基于 [Keep a Changelog](https://keepachangelog.com/)，
本项目遵循 [Semantic Versioning](https://semver.org/)。

---

## [Unreleased]

### 新增

#### TTS 扩展
- 音频域 TTS 节点已扩展为路由器，支持 4 个生产级 TTS 后端：
  - **ChatTTS**（CC-BY-NC-4.0，24kHz，AR 流式 ~50ms）
  - **Fish Speech**（Apache-2.0，22.05kHz，多语言）
  - **GPT-SoVITS**（MIT，32kHz，少样本克隆）
  - **CosyVoice**（Apache-2.0，24kHz，Flow Matching 高质量）
- TTS 四层架构：`TextFrontend → AcousticModel → Vocoder → StreamAdapter`
- 流式 TTS 输出支持（AR 逐 token / Flow Matching 分块流式）
- 语音克隆支持（GPT-SoVITS / Fish / CosyVoice）
- 韵律控制（ChatTTS 的 oral/laugh/break/speed 标记）
- SFT 情感指令（CosyVoice）
- HFModelManager 统一管理 TTS 模型权重下载与路径解析
- 权重转换器（`weights.py`）支持原版模型到 Mosaic 格式转换
- 30 个 TTS 单元测试

#### 视频模型支持
- **Wan2.1 / Wan2.2** 集成（WanVideo 节点）
  - 自动补全 `-Diffusers` 后缀
  - 4k+1 帧数校验
  - 14B（~30GB）/ 1.3B（~8GB）/ 2.2-A14B（~30GB）
- **HunyuanVideo** 集成（HunyuanVideo 节点）
  - VAE chunking 专属优化
  - 默认 60GB 显存，可配 CPU offload
- **LTX-Video** 集成（LTXVideo 节点）
  - 轻量快速（~12GB 显存）
  - 30fps 高帧率输出
- HFModelManager 统一管理视频模型权重

#### 框架核心
- 异步执行：`AsyncTask` + `TaskManager` 支持并发任务编排
- 插件系统：entry_points + 装饰器 + 目录扫描三种注册机制
- CLI 工具：`mosaic list` / `info` / `create-node` / `run` / `version` / `doctor`
- 事件总线：进度、错误、中间结果实时上报
- 显存调度器：LRU 自动加载/卸载

#### 文档
- `docs/getting-started.md` — 5 分钟上手指南
- `docs/architecture.md` — 架构设计文档
- `docs/nodes-reference.md` — 42 节点参考手册
- `docs/pipeline-guide.md` — 管道使用指南
- `docs/tts-guide.md` — TTS 完整指南
- `docs/video-models.md` — 视频模型指南
- `docs/plugin-development.md` — 插件开发指南
- `docs/cli-reference.md` — CLI 参考手册
- 11 个完整示例（`examples/01-11_*.py`）
- `scripts/generate_api_docs.py` — 自动 API 文档生成器

### 改进
- `safe_load_pipeline()` 统一 Pipeline 加载工具
  - T5 tokenizer 预导入（解决 lazy loading 问题）
  - fp16 variant 回退
  - 详细错误诊断

### 修复
- 修复 Wan-AI 原始仓库无法直接 `from_pretrained` 的问题（自动补全 `-Diffusers` 后缀）

### 测试
- 新增 TTS 30 个单元测试（4 后端）
- 新增视频 30 个单元测试（WanVideo / HunyuanVideo / LTXVideo）
- 完整测试套件：phase1-7 + tts

---

## [0.1.0] - 2025-01-15

### 新增
- 初始版本发布
- 九大域 39 节点：
  - 文本域（6）：TextGenerator、Chat、TextRewriter、Translator、TextSummarizer、TextClassifier
  - 图像域（6）：TextToImage、ImageToImage、Inpainting、Upscaler、BackgroundRemover、Stylizer
  - 视频域（5）：TextToVideo、ImageToVideo、VideoContinuation、FrameInterpolator、FrameExtractor
  - 音频域（5）：TTS、ASR、MusicGenerator、SoundEffect、VoiceClone
  - 字幕域（3）：SubtitleGenerator、SubtitleTranslator、SubtitleAligner
  - 一致性域（3）：IdentityKeeper、StyleKeeper、CrossFrameConsistency
  - 数字人域（4）：AvatarDriver、LipSyncer、MotionGenerator、RealtimeRenderer
  - 导出域（3）：VideoEncoder、Livestreamer、MultiFormatExporter
  - RAG 域（4）：DocumentParser、VectorIndexer、Retriever、CitationGenerator
- 核心框架：Node / Pipeline / Registry / Scheduler / EventBus
- 基础 CLI 工具

---

## 版本说明

- **主版本号**：不兼容的 API 变更
- **次版本号**：向下兼容的新功能
- **修订号**：向下兼容的问题修复

---

## 贡献者

Mosaic 由开源社区贡献者共同维护。

详见 [GitHub Contributors](https://github.com/your-org/mosaic/graphs/contributors)。
