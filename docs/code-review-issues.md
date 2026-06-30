# Mosaic 代码审查问题清单

> 审查日期：2026-06-30
> 审查范围：`mosaic/nodes/audio/`、`mosaic/nodes/text/`、`mosaic/nodes/image/`、`mosaic/nodes/video/`、`mosaic/nodes/digital_human/`、`mosaic/nodes/consistency/`、`mosaic/core/`、`mosaic/cli/`
> 状态：仅记录，暂不修改
> 组织方式：按模块/目录分组，便于特定模块精细化修改

---

## 问题统计

### 按模块统计

| 模块 | 目录路径 | 高 | 中 | 低 | 合计 |
|------|---------|---|---|---|------|
| TTS 文本前端 | `tts_backends/text_frontends/` | 2 | 9 | 2 | **13** |
| TTS 后端实现 | `tts_backends/implementations/` + `base.py` | 4 | 7 | 1 | **12** |
| TTS 后端注册表 | `tts_backends/registry.py` | 0 | 1 | 0 | **1** |
| TTS 节点 | `nodes/audio/tts.py` | 2 | 3 | 1 | **6** |
| 参考音频处理 | `nodes/audio/_ref_audio_utils.py` | 0 | 3 | 1 | **4** |
| 声学模型 | `tts_backends/acoustic_models/` | 0 | 4 | 1 | **5** |
| 流式输出 | `tts_backends/streaming/` | 0 | 2 | 0 | **2** |
| 文本节点 | `nodes/text/` | 1 | 4 | 2 | **7** |
| 图像节点 | `nodes/image/` | 1 | 5 | 4 | **10** |
| 视频节点 | `nodes/video/` | 2 | 5 | 3 | **10** |
| 数字人节点 | `nodes/digital_human/` | 0 | 0 | 5 | **5** |
| 一致性节点 | `nodes/consistency/` | 0 | 3 | 2 | **5** |
| 调度器 | `core/scheduler.py` | 0 | 5 | 2 | **7** |
| 注册表 | `core/registry.py` | 0 | 3 | 3 | **6** |
| 类型系统 | `core/types.py` | 1 | 5 | 1 | **7** |
| 管道与节点 | `core/pipeline.py` + `node.py` + `branch.py` | 1 | 5 | 4 | **10** |
| 事件/任务/上下文 | `core/events.py` + `task.py` + `context.py` | 1 | 2 | 3 | **6** |
| 模型缓存与插件 | `core/model_cache.py` + `plugin.py` | 0 | 2 | 3 | **5** |
| CLI | `cli/` | 1 | 2 | 4 | **7** |
| 公共工具 | `nodes/_pipeline_utils.py` | 0 | 3 | 1 | **4** |
| 全局问题 | 跨模块 | 0 | 0 | 2 | **2** |
| **合计** | | **16** | **73** | **45** | **134** |

---

## 第一部分：音频 TTS 模块

### 1.1 文本前端与 Tokenizer（13 项）

> 模块路径：`mosaic/nodes/audio/tts_backends/text_frontends/`
> 涉及文件：`chat_tokenizer.py`、`fish_tokenizer.py`、`sovits_tokenizer.py`、`cosyvoice_tokenizer.py`

#### [高] ChatTTS preprocess 破坏内嵌韵律标记（原 A1-1）
- **文件**：`chat_tokenizer.py`
- **行号**：512-538
- **描述**：`preprocess` 第 529 行将所有数字转中文，第 532-536 行的过滤正则 `[^\u4e00-\u9fffa-zA-Z0-9,.!?;:'\"()\\[\\]\\s]` 不允许下划线 `_`。用户在 text 中写 `[break_4]` 时，先被转成 `[break四]`，再被过滤掉 `_` 变成 `[break4]`，最终无法匹配 `SPECIAL_TOKENS` 里的 `[break_4]`，韵律标记完全失效。当前设计仅支持通过 `prosody_prompt` 参数注入标记，但这一限制未在文档/校验中说明。

#### [高] 数字规范化粗暴逐位转中文（原 B3-1）
- **文件**：`chat_tokenizer.py` 第 529 行；`sovits_tokenizer.py` 同名 `_DIGIT_TO_CN`
- **描述**：`2024` → `二零二四`（非"两千零二十四"），`3.14` → `三.一四`，电话号 `13800138000` → 逐位中文。无整数/小数/日期/时间/货币/电话/序号的上下文感知规范化。

#### [中] insert_prosody_tokens 对未知标记静默丢弃（原 A1-2）
- **文件**：`chat_tokenizer.py`
- **行号**：461-507
- **描述**：正则匹配任意 `[xxx]`，未命中 break/laugh/oral/speed 分类的标记（如拼写错误 `[braek_4]`）被静默丢弃，无校验、无告警。

#### [中] break 标记重复插入逻辑可疑（原 A1-3）
- **文件**：`chat_tokenizer.py`
- **行号**：500-505
- **描述**：`break_str = "".join(t for t in tags if t.startswith("[break"))` 把所有 break 标记拼接成一个字符串，然后在每个标点后都插入这一整串。若用户同时指定多个 break 级别，每个标点后都会塞入全部 break 标记，语义错误。

#### [中] ChatTTS 把语言字符串当文本编码（原 A2-1）
- **文件**：`chat_tokenizer.py`
- **行号**：309
- **描述**：将 `"zh"`/`"en"` 直接作为普通文本做字符级编码，并非 ChatTTS 真实的语言条件机制。不校验 language 是否在 `spec.supported_languages=["zh","en"]` 内，传 `"ja"` 会静默编码为 `j`、`a` 两个 token 注入。

#### [中] Fish 语言映射对未知语言静默回退为中文（原 A2-2）
- **文件**：`fish_tokenizer.py`
- **行号**：89-94, 273
- **描述**：`_LANG_TOKEN_MAP` 仅 zh/en/ja/ko，`language="yue"`、`"de"`、`"fr"` 等未覆盖语言会静默映射到 `<zh>`，无告警。

#### [中] SoVITS 语言分支对 ko/yue 回退为中文（原 A2-3）
- **文件**：`sovits_tokenizer.py`
- **行号**：301-313
- **描述**：仅处理 zh/en/ja，else 分支默认走中文 G2P。但 `spec.supported_languages` 声称支持 `["zh","en","ja","ko","yue"]`，ko/yue 实际被当中文处理，与声明不符。

#### [中] CosyVoice 完全忽略 language 参数（原 A2-4）
- **文件**：`cosyvoice_tokenizer.py`
- **行号**：199 附近
- **描述**：文档写"language 仅作记录，不影响编码逻辑"，但 `spec.supported_languages` 声明 `["zh","en","ja","ko","yue","de","fr"]` 七种语言。声明与实现脱节。

#### [中] ChatTTS 解码 speaker 失败静默返回 None（原 A3-3）
- **文件**：`chat_tokenizer.py`
- **行号**：374-376
- **描述**：Base16384 解码失败时返回 None，调用方无法区分"未提供 speaker"与"speaker 字符串损坏"，且无告警日志。

#### [中] Fish/SoVITS 英文文本缺数字与符号规范化（原 B3-2）
- **文件**：`fish_tokenizer.py`、`sovits_tokenizer.py`
- **行号**：sovits `_g2p_english` 第 526 行
- **描述**：英文分词正则 `r"[A-Za-z']+|[,.!?;:]"` 不识别连字符、数字、`$`、`%`、`@` 等，`"3.14"`、`"state-of-the-art"`、`"100%"` 会被丢弃或错切。

#### [中] Base16384 remainder 编解码不对称（原 E5-2）
- **文件**：`chat_tokenizer.py`
- **行号**：917-947 vs 417-456
- **描述**：encode 对非 7 字节倍数载荷做左填零，decode 做右移丢弃尾部位。两者不对称，round-trip 可能出错。

#### [低] _encode_text 子词匹配窗口硬编码（原 A1-4）
- **文件**：`chat_tokenizer.py`
- **行号**：221
- **描述**：`j = min(n, i + 20)` 最大子词长度硬编码为 20，缺乏与词表最大长度的关联校验。

#### [低] ChatTTS preprocess 过滤正则 CJK 范围仅覆盖基本块（原 E1-1）
- **文件**：`chat_tokenizer.py`
- **行号**：532-536
- **描述**：`\u4e00-\u9fff` 仅 CJK 统一表意文字基本块，遗漏 Extension A/B/C、兼容汉字、全角符号等。

---

### 1.2 TTS 后端实现（12 项）

> 模块路径：`mosaic/nodes/audio/tts_backends/implementations/` + `tts_backends/base.py`
> 涉及文件：`chattts_backend.py`、`cosyvoice_backend.py`、`fish_backend.py`、`sovits_backend.py`、`base.py`

#### [高] speaker 含义跨后端不统一且无统一校验（原 A3-1）
- **文件**：`tts_backends/base.py` 及各 `implementations/*.py`
- **描述**：同一个 `speaker: str | None` 参数在不同后端语义完全不同（ChatTTS 当嵌入字符串；Fish 当音频路径/tensor；SoVITS/CosyVoice 当"缓存名 or 音频路径"），基类无任何统一校验或类型区分。调用方切换后端时极易踩坑。

#### [高] 流式生成中途异常未清理会话与 KV 缓存（原 D1-1）
- **文件**：`tts_backends/base.py` 306-398；`chattts_backend.py` 657-779；`sovits_backend.py` 716-729；`cosyvoice_backend.py` 733-749
- **描述**：`for ... in generate_stream():` 循环外层无 `try/finally`。若中途抛异常（CUDA OOM、解码失败），`StreamSession` 缓冲与声学模型 KV cache 不会被释放/重置，资源泄漏且后续合成可能受污染状态影响。

#### [高] SoVITS 用 32kHz 波形喂给 HuBERT，采样率疑似不匹配（原 C2-1）
- **文件**：`sovits_backend.py`
- **行号**：1045-1052, 1006
- **描述**：`chinese-hubert-base` 通常以 16kHz 训练，这里却把 32kHz 波形并告知 extractor `sampling_rate=32000`，可能造成 SSL 语义 token 质量严重下降。GPT-SoVITS 原始管线参考音频一般用 16kHz。

#### [高] CosyVoice LLM 加载失败后用 token_ids 直接当 text_feats（原 E4-2）
- **文件**：`cosyvoice_backend.py`
- **行号**：516-523, 986-1017
- **描述**：LLM 加载失败时 `text_feats = token_ids`（整数 token id 张量）直接喂给 FlowMatchingModel 当条件，维度/语义全错，产出垃圾音频却无任何错误抛出。

#### [中] Fish 内置 speaker 名称无法使用（原 A3-2）
- **文件**：`fish_backend.py`
- **描述**：`list_speakers` 返回 `["default","male","female"]`，但 `encode_speaker`/`synthesize` 对 str 类型一律当音频路径处理。传入 `speaker="default"` 会被当作不存在的音频文件路径，触发深层晦涩错误。

#### [中] SoVITS/CosyVoice 未知 speaker 静默返回 None（原 A3-4）
- **文件**：`sovits_backend.py` 952-989、`cosyvoice_backend.py` 1039-1058
- **描述**：speaker 既不在缓存也不是文件时返回 None，合成继续用默认音色但不告知用户"指定 speaker 未生效"。

#### [中] 多处异常被静默吞掉（原 E3-1）
- **文件**：`sovits_backend.py` 980-986/1064-1066、`cosyvoice_backend.py` 多处
- **描述**：失败仅 `logger.debug` 或直接返回 None，生产环境难以定位问题。

#### [中] fish clone_voice 对非法路径给出深层错误（原 E3-2）
- **文件**：`fish_backend.py` 782-786
- **描述**：`clone_voice` 把 str 当 speaker 透传 synthesize，路径不存在/格式错误时报错点深在声学模型，无前置校验。

#### [中] SoVITS SSL 提取 argmax 量化、说话人嵌入用 one-hot 均值（原 E4-3）
- **文件**：`sovits_backend.py` 1060-1063, 1068-1096
- **描述**：argmax 量化与真实 k-means 语义 token 差异大；one-hot 均值作 speaker embedding 几乎无说话人区分力，语音克隆效果存疑。

#### [中] CosyVoice LLM hidden_dim 与 FlowMatching condition_dim 可能不匹配（原 E4-6）
- **文件**：`cosyvoice_backend.py` 381-391 vs `_load_llm`
- **描述**：LLM `last_hidden_state` 维度为 896，而 FlowMatchingModel `condition_dim=512`，若内部无投影对齐会维度不匹配崩溃。

#### [中] 类型注解过度使用 Any（原 E5-1）
- **文件**：各后端
- **描述**：核心数据流类型用 Any 丢失了静态检查价值。

#### [低] stream_batch 等魔法数字硬编码（原 D2-2）
- **文件**：`chattts_backend.py` 756、`fish_backend.py` 728、`sovits_backend.py` 700
- **描述**：`stream_batch=24`/`16` 硬编码，无法按模型/延迟需求调整。

---

### 1.3 TTS 后端注册表（1 项）

> 模块路径：`mosaic/nodes/audio/tts_backends/registry.py`

#### [中] auto_select 语言过滤与各后端实际支持不一致（原 A2-5）
- **文件**：`tts_backends/registry.py`
- **行号**：172-177
- **描述**：`auto_select` 信任 `spec.supported_languages` 做过滤，但如上所述 spec 声称支持的语言在后端/tokenizer 层并未真正落地，会导致选出的后端实际无法正确处理该语言。

---

### 1.4 TTS 节点（6 项）

> 模块路径：`mosaic/nodes/audio/tts.py`

#### [高] 句子分割正则误切缩写、小数、序号（原 B1-1）
- **文件**：`tts.py`
- **行号**：96-134，第 112 行
- **描述**：`re.split(r"[。！？.!?；;\n]+", text)` 中的 `.` 会把 `Mr.`、`Dr.`、`e.g.`、`3.14`、`No.1`、`2024.06.30` 全部错误切断。完全没有缩写/小数/日期保护逻辑。英文场景严重缺陷。

#### [高] 长文本拆分仅对 edge/transformers 生效，扩展后端不拆分（原 B1-3）
- **文件**：`tts.py`
- **行号**：519 附近
- **描述**：ChatTTS/Fish/SoVITS/CosyVoice 四个扩展后端的 `synthesize`/`synthesize_stream` 直接把整段文本喂给声学模型，不做长度拆分。当文本超过模型 `max_position_embeddings`（LLaMA AR 为 2048）时会越界报错或静默截断。长文本合成不可用。

#### [中] 长句二次切分后用空格重组，污染中文（原 B1-2）
- **文件**：`tts.py`
- **行号**：122-130
- **描述**：超长句按 `r"[，,、\s]+"` 切分后用 `" ".join(...)` 重组，会在中文片段间插入空格，可能影响后续 G2P/分词。

#### [中] 完全缺失 SSML 解析（原 B2-1）
- **文件**：整个 `nodes/audio/` 目录
- **描述**：无任何 SSML 解析逻辑。`<speak>`、`<break time="1s"/>`、`<emphasis>` 等标签会被原样送入 TTS；ChatTTS 的 preprocess 过滤会吃掉 `<>` 但残留内容，edge-tts 会把标签当正文朗读。

#### [中] edge-tts 采样率来源不一致（原 C2-3）
- **文件**：`tts.py`
- **行号**：224, 239
- **描述**：soundfile 分支用文件实际采样率，transformers 回退固定 24000，两条路径输出采样率可能不一致。

#### [低] max_length=200 硬编码且无配置（原 B1-4）
- **文件**：`tts.py`
- **行号**：112 附近
- **描述**：拆分阈值 200 字为魔法数字，未与后端实际上下文长度关联，也不可由调用方覆盖。

---

### 1.5 参考音频处理（4 项）

> 模块路径：`mosaic/nodes/audio/_ref_audio_utils.py`

#### [中] mp3 等格式支持依赖隐式回退（原 C1-1）
- **文件**：`_ref_audio_utils.py`
- **行号**：111-131
- **描述**：优先用 soundfile，失败回退 librosa。soundfile 是否支持 mp3 取决于系统 libsndfile 版本。错误消息只提示 `pip install soundfile`，未说明格式限制与 librosa 回退路径。

#### [中] 重采样回退用线性插值，质量差（原 C2-2）
- **文件**：`_ref_audio_utils.py`
- **行号**：134-146
- **描述**：librosa 不可用时退化为 `np.linspace` 线性插值重采样，对 2x 以上采样率变换会产生明显伪影，且无告警。

#### [中] 多声道合并假设 shape 为 [samples, channels]（原 C3-1）
- **文件**：`_ref_audio_utils.py`
- **行号**：76-77
- **描述**：`waveform.mean(axis=-1)` 假设 soundfile 返回 `[samples, channels]`，但若上游传入的是 `[channels, samples]` 布局会被错误求平均。缺少对 shape 的判别与转置保护。

#### [低] 无显式格式探测与统一转码（原 C1-2）
- **文件**：`_ref_audio_utils.py`
- **描述**：未基于扩展名/MAGIC 做格式探测，也未把任意输入统一转 wav。

---

### 1.6 声学模型（5 项）

> 模块路径：`mosaic/nodes/audio/tts_backends/acoustic_models/`
> 涉及文件：`llama_ar.py`、`gpt2_ar.py`、`flow_matching.py`

#### [中] generate_stream 仅捕获 RuntimeError（原 D1-2）
- **文件**：`llama_ar.py`
- **行号**：603-609, 711-717
- **描述**：KV cache 回退仅 catch RuntimeError，CUDA OOM 等非 RuntimeError 会直接冒泡，无优雅降级。

#### [中] flow_matching load_weights 用 strict=False 静默加载（原 E4-1）
- **文件**：`flow_matching.py`
- **行号**：849-856
- **描述**：`load_state_dict(strict=False)` 失败时仅 debug 日志，权重缺失/多余键不告警，可能以半初始化权重推理。

#### [中] llama_ar generate 输出张量做了两次 .T，疑似形状 bug（原 E4-5）
- **文件**：`llama_ar.py`
- **行号**：542-656, 760
- **描述**：`cat(dim=0).T` 后再 `.T`，两次转置相互抵消，返回 `[steps, num_vq]`，但文档声称 `[num_vq, generated_len]`，DVAE 解码器期望 `[num_vq, frames]`。形状与文档/下游期望不符。

#### [中] 大量魔法数字未命名常量化（原 E2-1）
- **文件**：`llama_ar.py`、`gpt2_ar.py`、`flow_matching.py`（及 `chat_tokenizer.py`、`cosyvoice_backend.py`、`sovits_backend.py`）
- **描述**：如 `_BASE16384_OFFSET`、子词窗口 20、spk 归一化 eps=1e-9、velocity clamp [-10,10]、`spk_cond * 0.1`、`mel_fps=86.13`、`int(len(text)*15)`、`target_length_seconds=30.0`、`semantic_vocab_size=768` 等，应抽为命名常量或可配置参数。

#### [低] gpt2_ar stop 条件 .item() 对 batch>1 会崩（原 E4-4）
- **文件**：`gpt2_ar.py`
- **行号**：199-240
- **描述**：`.item()` 仅支持单元素张量，batch>1 时抛 RuntimeError。当前 batch 恒为 1 故未触发。

---

### 1.7 流式输出（2 项）

> 模块路径：`mosaic/nodes/audio/tts_backends/streaming/`

#### [中] chunk_size / overlap 无边界校验（原 D2-1）
- **文件**：`streaming/base.py`
- **描述**：未校验 `chunk_size > 0`、`overlap < chunk_size`。`chunk_size=0` 时返回空块；`overlap >= chunk_size` 会导致交叉淡化逻辑异常。

#### [中] 无流式取消/中断机制（原 D3-1）
- **文件**：`streaming/base.py`、各后端 synthesize_stream
- **描述**：StreamSession 无 `cancel()`/`abort()`；消费者停止迭代时仅触发 GeneratorExit，声学模型 CUDA 资源、KV cache 无显式释放钩子。

---

## 第二部分：生成节点模块

### 2.1 文本节点（7 项）

> 模块路径：`mosaic/nodes/text/`
> 涉及文件：`chat.py`、`generator.py`、`summarizer.py`、`translator.py`、`classifier.py`、`rewriter.py`

#### [高] int()/float() 类型转换无异常处理（原 A1-文本部分）
- **文件**：`chat.py` 100-103、`generator.py` 89-92、`summarizer.py` 150/175-176、`translator.py` 167-168
- **描述**：所有节点的 `int()`/`float()` 参数转换无 try/except，传入非数字字符串（如 `"abc"`）会抛出难以理解的 `ValueError`，而非给出清晰的参数错误提示。

#### [中] 参数范围检查不全面（原 A2-文本部分）
- **文件**：`chat.py`、`generator.py`、`summarizer.py`、`translator.py`
- **描述**：`max_new_tokens` 无下限检查（可为 0 或负数）；`temperature`/`top_p` 无范围限制（应限制在 0-2 / 0-1）。传入越界值不会报错，但会导致模型推理异常。

#### [中] 超长 prompt 无上下文长度保护（原 D2）
- **文件**：`chat.py` 100-118、`generator.py` 89-106、`summarizer.py` 145-184、`translator.py` 155-176
- **描述**：无超长 prompt 检测。若累计 token 数超过模型上下文长度，会在模型推理时抛出底层错误，而非提前友好提示。

#### [中] 翻译目标语言未校验是否支持（原 D1）
- **文件**：`translator.py`
- **行号**：160-164
- **描述**：仅校验 `target_language` 是非空字符串，不检查是否在 `_LANGUAGE_NAMES` 支持的语言列表中。传入不支持的语言代码不会报错。

#### [中] 空字符串 prompt 处理不一致（原 C1-文本部分）
- **文件**：`generator.py` 81-86
- **描述**：仅检查 `isinstance(prompt, str)`，未检查 `prompt.strip()`，允许空字符串和纯空白字符串通过。注：视频节点（text_to_video.py 等）正确使用了 `not prompt.strip()` 检查。

#### [低] 分类器标签解析正则表达式边界情况（原 D3）
- **文件**：`classifier.py`
- **行号**：283
- **描述**：`re.split(r"[、,，\n;；]+", text)` 未处理制表符 `\t`、空格分隔、冒号 `:` 等分隔符。

#### [低] 改写器后缀移除逻辑不全面（原 D4）
- **文件**：`rewriter.py`
- **行号**：162-166
- **描述**：`suffix_markers` 仅检查 `"\n\n希望"`、`"\n\n注："` 等固定模式，若模型输出使用 `"\n希望"` 或 `"\n\n以上是"` 等变体则无法移除。

---

### 2.2 图像节点（10 项）

> 模块路径：`mosaic/nodes/image/`
> 涉及文件：`text_to_image.py`、`image_to_image.py`、`inpainting.py`、`upscaler.py`、`stylizer.py`、`background_remover.py`

#### [高] int()/float() 类型转换无异常处理（原 A1-图像部分）
- **文件**：`text_to_image.py` 125-134、`image_to_image.py` 138-142、`inpainting.py` 145-146、`upscaler.py` 131-134、`stylizer.py` 186-194
- **描述**：同文本节点，`int()`/`float()` 参数转换无 try/except。

#### [中] 参数范围检查不全面（原 A2-图像部分）
- **文件**：`text_to_image.py`、`image_to_image.py`、`inpainting.py`、`upscaler.py`、`stylizer.py`
- **描述**：`width`/`height` 对齐到 8 的倍数但无上限（可传入 999999 导致 OOM）；`guidance_scale`/`num_inference_steps`/`strength` 均无范围校验。

#### [中] 空字符串 prompt 处理不一致（原 C1-图像部分）
- **文件**：`text_to_image.py` 113-118、`image_to_image.py` 126-131、`inpainting.py` 133-138
- **描述**：仅检查 `isinstance(prompt, str)`，未检查 `prompt.strip()`。

#### [中] 图像 mask 尺寸不匹配时静默 resize（原 C2）
- **文件**：`inpainting.py`
- **行号**：153
- **描述**：`mask_image = mask_image.resize(image.size)` 直接将 mask 强制 resize 到 image 尺寸，不做任何警告或校验，可能导致 inpainting 结果错误。

#### [中] Image.open 对 bytes 类型的潜在失败（原 C3）
- **文件**：`background_remover.py`
- **行号**：188
- **描述**：`Image.open(output_bytes) if hasattr(output_bytes, "read") else Image.open(output_bytes)` 两个分支调用相同代码，逻辑冗余。若 `output_bytes` 是 `bytes` 类型而非文件对象，`Image.open` 需要 `io.BytesIO` 包装。

#### [中] 无大图像内存保护（原 E3-图像部分）
- **文件**：`text_to_image.py` 125-129、`inpainting.py` 152、`stylizer.py` 206
- **描述**：`width`/`height` 无上限，用户可传入超大尺寸导致 OOM。仅 `upscaler.py` 有 `_limit_image_size` 限制。

#### [低] 必需参数缺失时的错误提示不一致（原 A3）
- **文件**：`inpainting.py` 130、`background_remover.py` 152、`image_to_image.py` 123、`upscaler.py` 123
- **描述**：错误提示未包含实际接收到的类型信息，如 `"Inpainting requires 'mask_image' (PIL.Image)."` 未说明收到的是什么类型。

#### [低] 输出图像为 None 时的处理不一致（原 C4）
- **文件**：`image_to_image.py` 170/182、`inpainting.py` 178/189、`stylizer.py` 238/253
- **描述**：`result_image` 可能为 None，但后续 `result_image.size` 会抛出 `AttributeError`。

#### [低] 硬编码魔法数字（原 F2-图像部分）
- **文件**：`background_remover.py` 207、`stylizer.py` 226、`text_to_image.py` 128-129
- **描述**：如 `input_size = (1024, 1024)`、`set_ip_adapter_scale(0.6)` 等魔法数字。

#### [低] 冗余条件判断（原 F3-图像部分）
- **文件**：`background_remover.py` 188
- **描述**：`Image.open(output_bytes) if hasattr(...) else Image.open(output_bytes)` 两个分支完全相同。

---

### 2.3 视频节点（10 项）

> 模块路径：`mosaic/nodes/video/`
> 涉及文件：`text_to_video.py`、`hunyuan_video.py`、`ltx_video.py`、`wan_video.py`、`image_to_video.py`、`video_continuation.py`、`frame_interpolation.py`

#### [高] arr.max() 对空数组的潜在崩溃（原 E1）
- **文件**：`text_to_video.py` 290、`hunyuan_video.py` 206、`ltx_video.py` ~185、`wan_video.py` 233
- **描述**：`if arr.max() <= 1.0:` 若 `arr` 为空数组（shape[0]==0），`arr.max()` 会抛出 `ValueError: zero-size array to reduction operation maximum`。

#### [高] HunyuanVideo 缺少 num_frames 校验/调整（原 E2）
- **文件**：`hunyuan_video.py`
- **行号**：253
- **描述**：无 `_adjust_num_frames` 方法（CogVideoX 节点有），传入不支持的帧数可能导致模型推理失败。

#### [中] int()/float() 类型转换无异常处理（原 A1-视频部分）
- **文件**：`text_to_video.py` 345-355、`hunyuan_video.py` 253-264、`ltx_video.py` 238-249、`wan_video.py` 287-309、`image_to_video.py` 326-336
- **描述**：同文本/图像节点，`int()`/`float()` 参数转换无 try/except。

#### [中] 参数范围检查不全面（原 A2-视频部分）
- **文件**：`text_to_video.py`、`hunyuan_video.py`、`ltx_video.py`、`wan_video.py`、`image_to_video.py`
- **描述**：`num_frames`/`fps` 无范围校验；`guidance_scale`/`num_inference_steps` 无上下限。传入不合理值可能导致模型推理失败或 OOM。

#### [中] 模型名称大小写敏感匹配（原 B1）
- **文件**：`ltx_video.py` 282、`wan_video.py` 342
- **描述**：`"13B" in self._model_name` 和 `"1.3B" in self._model_name` 大小写敏感，若模型名为 `"ltx-13b"` 或 `"LTX-13B"` 等变体则匹配失败，导致 OOM 提示中显存估算错误。

#### [中] 模型路径不存在时无友好校验（原 B2-视频部分）
- **文件**：`text_to_video.py`、`hunyuan_video.py`、`ltx_video.py`、`wan_video.py`
- **描述**：除 `frame_interpolation.py` 有 `os.path.exists` 检查外，其他视频节点均无本地路径存在性预检查，直接调用 `from_pretrained` 失败时报错信息不够友好。

#### [中] 视频帧数边界条件（原 E4）
- **文件**：`image_to_video.py` 328、`frame_interpolation.py` 265-266
- **描述**：SVD 允许 1 帧"视频"；frame_interpolation 未检查输入视频是否至少有 2 帧（插帧需要至少 2 帧）。

#### [低] 视频节点间大量重复代码（原 F1）
- **文件**：`text_to_video.py` 与 `video_continuation.py`；`hunyuan_video.py`、`ltx_video.py`、`wan_video.py`
- **描述**：`_adjust_num_frames`、`_prepare_seed`、`_extract_frames_from_output` 三个方法在多个文件中几乎完全重复。应提取到 `BaseVideoNode` 或公共工具模块。

#### [低] 冗余条件判断（原 F3-视频部分）
- **文件**：`wan_video.py` 132
- **描述**：`if "Wan-AI/" in name or name.startswith("Wan-AI/"):` 两个条件等价，`or` 逻辑冗余。

#### [低] 日志信息不充分（原 F5-视频部分）
- **文件**：`frame_interpolation.py` 180-186
- **描述**：RIFE 模型不可用时回退到 linear，但日志未记录原始请求的 method 参数，不利于排查问题。

---

### 2.4 数字人节点（5 项）

> 模块路径：`mosaic/nodes/digital_human/`
> 涉及文件：`avatar_driver.py`、`lip_syncer.py`、`motion_generator.py`、`realtime_renderer.py`

#### [低] 硬编码魔法数字（原 F2-数字人部分）
- **文件**：`avatar_driver.py` 772/796/797、`lip_syncer.py` 749/767-773、`realtime_renderer.py` 765-766/778
- **描述**：如 `energy * 3.0`、`mouth_open_proxy / 0.5`、`mouth_width / eye_dist - 0.4` 等魔法数字。

#### [低] 异常处理过于宽泛（原 F4-数字人部分）
- **文件**：`realtime_renderer.py` 272/296/336/348/357/647、`lip_syncer.py` 279
- **描述**：多处 `except Exception` 捕获所有异常，可能掩盖编程错误。

#### [低] 日志信息不充分（原 F5-数字人部分）
- **文件**：`realtime_renderer.py` 314-320
- **描述**：ONNX 不可用时仅 debug 级别日志。

#### [低] motion_generator.py _walk 函数逻辑错误（原 E5）
- **文件**：`motion_generator.py`
- **行号**：216
- **描述**：`hasattr(lift, "__getitem__")` 检查在 numpy 数组上恒为 True，使得 else 分支永远不会执行，属于死代码。

#### [低] 模型路径不存在时无友好校验（原 B2-数字人部分）
- **文件**：`lip_syncer.py` 224-234
- **描述**：Wav2Lip 和 SadTalker 路径无预检查。

---

### 2.5 一致性节点（5 项）

> 模块路径：`mosaic/nodes/consistency/`
> 涉及文件：`cross_frame_consistency.py`、`identity_keeper.py`、`style_keeper.py`

#### [中] 设备参数处理不一致（原 B3）
- **文件**：`cross_frame_consistency.py`
- **行号**：185-193
- **描述**：`_resolve_target_device` 检查 `self._scheduler.is_gpu`，但其他 consistency 节点使用基类的 `_resolve_device`，设备降级逻辑不统一。

#### [中] 无大图像内存保护（原 E3-一致性部分）
- **文件**：`cross_frame_consistency.py` 637-641、`identity_keeper.py` 372-375、`style_keeper.py` 493-496
- **描述**：`width`/`height` 无上限，用户可传入超大尺寸导致 OOM。

#### [中] CrossFrameConsistency 的 ensure_loaded 调用时机不一致（原 F7）
- **文件**：`cross_frame_consistency.py`
- **行号**：604-630
- **描述**：`run()` 方法在输入校验之后才调用 `ensure_loaded`，而其他节点在 `run()` 开头就调用。

#### [低] 异常处理过于宽泛（原 F4-一致性部分）
- **文件**：`cross_frame_consistency.py` 441/451
- **描述**：多处 `except Exception` 捕获所有异常，可能掩盖编程错误。

#### [低] consistency 节点未调用 _emit_progress（原 F6）
- **文件**：`cross_frame_consistency.py` 687-700、`identity_keeper.py`、`style_keeper.py`
- **描述**：长序列生成时用户无法获知进度。

---

## 第三部分：核心框架模块

### 3.1 调度器（7 项）

> 模块路径：`mosaic/core/scheduler.py`

#### [中] LRU 淘汰逻辑中 exclude 分支的 rotate 破坏 LRU 语义（原 A1）
- **文件**：`scheduler.py`
- **行号**：288-296
- **描述**：当 `victim == exclude` 时调用 `self._lru.rotate(1)`，会将队首元素移到队尾，错误地修改 LRU 访问顺序。`all(n == exclude for n in self._lru)` 的 O(n) 检查在每次循环中都执行，效率低下。

#### [中] 显存估算与实际使用脱节（原 A2）
- **文件**：`scheduler.py`
- **行号**：214-226, 312-319
- **描述**：`_estimate_memory` 完全依赖静态估算值，不反映实际推理时的显存占用（随 batch_size、分辨率、步数动态变化）。`_query_used_memory()` 仅在 `status()` 中展示，未参与容量决策。

#### [中] 多 GPU 支持缺失，显存查询硬编码 device 0（原 A3）
- **文件**：`scheduler.py`
- **行号**：144, 155
- **描述**：`torch.cuda.get_device_properties(0)` 和 `torch.cuda.memory_allocated()` 硬编码 device 0，即使用户传入 `device="cuda:1"` 也只查 device 0。

#### [中] 持锁加载导致性能瓶颈（原 A4）
- **文件**：`scheduler.py`
- **行号**：247-266, 321-336
- **描述**：`ensure_loaded` 在 `with self._lock` 内调用 `node.load()`，后者可能耗时数十秒甚至数分钟。由于持锁，其他线程的调用全部阻塞。

#### [中] _do_load 中 node.load() 失败后状态不一致（原 A6）
- **文件**：`scheduler.py`
- **行号**：321-336
- **描述**：`node.load()` 可能部分成功（如已下载权重但设备迁移失败），节点内部 `_loaded` 已设为 `True`，而调度器的 `_loaded_names` 未更新。后续 `ensure_loaded` 检查 `node.is_loaded()` 返回 `True`，跳过重新加载，但模型实际不可用。

#### [低] release 方法中的冗余赋值（原 A5）
- **文件**：`scheduler.py`
- **行号**：359-362
- **描述**：`if self._is_gpu: freed = freed else: freed = 0.0` — `freed = freed` 是空操作。

#### [低] 全局单例 get_scheduler 不可重置（原 A7）
- **文件**：`scheduler.py`
- **行号**：472-490
- **描述**：没有类似 `EventBus._reset_singleton` 的清理机制，测试间隔离不完整。

---

### 3.2 注册表（6 项）

> 模块路径：`mosaic/core/registry.py`

#### [中] NodeRegistry 完全无锁，非线程安全（原 B1）
- **文件**：`registry.py`
- **行号**：41-44, 47-99, 113-144
- **描述**：`_nodes`/`_instances`/`_scanned` 均无锁保护。多线程插件加载或并行管道中动态注册节点时，`register`/`get`/`discover` 的并发访问会导致竞态条件。

#### [中] discover 静默吞掉所有模块导入异常（原 B3）
- **文件**：`registry.py`
- **行号**：238-242
- **描述**：`except Exception: continue` 静默跳过导入失败的模块，不记录任何日志。用户无法得知某节点被跳过。

#### [中] _safe_describe 实例化节点类可能产生副作用（原 B4）
- **文件**：`registry.py`
- **行号**：267-284
- **描述**：`_safe_describe` 调用 `node_class()` 实例化节点以获取 `describe()`，但 `__init__` 会调用 `get_scheduler()` 触发 GPU 检测。`list_nodes`/`list_domains` 每次调用都会实例化所有节点类。

#### [低] discover 的预检查与 register 之间存在 TOCTOU 竞态（原 B2）
- **文件**：`registry.py`
- **行号**：257-259
- **描述**：`discover` 在调用 `register` 前检查 `if attr_value.name not in self._nodes`，但二者之间状态可能变化。

#### [低] 类名别名注册可能导致名称冲突（原 B5）
- **文件**：`registry.py`
- **行号**：93-96
- **描述**：`register` 以类名 `__name__` 作为别名注册，若类名与另一个节点的 `name` 属性相同，别名注册被跳过且无提示。

#### [低] unregister 不清理已缓存实例的类名别名（原 B6）
- **文件**：`registry.py`
- **行号**：102-110
- **描述**：边界情况下的清理不彻底。

---

### 3.3 类型系统（7 项）

> 模块路径：`mosaic/core/types.py`

#### [高] MosaicData.__eq__ 对含 numpy 数组的行为异常（原 C1）
- **文件**：`types.py`
- **行号**：197-202
- **描述**：`__eq__` 执行 `self._data == other._data`，当 `_data` 中包含 numpy 数组时，`bool(array([True, True]))` 会抛 `ValueError: The truth value of an array is ambiguous`。`AudioData`（含 waveform 数组）、`MotionData`（含 keypoints 数组）均受影响。

#### [中] from_dict 对未知 data_type 静默降级为基类（原 C2）
- **文件**：`types.py`
- **行号**：238-252
- **描述**：若 `dtype` 不在注册表中，回退到 `MosaicData`，丢失原始子类类型信息，无任何警告。

#### [中] ImageData.__init__ 对非 PIL 输入的 size 推断错误（原 C3）
- **文件**：`types.py`
- **行号**：349-350
- **描述**：若 `image` 是 numpy 数组，`image.size` 返回元素总数（整数），而非 `(width, height)` 元组。

#### [中] AudioData 的 dtype 转换不做振幅归一化（原 C4）
- **文件**：`types.py`
- **行号**：417-430
- **描述**：`uint8→float32` 会产生 0-255 的值而非 [-1, 1]，下游代码若假设 [-1, 1] 范围会产生错误。注释说"归一化"但实际只做类型转换。

#### [中] validate 方法从未在管道执行中被调用（原 C6）
- **文件**：`types.py`
- **行号**：254-261 及各子类
- **描述**：各数据类型定义了 `validate` 类方法，但管道 `execute`/`execute_result` 从未调用它们，校验逻辑形同虚设。

#### [中] VideoData 帧序列化效率极低（原 C7）
- **文件**：`types.py`
- **行号**：109-139
- **描述**：`frames` 是 PIL.Image 列表，`to_dict` 会逐帧 base64 编码为 PNG。对于 100+ 帧的视频，极其缓慢且产生巨大 JSON 字符串。

#### [低] _b64_to_image 返回的 PIL 图像是懒加载的（原 C5）
- **文件**：`types.py`
- **行号**：78-85
- **描述**：`Image.open(io.BytesIO(raw))` 返回懒加载图像，数据在首次访问时才读取。建议调用 `.load()` 或 `.copy()` 立即加载。

---

### 3.4 管道与节点（10 项）

> 模块路径：`mosaic/core/pipeline.py`、`mosaic/core/node.py`、`mosaic/core/branch.py`

#### [高] 管道直接调用 node.load() 绕过调度器容量检查（原 D1）
- **文件**：`pipeline.py`
- **行号**：841-842, 895-896
- **描述**：`_execute_serial` 和 `_run_single_node` 在 `run` 前调用 `dn.node.load()`。所有节点的 `run` 方法内部又调用 `self._scheduler.ensure_loaded(self)`，但 `ensure_loaded` 检查 `node.is_loaded()` 返回 `True`（因 `load()` 已执行），直接跳过 `_ensure_capacity` 容量检查。结果：调度器的 LRU 淘汰和显存容量管理在管道执行路径中完全失效，多个大模型可同时加载导致 OOM。

#### [中] 并行执行中 fail_fast=True 时取消机制不完善（原 D2）
- **文件**：`pipeline.py`
- **行号**：794-798, 807-808
- **描述**：`f.cancel()` 只能取消尚未启动的 future，已在运行的线程无法中断。`executor.shutdown(wait=True)` 会等待所有运行中的线程完成。节点无协作式取消检查机制。

#### [中] _ConditionalNode 加载所有候选路径，浪费显存（原 D3）
- **文件**：`pipeline.py`
- **行号**：115-119
- **描述**：`_ConditionalNode.load()` 加载所有 `self._paths` 中的子管道，但运行时只执行其中一条，可能浪费大量显存。

#### [中] Node 类属性 input_types/output_types 为可变共享列表（原 D5）
- **文件**：`node.py`
- **行号**：136-137
- **描述**：`input_types: list[str] = []` 和 `output_types: list[str] = []` 是类级别的可变默认值。所有未覆写的子类实例共享同一列表对象。

#### [中] Node.__init__ 静默忽略未知 kwargs（原 D6）
- **文件**：`node.py`
- **行号**：139-143
- **描述**：不在类属性中的 kwargs 被静默丢弃。用户拼错参数名（如 `devce="cuda"`）不会报错。

#### [中] Branch.input_strategy="distribute" 未在 Pipeline 执行中实现（原 G16）
- **文件**：`branch.py` 53-59, `pipeline.py` 399-423/942-964
- **描述**：`Branch` 支持 `input_strategy="distribute"`，但 `_add_branch` 和 `_gather_input` 中无任何处理逻辑，`distribute` 策略被静默忽略。这是已声明但未实现的功能。

#### [低] _gather_input 多前驱合并时标签冲突无检测（原 D4）
- **文件**：`pipeline.py`
- **行号**：959-964
- **描述**：fan-in 合并时若两个前驱有相同标签，后者覆盖前者，无任何警告。

#### [低] dry_run 类型匹配过于宽松（原 D7）
- **文件**：`pipeline.py`
- **行号**：517-533
- **描述**：只要有交集就通过，空列表视为"接受任意类型"，无法检测语义不匹配。

#### [低] Pipeline 类的 input_types/output_types 与 describe 不一致（原 G14）
- **文件**：`pipeline.py`
- **行号**：241-242, 311-332
- **描述**：`describe()` 动态从 DAG 收集类型，但 `accepts()`/`produces()` 返回空列表 `[]`，嵌套管道的类型契约检查可能失效。

#### [低] Pipeline.__or__/__ror__ 创建的匿名管道丢失上下文（原 G15）
- **文件**：`pipeline.py`
- **行号**：335-347
- **描述**：新管道与原管道共享同一节点实例，若两个管道并发执行，共享节点的状态可能冲突。

---

### 3.5 事件总线与异步任务（6 项）

> 模块路径：`mosaic/core/events.py`、`mosaic/core/task.py`、`mosaic/core/context.py`

#### [高] AsyncTask 使用 threading.Lock（非 RLock）导致回调中死锁（原 G2）
- **文件**：`task.py`
- **行号**：153, 334-339, 349-355
- **描述**：`on_complete`/`on_error` 在 `with self._lock` 内同步调用回调，若回调内部访问 `task.result()`/`task.status` 等属性（也获取 `self._lock`），会因 `Lock` 不可重入而死锁。

#### [中] AsyncTask._run 未捕获 BaseException（原 G3）
- **文件**：`task.py`
- **行号**：435-505
- **描述**：`except Exception as exc` 不捕获 `KeyboardInterrupt`/`SystemExit`。若工作线程被中断，`_done_event` 永远不会被 set，`wait()` 会永久阻塞。

#### [中] Context.snapshot/load_snapshot 不对称（原 G5）
- **文件**：`context.py`
- **行号**：357-412
- **描述**：`snapshot()` 保存 `config` 和 `artifacts`，但 `load_snapshot()` 只恢复 `artifacts`，不恢复 `config`。快照的 round-trip 不完整。

#### [低] EventBus 单例不可重置，测试间状态泄漏（原 G1）
- **文件**：`events.py`
- **行号**：139-149, 344-355
- **描述**：`shutdown()` 不清除 `_subscribers`，跨测试用例的订阅者会互相干扰。

#### [低] Context.__enter__ 将 RunConfig.__dict__ 直接放入事件 payload（原 G4）
- **文件**：`context.py`
- **行号**：196
- **描述**：传入的是 `__dict__` 引用而非拷贝，若 config 后续被修改，已发出的事件 payload 会随之变化。

#### [低] Context.emit 回调异常被 continue 静默吞掉（原 G6）
- **文件**：`context.py`
- **行号**：229-239
- **描述**：`except Exception: continue` 不记录任何日志。

---

### 3.6 模型缓存与插件（5 项）

> 模块路径：`mosaic/core/model_cache.py`、`mosaic/core/plugin.py`

#### [中] ModelCache 缓存键不含设备信息（原 G7）
- **文件**：`model_cache.py`
- **行号**：36-41
- **描述**：缓存键为 `(class_name, model_name, dtype)`，不含 device。同一模型在 `cuda:0` 和 `cuda:1` 上加载会命中同一缓存条目，返回错误设备上的模型实例。

#### [中] ModelCache 无容量限制和淘汰策略（原 G8）
- **文件**：`model_cache.py`
- **行号**：31-34
- **描述**：`_cache` 字典无大小限制，持续缓存所有加载过的模型，长时间运行的服务会因缓存过多模型导致显存泄漏。

#### [低] PluginManager._scan_directory 模块名生成不健壮（原 G9）
- **文件**：`plugin.py`
- **行号**：283-287
- **描述**：若文件名含空格、连字符等非 Python 标识符字符，生成的模块名无效。

#### [低] PluginManager 的 _loaded 标志阻止重新加载（原 G10）
- **文件**：`plugin.py`
- **行号**：205-207
- **描述**：`load_plugins()` 检查 `if self._loaded: return 0`，且无公开方法重置此标志。

#### [低] 跨平台路径分隔符处理不完整（原 G11）
- **文件**：`plugin.py` 284-287, `doctor.py` 244
- **描述**：Windows 上若路径中混合使用 `/` 和 `\\`，`replace(os.sep, ".")` 只替换 `\\` 不替换 `/`。

---

## 第四部分：CLI 与公共工具

### 4.1 CLI（7 项）

> 模块路径：`mosaic/cli/`
> 涉及文件：`main.py`、`doctor.py`

#### [高] create-node 命令参数式调用完全不可用（原 F1）
- **文件**：`main.py`
- **行号**：437-444
- **描述**：`_cmd_create_node` 构造 `NodeGenerator(domain=..., name=..., ...)`，但 `NodeGenerator.__init__(self)` 不接受任何参数。此调用会抛 `TypeError`。即 `mosaic create-node --name foo --domain text` 命令必定失败。

#### [中] run 命令中 node.name = node_alias 修改实例属性可能干扰调度器（原 F2）
- **文件**：`main.py`
- **行号**：559-560
- **描述**：调度器以 `node.name` 为键跟踪节点。若两个节点使用相同别名，调度器中后者会覆盖前者的跟踪记录。

#### [中] doctor 命令退出码不反映错误（原 F3）
- **文件**：`doctor.py`
- **行号**：257-338
- **描述**：即使有 `error` 级别问题（如 Python 版本过低、必需依赖缺失），退出码仍为 0。`mosaic doctor && echo OK` 会在环境不达标时仍输出 OK。

#### [低] doctor 模型缓存检查不完整（原 F4）
- **文件**：`doctor.py`
- **行号**：240-251
- **描述**：只检查 `HF_HOME` 和 `~/.cache/huggingface`，未检查 `TRANSFORMERS_CACHE`、`HF_HUB_CACHE` 等。

#### [低] _load_pipeline_file 自动检测逻辑可能误判（原 F5）
- **文件**：`main.py`
- **行号**：224-229
- **描述**：未知扩展名时"先尝试 JSON，失败则尝试 YAML"，依赖异常控制流。

#### [低] _format_table 截断逻辑在小宽度时异常（原 F6）
- **文件**：`main.py`
- **行号**：87-88
- **描述**：`text[: max_widths[i] - 3] + "..."`，若 `max_widths[i] < 3`，负索引产生意外结果。

#### [低] create-node 交互模式中 input() 无编码处理（原 F7）
- **文件**：`main.py`
- **行号**：448-462
- **描述**：Windows 控制台（默认 GBK 编码）下，`input()` 可能因编码问题抛 `UnicodeDecodeError`。

---

### 4.2 公共工具（4 项）

> 模块路径：`mosaic/nodes/_pipeline_utils.py`

#### [中] fp16 variant 回退时丢失第一次失败的错误信息（原 E1）
- **文件**：`_pipeline_utils.py`
- **行号**：222-243
- **描述**：fp16 variant 加载失败后 `except (OSError, ValueError, EnvironmentError) as exc: pass` 异常被完全丢弃。第二次尝试若也失败，用户不知道 fp16 尝试也失败了。

#### [中] fp16 回退的异常捕获范围不完整（原 E2）
- **文件**：`_pipeline_utils.py`
- **行号**：229
- **描述**：第一次尝试只捕获 `(OSError, ValueError, EnvironmentError)`，若 `from_pretrained` 抛出 `RuntimeError`（如 CUDA OOM），不会触发 fp32 回退。`EnvironmentError` 在 Python 3 中是 `OSError` 的别名，捕获二者冗余。

#### [中] safe_load_model 的 dtype 回退逻辑可能掩盖真正的错误（原 E3）
- **文件**：`_pipeline_utils.py`
- **行号**：311-321
- **描述**：`TypeError` 也可能由其他原因引起，此时回退到 `torch_dtype=` 可能掩盖根因。

#### [低] _preimport_t5_components 对 T5Tokenizer 缺失的判断逻辑脆弱（原 E4）
- **文件**：`_pipeline_utils.py`
- **行号**：40-59
- **描述**：transformers 的 `_LazyModule` 机制下，`hasattr` 可能触发实际导入并抛 `ImportError`。

---

## 第五部分：全局问题（2 项）

> 跨模块问题，不局限于单一目录

#### [低] 日志配置缺失，无统一初始化（原 G12）
- **文件**：整个项目
- **描述**：各模块使用 `logging.getLogger("mosaic.xxx")`，但项目中无统一的日志配置。关键信息（如 LRU 淘汰、模块导入失败等）可能被静默丢弃。

#### [低] 环境变量处理不集中，散落各处（原 G13）
- **文件**：`doctor.py:242` (`HF_HOME`)、`scheduler.py`（无 env var 支持）、各节点
- **描述**：无集中管理，如 `MOSAIC_MEMORY_LIMIT`、`CUDA_VISIBLE_DEVICES` 等未支持。

---

## 第六部分：优先修复建议（Top 10）

以下 10 个问题建议优先修复，按影响范围和严重程度排序：

| 优先级 | 模块 | 问题编号 | 描述 | 影响 |
|--------|------|---------|------|------|
| 1 | 管道与节点 | 3.4-D1 | 管道直接调用 `node.load()` 绕过调度器容量检查 | 多个大模型可同时加载导致 OOM |
| 2 | TTS 节点 | 1.4-B1-3 | 长文本拆分仅对 edge/transformers 生效，扩展后端不拆分 | ChatTTS/Fish/SoVITS/CosyVoice 长文本合成不可用 |
| 3 | 文本/图像/视频节点 | 2.1~2.3-A1 | 所有节点的 int()/float() 参数转换无异常处理 | 传入非数字字符串产生难以理解的 ValueError |
| 4 | CLI | 4.1-F1 | create-node 命令参数式调用完全不可用 | CLI 命令必定失败 |
| 5 | 类型系统 | 3.3-C1 | MosaicData.__eq__ 对含 numpy 数组的行为异常 | 含数组数据的比较操作抛异常 |
| 6 | 事件/任务 | 3.5-G2 | AsyncTask 使用 threading.Lock 导致回调中死锁 | 回调中访问 task 属性会死锁 |
| 7 | TTS 后端实现 | 1.2-E4-2 | CosyVoice LLM 加载失败后用 token_ids 直接当 text_feats | 产出垃圾音频却无任何错误抛出 |
| 8 | 视频节点 | 2.3-E1 | 视频节点 arr.max() 对空数组的潜在崩溃 | 完全不可恢复的错误 |
| 9 | TTS 节点 | 1.4-B1-1 | 句子分割正则误切缩写、小数、序号 | 英文场景严重缺陷 |
| 10 | 文本前端 | 1.1-A1-1 | ChatTTS preprocess 破坏内嵌韵律标记 | 韵律标记完全失效 |

---

## 附录：模块修复索引

以下表格列出每个模块的问题数量和建议修复顺序，便于按模块分派任务：

| 修复批次 | 模块 | 目录 | 问题数 | 高优先问题 |
|---------|------|------|--------|-----------|
| 第 1 批 | 管道与节点 | `core/pipeline.py` + `node.py` + `branch.py` | 10 | D1 (绕过调度器) |
| 第 1 批 | TTS 节点 | `nodes/audio/tts.py` | 6 | B1-1, B1-3 (拆分规则) |
| 第 1 批 | CLI | `cli/` | 7 | F1 (create-node 不可用) |
| 第 2 批 | 类型系统 | `core/types.py` | 7 | C1 (__eq__ 崩溃) |
| 第 2 批 | 事件/任务/上下文 | `core/events.py` + `task.py` + `context.py` | 6 | G2 (死锁) |
| 第 2 批 | TTS 后端实现 | `tts_backends/implementations/` | 12 | D1-1, C2-1, E4-2 |
| 第 3 批 | 文本前端 | `tts_backends/text_frontends/` | 13 | A1-1, B3-1 |
| 第 3 批 | 视频节点 | `nodes/video/` | 10 | E1, E2 |
| 第 3 批 | 图像节点 | `nodes/image/` | 10 | A1 |
| 第 4 批 | 调度器 | `core/scheduler.py` | 7 | — |
| 第 4 批 | 公共工具 | `nodes/_pipeline_utils.py` | 4 | — |
| 第 4 批 | 声学模型 | `tts_backends/acoustic_models/` | 5 | — |
| 第 5 批 | 文本节点 | `nodes/text/` | 7 | — |
| 第 5 批 | 注册表 | `core/registry.py` | 6 | — |
| 第 5 批 | 参考音频 | `nodes/audio/_ref_audio_utils.py` | 4 | — |
| 第 6 批 | 一致性节点 | `nodes/consistency/` | 5 | — |
| 第 6 批 | 数字人节点 | `nodes/digital_human/` | 5 | — |
| 第 6 批 | 模型缓存与插件 | `core/model_cache.py` + `plugin.py` | 5 | — |
| 第 6 批 | 流式输出 | `tts_backends/streaming/` | 2 | — |
| 第 6 批 | TTS 注册表 | `tts_backends/registry.py` | 1 | — |
| 第 6 批 | 全局问题 | 跨模块 | 2 | — |

---

*本文档仅记录问题，暂不修改代码。按模块分组，便于分派和跟踪修复进度。*
