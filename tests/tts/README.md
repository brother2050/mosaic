# tests/tts — TTS 子系统测试用例说明

本目录覆盖 mosaic TTS 框架的基础架构、文本前端、声学模型、声码器、
ChatTTS 后端、Fish Speech 后端与权重转换器。测试遵循以下原则：

- 不依赖真实预训练权重（小模型参数 / mock / 随机初始化）。
- 可选依赖缺失时用 `pytest.mark.skipif` 跳过。
- 测试函数统一以 `test_` 前缀命名。
- 当前环境已安装 `torch`、`safetensors`，未安装 `transformers`、`vocos`；
  因此依赖 `transformers` 的用例会跳过。

运行：

```bash
cd /workspace/mosaic && python -m pytest tests/tts/ -v --tb=short
```

共 149 个用例：137 通过 / 12 跳过（均为 `transformers` 缺失导致）。

---

## 公共 fixtures（conftest.py）

| Fixture | 说明 |
| --- | --- |
| `mock_text_frontend` | 简单 TextFrontend mock，不需要真实词表 |
| `mock_acoustic_model` | 简单 AcousticModel mock |
| `mock_vocoder` | 简单 Vocoder mock |
| `sample_token_ids` | 随机 token ids tensor `[1, 20]`（torch 可用） |
| `sample_mel` | 随机 mel spectrogram tensor `[1, 80, 50]` |
| `sample_audio_codes` | 随机音频码 tensor `[4, 50]` |

---

## test_tts_framework.py — TTS 框架基础架构

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_TTSFW_01 | TTSBackend 抽象类不能直接实例化 | 抛出 `TypeError` |
| test_TTSFW_02 | TextFrontend 抽象类不能直接实例化 | 抛出 `TypeError` |
| test_TTSFW_03 | AcousticModel 抽象类不能直接实例化 | 抛出 `TypeError` |
| test_TTSFW_04 | Vocoder 抽象类不能直接实例化 | 抛出 `TypeError` |
| test_TTSFW_05 | TTSBackendSpec 创建和字段 | 各字段（name、supported_languages、sample_rate 等）取值正确 |
| test_TTSFW_06 | TTSBackendRegistry 注册和获取 | `register` 后 `get` 返回同一类对象，`list_backends` 含其名称 |
| test_TTSFW_07 | TTSBackendRegistry auto_select 自动选择 | 按独占语言约束自动选中已注册后端 |
| test_TTSFW_08 | StreamAdapter 创建 StreamSession | `create_stream` 返回 `StreamSession` 实例 |
| test_TTSFW_09 | StreamSession push/pop 基本流程 | 推入一个 chunk 后 `pop` 返回 `AudioData`，形状/采样率正确 |
| test_TTSFW_10 | StreamSession overlap-add 平滑 | overlap 区域由前段平滑过渡到后段，区域外保持后段值 |
| test_TTSFW_11 | StreamSession flush 输出剩余 | 不足一个 chunk 时 `pop` 为 None，`flush` 输出剩余样本 |
| test_TTSFW_12 | StreamSession on_chunk_ready 回调触发 | 推入超过一个 chunk 时回调被触发至少一次 |

---

## test_chat_tokenizer.py — ChatTokenizer

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_CTKN_01 | 基本分词，输出 tensor | 返回 `torch.Tensor`，形状 `[1, seq_len]` |
| test_CTKN_02 | 中文文本分词正确 | 首尾为 `[Stts]`/`[Ptts]`，长度大于结构标记数 |
| test_CTKN_03 | 英文文本分词正确 | 首尾结构标记正确 |
| test_CTKN_04 | 中英混合文本分词 | 同时包含中英字符 token，结构标记正确 |
| test_CTKN_05 | 韵律标记插入 [laugh_0] | token 序列含 `[laugh_0]`（id=4） |
| test_CTKN_06 | 韵律标记插入 [break_4] | token 序列含 `[break_4]`（id=13） |
| test_CTKN_07 | 多个韵律标记同时使用 | `[oral_2]`/`[laugh_0]`/`[break_4]` 均出现 |
| test_CTKN_08 | preprocess 文本清洗 | 全角标点转半角、数字转中文 |
| test_CTKN_09 | 特殊标记 [Stts]/[Ptts]/[spk_emb] 正确添加 | 带 speaker 用 `[spk_emb]`，否则用 `[empty_spk]` |
| test_CTKN_10 | encode_speaker 编码和解码一致性 | 说话人张量经编码再解码后 shape 一致、数值近似 |
| test_CTKN_11 | encode_speaker(None) 返回 None | 返回 `None` |
| test_CTKN_12 | detokenize 与 tokenize 对称 | detokenize 结果含结构标记与原文字符 |

---

## test_llama_ar.py — LlamaARModel

> 整个模块在 `transformers` 缺失时跳过（按设计要求）。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_LAR_01 | 构造成功，属性正确 | `model_type='ar'`、`vocab_size>0`、`hidden_size` 正确 |
| test_LAR_02 | vocab_size 计算正确 | `num_text_tokens + num_audio_tokens * num_vq` |
| test_LAR_03 | model_type == "ar" | 为 `'ar'` |
| test_LAR_04 | 未加载时 get_input_embeddings 返回 None | 返回 `None` |
| test_LAR_05 | 未加载时 get_output_head 返回 None | 返回 `None` |
| test_LAR_06 | 未加载时 generate 抛出 RuntimeError | 抛出 `RuntimeError` |
| test_LAR_07 | 未加载时 generate_stream 抛出 RuntimeError | 抛出 `RuntimeError` |
| test_LAR_08 | unload_weights 安全调用（未加载时） | 不抛异常 |
| test_LAR_09 | DualEmbedding 类可导入 | 为 `type` |
| test_LAR_10 | hidden_size 属性正确 | 等于构造值 |
| test_LAR_11 | num_vq 属性正确 | `._num_vq` 等于构造值 |
| test_LAR_12 | 构造参数正确存储 | `_model_path`/`_num_text_tokens`/`_num_audio_tokens`/`_num_vq`/`_hidden_size`/`_num_layers` 均正确 |

---

## test_dvae.py — DVAEDecoder

> 依赖 `torch`。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_DVAE_01 | 构造成功 | 实例非空 |
| test_DVAE_02 | forward 输出 mel shape 正确 [mel_bins, frames] | 2D 输入 `[num_vq, frames]` → `[mel_bins, frames]` |
| test_DVAE_03 | forward batch 输入 [batch, mel_bins, frames] | 3D 输入 → `[batch, mel_bins, frames]` |
| test_DVAE_04 | forward_chunk 流式解码 | `reset_stream_buffer` 后 `forward_chunk` 输出 mel 维度正确 |
| test_DVAE_05 | 不同 num_vq 值的兼容性 | num_vq=2/8 均可构造与前向 |
| test_DVAE_06 | load_weights 不存在的路径处理 | 优雅降级：不抛异常，标记已加载且仍可前向（注：当前实现不抛异常） |

---

## test_vocos.py — VocosVocoder

> 依赖 `torch`；`vocos` 包未安装，本组用例不触发 vocos 包。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_VOCOS_01 | 构造成功，类属性正确 | `vocoder_type='vocos'`、`input_type='mel'` |
| test_VOCOS_02 | vocoder_type == "vocos" | 为 `'vocos'` |
| test_VOCOS_03 | input_type == "mel" | 为 `'mel'` |
| test_VOCOS_04 | sample_rate 正确 | 为 `24000` |
| test_VOCOS_05 | 未加载时 decode 抛出 RuntimeError | 抛出 `RuntimeError` |
| test_VOCOS_06 | get_mel_basis 返回正确形状 | 形状 `[n_mels, n_fft//2+1]` |

---

## test_chattts_backend.py — ChatTTSBackend

> test_CHTTS_01~08 不依赖 `transformers`；test_CHTTS_09~16 依赖 `transformers`，缺失时跳过。
> 加载用例经 monkeypatch 使用 2 层 / hidden=64 的极小 LlamaARModel，随机初始化即可验证编排逻辑。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_CHTTS_01 | 类属性正确 (name, spec) | `name='chattts'`，`spec` 为 `TTSBackendSpec` |
| test_CHTTS_02 | spec 字段完整 | supported_languages / supports_streaming / vocoder_type 等正确 |
| test_CHTTS_03 | spec.default_params 包含 temperature, top_p, top_k | 三个键均存在 |
| test_CHTTS_04 | spec.sample_rate == 24000 | 为 `24000` |
| test_CHTTS_05 | spec.min_gpu_memory_gb == 2.0 | 为 `2.0` |
| test_CHTTS_06 | spec.model_license == "CC BY-NC 4.0" | 为 `'CC BY-NC 4.0'` |
| test_CHTTS_07 | list_speakers 返回非空列表 | 列表非空 |
| test_CHTTS_08 | check_dependencies 返回 bool | 返回 `bool` |
| test_CHTTS_09 | load 成功 | `is_loaded` 为 `True` |
| test_CHTTS_10 | synthesize 返回 AudioData | 返回 `AudioData`，采样率 24000，波形非空 |
| test_CHTTS_11 | synthesize 空文本抛 ValueError | 抛出 `ValueError` |
| test_CHTTS_12 | synthesize_stream 产出 chunk | 至少一个 `AudioData` chunk |
| test_CHTTS_13 | describe 返回 spec | 返回 `TTSBackendSpec`，`name='chattts'` |
| test_CHTTS_14 | list_speakers（加载后） | 非空列表 |
| test_CHTTS_15 | unload 后 is_loaded 为 False | 卸载后 `is_loaded` 为 `False` |
| test_CHTTS_16 | 重复 load 幂等 | 已加载时再次 load 不报错，仍为已加载 |

---

## test_chattts_weight_converter.py — ChatTTSWeightConverter

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_CVT_01 | 构造成功 | 实例非空 |
| test_CVT_02 | COMPONENTS 包含四大组件 | 含 text_frontend / acoustic_model / vocoder / dvae |
| test_CVT_03 | GPT_TO_LLAMA_MAP 包含关键映射 | 含 embed_tokens / norm / lm_head 映射 |
| test_CVT_04 | list_formats 继承正常工作 | 目录含 `.safetensors` → `['safetensors_dir']`；不存在路径 → `[]` |
| test_CVT_05 | dry_run 不存在的路径抛出 FileNotFoundError | 抛出 `FileNotFoundError` |

---

## test_fish_tokenizer.py — FishTokenizer

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_FTKN_01 | 基本分词，输出 tensor | 返回 `torch.Tensor`，形状 `[1, seq_len]` |
| test_FTKN_02 | 中文文本分词正确 | 序列含 `<s>`(0) 和 `<audio>`(3) |
| test_FTKN_03 | 英文文本分词正确 | 序列含 `<en>`(7) 标记 |
| test_FTKN_04 | 日文文本分词 | 序列含 `<ja>`(8) 标记 |
| test_FTKN_05 | 中英混合文本分词 | 输出非空 |
| test_FTKN_06 | token ids 在正确范围内 | 所有 id >= 0 且 < vocab_size |
| test_FTKN_07 | 特殊标记正确添加 | 序列以 `<s>` 开头、`<audio>` 结尾 |
| test_FTKN_08 | 语言标记插入 | 不同 language 对应不同语言标记 |
| test_FTKN_09 | preprocess 文本清洗 | 控制字符已清除 |
| test_FTKN_10 | encode_speaker(None) 返回 None | 返回 `None` |
| test_FTKN_11 | encode_speaker(路径) 返回路径 | 返回路径字符串 |
| test_FTKN_12 | detokenize 对称性 | 文本部分可恢复 |
| test_FTKN_13 | voice clone 序列构造 | 含 `<clone>`(4) 标记 |

---

## test_fish_ar.py — FishLlamaARModel

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_FAR_01 | 构造成功 | 实例非空 |
| test_FAR_02 | unload_weights 安全调用 | 不抛异常 |
| test_FAR_03 | generate 未加载时抛 RuntimeError | 抛出 `RuntimeError` |
| test_FAR_04 | generate_stream 未加载时抛 RuntimeError | 抛出 `RuntimeError` |
| test_FAR_05 | UnifiedEmbedding 正确 | 为 `nn.Module`，forward shape 正确 |
| test_FAR_06 | vocab_size 计算 | text + audio == vocab_size |
| test_FAR_07 | model_type == "ar" | 为 `'ar'` |
| test_FAR_08 | 未加载时 get_input_embeddings 返回 None | 返回 `None` |
| test_FAR_09 | 未加载时 get_output_head 返回 None | 返回 `None` |
| test_FAR_10 | codec_type 属性正确 | 等于构造值 |
| test_FAR_11 | hidden_size 属性正确 | 等于构造值 |
| test_FAR_12 | UnifiedEmbedding forward shape | `[batch, seq, hidden]` |
| test_FAR_13 | LlamaARModelBase 是 AcousticModel 子类 | `issubclass` 为 True |
| test_FAR_14 | FishLlamaARModel 继承 LlamaARModelBase | `issubclass` 为 True |
| test_FAR_15 | ChatTTS LlamaARModel 仍继承 LlamaARModelBase | 回归测试 `issubclass` 为 True |

---

## test_vq_decoder.py — VQDecoder

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_VQD_01 | 基本前向推理 | mel shape 最后一维为 mel_bins |
| test_VQD_02 | 2D token ids 输入 | mel 维度为 80 |
| test_VQD_03 | 输出值合理 | 所有值有限 |
| test_VQD_04 | forward_chunk 流式解码 | 输出非空 |
| test_VQD_05 | 单码本输入 | 正常工作 |
| test_VQD_06 | 多码本输入 | 正常工作 |
| test_VQD_07 | load_weights 优雅降级 | 不崩溃 |
| test_VQD_08 | reset_stream_buffer 安全调用 | 不抛异常 |

---

## test_hifi_gan.py — HiFiGanVocoder

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_HFG_01 | 构造成功，类属性正确 | `vocoder_type='hifi_gan'` |
| test_HFG_02 | sample_rate 正确 | 为 `22050` |
| test_HFG_03 | 未加载时 decode 抛 RuntimeError | 抛出 `RuntimeError` |
| test_HFG_04 | 未加载时 decode_chunk 抛 RuntimeError | 抛出 `RuntimeError` |
| test_HFG_05 | decode 返回元组 | `(waveform, sample_rate)` |
| test_HFG_06 | decode waveform shape 正确 | 为 tensor |
| test_HFG_07 | decode_chunk 流式解码 | 返回元组 |
| test_HFG_08 | get_mel_basis 形状正确 | `[80, 513]` |
| test_HFG_09 | unload_weights 安全调用 | 不抛异常 |
| test_HFG_10 | 不同 n_mels 兼容 | n_mels=128 正常工作 |

---

## test_fish_backend.py — FishSpeechBackend

> test_FISH_01~14 不依赖 `transformers`；test_FISH_15~18 依赖 `transformers`，缺失时跳过。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_FISH_01 | 类属性正确 | `name='fish'` |
| test_FISH_02 | spec 字段完整 | supported_languages / vocoder_type 等正确 |
| test_FISH_03 | default_params 包含五个参数 | temperature / top_p / top_k / repetition_penalty / max_new_tokens |
| test_FISH_04 | spec.sample_rate == 22050 | 为 `22050` |
| test_FISH_05 | spec.min_gpu_memory_gb == 3.0 | 为 `3.0` |
| test_FISH_06 | spec.model_license == "Apache-2.0" | 为 `'Apache-2.0'` |
| test_FISH_07 | list_speakers 返回非空列表 | 列表非空 |
| test_FISH_08 | list_speakers 返回列表类型 | 为 `list` |
| test_FISH_09 | check_dependencies 返回 bool | 返回 `bool` |
| test_FISH_10 | 构造成功 | 不报错 |
| test_FISH_11 | 构造参数正确存储 | model_path / codec_type / language |
| test_FISH_12 | is_loaded 初始为 False | 为 `False` |
| test_FISH_13 | _CompositeVocoder 类属性正确 | vocoder_type / input_type / sample_rate |
| test_FISH_14 | describe 返回正确信息 | spec.name == 'fish' |
| test_FISH_15 | load 成功 | `is_loaded` 为 `True` |
| test_FISH_16 | synthesize 返回 AudioData | 采样率 22050，波形非空 |
| test_FISH_17 | unload 后 is_loaded 为 False | 卸载后为 `False` |
| test_FISH_18 | describe（加载后） | 返回正确 spec |

---

## test_fish_weight_converter.py — FishWeightConverter

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_FCVT_01 | 构造成功 | 实例非空 |
| test_FCVT_02 | COMPONENTS 包含五大组件 | 含 vq_decoder / audio_encoder |
| test_FCVT_03 | FISH_TO_LLAMA_MAP 包含关键映射 | 含 embed_tokens |
| test_FCVT_04 | list_formats 继承正常 | 返回 list |
| test_FCVT_05 | dry_run 不存在路径抛异常 | 抛出 `FileNotFoundError` 或 `OSError` |

---

## test_chat_fish_regression.py — ChatTTS 与 Fish 共存回归测试

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_REGR_01 | ChatTTS 类属性不变 | name='chattts', sample_rate=24000 |
| test_REGR_02 | Fish 类属性正确 | name='fish', sample_rate=22050 |
| test_REGR_03 | 两后端采样率不同 | 24000 != 22050 |
| test_REGR_04 | 两后端声码器类型不同 | vocos != hifi_gan |
| test_REGR_05 | LlamaARModel 继承 LlamaARModelBase | `issubclass` 为 True |
| test_REGR_06 | FishLlamaARModel 继承 LlamaARModelBase | `issubclass` 为 True |
| test_REGR_07 | 两 AR 模型 model_type == "ar" | 均为 `'ar'` |
| test_REGR_08 | Vocos 与 HiFiGAN 互不干扰 | vocoder_type 不同 |
| test_REGR_09 | 两后端可独立实例化 | 均不报错 |
| test_REGR_10 | Registry 同时包含 chattts 和 fish | 延迟注册后两者均在 |
| test_REGR_11 | TTS 节点 list_backends 包含基础后端 | 含 edge_tts / transformers |
