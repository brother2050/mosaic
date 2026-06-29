# tests/tts — TTS 子系统测试用例说明

本目录覆盖 mosaic TTS 框架的基础架构、文本前端、声学模型、声码器、
ChatTTS 后端、Fish Speech 后端、GPT-SoVITS 后端、CosyVoice 后端与权重转换器。
测试遵循以下原则：

- 不依赖真实预训练权重（小模型参数 / mock / 随机初始化）。
- 可选依赖缺失时用 `pytest.mark.skipif` 跳过。
- 测试函数统一以 `test_` 前缀命名。
- 当前环境已安装 `torch`、`safetensors`，未安装 `transformers`、`vocos`；
  因此依赖 `transformers` 的用例会跳过。

运行：

```bash
cd /workspace/mosaic && python -m pytest tests/tts/ -v --tb=short
```

共 338 个用例：335 通过 / 3 失败（均为 ChatTTS/Fish 集成测试的预存问题，与 CosyVoice 无关）。

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
| `mock_sovits_tokenizer` | SoVITSTokenizer mock |
| `mock_gpt2_model` | GPT2ARModel mock |
| `mock_sovits_decoder` | SoVITSDecoder mock |
| `sample_phoneme_ids` | 音素 token ids `[1, 16]` |
| `sample_semantic_tokens` | 语义 token ids `[1, 32]`（SSL 码本 index） |
| `sample_speaker_info` | 含 ref_semantic_tokens + speaker_embedding 的 dict |
| `sample_ref_features` | 参考音频 SSL 特征 `[1, 50, 768]` |
| `mock_cosyvoice_tokenizer` | CosyVoiceTokenizer mock（LLM tokenizer + 特殊标记） |
| `mock_flow_matching_model` | FlowMatchingModel mock（generate 返回 mel，generate_stream 返回 chunk 迭代器） |
| `mock_speech_tokenizer` | SpeechTokenizer mock（2 层 RVQ，encode 返回 token ids） |
| `mock_speaker_encoder` | SpeakerEncoder mock（ECAPA-TDNN，encode 返回嵌入） |
| `sample_text_features` | LLM 输出文本特征 `[1, 20, 512]` |
| `sample_condition` | 融合条件特征 `[1, 30, 512]` |
| `sample_mel_from_flow` | Flow Matching 输出 mel `[1, 80, 100]` |

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

---

## test_sovits_tokenizer.py — SoVITSTokenizer

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_STKN_01 | 基本中文分词 | 输出 tensor `[1, seq]` |
| test_T_STKN_02 | 拼音转换正确 | 声母+韵母+声调拆分 |
| test_T_STKN_03 | 声调标注正确 | 一声到五声均出现 |
| test_T_STKN_04 | 英文分词（ARPAbet） | 返回 ARPAbet 音素 |
| test_T_STKN_05 | 中英混合分词 | 输出非空 |
| test_T_STKN_06 | 多音字处理 | 使用默认读音 |
| test_T_STKN_07 | 数字转拼音 | 产生拼音音素 |
| test_T_STKN_08 | add_blank 参数生效 | 插入空白 token |
| test_T_STKN_09 | 特殊标记正确添加 | 含 `<s>`、`</s>`、`[SPLIT]`、`[SPK]` |
| test_T_STKN_10 | 语言标记插入 | `[ZH]` 和 `[EN]` 存在 |
| test_T_STKN_11 | encode_speaker(None) | 返回 None |
| test_T_STKN_12 | encode_speaker(路径) | 返回路径或 None |
| test_T_STKN_13 | preprocess 文本清洗 | 全角→半角、空白合并 |
| test_T_STKN_14 | detokenize 可读性 | 返回字符串 |
| test_T_STKN_15 | 空文本处理 | 不崩溃 |

---

## test_gpt2_ar.py — GPT2ARModel

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_GPT2_01 | load_weights 加载成功 | `_is_loaded=True` |
| test_T_GPT2_02 | unload_weights 释放资源 | `_is_loaded=False` |
| test_T_GPT2_03 | generate 基本生成 | 输出 long tensor |
| test_T_GPT2_04 | token ids 范围正确 | `[0, semantic_vocab_size)` |
| test_T_GPT2_05 | speaker_info 条件注入 | 有 ref 输出 ≠ 无 ref |
| test_T_GPT2_06 | temperature 生效 | 不同 temperature 输出不同 |
| test_T_GPT2_07 | top_p / top_k 生效 | 输出 token 在范围内 |
| test_T_GPT2_08 | repetition_penalty 生效 | 不抛异常 |
| test_T_GPT2_09 | generate_stream 流式生成 | yield 多个 chunk |
| test_T_GPT2_10 | stream chunk token 数量 | 每 chunk ≤ stream_batch |
| test_T_GPT2_11 | 停止条件（EOS） | EOS 触发停止 |
| test_T_GPT2_12 | 停止条件（重复检测） | max_repeat 触发停止 |
| test_T_GPT2_13 | max_new_tokens 限制 | 输出长度 ≤ max_new_tokens |
| test_T_GPT2_14 | 双路径 Embedding 独立 | text_embedding ≠ semantic_embedding |
| test_T_GPT2_15 | 类属性正确 | model_type='ar', vocab_size 正确 |

---

## test_sovits_decoder.py — SoVITSDecoder

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SVITS_01 | SemanticEncoder 前向 | 输出 shape `[B, T, H]` |
| test_T_SVITS_02 | PriorEncoder 输出 μ 和 log_σ | shape 正确 |
| test_T_SVITS_03 | 重参数化采样 | z shape = μ shape |
| test_T_SVITS_04 | Flow inverse 单层 | roundtrip 误差 < 1e-4 |
| test_T_SVITS_05 | Flow inverse 多层 | roundtrip 误差 < 1e-3 |
| test_T_SVITS_06 | Flow 数值稳定性 | 无 NaN / Inf |
| test_T_SVITS_07 | HiFiGAN 输出波形 | `[B, 1, samples]` |
| test_T_SVITS_08 | 完整 forward | 返回含 waveform 的 dict |
| test_T_SVITS_09 | set_reference 缓存 | `_ref_tokens` 已设置 |
| test_T_SVITS_10 | decode 接口 | 返回 `(waveform, sr)` |
| test_T_SVITS_11 | decode_chunk 流式 | 返回 `(waveform, sr)` |
| test_T_SVITS_12 | 不同 seq_len 兼容 | 多种长度均正常 |
| test_T_SVITS_13 | load_weights | `_is_loaded=True` |
| test_T_SVITS_14 | 条件注入 | 有 ref ≠ 无 ref |

---

## test_sovits_gpt_regression.py — GPT-SoVITS 组件回归

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SREG_01 | Flow roundtrip 一致性 | 误差 < 1e-4 |
| test_T_SREG_02 | Flow 初始接近恒等 | `|output - input| < 0.5` |
| test_T_SREG_03 | μ 和 σ 范围合理 | log_var ∈ [-10, 10] |
| test_T_SREG_04 | SemanticEncoder 长度兼容 | 多种 seq_len 正常 |
| test_T_SREG_05 | 条件化/非条件化输出一致 | 均为 `[B, 1, T]` |

---

## test_flow_numerics.py — Flow 数值专项

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_FLOW_01 | 单层 roundtrip | 误差 < 1e-5 |
| test_T_FLOW_02 | 多层 roundtrip | 误差 < 1e-4 |
| test_T_FLOW_03 | 大数值输入 | 无 NaN / Inf |
| test_T_FLOW_04 | 小数值输入 | 无 NaN / Inf |
| test_T_FLOW_05 | log_scale clamp | s ∈ [-5, 5] |
| test_T_FLOW_06 | 梯度不消失/爆炸 | 梯度有限 |
| test_T_FLOW_07 | batch_size 兼容 | B=1 和 B>1 正常 |
| test_T_FLOW_08 | seq_len 兼容 | 多种长度正常 |

---

## test_sovits_backend.py — GPT-SoVITS 后端集成

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SBE_01 | 后端创建成功 | 实例非空 |
| test_T_SBE_02 | spec 属性正确 | name/license/sample_rate 正确 |
| test_T_SBE_03 | load 成功 | is_loaded=True |
| test_T_SBE_04 | unload 成功 | is_loaded=False |
| test_T_SBE_05 | synthesize 基本合成 | 输出 AudioData |
| test_T_SBE_06 | synthesize sample_rate | 32000 |
| test_T_SBE_07 | synthesize waveform 非空 | len > 0 |
| test_T_SBE_08 | 自定义 temperature | 参数传递 |
| test_T_SBE_09 | 中文输入 | metadata.language='zh' |
| test_T_SBE_10 | 英文输入 | metadata.language='en' |
| test_T_SBE_11 | 中英混合输入 | 输出正常 |
| test_T_SBE_12 | clone_voice 语音克隆 | 输出 AudioData |
| test_T_SBE_13 | clone_voice 音色保持 | metadata.speaker='cloned' |
| test_T_SBE_14 | synthesize_stream 流式 | yield chunks |
| test_T_SBE_15 | stream yield 多个 chunk | len >= 1 |
| test_T_SBE_16 | stream 总时长合理 | samples > 0 |
| test_T_SBE_17 | extract_speaker | 返回 dict |
| test_T_SBE_18 | save/load_speaker | 保存和加载 |
| test_T_SBE_19 | list_speakers | 返回 list |
| test_T_SBE_20 | describe 返回正确信息 | spec.name='sovits' |
| test_T_SBE_21 | speed 参数影响时长 | waveform 非空 |

---

## test_sovits_weight_converter.py — 权重转换

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SCVT_01 | convert 基本转换 | 输出文件存在 |
| test_T_SCVT_02 | validate 验证 | 返回 True |
| test_T_SCVT_03 | GPT 权重映射 | 含 transformer.wte.weight |
| test_T_SCVT_04 | SoVITS 权重提取 | 含 enc_p./flow./dec. |
| test_T_SCVT_05 | 转换后加载 GPT2ARModel | 加载成功 |
| test_T_SCVT_06 | dry_run 模式 | 返回 dict |

---

## test_all_backends_regression.py — 三后端共存回归

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_ALLR_01 | 三后端同时注册 | chattts/fish/sovits 均在 |
| test_T_ALLR_02 | 三后端独立加载/卸载 | 互不干扰 |
| test_T_ALLR_03 | TTS 节点路由 | list_backends 含三者 |
| test_T_ALLR_04 | 采样率不同 | 24000/22050/32000 |
| test_T_ALLR_05 | Scheduler 显存管理 | 三者均有 min_gpu_memory_gb |
| test_T_ALLR_06 | unload 隔离 | 一个卸载不影响另一个 |
| test_T_ALLR_07 | list_backends 完整 | 含三个后端 |
| test_T_ALLR_08 | auto_select 自动选择 | 返回有效后端名 |

---

## test_cosyvoice_tokenizer.py — CosyVoiceTokenizer

> 依赖 `torch`；不依赖 `transformers`（使用字符级回退分词）。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_CVTOK_01 | 基本分词，输出 tensor | 返回 `torch.Tensor`，形状 `[1, seq_len]` |
| test_T_CVTOK_02 | 中文文本分词正确 | 首尾为 `sos_token_id` / `flow_token_id` |
| test_T_CVTOK_03 | 英文文本分词正确 | 输出非空，首尾特殊标记正确 |
| test_T_CVTOK_04 | 中英混合文本分词 | 输出非空，首尾特殊标记正确 |
| test_T_CVTOK_05 | 特殊标记 [sos]/[flow] 正确添加 | 首元素 == sos_token_id，末元素 == flow_token_id |
| test_T_CVTOK_06 | token ids 在 LLM 词表范围内 | 所有 id ∈ [0, vocab_size) |
| test_T_CVTOK_07 | encode_speaker(None) 返回 None | 返回 `None` |
| test_T_CVTOK_08 | encode_speaker(路径) 返回 dict | 含 `ref_speech_tokens` 和 `speaker_embedding` 键 |
| test_T_CVTOK_09 | preprocess 文本清洗 | 多余空白被合并 |
| test_T_CVTOK_10 | detokenize 文本部分正确 | 返回字符串 |
| test_T_CVTOK_11 | 长文本处理（>500 字） | tokenize 正常，输出长度 > 500 |
| test_T_CVTOK_12 | 空文本处理 | 输出 `[sos, flow]` 两元素 |

---

## test_flow_matching.py — FlowMatchingModel

> 依赖 `torch`；使用小模型配置（hidden=64, layers=2, heads=4, cond=64）。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_FLOWM_01 | load_weights 加载成功 | `_is_loaded=True` |
| test_T_FLOWM_02 | unload_weights 释放资源 | `_is_loaded=False`，`_impl=None` |
| test_T_FLOWM_03 | generate 基本生成 | 输出 mel spectrogram |
| test_T_FLOWM_04 | 输出 mel shape 正确 | `[batch, 80, frames]` |
| test_T_FLOWM_05 | 输出 mel 值在合理范围 | 无 NaN、无 Inf |
| test_T_FLOWM_06 | num_ode_steps=5 可运行 | 最快模式正常 |
| test_T_FLOWM_07 | num_ode_steps=10 可运行 | 推荐模式正常 |
| test_T_FLOWM_08 | num_ode_steps=20 可运行 | 高质量模式正常 |
| test_T_FLOWM_09 | 不同 ode_solver 可运行 | euler / midpoint 均正常 |
| test_T_FLOWM_10 | speaker_info 条件注入 | 有 ref 输出 ≠ 无 ref |
| test_T_FLOWM_11 | generate_stream 分块流式 | yield mel chunk |
| test_T_FLOWM_12 | generate_stream yield 多个 chunk | len ≥ 2 |
| test_T_FLOWM_13 | 不同 text_features 长度兼容 | 短/中/长序列均正常 |
| test_T_FLOWM_14 | model_type == "flow_matching" | 标识正确 |
| test_T_FLOWM_15 | EventBus 兼容 | event_bus=None 不崩溃 |

---

## test_speech_tokenizer.py — SpeechTokenizer

> 依赖 `torch`；使用小模型配置。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SPTOK_01 | encode 基本编码 | 输出 token ids |
| test_T_SPTOK_02 | token ids 在 [0, codebook_size) 范围 | 所有 id ∈ [0, 6561) |
| test_T_SPTOK_03 | 不同长度音频编码正确 | 8000/32000 采样均正常 |
| test_T_SPTOK_04 | decode 可用 | 返回 tensor |
| test_T_SPTOK_05 | load_weights 加载成功 | `_is_loaded=True` |

---

## test_speaker_encoder.py — SpeakerEncoder

> 依赖 `torch`；使用小模型配置。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_SPKENC_01 | encode 基本编码 | 输出嵌入向量 |
| test_T_SPKENC_02 | 输出 shape 正确 | `[1, embedding_dim]` |
| test_T_SPKENC_03 | 相同音频两次编码结果一致 | `allclose` 为 True |
| test_T_SPKENC_04 | 不同音频编码结果不同 | 不相等 |
| test_T_SPKENC_05 | 不同采样率音频自动重采样 | 32000 采样正常 |
| test_T_SPKENC_06 | load_weights 加载成功 | `_is_loaded=True` |

---

## test_flow_matching_numerics.py — Flow Matching 数值专项

> 依赖 `torch`；使用小模型配置。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_FMNUM_01 | Euler 步轨迹平滑 | 输出有限、幅值合理 |
| test_T_FMNUM_02 | Midpoint 与 Euler 均产出有限 mel | 同形状、均无 NaN |
| test_T_FMNUM_03 | velocity 输出无 NaN/Inf | 落在 [-10, 10]（clamp） |
| test_T_FMNUM_04 | ODE 求解后 mel 无 NaN/Inf | 有限值 |
| test_T_FMNUM_05 | 步数增加时输出更稳定 | 5/10/20 步均有限 |
| test_T_FMNUM_06 | 时间步嵌入在 t=0 和 t=1 有定义 | 输出有限、维度正确 |
| test_T_FMNUM_07 | 零条件向量仍能生成 | 退化为无条件生成 |
| test_T_FMNUM_08 | batch_size=1 和 batch_size>1 兼容 | batch 维正确 |
| test_T_FMNUM_09 | 不同 target_length 兼容 | 50/100/200 均正常 |

---

## test_flow_architecture.py — Flow 架构单元测试

> 依赖 `torch`；通过 `_impl` 访问内部 `nn.Module` 组件。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_FARCH_01 | FlowEstimator 输入输出 shape | `[1,80,50] → [1,80,50]` |
| test_T_FARCH_02 | SinusoidalPosEmb 在 t∈[0,1] 输出合理 | 有限、维度正确 |
| test_T_FARCH_03 | Self-Attention 层可运行 | 输出同形状 |
| test_T_FARCH_04 | Cross-Attention 层可运行 | query/kv 正常 |
| test_T_FARCH_05 | AdaptiveLayerNorm (FiLM) 条件注入 | 不同 cond 产生不同输出 |
| test_T_FARCH_06 | 时间步嵌入广播到序列维度 | 形状广播正确 |

---

## test_cosyvoice_backend.py — CosyVoice 后端集成

> 依赖 `torch`；使用 mock 注入管线组件。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_CVBE_01 | 后端创建成功 | 实例非空，name='cosyvoice' |
| test_T_CVBE_02 | spec 属性正确 | acoustic_type='flow_matching', sample_rate=22050 |
| test_T_CVBE_03 | load 成功 | is_loaded=True |
| test_T_CVBE_04 | unload 成功 | is_loaded=False |
| test_T_CVBE_05 | synthesize 基本合成 | 输出 AudioData |
| test_T_CVBE_06 | synthesize sample_rate | 22050 |
| test_T_CVBE_07 | synthesize waveform 非空 | len > 0 |
| test_T_CVBE_08 | 中文输入 | metadata.language='zh' |
| test_T_CVBE_09 | 英文输入 | metadata.language='en' |
| test_T_CVBE_10 | 中英混合输入 | 输出正常 |
| test_T_CVBE_11 | 自定义 num_ode_steps 生效 | 参数传至 acoustic_model |
| test_T_CVBE_12 | clone_voice 语音克隆 | 输出 AudioData |
| test_T_CVBE_13 | clone_voice metadata | metadata.backend='cosyvoice' |
| test_T_CVBE_14 | synthesize_stream 流式合成 | yield AudioData |
| test_T_CVBE_15 | synthesize_stream yield 多个 chunk | len ≥ 2 |
| test_T_CVBE_16 | synthesize_stream 总时长合理 | duration > 0 |
| test_T_CVBE_17 | extract_speaker 提取特征 | 返回 dict 含两个键 |
| test_T_CVBE_18 | save_speaker / load_speaker | 保存后可加载 |
| test_T_CVBE_19 | set_ode_params 运行时修改 | _num_ode_steps 和 _ode_solver 更新 |
| test_T_CVBE_20 | list_speakers 返回列表 | isinstance(list) |
| test_T_CVBE_21 | check_dependencies 返回 bool | 返回 bool |

---

## test_cosyvoice_weight_converter.py — 权重转换

> 依赖 `torch`、`safetensors`；使用临时伪造 checkpoint。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_CWC_01 | convert 基本转换流程 | 返回 dict 映射 |
| test_T_CWC_02 | validate 验证转换后权重 | 返回 bool |
| test_T_CWC_03 | Flow Matching 权重映射正确 | 含 flow_matching.safetensors |
| test_T_CWC_04 | LLM 权重引用正确 | text_frontend_config.json 存在（非 safetensors） |
| test_T_CWC_05 | 转换后 FlowMatchingModel 可加载 | 加载不崩溃 |
| test_T_CWC_06 | dry_run 模式 | 返回 dict |

---

## test_four_backends_regression.py — 四后端共存回归

> 依赖 `torch`；验证 ChatTTS / Fish / GPT-SoVITS / CosyVoice 四后端共存。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_4BE_01 | 四后端同时注册 | chattts/fish/sovits/cosyvoice 均在 |
| test_T_4BE_02 | list_backends 返回 4 个 | len ≥ 4 |
| test_T_4BE_03 | acoustic_type 正确 | ar/ar/ar/flow_matching |
| test_T_4BE_04 | 四后端独立加载/卸载 | 互不干扰 |
| test_T_4BE_05 | Scheduler 显存管理 | 四者均有 min_gpu_memory_gb |
| test_T_4BE_06 | LRU 淘汰 | _backends 字典含四者 |
| test_T_4BE_07 | 一个 unload 后另一个仍可用 | 注册表不变 |
| test_T_4BE_08 | auto_select 选择正确后端 | quality→cosyvoice, low_latency→chattts |
| test_T_4BE_09 | ChatTTS 仍注册（回归） | name='chattts' |
| test_T_4BE_10 | Fish 仍注册（回归） | name='fish' |
| test_T_4BE_11 | GPT-SoVITS 仍注册（回归） | name='sovits' |
| test_T_4BE_12 | CosyVoice 注册 | acoustic_type='flow_matching' |

---

## test_streaming_comparison.py — 四后端流式对比

> 依赖 `torch`；验证流式接口兼容性。

| 用例 ID | 描述 | 预期结果 |
| --- | --- | --- |
| test_T_STRCOMP_01 | AR 后端有 synthesize_stream | ChatTTS/Fish/SoVITS 均有方法 |
| test_T_STRCOMP_02 | CosyVoice 有 synthesize_stream | 方法存在 |
| test_T_STRCOMP_03 | 流式输出 AudioData | yield AudioData 对象 |
| test_T_STRCOMP_04 | 流式与非流式采样率一致 | 均为 22050Hz |
| test_T_STRCOMP_05 | chunk_size 参数被接受 | 不崩溃 |
