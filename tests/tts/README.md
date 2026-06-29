# tests/tts — TTS 子系统测试用例说明

本目录覆盖 mosaic TTS 框架的基础架构、文本前端、声学模型、声码器、
ChatTTS 后端与权重转换器。测试遵循以下原则：

- 不依赖真实预训练权重（小模型参数 / mock / 随机初始化）。
- 可选依赖缺失时用 `pytest.mark.skipif` / `pytest.importorskip` 跳过。
- 测试函数统一以 `T_XXXX_NN` 命名（已通过 `pyproject.toml` 的
  `python_functions = ["test*", "T_*"]` 启用收集）。
- 当前环境已安装 `torch`、`safetensors`，未安装 `transformers`、`vocos`；
  因此依赖 `transformers` 的用例会跳过。

运行：

```bash
cd /workspace/mosaic && python -m pytest tests/tts/ -v --tb=short
```

共 69 个用例：49 通过 / 20 跳过（均为 `transformers` 缺失导致）。

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
