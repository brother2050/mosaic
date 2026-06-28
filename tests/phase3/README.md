# Phase 3 测试验收清单

> 覆盖音频域 5 个节点 + 字幕域 3 个节点 + 数据类型 + 基类 + 端到端集成

## 测试统计

| 文件 | 测试用例数 | 通过 | 跳过 | 说明 |
|------|-----------|------|------|------|
| test_audio_types.py | 10 | 10 | 0 | 音频数据类型 |
| test_subtitle_types.py | 13 | 13 | 0 | 字幕数据类型 |
| test_tts.py | 9 | 9 | 0 | TTS 节点 |
| test_asr.py | 9 | 8 | 1 | ASR 节点 |
| test_music_generator.py | 7 | 7 | 0 | 音乐生成节点 |
| test_sound_effect.py | 7 | 7 | 0 | 音效生成节点 |
| test_voice_clone.py | 7 | 7 | 0 | 语音克隆节点 |
| test_audio_base.py | 10 | 9 | 1 | 音频域基类 |
| test_subtitle_generator.py | 9 | 9 | 0 | 字幕生成节点 |
| test_subtitle_translator.py | 8 | 8 | 0 | 字幕翻译节点 |
| test_subtitle_aligner.py | 9 | 9 | 0 | 字幕对齐节点 |
| test_subtitle_base.py | 16 | 16 | 0 | 字幕域基类 |
| test_integration.py | 8 | 8 | 0 | 端到端集成 |
| **总计** | **122** | **120** | **2** | |

---

## 测试用例明细

### 1. test_audio_types.py — 音频数据类型测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_AUDTYPE_01 | test_create_with_waveform_and_sample_rate | AudioData 创建成功，waveform/sample_rate/data_type 正确 |
| T_AUDTYPE_01 | test_create_with_default_sample_rate | 默认采样率 22050 |
| T_AUDTYPE_01 | test_create_with_metadata | metadata 正确存储 |
| T_AUDTYPE_02 | test_roundtrip | 序列化→反序列化后 waveform 与原始一致 |
| T_AUDTYPE_02 | test_dict_like_access | 支持字典式访问 |
| T_AUDTYPE_03 | test_low_sample_rate | 8000Hz 采样率处理正常 |
| T_AUDTYPE_03 | test_high_sample_rate | 48000Hz 采样率处理正常 |
| T_AUDTYPE_04 | test_mono_audio | 单声道 (samples,) 形状正确 |
| T_AUDTYPE_04 | test_stereo_audio | 立体声 (2, samples) 形状正确 |
| T_AUDTYPE_05 | test_duration_calculation | 时长 = 样本数 / 采样率 |

### 2. test_subtitle_types.py — 字幕数据类型测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_SUBTYPE_01 | test_create_with_segments | SubtitleData 创建成功，segments/format/data_type 正确 |
| T_SUBTYPE_01 | test_create_with_format | 指定 format 参数生效 |
| T_SUBTYPE_01 | test_create_with_metadata | metadata 正确存储 |
| T_SUBTYPE_01 | test_empty_segments_default | 默认 segments 为空列表 |
| T_SUBTYPE_02 | test_roundtrip | 序列化→反序列化后 segments 一致 |
| T_SUBTYPE_02 | test_roundtrip_with_extra_fields | 额外字段（speaker）保留 |
| T_SUBTYPE_03 | test_parse_srt | 标准 SRT 解析为 5 个片段 |
| T_SUBTYPE_03 | test_parse_srt_empty | 空字符串返回空列表 |
| T_SUBTYPE_03 | test_parse_srt_single_segment | 单段 SRT 解析正确 |
| T_SUBTYPE_04 | test_parse_vtt | 标准 VTT 解析为 5 个片段 |
| T_SUBTYPE_04 | test_parse_vtt_empty | 空字符串返回空列表 |
| T_SUBTYPE_04 | test_parse_vtt_header_only | 仅头部返回空列表 |
| T_SUBTYPE_05 | test_empty_subtitle_data | 空字幕 segments 为空，通过校验 |

### 3. test_tts.py — TTS 节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_TTS_01 | test_basic_tts_output | 输出 AudioData 非空 |
| T_TTS_02 | test_sample_rate_correct | 输出 sample_rate 正确 |
| T_TTS_03 | test_waveform_shape_correct | waveform 为 1 维 numpy.ndarray |
| T_TTS_04 | test_long_text_sentence_split | 长文本自动分句处理 |
| T_TTS_05 | test_language_param | 指定语言参数生效 |
| T_TTS_05 | test_specified_language_in_input | 输入中指定 language 生效 |
| T_TTS_06 | test_describe | describe 返回 name/domain/input_types/output_types |
| — | test_edge_tts_backend | edge-tts 后端初始化正确 |
| — | test_missing_text | 缺少 text 抛出 ValueError |
| — | test_empty_text | 空文本抛出 ValueError |

### 4. test_asr.py — ASR 节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_ASR_01 | test_basic_asr_output | 输出识别文本非空 |
| T_ASR_02 | test_segments_with_timestamps | segments 包含 start/end/text |
| T_ASR_03 | test_language_auto_detect | 自动检测语言 |
| T_ASR_04 | test_specified_language | 指定语言参数生效 |
| T_ASR_05 | test_translate_task | translate 任务输出英文 |
| T_ASR_06 | test_long_audio_handling | 长音频（>30s）处理正常 |
| T_ASR_07 | test_from_file_path | 从文件路径输入识别成功 |
| — | test_missing_audio | 缺少 audio 抛出 ValueError |
| — | test_describe | describe 返回正确信息 |

### 5. test_music_generator.py — 音乐生成节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_MUSIC_01 | test_basic_music_generation | 输出 AudioData 非空 |
| T_MUSIC_02 | test_duration_param | 指定 duration 参数生效 |
| T_MUSIC_02 | test_duration_clamp | 超过 30s 的 duration 被截断 |
| T_MUSIC_03 | test_different_styles | 不同 prompt 都生成音频 |
| T_MUSIC_03 | test_guidance_scale_param | guidance_scale 参数生效 |
| T_MUSIC_04 | test_describe | describe 标注模型信息 |
| — | test_missing_prompt | 缺少 prompt 抛出 ValueError |
| — | test_empty_prompt | 空 prompt 抛出 ValueError |

### 6. test_sound_effect.py — 音效生成节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_SFX_01 | test_basic_sound_effect | 输出 AudioData 非空 |
| T_SFX_02 | test_different_descriptions | 不同描述生成音效 |
| T_SFX_02 | test_duration_param | 指定 duration 参数 |
| T_SFX_03 | test_negative_prompt | negative_prompt 参数生效 |
| T_SFX_03 | test_num_inference_steps | num_inference_steps 参数生效 |
| T_SFX_04 | test_describe | describe 标注模型信息 |
| — | test_missing_prompt | 缺少 prompt 抛出 ValueError |
| — | test_empty_prompt | 空 prompt 抛出 ValueError |

### 7. test_voice_clone.py — 语音克隆节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_CLONE_01 | test_basic_voice_clone | 输出 AudioData 和 reference_audio |
| T_CLONE_02 | test_duration_vs_text_length | 长文本输出更长波形 |
| T_CLONE_03 | test_from_file_path | 从文件路径输入参考音频成功 |
| T_CLONE_04 | test_describe | describe 标注模型信息 |
| — | test_missing_reference_audio | 缺少 reference_audio 抛出 ValueError |
| — | test_missing_text | 缺少 text 抛出 ValueError |
| — | test_language_param | 语言参数可指定 |

### 8. test_audio_base.py — 音频域基类测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_ABASE_01 | test_load_from_file | 从 wav 文件加载音频 |
| T_ABASE_02 | test_load_from_array | 从 ndarray 加载音频 |
| T_ABASE_02 | test_load_from_audio_data | 从 AudioData 加载音频 |
| T_ABASE_02 | test_load_invalid_type | 无效类型抛出 TypeError |
| T_ABASE_03 | test_resample_down | 降采样后长度正确 |
| T_ABASE_03 | test_resample_same_rate | 相同采样率返回原波形 |
| T_ABASE_03 | test_resample_up | 升采样后长度正确 |
| T_ABASE_03 | test_resample_stereo | 立体声重采样保持通道数 |
| T_ABASE_04 | test_mono_passthrough | 单声道保持不变 |
| T_ABASE_04 | test_stereo_to_mono | 立体声转单声道正确 |
| T_ABASE_05 | test_normalize | 归一化后最大值 ≤ 1.0 |
| T_ABASE_05 | test_normalize_large_values | 大数据幅值归一化正确 |
| T_ABASE_05 | test_normalize_zero | 全零波形归一化不变 |

### 9. test_subtitle_generator.py — 字幕生成节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_SUBGEN_01 | test_generate_from_audio | 输出 SubtitleData |
| T_SUBGEN_02 | test_segments_structure | segments 包含 start/end/text |
| T_SUBGEN_03 | test_timeline_continuous | 时间轴连续无重叠 |
| T_SUBGEN_04 | test_srt_output | 输出 SRT 格式正确 |
| T_SUBGEN_05 | test_vtt_output | 输出 VTT 格式正确 |
| T_SUBGEN_06 | test_word_timestamps | word_timestamps 参数生效 |
| T_SUBGEN_07 | test_max_chars_per_line | max_chars_per_line 参数生效 |
| T_SUBGEN_08 | test_long_audio | 长音频处理正常 |
| — | test_missing_audio | 缺少 audio 抛出异常 |

### 10. test_subtitle_translator.py — 字幕翻译节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_SUBTRANS_01 | test_translate_chinese_to_english | 中文字幕翻译为英文 |
| T_SUBTRANS_02 | test_timeline_preserved | 时间轴保持不变 |
| T_SUBTRANS_03 | test_speaker_preserved | speaker 信息保留 |
| T_SUBTRANS_04 | test_empty_subtitle | 空字幕返回空列表 |
| T_SUBTRANS_05 | test_long_line_split | 翻译后长行自动拆分 |
| — | test_missing_subtitle | 缺少 subtitle 抛出异常 |
| — | test_describe | describe 返回正确信息 |

### 11. test_subtitle_aligner.py — 字幕对齐节点测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_ALIGN_01 | test_basic_alignment | 输出 SubtitleData |
| T_ALIGN_02 | test_alignment_score_range | alignment_score 在 0-1 范围 |
| T_ALIGN_03 | test_time_shift_calculated | time_shift 计算正确 |
| T_ALIGN_04 | test_duration_mismatch | 时长不匹配时仍返回字幕 |
| T_ALIGN_05 | test_whisper_method | whisper 对齐方法生效 |
| T_ALIGN_05 | test_dtw_method | DTW 对齐方法生效 |
| T_ALIGN_05 | test_dtw_alignment_score | DTW 方法输出 alignment_score |
| — | test_missing_subtitle | 缺少 subtitle 抛出异常 |
| — | test_missing_audio | 缺少 audio 抛出异常 |

### 12. test_subtitle_base.py — 字幕域基类测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_SBASE_01 | test_parse_srt | 解析 5 个片段 |
| T_SBASE_01 | test_parse_srt_empty | 空字符串返回空列表 |
| T_SBASE_01 | test_parse_srt_no_index | 无序号 SRT 可解析 |
| T_SBASE_01 | test_parse_srt_multiline_text | 多行文本解析正确 |
| T_SBASE_02 | test_parse_vtt | 解析 5 个片段 |
| T_SBASE_02 | test_parse_vtt_empty | 空字符串返回空列表 |
| T_SBASE_02 | test_parse_vtt_header_only | 仅头部返回空列表 |
| T_SBASE_02 | test_parse_vtt_with_cue_id | 带 cue ID 的 VTT 解析 |
| T_SBASE_03 | test_to_srt | 输出 SRT 格式正确 |
| T_SBASE_03 | test_to_srt_empty | 空片段返回空字符串 |
| T_SBASE_04 | test_to_vtt | 输出 VTT 格式正确 |
| T_SBASE_05 | test_format_srt | SRT 时间戳格式正确 |
| T_SBASE_05 | test_format_vtt | VTT 时间戳格式正确 |
| T_SBASE_05 | test_format_zero | 零时间戳格式化 |
| T_SBASE_05 | test_format_negative | 负时间戳转为 0 |
| T_SBASE_06 | test_parse_srt_timestamp | SRT 时间戳解析 |
| T_SBASE_06 | test_parse_vtt_timestamp | VTT 时间戳解析 |
| T_SBASE_06 | test_parse_short_format | 短格式 MM:SS 解析 |
| T_SBASE_06 | test_parse_seconds_only | 仅秒数解析 |
| T_SBASE_06 | test_parse_invalid | 无效格式返回 0.0 |
| T_SBASE_07 | test_merge_basic | 短片段合并正确 |
| T_SBASE_07 | test_merge_empty | 空列表返回空列表 |
| T_SBASE_07 | test_merge_single | 单片段保持不变 |
| T_SBASE_07 | test_merge_preserve_speaker | 合并时保留 speaker |
| T_SBASE_08 | test_split_basic | 长片段拆分正确，时间轴连续 |
| T_SBASE_08 | test_split_no_op | 短片段不拆分 |
| T_SBASE_08 | test_split_empty | 空列表返回空列表 |
| T_SBASE_08 | test_split_hard_cut | 无标点长文本按字数硬切 |

### 13. test_integration.py — 端到端集成测试

| ID | 测试名称 | 预期结果 |
|----|---------|---------|
| T_E2E_P3_01 | test_tts_to_asr_roundtrip | TTS→ASR 文本→语音→文本完整流程 |
| T_E2E_P3_02 | test_asr_to_subtitle | ASR→字幕生成流程 |
| T_E2E_P3_03 | test_subtitle_to_translation | 字幕生成→字幕翻译流程 |
| T_E2E_P3_04 | test_full_pipeline | TTS→字幕生成→字幕对齐完整流程 |
| T_E2E_P3_05 | test_load_unload_memory_release | 单节点 load/unload 显存释放 |
| T_E2E_P3_05 | test_multiple_nodes_memory | 多节点 load/unload 显存释放 |
| T_E2E_P3_06 | test_events_triggered | 运行中 NODE_START/NODE_COMPLETE 事件触发 |
| T_E2E_P3_06 | test_error_event | 错误时 NODE_ERROR 事件触发 |

---

## 运行命令

```bash
# 运行全部 Phase 3 测试
cd /workspace/mosaic && python -m pytest tests/phase3/ -v

# 跳过集成测试
cd /workspace/mosaic && python -m pytest tests/phase3/ -v -m "not integration"

# 仅运行集成测试
cd /workspace/mosaic && python -m pytest tests/phase3/ -v -m integration

# 运行单个文件
cd /workspace/mosaic && python -m pytest tests/phase3/test_audio_types.py -v
```

## 依赖要求

- pytest
- numpy
- soundfile（可选，用于从文件加载/保存音频的测试；mock 环境会自动跳过）
- TTS / edge_tts / transformers / diffusers（测试使用 mock 注入，无需真实安装）

## 跳过说明

- `test_load_from_file` / `test_save_and_load`（test_audio_base.py）：需要真实 soundfile 才能写入/读取文件
- `test_from_file_path`（test_asr.py）：需要真实 soundfile 写入临时 wav 文件