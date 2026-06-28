# Phase 4 测试验收清单

## 概述

Phase 4 测试覆盖视频域（5 个节点）、导出域（3 个节点）、异步执行（AsyncTask / TaskManager / Pipeline.run_async）以及端到端集成场景。

- **测试文件数**：16
- **测试用例总数**：134
- **测试框架**：pytest
- **Mock 策略**：全部使用合成 PIL 帧 + mock torch/diffusers/FFmpeg，不依赖外部视频文件或真实模型

---

## 一、视频数据类型测试（test_video_types.py）— 5 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_VIDTYPE_01 | VideoData 创建，包含 frames 和 fps | frames 和 fps 正确赋值，data_type 为 "video" |
| T_VIDTYPE_02 | duration 属性计算正确 | duration = frame_count / fps |
| T_VIDTYPE_03 | width、height 属性正确 | 从首帧 PIL.Image.size 获取 |
| T_VIDTYPE_04 | frame_count 属性正确 | len(frames) 与帧数一致 |
| T_VIDTYPE_05 | VideoData 序列化/反序列化 | to_dict / data_from_dict 往返，帧保持 PIL.Image 类型 |

---

## 二、视频域基类测试（test_video_base.py）— 8 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_VBASE_01 | _load_video 从文件加载 | 从 video 文件逐帧读取为 PIL.Image，返回 VideoData |
| T_VBASE_02 | _save_video 保存为文件 | 帧列表保存为 video 文件，文件存在且非空 |
| T_VBASE_03 | _extract_frames 提取指定帧 | 按索引提取帧，越界索引被忽略 |
| T_VBASE_04 | _resize_frames 批量 resize | 所有帧 resize 到目标尺寸 |
| T_VBASE_05 | _frames_to_tensor 转换正确 | PIL.Image → (N, C, H, W) torch.Tensor |
| T_VBASE_06 | _tensor_to_frames 转换正确 | torch.Tensor → PIL.Image 列表，往返一致 |
| T_VBASE_07 | _ensure_even_dimensions 处理奇数尺寸 | 奇数减 1 变偶数，最小为 (2, 2) |
| T_VBASE_08 | _get_frame_at 按时间戳取帧 | timestamp * fps 计算帧索引，超范围钳制 |

---

## 三、文生视频节点测试（test_text_to_video.py）— 6 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_T2V_01 | 基本文生视频 | 输出 MosaicData 包含 VideoData |
| T_T2V_02 | 输出帧数与请求一致 | num_frames 参数对应输出帧数 |
| T_T2V_03 | fps 参数生效 | 输出 VideoData.fps 等于请求的 fps |
| T_T2V_04 | 指定 seed 可复现 | seed 参数正确传递，输出包含 seed |
| T_T2V_05 | describe 标注显存需求和许可证 | model_info 包含 vram_gb 和 license |
| T_T2V_06 | load/unload 状态正确 | 加载后 is_loaded()=True，卸载后=False |

---

## 四、图生视频节点测试（test_image_to_video.py）— 5 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_I2V_01 | 基本图生视频 | 输出 MosaicData 包含 VideoData |
| T_I2V_02 | 输出帧数正确 | num_frames 参数对应输出帧数 |
| T_I2V_03 | motion_bucket_id 参数生效 | metadata 中 motion_bucket_id 正确 |
| T_I2V_04 | 输入图片自动 resize | 非目标尺寸的图片被 resize 到 1024x576 |
| T_I2V_05 | describe 标注信息 | model_info 包含 name 和 license |

---

## 五、视频续写节点测试（test_video_continuation.py）— 4 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CONT_01 | 基本续写 | 输出帧数 > 输入帧数 |
| T_CONT_02 | overlap_frames 参数生效 | 输出中 overlap_frames 正确 |
| T_CONT_03 | 输出包含原始帧和续写帧 | total_frames > 0 |
| T_CONT_04 | continuation_video 单独可访问 | 输出包含 continuation_video 键，为 VideoData |

---

## 六、插帧节点测试（test_frame_interpolation.py）— 4 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_INTERP_01 | 2x 插帧 | 帧数翻倍（4→7，即 2N-1） |
| T_INTERP_02 | target_fps 参数生效 | new_fps > 原始 fps |
| T_INTERP_03 | 线性插值方法可运行 | linear 模式无需模型即可完成 |
| T_INTERP_04 | 输出 fps 正确 | 2x 插帧后 fps=60，4x 后 fps=120 |

---

## 七、拆帧节点测试（test_frame_extractor.py）— 6 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_EXTRACT_01 | all 模式提取所有帧 | 提取全部 20 帧 |
| T_EXTRACT_02 | interval 模式按间隔提取 | interval=5 提取 4 帧 |
| T_EXTRACT_03 | timestamps 模式按时间戳提取 | 按给定时间戳列表提取帧 |
| T_EXTRACT_04 | 输出 timestamps 列表正确 | 首帧时间戳=0.0，末帧=19/30 |
| T_EXTRACT_05 | 从文件路径输入 | mock _load_video 后正确提取帧 |
| T_EXTRACT_06 | keyframe 模式 | 至少提取首帧作为关键帧 |

---

## 八、视频编码节点测试（test_video_encoder.py）— 7 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_ENC_01 | 基本 mp4 编码 | 输出文件存在且非空，stdin 有帧数据写入 |
| T_ENC_02 | 不同格式编码 | mp4/webm/avi 各自编码成功，文件扩展名正确 |
| T_ENC_03 | quality 参数生效 | -crf 值随 quality 变化 |
| T_ENC_04 | 音视频合并 | subprocess.run 被调用，命令包含 -i 音频输入 |
| T_ENC_05 | 字幕烧录 | subprocess.run 被调用，命令包含 subtitles 滤镜 |
| T_ENC_06 | 奇数尺寸帧自动处理 | 输出 resolution 宽高均为偶数 |
| T_ENC_07 | 输出包含正确的元信息 | output_path/format/codec/duration/file_size/resolution 均正确 |

---

## 九、直播推流节点测试（test_livestream.py）— 4 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_LIVE_01 | describe 标注为不需要 GPU | model_info 为 None（导出域节点） |
| T_LIVE_02 | stream_url 参数校验 | 缺少/空/空白 stream_url 返回 status="failed" |
| T_LIVE_03 | 无效地址时给出友好错误 | 推流失败返回 error 字段，消息非空 |
| T_LIVE_04 | 协议参数（rtmp/srt）正确传递 | RTMP 使用 -f flv，SRT 使用 -f mpegts |

---

## 十、多格式导出测试（test_multi_format_export.py）— 7 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_MULTI_01 | 视频多格式导出（mp4 + gif） | outputs 字典包含 mp4 和 gif |
| T_MULTI_02 | 图像多格式导出（png + jpg + webp） | 三种格式文件均存在且非空 |
| T_MULTI_03 | 音频多格式导出（wav + mp3） | 两种格式文件均存在且非空 |
| T_MULTI_04 | 字幕多格式导出（srt + vtt） | SRT 包含时间戳，VTT 以 WEBVTT 开头 |
| T_MULTI_05 | outputs 字典格式正确 | format→filepath 映射，路径为绝对路径 |
| T_MULTI_06 | 不支持的格式给出警告 | 不支持的格式被跳过，日志含警告 |
| T_MULTI_07 | total_size 计算正确 | total_size 等于各文件大小之和 |

---

## 十一、AsyncTask 测试（test_async_task.py）— 10 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_ASYNC_01 | 创建 AsyncTask | 初始状态 pending，progress=0.0，current_node=None |
| T_ASYNC_02 | 状态流转 | pending → running → completed |
| T_ASYNC_03 | wait() 阻塞并返回结果 | 返回 PipelineResult，success=True |
| T_ASYNC_04 | wait(timeout) 超时 | 慢任务超时抛出 TimeoutError |
| T_ASYNC_05 | cancel() 设置取消标志 | is_cancelled 变为 True |
| T_ASYNC_06 | on_complete 回调被触发 | 回调收到 PipelineResult 参数 |
| T_ASYNC_07 | on_error 回调在失败时被触发 | 失败节点触发错误回调，收到 Exception |
| T_ASYNC_08 | on_progress 回调收到进度更新 | 回调参数为 (float, str)，progress 在 0.0~1.0 |
| T_ASYNC_09 | is_ready() 正确反映状态 | pending 返回 False，completed 返回 True |
| T_ASYNC_10 | to_dict() 序列化正确 | 包含 11 个必要键，值匹配当前状态 |

---

## 十二、TaskManager 测试（test_task_manager.py）— 8 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_TM_01 | submit 提交任务返回 AsyncTask | 返回 AsyncTask，task_id 和 pipeline_name 正确 |
| T_TM_02 | get 按 task_id 获取任务 | 返回同一任务实例 |
| T_TM_03 | list_tasks 返回所有任务 | 3 个任务全部返回 |
| T_TM_04 | list_tasks(status="running") 过滤 | 仅返回对应状态的任务 |
| T_TM_05 | cancel 取消指定任务 | cancel() 返回 True，任务被取消 |
| T_TM_06 | cancel_all 取消所有运行中任务 | 返回非负整数，至少一个任务被取消 |
| T_TM_07 | cleanup 清理过期任务 | 删除已完成任务，get() 返回 None |
| T_TM_08 | status_summary 返回统计信息 | 字典包含 6 个键，计数之和等于 total |

---

## 十三、异步管道测试（test_async_pipeline.py）— 4 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_APIPE_01 | run_async 返回 AsyncTask | 返回 AsyncTask 实例，pipeline_name 正确 |
| T_APIPE_02 | 异步执行最终得到 PipelineResult | wait() 返回 PipelineResult，success=True |
| T_APIPE_03 | 多个异步任务并行执行 | 3 个任务有不同 task_id，并行总时间 < 2.5s |
| T_APIPE_04 | EventBus 事件被转发 | NODE_START 和 NODE_COMPLETE 事件均被触发 |

---

## 十四、端到端集成测试（test_integration.py）— 7 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_E2E_P4_01 | 文本生成→文生图→图生视频 跨域管道 | 最终输出包含 VideoData（10 帧） |
| T_E2E_P4_02 | 文生视频→插帧→视频编码 视频增强 | 插帧后帧数 >= 原始帧数，编码输出路径存在 |
| T_E2E_P4_03 | 文生视频→拆帧→去背景 视频后处理 | 帧提取输出 image，背景移除后输出 RGBA |
| T_E2E_P4_04 | 文生视频→编码→多格式导出 完整导出 | 导出 2 种格式（mp4 + webm） |
| T_E2E_P4_05 | 异步执行完整管道，等待结果 | PipelineResult.success=True，包含 video |
| T_E2E_P4_06 | 异步执行中查询进度 | 执行期间出现 running 状态，最终成功 |
| T_E2E_P4_07 | PipelineResult 包含各节点耗时 | node_durations 非空，总耗时 >= 各节点之和 |

---

## 测试统计

| 分类 | 用例数 |
|------|--------|
| 视频数据类型 | 5 |
| 视频域基类 | 8 |
| 文生视频 | 6 |
| 图生视频 | 5 |
| 视频续写 | 4 |
| 插帧 | 4 |
| 拆帧 | 6 |
| 视频编码 | 7 |
| 直播推流 | 4 |
| 多格式导出 | 7 |
| AsyncTask | 10 |
| TaskManager | 8 |
| 异步管道 | 4 |
| 端到端集成 | 7 |
| **总计** | **85** |

## 运行命令

```bash
# 运行 Phase 4 全部测试
python -m pytest tests/phase4/ -v

# 仅运行集成测试
python -m pytest tests/phase4/ -v -m integration

# 运行全部测试（Phase 1-4）
python -m pytest tests/phase1/ tests/phase2/ tests/phase3/ tests/phase4/ -v

# 跳过集成测试
python -m pytest tests/phase4/ -v -m "not integration"
```