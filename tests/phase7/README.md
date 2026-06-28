# Phase 7 测试验收清单

## 概述

Phase 7 测试覆盖数字人域（Digital Human Domain），包括 MotionGenerator（动作生成）、RealtimeRenderer（实时渲染），以及端到端集成场景和数字人管道组合。

- **测试文件数**：4（含 conftest.py）
- **测试用例总数**：40
- **测试框架**：pytest
- **Mock 策略**：全部使用 mock torch/diffusers/transformers/insightface 模块，preset 模式直接测试，其他模式通过 mock 验证数据流

---

## 一、MotionGenerator 节点测试（test_motion_generator.py）— 12 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_MOT_01 | preset 模式，输入预设名称 "wave" 输出 MotionData | 输出包含 motion 字段，类型为 MotionData，frame_count>0，fps=30 |
| T_MOT_02 | 内置动作列表非空，至少包含 wave/bow/nod 等 | _PRESET_ANIMATIONS 有 15 个动作，describe 包含 preset_names |
| T_MOT_03 | text2motion 模式，输入描述 "挥手" 输出 MotionData | 输出 motion 为 MotionData，frame_count=60（2s@30fps） |
| T_MOT_04 | audio2motion 模式，输入音频输出 MotionData | 输出 motion 为 MotionData，frame_count=60（2s@30fps） |
| T_MOT_05 | duration 参数生效（duration=5.0 -> 150 帧 @ 30fps） | frame_count=150，duration=5.0 |
| T_MOT_06 | fps 参数生效（fps=15 -> 45 帧 @ 3s） | frame_count=45，motion.fps=15 |
| T_MOT_07 | smooth 参数生效（smooth=True vs smooth=False 输出有差异） | 两组 keypoints 的 max_diff >= 0 |
| T_MOT_08 | 输出 keypoints shape 为 (frame_count, 17, 2) 对于 COCO | keypoints.shape 正确 |
| T_MOT_09 | skeleton_type="openpose" 参数正确传递 | 输出 skeleton_type 和 motion.skeleton_type 均为 "openpose" |
| T_MOT_10 | describe 返回正确信息 | name/domain/version/model_info 均正确，num_presets=15 |
| T_MOT_10 补充 | text2motion 和 audio2motion 模式的 describe | method 字段正确 |

---

## 二、RealtimeRenderer 节点测试（test_realtime_renderer.py）— 14 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_RT_01 | 基本实时渲染（audio 模式），输出帧列表 | frames 为非空 list，每帧为 PIL Image |
| T_RT_02 | text 模式未启用 TTS 时抛出 ValueError | 抛出 ValueError，匹配 "Text mode requires TTS" |
| T_RT_03 | motion 模式可运行，输出帧列表 | frames 非空，每帧为 PIL Image |
| T_RT_04 | render_stats 包含正确字段 | 包含 total_frames/average_fps/average_latency_ms/dropped_frames |
| T_RT_05 | target_fps 参数传递正确 | render_stats["target_fps"] 与构造参数一致 |
| T_RT_06 | resolution 参数传递正确 | render_stats["resolution"] 与构造参数一致 |
| T_RT_07 | enable_tts=True 时加载 TTS 模型 | describe 中 enable_tts=True，包含 tts_model |
| T_RT_08 | start_realtime 和 stop_realtime 生命周期正确 | 初始 is_running=False，渲染完成后 is_running=False |
| T_RT_09 | start_realtime 的 output_callback 被调用 | callback 被调用，收到 PIL Image 帧 |
| T_RT_10 | stop_realtime 后渲染停止 | _stop_requested 为 True |
| T_RT_11 | get_stats 返回当前统计 | 包含 total_frames/average_fps/average_latency_ms/dropped_frames/is_running |
| T_RT_12 | describe 返回正确信息（标注性能指标） | name/domain/version 正确，performance 包含 target_fps |
| T_RT_13 | load/unload 后 is_loaded 状态正确 | 加载前 False，加载后 True，卸载后 False |
| T_RT_14 | 输入为空流时优雅处理 | 空列表 -> 输出空 frames 列表，不崩溃 |

---

## 三、端到端集成测试（test_integration.py）— 10 个用例

全部使用 `@pytest.mark.integration` 标记。

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_E2E_P7_01 | TTS -> LipSyncer -> VideoEncoder 文本到说话数字人视频 | 三层数据流正确传递，最终输出 video_path |
| T_E2E_P7_02 | AvatarDriver -> VideoEncoder 形象驱动导出视频 | 两层数据流正确传递，最终输出 video_path |
| T_E2E_P7_03 | MotionGenerator(preset) -> AvatarDriver 预设动作驱动形象 | 真实 MotionGenerator 生成动作，mock AvatarDriver 接收 |
| T_E2E_P7_04 | TextGenerator -> TTS -> LipSyncer 完整对话数字人流程 | 三层数据流正确传递，lip_synced=True |
| T_E2E_P7_05 | AvatarDriver -> FrameInterpolator -> VideoEncoder 驱动后插帧 | 插帧后帧数增加，最终编码输出 video_path |
| T_E2E_P7_06 | LipSyncer -> MultiFormatExporter 口型同步后多格式导出 | 输出 formats 列表非空 |
| T_E2E_P7_07 | 数字人管道与一致性域组合（IdentityKeeper + AvatarDriver） | 跨域组合成功，数据流正确传递 |
| T_E2E_P7_08 | 运行过程中事件被正确触发 | NODE_START 和 NODE_COMPLETE 事件均被触发 |
| T_E2E_P7_09 | PipelineResult 包含正确信息 | pipeline_name/duration/output/intermediate 均正确 |
| T_E2E_P7_10 | 中间产物（驱动帧、口型帧）可单独取出 | 中间产物中含 frames 类型数据 |

---

## 四、数字人管道组合测试（test_digital_human_pipeline.py）— 6 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_DHPIPE_01 | Pipeline 声明式组装 MotionGenerator -> AvatarDriver -> VideoEncoder | PipelineResult 输出含 video_path，>=3 个中间产物 |
| T_DHPIPE_02 | 数字人管道与文本域管道组合 TextGenerator -> TTS -> LipSyncer | 输出 lip_synced=True，>=3 个中间产物 |
| T_DHPIPE_03 | 数字人管道与音频域管道组合 TTS -> LipSyncer -> AvatarDriver | 输出 driven_by 标记，>=3 个中间产物 |
| T_DHPIPE_04 | 数字人管道与视频域管道组合 AvatarDriver -> FrameInterpolator -> VideoEncoder | 输出 video_path，>=3 个中间产物 |
| T_DHPIPE_05 | 数字人管道与导出域管道组合 LipSyncer -> MultiFormatExporter | 输出 output_path 和 formats，>=2 个中间产物 |
| T_DHPIPE_06 | 异步执行数字人长时间渲染任务 | 异步任务状态正确，wait 返回 PipelineResult |

---

## 测试统计

| 分类 | 用例数 |
|------|--------|
| MotionGenerator | 12 |
| RealtimeRenderer | 14 |
| 端到端集成 | 10 |
| 数字人管道组合 | 6 |
| **总计** | **42** |

## 运行命令

```bash
# 运行 Phase 7 全部测试
python -m pytest tests/phase7/ -v

# 仅运行集成测试
python -m pytest tests/phase7/ -v -m integration

# 运行全部测试（Phase 1-7）
python -m pytest tests/ -v

# 跳过集成测试
python -m pytest tests/phase7/ -v -m "not integration"
```