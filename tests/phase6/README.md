# Phase 6 测试验收清单

## 概述

Phase 6 测试覆盖一致性域（Consistency Domain），包括 CrossFrameConsistency（跨帧一致性）、IdentityKeeper（身份保持）、StyleKeeper（风格保持），以及端到端集成场景和一致性管道组合。

- **测试文件数**：4（含 conftest.py）
- **测试用例总数**：35
- **测试框架**：pytest
- **Mock 策略**：全部使用 mock diffusers/torch 模块 + mock _run_pipeline，不依赖真实模型或外部文件

---

## 一、CrossFrameConsistency 节点测试（test_cross_frame_consistency.py）— 14 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CF_01 | 基本跨帧生成，输出 images 列表 | images 为非空 list，每帧为 PIL Image |
| T_CF_02 | 输出帧数与 prompts 列表长度一致 | len(images) == len(prompts) |
| T_CF_03 | consistency_scores 列表长度正确（等于帧数） | len(consistency_scores) == len(images) |
| T_CF_04 | average_consistency 在合理范围（0-1） | 0.0 <= average_consistency <= 1.0 |
| T_CF_05 | reference_image 可选参数生效 | 传入 sample_face_image 后 output 包含 reference_image |
| T_CF_06 | character_description 在输出中正确返回 | output["character_description"] 与输入一致 |
| T_CF_07 | consistency_strength 参数生效 | 不同 strength（0.3/0.95）节点均正常生成 |
| T_CF_08 | 指定 seed 可复现 | 两次相同 seed 输出相同的 images 数量和 seed 值 |
| T_CF_09 | 单帧输入（prompts 只有 1 个元素）正常工作 | 生成 1 张图片，consistency_scores 长度为 1 |
| T_CF_10 | 多帧输入（prompts 有 7 个元素）正常工作 | 帧数与 prompts 数一致 |
| T_CF_11 | method 参数切换（consistory / story-diffusion / all-in-one） | 三种方法都能实例化且 describe 正常 |
| T_CF_12 | describe 返回正确信息 | name/domain/version/model_info 均正确 |
| T_CF_13 | load/unload 后 is_loaded 状态正确 | 加载前 False，加载后 True，卸载后 False |
| T_CF_14 | 进度事件在多帧生成中被触发 | 至少收到 5 个 NODE_COMPLETE 事件，含 progress 信息 |

---

## 二、端到端集成测试（test_integration.py）— 8 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_E2E_P6_01 | IdentityKeeper + ImageToImage 身份保持后风格化 | IdentityKeeper 生成图片，mock i2i 正常处理 |
| T_E2E_P6_02 | StyleKeeper + ImageToImage 风格保持后进一步处理 | StyleKeeper 生成图片，mock i2i 正常处理 |
| T_E2E_P6_03 | CrossFrameConsistency + VideoEncoder 跨帧一致后编码 | CF 输出 3 帧，encoder 正确接收 frame_count=3 |
| T_E2E_P6_04 | TextToImage + IdentityKeeper 先生成参考图再保持身份 | T2I 生成参考图，IdentityKeeper 使用参考图生成 |
| T_E2E_P6_05 | 一致性节点与 Pipeline 串联（身份 + 风格） | Pipeline 串联两个节点，输出非空 |
| T_E2E_P6_06 | 运行过程中事件被正确触发 | NODE_START 和 NODE_COMPLETE 事件均被触发 |
| T_E2E_P6_07 | PipelineResult 包含正确信息 | pipeline_name/duration/output/intermediate 均正确 |
| T_E2E_P6_08 | 中间产物可单独取出 | 中间产物中含 image 类型数据 |

---

## 三、一致性管道组合测试（test_consistency_pipeline.py）— 6 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CPIPE_01 | 用 Pipeline 声明式组装身份保持流程 | PipelineResult 输出含 image 和 identity_score |
| T_CPIPE_02 | 用 Pipeline 声明式组装风格保持流程 | PipelineResult 输出含 image |
| T_CPIPE_03 | 用 Pipeline 声明式组装跨帧一致流程 | PipelineResult 输出含 images（3 帧） |
| T_CPIPE_04 | 一致性管道与图像域管道无缝对接（T2I → IdentityKeeper） | 跨域管道成功，中间产物 >= 2 个 |
| T_CPIPE_05 | 一致性管道与视频域管道无缝对接（CF → VideoEncoder） | 跨域管道成功，输出含 video_path，frame_count=3 |
| T_CPIPE_06 | 多个一致性节点串联（IdentityKeeper → StyleKeeper） | 串联成功，中间产物 >= 2 个 |

---

## 测试统计

| 分类 | 用例数 |
|------|--------|
| CrossFrameConsistency | 14 |
| 端到端集成 | 8 |
| 一致性管道组合 | 6 |
| **总计** | **28** |

## 运行命令

```bash
# 运行 Phase 6 全部测试
python -m pytest tests/phase6/ -v

# 仅运行集成测试
python -m pytest tests/phase6/ -v -m integration

# 运行全部测试（Phase 1-6）
python -m pytest tests/ -v

# 跳过集成测试
python -m pytest tests/phase6/ -v -m "not integration"
```