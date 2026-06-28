# Phase 1 测试套件验收清单

## 概览

| 文件 | 测试用例数 | 描述 |
|------|-----------|------|
| `test_types.py` | 8 组 (~30 个) | 数据类型测试 |
| `test_node.py` | 6 组 (~20 个) | 节点基类测试 |
| `test_registry.py` | 7 组 (~20 个) | 注册表测试 |
| `test_pipeline.py` | 10 组 (~25 个) | 管道测试 |
| `test_scheduler.py` | 6 组 (~20 个) | 调度器测试 |
| `test_events.py` | 5 组 (~20 个) | 事件总线测试 |
| `test_text_nodes.py` | 6 组 (~21 个) | 文本域节点测试 |
| `test_integration.py` | 4 组 (~8 个) | 端到端集成测试 |

---

## 1. 数据类型测试 (`test_types.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_TYPES_01 | 创建 MosaicData，设置和获取值 | 字典式读写操作正常，序列化/反序列化往返正确 |
| T_TYPES_02 | TextData 的创建、序列化、反序列化 | content/language/metadata 字段正确，validate 校验通过 |
| T_TYPES_03 | ImageData 的创建、序列化、反序列化 | PIL 图像 base64 编码/解码往返正确，size 保留 |
| T_TYPES_04 | AudioData 的创建、序列化、反序列化 | numpy 数组带 shape/dtype 元数据序列化往返正确 |
| T_TYPES_05 | VideoData 的创建、序列化、反序列化 | 帧列表（PIL 图像）序列化往返正确，fps 保留 |
| T_TYPES_06 | SubtitleData 的创建和转换 | segments 正确存储，validate 校验 start/end/text 必需键 |
| T_TYPES_07 | DocumentData 的创建和转换 | chunks 列表正确存储，validate 校验全部为字符串 |
| T_TYPES_08 | 类型校验 —— 传入错误类型时抛出异常 | _image_to_b64 拒绝非 PIL 图像；_array_to_dict 拒绝非 ndarray；tuple 保留语义 |

---

## 2. 节点基类测试 (`test_node.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_NODE_01 | Node 是抽象类，不能直接实例化 | 直接实例化 Node 或未实现抽象方法的子类抛出 TypeError |
| T_NODE_02 | 实现所有抽象方法后可以正常实例化 | mock 节点正常实例化，load/unload/run/describe 正常工作 |
| T_NODE_03 | __call__ 方法正确调用 run | __call__ 自动 load 然后调用 run，返回 MosaicData |
| T_NODE_04 | 上下文管理器 __enter__ 调用 load，__exit__ 调用 unload | with 块内 is_loaded=True，退出后 is_loaded=False |
| T_NODE_05 | is_loaded 状态正确反映 | 初始 unloaded，load 后 loaded，unload 后 unloaded |
| T_NODE_06 | describe 返回正确的 NodeSpec | NodeSpec 包含 name/domain/description/version/input_types/output_types |

---

## 3. 注册表测试 (`test_registry.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_REG_01 | 注册一个节点类，然后按名称获取 | register 后 get 返回实例，get_class 返回类，不存在则 KeyError |
| T_REG_02 | @registry.register 装饰器正常工作 | 装饰器自动注册，抽象类跳过，返回原类 |
| T_REG_03 | list_nodes 返回所有已注册节点 | 按名称排序，去重，返回 NodeSpec 列表 |
| T_REG_04 | list_nodes("text") 只返回文本域节点 | 按域过滤，text 域只含 text 节点 |
| T_REG_05 | list_domains 返回所有已注册的域 | 去重排序，返回域列表 |
| T_REG_06 | 获取不存在的节点名返回 None 或抛出友好错误 | get 抛出 KeyError，contains 返回 False |
| T_REG_07 | 重复注册同名节点的行为 | 不同类同名抛出 ValueError；同类重复注册不报错；unregister 移除 |

---

## 4. 管道测试 (`test_pipeline.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_PIPE_01 | 创建简单串行管道并执行 | 节点按序执行，累加器节点输出正确拼接 |
| T_PIPE_02 | 管道运算符 \| 语法正常工作 | node_a \| node_b 创建匿名 Pipeline，链式执行正确 |
| T_PIPE_03 | 运行后可以获取中间结果 | get_intermediate 按节点 id/name 获取中间产物 |
| T_PIPE_04 | 空管道抛出友好错误 | 空管道合法执行，add 方法追加节点 |
| T_PIPE_05 | 节点执行失败时管道正确报告错误 | RuntimeError 向上传播，不支持的元素类型抛出 TypeError |
| T_PIPE_06 | Pipeline 作为 Node 可以嵌套 | 嵌套管道 load/unload/run/describe 正常 |
| T_PIPE_07 | dry_run 模式只检查不执行 | 结构校验 + 类型匹配检查，不实际调用 run |
| T_PIPE_08 | Branch 和 Merge 的基本用法 | fan-out/条件分支 + dict/flatten/custom merge |
| T_PIPE_09 | 循环依赖检测，DAG 合法性检查 | validate 通过合法管道，dry_run 检测类型不匹配 |
| T_PIPE_10 | 进度回调被正确触发 | Context 回调 / callbacks 参数的 node_start/node_end 被触发 |

---

## 5. 调度器测试 (`test_scheduler.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_SCHED_01 | track 和 status 基本功能 | track 注册节点，status 返回完整字典 |
| T_SCHED_02 | ensure_loaded 加载模型 | 自动 track + load，已加载节点不重复加载，触发 MODEL_LOAD 事件 |
| T_SCHED_03 | release 释放模型 | release 卸载节点，触发 MODEL_UNLOAD 事件，release_all 释放全部 |
| T_SCHED_04 | 显存不足时 LRU 淘汰 | 超限时按 LRU 淘汰最久未使用节点，无法腾出抛 MemoryError |
| T_SCHED_05 | set_memory_limit 生效 | 修改上限，0 表示不限制，低于当前占用抛 ValueError |
| T_SCHED_06 | 无 GPU 时优雅降级 | CPU 模式不抛 MemoryError，device="cpu"，memory_used_gb=0 |

---

## 6. 事件总线测试 (`test_events.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_EVT_01 | on 注册监听，事件触发时 callback 被调用 | emit 触发回调，通配符订阅接收所有事件 |
| T_EVT_02 | off 取消监听后不再触发 | off 后回调不再触发，clear 清除全部/指定类型 |
| T_EVT_03 | callback 异常不影响管道运行 | 异常捕获不传播，其他回调正常触发 |
| T_EVT_04 | 多个 callback 按注册顺序触发 | 按 on 注册顺序依次调用 |
| T_EVT_05 | 事件对象包含正确的元数据 | timestamp > 0，payload 包含传入数据，repr 包含关键信息 |

---

## 7. 文本域节点测试 (`test_text_nodes.py`)

### TextGenerator

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_GEN_01 | 基本生成，返回非空文本 | generated_text 非空字符串，input_tokens/output_tokens > 0 |
| T_GEN_02 | 自定义参数生效 | max_new_tokens/temperature/top_p/do_sample 传入生效 |
| T_GEN_03 | describe 返回正确的模型信息 | spec.name="text-generator"，model_info 含模型名称 |

### Chat

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CHAT_01 | 单轮对话返回回复 | reply 非空，messages 含 assistant 回复 |
| T_CHAT_02 | 多轮对话保持上下文 | 返回的 messages 长度 = 原始 + 1 |
| T_CHAT_03 | system_prompt 生效 | system_prompt 传入后正常执行 |

### TextRewriter

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_REWRITE_01 | 基本改写，返回不同文本 | rewritten_text 和 original_text 都存在 |
| T_REWRITE_02 | 指定 instruction 改写 | instruction 参数传入生效 |
| T_REWRITE_03 | 保留原文语义 | original_text 与输入一致 |

### Translator

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_TRANS_01 | 中译英 | translated_text 存在，target_language="en" |
| T_TRANS_02 | 英译中 | translated_text 存在，target_language="zh" |
| T_TRANS_03 | auto 源语言检测 | source_language="auto"，describe 含 translation_mode |

### TextSummarizer

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_SUM_01 | 长文本摘要，压缩比 < 1 | summary 非空，compression_ratio 存在 |
| T_SUM_02 | 短文本直接返回 | compression_ratio=1.0，summary==原文，note 含 "skipped" |
| T_SUM_03 | bullet_points 风格 | style="bullet_points" 正常执行 |

### TextClassifier

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CLS_01 | 单标签分类，返回正确格式 | predicted_label 在 labels 中，含 scores 和 method |
| T_CLS_02 | 返回 scores 字典 | scores 为 dict，每个 label 都有分数 |
| T_CLS_03 | 多标签分类 | predicted_labels 为 list，含 scores 和 method |

---

## 8. 端到端集成测试 (`test_integration.py`)

| 用例 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_E2E_01 | "文本生成 → 翻译" 管道完整运行 | 管道执行成功，translated_text 存在，中间结果可访问 |
| T_E2E_02 | "文本生成 → 摘要 → 分类" 三节点管道 | 管道执行成功，predicted_label 在 labels 中 |
| T_E2E_03 | 运行过程中事件被正确触发 | NODE_START 和 NODE_COMPLETE 事件数量 >= 2 |
| T_E2E_04 | 运行结束后中间结果可访问 | 所有节点中间产物可获取，| 运算符链式管道也支持 |

---

## 运行方式

```bash
# 运行所有 Phase 1 测试
cd /workspace/mosaic
python -m pytest tests/phase1/ -v

# 仅运行核心框架测试（跳过集成测试）
python -m pytest tests/phase1/ -v -m "not integration"

# 仅运行集成测试
python -m pytest tests/phase1/ -v -m integration

# 运行特定文件
python -m pytest tests/phase1/test_types.py -v
```