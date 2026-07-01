# Phase 2 测试验收清单

## 测试用例汇总

| 编号 | 测试文件 | 测试用例数 | 描述 |
|------|----------|------------|------|
| 1 | `test_image_types.py` | 4 | 图像数据类型测试 |
| 2 | `test_text_to_image.py` | 7 | 文生图节点测试 |
| 3 | `test_image_to_image.py` | 5 | 图生图节点测试 |
| 4 | `test_inpainting.py` | 4 | 局部重绘节点测试 |
| 5 | `test_upscaler.py` | 4 | 超分辨率节点测试 |
| 6 | `test_background_remover.py` | 4 | 去背景节点测试 |
| 7 | `test_stylizer.py` | 5 | 风格化节点测试 |
| 8 | `test_image_base.py` | 4 | 图像域基类测试 |
| 9 | `test_pipeline_parallel.py` | 8 | Pipeline 并行增强测试 |
| 10 | `test_intermediate.py` | 7 | 中间产物检查测试 |
| 11 | `test_pipeline_result.py` | 7 | PipelineResult 测试 |
| 12 | `test_integration.py` | 7 | Phase 2 端到端集成测试 |
| **合计** | | **66** | |

---

## 1. test_image_types.py（图像数据类型测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_IMGTYPE_01` | ImageData 创建，包含 PIL.Image | 创建 ImageData 实例成功，内部持有正确的 PIL.Image 对象，mode 和 size 属性与原始图片一致 |
| `T_IMGTYPE_02` | ImageData 序列化/反序列化（图片转 base64） | 序列化后得到有效的 base64 字符串；反序列化后恢复的 PIL.Image 与原图逐像素一致（或差异在可接受范围内） |
| `T_IMGTYPE_03` | 不同尺寸图片的 ImageData 处理 | 分别使用 64x64、256x256、1024x1024 等多种尺寸图片创建 ImageData，均能正确处理，size 属性反映实际尺寸，序列化与反序列化正常 |
| `T_IMGTYPE_04` | RGBA 图片的 ImageData 处理 | RGBA 四通道图片创建 ImageData 成功，mode 为 "RGBA"；序列化/反序列化后 alpha 通道信息保留完整 |

---

## 2. test_text_to_image.py（文生图节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_T2I_01` | 基本文生图，输出是 PIL.Image 列表 | 调用 `generate(prompt="一只猫")` 返回 `list[PIL.Image.Image]` 类型的非空列表，每张图片可正常打开和显示 |
| `T_T2I_02` | 自定义尺寸（512x512）生效 | 指定 `width=512, height=512`，输出图片尺寸均为 512x512（或与模型输出尺寸一致，在合理误差范围内） |
| `T_T2I_03` | 指定 seed 可复现 | 使用相同 seed 调用两次，生成的图片逐像素相等或 SSIM 指标 >= 0.99 |
| `T_T2I_04` | negative_prompt 参数生效 | 传入 `negative_prompt` 参数不报错，且生成结果与不传时有明显差异（视觉上或通过哈希/SSIM 验证） |
| `T_T2I_05` | num_images > 1 时返回多张图 | 设置 `num_images=3`，返回列表中包含 3 张互不相同的 PIL.Image |
| `T_T2I_06` | describe 返回正确信息，含许可证 | 调用 `describe()` 返回非空字符串，包含模型名称、用途说明和许可证信息（如 "openrail++" 或类似字样） |
| `T_T2I_07` | load/unload 后 is_loaded 状态正确 | 初始 `is_loaded` 为 False；`load()` 后变为 True；`unload()` 后变为 False；再次 `load()` 后恢复为 True |

---

## 3. test_image_to_image.py（图生图节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_I2I_01` | 基本图生图，输出尺寸与输入相关 | 输入一张 512x512 图片，输出图片尺寸与输入一致（或为模型原生输出尺寸，但需在合理范围内，如 512x512 或 768x768） |
| `T_I2I_02` | strength=0 时接近原图 | 设置 `strength=0.0`，输出图片与原图在视觉上几乎一致，SSIM >= 0.95 |
| `T_I2I_03` | strength=1 时变化最大 | 设置 `strength=1.0`，输出图片与原图有显著差异，SSIM < 0.8 |
| `T_I2I_04` | 输入图片不是 8 的倍数时自动处理 | 输入 500x500 图片（非 8 的倍数），节点不报错，自动调整尺寸并正常生成输出 |
| `T_I2I_05` | describe 返回正确信息 | 调用 `describe()` 返回非空字符串，包含图生图相关描述和模型信息 |

---

## 4. test_inpainting.py（局部重绘节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_INP_01` | 基本局部重绘，遮罩区域被重绘 | 提供原图和 mask（白色区域表示需要重绘），输出图片中 mask 标记区域的内容与原图明显不同，非 mask 区域保持不变 |
| `T_INP_02` | image 和 mask 尺寸不一致时自动处理 | 输入 512x512 图片和 256x256 mask，节点自动将 mask resize 到与图片相同尺寸，不报错且正常生成结果 |
| `T_INP_03` | mask 不是二值图时自动阈值处理 | 输入灰度 mask（值在 0-255 范围），节点自动做二值化处理，正常生成结果 |
| `T_INP_04` | 全白 mask 时整张图被重绘 | 输入全白（255）mask，输出整张图均被重绘，与输入图整体有明显差异 |

---

## 5. test_upscaler.py（超分辨率节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_UP_01` | 4x 放大，输出尺寸正确 | 输入 128x128 图片，输出图片尺寸为 512x512（4x 放大），宽高均为输入的 4 倍 |
| `T_UP_02` | 输入已是高分辨率时的处理 | 输入 2048x2048 图片，节点正常处理（可能跳过放大或自动 resize 后处理），不报错，不 OOM |
| `T_UP_03` | tiny_image（32x32）的处理，检查警告 | 输入 32x32 极小图片，生成警告日志（提示图片过小可能影响效果），但输出仍正常生成且尺寸为 128x128 |
| `T_UP_04` | describe 标注放大倍数 | 调用 `describe()` 返回的字符串中包含放大倍数信息（如 "4x" 或 "scale" 等字样） |

---

## 6. test_background_remover.py（去背景节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_BGR_01` | 基本去背景，输出 RGBA 模式 | 输入 RGB 图片，输出图片 mode 为 "RGBA"，透明区域（alpha=0）对应原图背景区域 |
| `T_BGR_02` | 输出包含 mask（灰度图） | 除处理后的 RGBA 图片外，还输出一个灰度 mask 图，其中白色区域表示前景，黑色区域表示背景 |
| `T_BGR_03` | RGBA 输入图片的处理 | 输入已有 alpha 通道的 RGBA 图片，节点正确处理，不报错，输出仍为合理的 RGBA 图片 |
| `T_BGR_04` | 大图自动 resize 处理 | 输入 4096x4096 大图，节点自动 resize 到合适尺寸处理后输出，不 OOM，不报错 |

---

## 7. test_stylizer.py（风格化节点测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_STY_01` | 指定 "oil painting" 风格，输出非空 | 指定风格为 "oil painting"，输出为非空 PIL.Image 列表，图片在视觉上具有油画风格特征 |
| `T_STY_02` | strength 参数生效 | 设置 `strength=0.0` 时输出接近原图，`strength=1.0` 时风格化效果最强，两者有明显差异 |
| `T_STY_03` | prompt_extra 额外提示词生效 | 传入 `prompt_extra="blue tones"`，输出图片整体色调偏向蓝色，与不加额外提示词时有视觉差异 |
| `T_STY_04` | 指定 seed 可复现 | 使用相同 seed 和风格参数调用两次，输出图片一致（SSIM >= 0.99） |
| `T_STY_05` | 各种预设风格都可以运行 | 遍历所有预设风格（如 "oil painting"、"watercolor"、"sketch"、"anime" 等），每种风格均能正常运行不报错 |

---

## 8. test_image_base.py（图像域基类测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_IBASE_01` | BaseImageNode 的公共参数传递正确 | 创建继承 BaseImageNode 的子类实例，传入 `device`、`model_id` 等公共参数，子类能正确读取并应用这些参数 |
| `T_IBASE_02` | dtype 参数生效（float16 vs float32） | 分别以 `dtype=float16` 和 `dtype=float32` 加载模型，检查模型参数的实际 dtype 与传入值一致 |
| `T_IBASE_03` | enable_attention_slicing 参数生效 | 设置 `enable_attention_slicing=True`，模型加载后 attention slicing 功能已启用，可通过模型内部属性验证 |
| `T_IBASE_04` | 无 GPU 时降级到 CPU | 在无 CUDA 环境中，节点自动将 device 降级为 "cpu"，不抛出异常，且能正常加载模型（可能在 CPU 上较慢） |

---

## 9. test_pipeline_parallel.py（Pipeline 并行增强测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_PAR_01` | Branch 下两个 mock 节点真正并行执行 | 使用带延迟的 mock 节点，两个分支的总耗时接近单个延迟时间（而非两个延迟之和），证明并行执行生效 |
| `T_PAR_02` | Merge 默认合并策略，输出包含所有分支结果 | 两路分支分别输出不同值，Merge 节点默认将结果合并为一个列表或字典，包含所有分支的输出 |
| `T_PAR_03` | Merge(keep="branch_name") 只保留指定分支 | 设置 `Merge(keep="branch_a")`，结果中仅包含 branch_a 的输出，不包含其他分支的输出 |
| `T_PAR_04` | 自定义 Merge 函数 | 传入自定义合并函数（如 `lambda results: sum(results)`），Merge 节点按自定义逻辑合并结果，输出符合预期 |
| `T_PAR_05` | 并行分支输入分配正确 | 传入一个字典，每个分支收到其对应 key 的值，验证各分支的输入数据正确无误 |
| `T_PAR_06` | fail_fast=True 时一个失败立即停止 | 其中一个分支抛出异常，设置 `fail_fast=True`，整体立即抛出异常，其他分支被终止，耗时明显短于等待所有分支完成 |
| `T_PAR_07` | fail_fast=False 时收集所有结果和错误 | 其中一个分支失败，设置 `fail_fast=False`，其他分支正常完成，结果中包含成功分支的输出和失败分支的异常信息 |
| `T_PAR_08` | 并行分支数量 > 2 时正常工作（3 路并行） | 创建 3 路并行分支，每路使用不同延迟的 mock 节点，总耗时接近单路最大延迟，3 路输出均正确 |

---

## 10. test_intermediate.py（中间产物检查测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_INT_01` | 运行后 get_intermediate 获取指定节点输出 | Pipeline 运行后，调用 `get_intermediate("node_a")` 返回该节点的实际输出值，类型与节点定义一致 |
| `T_INT_02` | list_intermediate 返回所有中间产物节点名 | 调用 `list_intermediate()` 返回一个列表，包含 Pipeline 中所有产生中间产物的节点名称，不含不产生输出的节点 |
| `T_INT_03` | get_all_intermediate 返回全部 | 调用 `get_all_intermediate()` 返回一个字典，key 为节点名，value 为该节点的输出值，所有中间产物节点均在字典中 |
| `T_INT_04` | snapshot 导出可序列化字典 | 调用 `snapshot()` 返回一个字典，可成功通过 `json.dumps()` 序列化（或通过自定义序列化器处理），不抛出异常 |
| `T_INT_05` | save_snapshot 保存到文件 | 调用 `save_snapshot("snapshot.json")` 成功创建文件，文件内容可读，格式正确 |
| `T_INT_06` | load_snapshot 从文件加载 | 从 `save_snapshot` 保存的文件中调用 `load_snapshot("snapshot.json")`，恢复的中间产物与原始数据一致 |
| `T_INT_07` | max_intermediate 限制存储数量 | 设置 `max_intermediate=3`，Pipeline 中有 5 个节点产生输出，运行后 `list_intermediate()` 最多返回 3 个节点名，按策略保留最近或最重要的 |

---

## 11. test_pipeline_result.py（PipelineResult 测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_RES_01` | 成功运行后 result.success == True | 运行一个正常完成的 Pipeline，`result.success` 为 `True` |
| `T_RES_02` | 失败运行后 result.success == False | 运行一个包含会抛出异常的节点的 Pipeline，`result.success` 为 `False` |
| `T_RES_03` | result.duration > 0 | 任意 Pipeline 运行后，`result.duration` 为大于 0 的浮点数，单位为秒 |
| `T_RES_04` | result.node_durations 包含每个节点的耗时 | `result.node_durations` 是一个字典，key 为各节点名，value 为该节点的执行耗时（秒），所有节点均在字典中 |
| `T_RES_05` | result.summary() 输出可读摘要 | 调用 `result.summary()` 返回一个字符串，包含成功/失败状态、总耗时、各节点耗时、错误信息（如有）等关键信息 |
| `T_RES_06` | result.to_dict() 可序列化 | 调用 `result.to_dict()` 返回一个字典，可通过 `json.dumps()` 成功序列化，不抛出异常 |
| `T_RES_07` | result.failed_nodes 在有错误时返回正确列表 | 运行包含失败节点的 Pipeline，`result.failed_nodes` 返回一个列表，包含所有失败节点的名称和对应的异常信息 |

---

## 12. test_integration.py（Phase 2 端到端集成测试）

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| `T_E2E_P2_01` | "文本生成 → 文生图" 跨域管道 | 构建 TextGenerationNode → TextToImageNode 的 Pipeline，输入文本 prompt，最终输出为 PIL.Image 列表，无异常 |
| `T_E2E_P2_02` | "文生图 → 去背景" 图像域内管道 | 构建 TextToImageNode → BackgroundRemoverNode 的 Pipeline，输出 RGBA 模式图片，背景被成功移除 |
| `T_E2E_P2_03` | "文生图 → 同时(去背景 + 风格化)" 并行分支 | 构建 TextToImageNode → Branch → [BackgroundRemoverNode, StylizerNode] 的 Pipeline，两路分支均正常执行并输出结果 |
| `T_E2E_P2_04` | 并行分支后 Merge，结果可访问 | 在 `T_E2E_P2_03` 的基础上添加 Merge 节点，合并后结果中包含去背景和风格化两个分支的输出，均可正常访问 |
| `T_E2E_P2_05` | 运行后中间产物（图片）可单独取出并保存为文件 | Pipeline 运行完成后，通过 `get_intermediate` 获取中间产物图片，调用 `save()` 方法成功保存为 PNG/JPEG 文件，文件存在且可正常打开 |
| `T_E2E_P2_06` | 整个流程事件被正确触发（node_start、node_complete） | 注册事件回调，Pipeline 运行期间每个节点都触发 `node_start` 和 `node_complete` 事件，事件参数包含正确的节点名和时间戳 |
| `T_E2E_P2_07` | PipelineResult 包含正确的摘要信息 | 运行完整 Pipeline 后，`result.summary()` 包含所有节点名称、各节点耗时、总耗时、成功状态，且信息准确无误 |

---

## 验收标准

所有 66 个测试用例需全部通过（PASS），方可视为 Phase 2 测试验收通过。具体要求如下：

- **通过率要求**：100%（所有测试用例必须通过，不允许有 FAIL 或 SKIP）
- **执行环境**：需在有 GPU 的环境中执行（部分测试如 `T_IBASE_04` 需在无 GPU 环境中验证降级行为，可单独执行）
- **依赖项**：确保所有模型文件已下载，Python 依赖已安装（参见 `pyproject.toml`）
- **执行命令**：`pytest tests/phase2/ -v --tb=short`
- **报告格式**：执行完成后生成 JUnit XML 报告（`pytest tests/phase2/ -v --tb=short --junitxml=phase2_results.xml`）