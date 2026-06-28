# Phase 5 测试验收清单

## 概述

Phase 5 测试覆盖 RAG 域（检索增强生成），包括 Retriever（向量检索）、CitationGenerator（引用生成）、以及端到端集成场景和 RAG 管道组合。

- **测试文件数**：4
- **测试用例总数**：33
- **测试框架**：pytest
- **Mock 策略**：全部使用 mock 嵌入模型 + mock FAISS 索引 + mock LLM，不依赖真实模型或外部文件

---

## 一、Retriever 节点测试（test_retriever.py）— 12 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_RET_01 | 基本检索，返回 results 列表 | results 为非空 list |
| T_RET_02 | top_k 参数生效（返回数量正确，不超过 top_k） | len(results) <= top_k |
| T_RET_03 | score_threshold 过滤生效（高分过滤后只保留高分结果） | 所有结果 score >= threshold |
| T_RET_04 | 结果按 score 降序排列 | scores 序列递减 |
| T_RET_05 | top_score 是最高分 | top_score == max(scores) |
| T_RET_06 | result_count 与实际结果数量一致 | result_count == len(results) |
| T_RET_07 | query 字段正确回传 | result["query"] == 输入 query |
| T_RET_08 | 每个结果包含 content、score、source、metadata 四个字段 | 四个字段都存在 |
| T_RET_09 | filter_metadata 元信息过滤 | 过滤后结果 topic 均为指定值 |
| T_RET_10 | collection_name 指定不同集合 | 不同 collection 返回不同结果 |
| - | 缺少 query 时抛出 ValueError | ValueError 被抛出 |
| - | describe 返回正确信息 | 包含 name/domain/output_types |

---

## 二、CitationGenerator 节点测试（test_citation_generator.py）— 16 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_CITE_01 | 基本引用生成，输出 answer 非空 | answer 为非空字符串 |
| T_CITE_02 | citations 列表非空 | citations 为非空 list |
| T_CITE_03 | 每个 citation 包含 citation_id、source、content、score | 四个字段都存在 |
| T_CITE_04 | inline 引用风格，answer 中包含 [1] 标记 | "[1]" in answer |
| T_CITE_05 | footnote 引用风格 | answer 非空 |
| T_CITE_06 | academic 引用风格 | answer 非空 |
| T_CITE_07 | sources_used 数量正确 | sources_used == len(citations) |
| T_CITE_08 | language 参数生效（zh 使用中文 prompt，en 使用英文 prompt） | zh 含 "参考资料"，en 含 "Reference materials" |
| T_CITE_09 | 检索结果为空时的回答处理（应抛出 ValueError） | ValueError 被抛出 |
| T_CITE_10 | temperature 参数生效 | temperature=0 时正常执行 |
| T_CITE_11 | describe 返回正确信息 | 包含 name/domain/output_types |
| - | _parse_citations 正确解析单个引用 | 解析出 1 个引用 |
| - | _parse_citations 正确解析多个引用 | 解析出 3 个引用 |
| - | _parse_citations 正确解析范围引用 [1-3] | 解析出 3 个引用 |
| - | _parse_citations 无引用时返回空列表 | 返回 [] |
| - | _build_context 创建编号条目 | context 包含 [1][2][3] |

---

## 三、端到端集成测试（test_integration.py）— 8 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_E2E_P5_01 | 文档解析 → 向量化索引 完整流程 | 解析得到 DocumentData，索引后 indexed_count > 0 |
| T_E2E_P5_02 | 文档解析 → 向量化 → 检索 完整流程 | 检索结果非空，每个结果包含 content/score/source/metadata |
| T_E2E_P5_03 | 文档解析 → 向量化 → 检索 → 引用生成 完整 RAG 管道 | 生成 answer 和 citations |
| T_E2E_P5_04 | 解析多个文档建立索引后统一检索 | 多个文档索引后可检索到结果 |
| T_E2E_P5_05 | 索引持久化后重新加载并检索（结果一致） | 索引文件保存成功，重新加载后可检索 |
| T_E2E_P5_06 | 不同 query 返回不同检索结果 | 两次查询的 content 集合不同 |
| T_E2E_P5_07 | 运行过程中事件被正确触发 | NODE_START 和 NODE_COMPLETE 事件被触发 |
| T_E2E_P5_08 | PipelineResult 包含正确信息 | success=True, output 含 results, pipeline_name 正确 |

---

## 四、RAG 管道组合测试（test_rag_pipeline.py）— 5 个用例

| 测试 ID | 描述 | 预期结果 |
|---------|------|----------|
| T_RPIPE_01 | 用 Pipeline 声明式组装完整 RAG 流程 | 管道成功执行，输出含 answer 和 citations |
| T_RPIPE_02 | RAG 管道支持多次查询（索引一次，查询多次） | 两次不同查询返回不同 answer |
| T_RPIPE_03 | RAG 管道与文本域管道组合（生成查询 → RAG 检索 → 回答） | 跨域管道成功，中间产物可检查 |
| T_RPIPE_04 | 中间产物可检查（索引数据、检索结果可单独取出） | 检索结果含 score=0.95，引用含 citation_id |
| - | 中间产物数量正确 | len(intermediate) >= 2 |

---

## 测试统计

| 分类 | 用例数 |
|------|--------|
| Retriever | 12 |
| CitationGenerator | 16 |
| 端到端集成 | 8 |
| RAG 管道组合 | 5 |
| **总计** | **41** |

## 运行命令

```bash
# 运行 Phase 5 全部测试
python -m pytest tests/phase5/ -v

# 仅运行集成测试
python -m pytest tests/phase5/ -v -m integration

# 运行全部测试（Phase 1-5）
python -m pytest tests/phase1/ tests/phase2/ tests/phase3/ tests/phase4/ tests/phase5/ -v

# 跳过集成测试
python -m pytest tests/phase5/ -v -m "not integration"
```