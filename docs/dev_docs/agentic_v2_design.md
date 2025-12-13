# Agentic Retrieval V2 设计文档

## 概述

本文档描述 Agentic Retrieval V2 的设计方案，核心改进是将 Multi-Query 生成前移到 Round 1，基于问题类型差异化处理，以提升首轮召回率、减少 Round 2 触发。

**设计原则**: 不修改现有 `agentic.py` 代码，新建独立文件，复用公共函数。

---

## 1. 背景与动机

### 1.1 当前问题

- **72% 的查询触发 Round 2**：现有流程在 Round 1 只做单 Query 检索，Sufficiency Check 经常判定 insufficient
- **Multi-Query 延迟**：只有在 Round 2 才生成 Multi-Query，导致复杂问题延迟高 (~2600ms)
- **问题类型未差异化**：简单问题（如 ATTRIBUTE_LOCATION）和复杂问题（如 EVENT_AGGREGATION）使用相同流程

### 1.2 改进目标

| 指标 | 当前 | 预期 |
|------|------|------|
| Insufficient 比例 | 72% | 35-45% |
| 简单问题延迟 | ~1000ms | ~1000ms (无变化) |
| 复杂问题延迟 | ~2600ms | ~1400ms (如果首轮成功) |

---

## 2. 新流程设计

```
Query
  │
  ▼
问题分类 (QuestionClassifier)
  │
  ├─── 简单问题 (ATTRIBUTE_LOCATION, ATTRIBUTE_IDENTITY 等高置信度)
  │         │
  │         ▼
  │    直接单 Query Hybrid Search → Top 20 → Rerank Top 10 → Sufficiency Check
  │
  └─── 复杂问题 (EVENT_*, COUNTING, REASONING_* 等)
            │
            ▼
       Multi-Query 生成 (2-3个) → 并行 Hybrid Search → RRF 融合 Top 20 → Rerank Top 10
            │
            ▼
       Sufficiency Check
            │
            ├─── Sufficient → Return Top 20 (+ Cluster Expansion)
            │
            └─── Insufficient → Round 2
                      │
                      ▼
                 基于 missing_info 生成补充 Query
                      +
                 每个 Query 扩大召回 (top_n: 20 → 30)
                      │
                      ▼
                 合并 + Final Rerank → Top 20
```

---

## 3. 问题类型差异化处理

### 3.1 类型策略矩阵

| 问题类型 | 策略 | 改写重点 |
|----------|------|----------|
| `ATTRIBUTE_LOCATION` | 跳过 Multi-Query | - |
| `ATTRIBUTE_IDENTITY` | 跳过 Multi-Query | - |
| `EVENT_TEMPORAL` | **特殊处理** | 时间粒度变体 + 事件同义词 |
| `EVENT_ACTIVITY` | Multi-Query | 活动同义词 + 上下位概念 |
| `EVENT_AGGREGATION` | Multi-Query | 完整性 + 不同方面 |
| `COUNTING` | Multi-Query | 列举变体 + 完整覆盖 |
| `REASONING_*` | Multi-Query | 多角度证据 |
| `GENERAL` | Multi-Query | 同义词扩展 |

### 3.2 时间问题 (EVENT_TEMPORAL) 特殊处理

**问题特点**: 92% 准确率，错误主要来自相对时间表达（"The week before X"）

**改写策略示例**:
```
原始: "When did John have his first firefighter call-out?"

改写为:
1. 精确实体查询: "John firefighter call-out first time"
2. 时间上下文查询: "John volunteer firefighter July 2023"
3. 事件同义查询: "John firefighting started when"
```

**Prompt 要点**:
- 提取核心实体（人名 + 事件动词）
- 生成不同时间粒度（日期/周/月/年）
- 包含事件的同义表达（call-out → started, began, first）
- **不要**猜测具体日期，让检索去找

---

## 4. Round 2 增强逻辑

当 Round 1 insufficient 时：

### 4.1 基于 missing_info 生成补充 Query
- 利用 Sufficiency Check 返回的 `missing_information` 列表
- 生成 1-2 个针对性补充 query

### 4.2 扩大召回范围
- 每个 query 的 `top_n` 从 20 增加到 30
- RRF 融合后取更多候选 (Top 30)

### 4.3 合并策略
- Round 1 结果 + Round 2 新结果去重合并
- Final Rerank 选出 Top 20

---

## 5. 实现步骤

### Step 1: 新建 Prompt 文件

**文件**: `src/prompts/memory/en/eval/search/type_aware_multi_query_prompts.py`

核心内容：
- `TYPE_AWARE_PROMPTS`: 不同问题类型的 Prompt 模板字典
- `SKIP_MULTI_QUERY_TYPES`: 跳过 Multi-Query 的问题类型集合
- `should_use_multi_query()`: 判断函数

### Step 2: 新建 agentic_v2.py

**文件**: `src/retrieval/offline/pipelines/agentic_v2.py`

复用的公共函数：
- `hybrid_search_with_rrf` - from search_utils.py
- `multi_rrf_fusion` - from search_utils.py
- `reranker_search` - from rerank.py
- `check_sufficiency` - from llm_utils.py
- `generate_multi_queries` - from llm_utils.py (Round 2)

新增函数：
- `generate_multi_queries_by_type()`: 基于问题类型在首轮生成 Multi-Query
- `agentic_retrieval_v2()`: V2 主流程

### Step 3: 更新配置

**文件**: `config/eval/systems/parallax.yaml`

> **注意**: MemUnit 数量约 37-93 个/conv，平均 ~67 个

```yaml
retrieval:
  mode: "agentic_v2"  # 新增模式

  # V2 专用配置
  agentic_v2:
    # Multi-Query 配置
    skip_high_confidence_types: true
    confidence_threshold: 0.85
    num_queries: 3

    # Round 1 配置
    round1_per_query_top_n: 20
    round1_fusion_top_n: 20
    round1_rerank_top_n: 10

    # Round 2 配置
    round2_per_query_top_n: 30
    round2_fusion_top_n: 30
    final_rerank_top_n: 20
```

### Step 4: 接入评估流程

在评估入口根据 `retrieval.mode` 选择调用：
```python
if config.retrieval.mode == "agentic_v2":
    from retrieval.offline.pipelines.agentic_v2 import agentic_retrieval_v2
    results = await agentic_retrieval_v2(query, config, ...)
else:
    from retrieval.offline.pipelines.agentic import agentic_retrieval
    results = await agentic_retrieval(query, config, ...)
```

---

## 6. 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/retrieval/offline/pipelines/agentic_v2.py` | **新增** | V2 主流程 |
| `src/prompts/memory/en/eval/search/type_aware_multi_query_prompts.py` | **新增** | 类型相关 prompt |
| `config/eval/systems/parallax.yaml` | 更新 | 新增 `agentic_v2` 配置 |
| `src/retrieval/offline/pipelines/agentic.py` | **不修改** | 保留原有流程 |

---

## 7. 测试计划

1. **单元测试**: 验证 `agentic_v2` 流程独立运行
2. **对比实验**:
   - `mode: "agentic"` vs `mode: "agentic_v2"`
   - 指标：准确率、Insufficient 比例、延迟
3. **回滚方案**: 只需改配置 `mode: "agentic"` 即可回退

---

## 8. 变更日志

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2025-12-13 | 1.0 | 初始设计文档 |
| 2025-12-13 | 1.1 | 完成实现：新增 agentic_v2.py、type_aware_multi_query_prompts.py，更新配置和评估入口 |
| 2025-12-13 | 1.2 | 修复 hybrid 配置读取问题（支持嵌套结构 config.retrieval.hybrid.*）|
| 2025-12-13 | 1.3 | 添加 traversal_stats 统计支持，配置默认切换为 agentic_v2 |
| 2025-12-13 | 1.4 | V2 优化三项改进：(1) 时间问题增加前后周查询变体；(2) Round 2 召回量提升 30→40；(3) REASONING 类问题自动扩大召回窗口 1.5x |
