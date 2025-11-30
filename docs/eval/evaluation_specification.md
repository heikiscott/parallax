# Parallax 评测 (Eval) 全流程详解

> 基于 `locomo-q30-1` 实际运行结果的完整走读

## 目录

1. [架构概览](#1-架构概览)
2. [数据流总览](#2-数据流总览)
3. [Stage 1: MemUnit 抽取](#3-stage-1-memunit-抽取)
4. [Stage 2: 索引构建](#4-stage-2-索引构建)
5. [Stage 3: 记忆检索](#5-stage-3-记忆检索)
6. [Stage 4: 答案生成](#6-stage-4-答案生成)
7. [Stage 5: 评估](#7-stage-5-评估)
8. [完整案例：问题 "When did Caroline go to the LGBTQ support group?"](#8-完整案例)
9. [检索与生成字段设计原则](#9-检索与生成字段设计原则)
10. [Eval 与 Src 模块的差异](#10-eval-与-src-模块的差异)
11. [常见问题解答](#11-常见问题解答)

---

## 1. 架构概览

### 1.1 目录结构

```
eval/
├── cli.py                              # 命令行入口
├── run_locomo.py                       # 快捷运行脚本
├── config/
│   ├── datasets/                       # 数据集配置
│   │   ├── locomo-q30.yaml
│   │   └── ...
│   └── systems/                        # 系统配置
│       └── parallax.yaml
├── core/
│   ├── pipeline.py                     # Pipeline 编排器
│   ├── data_models.py                  # 标准数据模型
│   └── stages/                         # 各阶段执行逻辑
├── adapters/
│   └── parallax/
│       ├── parallax_adapter.py         # 适配器
│       ├── stage1_memunits_extraction.py
│       ├── stage2_index_building.py
│       ├── stage3_memory_retrivel.py
│       └── stage4_response.py
└── results/
    └── locomo-q30-1/                   # 本文档使用的实际案例
        ├── memunits/
        │   └── memunit_list_conv_0.json   # 14MB, 26个MemUnit
        ├── bm25_index/
        │   └── bm25_index_conv_0.pkl      # 4.5MB
        ├── vectors/
        │   └── embedding_index_conv_0.pkl # 6.4MB
        ├── search_results.json            # 1.8MB
        ├── answer_results.json            # 826KB
        └── eval_results.json              # 16KB
```

### 1.2 Pipeline 流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Eval Pipeline 全流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  原始数据                Stage 1                 Stage 2                    │
│  ┌──────────┐           ┌──────────┐            ┌──────────┐               │
│  │ LoComo   │  ──────▶  │ MemUnit  │  ──────▶   │  Index   │               │
│  │ JSON     │           │ 抽取     │            │  构建    │               │
│  └──────────┘           └──────────┘            └──────────┘               │
│       │                      │                       │                      │
│       ▼                      ▼                       ▼                      │
│  locomo-q30.json      memunit_list_conv_0.json   bm25_index.pkl            │
│  (对话+QA对)           (26个MemUnit)              embedding_index.pkl       │
│                                                                             │
│                                                                             │
│  Stage 3                Stage 4                 Stage 5                     │
│  ┌──────────┐           ┌──────────┐            ┌──────────┐               │
│  │  记忆    │  ──────▶  │  答案    │  ──────▶   │  评估    │               │
│  │  检索    │           │  生成    │            │          │               │
│  └──────────┘           └──────────┘            └──────────┘               │
│       │                      │                       │                      │
│       ▼                      ▼                       ▼                      │
│  search_results.json   answer_results.json      eval_results.json          │
│  (30个问题的检索结果)   (30个生成答案)           (准确率: 90%)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流总览

### 2.1 原始输入数据

**文件**: `eval/data/locomo/locomo-q30.json`

**数据结构**:
```json
[
  {
    "conversation": {
      "speaker_a": "Caroline",
      "speaker_b": "Melanie",
      "session_1": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you! How have you been?"},
        {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hey Caroline! Good to see you! I'm swamped with the kids & work..."}
      ],
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_2": [...],
      "session_2_date_time": "1:14 pm on 25 May, 2023"
    },
    "qa_pairs": [
      {
        "question_id": "locomo-q30_0_qa0",
        "question": "When did Caroline go to the LGBTQ support group?",
        "answer": "7 May 2023",
        "category": 2
      }
    ]
  }
]
```

**数据统计** (locomo-q30):
- 对话数: 1
- Session 数: 多个 (跨越 May-July 2023)
- 说话者: Caroline 和 Melanie
- QA 问题数: 30

---

## 3. Stage 1: MemUnit 抽取

### 3.1 处理流程

**代码位置**: `eval/adapters/parallax/stage1_memunits_extraction.py`

```
原始消息流 ──▶ 边界检测 ──▶ Narrative生成 ──▶ EventLog提取 ──▶ Embedding生成 ──▶ MemUnit
```

### 3.2 输出: MemUnit 结构

**文件**: `results/locomo-q30-1/memunits/memunit_list_conv_0.json` (14MB, 26个MemUnit)

**单个 MemUnit 完整字段**:

```json
{
  "unit_id": "05a303b0-2f49-4ebf-81b2-ca3e95944665",
  "user_id_list": ["caroline_locomo-q30_0", "melanie_locomo-q30_0"],
  "participants": ["caroline_locomo-q30_0", "melanie_locomo-q30_0"],
  "original_data": [
    {
      "speaker_id": "caroline_locomo-q30_0",
      "user_name": "Caroline",
      "content": "Hey Mel! Good to see you! How have you been?",
      "timestamp": "2023-05-08T13:56:00+08:00"
    }
  ],
  "timestamp": "2023-05-08T13:56:30+08:00",
  "type": "Conversation",
  "summary": "On May 8, 2023 at 1:56 PM UTC, Caroline greeted her friend Melanie...",
  "subject": "Caroline and Melanie's Catch-Up on Family and Work May 8, 2023",
  "narrative": "On May 8, 2023 at 1:56 PM UTC, Caroline greeted her friend Melanie with enthusiasm, expressing joy at seeing her again...",
  "event_log": {
    "time": "May 08, 2023(Monday) at 01:56 PM",
    "atomic_fact": [
      "Caroline greeted her friend Melanie with enthusiasm.",
      "Caroline expressed joy at seeing Melanie again.",
      "Caroline inquired about Melanie's well-being."
    ],
    "fact_embeddings": [
      [-0.0002803802490234375, -0.021240234375, ...],
      [...]
    ]
  }
}
```

### 3.3 MemUnit 字段说明与用途总结

| 字段 | 类型 | BM25检索 | Embedding检索 | 答案生成 | 说明 |
|------|------|:--------:|:-------------:|:--------:|------|
| `unit_id` | string | - | - | - | 唯一标识符 (UUID) |
| `original_data` | list[dict] | - | - | - | 原始消息数据 |
| `timestamp` | string | - | - | - | ISO格式时间戳 |
| **`summary`** | string | **回退×2** | **回退** | ❌ | 简短摘要 |
| **`subject`** | string | **回退×3** | **回退** | **✅** | 主题标题 |
| **`narrative`** | string | **回退×1** | **回退** | **✅** | 详细叙事描述 |
| **`event_log.atomic_fact`** | list[str] | **✅优先** | - | - | 原子事实列表 |
| **`event_log.fact_embeddings`** | list[list[float]] | - | **✅优先** | - | 事实向量 |

> **重要说明**:
>
> - `narrative` 字段在 Eval 中只存储为**纯字符串**，**不使用** `EpisodeMemory` 结构化类
> - Eval 流程**不需要** `EpisodeMemory` 的结构化检索和存储
> - `narrative` 本身**没有单独做 embedding**，语义检索使用的是 `atomic_fact` 的 embedding

---

## 4. Stage 2: 索引构建

### 4.1 处理流程

**代码位置**: `eval/adapters/parallax/stage2_index_building.py`

```
MemUnit列表 ──▶ BM25索引构建 ──▶ bm25_index_conv_0.pkl
            ──▶ Embedding索引 ──▶ embedding_index_conv_0.pkl
```

### 4.2 BM25 索引构建

#### 4.2.1 索引字段选择逻辑

**代码函数**: `build_searchable_text(doc)` (stage2_index_building.py:53-94)

```python
def build_searchable_text(doc: dict) -> str:
    """
    Build searchable text from a document with weighted fields.

    Priority:
    1. If event_log exists, use atomic_fact for indexing  # ← 优先使用
    2. Otherwise, fall back to original fields:           # ← 回退策略
       - "subject" corresponds to "title" (weight * 3)
       - "summary" corresponds to "summary" (weight * 2)
       - "narrative" corresponds to "content" (weight * 1)
    """
    parts = []

    # 优先使用event_log的atomic_fact（如果存在）
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            for fact in atomic_facts:
                if isinstance(fact, dict) and "fact" in fact:
                    parts.append(fact["fact"])
                elif isinstance(fact, str):
                    parts.append(fact)
            return " ".join(str(fact) for fact in parts if fact)

    # 回退到原有字段（保持向后兼容）
    if doc.get("subject"):
        parts.extend([doc["subject"]] * 3)  # ×3 权重
    if doc.get("summary"):
        parts.extend([doc["summary"]] * 2)  # ×2 权重
    if doc.get("narrative"):
        parts.append(doc["narrative"])         # ×1 权重

    return " ".join(str(part) for part in parts if part)
```

#### 4.2.2 BM25 索引字段优先级

| 优先级 | 条件 | 使用字段 | 权重 |
|--------|------|----------|------|
| **1 (优先)** | 存在 `event_log.atomic_fact` | `atomic_fact` 列表拼接 | 无权重(直接拼接) |
| 2 (回退) | 无 event_log | `subject` | ×3 (重复3次) |
| 2 (回退) | 无 event_log | `summary` | ×2 (重复2次) |
| 2 (回退) | 无 event_log | `narrative` | ×1 |

**实际行为**: 在 locomo-q30-1 中，所有 MemUnit 都有 `event_log.atomic_fact`，因此：

- **BM25 只索引了 `atomic_fact`**
- **没有使用 `subject`、`summary`、`narrative`**

### 4.3 Embedding 索引构建

#### 4.3.1 索引字段选择逻辑

**代码函数**: `build_emb_index()` (stage2_index_building.py:178-369)

```python
for doc_idx, doc in enumerate(original_docs):
    # 优先使用event_log（如果存在）
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list) and atomic_facts:
            # 🔥 关键：每个atomic_fact单独计算embedding（MaxSim策略）
            for fact_idx, fact in enumerate(atomic_facts):
                texts_to_embed.append(fact_text)
                doc_field_map.append((doc_idx, f"atomic_fact_{fact_idx}"))
            continue

    # 回退到原有字段（保持向后兼容）
    for field in ["subject", "summary", "narrative"]:
        if text := doc.get(field):
            texts_to_embed.append(text)
            doc_field_map.append((doc_idx, field))
```

#### 4.3.2 Embedding 索引字段优先级

| 优先级 | 条件 | 使用字段 | 处理方式 |
|--------|------|----------|----------|
| **1 (优先)** | 存在 `event_log.atomic_fact` | 每个 `atomic_fact` | **单独 embedding**（用于MaxSim） |
| 2 (回退) | 无 event_log | `subject` | 单独 embedding |
| 2 (回退) | 无 event_log | `summary` | 单独 embedding |
| 2 (回退) | 无 event_log | `narrative` | 单独 embedding |

**重要**:

- `narrative` 字段**没有单独做 embedding**！
- 语义检索使用的是 `event_log.fact_embeddings`（每个 atomic_fact 的 embedding）
- 只有在**回退模式**（无 event_log 时）才会对 narrative 做 embedding

---

## 5. Stage 3: 记忆检索

### 5.1 检索流程 (Agentic Retrieval)

**代码位置**: `eval/adapters/parallax/stage3_memory_retrivel.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agentic Retrieval 流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户问题: "When did Caroline go to the LGBTQ support group?"               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Round 1: 初次检索                                                    │   │
│  │                                                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐                              │   │
│  │   │ BM25 检索    │    │ Embedding    │                              │   │
│  │   │ (atomic_fact)│    │ 检索(MaxSim) │                              │   │
│  │   └──────┬───────┘    └──────┬───────┘                              │   │
│  │          │                   │                                       │   │
│  │          └───────┬───────────┘                                       │   │
│  │                  ▼                                                   │   │
│  │          ┌──────────────┐                                           │   │
│  │          │ RRF 融合     │  Reciprocal Rank Fusion                   │   │
│  │          │ score = Σ 1/(k+rank)                                     │   │
│  │          └──────┬───────┘                                           │   │
│  │                 ▼                                                    │   │
│  │          ┌──────────────┐                                           │   │
│  │          │ Reranker     │  Top 20 → Top 5                           │   │
│  │          └──────┬───────┘                                           │   │
│  │                 ▼                                                    │   │
│  │          ┌──────────────┐                                           │   │
│  │          │ LLM 充分性   │  判断检索结果是否足够回答问题              │   │
│  │          │ 检查         │                                           │   │
│  │          └──────┬───────┘                                           │   │
│  │                 │                                                    │   │
│  │                 ├── 足够 ──▶ 返回 Top 20                            │   │
│  │                 │                                                    │   │
│  │                 └── 不足 ──▶ Round 2 (多查询检索)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 检索字段总结

| 检索方式 | 使用字段 | 说明 |
|---------|---------|------|
| **BM25** | `event_log.atomic_fact` | 关键词匹配，原子事实拼接后分词 |
| **Embedding (MaxSim)** | `event_log.fact_embeddings` | 语义匹配，找最相关的 atomic_fact |
| **Reranker** | `event_log.atomic_fact` 或 `narrative` | 格式化为多行文本进行重排 |

### 5.3 MaxSim 策略说明

```python
def compute_maxsim_score(query_emb, atomic_fact_embs):
    """
    MaxSim 策略：找到与 query 最相关的单个 atomic_fact
    - 只要有一个 atomic_fact 与 query 强相关，就认为整个 MemUnit 相关
    - 避免被不相关的 fact 稀释分数
    """
    similarities = [cosine_similarity(query_emb, fact_emb) for fact_emb in atomic_fact_embs]
    return max(similarities)
```

---

## 6. Stage 4: 答案生成

### 6.1 处理流程

**代码位置**: `eval/adapters/parallax/stage4_response.py` 和 `parallax_adapter.py`

```
检索结果 ──▶ Context构建 ──▶ Prompt填充 ──▶ LLM生成 ──▶ 答案提取
```

### 6.2 Context 构建 - 使用的字段

**代码位置**: `parallax_adapter.py:548-566`

```python
# 从检索到的 MemUnit 中提取内容
for doc, score in top_results[:response_top_k]:  # 默认 top_k=10
    subject = doc.get('subject', 'N/A')      # ✅ 使用 subject
    narrative = doc.get('narrative', 'N/A')  # ✅ 使用 narrative
    doc_text = f"{subject}: {narrative}\n---"
    retrieved_docs_text.append(doc_text)
```

### 6.3 填充到答案生成的字段

| 字段 | 是否使用 | 用途 |
|------|:--------:|------|
| **`subject`** | ✅ | 作为每个记忆块的标题 |
| **`narrative`** | ✅ | 作为每个记忆块的详细内容 |
| `summary` | ❌ | 不使用 |
| `event_log.atomic_fact` | ❌ | 只用于检索，不用于生成 |
| `original_data` | ❌ | 不使用 |

### 6.4 Context 模板

```python
TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
```

**实际 Context 示例**:
```
Episodes memories for conversation between Caroline and Melanie:

    Caroline's Empowering Experience at the LGBTQ Support Group on May 7, 2023: On May 8, 2023 at 1:59 PM UTC, Caroline shared her experience of attending an LGBTQ support group the previous day (May 7, 2023)...
---

Caroline's Journey into Counseling: On June 27, 2023, Melanie and Caroline engaged in a meaningful conversation...
---
```

---

## 7. Stage 5: 评估

### 7.1 评估方式

**评估器**: LLM Judge (gpt-4o-mini)

**运行次数**: 3 次 (取平均，提高稳定性)

### 7.2 评估结果

**文件**: `results/locomo-q30-1/eval_results.json`

```json
{
  "total_questions": 30,
  "correct": 27,
  "accuracy": 0.9,
  "metadata": {
    "model": "gpt-4o-mini",
    "num_runs": 3,
    "mean_accuracy": 0.9,
    "category_accuracies": {
      "1": {"mean": 0.875, "total": 8},
      "2": {"mean": 0.928, "total": 14},
      "3": {"mean": 0.833, "total": 6}
    }
  }
}
```

---

## 8. 完整案例

### 问题: "When did Caroline go to the LGBTQ support group?"

#### Step 1: 原始对话

```
Caroline: "I went to a LGBTQ support group yesterday and it was so powerful."
                                          ↑
                                      关键信息
```

#### Step 2: 抽取的 MemUnit

```json
{
  "subject": "Caroline's Empowering Experience at the LGBTQ Support Group on May 7, 2023",
  "narrative": "On May 8, 2023 at 1:59 PM UTC, Caroline shared her experience of attending an LGBTQ support group the previous day (May 7, 2023)...",
  "event_log": {
    "atomic_fact": [
      "Caroline attended an LGBTQ support group on May 7, 2023",
      "Caroline described the support group experience as powerful and inspiring"
    ],
    "fact_embeddings": [[...], [...]]
  }
}
```

#### Step 3: 检索过程

1. **BM25 检索**: 匹配 "LGBTQ support group" 在 `atomic_fact` 中
2. **Embedding 检索**: MaxSim 找到 "Caroline attended an LGBTQ support group on May 7, 2023"
3. **RRF 融合**: 该 MemUnit 排名第一，score = 0.997

#### Step 4: 生成的 Context

```
Episodes memories for conversation between Caroline and Melanie:

    Caroline's Empowering Experience at the LGBTQ Support Group on May 7, 2023: On May 8, 2023 at 1:59 PM UTC, Caroline shared her experience...
---
```

**注意**: Context 只使用 `subject` + `narrative`，不使用 `atomic_fact`。

#### Step 5: LLM 生成答案

**生成答案**: "Caroline went to the LGBTQ support group on May 7, 2023."

**金标答案**: "7 May 2023"

**判断**: ✅ True

---

## 9. 检索与生成字段设计原则

### 9.1 顶层设计原则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           字段分工设计原则                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📌 核心思想：检索精准，生成丰富                                             │
│                                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐                       │
│  │   检索阶段          │     │   生成阶段          │                       │
│  │   (Retrieval)       │     │   (Generation)      │                       │
│  ├─────────────────────┤     ├─────────────────────┤                       │
│  │ • 目标：高召回率    │     │ • 目标：高可读性    │                       │
│  │ • 使用：atomic_fact │     │ • 使用：narrative   │                       │
│  │ • 特点：细粒度、精准│     │ • 特点：上下文完整  │                       │
│  └─────────────────────┘     └─────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 为什么这样设计？

#### 9.2.1 检索阶段使用 `atomic_fact`

| 优势 | 说明 |
|------|------|
| **细粒度匹配** | 每个 atomic_fact 是一个独立事实，可以精准匹配用户查询 |
| **MaxSim 友好** | 只要有一个 fact 匹配，整个 MemUnit 就能被召回 |
| **避免噪声** | 不会被 narrative 中的无关内容稀释相关性分数 |

#### 9.2.2 生成阶段使用 `narrative`

| 优势 | 说明 |
|------|------|
| **上下文完整** | narrative 包含完整的事件描述，不会遗漏信息 |
| **可读性强** | 自然语言叙述，LLM 易于理解 |
| **时间线清晰** | 包含时间戳和事件顺序 |

### 9.3 字段流转图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         字段在各阶段的流转                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1 (抽取)           Stage 2 (索引)          Stage 3 (检索)        │
│  ┌─────────────┐          ┌─────────────┐         ┌─────────────┐       │
│  │ 生成字段:   │          │ BM25索引:   │         │ BM25检索:   │       │
│  │ • subject   │   ──▶    │ atomic_fact │   ──▶   │ atomic_fact │       │
│  │ • narrative │          │ (优先)      │         │             │       │
│  │ • summary   │          │             │         │ Emb检索:    │       │
│  │ • atomic_   │          │ Emb索引:    │         │ fact_       │       │
│  │   fact      │          │ fact_       │         │ embeddings  │       │
│  │ • fact_     │          │ embeddings  │         │ (MaxSim)    │       │
│  │   embeddings│          │ (优先)      │         │             │       │
│  └─────────────┘          └─────────────┘         └─────────────┘       │
│                                                                          │
│  Stage 4 (答案生成)                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Context = subject + ": " + narrative                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Eval 与 Src 模块的差异

### 10.1 存储方式

| 对比项 | Eval 模块 | Src 模块 |
|-------|----------|---------|
| **MemUnit 存储** | JSON 文件 | 存储抽象层 |
| **索引存储** | Pickle 文件 | 内存/数据库 |
| **检索结果** | JSON 文件 | 实时返回 |

### 10.2 数据结构

| 对比项 | Eval 模块 | Src 模块 |
|-------|----------|---------|
| **narrative 存储** | 纯字符串 | `EpisodeMemory` 结构化类 |
| **MemUnit 类型** | JSON dict | `MemUnit` dataclass |

### 10.3 设计目的

**Eval 模块**:
- 离线评测，追求**可复现性**
- 结果持久化到文件，便于**调试和对比**
- 支持断点续传

**Src 模块**:
- 在线服务，追求**低延迟**
- 实时处理单条消息
- 集成到应用系统

---

## 11. 常见问题解答

### Q1: BM25索引的构建只需要narrative文本吗？summary和subject用到了吗？

**答案**:
- **优先使用 `event_log.atomic_fact`**（如果存在）
- **只有在没有 event_log 时**，才回退使用 subject(×3权重) + summary(×2权重) + narrative(×1权重)
- 在 locomo-q30-1 中，所有 MemUnit 都有 event_log，因此 **BM25 只索引了 atomic_fact**

### Q2: narrative在MemUnit中只存了字符串，eval流程是否需要EpisodeMemory结构化？

**答案**:
- **不需要**。Eval 流程中 `narrative` 只是纯字符串
- Eval 不使用 `EpisodeMemory` 结构化类
- 结构化存储是 Src 模块的设计，Eval 为了简化采用纯 JSON

#### 详细说明：Eval 流程中不创建独立的 EpisodeMemory 对象

**代码位置**: `eval/adapters/parallax/stage1_memunits_extraction.py:260-264`

```python
# Eval 调用方式
episode_result = await episode_extractor.extract_memory(
    episode_request, use_group_prompt=True  # ← 关键参数
)
memunit.narrative = episode_result.narrative     # ← 只取 narrative 字符串
memunit.subject = episode_result.subject     # ← 只取 subject 字符串
```

**`use_group_prompt=True` 时的返回类型**:

- 返回 `MemUnit` 对象（不是 `List[EpisodeMemory]`）
- `episode_result.narrative` 是**纯字符串**
- 直接赋值给 `memunit.narrative`

**对比：`use_group_prompt=False` 时（生产环境）**:

```python
# src/memory/extraction/memory/episode_memory_extractor.py:429-432
async def generate_memory_for_user(user_id: str, user_name: str) -> EpisodeMemory:
    # 为每个参与者创建独立的 EpisodeMemory 对象
    return EpisodeMemory(
        user_id=user_id,
        episode_id=generate_uuid(),
        narrative=content,
        subject=title,
        ...
    )
```

**设计原因**:

| 场景 | 模式 | 返回类型 | 说明 |
|------|------|----------|------|
| **Eval (评测)** | `use_group_prompt=True` | `MemUnit` (含 narrative 字符串) | 群组视角，单一叙事，简化存储 |
| **Production (生产)** | `use_group_prompt=False` | `List[EpisodeMemory]` | 多用户视角，为每人生成独立记忆 |

**结论**:

- `EpisodeMemory` 类在 Eval 流程中**完全没有用到**
- Eval 只使用 `EpisodeMemoryExtractor` 来**生成 narrative 字符串**
- 生成的字符串**直接存储在 `MemUnit.narrative` 字段中**
- 这是因为 Eval 只需要群组视角的单一叙事，不需要为每个参与者生成独立的个人视角记忆

### Q3: narrative除了用于BM25检索之外也做了embedding吗？

**答案**:
- **没有**。`narrative` 本身没有单独做 embedding
- 语义检索使用的是 `event_log.fact_embeddings`（每个 atomic_fact 的 embedding）
- 只有在**回退模式**（无 event_log 时）才会对 narrative 做 embedding

### Q4: summary和subject这些字段在哪里用到了？

**答案**:
| 字段 | 检索时用途 | 生成时用途 |
|------|-----------|-----------|
| **`subject`** | 回退策略 (×3权重) | ✅ 作为 Context 中每个记忆块的标题 |
| **`summary`** | 回退策略 (×2权重) | ❌ **当前未使用** |

### Q5: 填写到答案中的只有narrative文本吗？除了narrative还有哪些？

**答案**:
- **`subject`**: 作为标题
- **`narrative`**: 作为详细内容
- 格式: `"{subject}: {narrative}\n---"`

---

## 附录: 文件大小统计 (locomo-q30-1)

| 文件 | 大小 | 说明 |
|------|------|------|
| memunit_list_conv_0.json | 14 MB | 26个MemUnit，包含embedding |
| bm25_index_conv_0.pkl | 4.5 MB | BM25索引 |
| embedding_index_conv_0.pkl | 6.4 MB | 向量索引 |
| search_results.json | 1.8 MB | 30个问题的检索结果 |
| answer_results.json | 826 KB | 30个生成答案 |
| eval_results.json | 16 KB | 评估结果 |
