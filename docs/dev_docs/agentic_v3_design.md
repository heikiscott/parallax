# Agentic Retrieval V3 设计文档

> **版本**: 1.0
> **日期**: 2024-12
> **作者**: Claude
> **状态**: 设计审阅中

---

## 目录

1. [概述](#1-概述)
2. [设计目标](#2-设计目标)
3. [ColBERT 技术选型](#3-colbert-技术选型)
4. [架构设计](#4-架构设计)
5. [核心组件](#5-核心组件)
6. [配置规范](#6-配置规范)
7. [索引管理](#7-索引管理)
8. [与 V2 对比](#8-与-v2-对比)
9. [实现计划](#9-实现计划)
10. [运行环境](#10-运行环境)

---

## 1. 概述

### 1.1 背景

Agentic V2 使用混合检索策略（Embedding + BM25 + RRF），虽然效果不错，但存在以下问题：

1. **多阶段复杂性**：需要协调 embedding 检索、BM25 检索、RRF 融合、Rerank 等多个环节
2. **参数繁多**：每个环节都有独立参数（emb_candidates, bm25_candidates, rrf_k, rerank_top_n 等）
3. **依赖外部服务**：Reranker 依赖 DeepInfra API

### 1.2 V3 方案

Agentic V3 采用 **纯 ColBERT** 检索，利用 ColBERT 的 Late Interaction 机制实现高精度检索，同时简化流程。

### 1.3 核心变化

| 组件 | V2 (当前) | V3 (新) |
|------|-----------|---------|
| 检索方法 | Embedding + BM25 + RRF | **纯 ColBERT MaxSim** |
| Rerank | DeepInfra BGE Reranker | **无需**（ColBERT 自带排序） |
| Cluster Expansion | 有 | **移除**（ColBERT token 匹配已足够） |
| 问题分类 | 保留 | 保留 |
| Type-Aware Multi-Query | 保留 | 保留 |
| Sufficiency Check | 保留 | 保留 |
| Round 2 | 保留 | 保留 |

---

## 2. 设计目标

### 2.1 主要目标

1. **提升检索精度**：利用 ColBERT 的 token-level 交互提升语义匹配质量
2. **简化流程**：移除 Reranker 和 Cluster Expansion，减少依赖
3. **降低参数复杂度**：从 10+ 参数简化为 4 个核心参数
4. **跨平台兼容**：支持 Windows/Linux/Mac，无需 WSL

### 2.2 非目标

- 不追求极致延迟（CPU 环境下接受较高延迟）
- 不考虑在线实时场景（专注离线评测）

---

## 3. ColBERT 技术选型

### 3.1 候选方案对比

| 包 | Windows 支持 | 安装方式 | 优点 | 缺点 |
|----|-------------|----------|------|------|
| **RAGatouille** | ❌ (仅 WSL2) | `pip install ragatouille` | 易用、封装好 | 不支持 Windows |
| **colbert-ai** | ⚠️ (部分) | `pip install colbert-ai` | Stanford 官方 | 依赖复杂，FAISS 问题 |
| **Jina ColBERT v2** | ✅ | `transformers + torch` | HF 原生、多语言 | 需自行实现索引 |

### 3.2 选择：Jina ColBERT v2 (HuggingFace)

**理由**：

1. **Windows 兼容**：直接运行，无需 WSL
2. **已有依赖**：`transformers` 和 `torch` 项目中已存在
3. **多语言支持**：支持 89+ 语言
4. **灵活控制**：可自定义索引和检索逻辑
5. **模型质量**：基于 ColBERT v2 架构，在多个 benchmark 表现优异

**模型信息**：

- Model ID: `jinaai/jina-colbert-v2`
- Embedding 维度: 128 per token
- 最大序列长度: 8192 tokens
- 参考：[HuggingFace Model Card](https://huggingface.co/jinaai/jina-colbert-v2)

### 3.3 ColBERT 原理简述

```
传统 Dense Retrieval:
  Query  → [CLS] embedding (1 × 768)
  Doc    → [CLS] embedding (1 × 768)
  Score  = dot(query_emb, doc_emb)

ColBERT Late Interaction:
  Query  → token embeddings (q_len × 128)
  Doc    → token embeddings (d_len × 128)
  Score  = Σ max(dot(q_i, d_j)) for each q_i  (MaxSim)
```

**优势**：每个 query token 都能找到最相关的 doc token，捕捉细粒度语义匹配。

---

## 4. 架构设计

### 4.1 V3 流程图

```
Query
  │
  ▼
Question Classification (复用 V2 rule-based)
  │
  ├─── 简单问题 (ATTRIBUTE_*) + 高置信度
  │         │
  │         ▼
  │    单 Query ColBERT Search → Top 10 → Sufficiency Check
  │
  └─── 复杂问题 (EVENT_*, COUNTING, REASONING_*)
            │
            ▼
       Type-Aware Multi-Query (2-3 queries)
            │
            ▼
       并行 ColBERT Search → RRF Fusion → Top 12-18
            │
            ▼
       Sufficiency Check (LLM)
            │
            ├─── Sufficient → 返回结果
            │
            └─── Insufficient → Round 2
                      │
                      ▼
                 Missing-Info Multi-Query (复用 V2)
                      │
                      ▼
                 ColBERT Search → Merge → 返回最终结果
```

### 4.2 关键设计决策

#### 4.2.1 无单独 Reranker

**原因**：ColBERT 的 MaxSim 已提供高质量排序，token-level 交互本身就是一种 "软" cross-attention。添加额外 reranker 是冗余的。

#### 4.2.2 移除 Cluster Expansion

**原因**：
- ColBERT 的 token-level 匹配已能捕捉同义词和语义相似
- 减少流程复杂度和延迟
- 评测后如发现召回不足，可重新考虑

#### 4.2.3 保留 Multi-Query

**原因**：
- 不同查询角度能覆盖不同信息需求
- RRF 融合多查询结果仍有价值
- Type-aware 策略在 V2 中证明有效

#### 4.2.4 更保守的参数

**原因**：ColBERT 精度更高，不需要像 V2 那样大量召回再筛选。

---

## 5. 核心组件

### 5.1 文件结构

```
src/
  retrieval/
    offline/
      pipelines/
        agentic_v3.py              # V3 主流程
        colbert_utils.py           # ColBERT 检索工具函数
      retrievers/
        colbert_retriever.py       # ColBERT 检索器类 (可选)
    services/
      colbert_service.py           # ColBERT 模型服务

scripts/
  build_colbert_index.py           # ColBERT 索引构建脚本

config/
  src/
    colbert.yaml                   # ColBERT 模型配置
```

### 5.2 ColBERT Service

**文件**: `src/retrieval/services/colbert_service.py`

```python
class ColBERTService:
    """ColBERT v2 模型服务，提供编码和评分功能。"""

    def __init__(self, model_name: str = "jinaai/jina-colbert-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False

    async def initialize(self):
        """延迟初始化模型。"""
        if self._initialized:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.eval()
        self._initialized = True

    async def encode_query(self, query: str) -> np.ndarray:
        """编码 query 为多向量表示。

        Returns:
            np.ndarray: shape [seq_len, 128]
        """
        await self.initialize()
        inputs = self.tokenizer(
            query, return_tensors="pt",
            max_length=128, truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, 128]
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.numpy()

    async def encode_documents(self, docs: List[str]) -> List[np.ndarray]:
        """批量编码文档为多向量表示。"""
        await self.initialize()
        all_embeddings = []
        for doc in docs:
            inputs = self.tokenizer(
                doc, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[0]
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeddings.append(emb.numpy())
        return all_embeddings

    def compute_maxsim(
        self, query_emb: np.ndarray, doc_emb: np.ndarray
    ) -> float:
        """计算 MaxSim 分数。

        MaxSim: 对每个 query token，找到最相似的 doc token，然后求和。

        Args:
            query_emb: [q_len, 128]
            doc_emb: [d_len, 128]

        Returns:
            float: MaxSim 分数
        """
        # 相似度矩阵: [q_len, d_len]
        sim_matrix = np.dot(query_emb, doc_emb.T)
        # 每个 query token 的最大相似度
        max_sim_per_token = np.max(sim_matrix, axis=1)
        # 求和
        return float(np.sum(max_sim_per_token))
```

### 5.3 ColBERT 检索工具

**文件**: `src/retrieval/offline/pipelines/colbert_utils.py`

```python
async def colbert_search(
    query: str,
    colbert_index: List[dict],
    top_n: int = 20,
    return_traversal_stats: bool = False
) -> List[Tuple[dict, float]]:
    """执行 ColBERT 检索。

    Args:
        query: 用户查询
        colbert_index: 预构建的 ColBERT 索引
            [{"doc": {...}, "embeddings": np.ndarray}, ...]
        top_n: 返回结果数

    Returns:
        [(doc, score), ...] 按分数降序排列
    """
    colbert_service = get_colbert_service()

    # 编码 query
    query_emb = await colbert_service.encode_query(query)

    # 对所有文档评分
    doc_scores = []
    for item in colbert_index:
        doc = item["doc"]
        doc_emb = item["embeddings"]
        score = colbert_service.compute_maxsim(query_emb, doc_emb)
        doc_scores.append((doc, score))

    # 按分数排序
    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def multi_colbert_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """RRF 融合多个 ColBERT 检索结果。"""
    if len(results_list) == 1:
        return results_list[0]

    doc_rrf_scores = {}
    doc_map = {}

    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = doc.get("unit_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]
```

### 5.4 V3 主流程

**文件**: `src/retrieval/offline/pipelines/agentic_v3.py`

```python
async def agentic_retrieval_v3(
    query: str,
    config: Any,
    llm_provider: Any,
    llm_config: dict,
    colbert_index: List[dict],
    enable_traversal_stats: bool = False,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Agentic Retrieval V3 - 纯 ColBERT 检索。

    与 V2 的主要区别：
    - 使用 ColBERT 替代 Embedding + BM25 + RRF
    - 无单独 Reranker（ColBERT 自带高质量排序）
    - 无 Cluster Expansion
    - 参数更保守（ColBERT 精度更高）

    Args:
        query: 用户查询
        config: 实验配置
        llm_provider: LLM Provider
        llm_config: LLM 配置
        colbert_index: 预构建的 ColBERT 索引
        enable_traversal_stats: 是否记录遍历统计

    Returns:
        (final_results, metadata)
    """
    # Step 1: 问题分类 (复用 V2)
    classifier = QuestionClassifier()
    classification = classifier.classify(query)

    # Step 2: 加载类型配置
    type_config = _get_type_retrieval_config_v3(config, classification.question_type)
    round1_top_n = type_config.get('round1_top_n', 12)
    round2_top_n = type_config.get('round2_top_n', 15)
    merge_budget = type_config.get('merge_budget', 20)
    final_top_n = type_config.get('final_top_n', 12)

    # Step 3: 决定是否使用 Multi-Query
    use_mq = should_use_multi_query(
        classification.question_type,
        classification.confidence
    )

    # Step 4: Round 1 ColBERT 检索
    if use_mq:
        queries, reasoning = await generate_type_aware_multi_queries(
            query, classification.question_type, llm_provider, llm_config
        )
        # 并行检索
        results_list = await asyncio.gather(*[
            colbert_search(q, colbert_index, round1_top_n)
            for q in queries
        ])
        round1_results = multi_colbert_rrf_fusion(results_list)[:round1_top_n]
    else:
        round1_results = await colbert_search(query, colbert_index, round1_top_n)

    # Step 5: Sufficiency Check
    is_sufficient, reasoning, missing_info = await check_sufficiency(
        query, round1_results[:10], llm_provider, llm_config
    )

    if is_sufficient:
        return round1_results[:final_top_n], metadata

    # Step 6: Round 2 (Insufficient)
    refined_queries, _ = await generate_multi_queries(
        query, round1_results[:10], missing_info, llm_provider, llm_config
    )
    round2_results_list = await asyncio.gather(*[
        colbert_search(q, colbert_index, round2_top_n)
        for q in refined_queries
    ])
    round2_results = multi_colbert_rrf_fusion(round2_results_list)

    # Step 7: Merge
    round1_ids = {doc.get("unit_id") for doc, _ in round1_results}
    round2_unique = [(d, s) for d, s in round2_results if d.get("unit_id") not in round1_ids]
    combined = round1_results + round2_unique[:max(0, merge_budget - len(round1_results))]

    return combined[:final_top_n], metadata
```

---

## 6. 配置规范

### 6.1 ColBERT 模型配置

**文件**: `config/services/colbert.yaml`

```yaml
colbert:
  # 模型选择
  model_name: "jinaai/jina-colbert-v2"

  # 设备配置
  device: "cpu"  # cuda | cpu

  # Tokenization 限制
  max_query_length: 128
  max_doc_length: 512

  # 批处理
  batch_size: 8  # CPU 模式下降低批大小

  # 归一化
  normalize_embeddings: true
```

### 6.2 V3 检索配置

**添加到**: `config/eval/systems/parallax.yaml`

```yaml
retrieval:
  # ===== Agentic V3 配置 =====
  agentic_v3:
    # Multi-Query 配置 (与 V2 相同)
    skip_high_confidence_types: true
    confidence_threshold: 0.85
    num_queries: 3

    # ColBERT 检索默认配置 (比 V2 更保守)
    round1_top_n: 12
    round2_top_n: 15
    merge_budget: 20
    final_top_n: 12
    rrf_k: 60

  # ===== V3 类型配置矩阵 =====
  v3_type_retrieval_configs:
    # --- 简单问题：ColBERT 精准，10 个足够 ---
    attribute_identity:
      round1_top_n: 10
      round2_top_n: 12
      merge_budget: 15
      final_top_n: 10

    attribute_preference:
      round1_top_n: 10
      round2_top_n: 12
      merge_budget: 15
      final_top_n: 10

    attribute_location:
      round1_top_n: 10
      round2_top_n: 12
      merge_budget: 15
      final_top_n: 10

    # --- 中等问题 ---
    event_activity:
      round1_top_n: 12
      round2_top_n: 15
      merge_budget: 20
      final_top_n: 12

    general:
      round1_top_n: 12
      round2_top_n: 15
      merge_budget: 20
      final_top_n: 12

    # --- 时间/聚合问题 ---
    event_temporal:
      round1_top_n: 15
      round2_top_n: 18
      merge_budget: 25
      final_top_n: 15

    event_aggregation:
      round1_top_n: 15
      round2_top_n: 18
      merge_budget: 25
      final_top_n: 15

    time_calculation:
      round1_top_n: 15
      round2_top_n: 18
      merge_budget: 25
      final_top_n: 15

    # --- 复杂问题 ---
    counting:
      round1_top_n: 18
      round2_top_n: 22
      merge_budget: 30
      final_top_n: 18

    reasoning_hypothetical:
      round1_top_n: 18
      round2_top_n: 22
      merge_budget: 30
      final_top_n: 18

    reasoning_inference:
      round1_top_n: 18
      round2_top_n: 22
      merge_budget: 30
      final_top_n: 18

    # --- 默认配置 ---
    default:
      round1_top_n: 12
      round2_top_n: 15
      merge_budget: 20
      final_top_n: 12
```

### 6.3 参数设计原则

**ColBERT 精度更高，参数整体比 V2 更保守**：

| 问题类型 | V2 Round1 | V3 Round1 | V2 Final | V3 Final | 降幅 |
|---------|----------|----------|----------|----------|------|
| 简单 (ATTRIBUTE_*) | 20 | **10** | 20 | **10** | 50% |
| 中等 (EVENT_ACTIVITY) | 20 | **12** | 20 | **12** | 40% |
| 时间/聚合 | 25 | **15** | 20 | **15** | 40% |
| 复杂 (COUNTING) | 30 | **18** | 25 | **18** | 40% |

---

## 7. 索引管理

### 7.1 索引格式

```python
# ColBERT 索引结构 (pickle 格式)
colbert_index = [
    {
        "doc": {
            "unit_id": "conv0_mu001",
            "narrative": "...",
            "summary": "...",
            "subject": "...",
            "event_log": {...}
        },
        "embeddings": np.ndarray  # shape: [seq_len, 128]
    },
    ...
]
```

### 7.2 索引构建脚本

**文件**: `scripts/build_colbert_index.py`

```python
async def build_colbert_index(data_dir: Path, save_dir: Path):
    """构建 ColBERT 索引。"""
    colbert_service = get_colbert_service()
    await colbert_service.initialize()

    memunit_files = sorted(glob.glob(str(data_dir / "memunit_list_conv_*.json")))

    for file_path in memunit_files:
        conv_index = Path(file_path).stem.split('_')[-1]

        with open(file_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        # 构建可搜索文本
        texts = [build_searchable_text(doc) for doc in docs]

        # 编码
        embeddings = await colbert_service.encode_documents(texts)

        # 构建索引
        colbert_index = [
            {"doc": doc, "embeddings": emb}
            for doc, emb in zip(docs, embeddings)
        ]

        # 保存
        output_path = save_dir / f"colbert_index_conv_{conv_index}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(colbert_index, f)
```

### 7.3 存储位置

```
eval/adapters/parallax/{experiment}/colbert_index/
  colbert_index_conv_0.pkl    # ~10-15 MB
  colbert_index_conv_1.pkl
  ...
```

### 7.4 存储估算

| 数据集 | MemUnits | 估算大小 |
|--------|----------|----------|
| 单个 conversation | ~46 | ~10-15 MB |
| LoCoMo10 (10 convs) | ~460 | ~100-150 MB |
| LoCoMo-all | ~460 | ~100-150 MB |

---

## 8. 与 V2 对比

### 8.1 流程对比

```
V2 流程:
Query → Classification → Multi-Query → Hybrid Search (Emb+BM25+RRF)
     → Rerank → Sufficiency → Round2 → Cluster Expansion → Final

V3 流程:
Query → Classification → Multi-Query → ColBERT Search
     → Sufficiency → Round2 → Final
```

### 8.2 参数对比

| 参数类型 | V2 | V3 |
|---------|----|----|
| 检索参数 | round1_per_query_top_n, round1_fusion_top_n, round1_rerank_top_n | **round1_top_n** |
| Hybrid 参数 | emb_candidates, bm25_candidates, rrf_k | 无 |
| Rerank 参数 | reranker model, batch_size, concurrency | 无 |
| Cluster 参数 | expansion_strategy, max_expansion | 无 |
| 参数总数 | 10+ | **4** |

### 8.3 依赖对比

| 依赖 | V2 | V3 |
|------|----|----|
| Embedding Service | ✅ | ❌ |
| BM25 Index | ✅ | ❌ |
| DeepInfra Reranker | ✅ | ❌ |
| Cluster Index | ✅ | ❌ |
| ColBERT Model | ❌ | ✅ |
| ColBERT Index | ❌ | ✅ |

### 8.4 预期优势

1. **精度提升**：ColBERT token-level 匹配优于单向量 + BM25
2. **流程简化**：移除 Rerank 和 Cluster Expansion
3. **参数精简**：从 10+ 参数降至 4 个
4. **无外部依赖**：不依赖 DeepInfra API

### 8.5 预期挑战

1. **索引大小**：ColBERT 索引比单向量大 ~20-50 倍
2. **CPU 延迟**：无 GPU 时检索较慢
3. **模型加载**：首次加载模型需 ~10-30 秒

---

## 9. 实现计划

### Phase 1: ColBERT 基础设施

1. 创建 `config/services/colbert.yaml`
2. 实现 `src/retrieval/services/colbert_service.py`
3. 单元测试：模型加载、编码、MaxSim 计算

### Phase 2: 检索工具

1. 实现 `src/retrieval/offline/pipelines/colbert_utils.py`
   - `colbert_search()` 函数
   - `multi_colbert_rrf_fusion()` 函数
2. 单元测试

### Phase 3: V3 主流程

1. 实现 `src/retrieval/offline/pipelines/agentic_v3.py`
   - 复用 V2 的问题分类、multi-query、sufficiency check
   - 替换检索为 ColBERT
2. 更新 `__init__.py` 导出
3. 集成测试

### Phase 4: 索引构建

1. 实现 `scripts/build_colbert_index.py`
2. 测试单个 conversation 索引构建
3. 验证索引大小

### Phase 5: 配置与集成

1. 更新 `config/eval/systems/parallax.yaml`
2. 更新 `eval/adapters/parallax/memory_retrieval.py`
3. 端到端测试

---

## 10. 运行环境

### 10.1 CPU 模式配置

```yaml
colbert:
  device: "cpu"
  batch_size: 8  # 降低批大小
```

### 10.2 性能预估 (CPU)

| 操作 | 预估时间 |
|------|----------|
| 模型加载 | ~10-30 秒（首次） |
| 索引构建 | ~1-2 秒/文档 |
| Query 编码 | ~100-200 ms |
| MaxSim 计算 (46 docs) | ~10-50 ms |
| 单次检索总延迟 | ~200-500 ms |

### 10.3 验证计划

1. 先跑单个 conversation (46 memunits) 验证可行性
2. 确认索引大小
3. 确认检索延迟
4. 与 V2 对比效果

---

## 附录

### A. 参考文档

- [Agentic V2 设计文档](agentic_v2_design.md)
- [Agentic 检索设计文档](agentic_retrieval_design.md)
- [Jina ColBERT v2 Model Card](https://huggingface.co/jinaai/jina-colbert-v2)
- [ColBERT 论文](https://arxiv.org/abs/2004.12832)

### B. 相关文件

- `src/retrieval/offline/pipelines/agentic_v2.py` - V2 实现
- `src/retrieval/offline/pipelines/search_utils.py` - 检索工具
- `config/eval/systems/parallax.yaml` - 配置文件
