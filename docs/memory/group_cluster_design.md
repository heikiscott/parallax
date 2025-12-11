# 记忆聚类系统设计文档 (Memory Clustering System Design)

> **版本**: v2.0
> **更新日期**: 2025-12-11
> **状态**: 包含 Group Event Cluster 实现分析和新聚类方案设计

---

## 📋 目录

1. [概述](#1-概述)
2. [现有实现：Group Event Cluster](#2-现有实现group-event-cluster)
3. [Group Event Cluster 效果分析](#3-group-event-cluster-效果分析)
4. [新聚类方案设计](#4-新聚类方案设计)
5. [检索流程设计](#5-检索流程设计)
6. [实施路线图](#6-实施路线图)

---

## 1. 概述

### 1.1 背景与动机

在 LoCoMo 评估中发现，记忆检索系统存在以下问题：

| 问题 | 说明 | 影响 |
|------|------|------|
| **信息分散** | 同一主题的信息分散在多个 MemUnit | 检索只命中部分，无法完整回答 |
| **关联缺失** | 相关 MemUnit 之间缺乏显式关联 | 无法进行多跳推理 |
| **上下文不足** | 单个 MemUnit 缺乏推理所需上下文 | 时序推理准确率低 |
| **覆盖率低** | 复杂问题难以通过单一检索解决 | 只有 7% 问题使用聚类增强 |

### 1.2 设计目标

| 目标 | 现状 | 目标 |
|-----|------|------|
| **覆盖率** | 6.8% (Event Cluster) | **70-80%** |
| **准确率提升** | +5.4% (Event Cluster) | **+10-15%** |
| **选择准确性** | 98% miss 是 LLM selection 问题 | **降低到 <30%** |
| **多层聚类** | 仅事件聚类 | **事件 + 语义 + 关系** |

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **多层聚类** | 不同粒度的聚类满足不同查询需求 |
| **高覆盖率** | 目标覆盖 70-80% 的问题 |
| **易于选择** | 减少 LLM selection 错误率 |
| **渐进式** | 保留 Event Cluster，逐步添加新聚类 |
| **可解释** | 每个聚类有明确的语义和用途 |
| **离线处理** | 聚类在索引构建阶段完成 |

---

## 2. 现有实现：Group Event Cluster

### 2.1 核心概念

**Group Event Cluster** 是基于 LLM 的事件聚类系统，将讨论同一事件/主题的 MemUnit 归类到一起。

#### 2.1.1 数据结构

```python
@dataclass
class GroupEventCluster:
    cluster_id: str              # "gec_001", "gec_002", ...
    topic: str                   # "Caroline's adoption plan"
    summary: str                 # 第三人称详细描述
    members: List[ClusterMember] # 按时间排序的成员列表
    first_timestamp: datetime
    last_timestamp: datetime
    created_at: datetime
    updated_at: datetime

@dataclass
class GroupEventClusterIndex:
    clusters: Dict[str, GroupEventCluster]
    unit_to_clusters: Dict[str, List[str]]  # MemUnit → [cluster_ids]
    conversation_id: str
    total_units: int
    llm_model: str
```

#### 2.1.2 聚类算法

```
Input: MemUnit 列表（按时间排序）
Output: GroupEventClusterIndex

For each MemUnit:
  1. 生成 MemUnit 摘要（1-2句话）
  2. 如果是第一个 → 创建新 cluster
  3. 否则：
     - 将现有 clusters（最多20个）和新 MemUnit 提交给 LLM
     - LLM 判断：归入现有 cluster 或创建新 cluster
     - 支持多分配：一个 MemUnit 可属于多个 clusters
  4. 每 N 个成员更新 cluster summary
```

#### 2.1.3 检索增强策略

目前支持 5 种策略：

| 策略 | 说明 | 使用场景 |
|-----|------|---------|
| `insert_after_hit` | 在命中文档后插入 cluster 成员 | 保持语义连贯性 |
| `append_to_end` | 在结果末尾追加 cluster 成员 | 保留原始排序 |
| `merge_by_score` | 按分数重新排序 | 高质量扩展文档可排前 |
| `replace_rerank` | 扩展后外部 rerank | 需要额外 rerank 模型 |
| **`cluster_rerank`** | **LLM 选择相关 clusters** | **当前 eval 使用** |

### 2.2 Cluster Rerank 策略详解

这是当前 eval 中使用的策略，流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 原始检索                                                 │
│  - 向量检索 MemUnits                                             │
│  - 返回 top-k 结果（如 top-20）                                  │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 提取候选 Clusters                                        │
│  - 从检索结果中提取所有相关的 clusters                            │
│  - 去重，得到候选 cluster 列表（如 20-40 个）                     │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: LLM Selection                                            │
│  - 输入：query + 候选 clusters (topic, summary, hit_count)       │
│  - LLM 选择最相关的 clusters（最多 N 个，如 3 个）               │
│  - 输出：selected_cluster_ids + reasoning                        │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 返回 Cluster Members                                     │
│  - 从选中的 clusters 中提取所有 members（按时间排序）            │
│  - 应用限制：per-cluster limit, total limit                      │
│  - 去重：如果 MemUnit 在多个 cluster 中，只保留一份              │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Hybrid Supplement（可选）                                │
│  - 补充原始检索结果中未在 cluster 中的 MemUnits                  │
│  - 确保不遗漏高分的直接匹配                                      │
└─────────────────────────────────────────────────────────────────┘
              ↓
        Final Results
```

#### 配置参数

```python
@dataclass
class GroupEventClusterRetrievalConfig:
    # 基础开关
    enable_group_event_cluster_retrieval: bool = True

    # 策略选择
    expansion_strategy: str = "cluster_rerank"

    # Cluster Rerank 参数
    cluster_rerank_max_clusters: int = 3          # LLM 最多选择的 clusters
    cluster_rerank_max_members_per_cluster: int = 10  # 每个 cluster 最多返回的 members
    cluster_rerank_total_max_members: int = 20    # 总共最多返回的 members

    # Hybrid 补充参数
    hybrid_enable_original_supplement: bool = True
    hybrid_original_supplement_count: int = 10
    hybrid_max_total_results: int = 30

    # 时间偏好
    prefer_time_adjacent: bool = True
    time_window_hours: Optional[int] = None

    # 分数衰减
    expansion_score_decay: float = 0.7
```

### 2.3 代码结构

```
src/memory/group_event_cluster/
├── __init__.py                  # 模块导出
├── schema.py                    # ClusterMember, GroupEventCluster, GroupEventClusterIndex
├── types.py                     # GroupEventClusterConfig, GroupEventClusterRetrievalConfig
├── clusterer.py                 # GroupEventClusterer (聚类算法)
├── retrieval.py                 # expand_with_cluster (检索增强)
├── storage.py                   # ClusterStorage, JsonClusterStorage
└── utils.py                     # Prompt 模板, 解析函数

eval/adapters/parallax/
└── stage1_5_group_event_cluster.py  # Eval 调用入口
```

---

## 3. Group Event Cluster 效果分析

### 3.1 评估数据

在 LoCoMo benchmark 上的表现：

| 指标 | 数值 |
|-----|------|
| 总问题数 | 1,540 |
| **Event Cluster 覆盖率** | **6.8%** (104/1540) |
| 复杂问题数（被路由到 cluster_rerank） | 111 |
| Event Cluster 使用率（复杂问题中） | 93.7% (104/111) |

#### 准确率对比

| 场景 | 准确率 |
|-----|--------|
| 简单问题（未使用 cluster） | 92.6% |
| 复杂问题（使用 cluster） | 87.5% |
| 复杂问题（未使用 cluster） | 57.1% (仅 7 个样本) |
| **Cluster HIT**（找对了 cluster） | **88.1%** (59/111) |
| **Cluster MISS**（找错或没找到） | **82.7%** (52/111) |
| **准确率提升** | **+5.4%** |

### 3.2 核心问题分析

#### 问题 1：覆盖率太低（6.8%）

```
1,540 个问题中：
  ✓ 104 个 (6.8%) 使用了 Event Cluster
  ✗ 1,436 个 (93.2%) 没有使用

为什么覆盖率低？
  1. 只有被路由到 cluster_rerank 的复杂问题才会使用
  2. Event 聚类粒度太细，很多问题无法匹配到事件
  3. 缺乏其他类型的聚类（语义状态、实体关系等）
```

#### 问题 2：LLM Selection 错误率高（98.1%）

在 52 个 Cluster MISS 案例中：

| 原因 | 数量 | 占比 | 问题本质 |
|-----|------|------|---------|
| **LLM 选错了** | 44 | **84.6%** | 正确 cluster **在候选列表中**，但 LLM 选了错的 |
| **LLM 全拒绝** | 7 | **13.5%** | LLM 太保守，认为所有候选都不相关 |
| **检索漏掉** | 1 | **1.9%** | 正确 cluster 没进入候选列表 |

**关键发现**：98.1% 的失败是 **LLM Selection 的问题**！

**为什么 LLM 会选错？**

```
1. Cluster Summaries 太相似
   例子："Caroline's LGBTQ support group" vs "Caroline's LGBTQ conference"
   LLM 难以区分细粒度差异

2. 从 20+ 候选中选择太难
   候选 clusters 平均 20-40 个
   LLM 需要在众多相似的 clusters 中做选择

3. Selection Prompt 不够好
   可能没有给出足够清晰的选择标准

4. 事件聚类本身的问题
   事件太细粒度，导致很多相似事件被分开
   例如："Melanie's pottery class" vs "Melanie's painting hobby"
```

#### 问题 3：双重错误累积

```
检索流程有两个错误源：
  第一步：检索 clusters → 可能漏掉正确的
  第二步：LLM selection → 高概率选错（98.1%）

任何一步出错 = 整个流程失败

vs 直接 MemUnit 检索：
  只有一步：检索 MemUnits
  没有 selection 的额外错误
  准确率：92.5%（比 cluster 的 87.5% 更高）
```

### 3.3 为什么直接 MemUnit 检索反而更好？

| 维度 | Event Cluster | 直接 MemUnit 检索 |
|-----|--------------|------------------|
| **流程** | 两步（检索 + LLM selection） | 一步（检索） |
| **错误源** | 双重错误累积 | 单一错误源 |
| **信息保留** | Cluster summary 可能丢细节 | 完整原始对话 |
| **选择难度** | 从 20+ 相似 clusters 选择 | 直接 embedding 匹配 |
| **准确率** | 87.5%（仅 6.8% 覆盖） | **92.5%**（全覆盖） |

### 3.4 Event Cluster 的价值在哪里？

虽然存在问题，但 Event Cluster 在找对时确实有帮助：

```
整体价值 = 5.4%（找对时提升） × 53.2%（找对概率） ≈ 2.9%

对于被路由到 cluster_rerank 的复杂问题：
  - Cluster HIT: 88.1% 准确率
  - Cluster MISS: 82.7% 准确率
  - 提升: +5.4%

说明：
  ✓ Event Cluster 的设计理念是对的（提供上下文有帮助）
  ✗ 但实现有问题（LLM selection 太容易出错）
  ✗ 覆盖率太低（只有 6.8%）
```

### 3.5 改进方向总结

| 问题 | 现状 | 改进方向 |
|-----|------|---------|
| **覆盖率低** | 6.8% | 引入新聚类类型（语义、关系）→ 70-80% |
| **LLM Selection 错误** | 98.1% miss | 预定义类别，减少 LLM 选择 |
| **粒度太细** | 事件级别 | 分层聚类：粗粒度 + 细粒度 |
| **双重错误** | 检索 + selection | 简化流程，减少错误源 |

---

## 4. 新聚类方案设计

### 4.1 三层聚类架构

基于分析，我们设计一个**多层聚类系统**，不同层次满足不同查询需求：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Layer Clustering System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Semantic State Clustering（语义状态聚类）             │
│  ├─ 预定义类别（10-15 个）                                       │
│  ├─ 粗粒度、高覆盖率（50-60%）                                   │
│  ├─ 易于选择（直接路由，无需 LLM）                               │
│  └─ 例子：career, hobbies, relationships, health, finance       │
│                                                                 │
│  Layer 2: Entity Relation Clustering（实体关系聚类）            │
│  ├─ 基于实体和关系                                              │
│  ├─ 中等覆盖率（30-40%）                                        │
│  ├─ 精确匹配（基于 NER）                                        │
│  └─ 例子：people (Caroline's mom), places (Italy), orgs         │
│                                                                 │
│  Layer 3: Event Clustering（事件聚类，现有的）                  │
│  ├─ 细粒度、具体事件                                            │
│  ├─ 低覆盖率（6-10%）                                           │
│  ├─ LLM 驱动                                                   │
│  └─ 例子：Caroline's adoption plan, Melanie's camping trip      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

查询路由：
  1. 分析查询类型
  2. 选择合适的聚类层
  3. 可能使用多层（混合检索）
```

### 4.2 Layer 1: Semantic State Clustering

#### 4.2.1 设计思路

**核心思想**：将 MemUnits 按**长期语义状态**分类，而非具体事件。

**与 Event Cluster 的区别**：

| 维度 | Event Cluster | Semantic State Cluster |
|-----|--------------|----------------------|
| **粒度** | 细（具体事件） | 粗（语义主题） |
| **时效性** | 时间相关 | 时间不敏感 |
| **示例** | "Caroline's pottery class on June 5" | "Caroline's hobbies" |
| **覆盖率** | 低（6.8%） | 高（50-60%） |
| **选择方式** | LLM selection | 直接路由/分类 |

#### 4.2.2 预定义类别

```python
# 预定义的语义状态类别
SEMANTIC_CATEGORIES = {
    # === 个人发展 ===
    "career_planning": {
        "name": "职业规划",
        "description": "Career plans, job changes, professional goals",
        "keywords": ["career", "job", "work", "professional", "promotion"]
    },
    "education_learning": {
        "name": "学习教育",
        "description": "Education, courses, learning new skills",
        "keywords": ["learn", "study", "course", "education", "school"]
    },

    # === 兴趣爱好 ===
    "hobbies_interests": {
        "name": "兴趣爱好",
        "description": "Hobbies, interests, recreational activities",
        "keywords": ["hobby", "interest", "painting", "pottery", "reading"]
    },

    # === 人际关系 ===
    "relationships_family": {
        "name": "家庭关系",
        "description": "Family relationships, parenting, family events",
        "keywords": ["family", "parent", "child", "mother", "father"]
    },
    "relationships_friends": {
        "name": "朋友关系",
        "description": "Friendships, social connections",
        "keywords": ["friend", "friendship", "social"]
    },

    # === 健康状况 ===
    "health_physical": {
        "name": "身体健康",
        "description": "Physical health, fitness, medical issues",
        "keywords": ["health", "fitness", "medical", "exercise", "diet"]
    },
    "health_mental": {
        "name": "心理健康",
        "description": "Mental health, emotional wellbeing, therapy",
        "keywords": ["mental", "emotional", "therapy", "stress", "anxiety"]
    },

    # === 生活状态 ===
    "life_goals": {
        "name": "生活目标",
        "description": "Life goals, aspirations, future plans",
        "keywords": ["goal", "dream", "aspiration", "future", "plan"]
    },
    "daily_routines": {
        "name": "日常习惯",
        "description": "Daily routines, habits, lifestyle",
        "keywords": ["routine", "habit", "daily", "lifestyle"]
    },
    "financial_status": {
        "name": "财务状况",
        "description": "Financial situation, income, expenses, savings",
        "keywords": ["money", "finance", "income", "expense", "saving"]
    },

    # === 身份认同 ===
    "identity_beliefs": {
        "name": "身份与信念",
        "description": "Identity, beliefs, values, LGBTQ+, religion",
        "keywords": ["identity", "belief", "value", "LGBTQ", "religion"]
    },

    # === 社会活动 ===
    "community_service": {
        "name": "社区服务",
        "description": "Volunteering, community service, activism",
        "keywords": ["volunteer", "community", "activism", "charity"]
    },

    # === 旅行出行 ===
    "travel_experiences": {
        "name": "旅行经历",
        "description": "Travel, trips, places visited",
        "keywords": ["travel", "trip", "visit", "vacation", "journey"]
    }
}
```

#### 4.2.3 数据结构

```python
@dataclass
class SemanticStateCluster:
    """语义状态聚类"""

    cluster_id: str              # "ssc_career", "ssc_hobbies", ...
    category: str                # "career_planning", "hobbies_interests", ...
    category_name: str           # "职业规划", "兴趣爱好", ...
    description: str             # 类别描述

    members: List[ClusterMember] # 按时间排序
    member_count: int

    # 可选：子分类
    sub_categories: Dict[str, List[str]]  # 例如 hobbies -> {painting: [unit_ids], pottery: [unit_ids]}

    created_at: datetime
    updated_at: datetime

@dataclass
class SemanticStateClusterIndex:
    """语义状态聚类索引"""

    clusters: Dict[str, SemanticStateCluster]  # category -> cluster
    unit_to_categories: Dict[str, List[str]]   # unit_id -> [categories]
    conversation_id: str
    total_units: int
```

#### 4.2.4 聚类算法

```python
class SemanticStateClusterer:
    """语义状态聚类器"""

    async def cluster_memunits(
        self,
        memunit_list: List[Dict],
        conversation_id: str
    ) -> SemanticStateClusterIndex:
        """
        对 MemUnits 进行语义状态聚类

        算法：
        1. 对每个 MemUnit 的 narrative 进行分类
        2. 使用 LLM 或文本分类模型判断属于哪些类别
        3. 一个 MemUnit 可以属于多个类别
        4. 构建索引
        """

        index = SemanticStateClusterIndex(
            clusters={},
            unit_to_categories={},
            conversation_id=conversation_id,
            total_units=len(memunit_list)
        )

        # 初始化所有类别的 cluster
        for category_id, category_info in SEMANTIC_CATEGORIES.items():
            index.clusters[category_id] = SemanticStateCluster(
                cluster_id=f"ssc_{category_id}",
                category=category_id,
                category_name=category_info["name"],
                description=category_info["description"],
                members=[],
                member_count=0,
                sub_categories={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        # 对每个 MemUnit 分类
        for memunit in memunit_list:
            categories = await self._classify_memunit(memunit)
            unit_id = memunit["unit_id"]

            for category in categories:
                # 添加到对应 cluster
                member = ClusterMember(
                    unit_id=unit_id,
                    timestamp=self._parse_timestamp(memunit.get("timestamp")),
                    summary=await self._generate_summary(memunit)
                )
                index.clusters[category].members.append(member)
                index.clusters[category].member_count += 1

                # 更新映射
                if unit_id not in index.unit_to_categories:
                    index.unit_to_categories[unit_id] = []
                index.unit_to_categories[unit_id].append(category)

        # 排序每个 cluster 的 members
        for cluster in index.clusters.values():
            cluster.members.sort(key=lambda m: m.timestamp)

        return index

    async def _classify_memunit(
        self,
        memunit: Dict
    ) -> List[str]:
        """
        将 MemUnit 分类到一个或多个语义类别

        方法 1: 基于 keywords 的简单匹配（快速，但可能不准确）
        方法 2: 使用 LLM 分类（准确，但较慢）
        方法 3: 使用轻量级文本分类模型（平衡）

        推荐：方法 3 或混合方法
        """
        narrative = memunit.get("narrative", "")

        # 示例：LLM 分类
        prompt = f"""
Classify the following memory unit into one or more semantic categories.

Available categories:
{self._format_categories()}

Memory unit:
{narrative}

Return a JSON list of applicable category IDs:
["category_1", "category_2", ...]
"""

        response = await self.llm_provider.generate(prompt)
        categories = self._parse_category_response(response)

        return categories
```

#### 4.2.5 检索集成

```python
async def retrieve_with_semantic_state(
    query: str,
    original_results: List[Tuple[dict, float]],
    semantic_index: SemanticStateClusterIndex,
    config: SemanticStateRetrievalConfig
) -> List[Tuple[dict, float]]:
    """
    使用语义状态聚类增强检索

    流程：
    1. 识别查询的语义类别（如 "职业规划"）
    2. 直接获取该类别下的所有 MemUnits
    3. 与原始检索结果合并
    """

    # Step 1: 识别查询类别
    query_categories = await _identify_query_categories(query)

    # Step 2: 收集相关 MemUnits
    expanded_units = []
    for category in query_categories:
        cluster = semantic_index.clusters.get(category)
        if cluster:
            # 获取最近的 N 个 MemUnits
            recent_members = cluster.members[-config.max_members_per_category:]
            expanded_units.extend(recent_members)

    # Step 3: 合并结果
    final_results = _merge_results(
        original_results,
        expanded_units,
        config
    )

    return final_results

async def _identify_query_categories(query: str) -> List[str]:
    """
    识别查询属于哪些语义类别

    方法：
    1. 关键词匹配（快速）
    2. LLM 判断（准确）
    """

    # 快速关键词匹配
    matched_categories = []
    for category_id, info in SEMANTIC_CATEGORIES.items():
        keywords = info["keywords"]
        if any(keyword in query.lower() for keyword in keywords):
            matched_categories.append(category_id)

    # 如果没匹配到，使用 LLM
    if not matched_categories:
        prompt = f"""
Which semantic category does this query belong to?

Query: {query}

Categories:
{format_categories()}

Return ONE category ID.
"""
        response = await llm_provider.generate(prompt)
        matched_categories = [response.strip()]

    return matched_categories
```

#### 4.2.6 优势分析

| 优势 | 说明 |
|-----|------|
| **高覆盖率** | 50-60% 的问题都涉及语义状态 |
| **易于选择** | 直接关键词匹配或简单分类，不需要复杂的 LLM selection |
| **清晰区分** | 类别之间差异明显，不易混淆 |
| **时间不敏感** | 不会因为多个时间点而混淆 |
| **可扩展** | 可以根据需要增加新类别 |

### 4.3 Layer 2: Entity Relation Clustering

#### 4.3.1 设计思路

**核心思想**：基于**实体和关系**进行聚类，支持实体相关的查询。

**适用场景**：
- "他妈妈最近怎么样？" → 查找 "mother" 实体相关的 MemUnits
- "他去过哪些国家？" → 查找 "location" 实体
- "他在哪里工作？" → 查找 "organization" 实体

#### 4.3.2 实体类型

```python
ENTITY_TYPES = {
    "person": {
        "name": "人物",
        "relation_types": ["family", "friend", "colleague", "mentor"],
        "examples": ["mom", "dad", "Caroline", "Melanie"]
    },
    "location": {
        "name": "地点",
        "relation_types": ["home", "work", "visited", "lived"],
        "examples": ["Italy", "France", "home", "office"]
    },
    "organization": {
        "name": "组织",
        "relation_types": ["employer", "school", "volunteer", "member"],
        "examples": ["company", "university", "NGO"]
    },
    "object": {
        "name": "物品",
        "relation_types": ["owns", "uses", "gift"],
        "examples": ["car", "house", "book", "necklace"]
    }
}
```

#### 4.3.3 数据结构

```python
@dataclass
class EntityRelationCluster:
    """实体关系聚类"""

    cluster_id: str              # "erc_person_001", "erc_location_001"
    entity_type: str             # "person", "location", "organization"
    entity_name: str             # "Caroline's mom", "Italy"
    entity_aliases: List[str]    # ["mom", "mother", "母亲"]
    relation_type: str           # "family", "friend", "visited"

    members: List[ClusterMember] # 提到该实体的 MemUnits
    member_count: int

    # 实体元数据
    entity_metadata: Dict[str, Any]  # 额外信息

    created_at: datetime
    updated_at: datetime

@dataclass
class EntityRelationClusterIndex:
    """实体关系聚类索引"""

    clusters: Dict[str, EntityRelationCluster]  # cluster_id -> cluster
    entity_to_clusters: Dict[str, List[str]]    # entity_name -> [cluster_ids]
    unit_to_entities: Dict[str, List[str]]      # unit_id -> [entity_names]
    conversation_id: str
```

#### 4.3.4 聚类算法

```python
class EntityRelationClusterer:
    """实体关系聚类器"""

    async def cluster_memunits(
        self,
        memunit_list: List[Dict],
        conversation_id: str
    ) -> EntityRelationClusterIndex:
        """
        基于实体和关系进行聚类

        算法：
        1. 对每个 MemUnit 进行 NER（命名实体识别）
        2. 识别实体类型和关系
        3. 为每个唯一实体创建一个 cluster
        4. 将提到该实体的 MemUnits 加入对应 cluster
        """

        index = EntityRelationClusterIndex(
            clusters={},
            entity_to_clusters={},
            unit_to_entities={},
            conversation_id=conversation_id
        )

        entity_counter = {}  # entity_type -> counter

        for memunit in memunit_list:
            unit_id = memunit["unit_id"]
            narrative = memunit.get("narrative", "")

            # NER: 提取实体
            entities = await self._extract_entities(narrative)

            for entity in entities:
                entity_key = f"{entity['type']}:{entity['name']}"

                # 创建或获取 cluster
                if entity_key not in index.entity_to_clusters:
                    # 创建新 cluster
                    if entity['type'] not in entity_counter:
                        entity_counter[entity['type']] = 0
                    entity_counter[entity['type']] += 1

                    cluster_id = f"erc_{entity['type']}_{entity_counter[entity['type']]:03d}"

                    cluster = EntityRelationCluster(
                        cluster_id=cluster_id,
                        entity_type=entity['type'],
                        entity_name=entity['name'],
                        entity_aliases=entity.get('aliases', []),
                        relation_type=entity.get('relation', 'mentioned'),
                        members=[],
                        member_count=0,
                        entity_metadata=entity.get('metadata', {}),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )

                    index.clusters[cluster_id] = cluster
                    index.entity_to_clusters[entity_key] = [cluster_id]

                # 添加 MemUnit 到 cluster
                cluster_ids = index.entity_to_clusters[entity_key]
                for cluster_id in cluster_ids:
                    member = ClusterMember(
                        unit_id=unit_id,
                        timestamp=self._parse_timestamp(memunit.get("timestamp")),
                        summary=f"Mentioned {entity['name']}"
                    )
                    index.clusters[cluster_id].members.append(member)
                    index.clusters[cluster_id].member_count += 1

                # 更新映射
                if unit_id not in index.unit_to_entities:
                    index.unit_to_entities[unit_id] = []
                index.unit_to_entities[unit_id].append(entity['name'])

        return index

    async def _extract_entities(
        self,
        narrative: str
    ) -> List[Dict]:
        """
        从 narrative 中提取实体

        方法：
        1. 使用 spaCy NER
        2. 使用 LLM 提取和分类
        """

        # 示例：使用 LLM
        prompt = f"""
Extract all named entities from the following text.

Text:
{narrative}

Return a JSON list of entities:
[
  {{
    "type": "person|location|organization|object",
    "name": "entity name",
    "aliases": ["alias1", "alias2"],
    "relation": "family|friend|visited|...",
    "metadata": {{}}
  }},
  ...
]
"""

        response = await self.llm_provider.generate(prompt)
        entities = self._parse_entity_response(response)

        return entities
```

#### 4.3.5 检索集成

```python
async def retrieve_with_entity_relation(
    query: str,
    original_results: List[Tuple[dict, float]],
    entity_index: EntityRelationClusterIndex,
    config: EntityRelationRetrievalConfig
) -> List[Tuple[dict, float]]:
    """
    使用实体关系聚类增强检索

    流程：
    1. 从查询中识别实体（如 "他妈妈"）
    2. 查找该实体对应的 cluster
    3. 返回 cluster 中的 MemUnits
    """

    # Step 1: 识别查询中的实体
    query_entities = await _extract_query_entities(query)

    # Step 2: 查找实体对应的 clusters
    related_units = []
    for entity in query_entities:
        entity_key = f"{entity['type']}:{entity['name']}"
        cluster_ids = entity_index.entity_to_clusters.get(entity_key, [])

        for cluster_id in cluster_ids:
            cluster = entity_index.clusters[cluster_id]
            # 获取最近的 N 个 MemUnits
            recent_members = cluster.members[-config.max_members_per_entity:]
            related_units.extend(recent_members)

    # Step 3: 合并结果
    final_results = _merge_results(
        original_results,
        related_units,
        config
    )

    return final_results
```

#### 4.3.6 优势分析

| 优势 | 说明 |
|-----|------|
| **精确匹配** | 基于实体，不会引入无关信息 |
| **清晰关系** | 明确的实体-MemUnit 关系 |
| **支持别名** | 可以处理实体的不同称呼 |
| **与知识图谱兼容** | 可以扩展为知识图谱 |

### 4.4 Layer 3: Event Clustering（保留现有）

#### 4.4.1 保留原因

虽然 Event Cluster 覆盖率低、LLM selection 错误率高，但它在某些场景下仍有价值：

1. **细粒度事件查询**
   - "Caroline 的领养计划进展如何？"
   - 需要跟踪一个具体事件的完整时间线

2. **因果推理**
   - 事件之间的因果关系
   - 按时间顺序理解发展过程

3. **补充其他层**
   - 当语义状态或实体关系无法满足时
   - 提供更细粒度的聚类

#### 4.4.2 改进方向

| 改进点 | 当前问题 | 改进方案 |
|-------|---------|---------|
| **降低 LLM selection 错误** | 98% miss 是 selection 问题 | 减少候选 clusters 数量（10 → 5）<br>改进 selection prompt<br>提供更多上下文 |
| **提高 cluster 区分度** | Summaries 太相似 | 在 topic 中加入更多区分信息<br>改进 summary 生成 |
| **减少时间混淆** | 多个时间点导致混淆 | 在 summary 中明确时间范围<br>支持时间过滤 |
| **降低双重错误** | 检索 + selection 都可能错 | 提高检索召回率<br>fallback 机制 |

---

## 5. 检索流程设计

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Input                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Query Analysis                               │
│  - 识别查询类型（semantic, entity, event, hybrid）               │
│  - 提取关键信息（类别、实体、事件）                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ▼                   ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  Direct MemUnit Retrieval│  │  Cluster-Enhanced Retrieval│
│  - 向量检索               │  │  - 选择合适的聚类层        │
│  - Top-k MemUnits        │  │  - 获取相关 MemUnits      │
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Result Fusion                               │
│  - 合并来自不同源的结果                                           │
│  - 去重、排序                                                    │
│  - 应用配置（max_results, score_threshold）                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Final Results                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Query Analysis（查询分析）

```python
@dataclass
class QueryAnalysisResult:
    """查询分析结果"""

    query_type: str              # "semantic", "entity", "event", "hybrid"

    # Semantic State 相关
    semantic_categories: List[str]  # ["career_planning", ...]

    # Entity Relation 相关
    entities: List[Dict]         # [{"type": "person", "name": "mom"}, ...]

    # Event 相关
    event_keywords: List[str]    # ["adoption", "plan", ...]

    # 其他
    time_constraint: Optional[Dict]  # {"type": "recent", "days": 30}
    complexity: str              # "simple", "complex"

class QueryAnalyzer:
    """查询分析器"""

    async def analyze(self, query: str) -> QueryAnalysisResult:
        """
        分析查询，识别查询类型和相关信息

        方法：
        1. 关键词匹配（快速）
        2. LLM 分析（准确）
        3. 混合方法
        """

        # 示例：使用 LLM
        prompt = f"""
Analyze the following query and identify its type and relevant information.

Query: {query}

Return a JSON object:
{{
  "query_type": "semantic|entity|event|hybrid",
  "semantic_categories": ["category1", ...],
  "entities": [{{"type": "person|location|...", "name": "..."}}],
  "event_keywords": ["keyword1", ...],
  "time_constraint": {{"type": "recent|range|...", "days": 30}},
  "complexity": "simple|complex"
}}
"""

        response = await self.llm_provider.generate(prompt)
        result = self._parse_analysis_response(response)

        return result
```

### 5.3 Cluster Selection Strategy（聚类选择策略）

```python
class ClusterSelectionStrategy:
    """聚类选择策略"""

    async def select_clusters(
        self,
        query_analysis: QueryAnalysisResult,
        semantic_index: SemanticStateClusterIndex,
        entity_index: EntityRelationClusterIndex,
        event_index: GroupEventClusterIndex,
        config: MultiLayerRetrievalConfig
    ) -> Dict[str, List[str]]:
        """
        根据查询分析结果选择合适的聚类

        返回：
        {
          "semantic": [cluster_id, ...],
          "entity": [cluster_id, ...],
          "event": [cluster_id, ...]
        }
        """

        selected = {
            "semantic": [],
            "entity": [],
            "event": []
        }

        # 1. Semantic State Clusters
        if query_analysis.semantic_categories:
            for category in query_analysis.semantic_categories:
                cluster_id = f"ssc_{category}"
                if cluster_id in semantic_index.clusters:
                    selected["semantic"].append(cluster_id)

        # 2. Entity Relation Clusters
        if query_analysis.entities:
            for entity in query_analysis.entities:
                entity_key = f"{entity['type']}:{entity['name']}"
                cluster_ids = entity_index.entity_to_clusters.get(entity_key, [])
                selected["entity"].extend(cluster_ids)

        # 3. Event Clusters
        if query_analysis.query_type in ["event", "hybrid"]:
            # 使用现有的 cluster_rerank 逻辑
            # 但限制候选数量，降低 LLM selection 错误
            event_clusters = await self._select_event_clusters(
                query_analysis,
                event_index,
                max_candidates=5  # 降低从 20 到 5
            )
            selected["event"] = event_clusters

        return selected
```

### 5.4 Result Fusion（结果融合）

```python
async def fuse_multi_layer_results(
    original_results: List[Tuple[dict, float]],
    semantic_results: List[Tuple[dict, float]],
    entity_results: List[Tuple[dict, float]],
    event_results: List[Tuple[dict, float]],
    config: MultiLayerRetrievalConfig
) -> List[Tuple[dict, float]]:
    """
    融合多层聚类的检索结果

    策略：
    1. 去重：同一个 MemUnit 只保留一份
    2. 分数合并：如果 MemUnit 在多个结果中，取最高分或加权平均
    3. 排序：按最终分数排序
    4. 限制：应用 max_results 限制
    """

    # 去重和分数合并
    unit_scores: Dict[str, float] = {}
    unit_docs: Dict[str, dict] = {}

    for results, source, weight in [
        (original_results, "original", config.original_weight),
        (semantic_results, "semantic", config.semantic_weight),
        (entity_results, "entity", config.entity_weight),
        (event_results, "event", config.event_weight)
    ]:
        for doc, score in results:
            unit_id = doc.get("unit_id")
            if not unit_id:
                continue

            # 加权分数
            weighted_score = score * weight

            if unit_id not in unit_scores:
                unit_scores[unit_id] = weighted_score
                unit_docs[unit_id] = doc
            else:
                # 取最高分或加权平均
                if config.fusion_strategy == "max":
                    unit_scores[unit_id] = max(unit_scores[unit_id], weighted_score)
                elif config.fusion_strategy == "average":
                    unit_scores[unit_id] = (unit_scores[unit_id] + weighted_score) / 2
                elif config.fusion_strategy == "sum":
                    unit_scores[unit_id] += weighted_score

    # 排序
    final_results = [
        (unit_docs[unit_id], score)
        for unit_id, score in sorted(
            unit_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    # 限制
    if config.max_total_results:
        final_results = final_results[:config.max_total_results]

    return final_results
```

### 5.5 配置设计

```python
@dataclass
class MultiLayerRetrievalConfig:
    """多层聚类检索配置"""

    # 是否启用各层
    enable_semantic_state: bool = True
    enable_entity_relation: bool = True
    enable_event_cluster: bool = True

    # 权重配置
    original_weight: float = 1.0
    semantic_weight: float = 0.8
    entity_weight: float = 0.9
    event_weight: float = 0.7

    # 融合策略
    fusion_strategy: str = "max"  # "max", "average", "sum"

    # 数量限制
    max_members_per_semantic: int = 10
    max_members_per_entity: int = 10
    max_members_per_event: int = 10
    max_total_results: int = 30

    # 查询分析
    auto_query_analysis: bool = True
    fallback_to_original: bool = True
```

---

## 6. 实施路线图

### 6.1 Phase 1: Semantic State Clustering（优先级：高）

**目标**：实现语义状态聚类，提升覆盖率到 50-60%

**任务**：
1. 设计和实现 `SemanticStateCluster` 数据结构
2. 实现 `SemanticStateClusterer`（LLM 或分类模型）
3. 实现检索集成 `retrieve_with_semantic_state`
4. 在 eval 中测试，验证覆盖率和准确率提升

**预期收益**：
- 覆盖率：6.8% → 50-60%
- 准确率提升：+8-12%（预估）

### 6.2 Phase 2: Entity Relation Clustering（优先级：中）

**目标**：实现实体关系聚类，支持实体相关查询

**任务**：
1. 设计和实现 `EntityRelationCluster` 数据结构
2. 实现 `EntityRelationClusterer`（NER + 关系提取）
3. 实现检索集成 `retrieve_with_entity_relation`
4. 在 eval 中测试

**预期收益**：
- 覆盖率：+30-40%
- 对实体相关问题准确率提升显著

### 6.3 Phase 3: Event Cluster 改进（优先级：低）

**目标**：改进现有 Event Cluster，降低 LLM selection 错误率

**任务**：
1. 减少候选 clusters 数量（20 → 5）
2. 改进 cluster selection prompt
3. 在 topic/summary 中增加区分信息
4. 添加 fallback 机制

**预期收益**：
- LLM selection 错误率：98% → 50-60%
- 对复杂事件查询准确率提升 +3-5%

### 6.4 Phase 4: Multi-Layer Integration（优先级：高）

**目标**：整合三层聚类，实现智能路由和结果融合

**任务**：
1. 实现 `QueryAnalyzer`
2. 实现 `ClusterSelectionStrategy`
3. 实现 `fuse_multi_layer_results`
4. 端到端测试和调优

**预期收益**：
- 整体覆盖率：70-80%
- 整体准确率提升：+10-15%

### 6.5 Phase 5: 优化和扩展（优先级：中）

**任务**：
1. 性能优化（缓存、并行处理）
2. 支持增量更新
3. 添加更多语义类别
4. 与知识图谱集成

---

## 7. 附录

### 7.1 现有代码位置

```
src/memory/group_event_cluster/
├── __init__.py
├── schema.py                    # GroupEventCluster, GroupEventClusterIndex
├── types.py                     # GroupEventClusterConfig, GroupEventClusterRetrievalConfig
├── clusterer.py                 # GroupEventClusterer
├── retrieval.py                 # expand_with_cluster, _expand_cluster_rerank
├── storage.py                   # ClusterStorage, JsonClusterStorage
└── utils.py                     # Prompt templates, parsing functions

eval/adapters/parallax/
└── stage1_5_group_event_cluster.py
```

### 7.2 新代码结构规划

```
src/memory/clustering/
├── __init__.py
├── base.py                      # 基础类和接口
│
├── semantic_state/              # Layer 1: Semantic State Clustering
│   ├── __init__.py
│   ├── schema.py                # SemanticStateCluster, SemanticStateClusterIndex
│   ├── categories.py            # SEMANTIC_CATEGORIES 定义
│   ├── clusterer.py             # SemanticStateClusterer
│   └── retrieval.py             # retrieve_with_semantic_state
│
├── entity_relation/             # Layer 2: Entity Relation Clustering
│   ├── __init__.py
│   ├── schema.py                # EntityRelationCluster, EntityRelationClusterIndex
│   ├── entity_types.py          # ENTITY_TYPES 定义
│   ├── clusterer.py             # EntityRelationClusterer
│   └── retrieval.py             # retrieve_with_entity_relation
│
├── event/                       # Layer 3: Event Clustering (现有的 group_event_cluster)
│   ├── __init__.py
│   ├── schema.py                # GroupEventCluster, GroupEventClusterIndex
│   ├── types.py
│   ├── clusterer.py
│   ├── retrieval.py
│   ├── storage.py
│   └── utils.py
│
└── multi_layer/                 # Multi-Layer Integration
    ├── __init__.py
    ├── query_analyzer.py        # QueryAnalyzer
    ├── cluster_selector.py      # ClusterSelectionStrategy
    ├── result_fusion.py         # fuse_multi_layer_results
    └── config.py                # MultiLayerRetrievalConfig
```

### 7.3 参考文献

1. LoCoMo Benchmark 评估结果
2. Group Event Cluster 设计文档（v1.0）
3. LangChain Multi-Query Retrieval
4. Semantic Scholar: "Hierarchical Clustering for Knowledge Graphs"
