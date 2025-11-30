# 记忆系统设计文档

## 1. 系统概览

记忆系统是一个完整的记忆抽取、存储和检索框架，用于从对话数据中提取结构化记忆，并支持高效的多模式检索。

### 1.1 核心流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           记忆系统架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  抽取流程 (Extraction):                                                  │
│  原始对话 → MemUnit提取 → Memory提取 → MongoDB存储 → ES/Milvus索引        │
│                                                                         │
│  检索流程 (Retrieval):                                                   │
│  查询请求 → 关键词/向量搜索 → ES/Milvus检索 → 结果聚合 → 返回              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **MemUnit** | 记忆单元，从原始对话中提取的最小记忆单位，包含摘要、情景描述等 |
| **EpisodeMemory** | 情景记忆，描述特定事件或对话的叙事性记忆 |
| **SemanticMemory** | 语义记忆，从情景中提取的通用知识和事实 |
| **ProfileMemory** | 用户画像，关于用户的结构化属性信息 |
| **GroupProfileMemory** | 群体画像，关于群组的结构化属性信息 |
| **EventLog** | 事件日志，原子化的事实记录 |

---

## 2. 数据模型 (Schema)

### 2.1 MemUnit 模型

**文件位置**: `src/memory/schema/memunit.py`

```python
class MemUnit(BaseModel):
    """记忆单元 - 从原始对话提取的最小记忆单位"""

    # ===== 标识字段 =====
    unit_id: str                    # 唯一标识 (UUID)
    user_id_list: List[str]         # 涉及的用户列表
    group_id: Optional[str]         # 群组ID (如果是群聊)

    # ===== 原始数据 =====
    original_data: List[Dict]       # 原始消息列表

    # ===== 核心内容 =====
    summary: str                    # 简短摘要 (必填)
    narrative: str                  # 详细叙事描述
    subject: Optional[str]          # 话题/标题
    participants: List[str]         # 参与者列表

    # ===== 衍生记忆 =====
    semantic_memories: List[SemanticMemoryItem]  # 关联语义记忆
    event_log: Optional[EventLog]                # 原子事实日志

    # ===== 元数据 =====
    source_type: SourceType         # 来源类型 (CONVERSATION等)
    timestamp: datetime             # 时间戳
    vector: Optional[List[float]]   # 向量表示
```

### 2.2 EpisodeMemory 模型

**文件位置**: `src/memory/schema/episode_memory.py`

```python
class EpisodeMemory(Memory):
    """情景记忆 - 描述特定事件或对话的叙事性记忆"""

    # ===== 标识字段 =====
    event_id: str                   # 情景记忆自身ID
    user_id: str                    # 记忆所属用户
    memunit_id_list: List[str]      # 源MemUnit ID列表

    # ===== 核心内容 =====
    narrative: str                  # 完整叙事文本
    summary: str                    # 简短摘要
    subject: str                    # 话题/标题

    # ===== 参与信息 =====
    participants: List[str]         # 参与者列表

    # ===== 检索字段 =====
    keywords: List[str]             # 关键词列表
    vector: List[float]             # 向量表示

    # ===== 元数据 =====
    timestamp: datetime             # 时间戳
    importance_score: float         # 重要性分数
```

### 2.3 Memory 基类

**文件位置**: `src/memory/schema/memory.py`

```python
class Memory(BaseModel):
    """所有记忆类型的基类"""

    user_id: str                    # 用户ID
    group_id: Optional[str]         # 群组ID
    source_type: SourceType         # 来源类型
    timestamp: datetime             # 时间戳
    importance_score: float = 0.0   # 重要性分数
```

### 2.4 ID字段关系图

```
MemUnit.unit_id              → MemUnit 的唯一标识 (UUID)
       │
       ▼
Memory.memunit_id_list       → 关联的 MemUnit ID 列表
       │
       ▼
EpisodeMemory.event_id       → EpisodeMemory 自身的 ID
       │
       ▼
ES/Milvus.event_id           → 存储在搜索引擎中的 Episode ID
```

---

## 3. 抽取流程 (Extraction Pipeline)

### 3.1 流程概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         抽取流程详解                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. MemorizeRequest 输入                                                │
│     ├─ history_raw_data_list: 历史消息                                   │
│     ├─ new_raw_data_list: 新消息                                        │
│     ├─ user_id_list: 用户列表                                           │
│     └─ source_type: 来源类型                                            │
│            │                                                            │
│            ▼                                                            │
│  2. MemUnit 提取                                                        │
│     └─ ConvMemUnitExtractor.extract_memunit()                           │
│            │                                                            │
│            ├──▶ 可选: EventLog 提取                                     │
│            └──▶ 可选: SemanticMemory 提取                               │
│            │                                                            │
│            ▼                                                            │
│  3. Memory 提取 (并行)                                                   │
│     ├─ EpisodeMemoryExtractor → EpisodeMemory                           │
│     ├─ ProfileMemoryExtractor → ProfileMemory                           │
│     └─ GroupProfileMemoryExtractor → GroupProfileMemory                 │
│            │                                                            │
│            ▼                                                            │
│  4. 持久化存储                                                           │
│     ├─ MongoDB: 原始记忆数据                                             │
│     ├─ Elasticsearch: 关键词索引                                         │
│     └─ Milvus: 向量索引                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键文件

| 层级 | 文件 | 职责 |
|------|------|------|
| **入口** | `src/services/mem_memorize.py` | 记忆化服务入口，协调整个流程 |
| **编排** | `src/memory/orchestrator/extraction_orchestrator.py` | 编排 MemUnit 和 Memory 的提取 |
| **MemUnit抽取** | `src/memory/extraction/memunit/conv_memunit_extractor.py` | 从对话提取 MemUnit |
| **Episode抽取** | `src/memory/extraction/memory/episode_memory_extractor.py` | 提取情景记忆 |
| **Profile抽取** | `src/memory/extraction/memory/profile/profile_memory_extractor.py` | 提取用户画像 |
| **Semantic抽取** | `src/memory/extraction/memory/semantic_memory_extractor.py` | 提取语义记忆 |

### 3.3 核心抽取逻辑

#### 3.3.1 MemUnit 提取

```python
# 位置: src/memory/extraction/memunit/conv_memunit_extractor.py

async def extract_memunit(request: MemorizeRequest) -> List[MemUnit]:
    """
    从对话数据提取 MemUnit

    步骤:
    1. 合并历史消息和新消息
    2. 调用 LLM 生成摘要和情景描述
    3. 提取参与者列表
    4. 可选: 提取 EventLog 和 SemanticMemory
    5. 返回 MemUnit 列表
    """
```

#### 3.3.2 EpisodeMemory 提取

```python
# 位置: src/memory/extraction/memory/episode_memory_extractor.py

async def extract_memory(request: MemoryExtractRequest) -> List[EpisodeMemory]:
    """
    从 MemUnit 提取 EpisodeMemory

    步骤:
    1. 接收 MemUnit 列表
    2. 为每个参与者生成不同视角的叙事
    3. 调用 LLM 生成 JSON 格式输出 (title, content)
    4. 可选: 异步触发语义记忆提取
    5. 返回 List[EpisodeMemory]
    """
```

### 3.4 提示词配置

系统支持两套提示词: **生产环境** 和 **Eval环境**

| 环境 | 提示词位置 | 用途 |
|------|-----------|------|
| **Production** | `src/prompts/memory/en/production/` | 生产环境，平衡质量和成本 |
| **Eval** | `src/prompts/memory/en/eval/add/` | 评估环境，更详细的提取 |

#### Eval 提示词要求

```
1. 时间处理:
   - 相对时间 + 绝对日期: "last week (May 7, 2023)"

2. 细节保留:
   - 人名: 完整名称 (不用代词)
   - 特殊名词: 品牌、地点、组织名称
   - 数字: 精确数字和数量

3. 频率信息:
   - 频率表示: "每周二和周四"
   - 重复次数: "提到三次"
   - 行为模式: "通常在早上8点"

4. 输出格式:
   {
     "title": "简洁标题 (10-20字)",
     "content": "第三人称叙事，按时间顺序..."
   }
```

---

## 4. 存储层 (Persistence)

### 4.1 存储架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          存储架构                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   MongoDB   │     │  Elasticsearch  │     │     Milvus      │       │
│  │  (主存储)    │────▶│   (关键词搜索)   │     │   (向量搜索)    │       │
│  └─────────────┘     └─────────────────┘     └─────────────────┘       │
│        │                     ▲                       ▲                  │
│        │                     │                       │                  │
│        └─────────────────────┴───────────────────────┘                  │
│                          同步服务                                        │
│              (MemUnitSyncService / MemUnitMilvusSyncService)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 MongoDB 存储

**文档模型位置**: `src/infra/adapters/out/persistence/document/memory/`

| 文件 | 模型 | 用途 |
|------|------|------|
| `episodic_memory.py` | `EpisodicMemory` | 情景记忆文档 |
| `memunit.py` | `MemUnit` | MemUnit 文档 |
| `semantic_memory.py` | `SemanticMemory` | 语义记忆文档 |
| `behavior_history.py` | `BehaviorHistory` | 行为历史文档 |

**Repository 位置**: `src/infra/adapters/out/persistence/repository/`

| 文件 | 主要方法 |
|------|---------|
| `episodic_memory_raw_repository.py` | `append_episodic_memory()`, `get_by_event_id()` |
| `memunit_raw_repository.py` | `insert()`, `get_by_unit_id()` |
| `semantic_memory_raw_repository.py` | `append_semantic_memory()` |

### 4.3 Elasticsearch 索引

**转换器位置**: `src/infra/adapters/out/search/elasticsearch/converter/`

| 文件 | 用途 |
|------|------|
| `episodic_memory_converter.py` | MongoDB → ES 文档转换 |
| `semantic_memory_converter.py` | 语义记忆转换 |
| `event_log_converter.py` | 事件日志转换 |

**ES 文档关键字段**:

```python
# 位置: src/infra/adapters/out/search/elasticsearch/memory/episodic_memory.py

class EpisodicMemoryDoc:
    event_id: str           # 主键
    user_id: str            # 用户ID
    group_id: str           # 群组ID
    search_content: str     # 组合搜索字段 (title + summary + narrative)
    keywords: List[str]     # 关键词列表
    subject: str            # 主题
    vector: List[float]     # 向量 (用于混合搜索)
    timestamp: datetime     # 时间戳
    memunit_id_list: List[str]  # 关联的 MemUnit ID
```

### 4.4 Milvus 向量存储

**转换器位置**: `src/infra/adapters/out/search/milvus/converter/`

| 文件 | 用途 |
|------|------|
| `episodic_memory_milvus_converter.py` | MongoDB → Milvus 文档转换 |

**Milvus Collection 字段**:

```python
# 关键字段
event_id: str           # 主键
user_id: str            # 用户ID (过滤条件)
vector: List[float]     # 向量 (搜索字段)
timestamp: int          # 时间戳 (过滤条件)
```

### 4.5 同步服务

**文件位置**:
- `src/services/memunit_sync.py` - ES 同步
- `src/services/memunit_milvus_sync.py` - Milvus 同步

```python
# 同步流程
async def sync_memunit(memunit: MemUnit, episodic_memories: List[EpisodeMemory]):
    """
    1. 转换 MongoDB 文档为 ES/Milvus 文档
    2. 批量索引到 ES
    3. 批量插入到 Milvus
    4. 刷新索引确保可搜索
    """
```

---

## 5. 检索流程 (Retrieval Pipeline)

### 5.1 检索入口

**文件位置**: `src/agents/memory_manager.py`

```python
class MemoryManager:
    """记忆管理器 - 负责记忆的检索和获取"""

    async def retrieve_mem(request: RetrieveMemoryRequest) -> RetrieveMemoryResponse:
        """根据检索方法分发到具体实现"""

    async def fetch_mem(request: FetchMemoryRequest) -> FetchMemoryResponse:
        """根据ID直接获取记忆"""
```

### 5.2 检索方法

系统支持三种检索方法:

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **KEYWORD** | BM25 关键词搜索 | 精确匹配、专有名词 |
| **VECTOR** | 向量相似度搜索 | 语义理解、模糊匹配 |
| **HYBRID** | 混合搜索 | 综合场景 |

### 5.3 关键词检索流程

```python
# 位置: src/agents/memory_manager.py

async def retrieve_mem_keyword(request: RetrieveMemoryRequest):
    """
    关键词检索流程:

    1. 分词: jieba.cut_for_search(query)
    2. 过滤停用词
    3. 构建时间范围: {"gte": start_time, "lte": end_time}
    4. 调用 ES 搜索:
       EpisodicMemoryEsRepository.multi_search(
           query_words, user_id, event_type, size, date_range
       )
    5. 分组处理: group_by_groupid_stratagy()
    6. 返回结果
    """
```

**ES 搜索字段**:
- `search_content`: 组合字段 (title + summary + narrative)
- `keywords`: 关键词列表
- `subject`: 主题

### 5.4 向量检索流程

```python
# 位置: src/agents/memory_manager.py

async def retrieve_mem_vector(request: RetrieveMemoryRequest):
    """
    向量检索流程:

    1. 向量化查询:
       query_vector = await vectorize_service.get_embedding(query)

    2. 选择 Repository (根据 memory_sub_type):
       - "semantic_memory" → SemanticMemoryMilvusRepository
       - "event_log" → EventLogMilvusRepository
       - 其他 → EpisodicMemoryMilvusRepository

    3. 调用 Milvus 搜索:
       await milvus_repo.vector_search(
           query_vector=query_vector_list,
           user_id=user_id,
           start_time=start_time,
           end_time=end_time,
           limit=top_k,
           radius=radius  # 相似度阈值
       )

    4. 分组处理并返回
    """
```

### 5.5 混合检索流程

```python
async def retrieve_mem_hybrid(request: RetrieveMemoryRequest):
    """
    混合检索流程:

    1. 并行执行关键词搜索和向量搜索
    2. 对结果进行相似度加权融合
    3. 按最终分数排序
    4. 返回 top_k 结果
    """
```

### 5.6 检索请求参数

```python
class RetrieveMemoryRequest(BaseModel):
    user_id: str                    # 用户ID
    query: str                      # 查询文本
    retrieve_method: RetrieveMethod # KEYWORD/VECTOR/HYBRID
    top_k: int = 10                 # 返回数量
    start_time: Optional[datetime]  # 开始时间
    end_time: Optional[datetime]    # 结束时间
    memory_sub_type: Optional[str]  # episode/semantic_memory/event_log
    radius: Optional[float]         # 向量相似度阈值
```

### 5.7 检索响应

```python
class RetrieveMemoryResponse(BaseModel):
    memories: List[MemoryModel]     # 记忆列表
    scores: List[float]             # 相似度分数
    importance_scores: List[float]  # 重要性分数
    original_data: List[Dict]       # 原始数据
    total_count: int                # 总数
```

---

## 6. 完整数据流

### 6.1 抽取流程数据流

```
MemorizeRequest
├─ history_raw_data_list: List[RawData]
├─ new_raw_data_list: List[RawData]
├─ user_id_list: List[str]
└─ source_type: SourceType.CONVERSATION
    │
    ▼ ExtractionOrchestrator.extract_memunit()
    │
MemUnit
├─ unit_id: str (UUID)
├─ original_data: List[Dict]
├─ summary: str
├─ narrative: str
├─ participants: List[str]
├─ semantic_memories: List[SemanticMemoryItem]
└─ event_log: EventLog
    │
    ├──▶ ExtractionOrchestrator.extract_memory()
    │    │
    │    ▼ EpisodeMemoryExtractor.extract_memory()
    │    │
    │    EpisodeMemory (每个参与者一份)
    │    ├─ user_id: str
    │    ├─ event_id: str
    │    ├─ memunit_id_list: [unit_id]
    │    ├─ narrative: str (个人视角)
    │    └─ summary, subject, keywords
    │
    └──▶ EpisodicMemoryRawRepository.append_episodic_memory()
         │
         ├──▶ 向量化: vectorize_service.get_embedding(narrative)
         │
         ├──▶ MongoDB 存储
         │
         └──▶ MemUnitSyncService.sync_memunit()
              │
              ├──▶ ES 索引 (search_content, keywords, subject)
              │
              └──▶ Milvus 索引 (vector)
```

### 6.2 检索流程数据流

```
RetrieveMemoryRequest
├─ user_id: str
├─ query: str
├─ retrieve_method: RetrieveMethod
├─ top_k: int
└─ start_time/end_time: datetime
    │
    ▼ MemoryManager.retrieve_mem()
    │
    ├─ KEYWORD:
    │  ├─ jieba.cut_for_search(query) + 停用词过滤
    │  ├─ EpisodicMemoryEsRepository.multi_search()
    │  └─ BM25 搜索结果
    │
    ├─ VECTOR:
    │  ├─ vectorize_service.get_embedding(query)
    │  ├─ EpisodicMemoryMilvusRepository.vector_search()
    │  └─ 向量相似度结果
    │
    └─ HYBRID:
       ├─ 并行执行 KEYWORD + VECTOR
       └─ 加权融合排序
    │
    ▼ group_by_groupid_stratagy()
    │
    ▼
RetrieveMemoryResponse
├─ memories: List[MemoryModel]
├─ scores: List[float]
└─ total_count: int
```

---

## 7. 关键文件索引

### 7.1 Schema 层

| 文件 | 作用 |
|------|------|
| `src/memory/schema/memory.py` | 记忆基类 |
| `src/memory/schema/memunit.py` | MemUnit 定义 |
| `src/memory/schema/episode_memory.py` | 情景记忆 |
| `src/memory/schema/profile_memory.py` | 用户画像 |
| `src/memory/schema/group_profile_memory.py` | 群体画像 |
| `src/memory/schema/source_type.py` | 来源类型枚举 |

### 7.2 抽取层

| 文件 | 作用 |
|------|------|
| `src/memory/orchestrator/extraction_orchestrator.py` | 抽取编排器 |
| `src/memory/extraction/memunit/conv_memunit_extractor.py` | MemUnit 抽取 |
| `src/memory/extraction/memory/episode_memory_extractor.py` | 情景记忆抽取 |
| `src/memory/extraction/memory/semantic_memory_extractor.py` | 语义记忆抽取 |
| `src/memory/extraction/memory/profile/profile_memory_extractor.py` | 用户画像抽取 |

### 7.3 存储层

| 文件 | 作用 |
|------|------|
| `src/infra/adapters/out/persistence/repository/episodic_memory_raw_repository.py` | MongoDB 情景记忆仓库 |
| `src/infra/adapters/out/persistence/repository/memunit_raw_repository.py` | MongoDB MemUnit 仓库 |
| `src/infra/adapters/out/search/repository/episodic_memory_es_repository.py` | ES 情景记忆仓库 |
| `src/infra/adapters/out/search/repository/episodic_memory_milvus_repository.py` | Milvus 情景记忆仓库 |

### 7.4 服务层

| 文件 | 作用 |
|------|------|
| `src/services/mem_memorize.py` | 记忆化服务入口 |
| `src/services/memunit_sync.py` | ES 同步服务 |
| `src/services/memunit_milvus_sync.py` | Milvus 同步服务 |
| `src/agents/memory_manager.py` | 记忆检索管理器 |
| `src/agents/fetch_memory_service.py` | 记忆获取服务 |

### 7.5 提示词层

| 文件 | 作用 |
|------|------|
| `src/prompts/memory/en/eval/add/episode_mem_prompts.py` | Eval 情景记忆提示词 |
| `src/prompts/memory/en/production/profile_mem_part3_prompts.py` | 生产用户画像提示词 |

---

## 8. 配置参考

### 8.1 环境变量

```bash
# LLM 配置
LLM_PROVIDER=openai          # 或 qwen 等
LLM_MODEL=gpt-4              # 模型名称
LLM_BASE_URL=https://...     # API 基础 URL
LLM_API_KEY=sk-...           # API 密钥
LLM_TEMPERATURE=0.3          # 温度 (抽取推荐 0.3)
LLM_MAX_TOKENS=16384         # 最大 token 数

# 数据库配置
MONGODB_URI=mongodb://...    # MongoDB 连接
ELASTICSEARCH_HOSTS=http://... # ES 连接
MILVUS_HOST=localhost        # Milvus 主机
MILVUS_PORT=19530            # Milvus 端口
```

### 8.2 检索参数推荐

| 场景 | 方法 | top_k | radius |
|------|------|-------|--------|
| 精确查询 | KEYWORD | 10 | - |
| 语义理解 | VECTOR | 20 | 0.7 |
| 综合搜索 | HYBRID | 15 | 0.6 |

---

## 9. 扩展指南

### 9.1 添加新的记忆类型

1. 在 `src/memory/schema/` 下定义新的 Schema
2. 在 `src/memory/extraction/memory/` 下实现 Extractor
3. 在 `src/infra/adapters/out/persistence/` 下添加 MongoDB 文档和 Repository
4. 在 `src/infra/adapters/out/search/` 下添加 ES/Milvus 转换器和 Repository
5. 在 `extraction_orchestrator.py` 中注册新的抽取器

### 9.2 自定义提示词

1. 在 `src/prompts/memory/` 下创建新的提示词文件
2. 在对应的 Extractor 中引入并使用
3. 可通过 `use_eval_prompts` 参数切换提示词集

---

## 10. 常见问题

### Q1: 为什么 EpisodeMemory 需要为每个参与者生成一份?

每个用户对同一对话可能有不同的视角和关注点。为每个参与者生成独立的记忆可以:
- 保持记忆的个性化
- 支持基于用户ID的精确检索
- 避免隐私泄露

### Q2: MemUnit 和 EpisodeMemory 的区别?

- **MemUnit**: 原始提取的记忆单元，包含完整对话信息，是存储的基础单位
- **EpisodeMemory**: 从 MemUnit 衍生的叙事性记忆，针对特定用户视角，用于检索

### Q3: 何时使用向量搜索 vs 关键词搜索?

- **关键词搜索**: 用户查询包含具体名词、品牌、人名等
- **向量搜索**: 用户查询是描述性的、语义性的
- **混合搜索**: 不确定时的默认选择
