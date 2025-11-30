# MemUnit 详细规格说明书

## 目录

1. [概述](#1-概述)
2. [核心架构澄清](#2-核心架构澄清)
3. [MemUnit 定义与定位](#3-memunit-定义与定位)
4. [字段详细说明](#4-字段详细说明)
5. [完整实例分析](#5-完整实例分析)
6. [抽取流程详解](#6-抽取流程详解)
7. [检索流程详解](#7-检索流程详解)
8. [字段映射关系](#8-字段映射关系)
9. [关键文件索引](#9-关键文件索引)

---

## 1. 概述

MemUnit（Memory Unit，记忆单元）是记忆系统的核心数据结构，作为原始对话数据和最终提取记忆之间的**中间表示层**。

### 1.1 核心定位

```
原始消息流 ──→ 边界检测 ──→ MemUnit ──→ 记忆提取 ──→ Memory
                              │
                              ├── 封装语义完整的对话片段
                              ├── 存储原始消息和提取结果
                              └── 作为下游记忆提取的输入单元
```

### 1.2 设计目标

| 目标 | 说明 |
|------|------|
| **语义完整性** | 一个 MemUnit 代表一段话题完整的对话 |
| **信息聚合** | 将分散的消息聚合为结构化单元 |
| **可追溯性** | 保留原始数据，支持记忆溯源 |
| **可扩展性** | 支持衍生多种类型的 Memory |

---

## 2. 核心架构澄清

### 2.1 存储和检索单位：EpisodeMemory 而非 MemUnit

**关键澄清**: 系统的存储和检索单位是 **EpisodeMemory**，而不是 MemUnit。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    存储和检索单位对比                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MemUnit (中间产物)                 EpisodeMemory (最终存储/检索单位)    │
│  ─────────────────                 ─────────────────────────────────    │
│  • 抽取流程的中间表示               • MongoDB 存储的实体                  │
│  • 包含群体视角的原始信息           • ES/Milvus 索引的实体                │
│  • 一个 MemUnit 生成多个 Memory     • 检索返回的单位                      │
│  • 通过 unit_id 被引用              • 每个用户一份独立记忆                │
│                                                                         │
│  关系: 1 MemUnit ───→ N EpisodeMemory (每个参与者一份)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**代码证据**:
```python
# 检索入口 (memory_manager.py)
async def retrieve_mem(...) -> RetrieveMemoryResponse:
    # 调用 EpisodicMemoryEsRepository 或 EpisodicMemoryMilvusRepository
    # 返回的是 EpisodeMemory 列表，不是 MemUnit

# 存储层 (episodic_memory_raw_repository.py)
class EpisodicMemoryRawRepository(BaseRepository[EpisodicMemory]):
    async def append_episodic_memory(self, episodic_memory: EpisodicMemory)
    # 存储的是 EpisodeMemory，不是 MemUnit
```

### 2.2 MemUnit 与 Memory 的包含关系

**关键澄清**: `semantic_memories` 和 `event_log` 是**嵌入在 MemUnit 内部**的，不是独立存储后相互引用。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    数据包含关系                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MemUnit                                                                │
│  ├── unit_id: "uuid-xxx"                                                │
│  ├── episode: "详细叙事..."                                              │
│  ├── summary: "简短摘要..."                                              │
│  ├── semantic_memories: [          ← 嵌入在内部，不是引用                │
│  │     {                                                                │
│  │       content: "用户喜欢咖啡",                                        │
│  │       evidence: "提到每天喝咖啡",                                     │
│  │       start_time: "2023-05-08",                                      │
│  │       embedding: [0.1, 0.2, ...]                                     │
│  │     }                                                                │
│  │   ]                                                                  │
│  └── event_log: {                  ← 嵌入在内部，不是引用                │
│        time: "May 08, 2023",                                            │
│        atomic_fact: [                                                   │
│          "Caroline greeted Melanie",                                    │
│          "Melanie mentioned work stress"                                │
│        ],                                                               │
│        fact_embeddings: [[...], [...]]                                  │
│      }                                                                  │
│                                                                         │
│  注意: SemanticMemory 也有独立存储版本，但那是另一个实体                  │
│       MemUnit 内的 semantic_memories 是 SemanticMemoryItem 类型          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**两种 Semantic Memory 的区别**:

| 类型 | 位置 | 说明 |
|------|------|------|
| `SemanticMemoryItem` | MemUnit.semantic_memories | 嵌入式，附加在 MemUnit 上的**前瞻性预测** |
| `SemanticMemory` | 独立存储在 MongoDB | 独立实体，**事实性知识**，可单独查询 |

```python
# SemanticMemoryItem (嵌入式) - 定义于 semantic_memory.py
@dataclass
class SemanticMemoryItem:
    content: str           # 预测性关联: "用户可能需要咖啡推荐"
    evidence: str          # 证据: "用户提到喜欢咖啡"
    start_time: str        # 生效时间
    embedding: List[float] # 向量

# SemanticMemory (独立存储) - 定义于 semantic_memory.py
@dataclass
class SemanticMemory:
    user_id: str           # 归属用户
    content: str           # 事实性知识: "用户会 Python"
    knowledge_type: str    # 知识类型
    source_episodes: List[str]  # 来源 episode ID
```

### 2.3 字段用途分类：检索 vs 返回给 Prompt

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    字段用途分类                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 用于检索的字段 (Retrieval Fields)                                │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │ 关键词检索 (ES BM25):                                           │   │
│  │   • search_content ← episode 分词后的词列表 (主搜索字段)         │   │
│  │   • subject       ← 标题/主题                                   │   │
│  │   • keywords      ← 关键词列表                                  │   │
│  │                                                                 │   │
│  │ 向量检索 (Milvus ANN):                                          │   │
│  │   • vector        ← episode 向量化后的嵌入                      │   │
│  │                                                                 │   │
│  │ 过滤条件:                                                        │   │
│  │   • user_id       ← 用户归属过滤                                │   │
│  │   • group_id      ← 群组过滤                                    │   │
│  │   • timestamp     ← 时间范围过滤                                │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 返回给 Prompt 的字段 (Response Fields)                           │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │ 核心内容 (构建 Prompt 上下文):                                   │   │
│  │   • episode       ← 详细叙事描述 (主要内容)                      │   │
│  │   • summary       ← 简短摘要                                    │   │
│  │   • subject       ← 标题                                        │   │
│  │                                                                 │   │
│  │ 元数据 (辅助理解):                                               │   │
│  │   • timestamp     ← 事件时间                                    │   │
│  │   • participants  ← 参与者列表                                  │   │
│  │                                                                 │   │
│  │ 不返回给 Prompt:                                                 │   │
│  │   • original_data ← 原始消息 (太长，仅用于调试/溯源)             │   │
│  │   • vector        ← 向量嵌入 (仅用于检索)                       │   │
│  │   • search_content← 分词结果 (仅用于 ES)                        │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    完整数据流                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始对话                                                                │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ MemUnit (中间产物)                                               │   │
│  │ ├── unit_id                                                     │   │
│  │ ├── original_data ─────────────────────────────────────────┐    │   │
│  │ ├── episode (LLM生成)                                      │    │   │
│  │ ├── summary                                                │    │   │
│  │ ├── subject                                                │    │   │
│  │ ├── semantic_memories: [...] (嵌入)                        │    │   │
│  │ └── event_log: {...} (嵌入)                                │    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                        │        │
│      │ 为每个参与者生成                                         │        │
│      ▼                                                        ▼        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ EpisodeMemory (用户A视角)    EpisodeMemory (用户B视角)          │   │
│  │ ├── episode_id               ├── episode_id                    │   │
│  │ ├── user_id: "A"             ├── user_id: "B"                  │   │
│  │ ├── memunit_id_list ─────────┴────→ [unit_id] (引用回MemUnit)  │   │
│  │ ├── episode (个人视角)       ├── episode (个人视角)             │   │
│  │ └── ...                      └── ...                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ├──────────────────────┬──────────────────────┐                   │
│      ▼                      ▼                      ▼                   │
│  MongoDB               Elasticsearch           Milvus                  │
│  (主存储)              (关键词索引)            (向量索引)              │
│  ───────               ─────────────            ──────                  │
│  存储完整              search_content           vector                  │
│  EpisodeMemory         subject                  user_id                │
│                        timestamp                timestamp              │
│                                                                         │
│      │                      │                      │                   │
│      └──────────────────────┴──────────────────────┘                   │
│                             │                                           │
│                             ▼                                           │
│                    ┌─────────────────┐                                  │
│                    │ 检索返回        │                                  │
│                    │ EpisodeMemory   │ ← 返回给 Prompt                  │
│                    │ (episode,       │                                  │
│                    │  summary,       │                                  │
│                    │  subject,       │                                  │
│                    │  timestamp)     │                                  │
│                    └─────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MemUnit 定义与定位

### 3.1 代码位置

**文件路径**: `src/memory/schema/memunit.py`

### 3.2 类定义

```python
@dataclass
class MemUnit:
    """
    记忆单元 (Memory Unit) - 对话内容提取的原子单位

    MemUnit 封装了通过边界检测识别出的一段语义完整的对话内容，
    作为下游记忆提取（情景记忆、语义记忆、用户画像等）的输入。
    """
```

### 3.3 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MemUnit 生命周期                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 创建阶段                                                             │
│     └─ ConvMemUnitExtractor.extract_memunit()                           │
│        └─ 边界检测 → 话题切分 → 生成 MemUnit                              │
│                                                                         │
│  2. 增强阶段                                                             │
│     └─ EpisodeMemoryExtractor.extract_memory()                          │
│        └─ LLM 生成 episode、summary、subject                             │
│        └─ 可选：提取 SemanticMemory、EventLog                            │
│                                                                         │
│  3. 存储阶段                                                             │
│     └─ MemUnitRawRepository.insert()                                    │
│        └─ 向量化 episode 字段                                            │
│        └─ 存储到 MongoDB                                                 │
│                                                                         │
│  4. 索引阶段                                                             │
│     └─ MemUnitSyncService.sync_memunit()                                │
│        └─ 转换为 ES 文档 (BM25 索引)                                      │
│        └─ 转换为 Milvus 文档 (向量索引)                                   │
│                                                                         │
│  5. 检索阶段                                                             │
│     └─ MemoryManager.retrieve_mem()                                     │
│        └─ 关键词搜索 / 向量搜索 / 混合搜索                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 字段详细说明

### 4.1 字段分组概览

```python
# ===== 1. 标识字段 (Identity) =====
unit_id: str                    # 唯一标识符

# ===== 2. 用户字段 (Users) =====
user_id_list: List[str]         # 涉及的所有用户ID
participants: List[str]         # 实际发言的参与者

# ===== 3. 原始数据 (Raw Data) =====
original_data: List[Dict]       # 原始消息列表

# ===== 4. 时间字段 (Timing) =====
timestamp: datetime             # 单元时间戳

# ===== 5. 上下文字段 (Context) =====
group_id: Optional[str]         # 群组ID
type: SourceType                # 数据源类型

# ===== 6. 内容字段 (Content) =====
summary: str                    # 简短摘要
subject: Optional[str]          # 话题/主题
keywords: Optional[List[str]]   # 关键词列表
linked_entities: Optional[List[str]]  # 关联实体
episode: Optional[str]          # 详细情景描述

# ===== 7. 提取结果字段 (Extracted) =====
semantic_memories: Optional[List[SemanticMemoryItem]]  # 语义记忆
event_log: Optional[EventLog]   # 事件日志

# ===== 8. 扩展字段 (Extension) =====
extend: Optional[Dict]          # 自定义元数据（包含 embedding、vector_model）
```

### 4.2 各字段详细说明

#### 4.2.1 标识字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `unit_id` | `str` | ✅ | 唯一标识符，UUID 格式 | `uuid.uuid4()` 自动生成 |

**示例值**:
```json
"unit_id": "3e50f955-a595-45c1-a678-40d6582eea64"
```

**使用场景**:
- 追踪 MemUnit 和衍生 Memory 的关联关系
- EpisodeMemory 的 `memunit_id_list` 字段引用此 ID
- 数据库查询和索引的主键

---

#### 4.2.2 用户字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `user_id_list` | `List[str]` | ✅ | 涉及的所有用户 ID | 请求参数传入 |
| `participants` | `List[str]` | ❌ | 实际发言的参与者 | 从 `original_data` 提取 |

**两者区别**:
- `user_id_list`: 请求时指定的相关用户，用于权限控制和记忆归属
- `participants`: 从消息中实际提取的发言者，是 `user_id_list` 的子集

**示例值**:
```json
"user_id_list": ["melanie_locomo-mini_0", "caroline_locomo-mini_0"],
"participants": ["melanie_locomo-mini_0", "caroline_locomo-mini_0"]
```

**提取逻辑** (`_extract_participant_ids` 方法):
```python
def _extract_participant_ids(self, chat_raw_data_list):
    participant_ids = set()
    for raw_data in chat_raw_data_list:
        # 1. 提取 speaker_id（发言者）
        if 'speaker_id' in raw_data:
            participant_ids.add(raw_data['speaker_id'])
        # 2. 提取 referList 中的 @提及用户
        if 'referList' in raw_data:
            for refer_item in raw_data['referList']:
                participant_ids.add(refer_item['_id'])
    return list(participant_ids)
```

---

#### 4.2.3 原始数据字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `original_data` | `List[Dict]` | ✅ | 原始消息列表 | 请求参数传入 |

**消息结构**:
```json
{
  "speaker_id": "caroline_locomo-mini_0",
  "user_name": "Caroline",
  "speaker_name": "Caroline",
  "content": "Hey Mel! Good to see you! How have you been?",
  "timestamp": "2023-05-08T13:56:00+08:00"
}
```

**字段说明**:
| 子字段 | 说明 |
|--------|------|
| `speaker_id` | 发言者的唯一 ID |
| `user_name` | 用户显示名称 |
| `speaker_name` | 发言者显示名称（通常与 user_name 相同）|
| `content` | 消息文本内容 |
| `timestamp` | ISO 8601 格式的时间戳 |

**消息类型处理**:
```python
SUPPORTED_MSG_TYPES = {
    1: None,       # TEXT - 保持原文本
    2: "[图片]",    # PICTURE
    3: "[视频]",    # VIDEO
    4: "[音频]",    # AUDIO
    5: "[文件]",    # FILE
    6: "[文件]",    # FILES
}
```

---

#### 4.2.4 时间字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `timestamp` | `datetime` | ❌ | 单元时间戳 | 取最后一条消息的时间 |

**时间处理逻辑**:
```python
ts_value = history_message_dict_list[-1].get("timestamp")

if isinstance(ts_value, str):
    timestamp = dt_from_iso_format(ts_value.replace("Z", "+00:00"))
elif isinstance(ts_value, (int, float)):
    timestamp = dt_from_timestamp(ts_value)
else:
    timestamp = get_now_with_timezone()
```

**示例值**:
```json
"timestamp": "2023-05-08T13:56:30+08:00"
```

---

#### 4.2.5 上下文字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `group_id` | `str` | ❌ | 群组 ID，私聊为 None | 请求参数传入 |
| `type` | `SourceType` | ❌ | 数据源类型 | 固定为 `SourceType.CONVERSATION` |

**SourceType 枚举**（定义于 `src/memory/schema/source_type.py`）:
```python
class SourceType(str, Enum):
    CONVERSATION = "Conversation"  # 对话类型
    DOCUMENT = "Document"          # 文档类型
    # ... 其他类型
```

---

#### 4.2.6 内容字段（核心）

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `summary` | `str` | ✅ | 简短摘要（1-2句话）| LLM 生成 |
| `subject` | `str` | ❌ | 话题/主题标题 | LLM 生成 |
| `keywords` | `List[str]` | ❌ | 关键词列表 | LLM 提取 |
| `linked_entities` | `List[str]` | ❌ | 关联实体（人名、地点、品牌等）| LLM 提取 |
| `episode` | `str` | ❌ | 详细情景描述（核心内容）| LLM 生成 |

**字段关系**:
```
summary (简短) ⊂ subject (标题) ⊂ episode (详细)
     │               │               │
     └── 1-2句话 ────┴── 10-20字 ────┴── 完整叙事段落
```

**示例值**:
```json
{
  "summary": "On May 8, 2023, at 1:56 PM UTC, Caroline greeted her friend Melanie with enthusiasm, expressing happiness to see her and inquiring about her well-being...",

  "subject": "Caroline and Melanie Catch Up on Life's Challenges May 8, 2023",

  "episode": "On May 8, 2023, at 1:56 PM UTC, Caroline greeted her friend Melanie with enthusiasm, expressing happiness to see her and inquiring about her well-being. Melanie responded positively but mentioned feeling overwhelmed with her responsibilities related to her children and work. She asked Caroline if there was anything new happening in her life. This interaction highlighted their friendship and mutual support, with Caroline showing genuine interest in Melanie's busy life while Melanie reciprocated by asking about Caroline's updates."
}
```

**Eval 提示词生成要求**:
```
1. 时间处理：相对时间 + 绝对日期
   示例: "last week (May 7, 2023)"

2. 人名保留：使用完整名称，不用代词
   ✅ "Caroline greeted Melanie"
   ❌ "She greeted her"

3. 数字精确：保留具体数字
   ✅ "married for 5 years"
   ❌ "married for several years"

4. 叙事视角：第三人称
   ✅ "Caroline expressed her excitement"
   ❌ "I am excited"
```

---

#### 4.2.7 提取结果字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `semantic_memories` | `List[SemanticMemoryItem]` | ❌ | 语义记忆关联列表 | SemanticMemoryExtractor |
| `event_log` | `EventLog` | ❌ | 事件日志（原子事实）| EventLogExtractor |

**SemanticMemoryItem 结构**:
```python
class SemanticMemoryItem:
    content: str          # 语义记忆内容
    category: str         # 分类
    confidence: float     # 置信度
```

**EventLog 结构**:
```json
{
  "time": "May 08, 2023(Monday) at 01:56 PM",
  "atomic_fact": [
    "Caroline greeted her friend Melanie with enthusiasm.",
    "Caroline expressed happiness to see Melanie.",
    "Caroline inquired about Melanie's well-being.",
    "Melanie responded positively to Caroline.",
    "Melanie mentioned feeling overwhelmed with her responsibilities related to her children and work.",
    "Melanie asked Caroline if there was anything new happening in her life."
  ],
  "fact_embeddings": [
    [0.0123, -0.0456, ...],  // 每个 atomic_fact 的向量
    ...
  ]
}
```

**EventLog 用途**:
- 提供细粒度的事实检索
- 支持事实级别的向量搜索
- 用于问答系统的证据定位

---

#### 4.2.8 扩展字段

| 字段 | 类型 | 必填 | 说明 | 来源 |
|------|------|------|------|------|
| `extend` | `Dict[str, Any]` | ❌ | 自定义元数据 | 运行时填充 |

**常见 extend 内容**:
```json
{
  "embedding": [0.0123, -0.0456, ...],  // episode 的向量表示
  "vector_model": "bge-m3"              // 向量化模型名称
}
```

**填充逻辑**:
```python
text_for_embed = episode_result.narrative or episode_result.summary or ""
if text_for_embed:
    vs = get_vectorize_service()
    vec = await vs.get_embedding(text_for_embed)
    episode_result.extend = episode_result.extend or {}
    episode_result.extend["embedding"] = vec.tolist()
    episode_result.extend["vector_model"] = vs.get_model_name()
```

---

## 5. 完整实例分析

### 5.1 原始对话数据（Eval 测试集）

**来源**: `eval/data/locomo/locomo-mini.json`

```json
{
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {
        "speaker": "Caroline",
        "dia_id": "D1:1",
        "text": "Hey Mel! Good to see you! How have you been?"
      },
      {
        "speaker": "Melanie",
        "dia_id": "D1:2",
        "text": "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?"
      }
    ]
  }
}
```

### 5.2 抽取出的 MemUnit

**来源**: `eval/results/locomo-mini/memunits/memunit_list_conv_0.json`

```json
{
  "unit_id": "3e50f955-a595-45c1-a678-40d6582eea64",

  "user_id_list": [
    "melanie_locomo-mini_0",
    "caroline_locomo-mini_0"
  ],

  "participants": [
    "melanie_locomo-mini_0",
    "caroline_locomo-mini_0"
  ],

  "original_data": [
    {
      "speaker_id": "caroline_locomo-mini_0",
      "user_name": "Caroline",
      "speaker_name": "Caroline",
      "content": "Hey Mel! Good to see you! How have you been?",
      "timestamp": "2023-05-08T13:56:00+08:00"
    },
    {
      "speaker_id": "melanie_locomo-mini_0",
      "user_name": "Melanie",
      "speaker_name": "Melanie",
      "content": "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?",
      "timestamp": "2023-05-08T13:56:30+08:00"
    }
  ],

  "timestamp": "2023-05-08T13:56:30+08:00",
  "group_id": null,
  "type": "Conversation",

  "summary": "On May 8, 2023, at 1:56 PM UTC, Caroline greeted her friend Melanie with enthusiasm, expressing happiness to see her and inquiring about her well-being. Melanie responded positively but mentioned feel...",

  "subject": "Caroline and Melanie Catch Up on Life's Challenges May 8, 2023",

  "keywords": null,
  "linked_entities": null,

  "episode": "On May 8, 2023, at 1:56 PM UTC, Caroline greeted her friend Melanie with enthusiasm, expressing happiness to see her and inquiring about her well-being. Melanie responded positively but mentioned feeling overwhelmed with her responsibilities related to her children and work. She asked Caroline if there was anything new happening in her life. This interaction highlighted their friendship and mutual support, with Caroline showing genuine interest in Melanie's busy life while Melanie reciprocated by asking about Caroline's updates.",

  "semantic_memories": null,

  "event_log": {
    "time": "May 08, 2023(Monday) at 01:56 PM",
    "atomic_fact": [
      "Caroline greeted her friend Melanie with enthusiasm.",
      "Caroline expressed happiness to see Melanie.",
      "Caroline inquired about Melanie's well-being.",
      "Melanie responded positively to Caroline.",
      "Melanie mentioned feeling overwhelmed with her responsibilities related to her children and work.",
      "Melanie asked Caroline if there was anything new happening in her life.",
      "The interaction highlighted the friendship and mutual support between Caroline and Melanie.",
      "Caroline showed genuine interest in Melanie's busy life.",
      "Melanie reciprocated by asking about Caroline's updates."
    ],
    "fact_embeddings": [[...], [...], ...]
  }
}
```

### 5.3 检索结果示例

**查询**: "When did Caroline go to the LGBTQ support group?"

**来源**: `eval/results/locomo-mini/search_results.json`

```json
{
  "query": "When did Caroline go to the LGBTQ support group?",
  "results": [
    {
      "content": "On May 8, 2023 at 1:59 PM UTC, Caroline shared her powerful experience attending an LGBTQ support group the previous day (May 7, 2023). She expressed how inspiring the transgender stories she heard were...",
      "score": 0.9981704950332642,
      "metadata": {
        "subject": "Caroline's Empowering Experience at the LGBTQ Support Group on May 7, 2023",
        "summary": "On May 8, 2023 at 1:59 PM UTC, Caroline shared her powerful experience attending an LGBTQ support group the previous day (May 7, 2023)..."
      }
    }
  ]
}
```

### 5.4 字段转换关系

```
MemUnit                    →  ES Document (检索)
────────────────────────────────────────────────
unit_id                    →  (不直接存储，通过 memunit_id_list 关联)
episode                    →  episode, search_content (分词后)
subject                    →  subject, title
summary                    →  summary
timestamp                  →  timestamp
user_id_list               →  (用于生成多个 EpisodeMemory)
participants               →  participants
group_id                   →  group_id
extend.embedding           →  (存储到 Milvus)
```

---

## 6. 抽取流程详解

### 6.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MemUnit 抽取流程                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 输入: MemUnitExtractRequest                                         │
│     ├─ history_raw_data_list: 历史消息                                   │
│     ├─ new_raw_data_list: 新消息                                        │
│     ├─ user_id_list: 用户列表                                           │
│     └─ group_id: 群组ID                                                 │
│            │                                                            │
│            ▼                                                            │
│  2. 消息预处理 (_data_process)                                           │
│     ├─ 过滤不支持的消息类型 (如系统消息)                                   │
│     └─ 非文本消息转为占位符 (如 "[图片]")                                  │
│            │                                                            │
│            ▼                                                            │
│  3. 边界检测 (_detect_boundary)                                         │
│     ├─ 格式化对话历史和新消息                                             │
│     ├─ 计算时间间隔                                                      │
│     ├─ 调用 LLM 判断是否话题结束                                          │
│     └─ 返回 BoundaryDetectionResult:                                    │
│        ├─ should_end: 是否应结束当前话题                                  │
│        ├─ should_wait: 是否应等待更多消息                                 │
│        ├─ topic_summary: 话题摘要                                        │
│        └─ confidence: 置信度                                             │
│            │                                                            │
│            ├─ should_end=True                                           │
│            │       │                                                    │
│            │       ▼                                                    │
│  4. 创建 MemUnit                                                        │
│     ├─ 生成 unit_id (UUID)                                              │
│     ├─ 提取 participants                                                │
│     ├─ 设置 timestamp (最后一条消息时间)                                  │
│     └─ 设置 summary (边界检测的 topic_summary)                            │
│            │                                                            │
│            ▼                                                            │
│  5. 触发 EpisodeMemoryExtractor                                         │
│     ├─ 调用 LLM 生成 episode (详细叙事)                                   │
│     ├─ 生成 subject (标题)                                               │
│     └─ 可选: 提取 SemanticMemory、EventLog                               │
│            │                                                            │
│            ▼                                                            │
│  6. 向量化                                                               │
│     ├─ 调用 vectorize_service.get_embedding(episode)                    │
│     └─ 存储到 extend.embedding                                          │
│            │                                                            │
│            ▼                                                            │
│  7. 返回 (MemUnit, StatusResult)                                        │
│                                                                         │
│            ├─ should_wait=True                                          │
│            │       │                                                    │
│            │       ▼                                                    │
│            └─ 返回 (None, StatusResult{should_wait=True})               │
│               └─ 等待更多消息再处理                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 关键代码

**边界检测提示词** (Eval 模式):
```python
# 位置: src/prompts/memory/en/eval/add/conv_prompts.py

CONV_BOUNDARY_DETECTION_PROMPT = """
Analyze the conversation and determine if a topic boundary exists.

Conversation History:
{conversation_history}

New Messages:
{new_messages}

Time Gap Information:
{time_gap_info}

Output JSON:
{
    "should_end": true/false,
    "should_wait": true/false,
    "reasoning": "...",
    "confidence": 0.0-1.0,
    "topic_summary": "..."
}
"""
```

**Episode 生成提示词** (Eval 模式):
```python
# 位置: src/prompts/memory/en/eval/add/episode_mem_prompts.py

GROUP_EPISODE_GENERATION_PROMPT = """
Generate a detailed narrative description of this conversation.

Requirements:
1. Use third-person perspective
2. Include timestamps: relative time (absolute date)
3. Preserve all names - never use pronouns
4. Keep specific numbers and quantities
5. Maintain brand names, locations, organizations

Output JSON:
{
    "title": "Concise title (10-20 characters)",
    "content": "Detailed narrative..."
}
"""
```

---

## 7. 检索流程详解

### 7.1 检索架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        检索流程                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RetrieveMemoryRequest                                                  │
│  ├─ query: "When did Caroline go to the LGBTQ support group?"           │
│  ├─ user_id: "caroline_locomo-mini_0"                                   │
│  ├─ retrieve_method: KEYWORD / VECTOR / HYBRID                          │
│  └─ top_k: 10                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MemoryManager.retrieve_mem()                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ├─────────────────┬─────────────────┬────────────────────┐       │
│       ▼                 ▼                 ▼                    │       │
│  ┌─────────┐      ┌─────────┐      ┌─────────────┐            │       │
│  │ KEYWORD │      │ VECTOR  │      │   HYBRID    │            │       │
│  └─────────┘      └─────────┘      └─────────────┘            │       │
│       │                 │                 │                    │       │
│       ▼                 ▼                 │                    │       │
│  Elasticsearch     Milvus               并行执行              │       │
│  (BM25)            (ANN)                + 融合               │       │
│       │                 │                 │                    │       │
│       └─────────────────┴─────────────────┘                    │       │
│                         │                                       │       │
│                         ▼                                       │       │
│              RetrieveMemoryResponse                            │       │
│              ├─ memories: List[MemoryModel]                    │       │
│              └─ scores: List[float]                            │       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 检索字段映射

#### 7.2.1 Elasticsearch (关键词检索)

**索引字段** (定义于 `EpisodicMemoryDoc`):

| 字段 | 类型 | 用途 | 来源 |
|------|------|------|------|
| `event_id` | Keyword | 主键 | MongoDB `_id` |
| `user_id` | Keyword | 过滤条件 | MemUnit.user_id_list 展开 |
| `search_content` | Text (分词) | **BM25 搜索主字段** | narrative 分词结果 |
| `subject` | Text | 辅助搜索 | MemUnit.subject |
| `narrative` | Text | 存储原文 | MemUnit.narrative |
| `timestamp` | Date | 时间过滤 | MemUnit.timestamp |
| `group_id` | Keyword | 过滤条件 | MemUnit.group_id |

**search_content 生成逻辑**:
```python
def _build_search_content(cls, source_doc):
    text_content = []
    if source_doc.narrative:
        text_content.append(source_doc.narrative)

    combined_text = ' '.join(text_content)
    search_content = list(jieba.cut(combined_text))

    # 过滤停用词
    query_words = filter_stopwords(search_content, min_length=2)
    return [word.strip() for word in query_words if word.strip()]
```

**搜索查询示例**:
```python
async def multi_search(self, query: List[str], user_id: str, ...):
    # 构建 BM25 查询
    search = EpisodicMemoryDoc.search()

    # 必须匹配 user_id
    search = search.filter("term", user_id=user_id)

    # search_content 字段匹配查询词
    should_queries = [Q("match", search_content=word) for word in query]
    search = search.query(Q("bool", should=should_queries, minimum_should_match=1))

    # 时间范围过滤
    if date_range:
        search = search.filter("range", timestamp=date_range)

    return await search.execute()
```

#### 7.2.2 Milvus (向量检索)

**Collection 字段**:

| 字段 | 类型 | 用途 |
|------|------|------|
| `event_id` | VARCHAR | 主键 |
| `user_id` | VARCHAR | 过滤条件 |
| `vector` | FLOAT_VECTOR | **向量搜索字段** |
| `timestamp` | INT64 | 时间过滤 |

**向量来源**:
```python
# MemUnit.extend.embedding
vec = await vectorize_service.get_embedding(episode)
```

**搜索查询示例**:
```python
async def vector_search(self, query_vector, user_id, limit=10, radius=0.7):
    search_params = {"metric_type": "COSINE", "params": {"radius": radius}}

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=limit,
        expr=f'user_id == "{user_id}"',
        output_fields=["event_id", "timestamp"]
    )
    return results
```

### 7.3 检索结果处理

```python
# 检索后的处理流程
results = await episodic_memory_es_repo.multi_search(query_words, user_id, ...)

# 分组策略
grouped_results = group_by_groupid_stratagy(results)

# 构建响应
response = RetrieveMemoryResponse(
    memories=[MemoryModel.from_es_doc(doc) for doc in results],
    scores=[hit.meta.score for hit in results],
    total_count=len(results)
)
```

---

## 8. 字段映射关系

### 8.1 完整映射图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MemUnit 字段映射关系                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MemUnit (Schema)                                                       │
│  ├─ unit_id ──────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  ├─ narrative ─┬─→ MongoDB.EpisodicMemory.narrative               │    │
│  │             ├─→ ES.EpisodicMemoryDoc.narrative                 │    │
│  │            ├─→ ES.EpisodicMemoryDoc.search_content (分词)      │    │
│  │            └─→ Milvus.vector (向量化后)                        │    │
│  │                                                                 │    │
│  ├─ subject ──┬─→ MongoDB.EpisodicMemory.subject                  │    │
│  │            └─→ ES.EpisodicMemoryDoc.subject, title              │    │
│  │                                                                 │    │
│  ├─ summary ──┬─→ MongoDB.EpisodicMemory.summary                  │    │
│  │            └─→ ES.EpisodicMemoryDoc.summary                    │    │
│  │                                                                 │    │
│  ├─ timestamp ┬─→ MongoDB.EpisodicMemory.timestamp                │    │
│  │            ├─→ ES.EpisodicMemoryDoc.timestamp                  │    │
│  │            └─→ Milvus.timestamp (int)                          │    │
│  │                                                                 │    │
│  ├─ user_id_list ─→ 展开为多个 EpisodeMemory                       │    │
│  │                  每个用户一份记忆                                │    │
│  │                                                                 │    │
│  ├─ participants ─→ MongoDB/ES.participants                       │    │
│  │                                                                 │    │
│  ├─ group_id ────→ MongoDB/ES/Milvus.group_id                     │    │
│  │                                                                 │    │
│  ├─ event_log ───→ MongoDB.EventLog                               │    │
│  │                └─→ Milvus.EventLog (fact_embeddings)           │    │
│  │                                                                 │    │
│  └─ extend.embedding ─→ Milvus.vector                              │    │
│                                                                 ▼    │
│                                                                      │
│  EpisodeMemory.memunit_id_list ←─────────────────────────────────────┘  │
│  (关联回 MemUnit)                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 存储层对比

| MemUnit 字段 | MongoDB 字段 | ES 字段 | Milvus 字段 | 用途 |
|-------------|--------------|---------|-------------|------|
| `unit_id` | `unit_id` | - | - | 唯一标识 |
| `episode` | `episode` | `episode`, `search_content` | `vector` | 核心内容 |
| `subject` | `subject` | `subject`, `title` | - | 标题 |
| `summary` | `summary` | `summary` | - | 摘要 |
| `timestamp` | `timestamp` | `timestamp` | `timestamp` | 时间过滤 |
| `user_id_list` | - | `user_id` (展开) | `user_id` | 用户归属 |
| `participants` | `participants` | `participants` | - | 参与者 |
| `group_id` | `group_id` | `group_id` | `group_id` | 群组过滤 |
| `event_log` | `event_log` | - | `fact_embeddings` | 原子事实 |
| `extend.embedding` | `extend.embedding` | - | `vector` | 向量表示 |

---

## 9. 关键文件索引

### 9.1 Schema 定义

| 文件 | 说明 |
|------|------|
| [src/memory/schema/memunit.py](../../src/memory/schema/memunit.py) | MemUnit 数据类定义 |
| [src/memory/schema/source_type.py](../../src/memory/schema/source_type.py) | SourceType 枚举 |
| [src/memory/schema/episode_memory.py](../../src/memory/schema/episode_memory.py) | EpisodeMemory 定义 |

### 9.2 抽取逻辑

| 文件 | 说明 |
|------|------|
| [src/memory/extraction/memunit/conv_memunit_extractor.py](../../src/memory/extraction/memunit/conv_memunit_extractor.py) | MemUnit 抽取器 |
| [src/memory/extraction/memory/episode_memory_extractor.py](../../src/memory/extraction/memory/episode_memory_extractor.py) | Episode 抽取器 |
| [src/memory/orchestrator/extraction_orchestrator.py](../../src/memory/orchestrator/extraction_orchestrator.py) | 抽取编排器 |

### 9.3 提示词

| 文件 | 说明 |
|------|------|
| [src/prompts/memory/en/eval/add/conv_prompts.py](../../src/prompts/memory/en/eval/add/conv_prompts.py) | Eval 边界检测提示词 |
| [src/prompts/memory/en/eval/add/episode_mem_prompts.py](../../src/prompts/memory/en/eval/add/episode_mem_prompts.py) | Eval Episode 生成提示词 |

### 9.4 存储层

| 文件 | 说明 |
|------|------|
| [src/infra/adapters/out/persistence/document/memory/memunit.py](../../src/infra/adapters/out/persistence/document/memory/memunit.py) | MongoDB MemUnit 文档 |
| [src/infra/adapters/out/persistence/repository/memunit_raw_repository.py](../../src/infra/adapters/out/persistence/repository/memunit_raw_repository.py) | MongoDB Repository |
| [src/infra/adapters/out/search/elasticsearch/converter/episodic_memory_converter.py](../../src/infra/adapters/out/search/elasticsearch/converter/episodic_memory_converter.py) | ES 转换器 |
| [src/infra/adapters/out/search/repository/episodic_memory_es_repository.py](../../src/infra/adapters/out/search/repository/episodic_memory_es_repository.py) | ES Repository |

### 9.5 检索逻辑

| 文件 | 说明 |
|------|------|
| [src/agents/memory_manager.py](../../src/agents/memory_manager.py) | 记忆管理器（检索入口）|
| [src/agents/fetch_memory_service.py](../../src/agents/fetch_memory_service.py) | 记忆获取服务 |

### 9.6 Eval 测试数据

| 文件 | 说明 |
|------|------|
| [eval/data/locomo/locomo-mini.json](../../eval/data/locomo/locomo-mini.json) | 原始对话数据 |
| [eval/results/locomo-mini/memunits/memunit_list_conv_0.json](../../eval/results/locomo-mini/memunits/memunit_list_conv_0.json) | 抽取的 MemUnit |
| [eval/results/locomo-mini/search_results.json](../../eval/results/locomo-mini/search_results.json) | 检索结果 |

---

## 附录：验证规则

```python
def __post_init__(self):
    """初始化后验证必填字段"""
    if not self.unit_id:
        raise ValueError("unit_id 是必填字段")
    if not self.original_data:
        raise ValueError("original_data 是必填字段")
    if not self.summary:
        raise ValueError("summary 是必填字段")
```

| 字段 | 验证规则 |
|------|---------|
| `unit_id` | 必填，不能为空 |
| `original_data` | 必填，不能为空列表 |
| `summary` | 必填，不能为空字符串 |
