# Parallax 时间精确化表示方案设计

## 1. 现状调研

### 1.1 当前时间处理架构

```
原始消息 (timestamp: ISO string or int/float)
    ↓
├─ ConvMemUnitExtractor._data_process()
│  (消息预处理，保留 timestamp)
    ↓
├─ _calculate_time_gap()
│  (计算时间间隔，用于边界检测)
    ↓
├─ dt_from_iso_format() / dt_from_timestamp()
│  (解析为 datetime 对象，统一为带时区)
    ↓
└─ MemUnit.timestamp (datetime)
   └─ EventLog.time (str: "March 10, 2024(Sunday) at 2:00 PM")
      └─ atomic_fact[] (每个事实独立的 embedding)
```

### 1.2 现有时间字段分布

| 组件 | 时间字段 | 类型 | 精度 | 用途 |
|------|---------|------|------|------|
| **MemUnit** | `timestamp` | `datetime` | 秒级 | 单元整体时间戳 |
| **EventLog** | `time` | `str` | 人类可读 | "March 10, 2024(Sunday) at 2:00 PM" |
| **EventLog** | `atomic_fact[]` | `List[str]` | 无时间 | 原子事实列表 |
| **SemanticMemoryItem** | `start_time` | `str` | 日期 | "YYYY-MM-DD" |
| **SemanticMemoryItem** | `end_time` | `str` | 日期 | "YYYY-MM-DD" |
| **GroupEventCluster** | `first_timestamp` | `datetime` | 秒级 | 最早成员时间 |
| **GroupEventCluster** | `last_timestamp` | `datetime` | 秒级 | 最晚成员时间 |

### 1.3 评估中的时间错误分析

根据 `docs/evaluation/error_analysis_report.md`：

- **时间日期错误占比**: 18.6% (19个实例)
- **Category 2 (时间相关) 准确率**: 92.3% (相对较低)

**典型错误案例**：

| 问题 | 正确答案 | 错误答案 | 错误原因 |
|------|---------|---------|---------|
| When did Gina open her online clothing store? | 16 March, 2023 | late January 2023 | 混淆广告活动时间和开店时间 |
| When did Gina go to a dance class? | 21 July 2023 | July 14, 2023 | 日期错误 |
| When did John have his first firefighter call-out? | The sunday before 3 July 2023 | July 23, 2023 | 相对时间解析错误 |

**根本原因**：
1. 事件时间线管理不准确
2. 混淆了相关但不同事件的时间
3. 时间表达形式不统一（绝对时间 vs 相对时间）
4. 记忆中时间信息提取或存储有误

---

## 2. Zep 双时态模型技术分析

### 2.1 核心设计理念

Zep 采用 **Graphiti** 作为其核心时间感知知识图谱引擎，实现了创新的双时态架构：

```
G = (N, E, φ)
- N: 节点集合 (实体)
- E: 边集合 (关系)
- φ: 形式化关联函数
```

### 2.2 双时态模型 (Bi-Temporal Model)

Zep 维护两条独立的时间线：

| 时间线 | 符号 | 含义 | 用途 |
|--------|------|------|------|
| **事件时间线 (T)** | T | 事件实际发生的时间顺序 | 对话数据的动态特性建模 |
| **事务时间线 (T')** | T' | 数据摄入系统的事务顺序 | 传统数据库审计 |

### 2.3 边的时间属性结构

每条知识图谱边存储 **4个时间戳**：

```python
class Edge:
    # 事务时间 (T') - 系统记录时间
    t_created: datetime   # 事实在系统中创建的时间
    t_expired: datetime   # 事实在系统中失效的时间

    # 有效时间 (T) - 事实真实有效期
    t_valid: datetime     # 事实开始为真的时间
    t_invalid: datetime   # 事实停止为真的时间
```

**示例**：
```
事实: "John在ABC公司工作"
- t_valid: 2020-01-15 (入职日期)
- t_invalid: 2023-06-30 (离职日期)
- t_created: 2020-01-20 (系统记录时间)
- t_expired: NULL (仍在系统中有效)
```

### 2.4 时间冲突检测与解决

当新边被引入时：
1. LLM 比较它与语义相关的现有边
2. 识别潜在矛盾
3. 对于时间重叠的冲突，将受影响边的 `t_invalid` 设定为新边的 `t_valid`
4. 系统优先考虑新信息进行边的失效判定

### 2.5 时间表达处理

Zep 支持两类时间信息的提取：

| 类型 | 示例 | 处理方式 |
|------|------|---------|
| **绝对时间戳** | "艾伦·图灵出生于1912年6月23日" | 直接提取 |
| **相对时间戳** | "我两周前开始新工作" | 基于参考时间 `t_ref` 计算 |

---

## 3. 精确时间表示方案设计

### 3.1 方案概述

基于 Zep 的双时态模型思想，结合 Parallax 现有架构，设计 **三层时间精确化方案**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: 查询层                          │
│     时间感知检索 + 时间推理 + 时间线重建                      │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: 存储层                          │
│     TimedFact 精确时间索引 + 有效性区间                       │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: 提取层                          │
│     时间表达识别 + 相对时间解析 + 事实-时间绑定               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Layer 1: 时间表达提取层

#### 3.2.1 新增数据结构：TemporalExpression

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

class TemporalType(str, Enum):
    """时间表达类型"""
    ABSOLUTE = "absolute"      # 绝对时间: "2023年3月16日"
    RELATIVE = "relative"      # 相对时间: "两周前", "昨天"
    DURATION = "duration"      # 持续时间: "三年来", "五个月"
    RECURRING = "recurring"    # 周期性: "每周一", "每年夏天"
    VAGUE = "vague"           # 模糊时间: "很久以前", "最近"

class TemporalGranularity(str, Enum):
    """时间粒度"""
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    UNKNOWN = "unknown"

@dataclass
class TemporalExpression:
    """时间表达式 - 从文本中提取的时间信息"""

    # === 原始信息 ===
    original_text: str
    """原始时间表达文本，如 "两周前", "March 16, 2023" """

    temporal_type: TemporalType
    """时间类型"""

    # === 解析结果 ===
    resolved_timestamp: Optional[datetime] = None
    """解析后的精确时间戳（如果可解析）"""

    resolved_date_str: Optional[str] = None
    """解析后的日期字符串，格式: "YYYY-MM-DD" """

    confidence: float = 1.0
    """解析置信度 (0.0 - 1.0)"""

    granularity: TemporalGranularity = TemporalGranularity.UNKNOWN
    """时间粒度"""

    # === 有效性区间 (借鉴 Zep) ===
    valid_from: Optional[datetime] = None
    """事实开始有效的时间"""

    valid_until: Optional[datetime] = None
    """事实停止有效的时间（None 表示仍然有效）"""

    # === 上下文 ===
    reference_time: Optional[datetime] = None
    """用于解析相对时间的参考时间"""

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "temporal_type": self.temporal_type.value,
            "resolved_timestamp": self.resolved_timestamp.isoformat() if self.resolved_timestamp else None,
            "resolved_date_str": self.resolved_date_str,
            "confidence": self.confidence,
            "granularity": self.granularity.value,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "reference_time": self.reference_time.isoformat() if self.reference_time else None,
        }
```

#### 3.2.2 新增数据结构：TimedFact

```python
@dataclass
class TimedFact:
    """
    带时间标注的原子事实 - 替代原来的 atomic_fact 字符串

    这是时间精确化的核心数据结构，将事实与精确时间绑定。
    """

    # === 内容 ===
    fact_id: str
    """事实唯一标识符"""

    content: str
    """事实内容（原子事实文本）"""

    # === 时间信息 ===
    event_time: Optional[TemporalExpression] = None
    """事件发生时间（事实描述的事件何时发生）"""

    mention_time: Optional[datetime] = None
    """提及时间（用户何时提到这个事实）"""

    # === 有效性区间 (Zep 风格) ===
    valid_from: Optional[datetime] = None
    """事实开始有效的时间"""

    valid_until: Optional[datetime] = None
    """事实停止有效的时间（None 表示持续有效）"""

    is_current: bool = True
    """是否为当前有效的事实（支持时间冲突解决）"""

    # === 事实类型 ===
    fact_type: Literal["event", "state", "plan", "preference"] = "event"
    """
    事实类型:
    - event: 一次性事件 ("Gina opened her store on March 16")
    - state: 持续状态 ("John works at ABC company")
    - plan: 计划/意图 ("Caroline plans to adopt a cat")
    - preference: 偏好 ("John likes basketball")
    """

    # === 检索 ===
    embedding: Optional[list[float]] = None
    """向量嵌入"""

    # === 元数据 ===
    source_unit_id: Optional[str] = None
    """来源 MemUnit ID"""

    superseded_by: Optional[str] = None
    """被哪个更新的事实替代（用于时间冲突解决）"""

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "event_time": self.event_time.to_dict() if self.event_time else None,
            "mention_time": self.mention_time.isoformat() if self.mention_time else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_current": self.is_current,
            "fact_type": self.fact_type,
            "embedding": self.embedding,
            "source_unit_id": self.source_unit_id,
            "superseded_by": self.superseded_by,
        }
```

#### 3.2.3 增强的 EventLog 结构

```python
@dataclass
class EnhancedEventLog:
    """
    增强版事件日志 - 支持精确时间表示

    对比原版:
    - 原版 EventLog.atomic_fact: List[str]
    - 新版 EnhancedEventLog.timed_facts: List[TimedFact]
    """

    # === 时间字段 (保持兼容) ===
    time: str
    """事件发生时间 (人类可读格式，保持向后兼容)"""

    timestamp: datetime
    """精确时间戳 (新增)"""

    # === 内容字段 ===
    timed_facts: List[TimedFact]
    """带时间标注的原子事实列表 (替代 atomic_fact)"""

    # === 向后兼容 ===
    @property
    def atomic_fact(self) -> List[str]:
        """向后兼容：返回纯文本原子事实列表"""
        return [tf.content for tf in self.timed_facts]

    @property
    def fact_embeddings(self) -> List[List[float]]:
        """向后兼容：返回 embedding 列表"""
        return [tf.embedding for tf in self.timed_facts if tf.embedding]

    # === 时间查询方法 ===
    def get_facts_at_time(self, query_time: datetime) -> List[TimedFact]:
        """获取在指定时间点有效的所有事实"""
        return [
            tf for tf in self.timed_facts
            if tf.is_current and
               (tf.valid_from is None or tf.valid_from <= query_time) and
               (tf.valid_until is None or tf.valid_until > query_time)
        ]

    def get_facts_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[TimedFact]:
        """获取在指定时间范围内发生的事实"""
        return [
            tf for tf in self.timed_facts
            if tf.event_time and tf.event_time.resolved_timestamp and
               start_time <= tf.event_time.resolved_timestamp <= end_time
        ]
```

### 3.3 Layer 2: 时间索引存储层

#### 3.3.1 时间索引结构：TemporalFactIndex

```python
@dataclass
class TemporalFactIndex:
    """
    时间事实索引 - 支持高效的时间查询

    类似 GroupEventClusterIndex，但按时间维度组织事实。
    """

    # === 事实存储 ===
    facts: Dict[str, TimedFact] = field(default_factory=dict)
    """fact_id -> TimedFact 映射"""

    # === 时间索引 ===
    timeline: Dict[str, List[str]] = field(default_factory=dict)
    """
    日期 -> [fact_id, ...] 映射
    格式: "2023-03-16" -> ["fact_001", "fact_002", ...]
    用于快速查找某一天的所有事实
    """

    # === 实体-时间索引 ===
    entity_timeline: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    """
    实体 -> 日期 -> [fact_id, ...] 映射
    格式: "Gina" -> {"2023-03-16": ["fact_001"], "2023-07-21": ["fact_002"]}
    用于查询特定人物在特定时间的事实
    """

    # === 事件类型索引 ===
    event_type_index: Dict[str, List[str]] = field(default_factory=dict)
    """
    事件关键词 -> [fact_id, ...] 映射
    格式: "open store" -> ["fact_001", "fact_003"]
    用于查询特定类型事件的时间
    """

    # === 查询方法 ===
    def get_facts_by_date(self, date_str: str) -> List[TimedFact]:
        """获取指定日期的所有事实"""
        fact_ids = self.timeline.get(date_str, [])
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]

    def get_facts_by_entity_and_date(
        self,
        entity: str,
        date_str: str
    ) -> List[TimedFact]:
        """获取指定实体在指定日期的事实"""
        entity_dates = self.entity_timeline.get(entity, {})
        fact_ids = entity_dates.get(date_str, [])
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]

    def get_entity_timeline(self, entity: str) -> List[Tuple[str, TimedFact]]:
        """获取指定实体的完整时间线，按时间排序"""
        entity_dates = self.entity_timeline.get(entity, {})
        result = []
        for date_str, fact_ids in sorted(entity_dates.items()):
            for fid in fact_ids:
                if fid in self.facts:
                    result.append((date_str, self.facts[fid]))
        return result

    def query_when(
        self,
        entity: str,
        event_keywords: List[str]
    ) -> Optional[TimedFact]:
        """
        回答 "When did X do Y?" 类型问题

        示例: query_when("Gina", ["open", "store"])
        返回: TimedFact(content="Gina opened her store", event_time=2023-03-16)
        """
        # 1. 获取实体的所有事实
        entity_dates = self.entity_timeline.get(entity, {})

        # 2. 在所有事实中搜索包含关键词的
        for date_str, fact_ids in entity_dates.items():
            for fid in fact_ids:
                fact = self.facts.get(fid)
                if fact and all(kw.lower() in fact.content.lower() for kw in event_keywords):
                    return fact

        return None
```

### 3.4 Layer 3: 时间感知查询层

#### 3.4.1 时间推理引擎

```python
class TemporalReasoningEngine:
    """
    时间推理引擎 - 回答时间相关问题
    """

    def __init__(
        self,
        fact_index: TemporalFactIndex,
        llm_provider: LLMProvider
    ):
        self.fact_index = fact_index
        self.llm_provider = llm_provider

    async def answer_temporal_question(
        self,
        question: str,
        context_time: datetime
    ) -> TemporalAnswer:
        """
        回答时间相关问题

        支持的问题类型:
        1. "When did X...?" - 查询事件时间
        2. "What happened on X date?" - 查询特定日期事件
        3. "How long has X been...?" - 计算持续时间
        4. "What was X doing before/after Y?" - 时间顺序推理
        """

        # 1. 分析问题类型
        question_type = self._classify_question(question)

        # 2. 提取关键实体和时间表达
        entities, temporal_refs = self._extract_question_elements(question)

        # 3. 根据问题类型执行查询
        if question_type == "when":
            return await self._answer_when_question(entities, question)
        elif question_type == "what_on_date":
            return await self._answer_date_query(temporal_refs, entities)
        elif question_type == "duration":
            return await self._answer_duration_query(entities, context_time)
        else:
            return await self._answer_with_llm(question, entities, context_time)

    async def _answer_when_question(
        self,
        entities: List[str],
        question: str
    ) -> TemporalAnswer:
        """回答 "When did..." 类型问题"""

        # 提取事件关键词
        event_keywords = self._extract_event_keywords(question)

        for entity in entities:
            fact = self.fact_index.query_when(entity, event_keywords)
            if fact and fact.event_time:
                return TemporalAnswer(
                    answer=fact.event_time.resolved_date_str,
                    confidence=fact.event_time.confidence,
                    evidence=[fact],
                    reasoning=f"Found matching fact: {fact.content}"
                )

        return TemporalAnswer(
            answer=None,
            confidence=0.0,
            evidence=[],
            reasoning="No matching temporal fact found"
        )


@dataclass
class TemporalAnswer:
    """时间查询答案"""
    answer: Optional[str]
    confidence: float
    evidence: List[TimedFact]
    reasoning: str
```

### 3.5 提取 Prompt 增强

#### 3.5.1 增强版 Event Log Prompt

```python
ENHANCED_EVENT_LOG_PROMPT = """You are an expert temporal information extraction analyst.
Your task is to extract atomic facts WITH PRECISE TEMPORAL INFORMATION from the given episode.

---

### INPUT
- EPISODE_TEXT: The memory text
- REFERENCE_TIME: "{{TIME}}" (the episode timestamp for resolving relative times)

---

### OUTPUT FORMAT
Return a JSON object with enhanced temporal facts:

{
  "event_log": {
    "time": "<REFERENCE_TIME>",
    "timed_facts": [
      {
        "fact_id": "f001",
        "content": "<Atomic fact sentence>",
        "event_time": {
          "original_text": "<Original time expression from text, or null>",
          "temporal_type": "absolute|relative|duration|vague",
          "resolved_date": "<YYYY-MM-DD or null if cannot resolve>",
          "confidence": 0.0-1.0,
          "granularity": "year|month|week|day|hour"
        },
        "fact_type": "event|state|plan|preference",
        "valid_from": "<YYYY-MM-DD or null>",
        "valid_until": "<YYYY-MM-DD or null, null means still valid>",
        "entities": ["<entity1>", "<entity2>"]
      }
    ]
  }
}

---

### TEMPORAL EXTRACTION RULES

#### 1. Time Expression Recognition
Identify ALL time expressions in the text:
- Absolute: "March 16, 2023", "2023年3月", "last Monday"
- Relative: "yesterday", "two weeks ago", "last month"
- Duration: "for 3 years", "since 2020"
- Vague: "recently", "a long time ago", "soon"

#### 2. Relative Time Resolution
For relative expressions, ALWAYS resolve to absolute dates using REFERENCE_TIME:
- "yesterday" + "March 10, 2024" → "March 9, 2024"
- "two weeks ago" + "March 10, 2024" → "February 25, 2024"
- "last month" + "March 10, 2024" → "February 2024" (month granularity)

#### 3. Validity Intervals
For stateful facts, determine validity periods:
- "Gina opened her store on March 16" → valid_from: "2023-03-16", valid_until: null
- "John worked at ABC from 2020 to 2023" → valid_from: "2020-01-01", valid_until: "2023-12-31"
- "Caroline plans to adopt a cat" → valid_from: null (future), valid_until: null

#### 4. Fact Type Classification
- event: One-time occurrence with specific time
- state: Ongoing condition or status
- plan: Future intention or plan
- preference: Personal preference (usually no time bound)

#### 5. Confidence Scoring
- 1.0: Explicit absolute date in text
- 0.8-0.9: Relative time successfully resolved
- 0.5-0.7: Approximate resolution (e.g., "last month" → month only)
- 0.3-0.5: Vague time, best effort estimate
- 0.0: Cannot determine time

---

### CRITICAL REQUIREMENTS
1. NEVER confuse different events' times (e.g., ad campaign vs store opening)
2. ALWAYS include the original time expression when present
3. For ambiguous cases, set confidence < 1.0 and explain in reasoning
4. Extract ALL temporal information, even if implicit

---

### EXAMPLE

**Input:**
REFERENCE_TIME = "March 10, 2024(Sunday) at 2:00 PM"
EPISODE_TEXT = "Gina said she opened her online clothing store on March 16 last year. She also mentioned launching an ad campaign yesterday."

**Output:**
{
  "event_log": {
    "time": "March 10, 2024(Sunday) at 2:00 PM",
    "timed_facts": [
      {
        "fact_id": "f001",
        "content": "Gina opened her online clothing store on March 16, 2023.",
        "event_time": {
          "original_text": "March 16 last year",
          "temporal_type": "relative",
          "resolved_date": "2023-03-16",
          "confidence": 0.95,
          "granularity": "day"
        },
        "fact_type": "event",
        "valid_from": "2023-03-16",
        "valid_until": null,
        "entities": ["Gina", "online clothing store"]
      },
      {
        "fact_id": "f002",
        "content": "Gina launched an advertising campaign on March 9, 2024.",
        "event_time": {
          "original_text": "yesterday",
          "temporal_type": "relative",
          "resolved_date": "2024-03-09",
          "confidence": 1.0,
          "granularity": "day"
        },
        "fact_type": "event",
        "valid_from": "2024-03-09",
        "valid_until": null,
        "entities": ["Gina", "ad campaign"]
      }
    ]
  }
}

---

Now analyze the provided EPISODE_TEXT and extract ALL temporal facts.

### INPUT
- EPISODE_TEXT: "{{EPISODE_TEXT}}"
- REFERENCE_TIME: "{{TIME}}"
"""
```

---

## 4. 实施方案对比

### 4.1 方案 A: 完整双时态模型 (类 Zep)

**复杂度**: 高
**改动范围**: 大
**预期效果**: 最佳

| 优点 | 缺点 |
|------|------|
| 完整的时间推理能力 | 需要重构存储层 |
| 支持历史查询 | 实现复杂度高 |
| 支持时间冲突解决 | 迁移成本大 |

### 4.2 方案 B: TimedFact 增强方案 (推荐)

**复杂度**: 中
**改动范围**: 中
**预期效果**: 良好

| 优点 | 缺点 |
|------|------|
| 向后兼容 | 不支持完整历史查询 |
| 渐进式改进 | 时间冲突解决较弱 |
| 实现成本可控 | |

### 4.3 方案 C: Prompt 优化方案

**复杂度**: 低
**改动范围**: 小
**预期效果**: 中等

| 优点 | 缺点 |
|------|------|
| 改动最小 | 效果有限 |
| 快速验证 | 不解决根本问题 |

---

## 5. 推荐实施路径

### Phase 1: 快速验证 (1-2周)
1. 优化 Event Log Prompt，增强时间提取要求
2. 在 atomic_fact 中强制包含解析后的绝对日期
3. 评估效果

### Phase 2: 结构化改进 (2-4周)
1. 实现 `TemporalExpression` 和 `TimedFact` 数据结构
2. 修改 `EventLogExtractor` 使用新 Prompt
3. 实现 `TemporalFactIndex` 时间索引
4. 集成到检索流程

### Phase 3: 完整方案 (4-6周)
1. 实现 `TemporalReasoningEngine`
2. 添加时间冲突检测与解决
3. 实现有效性区间查询
4. 性能优化

---

## 6. 预期收益

| 指标 | 当前 | Phase 1 | Phase 2 | Phase 3 |
|------|------|---------|---------|---------|
| 时间类问题准确率 | 92.3% | 94-95% | 96-97% | 98%+ |
| 时间日期错误占比 | 18.6% | 12-14% | 6-8% | <5% |

---

## 参考资料

- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- [Graphiti GitHub Repository](https://github.com/getzep/graphiti)
- [Parallax 评估错误分析报告](../evaluation/error_analysis_report.md)
