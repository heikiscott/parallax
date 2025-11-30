"""
记忆单元模块 (MemUnit - Memory Unit)

定义 MemUnit，即从对话中提取的基本记忆单元。
MemUnit 是对话边界检测的输出，代表一段语义完整的对话内容。

处理流程:
========

    原始消息 --> 边界检测 --> MemUnit --> 记忆提取 --> Memory
                               │
                               ├── unit_id: 唯一标识
                               ├── original_data: 原始消息列表
                               ├── summary: 内容摘要
                               ├── episode: 情景描述
                               ├── semantic_memories: 语义关联
                               └── event_log: 事件日志

MemUnit 是原始输入数据和最终提取记忆之间的中间表示。
它封装了一组语义相关的消息，作为后续记忆提取的输入单元。

核心概念:
========
- 边界检测: 将连续的对话流切分为语义完整的片段
- 话题转换: 当对话主题发生变化时，生成新的 MemUnit
- 多用户: 一个 MemUnit 可能包含多个用户的消息

架构设计说明 - MemUnit 与 Memory 的关系:
========================================

**MemUnit 不是包含 Memory，而是"分解"为多种 Memory 类型**

**重要：MemUnit 采用嵌入式存储，不是引用式存储**
- `MemUnit.episode` 是**完整的内容字符串**，不是 EpisodeMemory 的 ID 引用
- `MemUnit.semantic_memories` 是**完整的对象列表**，不是 SemanticMemory 的 ID 列表
- `MemUnit.event_log` 是**完整的对象**，不是 EventLog 的 ID 引用

1. MemUnit 的角色:
   - **临时中间对象**: 在提取流程中承载数据
   - **存储到 MongoDB**: MemUnit 作为完整文档持久化到 `memunits` 集合
     * episode 字段存储完整的情景描述文本
     * semantic_memories 字段存储完整的语义记忆对象列表
     * event_log 字段存储完整的事件日志对象
   - **同步到检索引擎**: MemUnit.episode **复制**到 ES/Milvus 作为 EpisodeMemory

2. Memory 的生成（数据复制，非引用）:
   从 MemUnit 中**复制提取**出不同类型的 Memory 对象，各自独立存储：
   
   a) **EpisodeMemory (群组视角情景记忆)**
      - 来源: 复制 MemUnit.episode 的内容（字符串）
      - 存储: MongoDB episodic_memories 集合 + ES + Milvus
      - 关系: EpisodeMemory.episode = MemUnit.episode（内容相同，存储独立）
      - 用途: 向量检索、全文检索、System Prompt 上下文
   
   b) **个人 EpisodeMemory (个人视角情景记忆)**
      - 来源: 从 MemUnit.episode 进一步提取（LLM 改写为个人视角）
      - 存储: MongoDB episodic_memories 集合 + ES + Milvus
      - 用途: 个人化的记忆检索
   
   c) **SemanticMemory (语义记忆)**
      - 来源: 复制 MemUnit.semantic_memories 列表中的对象
      - 存储: MongoDB semantic_memories 集合 + ES + Milvus
      - 关系: 每个 SemanticMemory 文档对应 MemUnit.semantic_memories 中的一个对象
      - 用途: 回答具体问题、构建知识图谱
   
   d) **EventLog (事件日志)**
      - 来源: 复制 MemUnit.event_log 对象
      - 存储: MongoDB event_logs 集合 + ES + Milvus
      - 关系: EventLog 文档对应 MemUnit.event_log 对象
      - 用途: 时间线构建、因果关系追踪

3. 为什么采用嵌入式存储 + 复制分发的设计?
   - **MemUnit 的完整性**: MemUnit 作为历史快照，保留提取时的完整数据
   - **解耦提取与检索**: MemUnit 专注于边界检测，Memory 专注于检索优化
   - **多视角记忆**: 同一 episode 内容可以生成群组和个人视角的不同检索记录
   - **灵活检索**: 不同类型的 Memory 有独立的索引结构和检索策略
   - **可溯源**: 通过 MemUnit 可以追溯记忆的原始提取结果
   - **数据冗余**: 牺牲存储空间换取查询性能和数据独立性

使用示例:
========
    from memory.schema import MemUnit, SourceType

    memunit = MemUnit(
        unit_id="mu_123",
        user_id_list=["user_1", "user_2"],
        original_data=[
            {"speaker_id": "user_1", "content": "你好!", "timestamp": "..."},
            {"speaker_id": "user_2", "content": "嗨!", "timestamp": "..."},
        ],
        timestamp=datetime.now(),
        summary="两位用户互相打招呼",
        type=SourceType.CONVERSATION,
        episode="用户1向用户2问好，用户2热情回应..."
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import datetime

from utils.datetime_utils import to_iso_format
from .source_type import SourceType

if TYPE_CHECKING:
    from .semantic_memory import SemanticMemoryItem


@dataclass
class MemUnit:
    """
    记忆单元 (Memory Unit) - 对话内容提取的原子单位

    MemUnit 封装了通过边界检测识别出的一段语义完整的对话内容，
    作为下游记忆提取（情景记忆、语义记忆、用户画像等）的输入。

    字段分组说明:
    =============

    1. 标识字段 (Identity):
        - unit_id: 唯一标识符，用于追踪和关联

    2. 用户字段 (Users):
        - user_id_list: 涉及的所有用户ID
        - participants: 实际发言的参与者 (user_id_list 的子集)

    3. 原始数据 (Raw Data):
        - original_data: 原始消息列表，每条消息包含:
            - speaker_id: 发言者ID
            - speaker_name: 发言者名称
            - content: 消息内容
            - timestamp: 发送时间

    4. 时间字段 (Timing):
        - timestamp: 该单元的时间戳 (通常是最后一条消息的时间)

    5. 上下文字段 (Context):
        - group_id: 群组ID (私聊场景为 None)
        - type: 数据源类型 (通常为 CONVERSATION)

    6. 内容字段 (Content):
        - summary: 简短摘要 (1-2句话)
        - subject: 话题/主题
        - keywords: 关键词列表
        - linked_entities: 关联实体 (项目名、产品名等)
        - episode: 详细的情景描述

    7. 提取结果字段 (Extracted):
        - semantic_memories: 语义记忆关联列表
        - event_log: 事件日志 (带时间戳的原子事实)

    8. 扩展字段 (Extension):
        - extend: 自定义元数据

    验证规则:
    ========
    - unit_id: 必填
    - original_data: 必填，不能为空
    - summary: 必填，不能为空
    """

    # ===== 1. 标识字段 =====
    unit_id: str
    """
    唯一标识符 (UUID格式)
    - 产生方式: 在 MemUnit 创建时自动生成 (uuid.uuid4())
    - 使用方式: 
        1. 作为 MongoDB/ES/Milvus 中的主键或关联键
        2. 用于追踪记忆的处理流程
    """

    # ===== 2. 用户字段 =====
    user_id_list: List[str]
    """
    涉及的所有用户ID
    - 产生方式: 从输入请求中直接获取
    - 使用方式: 
        1. 检索时用于权限过滤 (filter by user_id)
        2. 构造 System Prompt 时作为上下文信息
    """

    participants: Optional[List[str]] = None
    """
    实际发言的参与者 (user_id_list 的子集)
    - 产生方式: 从 original_data 中解析提取 (speaker_id)
    - 使用方式: 
        1. 辅助判断对话的参与情况
        2. 在 System Prompt 中展示对话参与者
    """

    # ===== 3. 原始数据 =====
    original_data: List[Dict[str, Any]] = None
    """
    原始消息列表
    - 产生方式: 用户的原始输入 (ChatRawData)
    - 使用方式: 
        1. 作为 LLM 提取 summary/episode 的输入源
        2. 存档，用于追溯原始对话
        3. 一般不直接用于检索或 System Prompt (除非需要展示原始引用)
    """

    # ===== 4. 时间字段 =====
    timestamp: datetime.datetime = None
    """
    单元时间戳
    - 产生方式: 通常取最后一条消息的时间，或当前时间
    - 使用方式: 
        1. 检索时用于时间范围过滤 (start_time, end_time)
        2. 在 System Prompt 中展示记忆发生的时间
    """

    # ===== 5. 上下文字段 =====
    group_id: Optional[str] = None
    """
    群组ID
    - 产生方式: 从输入请求中获取 (私聊为 None)
    - 使用方式: 
        1. 检索时用于群组隔离和过滤
        2. 区分个人记忆和群组记忆
    """

    type: Optional[SourceType] = None
    """
    数据源类型
    - 产生方式: 根据输入源确定 (如 CONVERSATION)
    - 使用方式: 
        1. 决定后续的处理策略 (不同类型可能走不同的提取管道)
    """

    # ===== 6. 内容字段 (核心检索与Prompt字段) =====
    summary: str = None
    """
    简短摘要 (1-2句话)
    - 产生方式: LLM 提取 (Boundary Detection 阶段或后续提取)
    - 使用方式: 
        1. 存入 ES/Milvus 的 search_content 字段，用于全文检索
        2. 在 UI 上展示记忆预览
    """

    subject: Optional[str] = None
    """
    话题/主题
    - 产生方式: LLM 提取
    - 使用方式: 
        1. 存入 ES/Milvus 的 search_content/subject 字段，用于检索
        2. 帮助快速了解记忆主题
    """

    keywords: Optional[List[str]] = None
    """
    关键词列表
    - 产生方式: LLM 提取
    - 使用方式: 
        1. 存入 ES 的 keywords 字段，用于精确匹配或增强检索
    """

    linked_entities: Optional[List[str]] = None
    """
    关联实体
    - 产生方式: LLM 提取 (NER)
    - 使用方式: 
        1. 存入 ES 的 linked_entities 字段，用于实体关联检索
    """

    episode: Optional[str] = None
    """
    详细情景描述 (核心字段)
    
    **为什么是 str 而不是 EpisodeMemory 类？**
    
    设计哲学：MemUnit 只存储"提取的内容"，而 EpisodeMemory 是"检索优化的文档结构"
    
    1. **层次分离**:
       - `MemUnit.episode`: 纯内容层 - 仅存储 LLM 提取的情景描述文本
       - `EpisodeMemory`: 文档层 - 包含 ID、用户、时间、索引、embedding 等检索所需的完整结构
    
    2. **避免循环依赖**:
       - 如果 `episode` 是 `EpisodeMemory` 对象，会引入复杂的对象嵌套
       - `EpisodeMemory` 本身也有 `episode: str` 字段，会造成结构混乱
    
    3. **关系类比**:
       ```
       类比数据库设计:
       MemUnit.episode = "文本内容"       (相当于原始数据)
       EpisodeMemory    = 完整的表记录    (ID + 内容 + 索引 + embedding)
       
       就像你不会在 User 表里嵌套完整的 Post 对象，只存 post_content 字符串
       然后在 Posts 表里存完整的 {id, user_id, content, created_at, ...}
       ```
    
    4. **数据流转**:
       ```python
       # Step 1: MemUnit 提取阶段
       memunit = MemUnit(
           episode="用户A和用户B讨论了项目进展..."  # 纯文本
       )
       
       # Step 2: 保存到 MongoDB
       await memunit_repo.save(memunit)  # episode 作为字符串字段存储
       
       # Step 3: 同步到检索引擎 (memunit_sync.py)
       episode_memory = EpisodeMemory(
           episode_id=memunit.unit_id,
           episode=memunit.episode,  # 复制文本内容
           user_id=memunit.user_id,
           timestamp=memunit.timestamp,
           embedding=generate_embedding(memunit.episode),  # 生成向量
           # ... 其他检索优化字段
       )
       await es_repo.save(episode_memory)  # 存储到 ES/Milvus
       ```
    
    - 产生方式: LLM 提取 (EpisodeMemoryExtractor)，将对话转化为第三人称叙事
    - 使用方式: 
        1. **向量检索源**: 用于生成 Embedding (存入 extend['embedding'])
        2. **全文检索源**: 复制到 ES/Milvus 的 EpisodeMemory.episode 字段
        3. **System Prompt**: 检索命中后，将此字段内容放入 Prompt 作为长期记忆上下文
    """

    # ===== 7. 提取结果字段 =====
    semantic_memories: Optional[List['SemanticMemoryItem']] = None
    """
    语义记忆关联列表
    
    **为什么是 List[SemanticMemoryItem] 而不是 List[SemanticMemory]？**
    
    - `SemanticMemoryItem`: 轻量级数据类，仅包含提取的语义内容（content、类型、置信度等）
    - `SemanticMemory`: 完整的 MongoDB 文档类，包含 ID、用户、时间戳、embedding、索引等
    
    设计原因：
    1. MemUnit 只需要存储"提取的语义内容"，不需要完整的文档结构
    2. 避免在 MemUnit 中嵌套复杂的文档对象
    3. SemanticMemory 文档会从这些 Item 中生成，并独立存储到 semantic_memories 集合
    
    - 产生方式: LLM 提取，从对话中抽象出的事实或偏好
    - 使用方式: 
        1. 复制每个 Item 的内容，生成独立的 SemanticMemory 文档
        2. 存储到 MongoDB semantic_memories 集合 + ES + Milvus
        3. 用于回答具体问题、构建知识图谱
    """

    event_log: Optional[Any] = None
    """
    事件日志对象
    
    **为什么是 Any 而不是强类型的 EventLog？**
    
    - 这里存储的是 LLM 提取的原始事件日志数据（包含 atomic_fact 列表等）
    - EventLog 是完整的 MongoDB 文档类，会从这个数据中生成
    
    设计原因：
    1. MemUnit 只需要存储"提取的事件数据"（字典或数据类）
    2. 避免类型循环依赖和结构嵌套
    3. EventLog 文档会从这个对象中生成，并独立存储到 event_logs 集合
    
    - 产生方式: LLM 提取，关键事件记录（atomic_fact 列表 + embedding）
    - 使用方式: 
        1. 复制此对象的内容，生成独立的 EventLog 文档
        2. 存储到 MongoDB event_logs 集合 + ES + Milvus
        3. 用于构建时间线、因果关系追踪
    """

    # ===== 8. 扩展字段 =====
    extend: Optional[Dict[str, Any]] = None
    """
    自定义元数据
    - 产生方式: 程序处理过程中添加
    - 使用方式: 
        1. **embedding**: 存储 episode 的向量表示，用于 Milvus 检索
        2. vector_model: 记录使用的向量模型版本
    """

    def __post_init__(self):
        """初始化后验证必填字段"""
        if not self.unit_id:
            raise ValueError("unit_id 是必填字段")
        if not self.original_data:
            raise ValueError("original_data 是必填字段")
        if not self.summary:
            raise ValueError("summary 是必填字段")

    def __repr__(self) -> str:
        """返回简洁的字符串表示"""
        summary_preview = self.summary[:50] if self.summary else ""
        return (
            f"MemUnit(unit_id={self.unit_id}, "
            f"messages={len(self.original_data)}, "
            f"timestamp={self.timestamp}, "
            f"summary={summary_preview}...)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，用于序列化

        返回:
            适合 JSON 序列化或数据库存储的字典
        """
        return {
            # 标识字段
            "unit_id": self.unit_id,
            # 用户字段
            "user_id_list": self.user_id_list,
            "participants": self.participants,
            # 原始数据
            "original_data": self.original_data,
            # 时间字段
            "timestamp": to_iso_format(self.timestamp),
            # 上下文字段
            "group_id": self.group_id,
            "type": str(self.type.value) if self.type else None,
            # 内容字段
            "summary": self.summary,
            "subject": self.subject,
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            "episode": self.episode,
            # 提取结果字段
            "semantic_memories": (
                [item.to_dict() for item in self.semantic_memories]
                if self.semantic_memories
                else None
            ),
            "event_log": (
                self.event_log.to_dict() if hasattr(self.event_log, 'to_dict')
                else self.event_log
            ) if self.event_log else None,
            # 扩展字段
            "extend": self.extend,
        }
