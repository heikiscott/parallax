"""
记忆基类模块 (Memory Base Class)

定义所有具体记忆类型继承的基类 Memory。
提供通用的结构和序列化方法。

继承层次结构:
=============

    Memory (基类)
        │
        ├── EpisodeMemory (情景记忆)
        │       个人叙事性记忆，从特定用户视角描述
        │
        ├── ProfileMemory (用户画像)
        │       用户特征档案，包含技能、性格、偏好等
        │
        └── GroupProfileMemory (群体画像)
                群体特征，包含话题、角色、互动模式

设计原则:
========
- Memory 通常不直接实例化，而是使用具体的子类
- 子类在 __post_init__ 中自动设置 memory_type
- 所有记忆都可追溯到源 MemUnit (通过 memunit_id_list)

使用示例:
========
    # Memory 通常不直接实例化，使用子类:

    from memory.schema import EpisodeMemory, ProfileMemory

    episode = EpisodeMemory(
        user_id="user_123",
        timestamp=datetime.now(),
        memunit_id_list=["memunit_1"],
        narrative="用户讨论了项目需求..."
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import datetime

from utils.datetime_utils import to_iso_format
from .memory_type import MemoryType
from .source_type import SourceType

if TYPE_CHECKING:
    from .semantic_memory import SemanticMemoryItem


@dataclass
class Memory:
    """
    记忆基类 - 所有提取记忆类型的父类

    此数据类提供所有记忆类型共享的通用结构。
    子类 (EpisodeMemory, ProfileMemory 等) 在此基础上扩展特定字段。

    字段分组说明:
    =============

    1. 类型字段 (Type):
        - memory_type: 记忆类型 (EPISODE_SUMMARY, PROFILE 等)
          由子类在 __post_init__ 中自动设置

    2. 归属字段 (Ownership):
        - user_id: 记忆所属用户ID
          对于个人记忆，这是记忆的拥有者
          对于群体记忆，可能是代表用户或群管理员

    3. 时间字段 (Timing):
        - timestamp: 记忆创建时间或事件发生时间
          存储为 UTC，序列化时转换为 ISO 格式

    4. 溯源字段 (Provenance):
        - memunit_id_list: 用于创建此记忆的 MemUnit ID 列表

    5. 内容字段 (Content):
        - subject: 简短的主题/标题
        - summary: 简洁摘要 (1-3句话)
        - narrative: 详细的叙事描述

    6. 上下文字段 (Context):
        - group_id: 群组ID (个人记忆为 None)
        - participants: 参与者用户ID列表
        - type: 源数据类型 (通常为 CONVERSATION)

    7. 索引字段 (Indexing):
        - keywords: 关键词，用于搜索和分类
        - linked_entities: 关联实体 (项目名、产品名等)

    8. 关联字段 (Relations):
        - semantic_memories: 关联的语义记忆预测

    9. 扩展字段 (Extension):
        - extend: 自定义元数据，用于非标准字段
    """

    # ===== 1. 类型字段 =====
    memory_type: MemoryType  # 记忆类型 (由子类自动设置)

    # ===== 2. 归属字段 =====
    user_id: str  # 记忆所属用户ID

    # ===== 3. 时间字段 =====
    timestamp: datetime.datetime  # 创建时间/事件时间

    # ===== 4. 溯源字段 =====
    memunit_id_list: List[str]  # 用于创建此记忆的 MemUnit ID 列表

    # ===== 5. 内容字段 =====
    subject: Optional[str] = None  # 主题/标题
    summary: Optional[str] = None  # 简洁摘要
    narrative: Optional[str] = None  # 详细叙事

    # ===== 6. 上下文字段 =====
    group_id: Optional[str] = None  # 群组ID
    participants: Optional[List[str]] = None  # 参与者列表
    type: Optional[SourceType] = None  # 源数据类型

    # ===== 7. 索引字段 =====
    keywords: Optional[List[str]] = None  # 关键词
    linked_entities: Optional[List[str]] = None  # 关联实体

    # ===== 8. 关联字段 =====
    semantic_memories: Optional[List['SemanticMemoryItem']] = None  # 语义记忆关联

    # ===== 9. 扩展字段 =====
    extend: Optional[Dict[str, Any]] = None  # 自定义元数据

    def __post_init__(self):
        """
        初始化后钩子，供子类重写

        子类应重写此方法以自动设置 memory_type。
        设置 memory_type 后应调用 super().__post_init__()。
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，用于序列化

        类型转换处理:
        - datetime -> ISO 格式字符串
        - MemoryType/SourceType -> 字符串值
        - SemanticMemoryItem -> 通过 to_dict() 转换

        返回:
            适合 JSON 序列化或数据库存储的字典
        """
        # 处理时间戳转换 (可能是 datetime、str 或 None)
        timestamp_str = None
        if self.timestamp:
            if isinstance(self.timestamp, str):
                timestamp_str = self.timestamp if self.timestamp else None
            else:
                try:
                    timestamp_str = to_iso_format(self.timestamp)
                except Exception:
                    timestamp_str = str(self.timestamp) if self.timestamp else None

        return {
            # 类型字段
            "memory_type": self.memory_type.value if self.memory_type else None,
            # 归属字段
            "user_id": self.user_id,
            # 时间字段
            "timestamp": timestamp_str,
            # 溯源字段
            "memunit_id_list": self.memunit_id_list,
            # 内容字段
            "subject": self.subject,
            "summary": self.summary,
            "narrative": self.narrative,
            # 上下文字段
            "group_id": self.group_id,
            "participants": self.participants,
            "type": self.type.value if self.type else None,
            # 索引字段
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            # 关联字段
            "semantic_memories": (
                [item.to_dict() for item in self.semantic_memories]
                if self.semantic_memories
                else None
            ),
            # 扩展字段
            "extend": self.extend,
        }
