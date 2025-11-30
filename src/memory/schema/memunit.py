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
    unit_id: str  # 唯一标识符 (UUID格式)

    # ===== 2. 用户字段 =====
    user_id_list: List[str]  # 涉及的所有用户ID
    participants: Optional[List[str]] = None  # 实际发言的参与者

    # ===== 3. 原始数据 =====
    original_data: List[Dict[str, Any]] = None  # 原始消息列表

    # ===== 4. 时间字段 =====
    timestamp: datetime.datetime = None  # 单元时间戳

    # ===== 5. 上下文字段 =====
    group_id: Optional[str] = None  # 群组ID
    type: Optional[SourceType] = None  # 数据源类型

    # ===== 6. 内容字段 =====
    summary: str = None  # 简短摘要 (1-2句话)
    subject: Optional[str] = None  # 话题/主题
    keywords: Optional[List[str]] = None  # 关键词列表
    linked_entities: Optional[List[str]] = None  # 关联实体
    episode: Optional[str] = None  # 详细情景描述

    # ===== 7. 提取结果字段 =====
    semantic_memories: Optional[List['SemanticMemoryItem']] = None  # 语义记忆关联
    event_log: Optional[Any] = None  # 事件日志

    # ===== 8. 扩展字段 =====
    extend: Optional[Dict[str, Any]] = None  # 自定义元数据

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
