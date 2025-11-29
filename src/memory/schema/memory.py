"""Memory - Base class for extracted memories."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import datetime

from utils.datetime_utils import to_iso_format
from .memory_type import MemoryType
from .source_type import SourceType

if TYPE_CHECKING:
    from .semantic import SemanticMemoryItem


@dataclass
class Memory:
    """
    Simple result class for memory extraction.

    Contains the essential information for extracted memories.
    """

    memory_type: MemoryType
    user_id: str
    timestamp: datetime.datetime
    ori_event_id_list: List[str]

    subject: Optional[str] = None
    summary: Optional[str] = None
    episode: Optional[str] = None

    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    type: Optional[SourceType] = None
    keywords: Optional[List[str]] = None
    linked_entities: Optional[List[str]] = None

    memunit_event_id_list: Optional[List[str]] = None
    # 语义记忆联想预测字段
    semantic_memories: Optional[List['SemanticMemoryItem']] = None  # 语义记忆联想列表
    extend: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        # 安全处理 timestamp（可能是 datetime、str 或 None）
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
            "memory_type": self.memory_type.value if self.memory_type else None,
            "user_id": self.user_id,
            "timestamp": timestamp_str,
            "ori_event_id_list": self.ori_event_id_list,
            "subject": self.subject,
            "summary": self.summary,
            "episode": self.episode,
            "group_id": self.group_id,
            "participants": self.participants,
            "type": self.type.value if self.type else None,
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            "semantic_memories": (
                [item.to_dict() for item in self.semantic_memories]
                if self.semantic_memories
                else None
            ),
            "extend": self.extend,
        }
