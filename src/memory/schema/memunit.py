"""MemUnit - Memory Unit data structure."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import datetime

from utils.datetime_utils import to_iso_format
from .raw_data_type import RawDataType

if TYPE_CHECKING:
    from .semantic import SemanticMemoryItem


@dataclass
class MemUnit:
    """
    Boundary detection result following the specified schema.

    This class represents the result of boundary detection analysis
    and contains all the required fields for memory storage.
    """

    event_id: str
    user_id_list: List[str]
    # For downstream consumers we store normalized dicts extracted from RawData
    original_data: List[Dict[str, Any]]
    timestamp: datetime.datetime
    summary: str

    # Optional fields
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    keywords: Optional[List[str]] = None
    subject: Optional[str] = None
    linked_entities: Optional[List[str]] = None
    episode: Optional[str] = None  # 情景记忆内容

    # 语义记忆联想预测字段
    semantic_memories: Optional[List['SemanticMemoryItem']] = None  # 语义记忆联想列表
    # Event Log 字段
    event_log: Optional[Any] = None  # Event Log 对象
    # extend fields, can be used to store any additional information
    extend: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate the result after initialization."""
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.original_data:
            raise ValueError("original_data is required")
        if not self.summary:
            raise ValueError("summary is required")

    def __repr__(self) -> str:
        return f"MemUnit(event_id={self.event_id}, original_data={self.original_data}, timestamp={self.timestamp}, summary={self.summary})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),  # 转换为ISO格式字符串
            "summary": self.summary,
            "group_id": self.group_id,
            "participants": self.participants,
            "type": str(self.type.value) if self.type else None,
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            "subject": self.subject,
            "episode": self.episode,
            "semantic_memories": (
                [item.to_dict() for item in self.semantic_memories]
                if self.semantic_memories
                else None
            ),
            "event_log": (
                self.event_log.to_dict() if hasattr(self.event_log, 'to_dict')
                else self.event_log
            ) if self.event_log else None,
            "extend": self.extend,
        }
