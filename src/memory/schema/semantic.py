"""Semantic memory data structures."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import datetime

from utils.datetime_utils import to_iso_format


@dataclass
class SemanticMemory:
    """
    语义记忆数据模型

    用于存储从情景记忆中提取的语义知识
    """

    user_id: str
    content: str
    knowledge_type: str = "knowledge"
    source_episodes: List[str] = None
    created_at: datetime.datetime = None
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.source_episodes is None:
            self.source_episodes = []
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "source_episodes": self.source_episodes,
            "created_at": to_iso_format(self.created_at),
            "group_id": self.group_id,
            "participants": self.participants,
            "metadata": self.metadata,
        }


@dataclass
class SemanticMemoryItem:
    """
    语义记忆联想项目

    包含时间信息的语义记忆联想预测
    """

    content: str
    evidence: Optional[str] = None  # 原始证据，支持该联想预测的具体事实（不超过30字）
    start_time: Optional[str] = None  # 事件开始时间，格式：YYYY-MM-DD
    end_time: Optional[str] = None  # 事件结束时间，格式：YYYY-MM-DD
    duration_days: Optional[int] = None  # 持续时间（天数）
    source_episode_id: Optional[str] = None  # 来源事件ID
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "evidence": self.evidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_days": self.duration_days,
            "source_episode_id": self.source_episode_id,
            "embedding": self.embedding,
        }
