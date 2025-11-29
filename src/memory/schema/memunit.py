"""
MemUnit - Memory Unit Data Structure.

This module defines MemUnit, the fundamental unit of extracted memory content.
A MemUnit represents a coherent segment of conversation or content that has been
identified as a meaningful boundary for memory extraction.

Processing Flow:
----------------
    Raw Messages --> Boundary Detection --> MemUnit --> Memory Extraction --> Memory
                                              |
                                              +-- Contains: original_data, episode,
                                                  semantic_memories, event_log

A MemUnit is the intermediate representation between raw input data and
final extracted memories (EpisodeMemory, SemanticMemory, etc.).

Usage:
------
    from memory.schema import MemUnit, SourceType

    memunit = MemUnit(
        event_id="evt_123",
        user_id_list=["user_1", "user_2"],
        original_data=[
            {"speaker_id": "user_1", "content": "Hello!", "timestamp": "..."},
            {"speaker_id": "user_2", "content": "Hi there!", "timestamp": "..."},
        ],
        timestamp=datetime.now(),
        summary="A greeting exchange between two users",
        type=SourceType.CONVERSATION,
        episode="User 1 greeted User 2, who responded warmly..."
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
    Memory Unit - The atomic unit of extracted conversation content.

    A MemUnit captures a coherent segment of conversation or content that
    has been identified through boundary detection. It serves as the input
    for downstream memory extraction (episode, semantic, profile memories).

    Attributes:
        event_id (str): Unique identifier for this memory unit.
            Format: UUID string. Used for tracing and linking memories.

        user_id_list (List[str]): List of all user IDs involved in this unit.
            May differ from participants if some users are mentioned but not speaking.

        original_data (List[Dict[str, Any]]): Raw message data that forms this unit.
            Each dict typically contains:
            - speaker_id: Who sent the message
            - speaker_name: Display name of the sender
            - content: Message text
            - timestamp: When the message was sent
            - Additional fields depending on source type

        timestamp (datetime): When this memory unit was created/detected.
            Usually the timestamp of the last message in the unit.

        summary (str): Brief summary of the conversation content.
            Generated during boundary detection, 1-2 sentences.

        group_id (Optional[str]): ID of the group/chat where this occurred.
            None for direct messages or non-group contexts.

        participants (Optional[List[str]]): Active participants (speakers) in this unit.
            Subset of user_id_list who actually sent messages.

        type (Optional[SourceType]): Source type of the original data.
            Usually SourceType.CONVERSATION for chat messages.

        keywords (Optional[List[str]]): Key terms extracted from the content.
            Used for search and categorization.

        subject (Optional[str]): Topic or subject of the conversation.
            Example: "API authentication implementation"

        linked_entities (Optional[List[str]]): External entities referenced.
            Project names, product names, URLs, etc.

        episode (Optional[str]): Detailed episodic narrative of events.
            Group-level episode summary generated during extraction.

        semantic_memories (Optional[List[SemanticMemoryItem]]): Predicted associations.
            Forward-looking semantic associations about impact on users.

        event_log (Optional[Any]): Atomic event log extracted from this unit.
            Contains timestamped facts for fine-grained recall.

        extend (Optional[Dict[str, Any]]): Extension field for custom metadata.
            Use for source-specific fields not in the standard schema.

    Raises:
        ValueError: If event_id, original_data, or summary is missing/empty.
    """

    # === Required Fields ===
    event_id: str
    user_id_list: List[str]
    original_data: List[Dict[str, Any]]
    timestamp: datetime.datetime
    summary: str

    # === Context Fields ===
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    type: Optional[SourceType] = None

    # === Content Fields ===
    keywords: Optional[List[str]] = None
    subject: Optional[str] = None
    linked_entities: Optional[List[str]] = None
    episode: Optional[str] = None

    # === Extracted Content Fields ===
    semantic_memories: Optional[List['SemanticMemoryItem']] = None
    event_log: Optional[Any] = None

    # === Extension Field ===
    extend: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.original_data:
            raise ValueError("original_data is required")
        if not self.summary:
            raise ValueError("summary is required")

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"MemUnit(event_id={self.event_id}, "
            f"messages={len(self.original_data)}, "
            f"timestamp={self.timestamp}, "
            f"summary={self.summary[:50]}...)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MemUnit to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON/database storage.
        """
        return {
            "event_id": self.event_id,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),
            "summary": self.summary,
            "group_id": self.group_id,
            "participants": self.participants,
            "type": str(self.type.value) if self.type else None,
            "keywords": self.keywords,
            "subject": self.subject,
            "linked_entities": self.linked_entities,
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
