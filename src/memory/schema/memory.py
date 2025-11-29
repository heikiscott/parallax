"""
Memory Base Class.

This module defines the base Memory class that all specific memory types inherit from.
It provides the common structure and serialization methods for extracted memories.

Inheritance Hierarchy:
----------------------
    Memory (base class)
        |
        +-- EpisodeMemory
        |       Personal narrative memories
        |
        +-- ProfileMemory
        |       User profile and characteristics
        |
        +-- GroupProfileMemory
                Group collective profile

Usage:
------
    # Memory is typically not instantiated directly.
    # Use specific subclasses instead:

    from memory.schema import EpisodeMemory, ProfileMemory

    episode = EpisodeMemory(
        user_id="user_123",
        timestamp=datetime.now(),
        ori_event_id_list=["event_1"],
        episode="User discussed project requirements..."
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
    Base class for all extracted memories.

    This dataclass provides the common structure shared by all memory types.
    Subclasses (EpisodeMemory, ProfileMemory, etc.) extend this with type-specific fields.

    Attributes:
        memory_type (MemoryType): Type of this memory (EPISODE_SUMMARY, PROFILE, etc.).
            Set automatically by subclasses in __post_init__.

        user_id (str): ID of the user this memory belongs to.
            For personal memories, this is the memory owner.
            For group memories, this may be a representative user or group admin.

        timestamp (datetime): When this memory was created or when the events occurred.
            Stored in UTC, converted to ISO format for serialization.

        ori_event_id_list (List[str]): Original event IDs that this memory is derived from.
            Links back to source MemUnits for traceability.

        subject (Optional[str]): Brief subject/title of the memory content.
            Example: "Project kickoff meeting", "Technical discussion about API"

        summary (Optional[str]): Concise summary of the memory content.
            Typically 1-3 sentences describing the key points.

        episode (Optional[str]): Detailed narrative description.
            Full episodic content for EPISODE_SUMMARY type memories.

        group_id (Optional[str]): ID of the group where this memory originated.
            None for personal/individual memories.

        participants (Optional[List[str]]): User IDs of people involved in this memory.
            Useful for multi-party conversations.

        type (Optional[SourceType]): Source type of the original data.
            Usually SourceType.CONVERSATION for chat-based memories.

        keywords (Optional[List[str]]): Extracted keywords for search/indexing.
            Example: ["python", "api", "authentication"]

        linked_entities (Optional[List[str]]): Related entity references.
            Example: project names, product names, external references.

        memunit_event_id_list (Optional[List[str]]): MemUnit event IDs used to create this memory.
            Different from ori_event_id_list which may reference other sources.

        semantic_memories (Optional[List[SemanticMemoryItem]]): Associated semantic predictions.
            Forward-looking associations about how events may impact the user.

        extend (Optional[Dict[str, Any]]): Extension field for additional metadata.
            Use for custom fields that don't fit the standard schema.
    """

    # === Required Fields ===
    memory_type: MemoryType
    user_id: str
    timestamp: datetime.datetime
    ori_event_id_list: List[str]

    # === Content Fields ===
    subject: Optional[str] = None
    summary: Optional[str] = None
    episode: Optional[str] = None

    # === Context Fields ===
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    type: Optional[SourceType] = None

    # === Indexing Fields ===
    keywords: Optional[List[str]] = None
    linked_entities: Optional[List[str]] = None

    # === Relationship Fields ===
    memunit_event_id_list: Optional[List[str]] = None
    semantic_memories: Optional[List['SemanticMemoryItem']] = None

    # === Extension Field ===
    extend: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Post-initialization hook for subclasses.

        Subclasses should override this to set memory_type automatically.
        Always call super().__post_init__() after setting memory_type.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary for serialization.

        Handles type conversions:
        - datetime -> ISO format string
        - MemoryType/SourceType -> string value
        - SemanticMemoryItem -> dict via to_dict()

        Returns:
            Dictionary representation suitable for JSON serialization or database storage.
        """
        # Handle timestamp conversion (may be datetime, str, or None)
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
