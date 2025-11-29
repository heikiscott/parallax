"""GroupProfileMemory - Group profile memory data structure."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import datetime

from .memory import Memory
from .memory_type import MemoryType

# Note: TopicInfo is defined in the extractor module as it's extractor-specific
# This file only contains the core GroupProfileMemory class


@dataclass
class GroupProfileMemory(Memory):
    """
    Group Profile Memory aligned with design document.

    Contains group core information extracted from conversations.
    Evidences are now stored within topics and roles instead of separately.
    """

    group_name: Optional[str] = None

    # Topics with evidences and confidence (TopicInfo objects from extractor)
    topics: Optional[List[Any]] = field(default_factory=list)
    # Roles: role -> [{"user_id": "xxx", "user_name": "xxx", "confidence": "strong|weak", "evidences": [...]}]
    roles: Optional[Dict[str, List[Dict[str, str]]]] = field(default_factory=dict)

    def __post_init__(self):
        """Set memory_type to GROUP_PROFILE and call parent __post_init__."""
        self.memory_type = MemoryType.GROUP_PROFILE
        if self.topics is None:
            self.topics = []
        if self.roles is None:
            self.roles = {}
        super().__post_init__()
