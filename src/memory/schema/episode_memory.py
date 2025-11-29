"""EpisodeMemory - Episodic memory data structure."""

from dataclasses import dataclass, field

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class EpisodeMemory(Memory):
    """
    Episodic memory result class.

    Contains the essential information for extracted episodic memories.
    """

    event_id: str = field(default=None)

    def __post_init__(self):
        """Set memory_type to EPISODE_SUMMARY and call parent __post_init__."""
        self.memory_type = MemoryType.EPISODE_SUMMARY
        super().__post_init__()
