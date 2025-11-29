"""
Episode Memory - Personal Narrative Memory.

This module defines EpisodeMemory, which captures personal narrative experiences
from a specific user's perspective. Episode memories are the primary output
of the memory extraction pipeline.

Characteristics:
----------------
- Personal: Written from ONE user's point of view
- Narrative: Describes events as a story, not just facts
- Contextual: Includes emotions, reactions, and interpretations
- Traceable: Links back to source MemUnit via event_id

Example:
--------
    MemUnit (group event):
        "Alice and Bob discussed the new API design. Bob suggested REST,
         while Alice preferred GraphQL. They agreed to prototype both."

    EpisodeMemory for Alice:
        "I discussed API design with Bob. He suggested REST, but I prefer
         GraphQL for its flexibility. We agreed to prototype both approaches
         to make a data-driven decision."

    EpisodeMemory for Bob:
        "Alice and I had a productive discussion about API design. I proposed
         REST for its simplicity, and she suggested GraphQL. We'll prototype
         both to compare."

Usage:
------
    from memory.schema import EpisodeMemory

    episode = EpisodeMemory(
        user_id="alice_123",
        timestamp=datetime.now(),
        ori_event_id_list=["memunit_456"],
        event_id="episode_789",
        episode="I discussed the project timeline with the team...",
        summary="Project timeline discussion",
        subject="Sprint planning"
    )
"""

from dataclasses import dataclass, field

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class EpisodeMemory(Memory):
    """
    Episode Memory - Personal narrative from user's perspective.

    Extends Memory base class with episode-specific fields.
    Automatically sets memory_type to EPISODE_SUMMARY.

    Each EpisodeMemory represents how ONE user experienced a set of events.
    The same MemUnit typically generates multiple EpisodeMemory instances,
    one for each participant, each with their own perspective.

    Attributes:
        event_id (str): Unique identifier for this episode memory.
            Different from ori_event_id_list which contains source MemUnit IDs.
            This is the episode's own ID for storage and retrieval.

    Inherited from Memory:
        - memory_type: Auto-set to MemoryType.EPISODE_SUMMARY
        - user_id: The user whose perspective this episode represents
        - timestamp: When the events occurred
        - ori_event_id_list: Source MemUnit event IDs
        - episode: The full narrative text (main content)
        - summary: Brief summary of the episode
        - subject: Topic/title of the episode
        - group_id, participants, keywords, etc.

    Example:
        >>> episode = EpisodeMemory(
        ...     user_id="user_123",
        ...     timestamp=datetime.now(),
        ...     ori_event_id_list=["memunit_1"],
        ...     event_id="episode_1",
        ...     episode="Today I learned about the new feature...",
        ...     summary="Learning about new feature",
        ...     subject="Feature onboarding"
        ... )
    """

    # === Episode-Specific Field ===
    event_id: str = field(default=None)

    def __post_init__(self):
        """
        Initialize episode memory with correct type.

        Sets memory_type to EPISODE_SUMMARY and calls parent initialization.
        """
        self.memory_type = MemoryType.EPISODE_SUMMARY
        super().__post_init__()
