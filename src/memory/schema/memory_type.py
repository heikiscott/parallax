"""
Memory Type Enumeration.

This module defines the types of memories that can be extracted and stored
in the Parallax memory system.

Memory Type Hierarchy:
----------------------
    MemUnit (raw extraction unit)
        |
        +-- EpisodeMemory (EPISODE_SUMMARY)
        |       Personal narrative of events from user's perspective
        |
        +-- SemanticMemory (SEMANTIC_SUMMARY)
        |       Factual knowledge extracted from episodes
        |
        +-- EventLog (EVENT_LOG)
        |       Atomic facts with timestamps
        |
        +-- ProfileMemory (PROFILE)
        |       User skills, personality, preferences
        |
        +-- GroupProfileMemory (GROUP_PROFILE)
                Group topics, roles, dynamics

Usage:
------
    from memory.schema import MemoryType

    if memory.memory_type == MemoryType.EPISODE_SUMMARY:
        # Handle episode memory
        pass
"""

from enum import Enum


class MemoryType(Enum):
    """
    Types of memories that can be extracted from conversations.

    Each type represents a different aspect of memory storage:
    - Episode memories capture personal narratives
    - Semantic memories capture factual knowledge
    - Profile memories capture user characteristics
    - Group memories capture collective dynamics

    Attributes:
        EPISODE_SUMMARY: Personal episodic memory from user's perspective.
            Contains narrative descriptions of events as experienced by a specific user.

        SEMANTIC_SUMMARY: Factual semantic knowledge.
            Contains objective facts and knowledge extracted from conversations.

        EVENT_LOG: Timestamped atomic facts.
            Contains individual facts with precise time information.

        PROFILE: User profile and characteristics.
            Contains skills, personality traits, preferences, and behavioral patterns.

        GROUP_PROFILE: Group collective profile.
            Contains group topics, member roles, and interaction patterns.

        BASE_MEMORY: Stable, verifiable facts about a person (legacy, use PROFILE).

        PREFERENCES: User preferences (legacy, merged into PROFILE).

        RELATIONSHIPS: Interpersonal relationships (reserved for future use).

        CORE: Core memory for essential user info (legacy, use PROFILE).
    """

    # Primary memory types (actively used)
    EPISODE_SUMMARY = "episode_summary"
    SEMANTIC_SUMMARY = "semantic"
    EVENT_LOG = "event_log"
    PROFILE = "profile"
    GROUP_PROFILE = "group_profile"

    # Legacy/Reserved types (for backward compatibility)
    BASE_MEMORY = "baseMemory"
    PREFERENCES = "preferences"
    RELATIONSHIPS = "relationships"
    CORE = "core"
