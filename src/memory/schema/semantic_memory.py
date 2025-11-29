"""
Semantic Memory Data Structures.

This module defines data structures for semantic memories - factual knowledge
extracted from episodic memories that captures objective information and
forward-looking associations.

Two Classes:
------------
1. SemanticMemory: Stored factual knowledge (persisted in database)
2. SemanticMemoryItem: Predictive association (attached to MemUnit/Memory)

Example:
--------
    Episode: "Alice discussed using Python for the API project"

    SemanticMemory (factual):
        content: "Alice knows Python programming"

    SemanticMemoryItem (predictive):
        content: "Alice may need Python documentation resources"
        evidence: "Alice is working on Python API project"

Usage:
------
    from memory.schema import SemanticMemory, SemanticMemoryItem

    # Factual knowledge
    semantic = SemanticMemory(
        user_id="user_123",
        content="User is proficient in Python",
        knowledge_type="skill"
    )

    # Predictive association
    item = SemanticMemoryItem(
        content="User may benefit from advanced Python courses",
        evidence="User is learning Python",
        start_time="2024-01-01"
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import datetime

from utils.datetime_utils import to_iso_format


@dataclass
class SemanticMemory:
    """
    Semantic Memory - Factual knowledge extracted from episodes.

    Represents objective, factual information derived from episodic memories.
    This is stored separately in the database and can be queried independently.

    Attributes:
        user_id (str): ID of the user this knowledge belongs to.

        content (str): The factual knowledge statement.
            Example: "User has experience with React development"

        knowledge_type (str): Category of knowledge. Default: "knowledge".
            Common types: "skill", "preference", "fact", "relationship"

        source_episodes (List[str]): Episode IDs this knowledge was derived from.
            Provides traceability to original sources.

        created_at (datetime): When this knowledge was extracted.
            Auto-set to current time if not provided.

        group_id (Optional[str]): Group context where knowledge was learned.

        participants (Optional[List[str]]): People involved in the source events.

        metadata (Optional[Dict[str, Any]]): Additional structured information.
            Can include confidence scores, extraction method, etc.
    """

    # === Required Fields ===
    user_id: str
    content: str

    # === Classification Fields ===
    knowledge_type: str = "knowledge"

    # === Provenance Fields ===
    source_episodes: List[str] = None
    created_at: datetime.datetime = None

    # === Context Fields ===
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None

    # === Metadata ===
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.source_episodes is None:
            self.source_episodes = []
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
    Semantic Memory Item - Predictive association from events.

    Represents a forward-looking prediction about how events might impact
    or relate to a user. These are attached to MemUnits and Memories to
    capture potential future relevance.

    Unlike SemanticMemory (factual), SemanticMemoryItem is predictive:
    - "User may need X in the future"
    - "This event suggests user is interested in Y"

    Attributes:
        content (str): The predictive association statement.
            Example: "User may need project management tools soon"

        evidence (Optional[str]): Supporting fact from the source event.
            Brief (≤30 chars) quote or fact supporting this prediction.
            Example: "mentioned starting new project"

        start_time (Optional[str]): When this association becomes relevant.
            Format: "YYYY-MM-DD". May be inferred from event timing.

        end_time (Optional[str]): When this association expires.
            Format: "YYYY-MM-DD". None if indefinite.

        duration_days (Optional[int]): Expected relevance duration in days.
            Used for time-decay in retrieval scoring.

        source_episode_id (Optional[str]): Episode this was derived from.
            Links back to the source for verification.

        embedding (Optional[List[float]]): Vector embedding for similarity search.
            Pre-computed for efficient retrieval.
    """

    # === Required Field ===
    content: str

    # === Evidence Field ===
    evidence: Optional[str] = None

    # === Time Relevance Fields ===
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_days: Optional[int] = None

    # === Provenance Field ===
    source_episode_id: Optional[str] = None

    # === Search Field ===
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "evidence": self.evidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_days": self.duration_days,
            "source_episode_id": self.source_episode_id,
            "embedding": self.embedding,
        }
