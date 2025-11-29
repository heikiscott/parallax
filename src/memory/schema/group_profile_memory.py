"""
Group Profile Memory - Collective Group Characteristics.

This module defines GroupProfileMemory, which captures the collective profile
of a group including topics discussed and member roles.

Key Concepts:
-------------
1. Topics: Subjects/themes the group discusses regularly
2. Roles: Member functions within the group (leader, contributor, etc.)

Unlike ProfileMemory (individual), GroupProfileMemory captures:
- What the GROUP cares about (not individual preferences)
- How members RELATE to each other (roles, dynamics)
- Collective patterns (recurring topics, decision processes)

Data Structure:
---------------
topics: List of TopicInfo objects (defined in extractor module)
    Each topic has: name, status, confidence, evidences

roles: Dict mapping role type to list of users with that role
    Format: {
        "DECISION_MAKER": [
            {"user_id": "u1", "user_name": "Alice", "confidence": "strong", "evidences": [...]}
        ],
        "TOPIC_INITIATOR": [
            {"user_id": "u2", "user_name": "Bob", "confidence": "weak", "evidences": [...]}
        ]
    }

Role Types (defined in GroupRole enum in extractor):
- DECISION_MAKER: Makes final decisions
- OPINION_LEADER: Influences group opinions
- TOPIC_INITIATOR: Starts new discussions
- EXECUTION_PROMOTER: Drives execution
- CORE_CONTRIBUTOR: Key contributor
- COORDINATOR: Coordinates activities
- INFO_SUMMARIZER: Summarizes information

Usage:
------
    from memory.schema import GroupProfileMemory

    group_profile = GroupProfileMemory(
        user_id="group_admin",
        timestamp=datetime.now(),
        ori_event_id_list=["memunit_1", "memunit_2"],
        group_id="group_123",
        group_name="Engineering Team",
        topics=[topic_info_1, topic_info_2],
        roles={
            "DECISION_MAKER": [
                {"user_id": "alice", "user_name": "Alice", "confidence": "strong"}
            ]
        }
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class GroupProfileMemory(Memory):
    """
    Group Profile Memory - Collective characteristics of a group.

    Extends Memory base class with group-specific attributes.
    Automatically sets memory_type to GROUP_PROFILE.

    This captures the GROUP's collective identity, not individual profiles:
    - Topics: What the group discusses/cares about
    - Roles: Who plays what role in the group dynamics

    Attributes:
        group_name (Optional[str]): Display name of the group.
            Example: "Engineering Team", "Project Alpha Group"

        topics (Optional[List[Any]]): Discussion topics in the group.
            List of TopicInfo objects (from group_profile_memory_extractor).
            Each contains: topic name, status (exploring/consensus/etc),
            confidence level, and supporting evidences.

        roles (Optional[Dict[str, List[Dict[str, str]]]]): Member roles.
            Maps role type string to list of users holding that role.
            Each user entry contains:
            - user_id: User identifier
            - user_name: Display name
            - confidence: "strong" or "weak"
            - evidences: List of evidence strings

    Note:
        TopicInfo and GroupRole are defined in the extractor module
        (group_profile_memory_extractor.py) as they are extraction-specific.
        Consider moving them to schema/ if they become widely used.

    Example:
        >>> profile = GroupProfileMemory(
        ...     user_id="admin_123",
        ...     timestamp=datetime.now(),
        ...     ori_event_id_list=["mu_1"],
        ...     group_id="grp_456",
        ...     group_name="Dev Team",
        ...     topics=[...],  # TopicInfo objects
        ...     roles={"DECISION_MAKER": [{"user_id": "u1", ...}]}
        ... )
    """

    # === Group Identity ===
    group_name: Optional[str] = None

    # === Group Topics ===
    # List of TopicInfo objects (see group_profile_memory_extractor.py)
    # TopicInfo contains: topic, status, confidence, evidences
    topics: Optional[List[Any]] = field(default_factory=list)

    # === Member Roles ===
    # Format: {"ROLE_TYPE": [{"user_id": "...", "user_name": "...", "confidence": "...", "evidences": [...]}]}
    roles: Optional[Dict[str, List[Dict[str, str]]]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize group profile memory with correct type.

        Sets memory_type to GROUP_PROFILE and ensures topics/roles are initialized.
        """
        self.memory_type = MemoryType.GROUP_PROFILE
        if self.topics is None:
            self.topics = []
        if self.roles is None:
            self.roles = {}
        super().__post_init__()
