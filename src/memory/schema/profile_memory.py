"""
Profile Memory - User Profile and Characteristics.

This module defines ProfileMemory, which captures a user's professional profile,
skills, personality traits, and behavioral patterns extracted from conversations.

Profile Categories:
-------------------
1. Skills: Technical (hard_skills) and interpersonal (soft_skills) abilities
2. Personality: Traits, decision-making style, tendencies
3. Professional: Work responsibilities, projects, goals
4. Behavioral: Working habits, preferences, communication style
5. Motivational: Values, motivations, fears

Evidence-Based Design:
----------------------
All profile attributes use an evidence-based format to track provenance:
    [{"value": "Python", "level": "expert", "evidences": ["2024-01-01|conv_123"]}]

This allows:
- Tracing back to source conversations
- Confidence assessment based on evidence count
- Merging profiles over time

Usage:
------
    from memory.schema import ProfileMemory

    profile = ProfileMemory(
        user_id="user_123",
        timestamp=datetime.now(),
        ori_event_id_list=["memunit_456"],
        user_name="Alice",
        hard_skills=[
            {"value": "Python", "level": "expert", "evidences": ["2024-01|conv_1"]},
            {"value": "React", "level": "intermediate", "evidences": ["2024-02|conv_2"]}
        ],
        personality=[
            {"value": "analytical", "evidences": ["2024-01|conv_1"]}
        ]
    )
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class ProfileMemory(Memory):
    """
    Profile Memory - Comprehensive user profile extracted from conversations.

    Extends Memory base class with extensive profile attributes.
    Automatically sets memory_type to PROFILE.

    All list attributes follow the evidence-based format:
        [{"value": <string>, "evidences": [<date>|<conv_id>, ...], ...}]

    Attribute Groups:

    1. Basic Identity:
        user_name: Display name of the user

    2. Skills (evidence-based lists):
        hard_skills: Technical skills with proficiency levels
            Example: [{"value": "Python", "level": "expert", "evidences": [...]}]
        soft_skills: Interpersonal/soft skills
            Example: [{"value": "communication", "evidences": [...]}]

    3. Cognitive Style:
        way_of_decision_making: How the user makes decisions
        personality: Personality traits observed
        tendency: Behavioral tendencies

    4. Professional Context:
        projects_participated: Projects the user has worked on (ProjectInfo objects)
        user_goal: Career or personal goals
        work_responsibility: Job responsibilities and duties
        working_habit_preference: Work style preferences

    5. Personal Interests:
        interests: Hobbies and areas of interest

    6. Motivational Profile:
        motivation_system: What motivates the user
        fear_system: Concerns and fears
        value_system: Core values

    7. Communication Style:
        humor_use: How the user uses humor
        colloquialism: Colloquial expressions used

    8. Group Dynamics:
        group_importance_evidence: Evidence of user's importance in the group

    9. Reasoning:
        output_reasoning: Explanation of profile extraction reasoning
    """

    # === Basic Identity ===
    user_name: Optional[str] = None

    # === Skills (with proficiency levels) ===
    # Format: [{"value": "skill_name", "level": "beginner|intermediate|expert", "evidences": [...]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None
    soft_skills: Optional[List[Dict[str, Any]]] = None

    # === Extraction Reasoning ===
    output_reasoning: Optional[str] = None

    # === Cognitive Style ===
    # Format: [{"value": "trait_description", "evidences": [...]}]
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    personality: Optional[List[Dict[str, Any]]] = None
    tendency: Optional[List[Dict[str, Any]]] = None

    # === Professional Context ===
    # projects_participated: List of ProjectInfo objects (see profile_memory/types.py)
    projects_participated: Optional[List[Any]] = None
    user_goal: Optional[List[Dict[str, Any]]] = None
    work_responsibility: Optional[List[Dict[str, Any]]] = None
    working_habit_preference: Optional[List[Dict[str, Any]]] = None

    # === Personal Interests ===
    interests: Optional[List[Dict[str, Any]]] = None

    # === Motivational Profile ===
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None

    # === Communication Style ===
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None

    # === Group Dynamics ===
    # GroupImportanceEvidence object (see profile_memory/types.py)
    group_importance_evidence: Optional[Any] = None

    def __post_init__(self) -> None:
        """Initialize profile memory with correct type."""
        self.memory_type = MemoryType.PROFILE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ProfileMemory to dictionary for serialization.

        Extends base class to_dict() with all profile-specific fields.
        Handles nested objects (ProjectInfo, GroupImportanceEvidence) by
        calling their to_dict() methods if available.

        Returns:
            Dictionary containing all profile fields.
        """
        base_dict = super().to_dict()

        base_dict.update({
            # Basic Identity
            "user_name": self.user_name,

            # Skills
            "hard_skills": self.hard_skills,
            "soft_skills": self.soft_skills,

            # Reasoning
            "output_reasoning": self.output_reasoning,

            # Cognitive Style
            "way_of_decision_making": self.way_of_decision_making,
            "personality": self.personality,
            "tendency": self.tendency,

            # Professional Context
            "projects_participated": [
                p.to_dict() if hasattr(p, 'to_dict') else p
                for p in (self.projects_participated or [])
            ] if self.projects_participated else None,
            "user_goal": self.user_goal,
            "work_responsibility": self.work_responsibility,
            "working_habit_preference": self.working_habit_preference,

            # Personal Interests
            "interests": self.interests,

            # Motivational Profile
            "motivation_system": self.motivation_system,
            "fear_system": self.fear_system,
            "value_system": self.value_system,

            # Communication Style
            "humor_use": self.humor_use,
            "colloquialism": self.colloquialism,

            # Group Dynamics
            "group_importance_evidence": (
                self.group_importance_evidence.to_dict()
                if hasattr(self.group_importance_evidence, 'to_dict')
                else self.group_importance_evidence
            ) if self.group_importance_evidence else None,
        })

        return base_dict
