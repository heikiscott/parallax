"""ProfileMemory - User profile memory data structure."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class ProfileMemory(Memory):
    """
    Profile memory result class.

    Contains user profile information extracted from conversations.
    All list attributes now contain dicts with 'value' and 'evidences' fields.
    """

    user_name: Optional[str] = None

    # Skills: [{"value": "Python", "level": "高级", "evidences": ["2024-01-01|conv_123"]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None
    soft_skills: Optional[List[Dict[str, Any]]] = None

    output_reasoning: Optional[str] = None

    # Other attributes: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    personality: Optional[List[Dict[str, Any]]] = None
    # ProjectInfo objects from profile_memory/types.py
    projects_participated: Optional[List[Any]] = None
    user_goal: Optional[List[Dict[str, Any]]] = None
    work_responsibility: Optional[List[Dict[str, Any]]] = None
    working_habit_preference: Optional[List[Dict[str, Any]]] = None
    interests: Optional[List[Dict[str, Any]]] = None
    tendency: Optional[List[Dict[str, Any]]] = None

    # Motivational attributes
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None

    # GroupImportanceEvidence from profile_memory/types.py
    group_importance_evidence: Optional[Any] = None

    def __post_init__(self) -> None:
        """Ensure the memory type is set to PROFILE."""
        self.memory_type = MemoryType.PROFILE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Override to_dict() to include all ProfileMemory fields."""
        base_dict = super().to_dict()

        base_dict.update({
            "user_name": self.user_name,
            "hard_skills": self.hard_skills,
            "soft_skills": self.soft_skills,
            "output_reasoning": self.output_reasoning,
            "way_of_decision_making": self.way_of_decision_making,
            "personality": self.personality,
            "projects_participated": [
                p.to_dict() if hasattr(p, 'to_dict') else p
                for p in (self.projects_participated or [])
            ] if self.projects_participated else None,
            "user_goal": self.user_goal,
            "work_responsibility": self.work_responsibility,
            "working_habit_preference": self.working_habit_preference,
            "interests": self.interests,
            "tendency": self.tendency,
            "motivation_system": self.motivation_system,
            "fear_system": self.fear_system,
            "value_system": self.value_system,
            "humor_use": self.humor_use,
            "colloquialism": self.colloquialism,
            "group_importance_evidence": (
                self.group_importance_evidence.to_dict()
                if hasattr(self.group_importance_evidence, 'to_dict')
                else self.group_importance_evidence
            ) if self.group_importance_evidence else None,
        })

        return base_dict
