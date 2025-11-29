"""Dataclasses and type definitions for profile memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base_memory_extractor import MemoryExtractRequest


@dataclass
class ProjectInfo:
    """Project participation information."""

    project_id: str
    project_name: str
    entry_date: str
    subtasks: Optional[List[Dict[str, Any]]] = None
    user_objective: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    user_concerns: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "entry_date": self.entry_date,
            "subtasks": self.subtasks,
            "user_objective": self.user_objective,
            "contributions": self.contributions,
            "user_concerns": self.user_concerns,
        }


@dataclass
class ImportanceEvidence:
    """Aggregated evidence indicating user importance within a group."""

    user_id: str
    group_id: str
    speak_count: int = 0
    refer_count: int = 0
    conversation_count: int = 0


@dataclass
class GroupImportanceEvidence:
    """Group-level importance assessment for a user."""

    group_id: str
    evidence_list: List[ImportanceEvidence]
    is_important: bool

    def to_dict(self):
        return {
            "group_id": self.group_id,
            "evidence_list": [
                {
                    "user_id": e.user_id,
                    "group_id": e.group_id,
                    "speak_count": e.speak_count,
                    "refer_count": e.refer_count,
                    "conversation_count": e.conversation_count,
                }
                for e in self.evidence_list
            ],
            "is_important": self.is_important,
        }


@dataclass
class ProfileMemoryExtractRequest(MemoryExtractRequest):
    """Request payload used by ProfileMemoryExtractor."""

    pass
