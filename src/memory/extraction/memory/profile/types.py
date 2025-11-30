"""Type definitions for profile memory extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProjectInfo:
    """Project participation details."""

    project_id: str
    project_name: str = ""
    entry_date: str = ""
    subtasks: Optional[List[Dict[str, Any]]] = None
    user_objective: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    user_concerns: Optional[List[Dict[str, Any]]] = None


@dataclass
class ImportanceEvidence:
    """Evidence used to evaluate user importance in a group."""

    user_id: str
    group_id: str
    speak_count: int = 0
    refer_count: int = 0
    conversation_count: int = 0


@dataclass
class GroupImportanceEvidence:
    """Aggregated importance evidence for a user across a group."""

    group_id: str
    evidence_list: List[ImportanceEvidence] = field(default_factory=list)
    is_important: bool = False


@dataclass
class ProfileMemoryExtractRequest:
    """Request for extracting ProfileMemory from MemUnits."""

    from memory.schema import MemUnit, Memory

    memunit_list: List[MemUnit]
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None

    old_memory_list: Optional[List[Memory]] = None

    user_organization: Optional[List] = None

    evidence_list: Optional[List[ImportanceEvidence]] = None
