"""
Memory extraction request data class.
"""

from dataclasses import dataclass
from typing import List, Optional

from memory.schema import Memory, MemUnit


@dataclass
class MemoryExtractRequest:
    """Request for extracting Memory from MemUnits."""

    memunit_list: List[MemUnit]
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None

    old_memory_list: Optional[List[Memory]] = None

    user_organization: Optional[List] = None
