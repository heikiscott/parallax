"""
Memory Schema Module

This module contains all memory-related data structures and type definitions.
Each type is defined in its own file for better maintainability.
"""

from .memory_type import MemoryType
from .source_type import SourceType
from .memunit import MemUnit
from .memory import Memory
from .semantic_memory import SemanticMemory, SemanticMemoryItem
from .episode_memory import EpisodeMemory
from .profile_memory import ProfileMemory
from .group_profile_memory import GroupProfileMemory

# Backward compatibility alias
RawDataType = SourceType

__all__ = [
    "MemoryType",
    "SourceType",
    "RawDataType",  # Backward compatibility
    "MemUnit",
    "Memory",
    "SemanticMemory",
    "SemanticMemoryItem",
    "EpisodeMemory",
    "ProfileMemory",
    "GroupProfileMemory",
]
