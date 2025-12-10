"""Agents module - Memory services for online applications.

This module provides:
- FetchMemoryService: Memory retrieval by ID
- Memory models and DTOs

Note: MemoryManager has been removed. Use:
- services.mem_memorize.memorize for storing memories
- retrieval.online.retrieve_lightweight for lightweight retrieval
- retrieval.online.retrieve_agentic for agentic retrieval
"""

from .fetch_memory_service import get_fetch_memory_service, FetchMemoryService
from .memory_models import (
    MemoryType,
    RetrieveMethod,
    Metadata,
    BaseMemoryModel,
    MemoryModel,
)

__all__ = [
    # Fetch Memory Service
    "get_fetch_memory_service",
    "FetchMemoryService",
    # Models
    "MemoryType",
    "RetrieveMethod",
    "Metadata",
    "BaseMemoryModel",
    "MemoryModel",
]
