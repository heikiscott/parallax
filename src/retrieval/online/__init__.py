"""Online retrieval module - for API services with database queries.

This module contains retrieval components that query databases directly
(Milvus for vector search, Elasticsearch for keyword search).

Used by: API controllers (agentic_v3_controller)

Architecture:
- lightweight: Fast retrieval (embedding + BM25 + RRF)
- agentic: LLM-guided multi-round retrieval
- vector_store: Low-level vector store operations

Usage:
    from retrieval.online import retrieve_lightweight, retrieve_agentic

    # Fast retrieval
    result = await retrieve_lightweight(query, user_id, group_id)

    # LLM-guided retrieval
    result = await retrieve_agentic(query, user_id, group_id, llm_provider)
"""

from .lightweight import retrieve_lightweight
from .agentic import retrieve_agentic
from .vector_store import retrieve_from_vector_stores
from .utils import (
    format_datetime_field,
    parse_datetime_value,
    filter_semantic_memories_by_time,
)

__all__ = [
    # Main retrieval functions
    "retrieve_lightweight",
    "retrieve_agentic",
    # Low-level functions
    "retrieve_from_vector_stores",
    # Utilities
    "format_datetime_field",
    "parse_datetime_value",
    "filter_semantic_memories_by_time",
]
