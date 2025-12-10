"""Online retrieval module - for API services with database queries.

This module contains retrieval components that query databases directly
(Milvus for vector search, Elasticsearch for keyword search).

Used by: API controllers (agentic_v2_controller, agentic_v3_controller)

Architecture:
- Currently wraps agents.memory_manager.MemoryManager methods
- Future: Move implementations here, make MemoryManager a thin facade

Submodules:
- lightweight: Fast retrieval (embedding + BM25 + RRF)
- agentic: LLM-guided multi-round retrieval

Usage:
    from retrieval.online import OnlineRetriever

    retriever = OnlineRetriever()
    result = await retriever.retrieve_lightweight(query, user_id, group_id)
    result = await retriever.retrieve_agentic(query, user_id, group_id, llm_provider)
"""

from agents.memory_manager import MemoryManager

# Re-export MemoryManager as OnlineRetriever for semantic clarity
# This provides a cleaner interface while maintaining backward compatibility
OnlineRetriever = MemoryManager

__all__ = [
    "OnlineRetriever",
    "MemoryManager",  # Keep for backward compatibility
]
