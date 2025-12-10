"""Online retrieval module - for API services with database queries.

This module contains retrieval components that query databases directly
(Milvus for vector search, Elasticsearch for keyword search).

Used by: API controllers (agentic_v2_controller, agentic_v3_controller)

Note: Currently delegates to agents.memory_manager.MemoryManager.
Future refactoring will move the actual implementation here.

Submodules (planned):
- lightweight.py: Fast retrieval (embedding + BM25 + RRF)
- agentic.py: LLM-guided multi-round retrieval
- keyword.py: Keyword-based retrieval (ES)
- vector.py: Vector-based retrieval (Milvus)
- hybrid.py: Hybrid retrieval (keyword + vector)
- result_grouper.py: Group results by group_id
"""

# Re-export from MemoryManager for backward compatibility
# TODO: Move actual implementation here and make MemoryManager a thin wrapper

__all__ = []
