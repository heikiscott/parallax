"""Offline retrieval module - for evaluation with in-memory indices.

This module contains retrieval components that work with pre-loaded
in-memory indices (embedding index, BM25 index, document list).

Used by: eval/ evaluation framework

Submodules:
- pipelines: High-level retrieval pipelines (lightweight, agentic)
- retrievers: Individual retrieval strategies (embedding, BM25, hybrid)
- expanders: Query and result expansion (multi-query, cluster)
- rerankers: Reranking models and utilities
"""

from .pipelines import lightweight_retrieval, agentic_retrieval
from .pipelines.search_utils import reciprocal_rank_fusion

__all__ = [
    "lightweight_retrieval",
    "agentic_retrieval",
    "reciprocal_rank_fusion",
]
