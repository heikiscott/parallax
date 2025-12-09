"""Retrieval module - components for memory retrieval.

This module contains all retrieval-related functionality:
- core: Core types and utilities (Document, etc.)
- classification: Question classification for retrieval routing
- retrievers: Individual retrieval strategies (embedding, BM25, hybrid)
- expanders: Query and result expansion (multi-query, cluster)
- rerankers: Reranking models and utilities
- pipelines: High-level retrieval pipelines
- strategies: Strategy routing and orchestration
- services: Vectorization and reranking services

Note: Submodules are NOT eagerly imported to avoid dependency issues.
Import specific modules as needed:
    from src.retrieval.pipelines import lightweight_retrieval
    from src.retrieval.classification import classify_question
"""

from .core import Document

# Classification exports (eager import since it's commonly used)
from .classification import (
    QuestionType,
    RetrievalStrategy,
    ClassificationResult,
    classify_question,
    should_use_group_event_cluster,
)

__all__ = [
    "Document",
    # Classification
    "QuestionType",
    "RetrievalStrategy",
    "ClassificationResult",
    "classify_question",
    "should_use_group_event_cluster",
]
