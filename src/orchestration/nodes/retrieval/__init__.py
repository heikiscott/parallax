"""Retrieval Nodes - Node implementations for retrieval workflows.

This module contains all retrieval-related workflow nodes:
- Classification nodes: Question type classification
- Legacy nodes: Wrappers around existing retrieval pipelines
- (future) Basic retrieval nodes: Embedding, BM25, hybrid search
- (future) Expansion nodes: Multi-query, cluster expansion
- (future) Rerank nodes: Reranking logic

Usage:
    from orchestration.nodes.retrieval import classify_question_node

    result = await classify_question_node(state, context)
"""

from .classification_nodes import classify_question_node
from .legacy_nodes import lightweight_retrieval_node, agentic_retrieval_node

__all__ = [
    # Classification
    "classify_question_node",
    # Legacy pipeline wrappers
    "lightweight_retrieval_node",
    "agentic_retrieval_node",
]
