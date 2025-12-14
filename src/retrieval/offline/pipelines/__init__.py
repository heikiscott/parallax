"""Retrieval pipelines - high-level retrieval workflows.

This module provides complete retrieval pipelines that orchestrate
retrievers, expanders, and LLM-based processing.

Pipelines:
- lightweight: Fast retrieval without LLM (Embedding + BM25 + RRF)
- agentic: LLM-guided multi-round retrieval with sufficiency checking (V1)
- agentic_v2: Type-aware multi-query at Round 1
- agentic_v3: Pure ColBERT retrieval with type-aware multi-query

Example:
    from retrieval.offline.pipelines import lightweight_retrieval, agentic_retrieval

    # Fast retrieval
    results, metadata = await lightweight_retrieval(query, emb_index, bm25, docs, config)

    # LLM-guided retrieval (V1 - hybrid)
    results, metadata = await agentic_retrieval(
        query, config, llm_provider, llm_config,
        emb_index, bm25, docs, cluster_index
    )

    # ColBERT retrieval (V3)
    results, metadata = await agentic_retrieval_v3(
        query, config, llm_provider, llm_config,
        colbert_index
    )
"""

from .lightweight import lightweight_retrieval
from .agentic import agentic_retrieval
from .agentic_v3 import agentic_retrieval_v3
from .rerank import reranker_search
from .llm_utils import (
    format_documents_for_llm,
    check_sufficiency,
    generate_refined_query,
    generate_multi_queries,
)
from .search_utils import (
    search_with_emb_index,
    search_with_bm25_index,
    reciprocal_rank_fusion,
    multi_rrf_fusion,
    hybrid_search_with_rrf,
)

__all__ = [
    # Pipelines
    "lightweight_retrieval",
    "agentic_retrieval",
    "agentic_retrieval_v3",
    # Reranking
    "reranker_search",
    # Search utilities
    "search_with_emb_index",
    "search_with_bm25_index",
    "reciprocal_rank_fusion",
    "multi_rrf_fusion",
    "hybrid_search_with_rrf",
    # LLM utilities
    "format_documents_for_llm",
    "check_sufficiency",
    "generate_refined_query",
    "generate_multi_queries",
]
