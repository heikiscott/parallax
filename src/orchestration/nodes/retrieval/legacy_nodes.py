"""Legacy node wrappers for existing retrieval pipelines.

This module wraps the existing retrieval pipelines (lightweight, agentic) as
LangGraph nodes to enable gradual migration without breaking existing functionality.

These nodes call the mature pipelines in src/retrieval/offline/pipelines/.
"""

import logging
from typing import Dict, Any, List, Tuple

from ...state import RetrievalState, Document
from ...context import ExecutionContext
from .. import register_node

logger = logging.getLogger(__name__)


def _convert_to_documents(
    results: List[Tuple[dict, float]],
    retrieval_method: str
) -> List[Document]:
    """Convert pipeline results (doc, score) tuples to Document objects.

    Args:
        results: List of (doc_dict, score) tuples from retrieval pipelines
        retrieval_method: Name of the retrieval method used

    Returns:
        List of Document objects
    """
    documents = []
    for doc, score in results:
        documents.append(Document(
            id=doc.get("unit_id", doc.get("id", "")),
            content=doc.get("narrative", doc.get("content", "")),
            score=score,
            metadata={
                "retrieval_method": retrieval_method,
                "original_doc": doc,  # Preserve original document
            }
        ))
    return documents


@register_node("lightweight_retrieval_node")
async def lightweight_retrieval_node(
    state: RetrievalState,
    context: ExecutionContext
) -> Dict[str, Any]:
    """Lightweight retrieval node - fast, no LLM calls.

    Wraps the lightweight_retrieval pipeline which performs:
    1. Parallel Embedding + BM25 search
    2. RRF (Reciprocal Rank Fusion)
    3. Return top-K results

    Args:
        state: Current retrieval state with query
        context: Execution context with memory_index, bm25_index, etc.

    Returns:
        Dict with retrieved documents and metadata
    """
    from src.retrieval.offline.pipelines.lightweight import lightweight_retrieval

    query = state["query"]

    # Build config object from context
    class LightweightConfig:
        def __init__(self, ctx: ExecutionContext):
            self.lightweight_emb_top_n = ctx.get_config_value("emb_top_k", 50)
            self.lightweight_bm25_top_n = ctx.get_config_value("bm25_top_k", 50)
            self.lightweight_final_top_n = ctx.get_config_value("final_top_k", 20)

    config = LightweightConfig(context)

    # Get documents for BM25 from memory_index
    all_docs = []
    if hasattr(context.memory_index, 'get_all_docs'):
        all_docs = context.memory_index.get_all_docs()
    elif hasattr(context.memory_index, 'memories'):
        all_docs = [
            {"unit_id": m.id, "narrative": m.narrative}
            for m in context.memory_index.memories.values()
        ]

    # Execute lightweight retrieval
    results, metadata = await lightweight_retrieval(
        query=query,
        emb_index=context.memory_index,
        bm25=context.bm25_index,
        docs=all_docs,
        config=config,
    )

    # Convert to Document objects
    documents = _convert_to_documents(results, "lightweight")

    return {
        "documents": documents,
        "metadata": {
            "retrieval_method": "lightweight",
            "pipeline": "legacy_lightweight",
            **metadata
        }
    }


@register_node("agentic_retrieval_node")
async def agentic_retrieval_node(
    state: RetrievalState,
    context: ExecutionContext
) -> Dict[str, Any]:
    """Agentic retrieval node - multi-round LLM-guided retrieval.

    Wraps the agentic_retrieval pipeline which performs:
    1. Round 1: Hybrid search -> Rerank -> Sufficiency check
    2. If insufficient: LLM generates refined queries
    3. Round 2: Multi-query retrieval and merge
    4. Final rerank and cluster expansion

    Requires: llm_provider in context

    Args:
        state: Current retrieval state with query
        context: Execution context with services

    Returns:
        Dict with retrieved documents and metadata
    """
    from src.retrieval.offline.pipelines.agentic import agentic_retrieval

    query = state["query"]

    if context.llm_provider is None:
        logger.warning("agentic_retrieval_node requires llm_provider, falling back to hybrid")
        # Fall back to lightweight if no LLM
        return await lightweight_retrieval_node(state, context)

    # Build config object from context
    class AgenticConfig:
        def __init__(self, ctx: ExecutionContext):
            # Hybrid search params
            self.hybrid_emb_candidates = ctx.get_config_value("emb_top_k", 50)
            self.hybrid_bm25_candidates = ctx.get_config_value("bm25_top_k", 50)
            self.hybrid_rrf_k = ctx.get_config_value("rrf_k", 60)
            # Reranker params
            self.use_reranker = ctx.get_config_value("use_reranker", True)
            self.reranker_instruction = ctx.get_config_value("reranker_instruction", None)
            self.reranker_batch_size = ctx.get_config_value("reranker_batch_size", 20)
            self.reranker_max_retries = ctx.get_config_value("reranker_max_retries", 10)
            self.reranker_retry_delay = ctx.get_config_value("reranker_retry_delay", 0.8)
            self.reranker_timeout = ctx.get_config_value("reranker_timeout", 60.0)
            self.reranker_fallback_threshold = ctx.get_config_value("reranker_fallback_threshold", 0.3)
            # Multi-query
            self.use_multi_query = ctx.get_config_value("use_multi_query", True)
            # Cluster expansion
            self.group_event_cluster_retrieval_config = ctx.get_config_value(
                "group_event_cluster_retrieval_config",
                {"enable_group_event_cluster_retrieval": False}
            )

    config = AgenticConfig(context)

    # LLM config
    llm_config = context.get_config_value("llm_config", {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    })

    # Get documents for BM25 from memory_index
    all_docs = []
    if hasattr(context.memory_index, 'get_all_docs'):
        all_docs = context.memory_index.get_all_docs()
    elif hasattr(context.memory_index, 'memories'):
        all_docs = [
            {"unit_id": m.id, "narrative": m.narrative}
            for m in context.memory_index.memories.values()
        ]

    # Execute agentic retrieval
    results, metadata = await agentic_retrieval(
        query=query,
        config=config,
        llm_provider=context.llm_provider,
        llm_config=llm_config,
        emb_index=context.memory_index,
        bm25=context.bm25_index,
        docs=all_docs,
        cluster_index=context.cluster_index,
        enable_traversal_stats=context.get_config_value("enable_traversal_stats", False),
    )

    # Convert to Document objects
    documents = _convert_to_documents(results, "agentic")

    return {
        "documents": documents,
        "metadata": {
            "retrieval_method": "agentic",
            "pipeline": "legacy_agentic",
            **metadata
        }
    }
