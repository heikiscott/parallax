"""Lightweight retrieval pipeline - fast, no LLM calls.

This pipeline performs pure algorithmic retrieval:
1. Parallel Embedding + BM25 search
2. Top-K candidates from each
3. RRF (Reciprocal Rank Fusion)
4. Return top-K results

Advantages:
- Fast: No LLM calls, pure vector/lexical retrieval
- Low cost: No LLM API fees
- Stable: No network dependency, pure local computation

Use cases:
- Latency-sensitive scenarios
- Budget-limited scenarios
- Simple and clear queries
"""

import time
import asyncio
import logging
from typing import List, Tuple, Any, Optional

from .search_utils import (
    search_with_emb_index,
    search_with_bm25_index,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


async def lightweight_retrieval(
    query: str,
    emb_index: Any,
    bm25: Any,
    docs: List[dict],
    config: Any,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Lightweight fast retrieval without LLM calls.

    Flow:
    1. Parallel Embedding + BM25 search
    2. Each returns Top-50 candidates
    3. RRF fusion
    4. Return Top-20 results

    Args:
        query: User query
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list
        config: ExperimentConfig with retrieval parameters:
            - lightweight_emb_top_n: Embedding candidates (default 50)
            - lightweight_bm25_top_n: BM25 candidates (default 50)
            - lightweight_final_top_n: Final results (default 20)

    Returns:
        (final_results, metadata)
    """
    start_time = time.time()

    metadata = {
        "retrieval_mode": "lightweight",
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    # Get config values with defaults
    emb_top_n = getattr(config, 'lightweight_emb_top_n', 50)
    bm25_top_n = getattr(config, 'lightweight_bm25_top_n', 50)
    final_top_n = getattr(config, 'lightweight_final_top_n', 20)

    # Parallel Embedding + BM25 search
    emb_task = search_with_emb_index(query, emb_index, top_n=emb_top_n)
    bm25_task = asyncio.to_thread(search_with_bm25_index, query, bm25, docs, bm25_top_n)

    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)

    metadata["emb_count"] = len(emb_results)
    metadata["bm25_count"] = len(bm25_results)

    # RRF fusion
    if not emb_results and not bm25_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    elif not emb_results:
        final_results = bm25_results[:final_top_n]
    elif not bm25_results:
        final_results = emb_results[:final_top_n]
    else:
        fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=60)
        final_results = fused_results[:final_top_n]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    return final_results, metadata
