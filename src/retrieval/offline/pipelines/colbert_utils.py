"""ColBERT Search Utilities for Agentic V3 Pipeline.

This module provides ColBERT-based search functions to replace hybrid search in V3.

Core Functions:
- colbert_search: Execute ColBERT retrieval using MaxSim scoring
- multi_colbert_rrf_fusion: Fuse multiple ColBERT search results using RRF

ColBERT Index Format:
    [
        {
            "doc": { ... original document dict ... },
            "embeddings": np.ndarray  # [seq_len, 128]
        },
        ...
    ]
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np

from retrieval.services.colbert_service import get_colbert_service

logger = logging.getLogger(__name__)


async def colbert_search(
    query: str,
    colbert_index: List[Dict[str, Any]],
    top_n: int = 20,
    return_traversal_stats: bool = False
) -> List[Tuple[dict, float]] | Tuple[List[Tuple[dict, float]], dict]:
    """Execute ColBERT search using MaxSim scoring.

    This function scores all documents in the index against the query
    using ColBERT's late interaction mechanism (MaxSim).

    Args:
        query: User query string.
        colbert_index: Pre-built ColBERT index with structure:
            [{"doc": {...}, "embeddings": np.ndarray}, ...]
        top_n: Number of top results to return.
        return_traversal_stats: Whether to return traversal statistics.

    Returns:
        If return_traversal_stats=False:
            List of (doc, score) tuples sorted by score descending.
        If return_traversal_stats=True:
            Tuple of (results, stats_dict) where stats_dict contains:
            - total_docs_scored: Number of documents scored
            - scored_ids: Set of all scored document IDs
            - returned_ids: Set of returned document IDs
    """
    if not colbert_index:
        logger.warning("Empty ColBERT index provided")
        if return_traversal_stats:
            return [], {"total_docs_scored": 0, "scored_ids": set(), "returned_ids": set()}
        return []

    colbert_service = get_colbert_service()

    # Encode query
    query_emb = await colbert_service.encode_query(query)

    # Score all documents
    doc_scores = []
    scored_ids = set()

    for item in colbert_index:
        doc = item.get("doc")
        doc_emb = item.get("embeddings")

        if doc is None or doc_emb is None:
            continue

        if len(doc_emb) == 0:
            continue

        # Compute MaxSim score
        score = colbert_service.compute_maxsim(query_emb, doc_emb)
        doc_scores.append((doc, score))

        # Track scored IDs
        unit_id = doc.get("unit_id", "")
        if unit_id:
            scored_ids.add(unit_id)

    # Sort by score descending
    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    top_results = sorted_results[:top_n]

    if return_traversal_stats:
        returned_ids = set(doc.get("unit_id", "") for doc, _ in top_results if doc.get("unit_id"))
        stats = {
            "total_docs_scored": len(doc_scores),
            "scored_ids": scored_ids,
            "returned_ids": returned_ids,
        }
        return top_results, stats

    return top_results


def multi_colbert_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """Fuse multiple ColBERT search results using Reciprocal Rank Fusion (RRF).

    RRF is effective for combining results from multiple queries, as it:
    - Doesn't require score normalization
    - Rewards documents that appear in multiple result lists
    - Is robust to outliers

    RRF Score = Σ (1 / (k + rank_i)) for each query where document appears

    Args:
        results_list: List of result lists from different queries.
            Each result list is [(doc, score), ...] sorted by score descending.
        k: RRF constant (default 60). Higher k reduces the impact of top ranks.

    Returns:
        Fused results as [(doc, rrf_score), ...] sorted by RRF score descending.
    """
    if not results_list:
        return []

    if len(results_list) == 1:
        return results_list[0]

    # Accumulate RRF scores
    doc_rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}

    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            # Use unit_id as key, fallback to object id
            doc_id = doc.get("unit_id", str(id(doc)))

            if doc_id not in doc_map:
                doc_map[doc_id] = doc

            # RRF formula
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]


def deduplicate_results(
    results: List[Tuple[dict, float]],
    seen_ids: Optional[Set[str]] = None
) -> List[Tuple[dict, float]]:
    """Remove duplicate documents from results.

    Args:
        results: List of (doc, score) tuples.
        seen_ids: Optional set of already-seen IDs to exclude.

    Returns:
        Deduplicated results list.
    """
    if seen_ids is None:
        seen_ids = set()

    deduped = []
    for doc, score in results:
        doc_id = doc.get("unit_id", str(id(doc)))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped.append((doc, score))

    return deduped


def merge_round_results(
    round1_results: List[Tuple[dict, float]],
    round2_results: List[Tuple[dict, float]],
    merge_budget: int = 35
) -> Tuple[List[Tuple[dict, float]], Set[str], Set[str]]:
    """Merge Round 1 and Round 2 results with budget constraint.

    Round 1 results are prioritized, Round 2 fills remaining budget.

    Args:
        round1_results: Results from Round 1 retrieval.
        round2_results: Results from Round 2 retrieval.
        merge_budget: Maximum total results after merge.

    Returns:
        Tuple of:
        - merged_results: Combined results list
        - round1_ids: Set of IDs from Round 1
        - round2_added_ids: Set of IDs added from Round 2
    """
    round1_ids = {doc.get("unit_id", str(id(doc))) for doc, _ in round1_results}

    # Filter Round 2 to exclude duplicates
    round2_unique = [
        (doc, score) for doc, score in round2_results
        if doc.get("unit_id", str(id(doc))) not in round1_ids
    ]

    # Merge with budget
    combined = round1_results.copy()
    needed_from_round2 = max(0, merge_budget - len(combined))
    round2_slice = round2_unique[:needed_from_round2]
    combined.extend(round2_slice)

    round2_added_ids = {doc.get("unit_id", "") for doc, _ in round2_slice if doc.get("unit_id")}

    return combined, round1_ids, round2_added_ids


def build_origin_map(
    round1_ids: Set[str],
    round2_ids: Set[str]
) -> Dict[str, str]:
    """Build origin mapping for document IDs.

    Args:
        round1_ids: IDs from Round 1.
        round2_ids: IDs from Round 2.

    Returns:
        Dict mapping unit_id to origin ("round1" or "round2").
    """
    origin_map = {}
    for uid in round1_ids:
        if uid:
            origin_map[uid] = "round1"
    for uid in round2_ids:
        if uid and uid not in origin_map:
            origin_map[uid] = "round2"
    return origin_map
