"""Cross-Attention Search Utilities for Agentic V4 Pipeline.

This module provides cross-attention based search functions to replace ColBERT search in V4.

Core Functions:
- cross_attention_score: Calculate cross-attention scores (TO BE IMPLEMENTED)
- cross_attention_search: Execute cross-attention retrieval using the scoring function
- multi_cross_attention_rrf_fusion: Fuse multiple search results using RRF

Document Index Format (same as V3, but without embeddings):
    [
        {
            "doc": { ... original document dict ... },
        },
        ...
    ]

Note: Unlike ColBERT which requires pre-computed embeddings, cross-attention
computes scores in real-time, so no index building is needed.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Core Interface - TO BE IMPLEMENTED BY ML TEAM
# =============================================================================

async def cross_attention_score(
    query: str,
    contexts: List[str],
    context_ids: List[str],
) -> List[Tuple[str, float]]:
    """
    计算 query 与每个 context 的交叉注意力分数。

    【桩函数】目前使用 ColBERT 计算分数，用于测试 V4 流程。
    后续由 ML 团队替换为真正的交叉注意力实现。

    Args:
        query: 查询文本 (用户问题)
        contexts: 上下文文本列表 (候选文档内容)
        context_ids: 对应的上下文 ID 列表 (与 contexts 一一对应)

    Returns:
        [(context_id, attention_score), ...] 按分数降序排列
        - context_id: 文档 ID
        - attention_score: 注意力分数 (越高越相关)

    Example:
        >>> query = "What is Alice's favorite color?"
        >>> contexts = ["Alice loves blue.", "Bob prefers red.", "Alice went to Paris."]
        >>> context_ids = ["doc_1", "doc_2", "doc_3"]
        >>> results = await cross_attention_score(query, contexts, context_ids)
        >>> # results: [("doc_1", 0.95), ("doc_3", 0.42), ("doc_2", 0.21)]

    TODO: 替换为真正的交叉注意力实现
    """
    # ============================================================
    # 【桩函数开始】使用 ColBERT 计算分数，用于测试 V4 流程
    # ML 团队替换时：删除此函数全部内容，实现真正的交叉注意力
    # ============================================================
    import hashlib
    from retrieval.services.colbert_service import get_colbert_service

    # 桩函数内部缓存
    if not hasattr(cross_attention_score, '_cache'):
        cross_attention_score._cache = {}

    if not contexts or not context_ids:
        return []

    colbert_service = get_colbert_service()
    cache = cross_attention_score._cache

    # 编码 query
    query_emb = await colbert_service.encode_query(query)

    # 获取或计算 context embeddings（使用缓存加速）
    doc_embeddings = []
    contexts_to_encode = []
    contexts_to_encode_indices = []

    for i, content in enumerate(contexts):
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash in cache:
            doc_embeddings.append(cache[content_hash])
        else:
            doc_embeddings.append(None)
            contexts_to_encode.append(content)
            contexts_to_encode_indices.append(i)

    # 批量编码未缓存的 contexts
    if contexts_to_encode:
        new_embeddings = await colbert_service.encode_documents(contexts_to_encode)
        for idx, emb in zip(contexts_to_encode_indices, new_embeddings):
            content_hash = hashlib.md5(contexts[idx].encode('utf-8')).hexdigest()
            cache[content_hash] = emb
            doc_embeddings[idx] = emb

    # 计算 MaxSim 分数
    results = []
    for ctx_id, doc_emb in zip(context_ids, doc_embeddings):
        if doc_emb is None or len(doc_emb) == 0:
            score = 0.0
        else:
            score = colbert_service.compute_maxsim(query_emb, doc_emb)
        results.append((ctx_id, float(score)))

    # 按分数降序排列
    results.sort(key=lambda x: x[1], reverse=True)

    return results
    # ============================================================
    # 【桩函数结束】
    # ============================================================


# =============================================================================
# Search Function - Wraps cross_attention_score
# =============================================================================

async def cross_attention_search(
    query: str,
    doc_index: List[Dict[str, Any]],
    top_n: int = 20,
    return_traversal_stats: bool = False
) -> List[Tuple[dict, float]] | Tuple[List[Tuple[dict, float]], dict]:
    """Execute cross-attention search over document index.

    This function wraps cross_attention_score() and handles document extraction.
    It mirrors the interface of colbert_search() for consistency.

    Args:
        query: User query string.
        doc_index: Document index with structure:
            [{"doc": {...}}, ...]
            The doc dict should contain "narrative" or "content" field for text.
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
    if not doc_index:
        logger.warning("Empty document index provided")
        if return_traversal_stats:
            return [], {"total_docs_scored": 0, "scored_ids": set(), "returned_ids": set()}
        return []

    # Extract contexts and IDs from index
    contexts = []
    context_ids = []
    doc_map = {}

    for item in doc_index:
        doc = item.get("doc", item)  # Support both {"doc": {...}} and direct doc format

        # Get document content - try various field names
        content = (
            doc.get("narrative") or
            doc.get("content") or
            doc.get("text") or
            ""
        )
        doc_id = doc.get("unit_id", str(id(doc)))

        if content:
            contexts.append(content)
            context_ids.append(doc_id)
            doc_map[doc_id] = doc

    if not contexts:
        logger.warning("No valid content found in document index")
        if return_traversal_stats:
            return [], {"total_docs_scored": 0, "scored_ids": set(), "returned_ids": set()}
        return []

    # Call cross-attention scoring
    scored_results = await cross_attention_score(query, contexts, context_ids)

    # Build result list
    results = []
    scored_ids = set()

    for doc_id, score in scored_results:
        if doc_id in doc_map:
            results.append((doc_map[doc_id], score))
            scored_ids.add(doc_id)

    # Sort and truncate (should already be sorted, but ensure)
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    if return_traversal_stats:
        returned_ids = {doc.get("unit_id", "") for doc, _ in results if doc.get("unit_id")}
        stats = {
            "total_docs_scored": len(contexts),
            "scored_ids": scored_ids,
            "returned_ids": returned_ids,
        }
        return results, stats

    return results


# =============================================================================
# RRF Fusion - Same as V3 (algorithm is retriever-agnostic)
# =============================================================================

def multi_cross_attention_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """Fuse multiple cross-attention search results using Reciprocal Rank Fusion (RRF).

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


# =============================================================================
# Utility Functions - Same as V3
# =============================================================================

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


# =============================================================================
# Smart Score Truncation - Reduce noise in Answer context
# =============================================================================

def smart_score_truncate(
    results: List[Tuple[dict, float]],
    min_results: int = 8,
    max_results: int = 40,
    score_ratio: float = 0.5,
    gap_threshold: float = 0.30,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Smart truncation based on score distribution to reduce noise.

    Strategy:
    1. Always keep at least min_results
    2. Filter by score_ratio threshold (score >= top_score * ratio)
    3. Detect score gaps and truncate at significant drops
    4. Never exceed max_results

    Note: All parameters should be passed from config (config.retrieval.agentic_v4.smart_truncate
    and config.retrieval.v4_type_retrieval_configs). The default values here are fallbacks only.

    Args:
        results: List of (doc, score) tuples, sorted by score descending.
        min_results: Minimum number of results to return.
        max_results: Maximum number of results to return.
        score_ratio: Minimum score as ratio of top score.
        gap_threshold: Significant drop threshold for gap detection.

    Returns:
        Tuple of:
        - truncated_results: Filtered results list
        - truncation_metadata: Dict with truncation stats
    """
    if not results:
        return [], {"reason": "empty_input", "original_count": 0, "final_count": 0}

    original_count = len(results)

    # Edge case: fewer results than min_results
    if original_count <= min_results:
        return results, {
            "reason": "below_min",
            "original_count": original_count,
            "final_count": original_count,
        }

    top_score = results[0][1]
    if top_score <= 0:
        # Invalid scores, return min_results
        return results[:min_results], {
            "reason": "invalid_scores",
            "original_count": original_count,
            "final_count": min_results,
        }

    # Step 1: Apply score ratio threshold
    score_threshold = top_score * score_ratio
    ratio_filtered = [r for r in results if r[1] >= score_threshold]

    # Step 2: Detect score gap (significant drop)
    gap_cutoff_idx = len(ratio_filtered)  # Default: no gap found
    for i in range(1, len(ratio_filtered)):
        prev_score = ratio_filtered[i - 1][1]
        curr_score = ratio_filtered[i][1]
        if prev_score > 0:
            drop_rate = (prev_score - curr_score) / prev_score
            if drop_rate > gap_threshold:
                gap_cutoff_idx = i
                break

    # Apply gap cutoff
    gap_filtered = ratio_filtered[:gap_cutoff_idx]

    # Step 3: Enforce min/max constraints
    if len(gap_filtered) < min_results:
        # Take min_results from original (sorted by score)
        final_results = results[:min_results]
        reason = "enforced_min"
    elif len(gap_filtered) > max_results:
        final_results = gap_filtered[:max_results]
        reason = "enforced_max"
    else:
        final_results = gap_filtered
        reason = "gap_detected" if gap_cutoff_idx < len(ratio_filtered) else "ratio_filtered"

    # Build metadata
    metadata = {
        "reason": reason,
        "original_count": original_count,
        "final_count": len(final_results),
        "top_score": top_score,
        "score_threshold": score_threshold,
        "ratio_filtered_count": len(ratio_filtered),
        "gap_cutoff_idx": gap_cutoff_idx if gap_cutoff_idx < len(ratio_filtered) else None,
    }

    # Log score range
    if final_results:
        metadata["final_score_range"] = (final_results[-1][1], final_results[0][1])

    logger.info(f"  [Truncate] {original_count} -> {len(final_results)} ({reason}), "
                f"score range: {metadata.get('final_score_range', 'N/A')}")

    return final_results, metadata
