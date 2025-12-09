"""Agentic multi-round retrieval pipeline with LLM guidance.

This pipeline implements LLM-guided multi-round retrieval:
1. Round 1: Hybrid retrieval -> Top K -> Rerank -> Top N -> LLM sufficiency check
2. If sufficient: Return original Top K (before rerank)
3. If insufficient:
   - LLM generates improved query(s)
   - Round 2: Retrieve and merge results
   - Rerank merged results -> Return final results

Advantages:
- Adaptive: Automatically decides if additional retrieval is needed
- Accurate: LLM-guided query improvement
- Context-aware: Uses cluster expansion for related information

Use cases:
- Complex queries requiring multiple perspectives
- Queries where initial results may be incomplete
- High-accuracy requirements
"""

import time
import asyncio
import logging
from typing import List, Tuple, Optional, Any, Set

from .search_utils import (
    hybrid_search_with_rrf,
    multi_rrf_fusion,
)
from .llm_utils import (
    check_sufficiency,
    generate_refined_query,
    generate_multi_queries,
)
from .rerank import reranker_search

logger = logging.getLogger(__name__)


def _log_ids(prefix: str, docs: List[Tuple[dict, float]], limit: int = 20):
    """Log a short list of unit_ids for debugging."""
    ids = [d.get("unit_id", "") for d, _ in docs if d.get("unit_id")]
    if not ids:
        logger.info(f"  {prefix}: (no unit_ids)")
        return
    short = ids[:limit]
    suffix = " ..." if len(ids) > limit else ""
    logger.info(f"  {prefix}: {', '.join(short)}{suffix}")


def _build_origin_map(round1_ids: Set[str], round2_ids: Set[str], cluster_ids: Set[str]) -> dict:
    """Build origin mapping for unit_ids."""
    origin_map = {}
    for uid in round1_ids:
        if uid:
            origin_map[uid] = "round1"
    for uid in round2_ids:
        if uid and uid not in origin_map:
            origin_map[uid] = "round2"
    for uid in cluster_ids:
        if uid and uid not in origin_map:
            origin_map[uid] = "cluster"
    return origin_map


async def _apply_cluster_expansion(
    final_results: List[Tuple[dict, float]],
    metadata: dict,
    cluster_index: Optional[Any],
    config: Any,
    docs: List[dict],
    query: str,
    llm_provider: Optional[Any] = None,
    llm_config: Optional[dict] = None,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Apply cluster expansion to retrieval results.

    Args:
        final_results: Original retrieval results
        metadata: Retrieval metadata
        cluster_index: Cluster index (optional)
        config: Experiment configuration
        docs: All documents list
        query: Original query
        llm_provider: LLM Provider (for cluster_rerank strategy)
        llm_config: LLM config (for cluster_rerank strategy)

    Returns:
        (expanded_results, updated_metadata)
    """
    from memory.group_event_cluster import (
        GroupEventClusterRetrievalConfig,
        expand_with_cluster,
    )

    # Check if cluster expansion is enabled
    cluster_retrieval_cfg = getattr(config, 'group_event_cluster_retrieval_config', {})
    enable_expansion = cluster_retrieval_cfg.get('enable_group_event_cluster_retrieval', False)

    if not enable_expansion or cluster_index is None:
        logger.debug(f"  [Cluster] Expansion disabled or no cluster index")
        return final_results, metadata

    # Create config object
    cluster_config = GroupEventClusterRetrievalConfig.from_dict(cluster_retrieval_cfg)

    # Build all_docs_map (unit_id -> doc)
    all_docs_map = {doc.get("unit_id"): doc for doc in docs if doc.get("unit_id")}

    logger.info(f"  [Cluster Expansion] Strategy: {cluster_config.expansion_strategy}")
    logger.info(f"  [Cluster Expansion] Input: {len(final_results)} results, {len(cluster_index.clusters)} clusters")

    # Execute cluster expansion
    expanded_results, expansion_metadata = await expand_with_cluster(
        original_results=final_results,
        cluster_index=cluster_index,
        config=cluster_config,
        all_docs_map=all_docs_map,
        query=query,
        llm_provider=llm_provider,
        llm_config=llm_config,
    )

    # Log expansion statistics
    strategy = expansion_metadata.get("strategy", "unknown")
    if strategy == "cluster_rerank":
        final_count = expansion_metadata.get("final_count", 0)
        clusters_selected = expansion_metadata.get("clusters_selected", [])
        logger.info(f"  [Cluster Rerank] Selected {len(clusters_selected)} clusters -> {final_count} MemUnits")
    else:
        expanded_count = expansion_metadata.get("expanded_count", 0)
        clusters_hit = expansion_metadata.get("clusters_hit", [])
        logger.info(f"  [Cluster Expansion] Expanded {expanded_count} docs from {len(clusters_hit)} clusters")

    # Update metadata
    metadata["cluster_expansion"] = expansion_metadata

    # Optional: Full rerank after expansion
    needs_rerank = (
        cluster_config.expansion_strategy == "replace_rerank"
        or cluster_config.rerank_after_expansion
    )
    if needs_rerank and getattr(config, 'use_reranker', True) and expanded_results:
        try:
            expanded_results = await reranker_search(
                query=query,
                results=expanded_results,
                top_n=cluster_config.rerank_top_n_after_expansion,
                reranker_instruction=getattr(config, 'reranker_instruction', None),
                batch_size=getattr(config, 'reranker_batch_size', 10),
                max_retries=getattr(config, 'reranker_max_retries', 3),
                retry_delay=getattr(config, 'reranker_retry_delay', 2.0),
                timeout=getattr(config, 'reranker_timeout', 30.0),
                fallback_threshold=getattr(config, 'reranker_fallback_threshold', 0.3),
                config=config,
            )
            metadata.setdefault("cluster_expansion", {})["reranked"] = True
        except Exception as e:
            logger.warning(f"  [Cluster] Rerank after expansion failed: {e}")
            metadata.setdefault("cluster_expansion", {})["reranked"] = False

    return expanded_results, metadata


async def agentic_retrieval(
    query: str,
    config: Any,
    llm_provider: Any,
    llm_config: dict,
    emb_index: Any,
    bm25: Any,
    docs: List[dict],
    cluster_index: Optional[Any] = None,
    enable_traversal_stats: bool = False,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Agentic multi-round retrieval with LLM guidance.

    Flow:
    1. Round 1: Hybrid search -> Top 20 -> Rerank -> Top 5 -> LLM sufficiency check
    2. If sufficient: Return original Top 20 (before rerank)
    3. If insufficient:
       - LLM generates improved query(s)
       - Round 2: Retrieve and merge to 40 documents
       - Rerank 40 -> Return final results

    Args:
        query: User query
        config: Experiment configuration with retrieval parameters
        llm_provider: LLM Provider instance
        llm_config: LLM configuration dict
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list
        cluster_index: Optional cluster index for expansion
        enable_traversal_stats: Enable detailed traversal statistics

    Returns:
        (final_results, metadata)
    """
    start_time = time.time()

    metadata = {
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "refined_query": None,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    # Traversal statistics
    traversal_stats = {
        "total_memunits": len(emb_index),
        "round1_emb_scored_ids": set(),
        "round1_bm25_scored_ids": set(),
        "round1_returned_ids": set(),
        "round1_rerank_input_ids": set(),
        "round1_rerank_output_ids": set(),
        "round2_queries": [],
        "round2_all_scored_ids": set(),
        "round2_returned_ids": set(),
        "round2_rerank_input_ids": set(),
        "round2_rerank_output_ids": set(),
        "all_reranked_ids": set(),
    }

    logger.info(f"{'='*60}")
    logger.info(f"Agentic Retrieval: {query[:60]}...")
    logger.info(f"{'='*60}")

    # Get config values with defaults
    hybrid_emb_candidates = getattr(config, 'hybrid_emb_candidates', 50)
    hybrid_bm25_candidates = getattr(config, 'hybrid_bm25_candidates', 50)
    hybrid_rrf_k = getattr(config, 'hybrid_rrf_k', 60)
    use_reranker = getattr(config, 'use_reranker', True)
    use_multi_query = getattr(config, 'use_multi_query', True)

    # ========== Round 1: Hybrid search Top 20 ==========
    logger.info(f"  [Round 1] Hybrid search for Top 20...")

    if enable_traversal_stats:
        round1_top20, r1_stats = await hybrid_search_with_rrf(
            query=query,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=20,
            emb_candidates=hybrid_emb_candidates,
            bm25_candidates=hybrid_bm25_candidates,
            rrf_k=hybrid_rrf_k,
            return_traversal_stats=True,
        )
        traversal_stats["round1_emb_scored_ids"] = set(r1_stats.get("emb_scored_ids", []))
        traversal_stats["round1_bm25_scored_ids"] = set(r1_stats.get("bm25_scored_ids", []))
        traversal_stats["round1_returned_ids"] = set(r1_stats.get("fused_ids", []))
    else:
        round1_top20 = await hybrid_search_with_rrf(
            query=query,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=20,
            emb_candidates=hybrid_emb_candidates,
            bm25_candidates=hybrid_bm25_candidates,
            rrf_k=hybrid_rrf_k,
        )

    metadata["round1_count"] = len(round1_top20)
    logger.info(f"  [Round 1] Retrieved {len(round1_top20)} documents")
    _log_ids("[Round 1] Unit IDs", round1_top20)

    if not round1_top20:
        logger.warning(f"  [Warning] No results from Round 1")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # ========== Rerank Top 20 -> Top 5 for Sufficiency Check ==========
    logger.info(f"  [Rerank] Reranking Top 20 to get Top 5 for sufficiency check...")

    if use_reranker:
        reranked_top5 = await reranker_search(
            query=query,
            results=round1_top20,
            top_n=5,
            reranker_instruction=getattr(config, 'reranker_instruction', None),
            batch_size=getattr(config, 'reranker_batch_size', 10),
            max_retries=getattr(config, 'reranker_max_retries', 3),
            retry_delay=getattr(config, 'reranker_retry_delay', 2.0),
            timeout=getattr(config, 'reranker_timeout', 30.0),
            fallback_threshold=getattr(config, 'reranker_fallback_threshold', 0.3),
            config=config,
        )
        metadata["round1_reranked_count"] = len(reranked_top5)

        if enable_traversal_stats:
            input_ids = set(doc.get("unit_id", "") for doc, _ in round1_top20)
            output_ids = set(doc.get("unit_id", "") for doc, _ in reranked_top5)
            traversal_stats["round1_rerank_input_ids"] = input_ids
            traversal_stats["round1_rerank_output_ids"] = output_ids
            traversal_stats["all_reranked_ids"].update(input_ids)
    else:
        reranked_top5 = round1_top20[:5]
        metadata["round1_reranked_count"] = 5

    if not reranked_top5:
        logger.warning(f"  [Warning] Reranking failed, falling back to original Top 20")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_top20, metadata

    # ========== LLM Sufficiency Check ==========
    logger.info(f"  [LLM] Checking sufficiency on Top 5...")

    is_sufficient, reasoning, missing_info = await check_sufficiency(
        query=query,
        results=reranked_top5,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=5,
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning

    logger.info(f"  [LLM] Result: {'Sufficient' if is_sufficient else 'Insufficient'}")

    # ========== If sufficient: Return original Round 1 Top 20 ==========
    if is_sufficient:
        logger.info(f"  [Decision] Sufficient! Using original Round 1 Top 20 results")

        final_results = round1_top20

        # Cluster expansion
        final_results, metadata = await _apply_cluster_expansion(
            final_results=final_results,
            metadata=metadata,
            cluster_index=cluster_index,
            config=config,
            docs=docs,
            query=query,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )

        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000

        cluster_ids = set(metadata.get("cluster_expansion", {}).get("expanded_unit_ids", []))
        round1_ids = {doc.get("unit_id", "") for doc, _ in round1_top20}
        metadata["origin_map"] = _build_origin_map(round1_ids, set(), cluster_ids)

        if enable_traversal_stats:
            _record_traversal_stats(metadata, traversal_stats, final_results, is_multi_round=False)

        logger.info(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== If insufficient: Enter Round 2 ==========
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    logger.info(f"  [Decision] Insufficient, entering Round 2")

    if use_multi_query:
        # Generate multiple refined queries
        logger.info(f"  [LLM] Generating multiple refined queries...")

        refined_queries, query_strategy = await generate_multi_queries(
            original_query=query,
            results=reranked_top5,
            missing_info=missing_info,
            llm_provider=llm_provider,
            llm_config=llm_config,
            max_docs=5,
            num_queries=3,
        )

        metadata["refined_queries"] = refined_queries
        metadata["query_strategy"] = query_strategy
        metadata["num_queries"] = len(refined_queries)

        if enable_traversal_stats:
            traversal_stats["round2_queries"] = refined_queries

        # Parallel multi-query retrieval
        logger.info(f"  [Round 2] Executing {len(refined_queries)} queries in parallel...")

        multi_query_tasks = [
            hybrid_search_with_rrf(
                query=q,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=50,
                emb_candidates=hybrid_emb_candidates,
                bm25_candidates=hybrid_bm25_candidates,
                rrf_k=hybrid_rrf_k,
                return_traversal_stats=enable_traversal_stats,
            )
            for q in refined_queries
        ]

        raw_results = await asyncio.gather(*multi_query_tasks)

        # Parse results
        if enable_traversal_stats:
            multi_query_results = []
            for result in raw_results:
                if isinstance(result, tuple):
                    docs_result, stats = result
                    multi_query_results.append(docs_result)
                    traversal_stats["round2_all_scored_ids"].update(stats.get("emb_scored_ids", []))
                    traversal_stats["round2_all_scored_ids"].update(stats.get("bm25_scored_ids", []))
                else:
                    multi_query_results.append(result)
        else:
            multi_query_results = raw_results

        # RRF fusion of multi-query results
        logger.info(f"  [Multi-RRF] Fusing results from {len(refined_queries)} queries...")

        round2_results = multi_rrf_fusion(
            results_list=multi_query_results,
            k=hybrid_rrf_k,
        )
        round2_results = round2_results[:40]

        metadata["round2_count"] = len(round2_results)
        metadata["multi_query_total_docs"] = sum(len(r) for r in multi_query_results)

    else:
        # Single refined query (legacy mode)
        logger.info(f"  [LLM] Generating single refined query...")

        refined_query = await generate_refined_query(
            original_query=query,
            results=reranked_top5,
            missing_info=missing_info,
            llm_provider=llm_provider,
            llm_config=llm_config,
            max_docs=5,
        )

        metadata["refined_query"] = refined_query

        round2_results = await hybrid_search_with_rrf(
            query=refined_query,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=40,
            emb_candidates=hybrid_emb_candidates,
            bm25_candidates=hybrid_bm25_candidates,
            rrf_k=hybrid_rrf_k,
        )

        metadata["round2_count"] = len(round2_results)

    # ========== Merge Round 1 and Round 2 ==========
    logger.info(f"  [Merge] Combining Round 1 and Round 2 to ensure 40 documents...")

    round1_ids = {doc.get("unit_id", id(doc)) for doc, _ in round1_top20}
    round2_unique = [
        (doc, score) for doc, score in round2_results
        if doc.get("unit_id", id(doc)) not in round1_ids
    ]

    combined_results = round1_top20.copy()
    needed_from_round2 = 40 - len(combined_results)
    round2_slice = round2_unique[:needed_from_round2]
    combined_results.extend(round2_slice)

    logger.info(f"  [Merge] Combined total: {len(combined_results)} documents")

    # ========== Final Rerank ==========
    if use_reranker and len(combined_results) > 0:
        logger.info(f"  [Rerank] Reranking {len(combined_results)} documents...")

        final_results = await reranker_search(
            query=query,
            results=combined_results,
            top_n=20,
            reranker_instruction=getattr(config, 'reranker_instruction', None),
            batch_size=getattr(config, 'reranker_batch_size', 10),
            max_retries=getattr(config, 'reranker_max_retries', 3),
            retry_delay=getattr(config, 'reranker_retry_delay', 2.0),
            timeout=getattr(config, 'reranker_timeout', 30.0),
            fallback_threshold=getattr(config, 'reranker_fallback_threshold', 0.3),
            config=config,
        )
    else:
        final_results = combined_results[:20]

    # ========== Cluster Expansion ==========
    final_results, metadata = await _apply_cluster_expansion(
        final_results=final_results,
        metadata=metadata,
        cluster_index=cluster_index,
        config=config,
        docs=docs,
        query=query,
        llm_provider=llm_provider,
        llm_config=llm_config,
    )

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    cluster_ids = set(metadata.get("cluster_expansion", {}).get("expanded_unit_ids", []))
    round1_ids_set = {doc.get("unit_id", "") for doc, _ in round1_top20}
    round2_ids_set = {doc.get("unit_id", "") for doc, _ in round2_slice}
    metadata["origin_map"] = _build_origin_map(round1_ids_set, round2_ids_set, cluster_ids)

    if enable_traversal_stats:
        traversal_stats["round2_rerank_input_ids"] = set(
            doc.get("unit_id", "") for doc, _ in combined_results
        )
        traversal_stats["round2_rerank_output_ids"] = set(
            doc.get("unit_id", "") for doc, _ in final_results
        )
        traversal_stats["all_reranked_ids"].update(traversal_stats["round2_rerank_input_ids"])
        _record_traversal_stats(metadata, traversal_stats, final_results, is_multi_round=True)

    logger.info(f"  [Complete] Final: {len(final_results)} docs | Latency: {metadata['total_latency_ms']:.0f}ms")

    return final_results, metadata


def _record_traversal_stats(
    metadata: dict,
    traversal_stats: dict,
    final_results: List[Tuple[dict, float]],
    is_multi_round: bool,
) -> None:
    """Record traversal statistics to metadata."""
    final_ids = set(doc.get("unit_id", "") for doc, _ in final_results)
    total_mu = traversal_stats["total_memunits"]
    rerank_count = len(traversal_stats["all_reranked_ids"])

    if is_multi_round:
        r1_rerank_count = len(traversal_stats["round1_rerank_input_ids"])
        r2_rerank_count = len(traversal_stats["round2_rerank_input_ids"])
        r1_emb = len(traversal_stats["round1_emb_scored_ids"])
        r1_bm25 = len(traversal_stats["round1_bm25_scored_ids"])
        r2_scored = len(traversal_stats["round2_all_scored_ids"])

        metadata["traversal_stats"] = {
            "total_memunits": total_mu,
            "round1_emb_scored": r1_emb,
            "round1_bm25_scored": r1_bm25,
            "round2_all_scored": r2_scored,
            "round1_rerank_count": r1_rerank_count,
            "round2_rerank_count": r2_rerank_count,
            "total_reranked": rerank_count,
            "rerank_coverage_percent": round(rerank_count / total_mu * 100, 1) if total_mu else 0,
            "final_returned": len(final_ids),
            "is_multi_round": True,
        }
    else:
        metadata["traversal_stats"] = {
            "total_memunits": total_mu,
            "emb_scored": len(traversal_stats["round1_emb_scored_ids"]),
            "bm25_scored": len(traversal_stats["round1_bm25_scored_ids"]),
            "total_reranked": rerank_count,
            "rerank_coverage_percent": round(rerank_count / total_mu * 100, 1) if total_mu else 0,
            "final_returned": len(final_ids),
            "is_multi_round": False,
        }
