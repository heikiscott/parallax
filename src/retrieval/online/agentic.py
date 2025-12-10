"""Agentic retrieval for online API services.

This module provides LLM-guided multi-round retrieval:
- Round 1: RRF hybrid search + Rerank
- LLM sufficiency check
- Round 2: Multi-query expansion (if needed)
- Final fusion and rerank

Extracted from agents.memory_manager.MemoryManager.retrieve_agentic
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from .lightweight import retrieve_lightweight

logger = logging.getLogger(__name__)


async def retrieve_agentic(
    query: str,
    user_id: str = None,
    group_id: str = None,
    time_range_days: int = 365,
    top_k: int = 20,
    llm_provider=None,
    agentic_config=None,
) -> Dict[str, Any]:
    """Agentic retrieval: LLM-guided multi-round intelligent retrieval.

    Flow: Round 1 (RRF) → Rerank → LLM check → Round 2 (multi-query) → Fusion → Rerank

    Args:
        query: User query
        user_id: User ID filter
        group_id: Group ID filter
        time_range_days: Time range in days
        top_k: Number of results to return
        llm_provider: LLM provider for sufficiency check and query generation
        agentic_config: AgenticConfig for retrieval parameters

    Returns:
        Dict with memories, count, and metadata
    """
    # Validate parameters
    if llm_provider is None:
        raise ValueError("llm_provider is required for agentic retrieval")

    # Import dependencies
    from agents.agentic_utils import (
        AgenticConfig,
        check_sufficiency,
        generate_multi_queries,
    )
    from retrieval.services.rerank import get_rerank_service

    # Use default config
    if agentic_config is None:
        agentic_config = AgenticConfig()
    config = agentic_config

    start_time = time.time()
    metadata = {
        "retrieval_mode": "agentic",
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "missing_info": None,
        "refined_queries": None,
        "round2_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    logger.info(f"{'='*60}")
    logger.info(f"Agentic Retrieval: {query[:60]}...")
    logger.info(f"{'='*60}")

    try:
        # ========== Round 1: RRF hybrid retrieval ==========
        logger.info("Round 1: RRF retrieval...")

        round1_result = await retrieve_lightweight(
            query=query,
            user_id=user_id,
            group_id=group_id,
            time_range_days=time_range_days,
            top_k=config.round1_top_n,  # 20
            retrieval_mode="rrf",
            data_source="episode",
        )

        round1_memories = round1_result.get("memories", [])
        metadata["round1_count"] = len(round1_memories)
        metadata["round1_latency_ms"] = round1_result.get("metadata", {}).get(
            "total_latency_ms", 0
        )

        logger.info(f"Round 1: Retrieved {len(round1_memories)} memories")

        if not round1_memories:
            logger.warning("Round 1 returned no results")
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000
            return {"memories": [], "count": 0, "metadata": metadata}

        # ========== Rerank Round 1 results → Top 5 ==========
        if config.use_reranker:
            logger.info("Reranking Top 20 to Top 5 for sufficiency check...")
            rerank_service = get_rerank_service()

            # Convert format for rerank
            candidates_for_rerank = [
                {
                    "index": i,
                    "narrative": mem.get("narrative", ""),
                    "summary": mem.get("summary", ""),
                    "subject": mem.get("subject", ""),
                    "score": mem.get("score", 0),
                }
                for i, mem in enumerate(round1_memories)
            ]

            reranked_hits = await rerank_service._rerank_all_hits(
                query, candidates_for_rerank, top_k=config.round1_rerank_top_n
            )

            # Extract Top 5 for LLM
            top5_for_llm = []
            for hit in reranked_hits[: config.round1_rerank_top_n]:
                idx = hit.get("index", 0)
                if 0 <= idx < len(round1_memories):
                    mem = round1_memories[idx]
                    top5_for_llm.append((mem, hit.get("relevance_score", 0)))

            metadata["round1_reranked_count"] = len(top5_for_llm)
            logger.info(f"Rerank: Got Top {len(top5_for_llm)} for sufficiency check")
        else:
            # No reranker, use top 5 directly
            top5_for_llm = [
                (mem, mem.get("score", 0))
                for mem in round1_memories[: config.round1_rerank_top_n]
            ]
            metadata["round1_reranked_count"] = len(top5_for_llm)
            logger.info("No Rerank: Using original Top 5")

        if not top5_for_llm:
            logger.warning("No results for sufficiency check")
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000
            return round1_result

        # ========== LLM sufficiency check ==========
        logger.info("LLM: Checking sufficiency on Top 5...")

        is_sufficient, reasoning, missing_info = await check_sufficiency(
            query=query,
            results=top5_for_llm,
            llm_provider=llm_provider,
            max_docs=config.round1_rerank_top_n,
        )

        metadata["is_sufficient"] = is_sufficient
        metadata["reasoning"] = reasoning
        metadata["missing_info"] = missing_info

        logger.info(
            f"LLM Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}"
        )
        logger.info(f"LLM Reasoning: {reasoning}")

        # ========== If sufficient: return Round 1 results ==========
        if is_sufficient:
            logger.info("Decision: Sufficient! Using Round 1 results")
            metadata["final_count"] = len(round1_memories)
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000

            round1_result["metadata"] = metadata
            logger.info(f"Complete: Latency {metadata['total_latency_ms']:.0f}ms")
            return round1_result

        # ========== Round 2: LLM generates multiple refined queries ==========
        metadata["is_multi_round"] = True
        logger.info("Decision: Insufficient, entering Round 2")

        if missing_info:
            logger.info(f"Missing: {', '.join(missing_info)}")

        if config.enable_multi_query:
            logger.info("LLM: Generating multiple refined queries...")

            refined_queries, query_strategy = await generate_multi_queries(
                original_query=query,
                results=top5_for_llm,
                missing_info=missing_info,
                llm_provider=llm_provider,
                max_docs=config.round1_rerank_top_n,
                num_queries=config.num_queries,
            )

            metadata["refined_queries"] = refined_queries
            metadata["query_strategy"] = query_strategy
            metadata["num_queries"] = len(refined_queries)

            logger.info(f"Generated {len(refined_queries)} queries")
            for i, q in enumerate(refined_queries, 1):
                logger.debug(f"  Query {i}: {q[:80]}...")
        else:
            # Single query mode
            refined_queries = [query]
            metadata["refined_queries"] = refined_queries
            metadata["num_queries"] = 1

        # ========== Round 2: Parallel multi-query retrieval ==========
        logger.info(
            f"Round 2: Executing {len(refined_queries)} queries in parallel..."
        )

        # Parallel retrieve_lightweight calls
        round2_tasks = [
            retrieve_lightweight(
                query=q,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=config.round2_per_query_top_n,  # 50 per query
                retrieval_mode="rrf",
                data_source="episode",
            )
            for q in refined_queries
        ]

        round2_results_list = await asyncio.gather(*round2_tasks, return_exceptions=True)

        # Collect results from all queries
        all_round2_memories = []
        for i, result in enumerate(round2_results_list, 1):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
                continue

            memories = result.get("memories", [])
            if memories:
                all_round2_memories.extend(memories)
                logger.debug(f"Query {i}: Retrieved {len(memories)} memories")

        logger.info(
            f"Round 2: Total retrieved {len(all_round2_memories)} memories before dedup"
        )

        # ========== Deduplication and fusion ==========
        logger.info("Merge: Deduplicating and combining Round 1 + Round 2...")

        # Deduplicate by episode_id
        round1_episode_ids = {mem.get("episode_id") for mem in round1_memories}
        round2_unique = [
            mem
            for mem in all_round2_memories
            if mem.get("episode_id") not in round1_episode_ids
        ]

        # Merge: Round 1 (20) + deduplicated Round 2 (up to total 40)
        combined_memories = round1_memories.copy()
        needed_from_round2 = config.combined_total - len(combined_memories)
        combined_memories.extend(round2_unique[:needed_from_round2])

        metadata["round2_count"] = len(round2_unique[:needed_from_round2])
        logger.info(
            f"Merge: Round1={len(round1_memories)}, "
            f"Round2_unique={len(round2_unique[:needed_from_round2])}, "
            f"Total={len(combined_memories)}"
        )

        # ========== Final Rerank ==========
        if config.use_reranker and len(combined_memories) > 0:
            logger.info(f"Rerank: Reranking {len(combined_memories)} memories...")

            rerank_service = get_rerank_service()

            # Convert format
            candidates_for_rerank = [
                {
                    "index": i,
                    "narrative": mem.get("narrative", ""),
                    "summary": mem.get("summary", ""),
                    "subject": mem.get("subject", ""),
                    "score": mem.get("score", 0),
                }
                for i, mem in enumerate(combined_memories)
            ]

            reranked_hits = await rerank_service._rerank_all_hits(
                query,  # Use original query
                candidates_for_rerank,
                top_k=config.final_top_n,
            )

            # Extract final Top 20
            final_memories = []
            for hit in reranked_hits[: config.final_top_n]:
                idx = hit.get("index", 0)
                if 0 <= idx < len(combined_memories):
                    mem = combined_memories[idx].copy()
                    mem["score"] = hit.get("relevance_score", mem.get("score", 0))
                    final_memories.append(mem)

            logger.info(f"Rerank: Final Top {len(final_memories)} selected")
        else:
            # No Reranker, return Top N directly
            final_memories = combined_memories[: config.final_top_n]
            logger.info(f"No Rerank: Returning Top {len(final_memories)}")

        metadata["final_count"] = len(final_memories)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000

        logger.info(
            f"Complete: Final {len(final_memories)} memories | "
            f"Latency {metadata['total_latency_ms']:.0f}ms"
        )
        logger.info(f"{'='*60}\n")

        return {
            "memories": final_memories,
            "count": len(final_memories),
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Agentic retrieval failed: {e}", exc_info=True)

        # Fallback to lightweight
        logger.warning("Falling back to lightweight retrieval")

        fallback_result = await retrieve_lightweight(
            query=query,
            user_id=user_id,
            group_id=group_id,
            time_range_days=time_range_days,
            top_k=top_k,
            retrieval_mode="rrf",
            data_source="episode",
        )

        fallback_result["metadata"]["retrieval_mode"] = "agentic_fallback"
        fallback_result["metadata"]["fallback_reason"] = str(e)

        return fallback_result
