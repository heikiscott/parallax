"""Agentic Retrieval V2 - Type-Aware Multi-Query at Round 1.

This module implements an enhanced agentic retrieval pipeline that generates
multi-query variations at Round 1 based on question type classification.

Key differences from agentic.py (V1):
- Round 1 generates multi-queries BEFORE retrieval (for complex questions)
- Simple questions (ATTRIBUTE_LOCATION, ATTRIBUTE_IDENTITY) skip multi-query
- Question-type-aware prompt templates for better query generation
- Round 2 uses missing_info-based query generation (same as V1)

Design Philosophy:
- DO NOT modify existing agentic.py - this is a separate implementation
- Reuse public functions from search_utils.py, llm_utils.py, rerank.py
- Configuration-driven switching between V1 and V2

Flow:
```
Query
  │
  ▼
Question Classification
  │
  ├─── Simple (ATTRIBUTE_*) + High Confidence
  │         │
  │         ▼
  │    Single Query Hybrid → Top 20 → Rerank Top 10 → Sufficiency Check
  │
  └─── Complex (EVENT_*, COUNTING, REASONING_*)
            │
            ▼
       Type-Aware Multi-Query (2-3) → Parallel Hybrid → RRF Fusion Top 20
            │
            ▼
       Rerank Top 10 → Sufficiency Check
            │
            ├─── Sufficient → Return Top 20 (+ Cluster Expansion)
            │
            └─── Insufficient → Round 2 (missing_info-based queries)
                      │
                      ▼
                 Merge + Final Rerank → Top 20
```
"""

import time
import asyncio
import logging
import json
from typing import List, Tuple, Optional, Any, Set

from .search_utils import (
    hybrid_search_with_rrf,
    multi_rrf_fusion,
)
from .llm_utils import (
    check_sufficiency,
    generate_multi_queries,
)
from .rerank import reranker_search

from retrieval.classification.question_classifier import (
    QuestionClassifier,
    QuestionType,
    ClassificationResult,
)
from prompts.memory.en.eval.search.type_aware_multi_query_prompts import (
    should_use_multi_query,
    get_prompt_for_type,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Config Helpers (copied from agentic.py for independence)
# =============================================================================

def _get_reranker_config(config: Any, key: str, default: Any = None) -> Any:
    """Get reranker config value from nested config.reranker.xxx structure."""
    reranker_cfg = getattr(config, 'reranker', None)
    if reranker_cfg is None:
        raise ValueError(
            "Missing required config: 'reranker'. "
            "Please add reranker section to your config file."
        )
    value = getattr(reranker_cfg, key, None)
    if value is None and default is None:
        raise ValueError(
            f"Missing required config key: 'reranker.{key}'. "
            "Please set this value in your config file."
        )
    return value if value is not None else default


def _get_v2_config(config: Any, key: str, default: Any) -> Any:
    """Get V2-specific config value from config.retrieval.agentic_v2.xxx."""
    retrieval_cfg = getattr(config, 'retrieval', None)
    if retrieval_cfg is None:
        return default
    v2_cfg = getattr(retrieval_cfg, 'agentic_v2', None)
    if v2_cfg is None:
        return default
    return getattr(v2_cfg, key, default)


def _get_type_retrieval_config(config: Any, question_type: QuestionType) -> Optional[dict]:
    """Get type-specific retrieval config from config.retrieval.type_retrieval_configs.

    Args:
        config: Experiment configuration
        question_type: Classified question type

    Returns:
        Dict with type-specific params or None if not configured.
        Keys: round1_per_query_top_n, round1_fusion_top_n, round1_rerank_top_n,
              round2_per_query_top_n, round2_fusion_top_n, merge_budget, final_rerank_top_n
    """
    retrieval_cfg = getattr(config, 'retrieval', None)
    if retrieval_cfg is None:
        return None

    type_configs = getattr(retrieval_cfg, 'type_retrieval_configs', None)
    if type_configs is None:
        return None

    # Try to get config for this specific type (e.g., "reasoning_inference")
    type_key = question_type.value
    type_config = getattr(type_configs, type_key, None)

    # Fallback to default config
    if type_config is None:
        type_config = getattr(type_configs, 'default', None)

    if type_config is None:
        return None

    # Convert config object to dict
    if hasattr(type_config, 'to_dict'):
        return type_config.to_dict()
    elif hasattr(type_config, '_data'):
        return type_config._data
    elif hasattr(type_config, '__dict__'):
        return {k: v for k, v in type_config.__dict__.items() if not k.startswith('_')}
    else:
        # Assume it's already dict-like
        return dict(type_config)


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


# =============================================================================
# Type-Aware Multi-Query Generation (NEW in V2)
# =============================================================================

async def generate_type_aware_multi_queries(
    original_query: str,
    question_type: QuestionType,
    llm_provider: Any,
    llm_config: dict,
    num_queries: int = 3,
) -> Tuple[List[str], str]:
    """Generate multi-query variations based on question type at Round 1.

    Unlike llm_utils.generate_multi_queries which requires retrieved docs and missing_info,
    this function generates queries purely based on the original question and its type.

    Args:
        original_query: Original user query
        question_type: Classified question type
        llm_provider: LLM provider instance
        llm_config: LLM configuration dict
        num_queries: Target number of queries to generate (2-3)

    Returns:
        (list of queries, reasoning string)
    """
    prompt_template = get_prompt_for_type(question_type)
    prompt = prompt_template.format(original_query=original_query)

    try:
        response = await llm_provider.chat.completions.create(
            model=llm_config.get("model", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 1024),
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json") or line.startswith("```"):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        result = json.loads(content)
        queries = result.get("queries", [])
        reasoning = result.get("reasoning", "")

        # Validate queries
        if not queries or not isinstance(queries, list):
            logger.warning(f"  [TypeMQ] Invalid response format, falling back to original query")
            return [original_query], "Failed to generate multi-queries, using original"

        # Limit to num_queries
        queries = queries[:num_queries]

        # Always include original query if we have room
        if len(queries) < num_queries and original_query not in queries:
            queries.append(original_query)

        logger.info(f"  [TypeMQ] Generated {len(queries)} queries for type {question_type.value}")
        for i, q in enumerate(queries):
            logger.debug(f"    Query {i+1}: {q[:80]}...")

        return queries, reasoning

    except json.JSONDecodeError as e:
        logger.warning(f"  [TypeMQ] JSON parse error: {e}, using original query")
        return [original_query], f"JSON parse error: {e}"
    except Exception as e:
        logger.warning(f"  [TypeMQ] Query generation failed: {e}, using original query")
        return [original_query], f"Generation failed: {e}"


# =============================================================================
# Cluster Expansion (reused from agentic.py structure)
# =============================================================================

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

    This is copied from agentic.py for independence. The logic is identical.
    """
    from memory.group_event_cluster import (
        GroupEventClusterRetrievalConfig,
        expand_with_cluster,
    )

    cluster_retrieval_cfg = getattr(config, 'group_event_cluster_retrieval', None)
    if cluster_retrieval_cfg is None:
        raise ValueError(
            "Missing required config: 'group_event_cluster_retrieval'. "
            "Please add group_event_cluster_retrieval section to your config file."
        )

    if hasattr(cluster_retrieval_cfg, 'to_dict'):
        cluster_retrieval_cfg = cluster_retrieval_cfg.to_dict()
    elif hasattr(cluster_retrieval_cfg, '_data'):
        cluster_retrieval_cfg = cluster_retrieval_cfg._data

    if 'enabled' not in cluster_retrieval_cfg:
        raise ValueError(
            "Missing required config key: 'group_event_cluster_retrieval.enabled'. "
            "Please set enabled: true/false in your config file."
        )
    enable_expansion = cluster_retrieval_cfg['enabled']

    if not enable_expansion or cluster_index is None:
        logger.debug(f"  [Cluster] Expansion disabled or no cluster index")
        return final_results, metadata

    try:
        retrieval_config = GroupEventClusterRetrievalConfig(**cluster_retrieval_cfg)
    except Exception as e:
        logger.warning(f"  [Cluster] Config init failed: {e}")
        return final_results, metadata

    try:
        expanded_results, expansion_meta = await expand_with_cluster(
            query=query,
            base_results=final_results,
            cluster_index=cluster_index,
            retrieval_config=retrieval_config,
            all_docs=docs,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )

        metadata["cluster_expansion"] = expansion_meta
        logger.info(f"  [Cluster] Expanded: {len(final_results)} -> {len(expanded_results)} docs")
        _log_ids("[Cluster] Expanded IDs", expanded_results)

        if retrieval_config.rerank_after_expansion and len(expanded_results) > len(final_results):
            logger.info(f"  [Cluster] Reranking after expansion...")
            expanded_results = await reranker_search(
                query=query,
                results=expanded_results,
                top_n=retrieval_config.rerank_top_n_after_expansion,
                reranker_instruction=_get_reranker_config(config, 'instruction', None),
                batch_size=_get_reranker_config(config, 'batch_size', 20),
                max_retries=_get_reranker_config(config, 'max_retries', 10),
                retry_delay=_get_reranker_config(config, 'retry_delay', 0.8),
                timeout=_get_reranker_config(config, 'timeout', 60.0),
                fallback_threshold=_get_reranker_config(config, 'fallback_threshold', 0.3),
                config=config,
            )
            metadata.setdefault("cluster_expansion", {})["reranked"] = True

        return expanded_results, metadata

    except Exception as e:
        logger.warning(f"  [Cluster] Expansion failed: {e}")
        return final_results, metadata


# =============================================================================
# Main Entry Point: agentic_retrieval_v2
# =============================================================================

async def agentic_retrieval_v2(
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
    """Agentic Retrieval V2 - Type-aware multi-query at Round 1.

    Key difference from V1: Multi-query generation happens at Round 1 start
    for complex questions, improving first-round recall.

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
        "version": "v2",
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "final_count": 0,
        "total_latency_ms": 0.0,
        "question_type": None,
        "used_multi_query_round1": False,
    }

    # Traversal statistics for debugging
    traversal_stats = {
        "total_memunits": len(emb_index) if emb_index else 0,
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
    logger.info(f"Agentic Retrieval V2: {query[:60]}...")
    logger.info(f"{'='*60}")

    # ========== Step 1: Question Classification ==========
    classifier = QuestionClassifier()
    classification: ClassificationResult = classifier.classify(query)

    metadata["question_type"] = classification.question_type.value
    metadata["classification_confidence"] = classification.confidence
    metadata["classification_reasoning"] = classification.reasoning

    logger.info(f"  [Classify] Type: {classification.question_type.value} "
                f"(conf={classification.confidence:.2f})")

    # ========== Step 2: Load Type-Specific Retrieval Config ==========
    type_config = _get_type_retrieval_config(config, classification.question_type)

    # Get config values - support both flat and nested config structures
    # Nested: config.retrieval.hybrid.emb_candidates (from YAML)
    # Flat: config.hybrid_emb_candidates (legacy)
    retrieval_cfg = getattr(config, 'retrieval', None)
    hybrid_cfg = getattr(retrieval_cfg, 'hybrid', None) if retrieval_cfg else None

    if hybrid_cfg:
        hybrid_emb_candidates = getattr(hybrid_cfg, 'emb_candidates', 50)
        hybrid_bm25_candidates = getattr(hybrid_cfg, 'bm25_candidates', 50)
        hybrid_rrf_k = getattr(hybrid_cfg, 'rrf_k', 40)
    else:
        # Fallback to flat config (legacy support)
        hybrid_emb_candidates = getattr(config, 'hybrid_emb_candidates', 50)
        hybrid_bm25_candidates = getattr(config, 'hybrid_bm25_candidates', 50)
        hybrid_rrf_k = getattr(config, 'hybrid_rrf_k', 40)

    use_reranker = getattr(config, 'use_reranker', True)
    if retrieval_cfg:
        use_reranker = getattr(retrieval_cfg, 'use_reranker', use_reranker)

    num_queries = _get_v2_config(config, 'num_queries', 3)

    # ========== Apply Type-Specific Config (or use defaults) ==========
    if type_config:
        # Use type-specific config from YAML
        round1_per_query_top_n = type_config.get('round1_per_query_top_n', 20)
        round1_fusion_top_n = type_config.get('round1_fusion_top_n', 20)
        round1_rerank_top_n = type_config.get('round1_rerank_top_n', 10)
        round2_per_query_top_n = type_config.get('round2_per_query_top_n', 30)
        round2_fusion_top_n = type_config.get('round2_fusion_top_n', 30)
        merge_budget = type_config.get('merge_budget', 35)
        final_rerank_top_n = type_config.get('final_rerank_top_n', 20)

        logger.info(f"  [TypeConfig] {classification.question_type.value}: "
                    f"R1(per={round1_per_query_top_n}, fusion={round1_fusion_top_n}, rerank={round1_rerank_top_n}) "
                    f"R2(per={round2_per_query_top_n}, fusion={round2_fusion_top_n}) "
                    f"merge={merge_budget}, final={final_rerank_top_n}")

        metadata["type_config"] = {
            "type": classification.question_type.value,
            "round1_per_query_top_n": round1_per_query_top_n,
            "round1_fusion_top_n": round1_fusion_top_n,
            "round1_rerank_top_n": round1_rerank_top_n,
            "round2_per_query_top_n": round2_per_query_top_n,
            "round2_fusion_top_n": round2_fusion_top_n,
            "merge_budget": merge_budget,
            "final_rerank_top_n": final_rerank_top_n,
        }
    else:
        # Fallback to agentic_v2 default config
        round1_per_query_top_n = _get_v2_config(config, 'round1_per_query_top_n', 20)
        round1_fusion_top_n = _get_v2_config(config, 'round1_fusion_top_n', 20)
        round1_rerank_top_n = _get_v2_config(config, 'round1_rerank_top_n', 10)
        round2_per_query_top_n = _get_v2_config(config, 'round2_per_query_top_n', 30)
        round2_fusion_top_n = _get_v2_config(config, 'round2_fusion_top_n', 30)
        merge_budget = round2_fusion_top_n  # fallback: use round2_fusion_top_n as merge budget
        final_rerank_top_n = _get_v2_config(config, 'final_rerank_top_n', 20)

        logger.info(f"  [TypeConfig] Using default config (type_retrieval_configs not found)")

    # ========== Step 3: Decide Multi-Query Strategy ==========
    confidence_threshold = _get_v2_config(config, 'confidence_threshold', 0.85)
    use_mq_round1 = should_use_multi_query(
        classification.question_type,
        classification.confidence,
        threshold=confidence_threshold,
    )

    metadata["used_multi_query_round1"] = use_mq_round1
    logger.info(f"  [Strategy] Use Multi-Query at Round 1: {use_mq_round1}")

    # ========== Step 3: Round 1 Retrieval ==========
    if use_mq_round1:
        # === Multi-Query Path ===
        logger.info(f"  [Round 1] Generating type-aware multi-queries...")

        queries, query_reasoning = await generate_type_aware_multi_queries(
            original_query=query,
            question_type=classification.question_type,
            llm_provider=llm_provider,
            llm_config=llm_config,
            num_queries=num_queries,
        )

        metadata["round1_queries"] = queries
        metadata["round1_query_reasoning"] = query_reasoning

        logger.info(f"  [Round 1] Executing {len(queries)} queries in parallel...")

        # Parallel hybrid search for each query
        search_tasks = [
            hybrid_search_with_rrf(
                query=q,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=round1_per_query_top_n,
                emb_candidates=hybrid_emb_candidates,
                bm25_candidates=hybrid_bm25_candidates,
                rrf_k=hybrid_rrf_k,
                return_traversal_stats=enable_traversal_stats,
            )
            for q in queries
        ]
        raw_results = await asyncio.gather(*search_tasks)

        # Parse results and collect stats
        if enable_traversal_stats:
            multi_results = []
            for result in raw_results:
                if isinstance(result, tuple):
                    docs_result, stats = result
                    multi_results.append(docs_result)
                    traversal_stats["round1_emb_scored_ids"].update(stats.get("emb_scored_ids", []))
                    traversal_stats["round1_bm25_scored_ids"].update(stats.get("bm25_scored_ids", []))
                else:
                    multi_results.append(result)
        else:
            multi_results = raw_results

        # RRF fusion of multi-query results
        logger.info(f"  [Round 1] RRF fusion of {len(queries)} query results...")
        round1_results = multi_rrf_fusion(multi_results, k=hybrid_rrf_k)
        round1_results = round1_results[:round1_fusion_top_n]

        if enable_traversal_stats:
            traversal_stats["round1_returned_ids"] = set(
                doc.get("unit_id", "") for doc, _ in round1_results
            )

    else:
        # === Single Query Path ===
        logger.info(f"  [Round 1] Single query hybrid search...")

        if enable_traversal_stats:
            round1_results, r1_stats = await hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=round1_fusion_top_n,
                emb_candidates=hybrid_emb_candidates,
                bm25_candidates=hybrid_bm25_candidates,
                rrf_k=hybrid_rrf_k,
                return_traversal_stats=True,
            )
            traversal_stats["round1_emb_scored_ids"] = set(r1_stats.get("emb_scored_ids", []))
            traversal_stats["round1_bm25_scored_ids"] = set(r1_stats.get("bm25_scored_ids", []))
            traversal_stats["round1_returned_ids"] = set(r1_stats.get("fused_ids", []))
        else:
            round1_results = await hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=round1_fusion_top_n,
                emb_candidates=hybrid_emb_candidates,
                bm25_candidates=hybrid_bm25_candidates,
                rrf_k=hybrid_rrf_k,
            )

    metadata["round1_count"] = len(round1_results)
    logger.info(f"  [Round 1] Retrieved {len(round1_results)} documents")
    _log_ids("[Round 1] Unit IDs", round1_results)

    if not round1_results:
        logger.warning(f"  [Warning] No results from Round 1")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # ========== Step 4: Rerank for Sufficiency Check ==========
    logger.info(f"  [Rerank] Reranking to get Top {round1_rerank_top_n} for sufficiency check...")

    if use_reranker:
        reranked_for_check = await reranker_search(
            query=query,
            results=round1_results,
            top_n=round1_rerank_top_n,
            reranker_instruction=_get_reranker_config(config, 'instruction', None),
            batch_size=_get_reranker_config(config, 'batch_size', 20),
            max_retries=_get_reranker_config(config, 'max_retries', 10),
            retry_delay=_get_reranker_config(config, 'retry_delay', 0.8),
            timeout=_get_reranker_config(config, 'timeout', 60.0),
            fallback_threshold=_get_reranker_config(config, 'fallback_threshold', 0.3),
            config=config,
        )
        metadata["round1_reranked_count"] = len(reranked_for_check)

        if enable_traversal_stats:
            input_ids = set(doc.get("unit_id", "") for doc, _ in round1_results)
            output_ids = set(doc.get("unit_id", "") for doc, _ in reranked_for_check)
            traversal_stats["round1_rerank_input_ids"] = input_ids
            traversal_stats["round1_rerank_output_ids"] = output_ids
            traversal_stats["all_reranked_ids"].update(input_ids)
    else:
        reranked_for_check = round1_results[:round1_rerank_top_n]
        metadata["round1_reranked_count"] = len(reranked_for_check)

    if not reranked_for_check:
        logger.warning(f"  [Warning] Reranking failed, returning Round 1 results")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata

    # ========== Step 5: Sufficiency Check ==========
    logger.info(f"  [LLM] Checking sufficiency on Top {len(reranked_for_check)}...")

    is_sufficient, reasoning, missing_info = await check_sufficiency(
        query=query,
        results=reranked_for_check,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=round1_rerank_top_n,
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning

    logger.info(f"  [LLM] Result: {'Sufficient' if is_sufficient else 'Insufficient'}")

    # ========== If Sufficient: Return Round 1 Results ==========
    if is_sufficient:
        logger.info(f"  [Decision] Sufficient! Returning Round 1 results")

        final_results = round1_results

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
        round1_ids = {doc.get("unit_id", "") for doc, _ in round1_results}
        metadata["origin_map"] = _build_origin_map(round1_ids, set(), cluster_ids)

        if enable_traversal_stats:
            _record_traversal_stats(metadata, traversal_stats, final_results, is_multi_round=False)

        logger.info(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== Round 2: Missing Info Based Retrieval ==========
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    logger.info(f"  [Decision] Insufficient, entering Round 2")

    # Note: round2_per_query_top_n, round2_fusion_top_n, merge_budget, final_rerank_top_n
    # are already set from type_config or defaults in Step 2

    logger.info(f"  [LLM] Generating queries based on missing info...")

    refined_queries, query_strategy = await generate_multi_queries(
        original_query=query,
        results=reranked_for_check,
        missing_info=missing_info,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=round1_rerank_top_n,
        num_queries=3,
    )

    metadata["round2_queries"] = refined_queries
    metadata["round2_query_strategy"] = query_strategy

    if enable_traversal_stats:
        traversal_stats["round2_queries"] = refined_queries

    # Parallel retrieval for Round 2 queries
    logger.info(f"  [Round 2] Executing {len(refined_queries)} queries (top_n={round2_per_query_top_n})...")

    round2_tasks = [
        hybrid_search_with_rrf(
            query=q,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=round2_per_query_top_n,
            emb_candidates=hybrid_emb_candidates,
            bm25_candidates=hybrid_bm25_candidates,
            rrf_k=hybrid_rrf_k,
            return_traversal_stats=enable_traversal_stats,
        )
        for q in refined_queries
    ]
    raw_round2_results = await asyncio.gather(*round2_tasks)

    # Parse results and collect stats
    if enable_traversal_stats:
        round2_multi_results = []
        for result in raw_round2_results:
            if isinstance(result, tuple):
                docs_result, stats = result
                round2_multi_results.append(docs_result)
                traversal_stats["round2_all_scored_ids"].update(stats.get("emb_scored_ids", []))
                traversal_stats["round2_all_scored_ids"].update(stats.get("bm25_scored_ids", []))
            else:
                round2_multi_results.append(result)
    else:
        round2_multi_results = raw_round2_results

    # RRF fusion
    logger.info(f"  [Round 2] RRF fusion...")
    round2_results = multi_rrf_fusion(round2_multi_results, k=hybrid_rrf_k)
    round2_results = round2_results[:round2_fusion_top_n]

    metadata["round2_count"] = len(round2_results)

    if enable_traversal_stats:
        traversal_stats["round2_returned_ids"] = set(
            doc.get("unit_id", "") for doc, _ in round2_results
        )

    # ========== Merge Round 1 and Round 2 ==========
    logger.info(f"  [Merge] {classification.question_type.value}: "
                f"R1={len(round1_results)}, R2={len(round2_results)}, budget={merge_budget}")

    round1_ids = {doc.get("unit_id", id(doc)) for doc, _ in round1_results}
    round2_unique = [
        (doc, score) for doc, score in round2_results
        if doc.get("unit_id", id(doc)) not in round1_ids
    ]

    # Merge strategy: Round 1 results + unique Round 2 results (capped by merge_budget)
    # merge_budget is type-specific to control noise for time-sensitive questions
    combined_results = round1_results.copy()
    needed_from_round2 = merge_budget - len(combined_results)
    needed_from_round2 = max(0, needed_from_round2)  # Ensure non-negative
    round2_slice = round2_unique[:needed_from_round2]
    combined_results.extend(round2_slice)

    logger.info(f"  [Merge] Combined: {len(combined_results)} docs (R1={len(round1_results)} + R2={len(round2_slice)})")

    # ========== Final Rerank ==========
    if use_reranker and len(combined_results) > 0:
        logger.info(f"  [Rerank] Final reranking {len(combined_results)} -> {final_rerank_top_n}...")

        final_results = await reranker_search(
            query=query,
            results=combined_results,
            top_n=final_rerank_top_n,
            reranker_instruction=_get_reranker_config(config, 'instruction', None),
            batch_size=_get_reranker_config(config, 'batch_size', 20),
            max_retries=_get_reranker_config(config, 'max_retries', 10),
            retry_delay=_get_reranker_config(config, 'retry_delay', 0.8),
            timeout=_get_reranker_config(config, 'timeout', 60.0),
            fallback_threshold=_get_reranker_config(config, 'fallback_threshold', 0.3),
            config=config,
        )
    else:
        final_results = combined_results[:final_rerank_top_n]

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
    round1_ids_set = {doc.get("unit_id", "") for doc, _ in round1_results}
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

    logger.info(f"  [Complete] Final: {len(final_results)} docs | "
                f"Type: {classification.question_type.value} | "
                f"Latency: {metadata['total_latency_ms']:.0f}ms")

    return final_results, metadata
