"""Agentic Retrieval V4 - Cross-Attention with Type-Aware Multi-Query.

This module implements cross-attention based agentic retrieval, replacing ColBERT
from V3 with cross-attention scoring.

Key differences from V3:
- Uses Cross-Attention instead of ColBERT MaxSim
- No index building required (real-time scoring)
- Same agentic flow (classification, multi-query, sufficiency check)

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
  │    Single Query Cross-Attention → Top 10 → Sufficiency Check
  │
  └─── Complex (EVENT_*, COUNTING, REASONING_*)
            │
            ▼
       Type-Aware Multi-Query (2-3) → Parallel Cross-Attention → RRF Fusion
            │
            ▼
       Sufficiency Check
            │
            ├─── Sufficient → Return Results
            │
            └─── Insufficient → Round 2 (missing_info queries)
                      │
                      ▼
                 Cross-Attention Search → Merge → Return Final
```

Design Philosophy:
- Keep V3's agentic flow (classification, multi-query, sufficiency check)
- Replace ColBERT with Cross-Attention
- Cross-attention scoring to be implemented by ML team
"""

import time
import asyncio
import logging
import json
from typing import List, Tuple, Optional, Any, Set, Dict

from .cross_attention_utils import (
    cross_attention_search,
    multi_cross_attention_rrf_fusion,
    merge_round_results,
    build_origin_map,
    smart_score_truncate,
    get_truncation_config,
)
from .llm_utils import (
    check_sufficiency,
    generate_multi_queries,
)

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
# Config Helpers
# =============================================================================

def _get_v4_config(config: Any, key: str, default: Any) -> Any:
    """Get V4-specific config value from config.retrieval.agentic_v4.xxx."""
    retrieval_cfg = getattr(config, 'retrieval', None)
    if retrieval_cfg is None:
        return default
    v4_cfg = getattr(retrieval_cfg, 'agentic_v4', None)
    if v4_cfg is None:
        return default
    return getattr(v4_cfg, key, default)


def _get_type_retrieval_config_v4(config: Any, question_type: QuestionType) -> Optional[dict]:
    """Get type-specific retrieval config for V4.

    Looks up config.retrieval.v4_type_retrieval_configs.{type_name}
    V4 独立配置，不 fallback 到 V3。

    Args:
        config: Experiment configuration
        question_type: Classified question type

    Returns:
        Dict with type-specific params or None if not configured.
        Keys: round1_top_n, round2_top_n, merge_budget, final_top_n
    """
    retrieval_cfg = getattr(config, 'retrieval', None)
    if retrieval_cfg is None:
        return None

    # V4 独立配置，不 fallback 到 V3
    v4_type_configs = getattr(retrieval_cfg, 'v4_type_retrieval_configs', None)
    if v4_type_configs is None:
        return None

    # Try to get config for this specific type
    type_key = question_type.value
    type_config = getattr(v4_type_configs, type_key, None)

    # Fallback to default config
    if type_config is None:
        type_config = getattr(v4_type_configs, 'default', None)

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


# =============================================================================
# Type-Aware Multi-Query Generation (reused from V3)
# =============================================================================

async def generate_type_aware_multi_queries(
    original_query: str,
    question_type: QuestionType,
    llm_provider: Any,
    llm_config: dict,
    num_queries: int = 3,
) -> Tuple[List[str], str]:
    """Generate multi-query variations based on question type at Round 1.

    This function generates queries based on the question type classification,
    without needing retrieved documents or missing_info.

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
        content = await llm_provider.generate(
            prompt=prompt,
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 1024),
            response_format={"type": "json_object"},
        )
        content = content.strip()

        # Parse JSON response (handle markdown code blocks)
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

        if not queries or not isinstance(queries, list):
            logger.warning(f"  [TypeMQ] Invalid response, using original query")
            return [original_query], "Fallback to original"

        queries = queries[:num_queries]
        if len(queries) < num_queries and original_query not in queries:
            queries.append(original_query)

        logger.info(f"  [TypeMQ] Generated {len(queries)} queries for {question_type.value}")
        return queries, reasoning

    except json.JSONDecodeError as e:
        logger.warning(f"  [TypeMQ] JSON parse error: {e}")
        return [original_query], f"JSON error: {e}"
    except Exception as e:
        logger.warning(f"  [TypeMQ] Query generation failed: {e}")
        return [original_query], f"Error: {e}"


# =============================================================================
# Main Entry Point: agentic_retrieval_v4
# =============================================================================

async def agentic_retrieval_v4(
    query: str,
    config: Any,
    llm_provider: Any,
    llm_config: dict,
    doc_index: List[Dict[str, Any]],
    enable_traversal_stats: bool = False,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Agentic Retrieval V4 - Cross-Attention with type-aware multi-query.

    Key difference from V3: Uses Cross-Attention instead of ColBERT.
    Cross-attention computes scores in real-time (no pre-built index needed).

    Args:
        query: User query
        config: Experiment configuration
        llm_provider: LLM Provider instance
        llm_config: LLM configuration dict
        doc_index: Document index with structure:
            [{"doc": {...}}, ...] where doc contains "narrative" field
        enable_traversal_stats: Enable detailed traversal statistics

    Returns:
        (final_results, metadata)
    """
    start_time = time.time()

    metadata = {
        "version": "v4",
        "retrieval_method": "cross_attention",
        "is_multi_round": False,
        "round1_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "final_count": 0,
        "total_latency_ms": 0.0,
        "question_type": None,
        "used_multi_query_round1": False,
        "classification_confidence": 0.0,
        "classification_reasoning": "",
        "round1_queries": [],
        "round1_query_reasoning": "",
        "round2_queries": [],
        "round2_query_strategy": "",
        "missing_info": [],
        "type_config": {},
        "origin_map": {},
    }

    traversal_stats = {
        "total_docs": len(doc_index),
        "round1_scored_ids": set(),
        "round1_returned_ids": set(),
        "round2_scored_ids": set(),
        "round2_returned_ids": set(),
    }

    logger.info(f"{'='*60}")
    logger.info(f"Agentic Retrieval V4 (Cross-Attention): {query[:60]}...")
    logger.info(f"{'='*60}")

    # ========== Step 1: Question Classification ==========
    classifier = QuestionClassifier()
    classification: ClassificationResult = classifier.classify(query)

    metadata["question_type"] = classification.question_type.value
    metadata["classification_confidence"] = classification.confidence
    metadata["classification_reasoning"] = classification.reasoning

    logger.info(f"  [Classify] Type: {classification.question_type.value} "
                f"(conf={classification.confidence:.2f})")

    # ========== Step 2: Load Type-Specific Config ==========
    type_config = _get_type_retrieval_config_v4(config, classification.question_type)

    if type_config:
        round1_top_n = type_config.get('round1_top_n', 12)
        round2_top_n = type_config.get('round2_top_n', 15)
        merge_budget = type_config.get('merge_budget', 20)
        final_top_n = type_config.get('final_top_n', 12)

        logger.info(f"  [TypeConfig] {classification.question_type.value}: "
                    f"R1={round1_top_n}, R2={round2_top_n}, merge={merge_budget}, final={final_top_n}")

        metadata["type_config"] = {
            "type": classification.question_type.value,
            "round1_top_n": round1_top_n,
            "round2_top_n": round2_top_n,
            "merge_budget": merge_budget,
            "final_top_n": final_top_n,
        }
    else:
        # Fallback to agentic_v4 default config
        round1_top_n = _get_v4_config(config, 'round1_top_n', 12)
        round2_top_n = _get_v4_config(config, 'round2_top_n', 15)
        merge_budget = _get_v4_config(config, 'merge_budget', 20)
        final_top_n = _get_v4_config(config, 'final_top_n', 12)

        logger.info(f"  [TypeConfig] Using default config (v4_type_retrieval_configs not found)")

    num_queries = _get_v4_config(config, 'num_queries', 3)
    confidence_threshold = _get_v4_config(config, 'confidence_threshold', 0.85)
    rrf_k = _get_v4_config(config, 'rrf_k', 60)

    # ========== Step 3: Decide Multi-Query Strategy ==========
    use_mq_round1 = should_use_multi_query(
        classification.question_type,
        classification.confidence,
        threshold=confidence_threshold,
    )

    metadata["used_multi_query_round1"] = use_mq_round1
    logger.info(f"  [Strategy] Use Multi-Query at Round 1: {use_mq_round1}")

    # ========== Step 4: Round 1 Cross-Attention Retrieval ==========
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

        logger.info(f"  [Round 1] Executing {len(queries)} Cross-Attention queries in parallel...")

        # Parallel cross-attention search for each query
        search_tasks = [
            cross_attention_search(
                query=q,
                doc_index=doc_index,
                top_n=round1_top_n,
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
                    traversal_stats["round1_scored_ids"].update(stats.get("scored_ids", set()))
                else:
                    multi_results.append(result)
        else:
            multi_results = raw_results

        # RRF fusion of multi-query results
        logger.info(f"  [Round 1] RRF fusion of {len(queries)} query results...")
        round1_results = multi_cross_attention_rrf_fusion(multi_results, k=rrf_k)
        round1_results = round1_results[:round1_top_n]

        if enable_traversal_stats:
            traversal_stats["round1_returned_ids"] = set(
                doc.get("unit_id", "") for doc, _ in round1_results
            )
    else:
        # === Single Query Path ===
        logger.info(f"  [Round 1] Single query Cross-Attention search...")

        if enable_traversal_stats:
            round1_results, r1_stats = await cross_attention_search(
                query=query,
                doc_index=doc_index,
                top_n=round1_top_n,
                return_traversal_stats=True,
            )
            traversal_stats["round1_scored_ids"] = r1_stats.get("scored_ids", set())
            traversal_stats["round1_returned_ids"] = r1_stats.get("returned_ids", set())
        else:
            round1_results = await cross_attention_search(
                query=query,
                doc_index=doc_index,
                top_n=round1_top_n,
            )

    metadata["round1_count"] = len(round1_results)
    logger.info(f"  [Round 1] Retrieved {len(round1_results)} documents via Cross-Attention")
    _log_ids("[Round 1] Unit IDs", round1_results)

    if not round1_results:
        logger.warning(f"  [Warning] No results from Round 1")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # ========== Step 5: Sufficiency Check (on top results) ==========
    sufficiency_check_count = min(10, len(round1_results))
    docs_for_check = round1_results[:sufficiency_check_count]

    logger.info(f"  [LLM] Checking sufficiency on Top {sufficiency_check_count}...")

    is_sufficient, reasoning, missing_info = await check_sufficiency(
        query=query,
        results=docs_for_check,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=sufficiency_check_count,
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning

    logger.info(f"  [LLM] Result: {'Sufficient' if is_sufficient else 'Insufficient'}")

    # ========== If Sufficient: Return Round 1 Results ==========
    if is_sufficient:
        logger.info(f"  [Decision] Sufficient! Returning Round 1 Cross-Attention results")

        # Get type-specific truncation config
        question_type_str = classification.question_type.value
        truncation_config = get_truncation_config(question_type_str)

        # Apply smart truncation before final_top_n limit
        candidates = round1_results[:final_top_n]
        final_results, truncation_meta = smart_score_truncate(
            candidates,
            min_results=truncation_config["min_results"],
            max_results=truncation_config["max_results"],
            score_ratio=truncation_config["score_ratio"],
        )

        metadata["truncation"] = truncation_meta
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000

        round1_ids = {doc.get("unit_id", "") for doc, _ in round1_results}
        metadata["origin_map"] = build_origin_map(round1_ids, set())

        if enable_traversal_stats:
            metadata["traversal_stats"] = {
                "total_docs": traversal_stats["total_docs"],
                "round1_scored": len(traversal_stats["round1_scored_ids"]),
                "final_returned": len(final_results),
                "is_multi_round": False,
            }

        logger.info(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== Round 2: Missing Info Based Retrieval ==========
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    logger.info(f"  [Decision] Insufficient, entering Round 2")

    logger.info(f"  [LLM] Generating queries based on missing info...")

    refined_queries, query_strategy = await generate_multi_queries(
        original_query=query,
        results=docs_for_check,
        missing_info=missing_info,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=sufficiency_check_count,
        num_queries=3,
    )

    metadata["round2_queries"] = refined_queries
    metadata["round2_query_strategy"] = query_strategy

    # Parallel cross-attention retrieval for Round 2 queries
    logger.info(f"  [Round 2] Executing {len(refined_queries)} Cross-Attention queries...")

    round2_tasks = [
        cross_attention_search(
            query=q,
            doc_index=doc_index,
            top_n=round2_top_n,
            return_traversal_stats=enable_traversal_stats,
        )
        for q in refined_queries
    ]
    raw_round2_results = await asyncio.gather(*round2_tasks)

    # Parse results
    if enable_traversal_stats:
        round2_multi_results = []
        for result in raw_round2_results:
            if isinstance(result, tuple):
                docs_result, stats = result
                round2_multi_results.append(docs_result)
                traversal_stats["round2_scored_ids"].update(stats.get("scored_ids", set()))
            else:
                round2_multi_results.append(result)
    else:
        round2_multi_results = raw_round2_results

    # RRF fusion of Round 2 results
    logger.info(f"  [Round 2] RRF fusion...")
    round2_results = multi_cross_attention_rrf_fusion(round2_multi_results, k=rrf_k)
    round2_results = round2_results[:round2_top_n]

    metadata["round2_count"] = len(round2_results)

    if enable_traversal_stats:
        traversal_stats["round2_returned_ids"] = set(
            doc.get("unit_id", "") for doc, _ in round2_results
        )

    # ========== Merge Round 1 and Round 2 ==========
    logger.info(f"  [Merge] R1={len(round1_results)}, R2={len(round2_results)}, budget={merge_budget}")

    combined_results, round1_ids, round2_added_ids = merge_round_results(
        round1_results=round1_results,
        round2_results=round2_results,
        merge_budget=merge_budget,
    )

    logger.info(f"  [Merge] Combined: {len(combined_results)} docs")

    # Get type-specific truncation config
    question_type_str = classification.question_type.value
    truncation_config = get_truncation_config(question_type_str)

    # Apply smart truncation before final_top_n limit
    candidates = combined_results[:final_top_n]
    final_results, truncation_meta = smart_score_truncate(
        candidates,
        min_results=truncation_config["min_results"],
        max_results=truncation_config["max_results"],
        score_ratio=truncation_config["score_ratio"],
    )

    metadata["truncation"] = truncation_meta
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    metadata["origin_map"] = build_origin_map(round1_ids, round2_added_ids)

    if enable_traversal_stats:
        metadata["traversal_stats"] = {
            "total_docs": traversal_stats["total_docs"],
            "round1_scored": len(traversal_stats["round1_scored_ids"]),
            "round2_scored": len(traversal_stats["round2_scored_ids"]),
            "final_returned": len(final_results),
            "is_multi_round": True,
        }

    logger.info(f"  [Complete] Final: {len(final_results)} docs | "
                f"Type: {classification.question_type.value} | "
                f"Latency: {metadata['total_latency_ms']:.0f}ms")

    return final_results, metadata
