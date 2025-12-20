"""Agentic Retrieval V4 - Cross-Attention with Adaptive Routing and C-RAG Evaluation.

This module implements cross-attention based agentic retrieval with:
- Adaptive routing based on query complexity (SIMPLE/MODERATE/COMPLEX)
- C-RAG style three-way evaluation (correct/ambiguous/incorrect)

Key features:
- Uses Cross-Attention instead of ColBERT MaxSim
- No index building required (real-time scoring)
- **Adaptive routing**: Skip C-RAG for SIMPLE queries, force Round 2 for COMPLEX
- C-RAG three-way evaluation for MODERATE queries
- Corrective retrieval for incorrect results (redirects wrong direction)

Flow:
```
Query
  │
  ▼
Question Classification + Complexity Level
  │
  ├─── SIMPLE (ATTRIBUTE_* + High Confidence)
  │         │
  │         ▼
  │    Single Query Cross-Attention → Top N
  │         │
  │         ▼
  │    **SKIP C-RAG** → Return Round 1 directly (saves ~300ms)
  │
  ├─── MODERATE (EVENT_*, TIME_CALCULATION)
  │         │
  │         ▼
  │    Type-Aware Multi-Query → Cross-Attention → RRF Fusion
  │         │
  │         ▼
  │    C-RAG Three-Way Evaluation
  │         │
  │         ├─── CORRECT → Return Round 1
  │         ├─── AMBIGUOUS → Round 2 Supplementary → Merge R1+R2
  │         └─── INCORRECT → Round 2 Corrective → Return R2 only
  │
  └─── COMPLEX (COUNTING, AGGREGATION, REASONING_*)
            │
            ▼
       Type-Aware Multi-Query → Cross-Attention → RRF Fusion
            │
            ▼
       C-RAG Evaluation (but **FORCE Round 2** even if CORRECT)
            │
            ▼
       Round 2 Supplementary → Merge R1+R2 → Higher recall
```

Design Philosophy:
- SIMPLE queries: High-confidence attribute queries rarely need refinement
- MODERATE queries: Standard C-RAG flow with optional Round 2
- COMPLEX queries: Multi-hop/counting need comprehensive retrieval, force Round 2
- C-RAG evaluation distinguishes "incomplete" from "wrong direction"
- Backward compatible: metadata["is_sufficient"] maps to (eval_type == "correct")
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
)
from .llm_utils import (
    check_sufficiency,
    generate_multi_queries,
    evaluate_retrieval,
    generate_corrective_queries,
)

from retrieval.classification.question_classifier import (
    QuestionClassifier,
    QuestionType,
    ClassificationResult,
    ComplexityLevel,
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


def _get_smart_truncate_config(config: Any) -> dict:
    """Get smart truncate global config from config.retrieval.agentic_v4.smart_truncate.

    Returns:
        Dict with enabled, min_results, max_results, score_ratio, gap_threshold, gap_start_idx.

    Conservative defaults (robustness over aggressiveness):
    - min_results: 12 (was 8) - keep more results by default
    - score_ratio: 0.4 (was 0.5) - more relaxed score filter
    - gap_threshold: 0.40 (was 0.30) - larger gap needed to trigger truncation
    - gap_start_idx: 15 - only detect gaps after first 15 results
    """
    defaults = {
        "enabled": True,
        "min_results": 12,
        "max_results": 40,
        "score_ratio": 0.4,
        "gap_threshold": 0.40,
        "gap_start_idx": 15,
    }

    retrieval_cfg = getattr(config, 'retrieval', None)
    if retrieval_cfg is None:
        return defaults

    v4_cfg = getattr(retrieval_cfg, 'agentic_v4', None)
    if v4_cfg is None:
        return defaults

    truncate_cfg = getattr(v4_cfg, 'smart_truncate', None)
    if truncate_cfg is None:
        return defaults

    return {
        "enabled": getattr(truncate_cfg, 'enabled', defaults["enabled"]),
        "min_results": getattr(truncate_cfg, 'min_results', defaults["min_results"]),
        "max_results": getattr(truncate_cfg, 'max_results', defaults["max_results"]),
        "score_ratio": getattr(truncate_cfg, 'score_ratio', defaults["score_ratio"]),
        "gap_threshold": getattr(truncate_cfg, 'gap_threshold', defaults["gap_threshold"]),
        "gap_start_idx": getattr(truncate_cfg, 'gap_start_idx', defaults["gap_start_idx"]),
    }


def _get_truncation_params_for_type(config: Any, type_config: Optional[dict], global_truncate_cfg: dict) -> dict:
    """Get truncation parameters for a specific question type.

    Priority: type_config > global smart_truncate config > defaults.

    Args:
        config: Experiment configuration
        type_config: Type-specific retrieval config (may contain truncate_min, truncate_max, truncate_score_ratio)
        global_truncate_cfg: Global smart_truncate config from agentic_v4

    Returns:
        Dict with min_results, max_results, score_ratio, gap_threshold, gap_start_idx.
    """
    # Start with global config
    params = {
        "min_results": global_truncate_cfg["min_results"],
        "max_results": global_truncate_cfg["max_results"],
        "score_ratio": global_truncate_cfg["score_ratio"],
        "gap_threshold": global_truncate_cfg["gap_threshold"],
        "gap_start_idx": global_truncate_cfg.get("gap_start_idx", 15),
    }

    # Override with type-specific config if available
    if type_config:
        if "truncate_min" in type_config:
            params["min_results"] = type_config["truncate_min"]
        if "truncate_max" in type_config:
            params["max_results"] = type_config["truncate_max"]
        if "truncate_score_ratio" in type_config:
            params["score_ratio"] = type_config["truncate_score_ratio"]
        if "truncate_gap_start_idx" in type_config:
            params["gap_start_idx"] = type_config["truncate_gap_start_idx"]

    return params


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
# Smart Truncation Helper (extracted to reduce code duplication)
# =============================================================================

def _apply_smart_truncation(
    results: List[Tuple[dict, float]],
    final_top_n: int,
    config: Any,
    type_config: Optional[dict],
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Apply smart truncation to results.

    Extracted as helper to reduce code duplication across SIMPLE/CORRECT/AMBIGUOUS/INCORRECT paths.

    Args:
        results: List of (doc, score) tuples to truncate
        final_top_n: Maximum number of results before truncation
        config: Experiment configuration
        type_config: Type-specific retrieval config

    Returns:
        (truncated_results, truncation_metadata)
    """
    global_truncate_cfg = _get_smart_truncate_config(config)
    truncation_params = _get_truncation_params_for_type(config, type_config, global_truncate_cfg)

    candidates = results[:final_top_n]

    if global_truncate_cfg["enabled"]:
        final_results, truncation_meta = smart_score_truncate(
            candidates,
            min_results=truncation_params["min_results"],
            max_results=truncation_params["max_results"],
            score_ratio=truncation_params["score_ratio"],
            gap_threshold=truncation_params["gap_threshold"],
            gap_start_idx=truncation_params["gap_start_idx"],
        )
    else:
        final_results = candidates
        truncation_meta = {
            "reason": "disabled",
            "original_count": len(candidates),
            "final_count": len(candidates),
        }

    return final_results, truncation_meta


def _finalize_metadata(
    metadata: dict,
    final_results: List[Tuple[dict, float]],
    truncation_meta: dict,
    start_time: float,
    round1_ids: set,
    round2_ids: set,
    traversal_stats: Optional[dict] = None,
    is_multi_round: bool = False,
    extra_stats: Optional[dict] = None,
) -> None:
    """Finalize metadata with common fields.

    Mutates metadata dict in place.

    Args:
        metadata: Metadata dict to update
        final_results: Final results list
        truncation_meta: Truncation metadata
        start_time: Start time for latency calculation
        round1_ids: Set of Round 1 unit IDs
        round2_ids: Set of Round 2 unit IDs (empty set if no Round 2)
        traversal_stats: Optional traversal statistics dict
        is_multi_round: Whether this was a multi-round retrieval
        extra_stats: Optional extra stats to merge into traversal_stats
    """
    metadata["truncation"] = truncation_meta
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    metadata["origin_map"] = build_origin_map(round1_ids, round2_ids)

    if traversal_stats is not None:
        stats = {
            "total_docs": traversal_stats["total_docs"],
            "round1_scored": len(traversal_stats.get("round1_scored_ids", set())),
            "final_returned": len(final_results),
            "is_multi_round": is_multi_round,
        }
        if is_multi_round:
            stats["round2_scored"] = len(traversal_stats.get("round2_scored_ids", set()))
        if extra_stats:
            stats.update(extra_stats)
        metadata["traversal_stats"] = stats


async def _execute_round2_retrieval(
    queries: List[str],
    doc_index: List[Dict[str, Any]],
    top_n: int,
    rrf_k: int,
    enable_traversal_stats: bool,
    traversal_stats: dict,
) -> List[Tuple[dict, float]]:
    """Execute Round 2 retrieval with multiple queries.

    Extracted as helper to reduce code duplication between AMBIGUOUS and INCORRECT paths.

    Args:
        queries: List of queries to execute
        doc_index: Document index
        top_n: Number of results per query
        rrf_k: RRF fusion parameter
        enable_traversal_stats: Whether to collect traversal statistics
        traversal_stats: Traversal stats dict to update (mutated in place)

    Returns:
        List of (doc, score) tuples after RRF fusion
    """
    logger.info(f"  [Round 2] Executing {len(queries)} Cross-Attention queries...")

    round2_tasks = [
        cross_attention_search(
            query=q,
            doc_index=doc_index,
            top_n=top_n,
            return_traversal_stats=enable_traversal_stats,
        )
        for q in queries
    ]
    raw_results = await asyncio.gather(*round2_tasks)

    # Parse results and collect stats
    if enable_traversal_stats:
        multi_results = []
        for result in raw_results:
            if isinstance(result, tuple):
                docs_result, stats = result
                multi_results.append(docs_result)
                traversal_stats["round2_scored_ids"].update(stats.get("scored_ids", set()))
            else:
                multi_results.append(result)
    else:
        multi_results = raw_results

    # RRF fusion
    logger.info(f"  [Round 2] RRF fusion...")
    round2_results = multi_cross_attention_rrf_fusion(multi_results, k=rrf_k)
    round2_results = round2_results[:top_n]

    if enable_traversal_stats:
        traversal_stats["round2_returned_ids"] = set(
            doc.get("unit_id", "") for doc, _ in round2_results
        )

    return round2_results


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
    metadata["complexity_level"] = classification.complexity_level.value

    logger.info(f"  [Classify] Type: {classification.question_type.value} "
                f"(conf={classification.confidence:.2f}, complexity={classification.complexity_level.value})")

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
    eval_top_n = _get_v4_config(config, 'eval_top_n', 5)  # Number of docs for C-RAG evaluation

    # ========== Step 3: Decide Multi-Query Strategy ==========
    use_mq_round1 = should_use_multi_query(
        classification.question_type,
        classification.confidence,
        threshold=confidence_threshold,
        complexity_level=classification.complexity_level,
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

    # ========== Step 5: Adaptive Routing Based on Complexity Level ==========
    #
    # SIMPLE: Skip C-RAG evaluation entirely, return Round 1 directly
    #         - Saves ~300ms latency and LLM cost
    #         - For high-confidence attribute queries (identity, location, preference)
    #
    # MODERATE: Standard C-RAG flow
    #           - Round 1 → eval → maybe Round 2
    #
    # COMPLEX: Force Round 2 regardless of evaluation
    #          - For counting, aggregation, reasoning questions
    #          - Higher recall is critical

    if classification.complexity_level == ComplexityLevel.SIMPLE:
        # ========== SIMPLE PATH: Skip C-RAG, return Round 1 directly ==========
        logger.info(f"  [Adaptive] SIMPLE complexity - skipping C-RAG evaluation")

        final_results, truncation_meta = _apply_smart_truncation(
            round1_results, final_top_n, config, type_config
        )

        # SIMPLE-specific metadata
        metadata["evaluation_type"] = "skipped_simple"
        metadata["is_sufficient"] = True
        metadata["adaptive_path"] = "simple_skip_eval"

        round1_ids = {doc.get("unit_id", "") for doc, _ in round1_results}
        _finalize_metadata(
            metadata, final_results, truncation_meta, start_time,
            round1_ids=round1_ids, round2_ids=set(),
            traversal_stats=traversal_stats if enable_traversal_stats else None,
            is_multi_round=False,
            extra_stats={"skipped_eval": True} if enable_traversal_stats else None,
        )

        logger.info(f"  [Complete] SIMPLE path - Final: {len(final_results)} docs | "
                    f"Latency: {metadata['total_latency_ms']:.0f}ms (saved C-RAG eval)")
        return final_results, metadata

    # ========== MODERATE/COMPLEX: Continue with C-RAG Evaluation ==========
    sufficiency_check_count = min(eval_top_n, len(round1_results))
    docs_for_check = round1_results[:sufficiency_check_count]

    logger.info(f"  [LLM] Evaluating retrieval quality on Top {sufficiency_check_count}...")

    eval_type, confidence, reasoning, missing_info, incorrect_aspects, correct_direction = await evaluate_retrieval(
        query=query,
        results=docs_for_check,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=sufficiency_check_count,
    )

    # Store evaluation results in metadata
    metadata["evaluation_type"] = eval_type
    metadata["evaluation_confidence"] = confidence
    metadata["reasoning"] = reasoning
    # Backward compatibility: map to is_sufficient
    metadata["is_sufficient"] = (eval_type == "correct")

    logger.info(f"  [LLM] Result: {eval_type.upper()} (confidence: {confidence:.2f})")

    # ========== COMPLEX: Force Round 2 even if eval_type is "correct" ==========
    if classification.complexity_level == ComplexityLevel.COMPLEX and eval_type == "correct":
        logger.info(f"  [Adaptive] COMPLEX complexity - forcing Round 2 for higher recall")
        eval_type = "ambiguous"  # Treat as ambiguous to trigger supplementary retrieval
        missing_info = ["Additional context needed for complex multi-hop query"]
        metadata["forced_round2"] = True
        metadata["original_eval_type"] = "correct"

    # ========== PATH 1: CORRECT - Return Round 1 Results ==========
    if eval_type == "correct":
        logger.info(f"  [Decision] CORRECT! Returning Round 1 Cross-Attention results")

        final_results, truncation_meta = _apply_smart_truncation(
            round1_results, final_top_n, config, type_config
        )

        round1_ids = {doc.get("unit_id", "") for doc, _ in round1_results}
        _finalize_metadata(
            metadata, final_results, truncation_meta, start_time,
            round1_ids=round1_ids, round2_ids=set(),
            traversal_stats=traversal_stats if enable_traversal_stats else None,
            is_multi_round=False,
        )

        logger.info(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== PATH 2: AMBIGUOUS - Supplementary Retrieval (fill missing info) ==========
    if eval_type == "ambiguous":
        metadata["is_multi_round"] = True
        metadata["missing_info"] = missing_info
        metadata["round2_strategy"] = "supplementary"
        logger.info(f"  [Decision] AMBIGUOUS, entering Round 2 (supplementary retrieval)")

        # Generate supplementary queries
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

        # Execute Round 2
        round2_results = await _execute_round2_retrieval(
            queries=refined_queries,
            doc_index=doc_index,
            top_n=round2_top_n,
            rrf_k=rrf_k,
            enable_traversal_stats=enable_traversal_stats,
            traversal_stats=traversal_stats,
        )
        metadata["round2_count"] = len(round2_results)

        # Merge Round 1 and Round 2 (AMBIGUOUS: both rounds are relevant)
        logger.info(f"  [Merge] R1={len(round1_results)}, R2={len(round2_results)}, budget={merge_budget}")
        combined_results, round1_ids, round2_added_ids = merge_round_results(
            round1_results=round1_results,
            round2_results=round2_results,
            merge_budget=merge_budget,
        )
        logger.info(f"  [Merge] Combined: {len(combined_results)} docs")

        # Apply truncation and finalize
        final_results, truncation_meta = _apply_smart_truncation(
            combined_results, final_top_n, config, type_config
        )
        _finalize_metadata(
            metadata, final_results, truncation_meta, start_time,
            round1_ids=round1_ids, round2_ids=round2_added_ids,
            traversal_stats=traversal_stats if enable_traversal_stats else None,
            is_multi_round=True,
        )

        logger.info(f"  [Complete] Final: {len(final_results)} docs | "
                    f"Type: {classification.question_type.value} | "
                    f"Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== PATH 3: INCORRECT - Corrective Retrieval (redirect wrong direction) ==========
    # eval_type == "incorrect"
    metadata["is_multi_round"] = True
    metadata["incorrect_aspects"] = incorrect_aspects
    metadata["correct_direction"] = correct_direction
    metadata["round2_strategy"] = "corrective"
    logger.info(f"  [Decision] INCORRECT! Entering Round 2 (corrective retrieval)")
    logger.info(f"  [Incorrect] Wrong aspects: {incorrect_aspects}")
    logger.info(f"  [Incorrect] Correct direction: {correct_direction}")

    # Generate corrective queries
    logger.info(f"  [LLM] Generating corrective queries...")
    corrective_queries, query_strategy = await generate_corrective_queries(
        original_query=query,
        results=docs_for_check,
        incorrect_aspects=incorrect_aspects,
        correct_direction=correct_direction,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=sufficiency_check_count,
        num_queries=3,
    )
    metadata["round2_queries"] = corrective_queries
    metadata["round2_query_strategy"] = query_strategy

    # Execute Round 2
    round2_results = await _execute_round2_retrieval(
        queries=corrective_queries,
        doc_index=doc_index,
        top_n=round2_top_n,
        rrf_k=rrf_k,
        enable_traversal_stats=enable_traversal_stats,
        traversal_stats=traversal_stats,
    )
    metadata["round2_count"] = len(round2_results)

    # CRITICAL: For INCORRECT, do NOT merge with Round 1 (it's wrong direction)
    # Use Round 2 results only to avoid polluting with off-topic documents
    logger.info(f"  [Corrective] Using Round 2 only (R1 was incorrect direction)")

    # Apply truncation and finalize (Round 2 only, no Round 1)
    final_results, truncation_meta = _apply_smart_truncation(
        round2_results, final_top_n, config, type_config
    )

    round2_ids = {doc.get("unit_id", "") for doc, _ in round2_results}
    metadata["merge_strategy"] = "round2_only"
    _finalize_metadata(
        metadata, final_results, truncation_meta, start_time,
        round1_ids=set(), round2_ids=round2_ids,  # R1 excluded
        traversal_stats=traversal_stats if enable_traversal_stats else None,
        is_multi_round=True,
        extra_stats={"round1_excluded": True} if enable_traversal_stats else None,
    )

    logger.info(f"  [Complete] Final: {len(final_results)} docs (R2 only) | "
                f"Type: {classification.question_type.value} | "
                f"Latency: {metadata['total_latency_ms']:.0f}ms")

    return final_results, metadata
