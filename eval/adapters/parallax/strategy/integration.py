"""Integration helpers for strategy routing in the eval flow.

This module provides helper functions to integrate the strategy router
into the existing parallax adapter without major refactoring.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import RetrievalContext, RetrievalResult, StrategyType
from .router import StrategyRouter, create_default_router, CLASSIFIER_TO_STRATEGY

# Use TYPE_CHECKING for heavy imports
if TYPE_CHECKING:
    from memory.group_event_cluster import GroupEventClusterIndex
    from providers.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


async def route_and_retrieve(
    query: str,
    config: Any,  # ExperimentConfig
    llm_provider: Any,  # LLMProvider
    llm_config: Dict[str, Any],
    emb_index: Any,
    bm25: Any,
    docs: List[dict],
    cluster_index: Optional[Any] = None,  # GroupEventClusterIndex
    enable_traversal_stats: bool = False,
    router: Optional[StrategyRouter] = None,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """Execute strategy-routed retrieval.

    This is the main integration point for strategy-based retrieval.
    It can be used as a drop-in replacement for direct agentic_retrieval calls.

    Args:
        query: User query
        config: ExperimentConfig instance
        llm_provider: LLM provider for retrieval
        llm_config: LLM configuration dict
        emb_index: Embedding index
        bm25: BM25 index
        docs: List of documents
        cluster_index: Optional cluster index
        enable_traversal_stats: Enable detailed stats
        router: Optional pre-configured router (creates default if None)

    Returns:
        Tuple of (results, metadata) in the same format as agentic_retrieval
    """
    # Check if classification is enabled
    classification_enabled = getattr(config, 'enable_question_classification', True)

    if not classification_enabled:
        # Fall back to existing agentic retrieval
        from eval.adapters.parallax.stage3_memory_retrivel import agentic_retrieval
        return await agentic_retrieval(
            query=query,
            config=config,
            llm_provider=llm_provider,
            llm_config=llm_config,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            cluster_index=cluster_index,
            enable_traversal_stats=enable_traversal_stats,
        )

    # Create or use provided router
    if router is None:
        router = create_default_router(
            config=config,
            enable_classification=True,
        )

    # Build retrieval context
    context = RetrievalContext(
        emb_index=emb_index,
        bm25=bm25,
        docs=docs,
        cluster_index=cluster_index,
        llm_provider=llm_provider,
        llm_config=llm_config,
        config=config,
        enable_traversal_stats=enable_traversal_stats,
    )

    # Execute strategy-routed retrieval
    result = await router.route_and_retrieve(query, context)

    # Convert to legacy format (results, metadata)
    return result.results, result.metadata


def get_strategy_for_query(
    query: str,
    config: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Get the recommended strategy for a query without executing retrieval.

    This is useful for analysis and debugging.

    Args:
        query: The query to classify
        config: Optional config for strategy overrides

    Returns:
        Tuple of (strategy_name, classification_info)
    """
    from agents.question_classifier import classify_question

    classification = classify_question(query)

    # Check for overrides
    if config is not None:
        qc_config = getattr(config, 'question_classification_config', {})
        overrides = qc_config.get('strategy_overrides', {})
        question_type = classification.question_type.value.upper()

        if question_type in overrides:
            override_strategy = overrides[question_type]
            return override_strategy, {
                "question_type": classification.question_type.value,
                "original_strategy": classification.strategy.value,
                "override_applied": True,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
            }

    # Map to strategy type
    strategy_type = CLASSIFIER_TO_STRATEGY.get(
        classification.strategy,
        StrategyType.GEC_INSERT_AFTER_HIT
    )

    return strategy_type.value, {
        "question_type": classification.question_type.value,
        "strategy": classification.strategy.value,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning,
        "detected_patterns": classification.detected_patterns,
    }


def create_retrieval_context(
    config: Any,
    llm_provider: Any,  # LLMProvider
    llm_config: Dict[str, Any],
    emb_index: Any,
    bm25: Any,
    docs: List[dict],
    cluster_index: Optional[Any] = None,  # GroupEventClusterIndex
    enable_traversal_stats: bool = False,
) -> RetrievalContext:
    """Create a RetrievalContext from individual components.

    This is a convenience factory for building the context object.

    Args:
        config: ExperimentConfig
        llm_provider: LLM provider
        llm_config: LLM config dict
        emb_index: Embedding index
        bm25: BM25 index
        docs: Documents list
        cluster_index: Optional cluster index
        enable_traversal_stats: Enable stats tracking

    Returns:
        Configured RetrievalContext
    """
    return RetrievalContext(
        emb_index=emb_index,
        bm25=bm25,
        docs=docs,
        cluster_index=cluster_index,
        llm_provider=llm_provider,
        llm_config=llm_config,
        config=config,
        enable_traversal_stats=enable_traversal_stats,
    )
