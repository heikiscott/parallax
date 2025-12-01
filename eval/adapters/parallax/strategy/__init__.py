"""Retrieval Strategy Module - Extensible strategy routing for memory retrieval.

This module provides a pluggable architecture for routing queries to different
retrieval strategies based on question classification.

Architecture:
    QuestionClassifier → StrategyRouter → RetrievalStrategy (abstract) → Concrete Strategies

Strategies:
    - GEC_CLUSTER_RERANK: LLM selects clusters + Agentic fallback (event-type questions)
    - GEC_INSERT_AFTER_HIT: Original retrieval + cluster expansion (reasoning questions)
    - AGENTIC_ONLY: Pure Agentic retrieval (attribute/preference questions)

Usage:
    from eval.adapters.parallax.strategy import StrategyRouter, create_default_router

    # Create router with default configuration
    router = create_default_router(config)

    # Route a query to appropriate strategy
    strategy = router.route(query)
    results, metadata = await strategy.retrieve(query, context)
"""

from .base import (
    StrategyType,
    RetrievalContext,
    RetrievalResult,
    BaseRetrievalStrategy,
)
from .router import (
    StrategyRouter,
    create_default_router,
)
from .strategies import (
    GECClusterRerankStrategy,
    GECInsertAfterHitStrategy,
    AgenticOnlyStrategy,
)
from .integration import (
    route_and_retrieve,
    get_strategy_for_query,
    create_retrieval_context,
)

__all__ = [
    # Base classes
    "StrategyType",
    "RetrievalContext",
    "RetrievalResult",
    "BaseRetrievalStrategy",
    # Router
    "StrategyRouter",
    "create_default_router",
    # Strategies
    "GECClusterRerankStrategy",
    "GECInsertAfterHitStrategy",
    "AgenticOnlyStrategy",
    # Integration helpers
    "route_and_retrieve",
    "get_strategy_for_query",
    "create_retrieval_context",
]
