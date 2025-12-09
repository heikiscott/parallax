"""Retrieval routing module - Routes queries to appropriate retrieval policies.

This module provides:
- StrategyType: Enum of available retrieval strategies
- RetrievalContext: Context object for retrieval operations
- RetrievalResult: Result container with metadata
- BaseRetrievalStrategy: Abstract base class for policies
- StrategyRouter: Routes queries to appropriate policies
- Policy implementations: GECClusterRerankPolicy, etc.
- Integration helpers: route_and_retrieve, etc.
"""

from .types import (
    StrategyType,
    RetrievalContext,
    RetrievalResult,
    BaseRetrievalStrategy,
)
from .router import (
    StrategyRouter,
    create_default_router,
    CLASSIFIER_TO_STRATEGY,
    STRATEGY_NAME_TO_TYPE,
)
from .policies import (
    GECClusterRerankPolicy,
    GECInsertAfterHitPolicy,
    AgenticOnlyPolicy,
    # Aliases for backward compatibility during transition
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
    # Base types
    "StrategyType",
    "RetrievalContext",
    "RetrievalResult",
    "BaseRetrievalStrategy",
    # Router
    "StrategyRouter",
    "create_default_router",
    "CLASSIFIER_TO_STRATEGY",
    "STRATEGY_NAME_TO_TYPE",
    # Policy implementations (new names)
    "GECClusterRerankPolicy",
    "GECInsertAfterHitPolicy",
    "AgenticOnlyPolicy",
    # Strategy implementations (aliases)
    "GECClusterRerankStrategy",
    "GECInsertAfterHitStrategy",
    "AgenticOnlyStrategy",
    # Integration helpers
    "route_and_retrieve",
    "get_strategy_for_query",
    "create_retrieval_context",
]
