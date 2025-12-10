"""Concrete retrieval policy implementations.

This module contains the actual implementations of retrieval policies:
- GECClusterRerankPolicy: LLM-guided cluster selection + Agentic fallback
- GECInsertAfterHitPolicy: Original retrieval with cluster expansion
- AgenticOnlyPolicy: Pure Agentic retrieval without cluster expansion
"""

import copy
import logging
import time
from typing import Any, Dict, Optional

from .types import (
    BaseRetrievalStrategy,
    StrategyType,
    RetrievalContext,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class _ConfigWrapper:
    """Wrapper that delegates to original config but overrides GEC config.

    This class ensures the original config is never modified,
    which is critical for concurrent request safety.
    """

    def __init__(self, original_config: Any, gec_overrides: Dict[str, Any]):
        """Initialize wrapper with config and overrides.

        Args:
            original_config: Original ExperimentConfig object
            gec_overrides: Dict of keys to override in group_event_cluster_retrieval_config
        """
        self._original = original_config
        # Deep copy the GEC config to avoid any mutation
        original_gec_config = getattr(
            original_config, 'group_event_cluster_retrieval_config', {}
        )
        self._gec_config = copy.deepcopy(original_gec_config)
        self._gec_config.update(gec_overrides)

    @property
    def group_event_cluster_retrieval_config(self) -> Dict[str, Any]:
        """Return the overridden GEC config."""
        return self._gec_config

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to original config."""
        return getattr(self._original, name)


def _create_config_with_overrides(config: Any, overrides: Dict[str, Any]) -> _ConfigWrapper:
    """Create a config wrapper with GEC config overrides.

    This function ensures the original config is never modified,
    which is critical for concurrent request safety.

    Args:
        config: Original ExperimentConfig object
        overrides: Dict of keys to override in group_event_cluster_retrieval_config

    Returns:
        A ConfigWrapper with the overrides applied
    """
    return _ConfigWrapper(config, overrides)


class GECClusterRerankPolicy(BaseRetrievalStrategy):
    """GEC Cluster Rerank Policy - LLM selects clusters + Agentic fallback.

    This policy is optimal for event-type questions (temporal, activity, aggregation).
    It uses LLM to intelligently select relevant clusters and falls back to
    Agentic retrieval if no suitable clusters are found.

    Flow:
    1. Execute standard Agentic retrieval (Round 1 + optional Round 2)
    2. Apply cluster expansion with "cluster_rerank" strategy
    3. LLM selects the most relevant clusters
    4. Return combined results

    Use cases:
    - "When did X go to Y?" (EVENT_TEMPORAL)
    - "What activities does X do?" (EVENT_ACTIVITY)
    - "What books has X read?" (EVENT_AGGREGATION)
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the policy.

        Args:
            config: Optional ExperimentConfig for customization
        """
        super().__init__(StrategyType.GEC_CLUSTER_RERANK)
        self._config = config

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
    ) -> RetrievalResult:
        """Execute GEC Cluster Rerank retrieval.

        Args:
            query: User query
            context: RetrievalContext with indices and configuration

        Returns:
            RetrievalResult with cluster-enhanced results
        """
        from src.retrieval.offline.pipelines import agentic_retrieval

        start_time = time.time()

        # Use context config or instance config
        original_config = context.config or self._config
        if original_config is None:
            raise ValueError("ExperimentConfig is required for GECClusterRerankPolicy")

        # Create config wrapper to avoid modifying original (concurrent safety)
        config = _create_config_with_overrides(original_config, {
            'expansion_strategy': 'cluster_rerank',
        })

        logger.info(f"[GECClusterRerank] Executing for: {query[:50]}...")

        # Execute agentic retrieval with cluster expansion
        results, metadata = await agentic_retrieval(
            query=query,
            config=config,
            llm_provider=context.llm_provider,
            llm_config=context.llm_config,
            emb_index=context.emb_index,
            bm25=context.bm25,
            docs=context.docs,
            cluster_index=context.cluster_index,
            enable_traversal_stats=context.enable_traversal_stats,
        )

        metadata["strategy_execution_time_ms"] = (time.time() - start_time) * 1000
        metadata["strategy_name"] = self.name

        return RetrievalResult(
            results=results,
            metadata=metadata,
            strategy_type=self.strategy_type,
        )


class GECInsertAfterHitPolicy(BaseRetrievalStrategy):
    """GEC Insert After Hit Policy - Original retrieval + cluster expansion.

    This policy is optimal for reasoning/hypothetical questions that benefit
    from context expansion without LLM cluster selection overhead.

    Flow:
    1. Execute standard Agentic retrieval
    2. Apply cluster expansion with "insert_after_hit" strategy
    3. For each hit, insert related cluster members nearby
    4. Return expanded results

    Use cases:
    - "Would X do Y if...?" (REASONING_HYPOTHETICAL)
    - "What career path has X chosen?" (GENERAL/Career)
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the policy.

        Args:
            config: Optional ExperimentConfig for customization
        """
        super().__init__(StrategyType.GEC_INSERT_AFTER_HIT)
        self._config = config

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
    ) -> RetrievalResult:
        """Execute GEC Insert After Hit retrieval.

        Args:
            query: User query
            context: RetrievalContext with indices and configuration

        Returns:
            RetrievalResult with context-expanded results
        """
        from src.retrieval.offline.pipelines import agentic_retrieval

        start_time = time.time()

        original_config = context.config or self._config
        if original_config is None:
            raise ValueError("ExperimentConfig is required for GECInsertAfterHitPolicy")

        # Create config wrapper (concurrent safety)
        config = _create_config_with_overrides(original_config, {
            'expansion_strategy': 'insert_after_hit',
        })

        logger.info(f"[GECInsertAfterHit] Executing for: {query[:50]}...")

        results, metadata = await agentic_retrieval(
            query=query,
            config=config,
            llm_provider=context.llm_provider,
            llm_config=context.llm_config,
            emb_index=context.emb_index,
            bm25=context.bm25,
            docs=context.docs,
            cluster_index=context.cluster_index,
            enable_traversal_stats=context.enable_traversal_stats,
        )

        metadata["strategy_execution_time_ms"] = (time.time() - start_time) * 1000
        metadata["strategy_name"] = self.name

        return RetrievalResult(
            results=results,
            metadata=metadata,
            strategy_type=self.strategy_type,
        )


class AgenticOnlyPolicy(BaseRetrievalStrategy):
    """Agentic Only Policy - Pure Agentic retrieval without cluster expansion.

    This policy is optimal for attribute/preference questions that require
    precise keyword matching and don't benefit from cluster expansion.

    Flow:
    1. Execute standard Agentic retrieval (Round 1 + optional Round 2)
    2. Skip cluster expansion entirely
    3. Return Agentic results directly

    Use cases:
    - "What is X's identity?" (ATTRIBUTE_IDENTITY)
    - "What does X like?" (ATTRIBUTE_PREFERENCE)
    - "Where is X from?" (ATTRIBUTE_LOCATION)
    - "How long ago was...?" (TIME_CALCULATION)
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the policy.

        Args:
            config: Optional ExperimentConfig for customization
        """
        super().__init__(StrategyType.AGENTIC_ONLY)
        self._config = config

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
    ) -> RetrievalResult:
        """Execute pure Agentic retrieval.

        Args:
            query: User query
            context: RetrievalContext with indices and configuration

        Returns:
            RetrievalResult without cluster expansion
        """
        from src.retrieval.offline.pipelines import agentic_retrieval

        start_time = time.time()

        original_config = context.config or self._config
        if original_config is None:
            raise ValueError("ExperimentConfig is required for AgenticOnlyPolicy")

        # Create config wrapper, disable GEC expansion
        config = _create_config_with_overrides(original_config, {
            'enable_group_event_cluster_retrieval': False,
        })

        logger.info(f"[AgenticOnly] Executing for: {query[:50]}...")

        # Execute without cluster index
        results, metadata = await agentic_retrieval(
            query=query,
            config=config,
            llm_provider=context.llm_provider,
            llm_config=context.llm_config,
            emb_index=context.emb_index,
            bm25=context.bm25,
            docs=context.docs,
            cluster_index=None,  # Explicitly disable cluster
            enable_traversal_stats=context.enable_traversal_stats,
        )

        metadata["strategy_execution_time_ms"] = (time.time() - start_time) * 1000
        metadata["strategy_name"] = self.name
        metadata["cluster_expansion_disabled"] = True

        return RetrievalResult(
            results=results,
            metadata=metadata,
            strategy_type=self.strategy_type,
        )


# Aliases for backward compatibility
GECClusterRerankStrategy = GECClusterRerankPolicy
GECInsertAfterHitStrategy = GECInsertAfterHitPolicy
AgenticOnlyStrategy = AgenticOnlyPolicy
