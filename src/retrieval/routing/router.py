"""Strategy Router - Routes queries to appropriate retrieval policies.

This module provides the StrategyRouter class that uses question classification
to route queries to the most appropriate retrieval policy.
"""

import logging
from typing import Any, Dict, Optional

from src.retrieval.classification import (
    QuestionClassifier,
    ClassificationResult,
    RetrievalStrategy as ClassifierStrategy,
)

from .types import (
    BaseRetrievalStrategy,
    StrategyType,
    RetrievalContext,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# Mapping from classifier strategy to our strategy type
CLASSIFIER_TO_STRATEGY: Dict[ClassifierStrategy, StrategyType] = {
    ClassifierStrategy.GEC_CLUSTER_RERANK: StrategyType.GEC_CLUSTER_RERANK,
    ClassifierStrategy.GEC_INSERT_AFTER_HIT: StrategyType.GEC_INSERT_AFTER_HIT,
    ClassifierStrategy.AGENTIC_ONLY: StrategyType.AGENTIC_ONLY,
}

# Mapping from strategy string names to StrategyType
STRATEGY_NAME_TO_TYPE: Dict[str, StrategyType] = {
    "gec_cluster_rerank": StrategyType.GEC_CLUSTER_RERANK,
    "gec_insert_after_hit": StrategyType.GEC_INSERT_AFTER_HIT,
    "agentic_only": StrategyType.AGENTIC_ONLY,
}


class StrategyRouter:
    """Routes queries to appropriate retrieval policies based on classification.

    The router uses a QuestionClassifier to analyze queries and determine the
    best retrieval policy. It maintains a registry of available policies
    and handles policy selection and fallback logic.

    Example:
        # Create router with policies
        router = StrategyRouter()
        router.register_strategy(StrategyType.GEC_CLUSTER_RERANK, cluster_policy)
        router.register_strategy(StrategyType.AGENTIC_ONLY, agentic_policy)

        # Route and execute
        result = await router.route_and_retrieve(query, context)
    """

    def __init__(
        self,
        classifier: Optional[QuestionClassifier] = None,
        default_strategy_type: StrategyType = StrategyType.GEC_INSERT_AFTER_HIT,
        enable_classification: bool = True,
        strategy_overrides: Optional[Dict[str, str]] = None,
        log_classification: bool = True,
    ):
        """Initialize the strategy router.

        Args:
            classifier: QuestionClassifier instance (creates default if None)
            default_strategy_type: Fallback strategy when classification fails
            enable_classification: If False, always use default strategy
            strategy_overrides: Optional dict mapping question_type to strategy
                                e.g., {"EVENT_TEMPORAL": "agentic_only"}
            log_classification: Whether to log classification results
        """
        self._classifier = classifier or QuestionClassifier()
        self._default_strategy_type = default_strategy_type
        self._enable_classification = enable_classification
        self._strategy_overrides = strategy_overrides or {}
        self._log_classification = log_classification
        self._strategies: Dict[StrategyType, BaseRetrievalStrategy] = {}

    def register_strategy(
        self,
        strategy_type: StrategyType,
        strategy: BaseRetrievalStrategy,
    ) -> "StrategyRouter":
        """Register a policy for a given type.

        Args:
            strategy_type: The type of strategy
            strategy: The policy instance

        Returns:
            self for chaining
        """
        self._strategies[strategy_type] = strategy
        logger.debug(f"Registered policy: {strategy_type.value}")
        return self

    def unregister_strategy(self, strategy_type: StrategyType) -> "StrategyRouter":
        """Unregister a policy.

        Args:
            strategy_type: The type to unregister

        Returns:
            self for chaining
        """
        if strategy_type in self._strategies:
            del self._strategies[strategy_type]
            logger.debug(f"Unregistered policy: {strategy_type.value}")
        return self

    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseRetrievalStrategy]:
        """Get a registered policy by type.

        Args:
            strategy_type: The type to look up

        Returns:
            The policy instance or None if not registered
        """
        return self._strategies.get(strategy_type)

    @property
    def available_strategies(self) -> list:
        """List of registered strategy types."""
        return list(self._strategies.keys())

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query to determine the best policy.

        Args:
            query: The query to classify

        Returns:
            ClassificationResult with question type, strategy, and metadata
        """
        return self._classifier.classify(query)

    def select_strategy(self, query: str) -> tuple:
        """Select the best policy for a query.

        Args:
            query: The query to route

        Returns:
            Tuple of (strategy_type, classification_result, override_applied)
        """
        if not self._enable_classification:
            if self._log_classification:
                logger.debug(f"Classification disabled, using default: {self._default_strategy_type.value}")
            return self._default_strategy_type, None, False

        # Classify the question
        classification = self.classify(query)
        override_applied = False

        # Check for strategy override based on question type
        question_type_key = classification.question_type.value.upper()
        if question_type_key in self._strategy_overrides:
            override_strategy_name = self._strategy_overrides[question_type_key]
            override_type = STRATEGY_NAME_TO_TYPE.get(override_strategy_name.lower())
            if override_type:
                if self._log_classification:
                    logger.info(
                        f"Question classified as {classification.question_type.value} "
                        f"-> Override applied: {override_strategy_name} "
                        f"(original: {classification.strategy.value})"
                    )
                target_type = override_type
                override_applied = True
            else:
                logger.warning(f"Invalid override strategy: {override_strategy_name}")
                target_type = CLASSIFIER_TO_STRATEGY.get(
                    classification.strategy,
                    self._default_strategy_type
                )
        else:
            # Map classifier strategy to our strategy type
            target_type = CLASSIFIER_TO_STRATEGY.get(
                classification.strategy,
                self._default_strategy_type
            )

        if self._log_classification and not override_applied:
            logger.info(
                f"Question classified as {classification.question_type.value} "
                f"-> {classification.strategy.value} (confidence: {classification.confidence:.2f})"
            )

        # Check if we have this policy registered
        if target_type not in self._strategies:
            logger.warning(
                f"Policy {target_type.value} not registered, "
                f"falling back to {self._default_strategy_type.value}"
            )
            target_type = self._default_strategy_type

            # Double-check that default is registered (safety check)
            if target_type not in self._strategies:
                # Last resort: use any registered policy
                if self._strategies:
                    target_type = next(iter(self._strategies.keys()))
                    logger.warning(f"Default policy also not registered, using {target_type.value}")

        return target_type, classification, override_applied

    async def route_and_retrieve(
        self,
        query: str,
        context: RetrievalContext,
    ) -> RetrievalResult:
        """Route a query to the appropriate policy and execute retrieval.

        This is the main entry point for policy-based retrieval. It:
        1. Classifies the query
        2. Selects the appropriate policy (with optional override)
        3. Executes the retrieval
        4. Adds classification metadata to results

        Args:
            query: The query to process
            context: RetrievalContext with indices and configuration

        Returns:
            RetrievalResult with results and metadata
        """
        # Select policy
        strategy_type, classification, override_applied = self.select_strategy(query)

        # Get policy instance
        strategy = self._strategies.get(strategy_type)
        if strategy is None:
            raise ValueError(
                f"No policy registered for type {strategy_type.value}. "
                f"Available: {[s.value for s in self.available_strategies]}"
            )

        if self._log_classification:
            logger.info(f"Executing policy: {strategy.name}")

        # Execute retrieval
        result = await strategy.retrieve(query, context)

        # Add classification metadata
        if classification:
            result.metadata["classification"] = {
                "question_type": classification.question_type.value,
                "classifier_strategy": classification.strategy.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "detected_patterns": classification.detected_patterns,
                "override_applied": override_applied,
            }

        result.metadata["selected_strategy"] = strategy_type.value

        return result


def create_default_router(
    config: Optional[Any] = None,
    enable_classification: bool = True,
) -> StrategyRouter:
    """Create a StrategyRouter with default policies registered.

    This factory function creates a router with all standard policies
    pre-registered, ready for use. It reads configuration from
    config.question_classification_config if available.

    Args:
        config: Optional ExperimentConfig for policy configuration.
                Reads question_classification_config for:
                - default_strategy: Fallback strategy name
                - strategy_overrides: Dict of question_type -> strategy
                - log_classification: Whether to log classifications
        enable_classification: If False, always use default strategy

    Returns:
        Configured StrategyRouter instance
    """
    from .policies import (
        GECClusterRerankPolicy,
        GECInsertAfterHitPolicy,
        AgenticOnlyPolicy,
    )

    # Read question classification config from ExperimentConfig
    qc_config: Dict[str, Any] = {}
    if config is not None:
        qc_config = getattr(config, 'question_classification_config', {})

    # Parse config values
    default_strategy_name = qc_config.get('default_strategy', 'gec_insert_after_hit')
    default_strategy_type = STRATEGY_NAME_TO_TYPE.get(
        default_strategy_name.lower(),
        StrategyType.GEC_INSERT_AFTER_HIT
    )
    strategy_overrides = qc_config.get('strategy_overrides', {})
    log_classification = qc_config.get('log_classification', True)

    router = StrategyRouter(
        classifier=QuestionClassifier(),
        default_strategy_type=default_strategy_type,
        enable_classification=enable_classification,
        strategy_overrides=strategy_overrides,
        log_classification=log_classification,
    )

    # Register all policies
    router.register_strategy(
        StrategyType.GEC_CLUSTER_RERANK,
        GECClusterRerankPolicy(config=config)
    )
    router.register_strategy(
        StrategyType.GEC_INSERT_AFTER_HIT,
        GECInsertAfterHitPolicy(config=config)
    )
    router.register_strategy(
        StrategyType.AGENTIC_ONLY,
        AgenticOnlyPolicy(config=config)
    )

    logger.info(
        f"Created router with policies: "
        f"{[s.value for s in router.available_strategies]}, "
        f"default: {default_strategy_type.value}, "
        f"overrides: {strategy_overrides}"
    )

    return router
