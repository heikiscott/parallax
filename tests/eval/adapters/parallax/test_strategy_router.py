"""Tests for the Strategy Router module."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))
sys.path.insert(0, str(project_root))

from agents.question_classifier import (
    QuestionType,
    RetrievalStrategy as ClassifierStrategy,
)
from adapters.parallax.strategy import (
    StrategyRouter,
    StrategyType,
    RetrievalContext,
    RetrievalResult,
    BaseRetrievalStrategy,
    create_default_router,
    get_strategy_for_query,
)


class TestStrategyType:
    """Test StrategyType enum."""

    def test_strategy_types_exist(self):
        """Test that all strategy types are defined."""
        assert StrategyType.GEC_CLUSTER_RERANK.value == "gec_cluster_rerank"
        assert StrategyType.GEC_INSERT_AFTER_HIT.value == "gec_insert_after_hit"
        assert StrategyType.AGENTIC_ONLY.value == "agentic_only"


class TestRetrievalContext:
    """Test RetrievalContext dataclass."""

    def test_create_context(self):
        """Test creating a retrieval context."""
        ctx = RetrievalContext(
            emb_index={"test": "emb"},
            bm25={"test": "bm25"},
            docs=[{"doc": 1}],
        )
        assert ctx.emb_index == {"test": "emb"}
        assert ctx.bm25 == {"test": "bm25"}
        assert len(ctx.docs) == 1
        assert ctx.cluster_index is None
        assert ctx.llm_provider is None

    def test_context_with_all_fields(self):
        """Test creating a context with all fields."""
        mock_provider = MagicMock()
        mock_cluster = MagicMock()

        ctx = RetrievalContext(
            emb_index={"emb": True},
            bm25={"bm25": True},
            docs=[{"doc": 1}, {"doc": 2}],
            cluster_index=mock_cluster,
            llm_provider=mock_provider,
            llm_config={"model": "test"},
            config={"test": True},
            enable_traversal_stats=True,
        )

        assert ctx.cluster_index == mock_cluster
        assert ctx.llm_provider == mock_provider
        assert ctx.enable_traversal_stats is True


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_create_result(self):
        """Test creating a retrieval result."""
        results = [({"unit_id": "u1"}, 0.9), ({"unit_id": "u2"}, 0.8)]
        result = RetrievalResult(
            results=results,
            metadata={"test": True},
            strategy_type=StrategyType.GEC_CLUSTER_RERANK,
        )

        assert result.count == 2
        assert result.unit_ids == ["u1", "u2"]
        assert result.strategy_type == StrategyType.GEC_CLUSTER_RERANK

    def test_result_to_dict(self):
        """Test serializing result to dict."""
        results = [({"unit_id": "u1"}, 0.9)]
        result = RetrievalResult(
            results=results,
            metadata={"key": "value"},
            strategy_type=StrategyType.AGENTIC_ONLY,
        )

        d = result.to_dict()
        assert d["count"] == 1
        assert d["strategy_type"] == "agentic_only"
        assert len(d["results"]) == 1
        assert d["metadata"]["key"] == "value"


class TestStrategyRouter:
    """Test StrategyRouter class."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        return StrategyRouter()

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = MagicMock(spec=BaseRetrievalStrategy)
        strategy.strategy_type = StrategyType.GEC_CLUSTER_RERANK
        strategy.name = "gec_cluster_rerank"
        return strategy

    def test_register_strategy(self, router, mock_strategy):
        """Test registering a strategy."""
        router.register_strategy(StrategyType.GEC_CLUSTER_RERANK, mock_strategy)
        assert StrategyType.GEC_CLUSTER_RERANK in router.available_strategies
        assert router.get_strategy(StrategyType.GEC_CLUSTER_RERANK) == mock_strategy

    def test_unregister_strategy(self, router, mock_strategy):
        """Test unregistering a strategy."""
        router.register_strategy(StrategyType.GEC_CLUSTER_RERANK, mock_strategy)
        router.unregister_strategy(StrategyType.GEC_CLUSTER_RERANK)
        assert StrategyType.GEC_CLUSTER_RERANK not in router.available_strategies

    def test_classify_event_temporal(self, router):
        """Test classifying an event temporal question."""
        result = router.classify("When did Caroline go to the conference?")
        assert result.question_type == QuestionType.EVENT_TEMPORAL
        assert result.strategy == ClassifierStrategy.GEC_CLUSTER_RERANK

    def test_classify_attribute_identity(self, router):
        """Test classifying an attribute identity question."""
        result = router.classify("What is Caroline's identity?")
        assert result.question_type == QuestionType.ATTRIBUTE_IDENTITY
        assert result.strategy == ClassifierStrategy.AGENTIC_ONLY

    def test_classify_reasoning_hypothetical(self, router):
        """Test classifying a reasoning hypothetical question."""
        result = router.classify("Would Caroline pursue writing as a career?")
        assert result.question_type == QuestionType.REASONING_HYPOTHETICAL
        assert result.strategy == ClassifierStrategy.GEC_INSERT_AFTER_HIT

    def test_select_strategy_with_classification_disabled(self, router, mock_strategy):
        """Test selecting strategy with classification disabled."""
        router = StrategyRouter(enable_classification=False)
        strategy_type, classification, override_applied = router.select_strategy("any question")

        assert strategy_type == StrategyType.GEC_INSERT_AFTER_HIT  # default
        assert classification is None
        assert override_applied is False

    def test_select_strategy_event_temporal(self, router, mock_strategy):
        """Test selecting strategy for event temporal question."""
        router.register_strategy(StrategyType.GEC_CLUSTER_RERANK, mock_strategy)

        strategy_type, classification, override_applied = router.select_strategy(
            "When did Caroline attend the meeting?"
        )

        assert strategy_type == StrategyType.GEC_CLUSTER_RERANK
        assert classification is not None
        assert classification.question_type == QuestionType.EVENT_TEMPORAL
        assert override_applied is False

    def test_select_strategy_with_override(self, router, mock_strategy):
        """Test selecting strategy with question type override."""
        router = StrategyRouter(
            enable_classification=True,
            strategy_overrides={"EVENT_TEMPORAL": "agentic_only"},
        )
        router.register_strategy(StrategyType.AGENTIC_ONLY, mock_strategy)

        strategy_type, classification, override_applied = router.select_strategy(
            "When did Caroline attend the meeting?"
        )

        assert strategy_type == StrategyType.AGENTIC_ONLY
        assert classification is not None
        assert classification.question_type == QuestionType.EVENT_TEMPORAL
        assert override_applied is True


class TestGetStrategyForQuery:
    """Test the get_strategy_for_query helper function."""

    def test_event_temporal_query(self):
        """Test getting strategy for event temporal query."""
        strategy, info = get_strategy_for_query("When did Caroline go to the meeting?")

        assert strategy == "gec_cluster_rerank"
        assert info["question_type"] == "event_temporal"

    def test_attribute_identity_query(self):
        """Test getting strategy for attribute identity query."""
        strategy, info = get_strategy_for_query("What is Caroline's relationship status?")

        assert strategy == "agentic_only"
        assert info["question_type"] == "attribute_identity"

    def test_reasoning_hypothetical_query(self):
        """Test getting strategy for reasoning hypothetical query."""
        strategy, info = get_strategy_for_query(
            "Would Caroline likely have Dr. Seuss books?"
        )

        assert strategy == "gec_insert_after_hit"
        assert info["question_type"] == "reasoning_hypothetical"

    def test_with_strategy_override(self):
        """Test getting strategy with config override."""
        # Create mock config with override
        mock_config = MagicMock()
        mock_config.question_classification_config = {
            "strategy_overrides": {
                "EVENT_TEMPORAL": "agentic_only",  # Override to use agentic_only
            }
        }

        strategy, info = get_strategy_for_query(
            "When did Caroline attend the meeting?",
            config=mock_config,
        )

        assert strategy == "agentic_only"
        assert info["override_applied"] is True
        assert info["original_strategy"] == "gec_cluster_rerank"


class TestCreateDefaultRouter:
    """Test the create_default_router factory function."""

    def test_create_router_with_all_strategies(self):
        """Test that default router has all strategies registered."""
        router = create_default_router()

        assert StrategyType.GEC_CLUSTER_RERANK in router.available_strategies
        assert StrategyType.GEC_INSERT_AFTER_HIT in router.available_strategies
        assert StrategyType.AGENTIC_ONLY in router.available_strategies

    def test_router_with_classification_disabled(self):
        """Test creating router with classification disabled."""
        router = create_default_router(enable_classification=False)

        # Should still have strategies, but won't classify
        assert len(router.available_strategies) == 3


class TestConfigIsolation:
    """Test that config is not mutated by strategies."""

    def test_config_wrapper_does_not_modify_original(self):
        """Test that _ConfigWrapper does not modify the original config."""
        from adapters.parallax.strategy.strategies import _create_config_with_overrides

        # Create a mock config
        class MockConfig:
            group_event_cluster_retrieval_config = {
                "enable_group_event_cluster_retrieval": True,
                "expansion_strategy": "insert_after_hit",
                "max_clusters": 5,
            }
            other_setting = "original"

        original_config = MockConfig()
        original_gec_config = original_config.group_event_cluster_retrieval_config.copy()

        # Create wrapper with overrides
        wrapped = _create_config_with_overrides(original_config, {
            "enable_group_event_cluster_retrieval": False,
            "expansion_strategy": "cluster_rerank",
        })

        # Verify wrapper has the overrides
        assert wrapped.group_event_cluster_retrieval_config["enable_group_event_cluster_retrieval"] is False
        assert wrapped.group_event_cluster_retrieval_config["expansion_strategy"] == "cluster_rerank"
        assert wrapped.group_event_cluster_retrieval_config["max_clusters"] == 5  # preserved

        # Verify original is unchanged
        assert original_config.group_event_cluster_retrieval_config == original_gec_config
        assert original_config.group_event_cluster_retrieval_config["enable_group_event_cluster_retrieval"] is True
        assert original_config.group_event_cluster_retrieval_config["expansion_strategy"] == "insert_after_hit"

        # Verify other attributes are delegated
        assert wrapped.other_setting == "original"

    def test_multiple_wrappers_are_independent(self):
        """Test that multiple wrappers from the same config are independent."""
        from adapters.parallax.strategy.strategies import _create_config_with_overrides

        class MockConfig:
            group_event_cluster_retrieval_config = {
                "enable_group_event_cluster_retrieval": True,
                "expansion_strategy": "insert_after_hit",
            }

        original_config = MockConfig()

        # Create two different wrappers
        wrapper1 = _create_config_with_overrides(original_config, {
            "expansion_strategy": "cluster_rerank",
        })
        wrapper2 = _create_config_with_overrides(original_config, {
            "enable_group_event_cluster_retrieval": False,
        })

        # Verify they are independent
        assert wrapper1.group_event_cluster_retrieval_config["expansion_strategy"] == "cluster_rerank"
        assert wrapper1.group_event_cluster_retrieval_config["enable_group_event_cluster_retrieval"] is True

        assert wrapper2.group_event_cluster_retrieval_config["expansion_strategy"] == "insert_after_hit"
        assert wrapper2.group_event_cluster_retrieval_config["enable_group_event_cluster_retrieval"] is False

        # Verify original is still unchanged
        assert original_config.group_event_cluster_retrieval_config["expansion_strategy"] == "insert_after_hit"
        assert original_config.group_event_cluster_retrieval_config["enable_group_event_cluster_retrieval"] is True


class TestLoCoMoQuestionRouting:
    """Test routing of LoCoMo evaluation questions."""

    LOCOMO_QUESTIONS = [
        # Event-temporal -> GEC_CLUSTER_RERANK
        ("When did Caroline go to the LGBTQ support group?", "gec_cluster_rerank"),
        ("When did Melanie paint a sunrise?", "gec_cluster_rerank"),
        ("When did Caroline have a picnic?", "gec_cluster_rerank"),

        # Event-activity -> GEC_CLUSTER_RERANK
        ("What activities does Melanie partake in?", "gec_cluster_rerank"),
        ("What does Melanie do to destress?", "gec_cluster_rerank"),

        # Event-aggregation -> GEC_CLUSTER_RERANK
        ("What books has Melanie read?", "gec_cluster_rerank"),
        ("Where has Melanie camped?", "gec_cluster_rerank"),

        # Attribute-identity -> AGENTIC_ONLY
        ("What is Caroline's identity?", "agentic_only"),
        ("What is Caroline's relationship status?", "agentic_only"),

        # Attribute-preference -> AGENTIC_ONLY
        ("What do Melanie's kids like?", "agentic_only"),

        # Attribute-location -> AGENTIC_ONLY
        ("Where did Caroline move from 4 years ago?", "agentic_only"),

        # Reasoning-hypothetical -> GEC_INSERT_AFTER_HIT
        ("Would Caroline pursue writing as a career option?", "gec_insert_after_hit"),
        ("Would Caroline likely have Dr. Seuss books on her bookshelf?", "gec_insert_after_hit"),

        # Time-calculation -> AGENTIC_ONLY
        ("How long has Caroline had her current group of friends for?", "agentic_only"),
        ("How long ago was Caroline's 18th birthday?", "agentic_only"),
    ]

    @pytest.mark.parametrize("question,expected_strategy", LOCOMO_QUESTIONS)
    def test_locomo_question_routing(self, question, expected_strategy):
        """Test that LoCoMo questions are routed to correct strategies."""
        strategy, info = get_strategy_for_query(question)
        assert strategy == expected_strategy, (
            f"Question: {question}\n"
            f"Expected: {expected_strategy}\n"
            f"Got: {strategy}\n"
            f"Type: {info['question_type']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
