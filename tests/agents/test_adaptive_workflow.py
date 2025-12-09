"""Test script for adaptive_retrieval workflow.

This script tests the adaptive retrieval workflow which routes queries
to different retrieval strategies based on question classification.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestAdaptiveWorkflowConfig:
    """Test adaptive_retrieval.yaml configuration loading."""

    def test_config_loads_successfully(self):
        """Test that the config file can be loaded."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        assert config.name == "adaptive_retrieval"
        # 3 nodes: classify, lightweight, agentic (legacy pipelines include reranking internally)
        assert len(config.nodes) == 3
        # 3 edges: START->classify, lightweight->END, agentic->END
        assert len(config.edges) == 3
        assert len(config.conditional_edges) == 1

    def test_config_has_correct_nodes(self):
        """Test that all required nodes are defined."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        node_names = [n.name for n in config.nodes]
        assert "classify_question" in node_names
        assert "lightweight_retrieval" in node_names
        assert "agentic_retrieval" in node_names
        # Note: No separate rerank nodes - legacy pipelines handle reranking internally

    def test_config_has_conditional_routing(self):
        """Test that conditional routing is configured."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        cond_edge = config.conditional_edges[0]
        assert cond_edge.source == "classify_question"
        assert cond_edge.router == "route_by_question_type"
        assert "lightweight_retrieval" in cond_edge.destinations
        assert "agentic_retrieval" in cond_edge.destinations


class TestRouteByQuestionType:
    """Test the route_by_question_type router function."""

    def test_routes_event_temporal_to_agentic(self):
        """Event temporal questions should route to agentic retrieval."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "When did Caroline go to the conference?"}
        result = route_by_question_type(state)
        assert result == "agentic_retrieval"

    def test_routes_event_activity_to_agentic(self):
        """Event activity questions should route to agentic retrieval."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "What activities does Melanie partake in?"}
        result = route_by_question_type(state)
        assert result == "agentic_retrieval"

    def test_routes_attribute_preference_to_lightweight(self):
        """Attribute preference questions should route to lightweight retrieval."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "What is Caroline's favorite book?"}
        result = route_by_question_type(state)
        assert result == "lightweight_retrieval"

    def test_routes_attribute_location_to_lightweight(self):
        """Attribute location questions should route to lightweight retrieval."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "Where does Caroline live?"}
        result = route_by_question_type(state)
        assert result == "lightweight_retrieval"

    def test_routes_aggregation_to_agentic(self):
        """Aggregation questions should route to agentic retrieval."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "What books has Melanie read?"}
        result = route_by_question_type(state)
        assert result == "agentic_retrieval"

    def test_routes_general_questions_to_agentic(self):
        """General/unknown questions should route to agentic (safe fallback)."""
        from src.orchestration.workflow_builder import route_by_question_type

        state = {"query": "Tell me about the weather last week"}
        result = route_by_question_type(state)
        assert result == "agentic_retrieval"


class TestWorkflowBuilder:
    """Test workflow building from config."""

    def test_builds_workflow_successfully(self):
        """Test that workflow can be built from config."""
        from src.orchestration.context import ExecutionContext
        from src.orchestration.workflow_builder import WorkflowBuilder
        from src.orchestration.config_loader import ConfigLoader

        # Create minimal mock context
        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=None,
            rerank_service=None,
            llm_provider=None
        )

        builder = WorkflowBuilder(context)
        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        graph = builder.build_from_config(config)

        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_workflow_has_all_nodes(self):
        """Test that built workflow has all expected nodes."""
        from src.orchestration.context import ExecutionContext
        from src.orchestration.workflow_builder import WorkflowBuilder
        from src.orchestration.config_loader import ConfigLoader

        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=None,
            rerank_service=None,
            llm_provider=None
        )

        builder = WorkflowBuilder(context)
        config = ConfigLoader().load("adaptive_retrieval")
        graph = builder.build_from_config(config)

        # Check nodes exist
        node_names = list(graph.nodes.keys())
        assert "classify_question" in node_names
        assert "lightweight_retrieval" in node_names
        assert "agentic_retrieval" in node_names


class TestClassifyQuestionNode:
    """Test the classify_question_node function."""

    @pytest.mark.asyncio
    async def test_classifies_event_question(self):
        """Test classification of event temporal question."""
        from src.orchestration.nodes.retrieval.classification_nodes import classify_question_node
        from src.orchestration.context import ExecutionContext

        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=None,
            rerank_service=None,
            llm_provider=None
        )

        state = {"query": "When did Caroline attend the meeting?"}
        result = await classify_question_node(state, context)

        assert "question_type" in result
        assert "retrieval_strategy" in result
        assert result["question_type"] == "event_temporal"
        assert result["retrieval_strategy"] == "gec_cluster_rerank"

    @pytest.mark.asyncio
    async def test_classifies_attribute_question(self):
        """Test classification of attribute preference question."""
        from src.orchestration.nodes.retrieval.classification_nodes import classify_question_node
        from src.orchestration.context import ExecutionContext

        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=None,
            rerank_service=None,
            llm_provider=None
        )

        state = {"query": "What does Melanie like to eat?"}
        result = await classify_question_node(state, context)

        assert result["question_type"] == "attribute_preference"
        assert result["retrieval_strategy"] == "agentic_only"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
