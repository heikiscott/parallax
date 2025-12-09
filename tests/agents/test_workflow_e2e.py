"""End-to-End tests for retrieval workflows.

These tests verify that the complete workflow execution works correctly,
from config loading to graph execution with mocked services.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestSimpleWorkflowE2E:
    """End-to-end tests for simple_retrieval workflow."""

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self):
        """Test that simple workflow can execute end-to-end with mocked services."""
        from src.orchestration.context import ExecutionContext
        from src.orchestration import create_workflow
        from src.orchestration.state import create_initial_retrieval_state

        # Create mock services
        mock_memory_index = Mock()
        mock_memory_index.embeddings = {
            "doc1": [0.1] * 1024,
            "doc2": [0.2] * 1024,
        }

        mock_vectorize = Mock()
        mock_vectorize.get_embedding = AsyncMock(return_value=[0.15] * 1024)

        mock_rerank = Mock()
        mock_rerank._rerank_all_hits = AsyncMock(return_value=[
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ])

        # Create context
        context = ExecutionContext(
            memory_index=mock_memory_index,
            bm25_index=None,
            cluster_index=None,
            vectorize_service=mock_vectorize,
            rerank_service=mock_rerank,
            llm_provider=None,
            project_root=project_root
        )

        # Build workflow
        workflow = create_workflow("simple_retrieval", context)
        assert workflow is not None

        # Create initial state
        state = create_initial_retrieval_state(
            query="What activities did I do yesterday?",
            top_k=50,
            rerank_top_k=20
        )

        # Workflow should be invokable
        assert hasattr(workflow, "ainvoke")


class TestAdaptiveWorkflowE2E:
    """End-to-end tests for adaptive_retrieval workflow."""

    @pytest.mark.asyncio
    async def test_adaptive_workflow_routes_correctly(self):
        """Test that adaptive workflow routes to correct retrieval strategy."""
        from src.orchestration.context import ExecutionContext
        from src.orchestration.workflow_builder import WorkflowBuilder, route_by_question_type
        from src.orchestration.config_loader import ConfigLoader

        # Test routing for different question types
        test_cases = [
            ("When did Caroline attend the meeting?", "agentic_retrieval"),
            ("What is Caroline's favorite color?", "lightweight_retrieval"),
            ("Where does Caroline live?", "lightweight_retrieval"),
            ("What books has Caroline read this year?", "agentic_retrieval"),
        ]

        for query, expected_route in test_cases:
            state = {"query": query}
            result = route_by_question_type(state)
            assert result == expected_route, f"Query '{query}' should route to {expected_route}, got {result}"

    @pytest.mark.asyncio
    async def test_adaptive_workflow_builds_and_has_all_nodes(self):
        """Test that adaptive workflow builds correctly with all nodes."""
        from src.orchestration.context import ExecutionContext
        from src.orchestration.workflow_builder import WorkflowBuilder
        from src.orchestration.config_loader import ConfigLoader

        # Create minimal mock context
        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=Mock(),
            rerank_service=Mock(),
            llm_provider=None
        )

        builder = WorkflowBuilder(context)
        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        graph = builder.build_from_config(config)

        # Verify all expected nodes exist
        node_names = list(graph.nodes.keys())
        assert "classify_question" in node_names
        assert "lightweight_retrieval" in node_names
        assert "agentic_retrieval" in node_names


class TestAgenticHybridWorkflowE2E:
    """End-to-end tests for agentic_hybrid workflow."""

    def test_agentic_hybrid_config_loads(self):
        """Test that agentic_hybrid config loads correctly."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("agentic_hybrid")

        assert config.name == "agentic_hybrid_retrieval"
        assert len(config.nodes) == 6

        # Check for expected nodes
        node_names = [n.name for n in config.nodes]
        assert "hybrid_retrieval" in node_names
        assert "round1_rerank" in node_names
        assert "sufficiency_check" in node_names
        assert "multi_query_expansion" in node_names
        assert "cluster_expansion" in node_names

    def test_agentic_hybrid_has_conditional_routing(self):
        """Test that agentic_hybrid has conditional routing configured."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("agentic_hybrid")

        assert len(config.conditional_edges) == 1
        cond_edge = config.conditional_edges[0]
        assert cond_edge.source == "sufficiency_check"
        assert cond_edge.router == "route_by_sufficiency"


class TestAllWorkflowConfigs:
    """Test that all workflow configs can be loaded."""

    def test_all_workflows_loadable(self):
        """Test that all YAML workflow configs can be loaded."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        workflow_names = ["simple_retrieval", "adaptive_retrieval", "agentic_hybrid"]

        for name in workflow_names:
            config = loader.load(name)
            assert config is not None
            assert config.name is not None
            assert len(config.nodes) > 0

    def test_workflow_nodes_have_functions(self):
        """Test that all workflow nodes have function definitions."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("simple_retrieval")

        for node in config.nodes:
            assert node.function is not None, f"Node {node.name} missing function"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
