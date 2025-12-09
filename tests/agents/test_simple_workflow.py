"""Test script for simple_retrieval workflow.

This script tests the simple_retrieval workflow configuration and building.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.context import ExecutionContext
from src.orchestration.state import create_initial_retrieval_state
from src.orchestration import create_workflow
from src.orchestration.config_loader import ConfigLoader


class TestSimpleWorkflowConfig:
    """Test simple_retrieval.yaml configuration loading."""

    def test_config_loads_successfully(self):
        """Test that the config file can be loaded."""
        loader = ConfigLoader()
        config = loader.load("simple_retrieval")

        assert config.name == "simple_retrieval"
        assert len(config.nodes) > 0
        assert len(config.edges) > 0

    def test_config_has_required_nodes(self):
        """Test that required nodes are defined."""
        loader = ConfigLoader()
        config = loader.load("simple_retrieval")

        node_names = [n.name for n in config.nodes]
        # simple_retrieval should have a retrieval node
        assert "retrieval" in node_names


class TestSimpleWorkflowBuilder:
    """Test simple workflow building."""

    def test_builds_workflow_successfully(self):
        """Test that workflow can be built from config."""
        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=Mock(),
            rerank_service=Mock(),
            llm_provider=None,
            project_root=project_root
        )

        workflow = create_workflow("simple_retrieval", context)

        assert workflow is not None
        assert hasattr(workflow, "ainvoke")

    def test_workflow_has_nodes(self):
        """Test that built workflow has nodes."""
        context = ExecutionContext(
            memory_index=Mock(),
            bm25_index=None,
            cluster_index=None,
            vectorize_service=Mock(),
            rerank_service=Mock(),
            llm_provider=None,
            project_root=project_root
        )

        workflow = create_workflow("simple_retrieval", context)

        # Check that workflow has nodes
        assert len(workflow.nodes) > 0


class TestInitialState:
    """Test initial state creation."""

    def test_creates_initial_state(self):
        """Test that initial state is created correctly."""
        state = create_initial_retrieval_state(
            query="What activities did I do yesterday?",
            top_k=50,
            rerank_top_k=20
        )

        assert state["query"] == "What activities did I do yesterday?"
        assert state["top_k"] == 50
        assert state["rerank_top_k"] == 20
        assert state["documents"] == []

    def test_initial_state_with_defaults(self):
        """Test initial state with default values."""
        state = create_initial_retrieval_state(
            query="Test query"
        )

        assert state["query"] == "Test query"
        assert "documents" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
