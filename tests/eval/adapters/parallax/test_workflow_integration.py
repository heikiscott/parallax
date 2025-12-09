"""Tests for LangGraph Workflow Integration in Parallax Adapter.

These tests verify that the workflow integration module correctly
bridges LangGraph workflows with the Parallax evaluation pipeline.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestExecutionContextCreation:
    """Test ExecutionContext creation for workflows."""

    def test_creates_context_with_all_dependencies(self):
        """Test that context is created with all required dependencies."""
        from eval.adapters.parallax.workflow_integration import create_execution_context

        # Create mock dependencies
        mock_emb_index = Mock()
        mock_emb_index.embeddings = [[0.1, 0.2, 0.3]]
        mock_bm25 = Mock()
        mock_docs = [{"unit_id": "u1", "narrative": "Test doc"}]
        mock_cluster = Mock()
        mock_llm = Mock()

        context = create_execution_context(
            memory_index=mock_emb_index,
            bm25_index=mock_bm25,
            docs=mock_docs,
            cluster_index=mock_cluster,
            llm_provider=mock_llm,
        )

        assert context is not None
        assert context.bm25_index == mock_bm25
        assert context.cluster_index == mock_cluster
        assert context.llm_provider == mock_llm

    def test_context_memory_index_wrapper_get_all_docs(self):
        """Test that MemoryIndexWrapper provides get_all_docs()."""
        from eval.adapters.parallax.workflow_integration import create_execution_context

        mock_docs = [
            {"unit_id": "u1", "narrative": "Doc 1"},
            {"unit_id": "u2", "narrative": "Doc 2"},
        ]

        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=mock_docs,
        )

        # Test get_all_docs
        all_docs = context.memory_index.get_all_docs()
        assert len(all_docs) == 2
        assert all_docs[0]["unit_id"] == "u1"

    def test_context_memory_index_wrapper_memories_property(self):
        """Test that MemoryIndexWrapper provides memories property."""
        from eval.adapters.parallax.workflow_integration import create_execution_context

        mock_docs = [
            {"unit_id": "u1", "narrative": "Doc 1"},
            {"unit_id": "u2", "narrative": "Doc 2"},
        ]

        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=mock_docs,
        )

        # Test memories property (for legacy node compatibility)
        memories = context.memory_index.memories
        assert "u1" in memories
        assert memories["u1"].id == "u1"
        assert memories["u1"].narrative == "Doc 1"

    def test_context_with_custom_config(self):
        """Test that custom config is merged into context."""
        from eval.adapters.parallax.workflow_integration import create_execution_context

        custom_config = {
            "emb_top_k": 100,
            "use_reranker": True,
        }

        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=[],
            config=custom_config,
        )

        assert context.config.get("emb_top_k") == 100
        assert context.config.get("use_reranker") is True


class TestWorkflowCreation:
    """Test LangGraph workflow creation."""

    def test_creates_workflow_from_name(self):
        """Test that workflow is created from config name."""
        from eval.adapters.parallax.workflow_integration import (
            create_execution_context,
            create_retrieval_workflow,
        )

        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=[],
        )

        workflow = create_retrieval_workflow("simple_retrieval", context)

        assert workflow is not None
        assert hasattr(workflow, "ainvoke")

    def test_creates_adaptive_workflow(self):
        """Test that adaptive_retrieval workflow is created correctly."""
        from eval.adapters.parallax.workflow_integration import (
            create_execution_context,
            create_retrieval_workflow,
        )

        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=[],
            llm_provider=Mock(),
        )

        workflow = create_retrieval_workflow("adaptive_retrieval", context)

        assert workflow is not None
        # Check that expected nodes exist
        node_names = list(workflow.nodes.keys())
        assert "classify_question" in node_names


class TestWorkflowExecution:
    """Test workflow execution and result conversion."""

    @pytest.mark.asyncio
    async def test_run_workflow_retrieval_returns_results(self):
        """Test that run_workflow_retrieval returns properly formatted results."""
        from eval.adapters.parallax.workflow_integration import run_workflow_retrieval

        # Create a mock workflow
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "documents": [
                {
                    "id": "u1",
                    "content": "Alice went to the park",
                    "score": 0.95,
                    "metadata": {
                        "retrieval_method": "hybrid",
                        "original_doc": {"unit_id": "u1", "subject": "Alice"},
                    },
                },
                {
                    "id": "u2",
                    "content": "Bob attended a meeting",
                    "score": 0.85,
                    "metadata": {"retrieval_method": "hybrid"},
                },
            ],
            "metadata": {"total_time": 0.5},
            "question_type": "event_temporal",
        }

        results, metadata = await run_workflow_retrieval(
            workflow=mock_workflow,
            query="When did Alice go to the park?",
            top_k=50,
            rerank_top_k=20,
        )

        # Check results format
        assert len(results) == 2
        doc1, score1 = results[0]
        assert doc1["unit_id"] == "u1"
        assert doc1["narrative"] == "Alice went to the park"
        assert doc1["subject"] == "Alice"  # From original_doc
        assert score1 == 0.95

        # Check metadata
        assert metadata["workflow_executed"] is True
        assert metadata["question_type"] == "event_temporal"

    @pytest.mark.asyncio
    async def test_run_workflow_retrieval_handles_empty_results(self):
        """Test handling of empty workflow results."""
        from eval.adapters.parallax.workflow_integration import run_workflow_retrieval

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "documents": [],
            "metadata": {},
        }

        results, metadata = await run_workflow_retrieval(
            workflow=mock_workflow,
            query="Test query",
        )

        assert len(results) == 0
        assert metadata["workflow_executed"] is True


class TestWorkflowSearchOneShot:
    """Test the one-shot workflow_search convenience function."""

    @pytest.mark.asyncio
    async def test_workflow_search_creates_context_and_workflow(self):
        """Test that workflow_search correctly creates context and workflow.

        This test verifies the function structure without executing the full workflow,
        as the full workflow requires many dependencies (core.nlp, etc.).
        """
        from eval.adapters.parallax.workflow_integration import (
            create_execution_context,
            create_retrieval_workflow,
        )

        # Test context creation (already tested above, but verify it works in this flow)
        mock_docs = [{"unit_id": "u1", "narrative": "Test doc"}]
        context = create_execution_context(
            memory_index=Mock(),
            bm25_index=Mock(),
            docs=mock_docs,
            llm_provider=Mock(),
        )

        assert context is not None
        assert context.memory_index.get_all_docs() == mock_docs

        # Test workflow creation (this should work with mocked context)
        workflow = create_retrieval_workflow("adaptive_retrieval", context)
        assert workflow is not None
        assert "classify_question" in workflow.nodes

    @pytest.mark.asyncio
    async def test_workflow_search_function_signature(self):
        """Test that workflow_search function has correct signature."""
        from eval.adapters.parallax.workflow_integration import workflow_search
        import inspect

        sig = inspect.signature(workflow_search)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "query" in params
        assert "workflow_name" in params
        assert "emb_index" in params
        assert "bm25" in params
        assert "docs" in params

        # Check optional parameters
        assert "cluster_index" in params
        assert "llm_provider" in params
        assert "config" in params
        assert "top_k" in params
        assert "rerank_top_k" in params


class TestExperimentConfigWorkflow:
    """Test ExperimentConfig workflow-related fields."""

    def test_config_has_workflow_fields(self):
        """Test that ExperimentConfig has workflow-related fields."""
        from eval.adapters.parallax.config import ExperimentConfig

        config = ExperimentConfig()

        assert hasattr(config, "workflow_name")
        assert config.workflow_name == "adaptive_retrieval"  # default value

    def test_retrieval_mode_includes_workflow(self):
        """Test that retrieval_mode supports 'workflow' option."""
        from eval.adapters.parallax.config import ExperimentConfig

        config = ExperimentConfig()
        config.retrieval_mode = "workflow"

        assert config.retrieval_mode == "workflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
