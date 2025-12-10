"""Tests for legacy node wrappers.

These tests verify that the legacy nodes correctly wrap the existing
retrieval pipelines and integrate with the LangGraph workflow system.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path (same pattern as test_simple_workflow.py)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestLegacyNodesRegistration:
    """Test that legacy nodes are properly registered."""

    def test_nodes_are_registered(self):
        """Test that legacy nodes appear in NODE_REGISTRY."""
        from src.orchestration.nodes import NODE_REGISTRY, list_nodes

        node_names = list_nodes()
        assert "lightweight_retrieval_node" in node_names
        assert "agentic_retrieval_node" in node_names

    def test_get_node_returns_functions(self):
        """Test that get_node returns callable functions."""
        from src.orchestration.nodes import get_node

        lightweight_node = get_node("lightweight_retrieval_node")
        agentic_node = get_node("agentic_retrieval_node")

        assert callable(lightweight_node)
        assert callable(agentic_node)


class TestLightweightRetrievalNode:
    """Tests for lightweight_retrieval_node.

    Note: These tests mock the pipeline at the sys.modules level to avoid
    importing the actual pipeline modules which have complex dependencies.
    """

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        from src.orchestration.context import ExecutionContext

        context = Mock(spec=ExecutionContext)
        context.memory_index = Mock()
        context.memory_index.memories = {
            "mem1": Mock(id="mem1", narrative="Alice went to the park."),
            "mem2": Mock(id="mem2", narrative="Bob attended a meeting."),
        }
        context.bm25_index = Mock()
        context.get_config_value = Mock(side_effect=lambda k, d=None: {
            "emb_top_k": 50,
            "bm25_top_k": 50,
            "final_top_k": 20,
        }.get(k, d))

        return context

    @pytest.fixture
    def mock_state(self):
        """Create a mock retrieval state."""
        return {
            "query": "When did Alice go to the park?",
            "top_k": 20,
        }

    @pytest.mark.asyncio
    async def test_lightweight_node_calls_pipeline(self, mock_context, mock_state):
        """Test that lightweight node calls the pipeline correctly."""
        mock_results = [
            ({"unit_id": "mem1", "narrative": "Alice went to the park."}, 0.9),
            ({"unit_id": "mem2", "narrative": "Bob attended a meeting."}, 0.8),
        ]

        mock_pipeline = AsyncMock(return_value=(mock_results, {"retrieval_mode": "lightweight"}))

        # Create mock module
        mock_lightweight_module = MagicMock()
        mock_lightweight_module.lightweight_retrieval = mock_pipeline

        # Patch the module in sys.modules before the import happens in the function
        original_modules = {}
        modules_to_mock = [
            'src.retrieval.offline.pipelines.lightweight',
            'retrieval.offline.pipelines.lightweight',
        ]

        for mod_name in modules_to_mock:
            original_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_lightweight_module

        try:
            from src.orchestration.nodes.retrieval.legacy_nodes import lightweight_retrieval_node
            result = await lightweight_retrieval_node(mock_state, mock_context)

            assert "documents" in result
            assert "metadata" in result
            assert result["metadata"]["retrieval_method"] == "lightweight"
            assert len(result["documents"]) == 2
        finally:
            # Restore original modules
            for mod_name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original

    @pytest.mark.asyncio
    async def test_lightweight_node_converts_to_documents(self, mock_context, mock_state):
        """Test that results are converted to Document dicts."""
        mock_results = [
            ({"unit_id": "mem1", "narrative": "Test content"}, 0.95),
        ]

        mock_pipeline = AsyncMock(return_value=(mock_results, {}))
        mock_lightweight_module = MagicMock()
        mock_lightweight_module.lightweight_retrieval = mock_pipeline

        original_modules = {}
        modules_to_mock = [
            'src.retrieval.offline.pipelines.lightweight',
            'retrieval.offline.pipelines.lightweight',
        ]

        for mod_name in modules_to_mock:
            original_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_lightweight_module

        try:
            from src.orchestration.nodes.retrieval.legacy_nodes import lightweight_retrieval_node
            result = await lightweight_retrieval_node(mock_state, mock_context)

            doc = result["documents"][0]
            # Document is a TypedDict, so it's actually a dict
            assert isinstance(doc, dict)
            assert doc["id"] == "mem1"
            assert doc["content"] == "Test content"
            assert doc["score"] == 0.95
        finally:
            for mod_name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original


class TestAgenticRetrievalNode:
    """Tests for agentic_retrieval_node."""

    @pytest.fixture
    def mock_context_with_llm(self):
        """Create a mock execution context with LLM provider."""
        from src.orchestration.context import ExecutionContext

        context = Mock(spec=ExecutionContext)
        context.memory_index = Mock()
        context.memory_index.memories = {
            "mem1": Mock(id="mem1", narrative="Event description"),
        }
        context.bm25_index = Mock()
        context.cluster_index = Mock()
        context.llm_provider = Mock()
        context.get_config_value = Mock(side_effect=lambda k, d=None: {
            "emb_top_k": 50,
            "bm25_top_k": 50,
            "use_reranker": True,
            "use_multi_query": True,
            "llm_config": {"model": "gpt-4o-mini", "temperature": 0.0},
        }.get(k, d))

        return context

    @pytest.fixture
    def mock_context_no_llm(self):
        """Create a mock execution context without LLM provider."""
        from src.orchestration.context import ExecutionContext

        context = Mock(spec=ExecutionContext)
        context.memory_index = Mock()
        context.memory_index.memories = {}
        context.bm25_index = None
        context.llm_provider = None  # No LLM
        context.get_config_value = Mock(return_value=None)

        return context

    @pytest.fixture
    def mock_state(self):
        """Create a mock retrieval state."""
        return {
            "query": "What events did Alice attend last week?",
            "top_k": 20,
        }

    @pytest.mark.asyncio
    async def test_agentic_node_calls_pipeline(self, mock_context_with_llm, mock_state):
        """Test that agentic node calls the pipeline correctly."""
        mock_results = [
            ({"unit_id": "mem1", "narrative": "Event at conference"}, 0.92),
        ]

        mock_pipeline = AsyncMock(return_value=(mock_results, {
            "is_multi_round": False,
            "is_sufficient": True,
        }))

        mock_agentic_module = MagicMock()
        mock_agentic_module.agentic_retrieval = mock_pipeline

        original_modules = {}
        modules_to_mock = [
            'src.retrieval.offline.pipelines.agentic',
            'retrieval.offline.pipelines.agentic',
        ]

        for mod_name in modules_to_mock:
            original_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_agentic_module

        try:
            from src.orchestration.nodes.retrieval.legacy_nodes import agentic_retrieval_node
            result = await agentic_retrieval_node(mock_state, mock_context_with_llm)

            assert "documents" in result
            assert "metadata" in result
            assert result["metadata"]["retrieval_method"] == "agentic"
        finally:
            for mod_name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original

    @pytest.mark.asyncio
    async def test_agentic_node_fallback_without_llm(self, mock_context_no_llm, mock_state):
        """Test that agentic node falls back to lightweight when no LLM."""
        mock_results = [
            ({"unit_id": "mem1", "narrative": "Content"}, 0.8),
        ]

        mock_lightweight = AsyncMock(return_value=(mock_results, {}))
        mock_lightweight_module = MagicMock()
        mock_lightweight_module.lightweight_retrieval = mock_lightweight

        # Also need to mock agentic module since import happens before LLM check
        mock_agentic_module = MagicMock()
        mock_agentic_module.agentic_retrieval = AsyncMock()

        original_modules = {}
        modules_to_mock = [
            'src.retrieval.offline.pipelines.lightweight',
            'retrieval.offline.pipelines.lightweight',
            'src.retrieval.offline.pipelines.agentic',
            'retrieval.offline.pipelines.agentic',
        ]

        mock_modules = {
            'src.retrieval.offline.pipelines.lightweight': mock_lightweight_module,
            'retrieval.offline.pipelines.lightweight': mock_lightweight_module,
            'src.retrieval.offline.pipelines.agentic': mock_agentic_module,
            'retrieval.offline.pipelines.agentic': mock_agentic_module,
        }

        for mod_name in modules_to_mock:
            original_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_modules[mod_name]

        try:
            from src.orchestration.nodes.retrieval.legacy_nodes import agentic_retrieval_node
            result = await agentic_retrieval_node(mock_state, mock_context_no_llm)

            # Should fall back to lightweight
            assert result["metadata"]["retrieval_method"] == "lightweight"
        finally:
            for mod_name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original


class TestAdaptiveWorkflowConfig:
    """Test that adaptive_retrieval workflow config is valid."""

    def test_config_loads_successfully(self):
        """Test that adaptive_retrieval.yaml loads without errors."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        assert config.name == "adaptive_retrieval"
        assert len(config.nodes) >= 3  # classify, lightweight, agentic

    def test_config_has_required_nodes(self):
        """Test that config has all required nodes."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        node_names = [n.name for n in config.nodes]
        assert "classify_question" in node_names
        assert "lightweight_retrieval" in node_names
        assert "agentic_retrieval" in node_names

    def test_config_has_conditional_edges(self):
        """Test that config has conditional routing."""
        from src.orchestration.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load("adaptive_retrieval")

        assert len(config.conditional_edges) == 1
        cond_edge = config.conditional_edges[0]
        assert cond_edge.source == "classify_question"
        assert cond_edge.router == "route_by_question_type"


class TestConvertToDocuments:
    """Test the _convert_to_documents helper function."""

    def test_converts_tuples_to_documents(self):
        """Test conversion of (doc, score) tuples to Document dicts."""
        from src.orchestration.nodes.retrieval.legacy_nodes import _convert_to_documents

        results = [
            ({"unit_id": "u1", "narrative": "Content 1"}, 0.9),
            ({"unit_id": "u2", "narrative": "Content 2"}, 0.8),
        ]

        documents = _convert_to_documents(results, "test_method")

        assert len(documents) == 2
        # Document is TypedDict, so returns dict
        assert all(isinstance(d, dict) for d in documents)
        assert documents[0]["id"] == "u1"
        assert documents[0]["score"] == 0.9
        assert documents[0]["metadata"]["retrieval_method"] == "test_method"

    def test_preserves_original_doc(self):
        """Test that original doc is preserved in metadata."""
        from src.orchestration.nodes.retrieval.legacy_nodes import _convert_to_documents

        original_doc = {"unit_id": "u1", "narrative": "Content", "extra_field": "value"}
        results = [(original_doc, 0.9)]

        documents = _convert_to_documents(results, "test")

        assert documents[0]["metadata"]["original_doc"] == original_doc
        assert documents[0]["metadata"]["original_doc"]["extra_field"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
