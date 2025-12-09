"""Shared fixtures for retrieval tests."""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_memory_index():
    """Mock memory index with sample data."""
    index = AsyncMock()

    # Sample memory objects
    class MemoryObj:
        def __init__(self, id, narrative, summary=""):
            self.id = id
            self.narrative = narrative
            self.summary = summary or narrative

    memories = [
        MemoryObj("mem1", "Alice went to the park on Monday morning."),
        MemoryObj("mem2", "Bob attended a meeting about project planning."),
        MemoryObj("mem3", "Charlie bought groceries including apples and milk."),
        MemoryObj("mem4", "Alice met her friend Diana at the park."),
        MemoryObj("mem5", "Bob presented the quarterly sales report."),
    ]

    index.get_all_memories = AsyncMock(return_value=memories)
    index.search = AsyncMock(return_value=memories[:3])  # Return top 3 for embedding search

    return index


@pytest.fixture
def mock_vectorize_service():
    """Mock vectorization service."""
    service = AsyncMock()
    service.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])  # Dummy embedding
    return service


@pytest.fixture
def mock_rerank_service():
    """Mock reranking service."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    provider = AsyncMock()

    # Default: return "SUFFICIENT" for sufficiency checks
    provider.generate = AsyncMock(return_value="SUFFICIENT - The documents contain relevant information.")

    return provider


@pytest.fixture
def mock_cluster_index():
    """Mock cluster index."""
    index = AsyncMock()

    # Mock cluster members
    class ClusterMember:
        def __init__(self, id, content, score):
            self.id = id
            self.narrative = content  # ClusterExpander expects 'narrative' attribute
            self.content = content
            self.cluster_score = score  # ClusterExpander expects 'cluster_score'
            self.score = score

    async def get_cluster_members(doc_id):
        if doc_id == "mem1":
            return [
                ClusterMember("mem4", "Alice met her friend Diana at the park.", 0.8),
                ClusterMember("mem6", "Alice enjoys walking in the park.", 0.7),
            ]
        return []

    index.get_cluster_members = AsyncMock(side_effect=get_cluster_members)

    return index


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from src.retrieval.core import Document

    return [
        Document(
            id="doc1",
            content="Alice went to the park on Monday morning.",
            score=0.9,
            metadata={"retrieval_method": "embedding"}
        ),
        Document(
            id="doc2",
            content="Bob attended a meeting about project planning.",
            score=0.8,
            metadata={"retrieval_method": "embedding"}
        ),
        Document(
            id="doc3",
            content="Charlie bought groceries including apples and milk.",
            score=0.7,
            metadata={"retrieval_method": "embedding"}
        ),
    ]


@pytest.fixture
def mock_execution_context(
    mock_memory_index,
    mock_vectorize_service,
    mock_rerank_service,
    mock_llm_provider,
    mock_cluster_index
):
    """Mock execution context."""
    from src.orchestration.context import ExecutionContext

    context = Mock(spec=ExecutionContext)
    context.memory_index = mock_memory_index
    context.vectorize_service = mock_vectorize_service
    context.rerank_service = mock_rerank_service
    context.llm_provider = mock_llm_provider
    context.cluster_index = mock_cluster_index
    context.bm25_index = None  # Will be built on the fly

    # Mock get_config_value
    config_values = {
        "top_k": 20,
        "emb_top_k": 50,
        "bm25_top_k": 50,
        "rerank_top_k": 20,
        "num_queries": 3,
        "retrieval_top_k": 20,
        "final_top_k": 50,
        "expansion_strategy": "insert_after_hit",
        "max_expansion_per_hit": 3,
        "min_cluster_score": 0.5,
        "top_n_to_check": 5,
    }

    def get_config_value(key, default=None):
        return config_values.get(key, default)

    context.get_config_value = Mock(side_effect=get_config_value)

    return context
