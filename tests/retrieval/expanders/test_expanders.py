"""Tests for ClusterExpander and MultiQueryExpander."""

import pytest
from unittest.mock import AsyncMock
from src.retrieval.core import Document
from src.retrieval.expanders import ClusterExpander, MultiQueryExpander, cluster_expand, multi_query_expand


# ============================================================================
# ClusterExpander Tests
# ============================================================================

@pytest.mark.asyncio
async def test_cluster_expander_insert_after_hit(mock_cluster_index, sample_documents):
    """Test cluster expansion with insert_after_hit strategy."""
    expander = ClusterExpander(
        expansion_strategy="insert_after_hit",
        max_expansion_per_hit=3,
        min_cluster_score=0.5
    )

    # First document should trigger expansion
    docs_with_cluster = [
        Document(
            id="mem1",
            content="Alice went to the park.",
            score=0.9,
            metadata={}
        )
    ]

    results = await expander.expand(
        query="Alice park",
        documents=docs_with_cluster,
        cluster_index=mock_cluster_index
    )

    # Should have original + expanded documents
    assert len(results) > len(docs_with_cluster)
    # Check that expanded docs have cluster metadata
    expanded_docs = [d for d in results if "expansion_method" in d.get("metadata", {})]
    assert len(expanded_docs) > 0
    assert all(d["metadata"]["expansion_method"] == "cluster" for d in expanded_docs)


@pytest.mark.asyncio
async def test_cluster_expander_append_end(mock_cluster_index, sample_documents):
    """Test cluster expansion with append_end strategy."""
    expander = ClusterExpander(
        expansion_strategy="append_end",
        max_expansion_per_hit=3,
        min_cluster_score=0.5
    )

    docs_with_cluster = [
        Document(
            id="mem1",
            content="Alice went to the park.",
            score=0.9,
            metadata={}
        )
    ]

    results = await expander.expand(
        query="Alice park",
        documents=docs_with_cluster,
        cluster_index=mock_cluster_index
    )

    # Should have original + expanded documents
    assert len(results) > len(docs_with_cluster)
    # Original document should be first
    assert results[0]["id"] == "mem1"


@pytest.mark.asyncio
async def test_cluster_expand_convenience_function(mock_cluster_index):
    """Test cluster expand convenience function."""
    docs = [
        Document(id="mem1", content="Alice went to the park.", score=0.9, metadata={})
    ]

    results = await cluster_expand(
        query="Alice",
        documents=docs,
        cluster_index=mock_cluster_index,
        expansion_strategy="insert_after_hit"
    )

    assert len(results) > len(docs)


# ============================================================================
# MultiQueryExpander Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multi_query_expander_basic(mock_llm_provider, sample_documents):
    """Test multi-query expansion."""
    expander = MultiQueryExpander(
        num_queries=3,
        retrieval_top_k=20,
        final_top_k=50
    )

    # Mock LLM to return query variants
    mock_llm_provider.generate = AsyncMock(
        return_value="Alice in the park\nAlice at the park\nPark visit by Alice"
    )

    # Mock retriever function
    async def mock_retriever(query: str, top_k: int):
        return [
            Document(id=f"doc_{query[:5]}", content=f"Result for {query}", score=0.8, metadata={})
        ]

    results = await expander.expand(
        query="Alice park",
        documents=sample_documents,
        llm_provider=mock_llm_provider,
        retriever_func=mock_retriever
    )

    # Should have fused results from multiple queries
    assert len(results) > 0
    assert all(isinstance(doc, dict) and "id" in doc and "content" in doc for doc in results)
    # Check metadata
    assert results[0].get("metadata", {}).get("expansion_method") == "multi_query"


@pytest.mark.asyncio
async def test_multi_query_expand_convenience_function(mock_llm_provider):
    """Test multi-query expand convenience function."""
    docs = [
        Document(id="doc1", content="Alice went to the park.", score=0.9, metadata={})
    ]

    mock_llm_provider.generate = AsyncMock(
        return_value="park visit\nAlice at park\npark activity"
    )

    async def mock_retriever(query: str, top_k: int):
        return [Document(id="doc2", content=f"Result for {query}", score=0.7, metadata={})]

    results = await multi_query_expand(
        query="Alice park",
        documents=docs,
        llm_provider=mock_llm_provider,
        retriever_func=mock_retriever,
        num_queries=3
    )

    assert len(results) > 0
