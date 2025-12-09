"""Tests for BM25, Embedding, and Hybrid retrievers."""

import pytest
from src.retrieval.retrievers import BM25Retriever, HybridRetriever, bm25_search, hybrid_search


# ============================================================================
# BM25Retriever Tests
# ============================================================================

@pytest.mark.asyncio
async def test_bm25_retriever_basic(mock_memory_index):
    """Test basic BM25 retrieval."""
    retriever = BM25Retriever(top_k=3)

    results = await retriever.search(
        query="park",
        memory_index=mock_memory_index,
        bm25_index=None
    )

    assert len(results) > 0
    # Check that results are dict-like with required keys (TypedDict)
    assert all(isinstance(doc, dict) and "id" in doc and "content" in doc for doc in results)
    assert all("bm25" in doc.get("metadata", {}).get("retrieval_method", "") for doc in results)


@pytest.mark.asyncio
async def test_bm25_search_convenience_function(mock_memory_index):
    """Test BM25 search convenience function."""
    results = await bm25_search(
        query="meeting",
        memory_index=mock_memory_index,
        top_k=3
    )

    assert len(results) > 0
    assert all(isinstance(doc, dict) and "id" in doc and "content" in doc for doc in results)


# ============================================================================
# HybridRetriever Tests
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_retriever_basic(mock_memory_index, mock_vectorize_service):
    """Test basic hybrid retrieval (Embedding + BM25 + RRF)."""
    retriever = HybridRetriever(
        emb_top_k=50,
        bm25_top_k=50,
        final_top_k=20
    )

    results = await retriever.search(
        query="Alice park",
        memory_index=mock_memory_index,
        vectorize_service=mock_vectorize_service,
        bm25_index=None
    )

    assert len(results) > 0
    assert len(results) <= 20
    assert all(isinstance(doc, dict) and "id" in doc and "content" in doc for doc in results)
    # Check that RRF scores are present
    assert all("rrf_score" in doc.get("metadata", {}) for doc in results)


@pytest.mark.asyncio
async def test_hybrid_search_convenience_function(mock_memory_index, mock_vectorize_service):
    """Test hybrid search convenience function."""
    results = await hybrid_search(
        query="Bob meeting",
        memory_index=mock_memory_index,
        vectorize_service=mock_vectorize_service,
        emb_top_k=50,
        bm25_top_k=50,
        final_top_k=10
    )

    assert len(results) > 0
    assert len(results) <= 10
    assert all(isinstance(doc, dict) and "id" in doc and "content" in doc for doc in results)
