"""Low-level search utilities for retrieval pipelines.

This module contains core search functions used by both lightweight and agentic pipelines:
- BM25 search
- Embedding search (MaxSim)
- Hybrid search with RRF fusion
- RRF fusion algorithms
"""

import logging
import asyncio
from typing import List, Tuple, Optional, Set
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from retrieval.services import vectorize as vectorize_service

logger = logging.getLogger(__name__)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and document vectors.

    Args:
        query_vec: 1D numpy array for the query
        doc_vecs: 2D numpy array where each row is a document vector

    Returns:
        1D numpy array of cosine similarity scores
    """
    dot_product = np.dot(doc_vecs, query_vec)
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    denominator = query_norm * doc_norms
    denominator[denominator == 0] = 1e-9

    return dot_product / denominator


def compute_maxsim_score(query_emb: np.ndarray, atomic_fact_embs: List[np.ndarray]) -> float:
    """Compute MaxSim score between query and atomic fact embeddings.

    MaxSim strategy: Find the single most relevant atomic fact.
    Only needs one fact to be strongly related to consider the event_log relevant.

    Args:
        query_emb: Query embedding vector (1D numpy array)
        atomic_fact_embs: List of atomic fact embedding vectors

    Returns:
        Maximum similarity score (float, range [-1, 1], typically [0, 1])
    """
    if not atomic_fact_embs:
        return 0.0

    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return 0.0

    try:
        # Vectorized computation (2-3x faster)
        fact_matrix = np.array(atomic_fact_embs)
        fact_norms = np.linalg.norm(fact_matrix, axis=1)

        valid_mask = fact_norms > 0
        if not np.any(valid_mask):
            return 0.0

        dot_products = np.dot(fact_matrix[valid_mask], query_emb)
        sims = dot_products / (query_norm * fact_norms[valid_mask])

        return float(np.max(sims))

    except Exception:
        # Fallback to loop (compatibility)
        similarities = []
        for fact_emb in atomic_fact_embs:
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            sim = np.dot(query_emb, fact_emb) / (query_norm * fact_norm)
            similarities.append(sim)
        return max(similarities) if similarities else 0.0


def tokenize(text: str, stemmer, stop_words: set) -> list:
    """NLTK-based tokenization with stemming and stopword removal.

    Args:
        text: Input text
        stemmer: NLTK stemmer instance
        stop_words: Set of stopwords

    Returns:
        List of processed tokens
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]

    return processed_tokens


def search_with_bm25_index(
    query: str,
    bm25,
    docs,
    top_n: int = 5,
    return_all_scored: bool = False
):
    """Perform BM25 search using pre-loaded index.

    Args:
        query: Query text
        bm25: BM25 index
        docs: Document list
        top_n: Number of results to return
        return_all_scored: If True, return (top_n_results, all_doc_ids_scored)

    Returns:
        If return_all_scored=False: [(doc, score), ...]
        If return_all_scored=True: ([(doc, score), ...], [all scored unit_ids])
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)

    if not tokenized_query:
        logger.warning("Query is empty after tokenization.")
        return ([], []) if return_all_scored else []

    doc_scores = bm25.get_scores(tokenized_query)
    results_with_scores = list(zip(docs, doc_scores))
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)

    if return_all_scored:
        all_scored_ids = [doc.get("unit_id", f"unknown_{i}") for i, (doc, _) in enumerate(results_with_scores)]
        return sorted_results[:top_n], all_scored_ids

    return sorted_results[:top_n]


async def search_with_emb_index(
    query: str,
    emb_index,
    top_n: int = 5,
    query_embedding: Optional[np.ndarray] = None,
    return_all_scored: bool = False
):
    """Execute embedding search using MaxSim strategy.

    For documents with atomic_facts: compute max similarity across facts.
    For traditional documents: use subject/summary/narrative fields.

    Args:
        query: Query text
        emb_index: Pre-built embedding index
        top_n: Number of results to return
        query_embedding: Optional pre-computed query embedding
        return_all_scored: If True, return (top_n_results, all_doc_ids_scored)

    Returns:
        If return_all_scored=False: [(doc, score), ...]
        If return_all_scored=True: ([(doc, score), ...], [all scored unit_ids])
    """
    # Get query embedding
    if query_embedding is not None:
        query_vec = query_embedding
    else:
        query_vec = np.array(await vectorize_service.get_text_embedding(query))

    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return ([], []) if return_all_scored else []

    doc_scores = []

    for item in emb_index:
        doc = item.get("doc")
        embeddings = item.get("embeddings", {})

        if not embeddings:
            continue

        # Prefer atomic_facts (MaxSim strategy)
        if "atomic_facts" in embeddings:
            atomic_fact_embs = embeddings["atomic_facts"]
            if atomic_fact_embs:
                score = compute_maxsim_score(query_vec, atomic_fact_embs)
                doc_scores.append((doc, score))
                continue

        # Fallback to traditional fields (MaxSim across fields)
        field_scores = []
        for field in ["subject", "summary", "narrative"]:
            if field in embeddings:
                field_emb = embeddings[field]
                field_norm = np.linalg.norm(field_emb)

                if field_norm > 0:
                    sim = np.dot(query_vec, field_emb) / (query_norm * field_norm)
                    field_scores.append(sim)

        if field_scores:
            score = max(field_scores)
            doc_scores.append((doc, score))

    if not doc_scores:
        return ([], []) if return_all_scored else []

    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    if return_all_scored:
        all_scored_ids = [doc.get("unit_id", f"unknown_{i}") for i, (doc, _) in enumerate(doc_scores)]
        return sorted_results[:top_n], all_scored_ids

    return sorted_results[:top_n]


def reciprocal_rank_fusion(
    emb_results: List[Tuple[dict, float]],
    bm25_results: List[Tuple[dict, float]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """Fuse Embedding and BM25 results using RRF (Reciprocal Rank Fusion).

    RRF is a rank-based fusion that doesn't require score normalization.
    Formula: RRF_score(doc) = sum(1 / (k + rank_i))

    Args:
        emb_results: Embedding search results [(doc, score), ...]
        bm25_results: BM25 search results [(doc, score), ...]
        k: RRF constant (typically 60)

    Returns:
        Fused results [(doc, rrf_score), ...] sorted by RRF score descending
    """
    doc_rrf_scores = {}  # {unit_id: rrf_score}
    doc_map = {}         # {unit_id: doc}

    # Process embedding results
    for rank, (doc, score) in enumerate(emb_results, start=1):
        doc_id = doc.get("unit_id", id(doc))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Process BM25 results
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        doc_id = doc.get("unit_id", id(doc))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]


def multi_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """Fuse multiple query results using RRF (multi-query fusion).

    Similar to dual-path RRF but supports any number of result sets.
    Documents ranked highly across multiple queries get higher scores.

    Args:
        results_list: List of result lists from different queries
        k: RRF constant (default 60)

    Returns:
        Fused results [(doc, rrf_score), ...] sorted by RRF score descending
    """
    if not results_list:
        return []

    if len(results_list) == 1:
        return results_list[0]

    doc_rrf_scores = {}
    doc_map = {}

    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = doc.get("unit_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]


async def hybrid_search_with_rrf(
    query: str,
    emb_index,
    bm25,
    docs,
    top_n: int = 40,
    emb_candidates: int = 50,
    bm25_candidates: int = 50,
    rrf_k: int = 60,
    query_embedding: Optional[np.ndarray] = None,
    return_traversal_stats: bool = False
):
    """Hybrid search combining Embedding and BM25 with RRF fusion.

    Flow:
    1. Parallel Embedding (MaxSim) and BM25 search
    2. Each method returns top-N candidates
    3. RRF fusion of both result sets
    4. Return fused top-N documents

    Args:
        query: User query
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list (for BM25)
        top_n: Final number of results (default 40)
        emb_candidates: Embedding search candidates (default 50)
        bm25_candidates: BM25 search candidates (default 50)
        rrf_k: RRF parameter k (default 60)
        query_embedding: Optional pre-computed query embedding
        return_traversal_stats: Whether to return traversal statistics

    Returns:
        If return_traversal_stats=False: [(doc, rrf_score), ...]
        If return_traversal_stats=True: ([(doc, rrf_score), ...], stats_dict)
    """
    traversal_stats = {
        "emb_scored_ids": [],
        "bm25_scored_ids": [],
        "emb_returned_ids": [],
        "bm25_returned_ids": [],
        "fused_ids": [],
    }

    # Parallel execution
    emb_task = search_with_emb_index(
        query,
        emb_index,
        top_n=emb_candidates,
        query_embedding=query_embedding,
        return_all_scored=return_traversal_stats,
    )
    bm25_task = asyncio.to_thread(
        search_with_bm25_index,
        query,
        bm25,
        docs,
        bm25_candidates,
        return_all_scored=return_traversal_stats,
    )

    emb_raw, bm25_raw = await asyncio.gather(emb_task, bm25_task)

    # Parse results
    if return_traversal_stats:
        emb_results, emb_all_ids = emb_raw
        bm25_results, bm25_all_ids = bm25_raw
        traversal_stats["emb_scored_ids"] = emb_all_ids
        traversal_stats["bm25_scored_ids"] = bm25_all_ids
        traversal_stats["emb_returned_ids"] = [doc.get("unit_id", "") for doc, _ in emb_results]
        traversal_stats["bm25_returned_ids"] = [doc.get("unit_id", "") for doc, _ in bm25_results]
    else:
        emb_results = emb_raw
        bm25_results = bm25_raw

    # RRF fusion
    if not emb_results and not bm25_results:
        return ([], traversal_stats) if return_traversal_stats else []
    elif not emb_results:
        fused = bm25_results[:top_n]
    elif not bm25_results:
        fused = emb_results[:top_n]
    else:
        fused = reciprocal_rank_fusion(emb_results, bm25_results, k=rrf_k)[:top_n]

    if return_traversal_stats:
        traversal_stats["fused_ids"] = [doc.get("unit_id", "") for doc, _ in fused]
        return fused, traversal_stats

    return fused
