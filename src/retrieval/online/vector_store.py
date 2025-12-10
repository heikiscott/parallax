"""Vector store retrieval for online API services.

This module provides unified vector store retrieval supporting:
- Embedding search via Milvus
- BM25 search via Elasticsearch
- RRF fusion of both

Core vector store retrieval logic for online API services.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import jieba

from core.di import get_bean_by_type
from infra.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from retrieval.core.utils import reciprocal_rank_fusion
from retrieval.services.vectorize import get_vectorize_service

from .utils import filter_semantic_memories_by_time, format_datetime_field

logger = logging.getLogger(__name__)


async def retrieve_from_vector_stores(
    query: str,
    user_id: str = None,
    group_id: str = None,
    top_k: int = 20,
    retrieval_mode: str = "rrf",
    data_source: str = "episode",
    start_time: float = None,
    memory_scope: str = "all",
    current_time: Optional[datetime] = None,
    participant_user_id: Optional[str] = None,
    radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Unified vector store retrieval (embedding, bm25, or rrf mode).

    Args:
        query: Query text
        user_id: User ID filter
        group_id: Group ID filter
        top_k: Number of results to return
        retrieval_mode: Retrieval mode (embedding/bm25/rrf)
        data_source: Data source (episode/event_log/semantic_memory)
        start_time: Start time for latency calculation
        memory_scope: Memory scope (all/personal/group)
        current_time: Current time for semantic memory filtering
        participant_user_id: Filter by participant in group memories
        radius: COSINE similarity threshold

    Returns:
        Dict with memories, count, and metadata
    """
    if start_time is None:
        start_time = time.time()

    try:
        # 1. Embedding retrieval via Milvus
        embedding_results = []
        embedding_count = 0

        if retrieval_mode in ["embedding", "rrf"]:
            embedding_results, embedding_count = await _search_embedding(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                data_source=data_source,
                current_time=current_time,
                participant_user_id=participant_user_id,
                radius=radius,
            )

        # 2. BM25 retrieval via Elasticsearch
        bm25_results = []
        bm25_count = 0

        if retrieval_mode in ["bm25", "rrf"]:
            bm25_results, bm25_count = await _search_bm25(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                data_source=data_source,
                current_time=current_time,
                participant_user_id=participant_user_id,
            )

        # 3. Build results based on mode
        if retrieval_mode == "embedding":
            memories, metadata = _build_embedding_result(
                embedding_results, embedding_count, top_k, data_source, start_time
            )
        elif retrieval_mode == "bm25":
            memories, metadata = _build_bm25_result(
                bm25_results, bm25_count, top_k, data_source, start_time
            )
        else:  # rrf
            memories, metadata = _build_rrf_result(
                embedding_results, bm25_results, embedding_count, bm25_count,
                top_k, data_source, start_time
            )

        # Filter semantic memories by time
        memories = filter_semantic_memories_by_time(memories, data_source, current_time)
        metadata["final_count"] = len(memories)

        return {
            "memories": memories,
            "count": len(memories),
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Vector store retrieval failed: {e}", exc_info=True)
        return {
            "memories": [],
            "count": 0,
            "metadata": {
                "retrieval_mode": retrieval_mode,
                "data_source": data_source,
                "error": str(e),
                "total_latency_ms": (time.time() - start_time) * 1000
            }
        }


async def _search_embedding(
    query: str,
    user_id: str,
    group_id: str,
    top_k: int,
    data_source: str,
    current_time: Optional[datetime],
    participant_user_id: Optional[str],
    radius: Optional[float],
) -> Tuple[List[Tuple[dict, float]], int]:
    """Execute embedding search via Milvus."""
    # Get appropriate Milvus repository based on data_source
    if data_source == "semantic_memory":
        from infra.adapters.out.search.repository.semantic_memory_milvus_repository import (
            SemanticMemoryMilvusRepository,
        )
        milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
    elif data_source == "event_log":
        from infra.adapters.out.search.repository.event_log_milvus_repository import (
            EventLogMilvusRepository,
        )
        milvus_repo = get_bean_by_type(EventLogMilvusRepository)
    else:  # "episode"
        milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

    vectorize_service = get_vectorize_service()

    # Generate query vector
    query_vec = await vectorize_service.get_embedding(query)

    # Execute vector search
    retrieval_limit = max(top_k * 10, 100)  # At least 100 candidates

    milvus_kwargs = dict(
        query_vector=query_vec,
        user_id=user_id,
        group_id=group_id,
        limit=retrieval_limit,
        radius=radius,
    )
    if data_source == "semantic_memory":
        milvus_kwargs["current_time"] = current_time
    if participant_user_id and data_source in ("episode", "event_log", "semantic_memory"):
        milvus_kwargs["participant_user_id"] = participant_user_id

    milvus_results = await milvus_repo.vector_search(**milvus_kwargs)

    # Process results
    embedding_results = []
    for result in milvus_results:
        score = result.get('score', 0)
        similarity = score  # All data sources use COSINE
        embedding_results.append((result, similarity))

    # Sort by similarity
    embedding_results.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Milvus search complete: data_source={data_source}, count={len(embedding_results)}")

    return embedding_results, len(embedding_results)


async def _search_bm25(
    query: str,
    user_id: str,
    group_id: str,
    top_k: int,
    data_source: str,
    current_time: Optional[datetime],
    participant_user_id: Optional[str],
) -> Tuple[List[Tuple[dict, float]], int]:
    """Execute BM25 search via Elasticsearch."""
    # Get appropriate ES repository based on data_source
    if data_source == "semantic_memory":
        from infra.adapters.out.search.repository.semantic_memory_es_repository import (
            SemanticMemoryEsRepository,
        )
        es_repo = get_bean_by_type(SemanticMemoryEsRepository)
    elif data_source == "event_log":
        from infra.adapters.out.search.repository.event_log_es_repository import (
            EventLogEsRepository,
        )
        es_repo = get_bean_by_type(EventLogEsRepository)
    else:  # "episode"
        es_repo = get_bean_by_type(EpisodicMemoryEsRepository)

    # Tokenize query with jieba
    query_words = list(jieba.cut(query))

    # Execute ES search
    retrieval_size = max(top_k * 10, 100)  # At least 100 candidates
    es_kwargs = dict(
        query=query_words,
        user_id=user_id,
        group_id=group_id,
        size=retrieval_size,
    )
    if participant_user_id and data_source in ("episode", "event_log", "semantic_memory"):
        es_kwargs["participant_user_id"] = participant_user_id
    if data_source == "semantic_memory" and current_time is not None:
        es_kwargs["current_time"] = current_time

    hits = await es_repo.multi_search(**es_kwargs)

    # Process results
    bm25_results = []
    for hit in hits:
        source = hit.get('_source', {})
        bm25_score = hit.get('_score', 0)
        metadata = source.get('extend', {})
        result = {
            'score': bm25_score,
            'episode_id': source.get('episode_id', ''),
            'user_id': source.get('user_id', ''),
            'group_id': source.get('group_id', ''),
            'timestamp': source.get('timestamp', ''),
            'narrative': source.get('narrative', ''),
            'search_content': source.get('search_content', []),
            'metadata': metadata,
        }
        if isinstance(metadata, dict):
            result['start_time'] = metadata.get('start_time')
            result['end_time'] = metadata.get('end_time')
        else:
            result['start_time'] = None
            result['end_time'] = None
        bm25_results.append((result, bm25_score))

    logger.debug(f"ES search complete: data_source={data_source}, count={len(bm25_results)}")
    return bm25_results, len(bm25_results)


def _build_embedding_result(
    embedding_results: List[Tuple[dict, float]],
    embedding_count: int,
    top_k: int,
    data_source: str,
    start_time: float,
) -> Tuple[List[Dict], Dict]:
    """Build result for pure embedding mode."""
    final_results = embedding_results[:top_k]
    memories = []
    for result, score in final_results:
        # Get content field based on data_source
        if data_source == "episode":
            narrative = result.get('narrative', '')
        elif data_source == "semantic_memory":
            narrative = result.get('content', '')
        else:  # event_log
            narrative = result.get('atomic_fact', '')

        memories.append({
            'score': score,
            'episode_id': result.get('id', ''),
            'user_id': result.get('user_id', ''),
            'group_id': result.get('group_id', ''),
            'timestamp': result.get('timestamp', ''),
            'subject': result.get('metadata', {}).get('title', ''),
            'narrative': narrative,
            'summary': result.get('metadata', {}).get('summary', ''),
            'evidence': result.get('evidence', '') if data_source == "semantic_memory" else '',
            'metadata': result.get('metadata', {}),
        })

    metadata = {
        "retrieval_mode": "embedding",
        "data_source": data_source,
        "embedding_candidates": embedding_count,
        "total_latency_ms": (time.time() - start_time) * 1000
    }
    return memories, metadata


def _build_bm25_result(
    bm25_results: List[Tuple[dict, float]],
    bm25_count: int,
    top_k: int,
    data_source: str,
    start_time: float,
) -> Tuple[List[Dict], Dict]:
    """Build result for pure BM25 mode."""
    final_results = bm25_results[:top_k]
    memories = [result for result, score in final_results]

    metadata = {
        "retrieval_mode": "bm25",
        "data_source": data_source,
        "bm25_candidates": bm25_count,
        "total_latency_ms": (time.time() - start_time) * 1000
    }
    return memories, metadata


def _build_rrf_result(
    embedding_results: List[Tuple[dict, float]],
    bm25_results: List[Tuple[dict, float]],
    embedding_count: int,
    bm25_count: int,
    top_k: int,
    data_source: str,
    start_time: float,
) -> Tuple[List[Dict], Dict]:
    """Build result for RRF fusion mode."""
    fused_results = reciprocal_rank_fusion(embedding_results, bm25_results, k=60)
    final_results = fused_results[:top_k]

    # Unify format
    memories = []
    for doc, rrf_score in final_results:
        # Doc may come from Milvus or ES, need to unify format
        # Distinguish: Milvus has 'id' field, ES has 'episode_id' field
        if 'episode_id' in doc and 'id' not in doc:
            # From ES (already in standard format)
            memory = {
                'score': rrf_score,
                'episode_id': doc.get('episode_id', ''),
                'user_id': doc.get('user_id', ''),
                'group_id': doc.get('group_id', ''),
                'timestamp': doc.get('timestamp', ''),
                'subject': '',
                'narrative': doc.get('narrative', ''),
                'summary': '',
                'evidence': doc.get('evidence', ''),
                'metadata': doc.get('metadata', {}),
                'start_time': doc.get('start_time'),
                'end_time': doc.get('end_time'),
            }
        else:
            # From Milvus (need field name conversion)
            content_field = 'narrative'  # default
            evidence_field = ''
            if data_source == "semantic_memory":
                content_field = 'content'
                evidence_field = doc.get('evidence', '')
            elif data_source == "event_log":
                content_field = 'atomic_fact'

            start_val = doc.get('start_time')
            end_val = doc.get('end_time')
            memory = {
                'score': rrf_score,
                'episode_id': doc.get('id', ''),  # Milvus uses 'id'
                'user_id': doc.get('user_id', ''),
                'group_id': doc.get('group_id', ''),
                'timestamp': doc.get('timestamp', ''),
                'subject': doc.get('metadata', {}).get('title', '') if isinstance(doc.get('metadata'), dict) else '',
                'narrative': doc.get(content_field, ''),
                'summary': doc.get('metadata', {}).get('summary', '') if isinstance(doc.get('metadata'), dict) else '',
                'evidence': evidence_field,
                'metadata': doc.get('metadata', {}) if isinstance(doc.get('metadata'), dict) else {},
                'start_time': format_datetime_field(start_val),
                'end_time': format_datetime_field(end_val),
            }
        memories.append(memory)

    metadata = {
        "retrieval_mode": "rrf",
        "data_source": data_source,
        "embedding_candidates": embedding_count,
        "bm25_candidates": bm25_count,
        "total_latency_ms": (time.time() - start_time) * 1000
    }
    return memories, metadata
