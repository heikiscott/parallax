"""Lightweight retrieval for online API services.

This module provides fast retrieval without LLM guidance:
- Embedding search via Milvus
- BM25 search via Elasticsearch
- RRF fusion of both
- Profile retrieval via MongoDB

Provides the main retrieve_lightweight function for API services.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from infra.adapters.out.persistence.document.memory.user_profile import UserProfile

from .vector_store import retrieve_from_vector_stores

logger = logging.getLogger(__name__)


async def retrieve_lightweight(
    query: str,
    user_id: str = None,
    group_id: str = None,
    time_range_days: int = 365,
    top_k: int = 20,
    retrieval_mode: str = "rrf",
    data_source: str = "episode",
    memory_scope: str = "all",
    current_time: Optional[datetime] = None,
    radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Lightweight memory retrieval (Embedding + BM25 + RRF via Milvus/ES).

    Args:
        query: User query
        user_id: User ID filter
        group_id: Group ID filter (required for profile data source)
        time_range_days: Time range in days
        top_k: Number of results to return
        retrieval_mode: Retrieval mode
            - "embedding": Pure vector search (via Milvus)
            - "bm25": Pure keyword search (via ES)
            - "rrf": RRF fusion (default, Milvus + ES)
        data_source: Data source
            - "episode": From episode (default)
            - "event_log": From event_log
            - "semantic_memory": From semantic memory
            - "profile": Direct user profile retrieval by user_id + group_id
        memory_scope: Memory scope
            - "all": All memories (default, personal + group)
            - "personal": Personal memories only (group_id is empty)
            - "group": Group memories only (group_id is not empty)
        current_time: Current time for filtering semantic memories
        radius: COSINE similarity threshold

    Returns:
        Dict with memories, count, and metadata
    """
    start_time = time.time()

    # Legacy parameter compatibility
    if data_source == "memunit":
        data_source = "episode"

    # Handle profile data source separately
    if data_source == "profile":
        if not user_id or not group_id:
            raise ValueError("user_id and group_id are required for profile retrieval")
        return await _retrieve_profile_memories(
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            start_time=start_time,
        )

    # Process memory_scope parameter
    original_user_id = user_id
    scope_user_id = user_id
    scope_group_id = group_id
    participant_user_id: Optional[str] = None

    if memory_scope == "personal":
        # Personal memories: only pass user_id, not group_id
        scope_group_id = None
    elif memory_scope == "group":
        # Group memories: episode still needs user_id="", others only look at group_id
        if data_source == "episode":
            scope_user_id = ""  # Empty string indicates group episode
        else:
            scope_user_id = None  # Event log/semantic memory filter by group_id only

        if original_user_id and data_source in (
            "episode",
            "event_log",
            "semantic_memory",
        ):
            participant_user_id = original_user_id
    else:
        # "all": Don't filter by user_id to avoid missing group memories
        scope_user_id = None

    return await retrieve_from_vector_stores(
        query=query,
        user_id=scope_user_id,
        group_id=scope_group_id,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        data_source=data_source,
        start_time=start_time,
        memory_scope=memory_scope,
        current_time=current_time,
        participant_user_id=participant_user_id,
        radius=radius,
    )


async def _retrieve_profile_memories(
    user_id: str,
    group_id: str,
    top_k: int,
    start_time: float,
) -> Dict[str, Any]:
    """Retrieve user profile from user_profiles collection."""
    doc = await UserProfile.find_one(
        UserProfile.user_id == user_id,
        UserProfile.group_id == group_id,
        sort=[("version", -1)],
    )

    memories: List[Dict[str, Any]] = []
    if doc:
        memories.append(
            {
                "user_id": doc.user_id,
                "group_id": doc.group_id,
                "profile": doc.profile_data,
                "scenario": doc.scenario,
                "confidence": doc.confidence,
                "version": doc.version,
                "cluster_ids": doc.cluster_ids,
                "memunit_count": doc.memunit_count,
                "last_updated_cluster": doc.last_updated_cluster,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            }
        )

    metadata = {
        "retrieval_mode": "direct",
        "data_source": "profile",
        "profile_count": len(memories),
        "total_latency_ms": (time.time() - start_time) * 1000,
    }

    return {
        "memories": memories[:top_k],
        "count": len(memories[:top_k]),
        "metadata": metadata,
    }
