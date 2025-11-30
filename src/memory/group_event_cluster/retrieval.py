"""Cluster-enhanced retrieval functions.

This module provides functions to expand retrieval results using cluster information.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from .schema import ClusterMember, GroupEventCluster, GroupEventClusterIndex
from .config import ClusterRetrievalConfig

logger = logging.getLogger(__name__)


def expand_with_cluster(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: ClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Expand retrieval results using cluster index.

    Args:
        original_results: Original retrieval results [(doc, score), ...]
        cluster_index: Cluster index for lookups
        config: Cluster retrieval configuration
        all_docs_map: unit_id -> doc mapping for getting expanded document content

    Returns:
        (expanded_results, metadata)
        - expanded_results: Expanded result list
        - metadata: Expansion statistics
    """
    if not config.enable_cluster_expansion:
        return original_results, {"enabled": False}

    if config.expansion_strategy == "insert_after_hit":
        return _expand_insert_after_hit(
            original_results, cluster_index, config, all_docs_map
        )
    elif config.expansion_strategy == "append_to_end":
        return _expand_append_to_end(
            original_results, cluster_index, config, all_docs_map
        )
    elif config.expansion_strategy == "merge_by_score":
        return _expand_merge_by_score(
            original_results, cluster_index, config, all_docs_map
        )
    elif config.expansion_strategy == "replace_rerank":
        # For replace_rerank, just expand; actual rerank happens in caller
        return _expand_for_rerank(
            original_results, cluster_index, config, all_docs_map
        )
    else:
        logger.warning(f"Unknown expansion strategy: {config.expansion_strategy}")
        return original_results, {"enabled": False, "error": "unknown_strategy"}


def _expand_insert_after_hit(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: ClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Insert Cluster members after each hit document.

    This strategy maintains semantic coherence by placing related content
    immediately after the original hit.
    """
    expanded_results = []
    seen_unit_ids: Set[str] = set()
    dedup = config.deduplicate_expanded
    total_expanded = 0
    clusters_expanded: Dict[str, Dict[str, Any]] = {}

    # Calculate expansion budget
    expansion_budget = min(
        config.max_total_expansion,
        int(len(original_results) * config.expansion_budget_ratio)
    )

    for doc, score in original_results:
        unit_id = doc.get("unit_id")
        if not unit_id:
            expanded_results.append((doc, score))
            continue

        # Add original document
        if unit_id not in seen_unit_ids:
            expanded_results.append((doc, score))
            seen_unit_ids.add(unit_id)

        # Check if we still have expansion budget
        if total_expanded >= expansion_budget:
            continue

        # Get the Cluster this MemUnit belongs to
        cluster = cluster_index.get_cluster_by_unit(unit_id)
        if not cluster:
            continue

        # Get expansion candidates from Cluster
        candidates = _select_expansion_candidates(
            cluster=cluster,
            hit_unit_id=unit_id,
            seen_unit_ids=seen_unit_ids,
            config=config,
        )

        # Track cluster expansion info
        if cluster.cluster_id not in clusters_expanded:
            clusters_expanded[cluster.cluster_id] = {
                "hit_unit_ids": [],
                "expanded_unit_ids": [],
            }
        clusters_expanded[cluster.cluster_id]["hit_unit_ids"].append(unit_id)

        # Add expanded documents with decayed scores
        for member in candidates[:config.max_expansion_per_hit]:
            if total_expanded >= expansion_budget:
                break
            if dedup and member.unit_id in seen_unit_ids:
                continue

            expanded_doc = all_docs_map.get(member.unit_id)
            if not expanded_doc:
                continue

            expanded_score = score * config.expansion_score_decay
            expanded_results.append((expanded_doc, expanded_score))
            if dedup:
                seen_unit_ids.add(member.unit_id)
            total_expanded += 1
            clusters_expanded[cluster.cluster_id]["expanded_unit_ids"].append(member.unit_id)

    metadata = _build_metadata(
        strategy="insert_after_hit",
        original_count=len(original_results),
        expanded_count=total_expanded,
        final_count=len(expanded_results),
        clusters_expanded=clusters_expanded,
        expansion_budget=expansion_budget,
        config=config,
    )

    return expanded_results, metadata


def _expand_append_to_end(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: ClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Append all expanded documents to the end of results.

    This strategy preserves the original ranking completely.
    """
    seen_unit_ids: Set[str] = set()
    expanded_docs: List[Tuple[dict, float]] = []
    clusters_expanded: Dict[str, Dict[str, Any]] = {}

    # Calculate expansion budget
    expansion_budget = min(
        config.max_total_expansion,
        int(len(original_results) * config.expansion_budget_ratio)
    )

    # First pass: collect all unit_ids from original results
    for doc, _ in original_results:
        unit_id = doc.get("unit_id")
        if unit_id:
            seen_unit_ids.add(unit_id)

    # Second pass: collect expansion candidates
    total_expanded = 0
    for doc, score in original_results:
        if total_expanded >= expansion_budget:
            break

        unit_id = doc.get("unit_id")
        if not unit_id:
            continue

        cluster = cluster_index.get_cluster_by_unit(unit_id)
        if not cluster:
            continue

        candidates = _select_expansion_candidates(
            cluster=cluster,
            hit_unit_id=unit_id,
            seen_unit_ids=seen_unit_ids,
            config=config,
        )

        if cluster.cluster_id not in clusters_expanded:
            clusters_expanded[cluster.cluster_id] = {
                "hit_unit_ids": [],
                "expanded_unit_ids": [],
            }
        clusters_expanded[cluster.cluster_id]["hit_unit_ids"].append(unit_id)

        for member in candidates[:config.max_expansion_per_hit]:
            if total_expanded >= expansion_budget:
                break
            if member.unit_id in seen_unit_ids:
                continue

            expanded_doc = all_docs_map.get(member.unit_id)
            if not expanded_doc:
                continue

            expanded_score = score * config.expansion_score_decay
            expanded_docs.append((expanded_doc, expanded_score))
            seen_unit_ids.add(member.unit_id)
            total_expanded += 1
            clusters_expanded[cluster.cluster_id]["expanded_unit_ids"].append(member.unit_id)

    # Combine original and expanded
    final_results = list(original_results) + expanded_docs

    metadata = _build_metadata(
        strategy="append_to_end",
        original_count=len(original_results),
        expanded_count=total_expanded,
        final_count=len(final_results),
        clusters_expanded=clusters_expanded,
        expansion_budget=expansion_budget,
        config=config,
    )

    return final_results, metadata


def _expand_merge_by_score(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: ClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Merge expanded documents by score, re-sorting the combined results.

    This strategy allows high-relevance expanded documents to rank higher.
    """
    seen_unit_ids: Set[str] = set()
    all_docs: List[Tuple[dict, float]] = []
    clusters_expanded: Dict[str, Dict[str, Any]] = {}

    expansion_budget = min(
        config.max_total_expansion,
        int(len(original_results) * config.expansion_budget_ratio)
    )

    # Add original results first
    for doc, score in original_results:
        unit_id = doc.get("unit_id")
        if unit_id:
            seen_unit_ids.add(unit_id)
        all_docs.append((doc, score))

    # Collect and add expanded documents
    total_expanded = 0
    for doc, score in original_results:
        if total_expanded >= expansion_budget:
            break

        unit_id = doc.get("unit_id")
        if not unit_id:
            continue

        cluster = cluster_index.get_cluster_by_unit(unit_id)
        if not cluster:
            continue

        candidates = _select_expansion_candidates(
            cluster=cluster,
            hit_unit_id=unit_id,
            seen_unit_ids=seen_unit_ids,
            config=config,
        )

        if cluster.cluster_id not in clusters_expanded:
            clusters_expanded[cluster.cluster_id] = {
                "hit_unit_ids": [],
                "expanded_unit_ids": [],
            }
        clusters_expanded[cluster.cluster_id]["hit_unit_ids"].append(unit_id)

        for member in candidates[:config.max_expansion_per_hit]:
            if total_expanded >= expansion_budget:
                break
            if member.unit_id in seen_unit_ids:
                continue

            expanded_doc = all_docs_map.get(member.unit_id)
            if not expanded_doc:
                continue

            expanded_score = score * config.expansion_score_decay
            all_docs.append((expanded_doc, expanded_score))
            seen_unit_ids.add(member.unit_id)
            total_expanded += 1
            clusters_expanded[cluster.cluster_id]["expanded_unit_ids"].append(member.unit_id)

    # Sort by score descending
    final_results = sorted(all_docs, key=lambda x: x[1], reverse=True)

    metadata = _build_metadata(
        strategy="merge_by_score",
        original_count=len(original_results),
        expanded_count=total_expanded,
        final_count=len(final_results),
        clusters_expanded=clusters_expanded,
        expansion_budget=expansion_budget,
        config=config,
    )

    return final_results, metadata


def _expand_for_rerank(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: ClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Expand results for subsequent reranking.

    This is similar to append_to_end but intended for reranking afterward.
    """
    # Use append_to_end logic
    results, metadata = _expand_append_to_end(
        original_results, cluster_index, config, all_docs_map
    )
    metadata["strategy"] = "replace_rerank"
    metadata["requires_rerank"] = True
    return results, metadata


def _select_expansion_candidates(
    cluster: GroupEventCluster,
    hit_unit_id: str,
    seen_unit_ids: Set[str],
    config: ClusterRetrievalConfig,
) -> List[ClusterMember]:
    """
    Select expansion candidate members from a cluster.

    Strategies:
    1. If prefer_time_adjacent=True, prefer time-adjacent members
    2. Otherwise, select by time order
    """
    # Filter out already seen members
    candidates = [
        m for m in cluster.members
        if m.unit_id != hit_unit_id and m.unit_id not in seen_unit_ids
    ]

    if not candidates:
        return []

    if not config.prefer_time_adjacent:
        # Simple strategy: take first N by time order
        return candidates

    # Time-adjacent strategy: find hit member position, prefer adjacent ones
    hit_index = None
    for i, m in enumerate(cluster.members):
        if m.unit_id == hit_unit_id:
            hit_index = i
            break

    if hit_index is None:
        return candidates

    # Build priority: closer to hit position = higher priority
    # Alternate between after and before
    result = []
    before_idx = hit_index - 1
    after_idx = hit_index + 1

    while len(result) < len(candidates):
        added = False

        # First select after (newer information in time)
        if after_idx < len(cluster.members):
            member = cluster.members[after_idx]
            if member.unit_id not in seen_unit_ids and member.unit_id != hit_unit_id:
                result.append(member)
                added = True
            after_idx += 1

        # Then select before
        if before_idx >= 0:
            member = cluster.members[before_idx]
            if member.unit_id not in seen_unit_ids and member.unit_id != hit_unit_id:
                result.append(member)
                added = True
            before_idx -= 1

        # Both sides exhausted
        if not added and after_idx >= len(cluster.members) and before_idx < 0:
            break

    # Apply time window limit if configured
    if config.time_window_hours is not None:
        hit_member = cluster.members[hit_index]
        hit_time = hit_member.timestamp
        max_delta = timedelta(hours=config.time_window_hours)

        result = [
            m for m in result
            if abs((m.timestamp - hit_time).total_seconds()) <= max_delta.total_seconds()
        ]

    return result


def _build_metadata(
    strategy: str,
    original_count: int,
    expanded_count: int,
    final_count: int,
    clusters_expanded: Dict[str, Dict[str, Any]],
    expansion_budget: int,
    config: ClusterRetrievalConfig,
) -> Dict[str, Any]:
    """Build expansion metadata dictionary."""
    expanded_unit_ids = []
    for _, info in clusters_expanded.items():
        expanded_unit_ids.extend(info.get("expanded_unit_ids", []))

    return {
        "enabled": True,
        "strategy": strategy,
        "original_count": original_count,
        "expanded_count": expanded_count,
        "final_count": final_count,
        "clusters_hit": list(clusters_expanded.keys()),
        "clusters_expanded": clusters_expanded,
        "expanded_unit_ids": expanded_unit_ids,
        "expansion_budget": expansion_budget,
        "budget_used": expanded_count,
        "config": {
            "max_expansion_per_hit": config.max_expansion_per_hit,
            "expansion_budget_ratio": config.expansion_budget_ratio,
            "prefer_time_adjacent": config.prefer_time_adjacent,
            "time_window_hours": config.time_window_hours,
            "expansion_score_decay": config.expansion_score_decay,
        }
    }


def get_related_units_for_query(
    query_hit_unit_ids: List[str],
    cluster_index: GroupEventClusterIndex,
    max_per_cluster: int = 3,
    exclude_hits: bool = True,
) -> List[str]:
    """
    Get related unit_ids for a set of query hits.

    This is a utility function for simple expansion scenarios.

    Args:
        query_hit_unit_ids: List of unit_ids that matched the query
        cluster_index: Cluster index for lookups
        max_per_cluster: Maximum units to return per cluster
        exclude_hits: Whether to exclude the original hit unit_ids

    Returns:
        List of related unit_ids (time-sorted within each cluster)
    """
    related = []
    seen_clusters: Set[str] = set()
    hit_set = set(query_hit_unit_ids)

    for unit_id in query_hit_unit_ids:
        cluster = cluster_index.get_cluster_by_unit(unit_id)
        if not cluster or cluster.cluster_id in seen_clusters:
            continue

        seen_clusters.add(cluster.cluster_id)

        # Get members (already time-sorted)
        for member in cluster.members:
            if exclude_hits and member.unit_id in hit_set:
                continue
            if member.unit_id not in related:
                related.append(member.unit_id)
                if len([r for r in related if cluster_index.get_cluster_id_by_unit(r) == cluster.cluster_id]) >= max_per_cluster:
                    break

    return related
