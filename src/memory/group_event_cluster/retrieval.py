"""Cluster-enhanced retrieval functions.

This module provides functions to expand retrieval results using cluster information.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import logging

from .schema import ClusterMember, GroupEventCluster, GroupEventClusterIndex
from .config import GroupEventClusterRetrievalConfig
from .utils import (
    format_clusters_for_selection,
    parse_cluster_selection_response,
    CLUSTER_SELECTION_PROMPT,
)

if TYPE_CHECKING:
    from providers.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


async def expand_with_cluster(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: GroupEventClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
    # 新增可选参数（仅 cluster_rerank 策略需要）
    query: Optional[str] = None,
    llm_provider: Optional["LLMProvider"] = None,
    llm_config: Optional[dict] = None,
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Expand retrieval results using cluster index.

    Args:
        original_results: Original retrieval results [(doc, score), ...]
        cluster_index: Cluster index for lookups
        config: Cluster retrieval configuration
        all_docs_map: unit_id -> doc mapping for getting expanded document content
        query: Original query (required for cluster_rerank strategy)
        llm_provider: LLM provider (required for cluster_rerank strategy)
        llm_config: LLM configuration (optional for cluster_rerank strategy)

    Returns:
        (expanded_results, metadata)
        - expanded_results: Expanded result list
        - metadata: Expansion statistics
    """
    if not config.enable_group_event_cluster_retrieval:
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
    elif config.expansion_strategy == "cluster_rerank":
        # Cluster-level rerank: LLM selects relevant clusters
        if not query or not llm_provider:
            logger.warning("cluster_rerank requires query and llm_provider")
            return original_results, {"enabled": False, "error": "missing_params"}

        return await _expand_cluster_rerank(
            original_results, cluster_index, config, all_docs_map,
            query, llm_provider, llm_config
        )
    else:
        logger.warning(f"Unknown expansion strategy: {config.expansion_strategy}")
        return original_results, {"enabled": False, "error": "unknown_strategy"}


def _expand_insert_after_hit(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: GroupEventClusterRetrievalConfig,
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

        # Get ALL Clusters this MemUnit belongs to (supports multi-cluster)
        clusters = cluster_index.get_clusters_by_unit(unit_id)
        if not clusters:
            continue

        # Expand from each cluster
        for cluster in clusters:
            if total_expanded >= expansion_budget:
                break

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
    config: GroupEventClusterRetrievalConfig,
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

        clusters = cluster_index.get_clusters_by_unit(unit_id)
        if not clusters:
            continue

        for cluster in clusters:
            if total_expanded >= expansion_budget:
                break

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
    config: GroupEventClusterRetrievalConfig,
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

        clusters = cluster_index.get_clusters_by_unit(unit_id)
        if not clusters:
            continue

        for cluster in clusters:
            if total_expanded >= expansion_budget:
                break

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
    config: GroupEventClusterRetrievalConfig,
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
    config: GroupEventClusterRetrievalConfig,
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
    config: GroupEventClusterRetrievalConfig,
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
    Supports multi-cluster membership: iterates all clusters each unit belongs to.

    Args:
        query_hit_unit_ids: List of unit_ids that matched the query
        cluster_index: Cluster index for lookups
        max_per_cluster: Maximum units to return per cluster
        exclude_hits: Whether to exclude the original hit unit_ids

    Returns:
        List of related unit_ids (deduplicated)
    """
    related: List[str] = []
    seen_clusters: Set[str] = set()
    seen_units: Set[str] = set()
    hit_set = set(query_hit_unit_ids)

    for unit_id in query_hit_unit_ids:
        # Get ALL clusters this unit belongs to (multi-cluster support)
        clusters = cluster_index.get_clusters_by_unit(unit_id)
        if not clusters:
            continue

        for cluster in clusters:
            if cluster.cluster_id in seen_clusters:
                continue

            seen_clusters.add(cluster.cluster_id)
            members_added = 0

            # Get members (already time-sorted)
            for member in cluster.members:
                if members_added >= max_per_cluster:
                    break
                if exclude_hits and member.unit_id in hit_set:
                    continue
                if member.unit_id in seen_units:
                    continue

                related.append(member.unit_id)
                seen_units.add(member.unit_id)
                members_added += 1

    return related


# =============================================================================
# Cluster Rerank Strategy
# =============================================================================

async def _expand_cluster_rerank(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: GroupEventClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
    query: str,
    llm_provider: "LLMProvider",
    llm_config: Optional[dict] = None,
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    Cluster-level rerank expansion strategy.

    This strategy:
    1. Extracts unique clusters from original retrieval results
    2. Uses LLM to intelligently select the most relevant clusters
    3. Returns MemUnits from selected clusters in time order
    4. Applies member limits (per-cluster and total)

    Args:
        original_results: Original retrieval results [(doc, score), ...]
        cluster_index: Cluster index for lookups
        config: Cluster retrieval configuration
        all_docs_map: unit_id -> doc mapping
        query: Original user query
        llm_provider: LLM provider for cluster selection
        llm_config: Optional LLM configuration

    Returns:
        (expanded_results, metadata)
    """
    metadata = {
        "enabled": True,
        "strategy": "cluster_rerank",
        "original_count": len(original_results),
        "clusters_found": [],
        "clusters_selected": [],
        "selection_reasoning": "",
        "members_per_cluster": {},
        "final_count": 0,
        "truncated": False,
    }

    # Step 1: Extract unique clusters from original results
    cluster_hit_counts: Dict[str, int] = {}  # cluster_id -> hit count
    unique_clusters: List[GroupEventCluster] = []
    seen_cluster_ids: Set[str] = set()

    for doc, score in original_results:
        unit_id = doc.get("unit_id")
        if not unit_id:
            continue

        # Get ALL clusters this unit belongs to (multi-cluster support)
        clusters = cluster_index.get_clusters_by_unit(unit_id)
        if not clusters:
            continue

        for cluster in clusters:
            if cluster.cluster_id not in seen_cluster_ids:
                seen_cluster_ids.add(cluster.cluster_id)
                unique_clusters.append(cluster)

            cluster_hit_counts[cluster.cluster_id] = cluster_hit_counts.get(cluster.cluster_id, 0) + 1

    metadata["clusters_found"] = list(seen_cluster_ids)

    if not unique_clusters:
        logger.warning("  [Cluster Rerank] No clusters found in original results")
        return original_results, metadata

    # 打印找到的 clusters 及其 topic
    cluster_topics = [(c.cluster_id, c.topic) for c in unique_clusters]
    logger.info(f"  ┌─[Cluster Rerank] Found {len(unique_clusters)} clusters from {len(original_results)} results")
    for cid, topic in cluster_topics:
        hit_count = cluster_hit_counts.get(cid, 0)
        logger.info(f"  │  {cid}: \"{topic}\" (hits: {hit_count})")

    # Step 2: Use LLM to select relevant clusters
    clusters_info = format_clusters_for_selection(unique_clusters, cluster_hit_counts)

    prompt = CLUSTER_SELECTION_PROMPT.format(
        query=query,
        clusters_info=clusters_info,
        max_clusters=config.cluster_rerank_max_clusters,
    )

    try:
        response = await llm_provider.generate(prompt)
        selection_result = parse_cluster_selection_response(response)
        selected_cluster_ids = selection_result.get("selected_clusters", [])
        metadata["selection_reasoning"] = selection_result.get("reasoning", "")

        # Validate and limit selected clusters
        valid_selected_ids = [
            cid for cid in selected_cluster_ids
            if cid in seen_cluster_ids
        ][:config.cluster_rerank_max_clusters]

        # Check if LLM explicitly said no clusters are relevant
        reasoning_lower = metadata["selection_reasoning"].lower()
        no_relevant_indicators = [
            "none of the clusters",
            "no cluster",
            "no relevant",
            "not relevant",
            "cannot find",
            "unable to find",
        ]
        llm_found_no_relevant = any(indicator in reasoning_lower for indicator in no_relevant_indicators)

        if not valid_selected_ids or llm_found_no_relevant:
            if llm_found_no_relevant:
                logger.info("  ├─[Cluster Rerank] LLM found no relevant clusters, returning original results")
                metadata["clusters_selected"] = []
                metadata["fallback_to_original"] = True
                # Return original results instead of cluster-based results
                return original_results, metadata
            else:
                logger.warning("LLM selected no valid clusters, falling back to top hit clusters")
                # Fallback: select clusters with most hits
                sorted_clusters = sorted(
                    cluster_hit_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                valid_selected_ids = [cid for cid, _ in sorted_clusters[:config.cluster_rerank_max_clusters]]

        metadata["clusters_selected"] = valid_selected_ids
        logger.info(f"  ├─[Cluster Rerank] LLM selected {len(valid_selected_ids)} clusters: {valid_selected_ids}")
        if metadata["selection_reasoning"]:
            logger.info(f"  │  Reasoning: {metadata['selection_reasoning']}")

    except Exception as e:
        logger.error(f"  [Cluster Rerank] LLM selection failed: {e}")
        # Fallback: select clusters with most hits
        sorted_clusters = sorted(
            cluster_hit_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        valid_selected_ids = [cid for cid, _ in sorted_clusters[:config.cluster_rerank_max_clusters]]
        metadata["clusters_selected"] = valid_selected_ids
        metadata["selection_reasoning"] = f"Fallback due to error: {e}"

    # Step 3: Collect MemUnits from selected clusters (time-ordered, deduplicated)
    final_results: List[Tuple[dict, float]] = []
    total_members_added = 0
    seen_unit_ids: Set[str] = set()  # For deduplication across clusters
    unit_to_cluster_map: Dict[str, List[str]] = {}  # unit_id -> [cluster_ids] mapping

    for cluster_id in valid_selected_ids:
        cluster = cluster_index.get_cluster(cluster_id)
        if not cluster:
            continue

        members_added_for_cluster = 0

        # Members are already time-sorted in cluster
        for member in cluster.members:
            # Check total limit
            if total_members_added >= config.cluster_rerank_total_max_members:
                metadata["truncated"] = True
                break

            # Check per-cluster limit
            if members_added_for_cluster >= config.cluster_rerank_max_members_per_cluster:
                break

            # Deduplicate: skip if already added from another cluster
            if member.unit_id in seen_unit_ids:
                # Track that this unit also belongs to this cluster
                if member.unit_id in unit_to_cluster_map:
                    unit_to_cluster_map[member.unit_id].append(cluster_id)
                continue

            # Get document
            doc = all_docs_map.get(member.unit_id)
            if not doc:
                continue

            # Use a score based on cluster selection order and position
            # Higher score for earlier selected clusters and earlier members
            base_score = 1.0 - (valid_selected_ids.index(cluster_id) * 0.1)
            final_results.append((doc, base_score))
            seen_unit_ids.add(member.unit_id)
            unit_to_cluster_map[member.unit_id] = [cluster_id]
            members_added_for_cluster += 1
            total_members_added += 1

        metadata["members_per_cluster"][cluster_id] = members_added_for_cluster

        # Check if we hit total limit
        if total_members_added >= config.cluster_rerank_total_max_members:
            break

    metadata["cluster_members_count"] = len(final_results)
    metadata["unit_to_cluster"] = unit_to_cluster_map  # 每个 MemUnit 对应的 Cluster

    # Step 4: Hybrid Strategy - Supplement with original rerank results
    original_supplement_count = 0
    if config.hybrid_enable_original_supplement:
        # Calculate remaining budget
        remaining_budget = min(
            config.hybrid_original_supplement_count,
            config.hybrid_max_total_results - len(final_results)
        )

        if remaining_budget > 0:
            # Add original results that are not in cluster results (preserving rerank order)
            for doc, score in original_results:
                if remaining_budget <= 0:
                    break
                unit_id = doc.get("unit_id")
                if not unit_id or unit_id in seen_unit_ids:
                    continue

                # Use decayed score to indicate these are supplementary
                supplementary_score = score * 0.8  # Slight decay for supplementary results
                final_results.append((doc, supplementary_score))
                seen_unit_ids.add(unit_id)
                original_supplement_count += 1
                remaining_budget -= 1

            if original_supplement_count > 0:
                logger.info(f"  ├─[Hybrid] Supplemented {original_supplement_count} MemUnits from original rerank")

    metadata["original_supplement_count"] = original_supplement_count
    metadata["final_count"] = len(final_results)

    # 构建 cluster 详情（用于 checkpoint）
    cluster_details = {}
    for cluster_id in valid_selected_ids:
        cluster = cluster_index.get_cluster(cluster_id)
        if cluster:
            cluster_details[cluster_id] = {
                "topic": cluster.topic,
                "summary": cluster.summary,
                "member_count": len(cluster.members),
                "members_returned": metadata["members_per_cluster"].get(cluster_id, 0),
            }
    metadata["cluster_details"] = cluster_details

    # 打印每个选中 cluster 的详情和 MemUnit IDs
    logger.info(f"  ├─[Cluster Rerank] Selected clusters detail:")
    for cluster_id in valid_selected_ids:
        detail = cluster_details.get(cluster_id, {})
        cluster = cluster_index.get_cluster(cluster_id)
        member_ids = [m.unit_id[:8] for m in cluster.members] if cluster else []
        logger.info(f"  │  {cluster_id}: \"{detail.get('topic', 'N/A')}\" -> {detail.get('members_returned', 0)}/{detail.get('member_count', 0)} members")
        if member_ids:
            logger.info(f"  │    MemUnits: {', '.join(member_ids)}...")

    # 打印最终返回的 MemUnit IDs
    final_unit_ids = [doc.get("unit_id", "")[:8] for doc, _ in final_results]
    cluster_count = metadata.get("cluster_members_count", 0)
    supplement_count = metadata.get("original_supplement_count", 0)
    logger.info(f"  └─[Cluster Rerank] Final: {len(final_results)} MemUnits ({cluster_count} from clusters + {supplement_count} from original rerank)")
    logger.info(f"     Final MemUnit IDs: {', '.join(final_unit_ids)}")
    if metadata["truncated"]:
        logger.info(f"     ⚠️ Result truncated (reached max {config.cluster_rerank_total_max_members} members)")

    return final_results, metadata
