"""Utility functions for Group Event Cluster system.

This module provides:
1. Re-exported prompt templates from centralized location
2. Helper functions for formatting cluster data for LLM prompts
3. Response parsing utilities for LLM outputs

Prompts are defined in: src/prompts/memory/en/eval/group_event_cluster_prompts.py
"""

import json
import re

# Re-export prompts from centralized location
from prompts.memory.en.eval.group_event_cluster_prompts import (
    CLUSTER_ASSIGNMENT_PROMPT,
    CLUSTER_SUMMARY_PROMPT,
    UNIT_SUMMARY_PROMPT,
    CLUSTER_TOPIC_PROMPT,
    CLUSTER_SELECTION_PROMPT,
)

__all__ = [
    # Prompts (re-exported)
    "CLUSTER_ASSIGNMENT_PROMPT",
    "CLUSTER_SUMMARY_PROMPT",
    "UNIT_SUMMARY_PROMPT",
    "CLUSTER_TOPIC_PROMPT",
    "CLUSTER_SELECTION_PROMPT",
    # Helper functions
    "format_clusters_for_prompt",
    "format_members_for_summary",
    "parse_cluster_assignment_response",
    # Cluster Rerank helpers
    "format_clusters_for_selection",
    "parse_cluster_selection_response",
]


# =============================================================================
# Helper Functions
# =============================================================================

def format_clusters_for_prompt(
    clusters: list,
    max_clusters: int = 20,
    max_members_per_cluster: int = 3,
) -> str:
    """
    Format cluster information for inclusion in prompts.

    Args:
        clusters: List of GroupEventCluster objects
        max_clusters: Maximum clusters to include
        max_members_per_cluster: Maximum members to show per cluster

    Returns:
        Formatted string representation of clusters
    """
    if not clusters:
        return "No existing clusters yet."

    lines = []
    for i, cluster in enumerate(clusters[:max_clusters]):
        lines.append(f"\n### Cluster {cluster.cluster_id}")
        lines.append(f"- Topic: {cluster.topic}")
        lines.append(f"- Members ({len(cluster.members)} total):")

        # Show recent members (last N by time)
        recent_members = cluster.members[-max_members_per_cluster:]
        for member in recent_members:
            timestamp_str = member.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  - [{timestamp_str}] {member.summary}")

    if len(clusters) > max_clusters:
        lines.append(f"\n... and {len(clusters) - max_clusters} more clusters")

    return "\n".join(lines)


def format_members_for_summary(members: list) -> str:
    """
    Format cluster members for summary generation prompt.

    Args:
        members: List of ClusterMember objects (should be time-sorted)

    Returns:
        Formatted string representation of members
    """
    lines = []
    for i, member in enumerate(members, 1):
        timestamp_str = member.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{i}. [{timestamp_str}] {member.summary}")

    return "\n".join(lines)


# =============================================================================
# Response Parsing
# =============================================================================

def parse_cluster_assignment_response(response: str) -> dict:
    """
    Parse LLM response for cluster assignment.

    Always returns the multi-assignment format: {"assignments": [...], "reason": "..."}

    Handles LLM response variations (not data format compatibility):
    1. Multi-assignment format: {"assignments": [...], "reason": "..."}
    2. Single-assignment format (converted): {"decision": "...", "cluster_id": "...", ...}

    Args:
        response: Raw LLM response string

    Returns:
        Parsed decision dict: {"assignments": [...], "reason": "..."}
    """
    # Try to extract JSON from response
    parsed = None
    try:
        # First try direct JSON parse
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response (handle nested braces for arrays)
    if not parsed:
        # Try to match JSON with arrays
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    if parsed:
        # Check if it's new multi-assignment format
        if "assignments" in parsed:
            return parsed
        # It's legacy format, convert to new format for consistency
        if "decision" in parsed:
            return _convert_legacy_to_multi_assignment(parsed)
        return parsed

    # Fallback: try to parse key-value pairs from text
    result = {"assignments": [], "reason": "Failed to parse response"}

    if "EXISTING" in response.upper():
        # Try to extract cluster_id(s)
        cluster_matches = re.findall(r'gec_\d+', response)
        for cluster_id in cluster_matches:
            result["assignments"].append({"type": "EXISTING", "cluster_id": cluster_id})

    if "NEW" in response.upper():
        # Try to extract topic
        topic_match = re.search(r'"new_topic":\s*"([^"]+)"', response)
        if topic_match:
            result["assignments"].append({"type": "NEW", "new_topic": topic_match.group(1)})
        elif not result["assignments"]:
            result["assignments"].append({"type": "NEW", "new_topic": "Unknown topic"})

    # If no assignments found, default to NEW
    if not result["assignments"]:
        result["assignments"].append({"type": "NEW", "new_topic": "Unknown topic"})

    return result


def _convert_legacy_to_multi_assignment(legacy: dict) -> dict:
    """Convert single-assignment LLM response to multi-assignment format."""
    assignments = []

    if legacy.get("decision") == "EXISTING":
        cluster_id = legacy.get("cluster_id", "")
        if cluster_id:
            assignments.append({"type": "EXISTING", "cluster_id": cluster_id})
    elif legacy.get("decision") == "NEW":
        new_topic = legacy.get("new_topic", "Unknown topic")
        assignments.append({"type": "NEW", "new_topic": new_topic})

    return {
        "assignments": assignments,
        "reason": legacy.get("reason", ""),
    }


# =============================================================================
# Cluster Selection Helpers (for cluster_rerank strategy)
# =============================================================================

def format_clusters_for_selection(
    clusters: list,
    hit_counts: dict = None,
) -> str:
    """
    Format cluster information for LLM cluster selection.

    Args:
        clusters: List of GroupEventCluster objects
        hit_counts: Optional dict mapping cluster_id -> number of hits from original results

    Returns:
        Formatted string representation of clusters for selection prompt
    """
    if not clusters:
        return "No clusters available."

    lines = []
    for cluster in clusters:
        # Header with cluster ID and topic
        hit_info = ""
        if hit_counts and cluster.cluster_id in hit_counts:
            hit_info = f" (hits: {hit_counts[cluster.cluster_id]})"
        lines.append(f"\n### [{cluster.cluster_id}] {cluster.topic}{hit_info}")

        # Summary if available
        if cluster.summary:
            lines.append(f"Summary: {cluster.summary}")

        # Member count and time range
        lines.append(f"Members: {len(cluster.members)} MemUnits")
        if cluster.first_timestamp and cluster.last_timestamp:
            first_str = cluster.first_timestamp.strftime("%Y-%m-%d")
            last_str = cluster.last_timestamp.strftime("%Y-%m-%d")
            if first_str == last_str:
                lines.append(f"Time: {first_str}")
            else:
                lines.append(f"Time: {first_str} to {last_str}")

        # Show a few recent member summaries for context
        recent_members = cluster.members[-3:]  # Last 3 members
        if recent_members:
            lines.append("Recent content:")
            for member in recent_members:
                summary_preview = member.summary[:100] + "..." if len(member.summary) > 100 else member.summary
                lines.append(f"  - {summary_preview}")

    return "\n".join(lines)


def parse_cluster_selection_response(response: str) -> dict:
    """
    Parse LLM response for cluster selection.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed selection dict with keys: selected_clusters (list), reasoning (str)
    """
    # Try to extract JSON from response
    try:
        # First try direct JSON parse
        result = json.loads(response.strip())
        if "selected_clusters" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    json_match = re.search(r'\{[^{}]*"selected_clusters"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract cluster IDs directly
    result = {"selected_clusters": [], "reasoning": "Failed to parse response"}

    # Look for gec_XXX patterns
    cluster_matches = re.findall(r'gec_\d+', response)
    if cluster_matches:
        # Remove duplicates while preserving order
        seen = set()
        unique_clusters = []
        for c in cluster_matches:
            if c not in seen:
                seen.add(c)
                unique_clusters.append(c)
        result["selected_clusters"] = unique_clusters

    return result
