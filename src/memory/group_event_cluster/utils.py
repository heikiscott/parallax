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
)

__all__ = [
    # Prompts (re-exported)
    "CLUSTER_ASSIGNMENT_PROMPT",
    "CLUSTER_SUMMARY_PROMPT",
    "UNIT_SUMMARY_PROMPT",
    "CLUSTER_TOPIC_PROMPT",
    # Helper functions
    "format_clusters_for_prompt",
    "format_members_for_summary",
    "parse_cluster_assignment_response",
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

    Args:
        response: Raw LLM response string

    Returns:
        Parsed decision dict with keys: decision, cluster_id (optional),
        new_topic (optional), reason (optional)
    """
    # Try to extract JSON from response
    try:
        # First try direct JSON parse
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse key-value pairs
    result = {"decision": "NEW", "reason": "Failed to parse response"}

    if "EXISTING" in response.upper():
        result["decision"] = "EXISTING"
        # Try to extract cluster_id
        cluster_match = re.search(r'gec_\d+', response)
        if cluster_match:
            result["cluster_id"] = cluster_match.group()

    if "NEW" in response.upper():
        result["decision"] = "NEW"
        # Try to extract topic (look for quoted string or after "topic:")
        topic_match = re.search(r'"new_topic":\s*"([^"]+)"', response)
        if topic_match:
            result["new_topic"] = topic_match.group(1)

    return result
