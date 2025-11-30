"""LLM prompt templates for Group Event Cluster system.

This module contains all prompt templates used by the GroupEventClusterer
for clustering decisions and summary generation.
"""

# =============================================================================
# Cluster Assignment Prompt
# =============================================================================

CLUSTER_ASSIGNMENT_PROMPT = """You are a memory clustering assistant. Your task is to determine whether a new memory unit belongs to an existing event cluster or should create a new cluster.

## Context
We are building a group perspective memory system that clusters related memory units (MemUnits) by events. Each cluster represents a coherent event or topic discussed in a conversation.

## Existing Clusters
{existing_clusters}

## New Memory Unit
- Unit ID: {unit_id}
- Timestamp: {timestamp}
- Summary: {unit_summary}
- Narrative: {narrative}

## Task
Analyze the new memory unit and determine:
1. Does it belong to any existing cluster? (Consider: same event, same participants, temporal/causal connection)
2. If yes, which cluster? (Provide cluster_id)
3. If no, it should be a new cluster. Provide a topic name (10-30 chars) for the new cluster.

## Rules
- A memory unit belongs to a cluster if it discusses the SAME specific event
- "Same event" means: same core participants + same specific activity/topic + temporal or causal connection
- Different events about the same person are DIFFERENT clusters (e.g., "John's birthday" vs "John's promotion")
- When in doubt, create a new cluster (avoid over-merging)

## Response Format (JSON)
If belongs to existing cluster:
{{"decision": "EXISTING", "cluster_id": "gec_XXX", "reason": "brief explanation"}}

If should create new cluster:
{{"decision": "NEW", "new_topic": "Topic Name Here", "reason": "brief explanation"}}

Respond with JSON only, no additional text."""

# =============================================================================
# Cluster Summary Generation Prompt
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """You are a memory summarization assistant. Generate a comprehensive summary for a group event cluster.

## Cluster Information
- Topic: {topic}
- Members (chronological order):
{members_info}

## Task
Generate a summary that:
1. Uses third-person group perspective (e.g., "The group discussed...", "Caroline shared with Melanie...")
2. Captures the key facts: who, what, when, where, why
3. Shows the progression of the event/topic over time
4. Is 100-300 characters in length
5. Is written in the same language as the input content

## Response Format
Provide only the summary text, no additional formatting or explanation."""

# =============================================================================
# Unit Summary Extraction Prompt
# =============================================================================

UNIT_SUMMARY_PROMPT = """Extract a brief summary (1-2 sentences) from this memory content.

## Content
{narrative}

## Requirements
- Capture the core event or topic
- Include key participants if mentioned
- Keep it concise (1-2 sentences)
- Use third-person perspective
- Preserve the language of the original content

## Response
Provide only the summary text."""

# =============================================================================
# Cluster Topic Generation Prompt
# =============================================================================

CLUSTER_TOPIC_PROMPT = """Generate a short topic name for this event cluster.

## Event Description
{description}

## Requirements
- 10-30 characters
- Descriptive and specific
- Include key participant if relevant
- Use the same language as the input

## Examples
Good: "Caroline's adoption plan"
Good: "Weekend picnic planning"
Bad: "Conversation" (too vague)
Bad: "Discussion about various topics" (too generic)

## Response
Provide only the topic name."""
