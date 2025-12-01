"""LLM prompt templates for Group Event Cluster system.

This module contains all prompt templates used by the GroupEventClusterer
for clustering decisions and summary generation.
"""

# =============================================================================
# Cluster Assignment Prompt
# =============================================================================

CLUSTER_ASSIGNMENT_PROMPT = """You are a memory clustering assistant. Your task is to determine whether a new memory unit belongs to an existing thematic cluster or should create a new cluster.

## Context
We are building a group perspective memory system that clusters related memory units (MemUnits) by **TOPIC/THEME**, not by time. Each cluster represents a **specific topic or event theme** that may span across multiple conversations and time periods.

## Existing Clusters
{existing_clusters}

## New Memory Unit
- Unit ID: {unit_id}
- Timestamp: {timestamp}
- Summary: {unit_summary}
- Narrative: {narrative}

## Task
Analyze the new memory unit and determine:
1. Does it belong to any existing cluster based on **TOPIC SIMILARITY**? (Ignore timestamp - same topic discussed at different times should be in the same cluster)
2. If yes, which cluster? (Provide cluster_id)
3. If no, create a new cluster with a specific topic name (10-30 chars)

## CRITICAL Rules - Topic-Based Clustering
- **TOPIC is the PRIMARY criterion**, NOT time proximity
- Same topic discussed on different days/months → SAME cluster
- Different topics discussed in the same conversation → DIFFERENT clusters
- A cluster should represent ONE specific theme, such as:
  - "Caroline's LGBTQ journey" (all discussions about her LGBTQ experiences)
  - "Melanie's painting hobby" (all mentions of painting/art)
  - "Caroline's adoption plan" (all discussions about adoption)
  - "Family support and relationships" (discussions about family bonds)
- Do NOT create vague clusters like "Catch-up conversation" or "Daily chat"
- Each cluster should answer: "What SPECIFIC TOPIC is this about?"

## Examples
- MemUnit about LGBTQ support group on May 7 → "Caroline's LGBTQ journey"
- MemUnit about LGBTQ school event on June 9 → SAME cluster "Caroline's LGBTQ journey" (same topic, different time)
- MemUnit about painting on May 8 → "Melanie's painting hobby"
- MemUnit about career counseling on May 8 → NEW cluster "Caroline's career plans" (different topic, same day)

## Response Format (JSON)
If belongs to existing cluster:
{{"decision": "EXISTING", "cluster_id": "gec_XXX", "reason": "Same topic: [topic name]"}}

If should create new cluster:
{{"decision": "NEW", "new_topic": "Specific Topic Name", "reason": "New topic not covered by existing clusters"}}

Respond with JSON only, no additional text."""

# =============================================================================
# Cluster Summary Generation Prompt
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """You are a memory summarization assistant. Generate a comprehensive summary for a thematic cluster.

## Cluster Information
- Topic: {topic}
- Members (may span different time periods):
{members_info}

## Task
Generate a summary that:
1. Uses third-person group perspective (e.g., "Caroline shared with Melanie...", "They discussed...")
2. Focuses on the **THEME** and key information across all time periods
3. Highlights the evolution or continuity of this topic over time (if applicable)
4. Is 100-300 characters in length
5. Is written in the same language as the input content

## Note
This cluster groups memories by TOPIC, not by time. Members may come from different conversations/dates but share the same theme.

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

CLUSTER_TOPIC_PROMPT = """Generate a short, SPECIFIC topic name for this thematic cluster.

## Event Description
{description}

## Requirements
- 10-30 characters
- Must be a SPECIFIC THEME, not a generic description
- Format: "[Person]'s [specific topic]" or "[Specific activity/theme]"
- Include the main participant's name when relevant
- Use the same language as the input

## Good Examples (Specific Themes)
- "Caroline's LGBTQ journey"
- "Melanie's painting hobby"
- "Caroline's adoption plan"
- "Caroline's career goals"
- "Family picnic planning"
- "Mental health charity run"

## Bad Examples (Too Vague - AVOID)
- "Conversation"
- "Catch-up chat"
- "Daily discussion"
- "Caroline and Melanie's talk"
- "Life updates"

## Response
Provide only the topic name."""


# =============================================================================
# Cluster Selection Prompt (for cluster_rerank strategy)
# =============================================================================

CLUSTER_SELECTION_PROMPT = """You are a memory retrieval assistant. Your task is to select the most relevant event clusters for answering a user query.

## User Query
{query}

## Candidate Clusters
{clusters_info}

## Task
Analyze the query and select the cluster(s) that contain information needed to answer it.

## Selection Guidelines

**Choose the appropriate number of clusters based on query complexity:**

- **1 cluster**: The query asks about a specific single event (e.g., "When did Caroline decide to adopt?")
- **2-3 clusters**: The query involves related events or needs context from multiple discussions
- **4+ clusters**: The query is broad, comparative, or asks about patterns/trends across multiple events

**Selection principles:**
- Only select clusters that directly help answer the query
- Prefer fewer, highly relevant clusters over many marginally relevant ones
- If unsure whether a cluster is needed, include it (recall is important)
- Maximum {max_clusters} clusters can be selected

## Response Format (JSON)
{{"selected_clusters": ["gec_001", "gec_003"], "reasoning": "Brief explanation of selection"}}

If no cluster is relevant (rare):
{{"selected_clusters": [], "reasoning": "None of the clusters contain relevant information"}}

Respond with JSON only."""
