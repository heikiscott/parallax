"""LLM prompt templates for Group Event Cluster system.

This module contains all prompt templates used by the GroupEventClusterer
for clustering decisions and summary generation.
"""

# =============================================================================
# Cluster Assignment Prompt
# =============================================================================

CLUSTER_ASSIGNMENT_PROMPT = """You are a memory clustering assistant. Your task is to determine which thematic cluster(s) a new memory unit belongs to.

## Context
We are building a group perspective memory system that clusters related memory units (MemUnits) by **TOPIC/THEME**, not by time. Each cluster represents **ONE specific topic** that may span across multiple conversations and time periods.

**CRITICAL**: A single MemUnit can belong to MULTIPLE clusters if it discusses multiple distinct topics. You MUST assign to ALL relevant clusters, not just the primary one.

## Existing Clusters
{existing_clusters}

## New Memory Unit
- Unit ID: {unit_id}
- Timestamp: {timestamp}
- Summary: {unit_summary}
- Narrative: {narrative}

## Task
Carefully analyze the new memory unit and determine:
1. **List ALL topics** mentioned in this MemUnit (even briefly mentioned ones)
2. For EACH topic, check if an existing cluster matches
3. If a topic has no matching cluster, create a NEW one

## CRITICAL Rules

### Rule 1: Assign to ALL Relevant Clusters (Multi-Assignment)
- **ALWAYS** assign to EVERY cluster that relates to the content
- A MemUnit mentioning 3 topics → assign to 3 clusters
- Do NOT just pick the "best" or "primary" cluster - assign to ALL matching ones
- Example: MemUnit about "camping trip + kids' excitement + summer plans" → assign to clusters for camping, family activities, AND summer planning

### Rule 2: Single Topic Per Cluster
- **Each cluster = ONE specific topic** (never combine topics)
- **NEVER** create compound topics like:
  - ❌ "Caroline's career and Melanie's art"
  - ❌ "Family camping and adoption plans"
- **ALWAYS** use singular, focused topics like:
  - ✅ "Caroline's career plans"
  - ✅ "Melanie's camping trips"
  - ✅ "Caroline's adoption journey"

### Rule 3: Granular Topic Differentiation
- Create SPECIFIC topics, not broad categories
- Differentiate related but distinct topics:
  - ✅ "Melanie's pottery class" (separate from painting)
  - ✅ "Melanie's painting hobby" (separate from pottery)
  - ✅ "Caroline's adoption research" (separate from adoption meetings)
  - ✅ "Caroline's LGBTQ support group" (separate from LGBTQ conference)

## Examples of Multi-Assignment
- MemUnit: "Melanie talked about her kids' excitement for camping, and Caroline shared her adoption research"
  → Assign to: "Melanie's family activities", "Melanie's camping trips", "Caroline's adoption journey" (3 clusters)

- MemUnit: "Caroline attended LGBTQ conference and discussed her counseling career plans"
  → Assign to: "Caroline's LGBTQ advocacy", "Caroline's career plans" (2 clusters)

- MemUnit: "Melanie shared a painting she made last year"
  → Assign to: "Melanie's painting hobby" only (1 cluster)

## Response Format (JSON)
{{"assignments": [
    {{"type": "EXISTING", "cluster_id": "gec_001"}},
    {{"type": "EXISTING", "cluster_id": "gec_003"}},
    {{"type": "NEW", "new_topic": "Specific Single Topic"}}
], "reason": "Topic1 matches gec_001, Topic2 matches gec_003, Topic3 is new"}}

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

CLUSTER_TOPIC_PROMPT = """Generate a short, HIGHLY SPECIFIC topic name for this thematic cluster.

## Event Description
{description}

## Requirements
- 15-40 characters (can be longer if needed for specificity)
- Must be a HIGHLY SPECIFIC SINGLE THEME
- Format: "[Person]'s [specific action/item]" or "[Specific event/activity]"
- Include the main participant's name when relevant
- Use the same language as the input

## CRITICAL: MAXIMUM SPECIFICITY
- Be as specific as possible to avoid semantic overlap with other clusters
- Include distinguishing details (activity type, context, item)

## Specificity Hierarchy (prefer more specific):
- ❌ "Melanie's hobbies" → too broad
- ❌ "Melanie's art" → still broad
- ✅ "Melanie's painting hobby" → good
- ✅✅ "Melanie's lake sunrise painting" → best (if applicable)

- ❌ "Caroline's LGBTQ activities" → too broad
- ✅ "Caroline's LGBTQ support group" → good
- ✅ "Caroline's LGBTQ conference attendance" → good (different from support group)

## Good Examples (Highly Specific)
- "Melanie's pottery class"
- "Melanie's painting hobby"
- "Caroline's adoption research"
- "Caroline's adoption council meeting"
- "Caroline's LGBTQ support group"
- "Caroline's transgender conference"
- "Caroline's counseling career goal"
- "Melanie's family camping trips"
- "Melanie's charity 5K run"
- "Caroline's school LGBTQ speech"

## Bad Examples (AVOID)
- ❌ "Caroline's career and Melanie's art" (compound)
- ❌ "Art and creativity" (compound + vague)
- ❌ "Conversation" (too vague)
- ❌ "Personal updates" (too vague)
- ❌ "Support and encouragement" (too vague)
- ❌ "Life events" (too vague)

## Response
Provide only the topic name (single specific topic, no compounds)."""


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
