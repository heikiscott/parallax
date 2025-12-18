"""Type-Aware Multi-Query Generation Prompts for Agentic Retrieval V2.

This module provides question-type-specific prompt templates for generating
multiple complementary search queries at Round 1 (before any retrieval).

Unlike the standard multi_query_prompts.py which generates queries based on
retrieved documents and missing_info, these prompts generate queries purely
based on the original question and its classified type.

Design Philosophy:
- Simple questions (ATTRIBUTE_*) skip multi-query generation
- Complex questions use type-specific prompts for better coverage
- Temporal questions get special treatment for time-related variations
"""

from typing import Set, Optional

from retrieval.classification.question_classifier import QuestionType, ComplexityLevel


# =============================================================================
# Skip Multi-Query Types
# =============================================================================

SKIP_MULTI_QUERY_TYPES: Set[QuestionType] = {
    QuestionType.ATTRIBUTE_LOCATION,
    QuestionType.ATTRIBUTE_IDENTITY,
}


def should_use_multi_query(
    question_type: QuestionType,
    confidence: float,
    threshold: float = 0.85,
    complexity_level: Optional[ComplexityLevel] = None,
) -> bool:
    """Determine whether to use multi-query generation for a question.

    Args:
        question_type: Classified question type
        confidence: Classification confidence score (0.0-1.0)
        threshold: Minimum confidence to skip multi-query for simple types
        complexity_level: Optional complexity level for more direct control

    Returns:
        True if multi-query should be used, False to skip

    Logic:
        - SIMPLE complexity: Skip multi-query (single query is sufficient)
        - COMPLEX complexity: Always use multi-query (need comprehensive coverage)
        - MODERATE or None: Use existing type-based logic
    """
    # If complexity level is provided, use it for direct control
    if complexity_level is not None:
        if complexity_level == ComplexityLevel.SIMPLE:
            return False  # SIMPLE queries skip multi-query
        if complexity_level == ComplexityLevel.COMPLEX:
            return True   # COMPLEX queries always use multi-query

    # Fallback to type-based logic for MODERATE or when complexity not provided
    if question_type in SKIP_MULTI_QUERY_TYPES and confidence >= threshold:
        return False
    return True


# =============================================================================
# Type-Aware Prompt Templates
# =============================================================================

TYPE_AWARE_MULTI_QUERY_PROMPT_TEMPORAL = """You are an expert at query reformulation for temporal event retrieval.

**Task**: Generate 3 complementary search queries to find WHEN a specific event happened.

**Original Query**:
{original_query}

**Question Type**: EVENT_TEMPORAL (When did X happen?)

**Strategy for Temporal Questions**:
Temporal questions often fail due to:
1. Relative time expressions ("the week before X", "around April") being hard to match exactly
2. Event synonyms not being covered
3. Time granularity mismatch (date vs. week vs. month)
4. **Adjacent time periods being missed** - if event happened "week of April 3-9", searching only "April 9-15" will miss it

**Generate queries that**:
1. **Entity-focused query**: Extract core entities (person name + event/action keywords) without time speculation
   - Example: "John firefighter call-out first"
2. **Event synonym query**: Use alternative verbs/phrases for the same event
   - Example: "John firefighting started began"
3. **Temporal neighborhood query**: If the query mentions a specific time period, include ADJACENT periods (week before/after, early/late month)
   - Example: If asking about "April", include "late March early April" or "March April"
   - Example: If asking about "first week of May", include "late April early May"

**Critical Rules**:
- DO NOT guess or fabricate specific dates/times
- DO NOT add information not in the original query
- Extract and repeat exact entity names from the query
- Include event synonyms (started/began/first time/initially)
- **When time period is mentioned, generate queries covering adjacent periods to handle off-by-one-week errors**
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "First query focusing on entities and action",
    "Second query with event synonyms",
    "Third query covering adjacent time periods"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_ACTIVITY = """You are an expert at query reformulation for activity retrieval.

**Task**: Generate 2-3 complementary search queries to find activities/hobbies someone does.

**Original Query**:
{original_query}

**Question Type**: EVENT_ACTIVITY (What activities does X do?)

**Strategy for Activity Questions**:
1. **Specific activity query**: Target exact activity types mentioned or implied
2. **Synonym expansion query**: Use activity synonyms (hobbies/pastimes/interests/leisure)
3. **Context query**: Include related concepts (relaxation/destress/free time/weekends)

**Generate queries covering**:
- Specific named activities if mentioned
- General activity/hobby vocabulary variants
- Time contexts (regular/daily/weekly/weekend activities)

**Critical Rules**:
- Extract and repeat person name exactly as stated
- Include both general terms (activities, hobbies) and specifics
- DO NOT fabricate activities not implied by the query
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "Query with specific activity focus",
    "Query with synonym expansion",
    "Query with context expansion (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_AGGREGATION = """You are an expert at query reformulation for aggregation retrieval.

**Task**: Generate 2-3 complementary search queries to find ALL items/events in a category.

**Original Query**:
{original_query}

**Question Type**: EVENT_AGGREGATION (What X has person done/read/visited?)

**Strategy for Aggregation Questions**:
Aggregation questions require COMPLETENESS - missing items is the main failure mode.

1. **Direct enumeration query**: Ask for the list directly
   - Example: "What books has Alice read list"
2. **Category exploration query**: Use broader category terms
   - Example: "Alice reading literature novels fiction"
3. **Temporal spread query**: Cover different time periods
   - Example: "Alice books read recently past year"

**Critical Rules**:
- Focus on COMPLETENESS over precision
- Use list-oriented language (all, every, complete, full list)
- Include category synonyms and hypernyms
- Extract person name exactly as stated
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "Direct enumeration query",
    "Category exploration query",
    "Temporal or context spread query (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_COUNTING = """You are an expert at query reformulation for counting/listing retrieval.

**Task**: Generate 2-3 complementary search queries to find items for counting or listing.

**Original Query**:
{original_query}

**Question Type**: COUNTING (How many X? List all X?)

**Strategy for Counting Questions**:
Counting requires finding ALL instances - partial retrieval leads to wrong counts.

1. **Explicit count query**: Use counting language
   - Example: "How many times did Bob visit the gym"
2. **List all query**: Ask for enumeration
   - Example: "List all Bob's gym visits dates"
3. **Instance-focused query**: Target individual occurrences
   - Example: "Bob gym workout sessions each time"

**Critical Rules**:
- Use quantity words (how many, count, number of, total, all)
- Include list-oriented language (list, enumerate, each, every)
- Extract exact entities from the query
- DO NOT guess counts or fabricate instances
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "Explicit counting query",
    "List/enumerate query",
    "Instance-focused query (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_REASONING = """You are an expert at query reformulation for reasoning/inference retrieval.

**Task**: Generate 2-3 complementary search queries to gather evidence for reasoning questions.

**Original Query**:
{original_query}

**Question Type**: REASONING (Would X do Y? Could X have Z?)

**Strategy for Reasoning Questions**:
Reasoning questions need multiple pieces of evidence to support inference.

1. **Direct evidence query**: Find statements directly relevant to the hypothesis
   - Example: "Alice career interests writing passion"
2. **Background context query**: Find supporting context
   - Example: "Alice education background skills experience"
3. **Related behavior query**: Find similar past behaviors/preferences
   - Example: "Alice previous career choices decisions"

**Critical Rules**:
- Seek factual evidence, not speculation
- Cover multiple angles that could support or refute the hypothesis
- Extract exact entity names from the query
- DO NOT include hypothetical language in queries
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "Direct evidence query",
    "Background context query",
    "Related behavior query (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_PREFERENCE = """You are an expert at query reformulation for preference retrieval.

**Task**: Generate 2-3 complementary search queries to find someone's preferences/likes.

**Original Query**:
{original_query}

**Question Type**: ATTRIBUTE_PREFERENCE (What does X like/prefer?)

**Strategy for Preference Questions**:
1. **Direct preference query**: Ask about likes/favorites directly
2. **Behavior inference query**: Look for actions that indicate preference
3. **Category-specific query**: Target specific preference categories

**Generate queries covering**:
- Direct statements of preference (like, love, favorite, prefer)
- Behavioral evidence (does, chooses, buys, uses)
- Specific categories if mentioned (food, music, activities)

**Critical Rules**:
- Extract person name exactly
- Include preference verbs (like, love, enjoy, prefer, favorite)
- Cover both stated preferences and behavioral evidence
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "Direct preference query",
    "Behavior-based query",
    "Category-specific query (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""

TYPE_AWARE_MULTI_QUERY_PROMPT_GENERAL = """You are an expert at query reformulation for information retrieval.

**Task**: Generate 2-3 complementary search queries to maximize recall and precision.

**Original Query**:
{original_query}

**Question Type**: GENERAL

**Strategy**:
1. **High-precision anchor query**: Extract exact entities and constraints
2. **Synonym expansion query**: Add vocabulary variants
3. **Broader context query**: Include related concepts

**Critical Rules**:
- DO NOT introduce facts not in the original query
- Extract and repeat exact entity names
- Use synonyms and paraphrases for better coverage
- Keep queries 6-15 words each

**Output Format** (strict JSON):
{{
  "queries": [
    "High-precision anchor query",
    "Synonym expansion query",
    "Broader context query (optional)"
  ],
  "reasoning": "Brief explanation (1 sentence)"
}}

Now generate queries:
"""


# =============================================================================
# Prompt Template Mapping
# =============================================================================

TYPE_AWARE_PROMPTS = {
    QuestionType.EVENT_TEMPORAL: TYPE_AWARE_MULTI_QUERY_PROMPT_TEMPORAL,
    QuestionType.EVENT_ACTIVITY: TYPE_AWARE_MULTI_QUERY_PROMPT_ACTIVITY,
    QuestionType.EVENT_AGGREGATION: TYPE_AWARE_MULTI_QUERY_PROMPT_AGGREGATION,
    QuestionType.COUNTING: TYPE_AWARE_MULTI_QUERY_PROMPT_COUNTING,
    QuestionType.REASONING_HYPOTHETICAL: TYPE_AWARE_MULTI_QUERY_PROMPT_REASONING,
    QuestionType.REASONING_INFERENCE: TYPE_AWARE_MULTI_QUERY_PROMPT_REASONING,
    QuestionType.ATTRIBUTE_PREFERENCE: TYPE_AWARE_MULTI_QUERY_PROMPT_PREFERENCE,
    QuestionType.TIME_CALCULATION: TYPE_AWARE_MULTI_QUERY_PROMPT_TEMPORAL,  # Similar to temporal
    QuestionType.GENERAL: TYPE_AWARE_MULTI_QUERY_PROMPT_GENERAL,
}


def get_prompt_for_type(question_type: QuestionType) -> str:
    """Get the appropriate prompt template for a question type.

    Args:
        question_type: The classified question type

    Returns:
        Prompt template string with {original_query} placeholder
    """
    return TYPE_AWARE_PROMPTS.get(question_type, TYPE_AWARE_MULTI_QUERY_PROMPT_GENERAL)


__all__ = [
    "SKIP_MULTI_QUERY_TYPES",
    "should_use_multi_query",
    "TYPE_AWARE_PROMPTS",
    "get_prompt_for_type",
]
