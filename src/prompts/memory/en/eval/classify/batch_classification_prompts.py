"""Batch Question Classification Prompts.

This module provides prompts for batch classification of questions
in the Classify Stage of the evaluation pipeline.

The prompt is designed to:
1. Classify multiple questions in a single LLM call for efficiency
2. Produce consistent results with QuestionClassifier (rule-based)
3. Return structured JSON for easy parsing

NOTE: This prompt ONLY handles question type classification.
The retrieval strategy is determined by each pipeline independently:
- V4 pipeline: Uses cross-attention retrieval for all types
- V3 pipeline: Uses ColBERT retrieval
- V1/V2 pipeline: Uses GEC-based strategies

This prompt covers ALL 11 categories defined in QuestionType enum:
1. EVENT_TEMPORAL, EVENT_ACTIVITY, EVENT_AGGREGATION
2. ATTRIBUTE_IDENTITY, ATTRIBUTE_PREFERENCE, ATTRIBUTE_LOCATION
3. REASONING_HYPOTHETICAL, REASONING_INFERENCE
4. TIME_CALCULATION, COUNTING
5. GENERAL
"""

# =============================================================================
# BATCH CLASSIFICATION PROMPT
# =============================================================================
# This prompt classifies multiple questions at once.
# It ONLY determines the question TYPE - the retrieval strategy is decided
# by each pipeline independently based on its architecture.
# =============================================================================

BATCH_CLASSIFICATION_PROMPT = """You are a question classifier for a memory retrieval system.
Your task is to classify each question to determine its semantic type.

## IMPORTANT INSTRUCTIONS
1. You MUST classify ALL {num_questions} questions provided below
2. Return a JSON object with a "classifications" array
3. Each classification must include: id, category, confidence, reasoning
4. Do NOT skip any question

## Question Categories (11 total)

### 1. EVENT_TEMPORAL
**Description:** Questions asking WHEN a specific event happened or will happen.
**Key Indicators:** "when", "what time", "what date", specific time references
**Examples:**
- "When did Caroline go to the LGBTQ support group?" → Asks for a specific date/time
- "When did Melanie paint a sunrise?" → Asks when an activity occurred
- "When is Caroline going to the conference?" → Asks about future event timing
- "What time did Sam arrive at the party?" → Asks for specific time

### 2. EVENT_ACTIVITY
**Description:** Questions about activities, hobbies, or actions someone does/did.
**Key Indicators:** "what does X do", "what activities", "hobbies", action verbs
**Examples:**
- "What activities does Melanie partake in?" → Asks about general activities
- "What does Melanie do to destress?" → Asks about relaxation activities
- "What hobbies does Sam have?" → Asks about recurring activities
- "How does Caroline spend her weekends?" → Asks about regular activities

### 3. EVENT_AGGREGATION
**Description:** Questions requiring collection/aggregation of multiple events or items over time.
**Key Indicators:** "what books", "where has X been", "all the places", plural nouns referring to events
**Note:** Different from COUNTING - EVENT_AGGREGATION asks for a LIST of things, COUNTING asks for a NUMBER.
**Examples:**
- "What books has Melanie read?" → Requires collecting multiple book mentions (list)
- "Where has Melanie camped?" → Requires collecting multiple camping locations (list)
- "What restaurants have they visited?" → Requires aggregating multiple visits (list)
- "What movies did Sam watch last month?" → Collecting multiple events (list)

### 4. ATTRIBUTE_IDENTITY
**Description:** Questions about fixed personal attributes, identity, or status.
**Key Indicators:** "what is X's", "who is", identity/status words, profession, relationships
**Examples:**
- "What is Caroline's identity?" → Asks about core identity
- "What is Caroline's relationship status?" → Asks about relationship state
- "What is Sam's job?" → Asks about profession (stable attribute)
- "Who is Melanie's sister?" → Asks about family relationship

### 5. ATTRIBUTE_PREFERENCE
**Description:** Questions about preferences, likes, dislikes, or favorites.
**Key Indicators:** "favorite", "prefer", "like", "enjoy", "love", "hate"
**Examples:**
- "What do Melanie's kids like?" → Asks about preferences
- "What is Caroline's favorite book?" → Asks about favorite item
- "What kind of music does Sam prefer?" → Asks about preference
- "What food does Melanie hate?" → Asks about dislikes

### 6. ATTRIBUTE_LOCATION
**Description:** Questions about locations, origins, residences, or places.
**Key Indicators:** "where does X live", "where from", "hometown", location words
**Examples:**
- "Where did Caroline move from?" → Asks about origin location
- "Where does Melanie live?" → Asks about current residence
- "What city is Sam from?" → Asks about hometown
- "Where is their office located?" → Asks about place

### 7. REASONING_HYPOTHETICAL
**Description:** Hypothetical or conditional questions requiring speculation.
**Key Indicators:** "would", "could", "if", "likely", "probably", conditional phrasing
**Examples:**
- "Would Caroline pursue writing as a career option?" → Hypothetical career question
- "Would Caroline likely have Dr. Seuss books?" → Inference about ownership
- "If Sam had more time, what would he do?" → Hypothetical scenario
- "Could Melanie be interested in art?" → Inference about interests

### 8. REASONING_INFERENCE
**Description:** Questions requiring logical inference or deduction from facts.
**Key Indicators:** "why", "how come", "what caused", inference from multiple facts
**Note:** Different from HYPOTHETICAL - INFERENCE asks about actual reasons/causes, HYPOTHETICAL asks about possibilities.
**Examples:**
- "Why did Sam decide to change jobs?" → Requires inferring motivation
- "What caused Caroline to move?" → Requires inferring cause
- "How did Melanie become interested in painting?" → Requires tracing development
- "What led to their friendship?" → Requires connecting multiple facts

### 9. TIME_CALCULATION
**Description:** Questions requiring time calculations, durations, or time comparisons.
**Key Indicators:** "how long", "how many years/months/days", "since", "ago", duration words
**Examples:**
- "How long has Caroline had her friends?" → Requires duration calculation
- "How long ago was Caroline's birthday?" → Requires time difference calculation
- "How many years has Sam been working there?" → Requires duration from events
- "Since when has Melanie been married?" → Requires finding start point

### 10. COUNTING
**Description:** Questions asking for a specific COUNT or NUMBER of items/events.
**Key Indicators:** "how many", "how often", "count", "number of", "times"
**Note:** Different from EVENT_AGGREGATION - COUNTING asks for a NUMBER, AGGREGATION asks for a LIST.
**Examples:**
- "How many trips did Sam take in 2023?" → Asks for a number
- "How many times has Caroline visited her parents?" → Asks for frequency count
- "How often does Melanie exercise?" → Asks for frequency
- "How many books has Sam read this year?" → Asks for a count (not a list)

### 11. GENERAL
**Description:** Questions that don't fit clearly into any category above.
**Key Indicators:** Ambiguous phrasing, multi-aspect questions, unclear intent
**Examples:**
- Complex multi-part questions
- Questions with unusual phrasing
- Questions combining multiple categories

## Questions to Classify
{questions_list}

## Response Format
Return a JSON object with EXACTLY {num_questions} classifications:

{{
  "classifications": [
    {{
      "id": 1,
      "category": "EVENT_TEMPORAL",
      "confidence": 0.95,
      "reasoning": "Question asks 'when' - requesting specific timing of an event"
    }},
    {{
      "id": 2,
      "category": "COUNTING",
      "confidence": 0.90,
      "reasoning": "Question asks 'how many' - requesting a specific count"
    }},
    ...
  ]
}}

## Important Distinctions
1. COUNTING vs EVENT_AGGREGATION:
   - "How many trips?" → COUNTING (asks for number)
   - "What trips did X take?" → EVENT_AGGREGATION (asks for list)

2. REASONING_HYPOTHETICAL vs REASONING_INFERENCE:
   - "Would X do Y?" → HYPOTHETICAL (speculation about possibility)
   - "Why did X do Y?" → INFERENCE (reasoning about actual events)

CRITICAL: You MUST return EXACTLY {num_questions} classifications. Do not skip any question.
Respond with the JSON object only, no other text."""


# =============================================================================
# HELPER CONSTANTS
# =============================================================================

# Category to internal question_type mapping - ALL 11 categories
CATEGORY_TO_QUESTION_TYPE = {
    "EVENT_TEMPORAL": "event_temporal",
    "EVENT_ACTIVITY": "event_activity",
    "EVENT_AGGREGATION": "event_aggregation",
    "ATTRIBUTE_IDENTITY": "attribute_identity",
    "ATTRIBUTE_PREFERENCE": "attribute_preference",
    "ATTRIBUTE_LOCATION": "attribute_location",
    "REASONING_HYPOTHETICAL": "reasoning_hypothetical",
    "REASONING_INFERENCE": "reasoning_inference",
    "TIME_CALCULATION": "time_calculation",
    "COUNTING": "counting",
    "GENERAL": "general",
}

# Valid categories for validation
VALID_CATEGORIES = list(CATEGORY_TO_QUESTION_TYPE.keys())


def get_batch_classification_prompt(questions_list: str, num_questions: int) -> str:
    """Format the batch classification prompt with questions.

    Args:
        questions_list: Formatted string of questions (e.g., "1. Question 1\n2. Question 2")
        num_questions: Total number of questions in the batch

    Returns:
        Formatted prompt string ready for LLM
    """
    return BATCH_CLASSIFICATION_PROMPT.format(
        questions_list=questions_list,
        num_questions=num_questions,
    )
