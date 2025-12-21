"""Single Question Classification Prompts.

This module provides prompts for single-question classification.
Currently used by LLMQuestionClassifier for runtime classification.

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

This prompt should stay aligned with batch_classification_prompts.py
to ensure consistent classification results between single and batch modes.
"""

# =============================================================================
# SINGLE QUESTION CLASSIFICATION PROMPT
# =============================================================================
# This prompt classifies one question at a time.
# It ONLY determines the question TYPE - the retrieval strategy is decided
# by each pipeline independently based on its architecture.
# =============================================================================

SINGLE_CLASSIFICATION_PROMPT = """You are a question classifier for a memory retrieval system.
Your task is to classify the question to determine its semantic type.

## Question
{question}

## Question Categories (11 total)

### 1. EVENT_TEMPORAL
**Description:** Questions asking WHEN a specific event happened or will happen.
**Key Indicators:** "when", "what time", "what date", specific time references
**Examples:**
- "When did Caroline go to the LGBTQ support group?"
- "When did Melanie paint a sunrise?"
- "When is Caroline going to the conference?"

### 2. EVENT_ACTIVITY
**Description:** Questions about activities, hobbies, or actions someone does/did.
**Key Indicators:** "what does X do", "what activities", "hobbies", action verbs
**Examples:**
- "What activities does Melanie partake in?"
- "What does Melanie do to destress?"
- "What hobbies does Sam have?"

### 3. EVENT_AGGREGATION
**Description:** Questions requiring collection/aggregation of multiple events or items over time.
**Key Indicators:** "what books", "where has X been", "all the places", plural nouns referring to events
**Note:** Different from COUNTING - EVENT_AGGREGATION asks for a LIST of things, COUNTING asks for a NUMBER.
**Examples:**
- "What books has Melanie read?" (asks for list)
- "Where has Melanie camped?" (asks for list)
- "What restaurants have they visited?" (asks for list)

### 4. ATTRIBUTE_IDENTITY
**Description:** Questions about fixed personal attributes, identity, or status.
**Key Indicators:** "what is X's", "who is", identity/status words, profession, relationships
**Examples:**
- "What is Caroline's identity?"
- "What is Caroline's relationship status?"
- "What is Sam's job?"

### 5. ATTRIBUTE_PREFERENCE
**Description:** Questions about preferences, likes, dislikes, or favorites.
**Key Indicators:** "favorite", "prefer", "like", "enjoy", "love", "hate"
**Examples:**
- "What do Melanie's kids like?"
- "What is Caroline's favorite book?"
- "What kind of music does Sam prefer?"

### 6. ATTRIBUTE_LOCATION
**Description:** Questions about locations, origins, residences, or places.
**Key Indicators:** "where does X live", "where from", "hometown", location words
**Examples:**
- "Where did Caroline move from?"
- "Where does Melanie live?"
- "What city is Sam from?"

### 7. REASONING_HYPOTHETICAL
**Description:** Hypothetical or conditional questions requiring speculation.
**Key Indicators:** "would", "could", "if", "likely", "probably", conditional phrasing
**Examples:**
- "Would Caroline pursue writing as a career option?"
- "Would Caroline likely have Dr. Seuss books?"
- "If Sam had more time, what would he do?"

### 8. REASONING_INFERENCE
**Description:** Questions requiring logical inference or deduction from facts.
**Key Indicators:** "why", "how come", "what caused", inference from multiple facts
**Note:** Different from HYPOTHETICAL - INFERENCE asks about actual reasons/causes, HYPOTHETICAL asks about possibilities.
**Examples:**
- "Why did Sam decide to change jobs?"
- "What caused Caroline to move?"
- "How did Melanie become interested in painting?"

### 9. TIME_CALCULATION
**Description:** Questions requiring time calculations, durations, or time comparisons.
**Key Indicators:** "how long", "how many years/months/days", "since", "ago", duration words
**Examples:**
- "How long has Caroline had her friends?"
- "How long ago was Caroline's birthday?"
- "How many years has Sam been working there?"

### 10. COUNTING
**Description:** Questions asking for a specific COUNT or NUMBER of items/events.
**Key Indicators:** "how many", "how often", "count", "number of", "times"
**Note:** Different from EVENT_AGGREGATION - COUNTING asks for a NUMBER, AGGREGATION asks for a LIST.
**Examples:**
- "How many trips did Sam take in 2023?" (asks for number)
- "How many times has Caroline visited her parents?" (asks for count)
- "How often does Melanie exercise?" (asks for frequency)

### 11. GENERAL
**Description:** Questions that don't fit clearly into any category above.
**Key Indicators:** Ambiguous phrasing, multi-aspect questions, unclear intent

## Response Format (JSON)
{{"category": "EVENT_TEMPORAL", "confidence": 0.9, "reasoning": "Brief explanation"}}

## Important Distinctions
1. COUNTING vs EVENT_AGGREGATION:
   - "How many trips?" → COUNTING (asks for number)
   - "What trips did X take?" → EVENT_AGGREGATION (asks for list)

2. REASONING_HYPOTHETICAL vs REASONING_INFERENCE:
   - "Would X do Y?" → HYPOTHETICAL (speculation about possibility)
   - "Why did X do Y?" → INFERENCE (reasoning about actual events)

Respond with JSON only, no other text."""


# =============================================================================
# HELPER CONSTANTS (shared with batch_classification_prompts.py)
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

# Valid categories
VALID_CATEGORIES = list(CATEGORY_TO_QUESTION_TYPE.keys())


def get_single_classification_prompt(question: str) -> str:
    """Format the single question classification prompt.

    Args:
        question: The question to classify

    Returns:
        Formatted prompt string ready for LLM
    """
    return SINGLE_CLASSIFICATION_PROMPT.format(question=question)
