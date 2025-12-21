"""Temporal Questions C-RAG Evaluator (Time-Focused Three-Way Classification)

This prompt specializes in temporal reasoning for:
- event_temporal
- time_calculation

These questions require careful verification of dates, times, and temporal relationships.
"""

RETRIEVAL_EVALUATOR_TEMPORAL_PROMPT = """You are a temporal reasoning expert evaluating whether retrieved documents contain sufficient time-related information to answer a question.

**User Query**:
{query}

**Retrieved Documents**:
{retrieved_docs}

**Instructions**:
Follow these steps for temporal evaluation:

1. **Identify Time Constraints**: Extract ALL time-related requirements from the query:
   - Specific dates (e.g., "May 2023", "August 18")
   - Relative times (e.g., "last summer", "before December")
   - Time calculations (e.g., "how many months between X and Y")
   - Temporal relationships (e.g., "when did X happen", "which came first")

2. **Verify Time Information**: Check if documents provide:
   - Explicit dates/timestamps matching the query
   - Sufficient temporal context to calculate or infer the answer
   - Confirmation that events align with the specified timeframe

3. **Three-Way Classification**:

   **CORRECT**: Documents explicitly answer ALL temporal aspects of the query.
   - Dates/times are clearly stated and match the query requirements
   - Temporal calculations can be performed with confidence
   - No ambiguity about when events occurred
   - Example: Query asks "When did X happen?" and document says "X happened on May 5, 2023"

   **AMBIGUOUS**: Documents contain partial time information but have gaps or require inference.
   - Some dates are mentioned but not all required dates
   - Relative time references that need calculation (e.g., "next Saturday" without base date)
   - Temporal context exists but specific date/time is not explicit
   - Multiple possible time interpretations
   - Example: Query asks "When did X happen?" and document says "X happened sometime in May"

   **INCORRECT**: Documents don't address the temporal question or provide wrong time information.
   - Wrong time period (e.g., query asks about 2023, documents discuss 2024)
   - Different event with similar timing
   - No temporal information provided
   - Example: Query asks "When did X happen?" and documents only say "X happened recently"

4. **Conservative Temporal Verification**:
   ⚠️ **Be strict with time**: If ANY temporal detail is unclear, vague, or requires inference, mark as AMBIGUOUS.
   - "sometime in May" vs "May 15" → AMBIGUOUS (lacks specificity)
   - "last week" without reference date → AMBIGUOUS (needs calculation)
   - Year ambiguity (2023 vs 2024) → AMBIGUOUS or INCORRECT

5. **Time Calculation Support**:
   - For calculation questions, verify that ALL required dates are present
   - Check that time units are clear (days, weeks, months)
   - Ensure no ambiguity in how to perform the calculation

**Output Format** (strict JSON):
{{
  "evaluation_type": "correct" or "ambiguous" or "incorrect",
  "confidence": 0.0 to 1.0,
  "reasoning": "Explain the temporal verification result (1-2 sentences)",
  "missing_information": ["specific missing time details"],
  "incorrect_aspects": ["why temporal info is wrong"],
  "correct_direction": "what temporal info is needed"
}}

**Examples**:

Example 1 (CORRECT - Explicit Date):
Query: "When did Sam start painting?"
Documents:
  Document 1: "On May 15, 2023, Sam mentioned he started painting as a new hobby."
Output:
{{
  "evaluation_type": "correct",
  "confidence": 0.95,
  "reasoning": "Document explicitly states Sam started painting on May 15, 2023 with full temporal clarity.",
  "missing_information": [],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 2 (AMBIGUOUS - Vague Timeframe):
Query: "When did Sam start painting?"
Documents:
  Document 1: "Sam mentioned he started painting sometime in May 2023."
  Document 2: "Sam talked about his new painting hobby in spring."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.80,
  "reasoning": "Documents indicate May 2023 but lack the specific date when Sam started painting.",
  "missing_information": ["exact date in May 2023", "specific day Sam started painting"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 3 (AMBIGUOUS - Relative Time Without Base):
Query: "When did Evan and Sam decide to paint together?"
Documents:
  Document 1: "On September 11, 2023 (Monday), Evan suggested they paint together next Saturday."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.85,
  "reasoning": "Document mentions 'next Saturday' after Monday Sept 11, which should be Sept 16, but the specific date is not explicitly stated and requires calculation.",
  "missing_information": ["explicit date of the Saturday they decided on", "confirmation of the calculated date"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 4 (INCORRECT - Wrong Year):
Query: "When did Evan have a drunken night in 2023?"
Documents:
  Document 1: "On January 9, 2024, Evan had a drunken night with friends."
Output:
{{
  "evaluation_type": "incorrect",
  "confidence": 0.90,
  "reasoning": "Document references January 9, 2024, but the query asks about 2023 - wrong year.",
  "missing_information": [],
  "incorrect_aspects": ["year mismatch: document shows 2024, query asks for 2023"],
  "correct_direction": "events from 2023, specifically drunken night occurrences in 2023"
}}

Example 5 (AMBIGUOUS - Calculation Missing Data):
Query: "How many months between Sam's first and second doctor appointments?"
Documents:
  Document 1: "Sam had his first doctor appointment in June 2023."
  Document 2: "Sam mentioned going to the doctor again recently."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.80,
  "reasoning": "First appointment date (June 2023) is known, but second appointment date is vague ('recently') - cannot calculate months.",
  "missing_information": ["specific date of second doctor appointment", "month and year of second visit"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Now evaluate:
"""
