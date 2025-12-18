"""Corrective Query Generation Prompt for C-RAG

When retrieval is evaluated as INCORRECT, this prompt generates
corrective queries to redirect retrieval in the right direction.
"""

CORRECTIVE_QUERY_PROMPT = """You are an expert at query reformulation for corrective retrieval.

**Context**: The initial retrieval was INCORRECT - it retrieved irrelevant or off-topic documents. Your task is to generate new queries that will retrieve the CORRECT information.

**Original Query**:
{original_query}

**Retrieved Documents** (INCORRECT - off-topic):
{retrieved_docs}

**Why Retrieval Failed**:
{incorrect_aspects}

**Correct Direction**:
{correct_direction}

**Instructions**:

1. **Avoid Current Results**: Reformulate queries to retrieve DIFFERENT documents, not the same ones.

2. **Target Correct Direction**: Use keywords and phrases from "Correct Direction" to guide retrieval.

3. **Entity Disambiguation**: If wrong entity was retrieved (e.g., wrong person), explicitly include the correct entity name with disambiguating context.

4. **Add Constraints**: Include specific constraints (time, person names, event types) to exclude wrong results.

5. **Use Alternative Phrasing**: Try different vocabulary to break away from failed retrieval patterns.

6. **Generate 2-3 Diverse Queries**: Each query should approach the correct direction from a different angle.

**Output Format** (strict JSON, no additional text before or after):
{{
  "queries": [
    "first corrective query focusing on correct entity/topic",
    "second corrective query with alternative phrasing",
    "third corrective query targeting specific detail"
  ],
  "reasoning": "Brief strategy explanation (1-2 sentences)"
}}

**Examples**:

Example 1 (Wrong Entity):
Original Query: "What did Alice say about the project deadline?"
Incorrect Aspects: ["retrieved Bob's statement instead of Alice's"]
Correct Direction: "Alice's direct comments about the project deadline"
Output:
{{
  "queries": [
    "Alice's comments opinion about project deadline",
    "What Alice said mentioned about the deadline",
    "Alice project deadline discussion her view"
  ],
  "reasoning": "Emphasize 'Alice' as the subject and use variations of 'said/comments/mentioned' to find her direct statements."
}}

Example 2 (Wrong Topic):
Original Query: "What restaurant did they go to for dinner last week?"
Incorrect Aspects: ["retrieved coffee shop and lunch info instead of dinner"]
Correct Direction: "dinner event from last week, restaurant name"
Output:
{{
  "queries": [
    "dinner restaurant last week where they ate",
    "last week dinner outing restaurant name location",
    "evening meal restaurant they visited recently"
  ],
  "reasoning": "Focus on 'dinner/evening meal' and 'restaurant' keywords, exclude coffee/lunch context."
}}

Example 3 (Wrong Time Frame):
Original Query: "What happened at the team meeting yesterday?"
Incorrect Aspects: ["retrieved old meeting notes from last month"]
Correct Direction: "yesterday's team meeting events and discussions"
Output:
{{
  "queries": [
    "team meeting yesterday what happened discussed",
    "yesterday's meeting topics decisions outcomes",
    "recent team meeting yesterday agenda results"
  ],
  "reasoning": "Anchor on 'yesterday' and 'recent' to avoid retrieving older meeting records."
}}

Now generate corrective queries:
"""
