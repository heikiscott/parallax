"""Simple Questions C-RAG Evaluator (Three-Way Classification)

This prompt uses the standard C-RAG three-way classification for simple questions:
- attribute_identity, attribute_preference, attribute_location
- event_activity
- general

Copied from Conv8-14 original version for independent tuning.
"""

RETRIEVAL_EVALUATOR_SIMPLE_PROMPT = """You are a meticulous and detail-oriented retrieval evaluator following the C-RAG (Corrective RAG) framework. Your task is to critically assess the quality of retrieved documents and classify them into three categories to determine the next retrieval action.

**User Query**:
{query}

**Retrieved Documents**:
{retrieved_docs}

**Instructions**:
Follow these steps to classify the retrieval quality:

1. **Deconstruct the Query**: Break down the query into core information needs (key entities, facts, constraints like time/location/person). Identify ALL aspects that need to be answered.

2. **Document Relevance Analysis**:
   - Check if documents mention the key entities from the query
   - Verify if documents are on-topic vs off-topic
   - Assess if documents contain factual information vs speculation

3. **Three-Way Classification**:

   **CORRECT**: Retrieved documents are relevant AND contain sufficient information to answer the query COMPLETELY.
   - ALL key aspects of the query are directly and explicitly answered
   - Explicit facts are provided with full context (no vague references)
   - No missing information, no need for inference or assumptions
   - The answer can be constructed from the documents without guessing
   - ⚠️ **Be strict**: If ANY aspect requires inference, assumptions, or is unclear, classify as AMBIGUOUS instead

   **AMBIGUOUS**: Retrieved documents are relevant BUT information is incomplete, partial, or requires inference.
   - Key entities are mentioned but lack sufficient detail or context
   - Some aspects of the query are answered, but others are missing or unclear
   - Information is vague, indirect, or requires making inferences
   - Documents hint at the answer but don't explicitly state it
   - Time/location/person constraints are not fully confirmed
   - ⚠️ **Default to this category when uncertain**: If you're not 100% confident that ALL information is explicitly present, choose AMBIGUOUS

   **INCORRECT**: Retrieved documents are irrelevant or on the wrong topic entirely.
   - Wrong entities (e.g., different person with similar name)
   - Completely different topic or event
   - Retrieval went in the wrong direction
   - Documents are off-topic with no useful information

4. **Conservative Evaluation Principle**:
   ❗ **IMPORTANT**: When in doubt between CORRECT and AMBIGUOUS, always choose AMBIGUOUS. It's better to retrieve additional information than to provide an incomplete answer. Err on the side of caution.

5. **Evidence Collection**:
   - For CORRECT: Confirm that ALL query aspects are explicitly answered with full context
   - For AMBIGUOUS: List specific missing or unclear information needed
   - For INCORRECT: Identify why documents are off-topic and what the correct retrieval direction should be

**Output Format** (strict JSON, no additional text before or after):
{{
  "evaluation_type": "correct" or "ambiguous" or "incorrect",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation for your classification (1-2 sentences)",
  "missing_information": ["list", "of", "missing", "info"],
  "incorrect_aspects": ["why", "retrieval", "is", "wrong"],
  "correct_direction": "what should be retrieved instead"
}}

**Field Usage**:
- `missing_information`: Only populate for "ambiguous" (what info is needed)
- `incorrect_aspects`: Only populate for "incorrect" (why results are wrong)
- `correct_direction`: Only populate for "incorrect" (guidance for re-retrieval)
- Use empty arrays [] or empty string "" for unused fields

**Examples**:

Example 1 (CORRECT - Truly Complete):
Query: "What is Alice's favorite hobby?"
Documents:
  Document 1: "On March 15, Alice explicitly told me her favorite hobby is painting. She said 'I love painting more than anything else.'"
  Document 2: "Alice spends every weekend at art galleries and painting workshops."
Output:
{{
  "evaluation_type": "correct",
  "confidence": 0.95,
  "reasoning": "Document 1 explicitly and directly states Alice's favorite hobby is painting with a direct quote, fully answering the query.",
  "missing_information": [],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 2 (AMBIGUOUS - Previously Would Be CORRECT):
Query: "What is Alice's favorite hobby?"
Documents:
  Document 1: "Alice mentioned she loves painting. She spends weekends at art galleries."
  Document 2: "Alice's work involves creative projects."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.75,
  "reasoning": "While Alice is mentioned with painting, it's unclear if painting is her FAVORITE hobby or just one she enjoys. The query asks for 'favorite' which requires explicit confirmation.",
  "missing_information": ["explicit confirmation that painting is her favorite hobby, not just something she likes"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 3 (AMBIGUOUS - Missing Details):
Query: "Where did Bob go on vacation last year?"
Documents:
  Document 1: "Bob talked about his work project deadline around June."
  Document 2: "Bob likes to travel and explore new places. He recently came back from a trip."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.85,
  "reasoning": "Documents mention Bob and travel but lack specific destination and timeframe confirmation for 'last year'.",
  "missing_information": ["specific vacation destination", "confirmation it was last year", "travel dates"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 4 (INCORRECT - Wrong Entity):
Query: "What did Alice say about the project deadline?"
Documents:
  Document 1: "Bob mentioned the project deadline is next Friday."
  Document 2: "The team discussed the upcoming deadline in the meeting."
Output:
{{
  "evaluation_type": "incorrect",
  "confidence": 0.90,
  "reasoning": "Documents discuss Bob's statement and team discussion, but do not contain Alice's specific comments about the deadline.",
  "missing_information": [],
  "incorrect_aspects": ["retrieved Bob's statement instead of Alice's", "no direct quote or reference from Alice"],
  "correct_direction": "Alice's direct comments, quotes, or opinions about the project deadline"
}}

Example 5 (INCORRECT - Wrong Topic):
Query: "What restaurant did they go to for dinner last week?"
Documents:
  Document 1: "They discussed their favorite coffee shops in the city."
  Document 2: "The lunch meeting was held at a conference room."
Output:
{{
  "evaluation_type": "incorrect",
  "confidence": 0.92,
  "reasoning": "Documents discuss coffee shops and lunch meetings, not dinner or restaurants from last week.",
  "missing_information": [],
  "incorrect_aspects": ["wrong meal type (coffee/lunch instead of dinner)", "wrong time frame", "no restaurant mentioned"],
  "correct_direction": "dinner event from last week, specific restaurant name or location"
}}

Now evaluate:
"""
