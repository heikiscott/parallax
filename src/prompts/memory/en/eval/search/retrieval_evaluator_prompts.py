"""C-RAG Style Three-Way Retrieval Evaluator Prompt

This prompt classifies retrieval quality into three categories:
- CORRECT: Documents are relevant and sufficient
- AMBIGUOUS: Documents are relevant but incomplete
- INCORRECT: Documents are irrelevant or wrong direction
"""

RETRIEVAL_EVALUATOR_PROMPT = """You are an expert retrieval evaluator following the C-RAG (Corrective RAG) framework. Your task is to classify the quality of retrieved documents into three categories to determine the next retrieval action.

**User Query**:
{query}

**Retrieved Documents**:
{retrieved_docs}

**Instructions**:
Follow these steps to classify the retrieval quality:

1. **Deconstruct the Query**: Break down the query into core information needs (key entities, facts, constraints like time/location/person).

2. **Document Relevance Analysis**:
   - Check if documents mention the key entities from the query
   - Verify if documents are on-topic vs off-topic
   - Assess if documents contain factual information vs speculation

3. **Three-Way Classification**:

   **CORRECT**: Retrieved documents are relevant AND contain sufficient information to answer the query.
   - All key entities are mentioned with correct context
   - Explicit facts directly answer the query
   - No significant information gaps

   **AMBIGUOUS**: Retrieved documents are relevant BUT information is incomplete or partial.
   - Key entities are mentioned but context is missing
   - Some aspects of the query are answered, others are not
   - Information is vague, indirect, or requires inference

   **INCORRECT**: Retrieved documents are irrelevant or on the wrong topic entirely.
   - Wrong entities (e.g., different person with similar name)
   - Completely different topic or event
   - Retrieval went in the wrong direction

4. **Evidence Collection**:
   - For CORRECT: Simply confirm sufficiency
   - For AMBIGUOUS: List specific missing information needed
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

Example 1 (CORRECT):
Query: "What is Alice's favorite hobby?"
Documents:
  Document 1: "Alice mentioned she loves painting. She spends weekends at art galleries."
  Document 2: "Alice's work involves creative projects."
Output:
{{
  "evaluation_type": "correct",
  "confidence": 0.95,
  "reasoning": "Document 1 explicitly states Alice's hobby is painting with supporting context about art galleries.",
  "missing_information": [],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 2 (AMBIGUOUS):
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

Example 3 (INCORRECT - Wrong Entity):
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

Example 4 (INCORRECT - Wrong Topic):
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
