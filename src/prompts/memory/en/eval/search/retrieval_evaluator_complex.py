"""Complex Questions C-RAG Evaluator (Multi-Step Reasoning Three-Way Classification)

This prompt handles complex questions requiring aggregation, counting, or multi-step reasoning:
- event_aggregation
- counting
- reasoning_hypothetical
- reasoning_inference

These questions require comprehensive context and multiple pieces of information.
"""

RETRIEVAL_EVALUATOR_COMPLEX_PROMPT = """You are a meticulous and detail-oriented retrieval evaluator for complex questions requiring aggregation, counting, or multi-step reasoning. Your task is to assess whether retrieved documents contain ALL necessary information.

**User Query**:
{query}

**Retrieved Documents**:
{retrieved_docs}

**Instructions**:
Follow these steps for complex question evaluation:

1. **Deconstruct the Query**: Break down into ALL sub-questions and information needs:
   - For aggregation: What are ALL the individual items/events to collect?
   - For counting: What criteria define what should be counted? Are there timeframes or constraints?
   - For reasoning: What facts are needed to support the inference or hypothesis?

2. **Completeness Check**: Verify documents provide:
   - ALL instances/items needed (not just some)
   - Complete context for each instance
   - No missing items that should be included
   - Clear criteria to determine inclusion/exclusion

3. **Three-Way Classification**:

   **CORRECT**: Retrieved documents are comprehensive AND contain ALL information to answer completely.
   - ALL required instances/events/facts are explicitly present
   - Sufficient context to understand each instance
   - Clear boundaries (know when the list is complete)
   - No missing information, no need for assumptions
   - Can confidently construct the full answer from documents
   - ⚠️ **Be strict**: If you suspect anything might be missing, choose AMBIGUOUS

   **AMBIGUOUS**: Retrieved documents are relevant BUT may be incomplete or require inference.
   - Some instances are present but might be missing others
   - Uncertain if the list is complete
   - Information is partial or lacks full context
   - Requires making assumptions about completeness
   - Time periods or constraints are not fully verified
   - ⚠️ **Default to this when uncertain**: Better to retrieve more than to miss information

   **INCORRECT**: Retrieved documents are off-topic or addressing wrong questions.
   - Wrong entities, events, or time periods
   - Completely different topic
   - Documents don't relate to the query requirements

4. **Conservative Completeness Principle**:
   ❗ **IMPORTANT**: For aggregation/counting questions, unless you can confidently say "this is definitely ALL of them", choose AMBIGUOUS. Missing even one instance makes the answer wrong.

5. **Evidence for Completeness**:
   - For CORRECT: Explain why you're confident this is the complete set
   - For AMBIGUOUS: List what might be missing or why you're uncertain about completeness
   - For INCORRECT: Identify the mismatch between query and documents

**Output Format** (strict JSON):
{{
  "evaluation_type": "correct" or "ambiguous" or "incorrect",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation focusing on completeness (1-2 sentences)",
  "missing_information": ["specific items or context that might be missing"],
  "incorrect_aspects": ["why retrieval is wrong"],
  "correct_direction": "what should be retrieved instead"
}}

**Examples**:

Example 1 (CORRECT - Confident Completeness):
Query: "List all the cars Evan has owned."
Documents:
  Document 1: "Evan owned two cars in his lifetime: first an old Toyota Prius from 2020-2023, then a new Toyota Prius from 2023 onwards."
  Document 2: "Evan sold his old Prius in May 2023 and bought a new one immediately after."
  Document 3: "Evan confirmed he has only ever owned Toyota Prius models, no other vehicles."
Output:
{{
  "evaluation_type": "correct",
  "confidence": 0.90,
  "reasoning": "Documents explicitly state Evan owned two Prius cars and confirm these are the only vehicles he has owned, providing complete information.",
  "missing_information": [],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 2 (AMBIGUOUS - Uncertain Completeness):
Query: "List all the hobbies Sam has tried in 2023."
Documents:
  Document 1: "Sam started painting in May 2023."
  Document 2: "Sam mentioned trying kayaking in August 2023."
  Document 3: "Sam talked about his interests in photography."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.75,
  "reasoning": "Documents mention painting, kayaking, and photography, but it's unclear if this is the complete list or if Sam tried other hobbies in 2023.",
  "missing_information": ["confirmation this is ALL hobbies tried in 2023", "potential other hobbies not mentioned", "explicit statement of completeness"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 3 (AMBIGUOUS - Counting Uncertainty):
Query: "How many roadtrips did Evan take in May 2023?"
Documents:
  Document 1: "Evan took a roadtrip to the Rocky Mountains in late May 2023."
  Document 2: "Evan mentioned enjoying driving in May."
Output:
{{
  "evaluation_type": "ambiguous",
  "confidence": 0.80,
  "reasoning": "Documents confirm at least one roadtrip to Rocky Mountains, but unclear if there were others in May 2023.",
  "missing_information": ["confirmation of total number of roadtrips in May", "whether Rocky Mountains was the only trip", "any other roadtrips that month"],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 4 (CORRECT - Aggregation with Context):
Query: "What health incidents did Evan face in 2023?"
Documents:
  Document 1: "In 2023, Evan had three health incidents: a twisted knee in August, a knee requiring a cast in October, and a back strain in November."
  Document 2: "Evan confirmed these were all his health issues in 2023 and he had a tough year physically."
  Document 3: "Evan mentioned he stayed healthy otherwise and these were the only medical problems."
Output:
{{
  "evaluation_type": "correct",
  "confidence": 0.95,
  "reasoning": "Documents explicitly list all three health incidents and confirm these were the only medical issues Evan faced in 2023.",
  "missing_information": [],
  "incorrect_aspects": [],
  "correct_direction": ""
}}

Example 5 (INCORRECT - Wrong Scope):
Query: "What personal health incidents does Evan face in 2023?"
Documents:
  Document 1: "Evan's son fell off his bike in December 2023."
  Document 2: "Evan's partner got the flu in March 2023."
Output:
{{
  "evaluation_type": "incorrect",
  "confidence": 0.90,
  "reasoning": "Documents discuss health incidents of Evan's family members (son and partner), not Evan's personal health issues.",
  "missing_information": [],
  "incorrect_aspects": ["retrieved family members' health instead of Evan's personal health", "wrong entity (son and partner vs Evan himself)"],
  "correct_direction": "Evan's own personal health incidents, injuries, or medical issues in 2023"
}}

Now evaluate:
"""
