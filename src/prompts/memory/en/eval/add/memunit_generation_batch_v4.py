# V4 Batch Prompts for MemUnit Generation
#
# This module contains batch versions of:
# - Session segmentation (fine-grained MemUnit splitting)
# - Narrative generation (episodic memory)
# - EventLog generation (atomic facts)
#
# Key principle: Batch prompts MUST preserve all rules from single-item prompts,
# only adding batch-specific output format changes.

# =============================================================================
# Session Segmentation Prompt (per session, outputs multiple units)
# =============================================================================
SESSION_SEGMENTATION_PROMPT = """
You are a conversation segmentation expert. Your task is to split a session into fine-grained semantic units.

**Session Info:**
- Session ID: {session_id}
- Session Time: {session_time}

**Conversation:**
{conversation}

**Core Concept: Semantic Boundaries**

A semantic boundary exists where one topic/exchange ENDS and another BEGINS. Your task is to identify these boundaries and create units accordingly.

**CRITICAL: A single message often contains MULTIPLE semantic parts!**
- Common pattern: "Answer to previous question. **New question to the other person?**"
- Common pattern: "**Response/comment.** By the way, new topic..."
- You MUST split such messages at the semantic boundary!

---

**Segmentation Principles:**

1. **One Complete Exchange = One Unit**:
   - Each unit should contain ONE semantic exchange: Question → Answer, or Statement → Response
   - The boundary is where one exchange ends and a new one begins

2. **Identifying Semantic Boundaries Within Messages (VERY COMMON)**:
   Messages often contain multiple semantic parts. Look for these patterns:
   - **Answer + New Question**: "I'm doing great! **What about you?**" → split!
   - **Comment + New Topic**: "That's cool! **By the way, did you hear about...**" → split!
   - **Response + Follow-up Question**: "Wow, that's amazing! **How did you do it?**" → split!

   When you see these patterns, the message belongs to TWO units:
   - First part goes with the PREVIOUS exchange
   - Second part starts a NEW exchange

3. **Message Count Per Unit is Flexible**:
   - **2 messages**: Simple Q → A (most common after proper splitting)
   - **3+ messages**: Multiple consecutive messages from same speaker, OR statement + response that continues the same topic

4. **Key Rules**:
   - A response/answer MUST stay with what it responds to
   - A question MUST stay with its answer
   - When splitting, each part retains the original dia_id
   - **Pure reactions merge with previous**: Simple reactions like "Wow!", "That's cool!", "Awesome!" should be included in the SAME unit as what they're responding to, not as standalone units. Only create a new unit when there's new semantic content (new topic, new question, etc.)

5. **CRITICAL - Preserve Complete Message Text**:
   - When a message appears in a unit, include its COMPLETE text - NEVER split into separate sentences
   - Example: If a message is "I'm so happy. This is great. Thanks!" - include ALL of it, not just one sentence
   - Sentence-level splitting is WRONG. Only split at semantic boundaries (topic change, new question, etc.)

**Output Format:**
Return a JSON object with the following structure:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "SpeakerName", "content": "actual text content"}},
                {{"dia_id": "D1:2", "speaker": "SpeakerName", "content": "actual text content"}}
            ],
            "summary": "Brief description of what this unit is about"
        }}
    ]
}}

**Important Rules:**
- `content` MUST contain the COMPLETE original text - no splitting or truncation
- `dia_id` must reference the original message ID exactly
- `speaker` must match the EXACT speaker name from the input
- `summary` should be concise (under 20 words)
- Maintain chronological order within each unit
- Every message must be included (no content loss)

---

**Example 1 - Answer + New Question pattern (VERY COMMON):**
This is the most common pattern in real conversations - someone answers and then asks back.

Input:
[D1:1] Tom: Hey Sarah! How have you been?
[D1:2] Sarah: Hey Tom! I'm good, been busy with work. What's new with you?
[D1:3] Tom: I just got back from a trip to Japan!
[D1:4] Sarah: Wow, that's amazing! How was it?

**Semantic boundary analysis:**
- D1:2 contains TWO parts:
  - "Hey Tom! I'm good, been busy with work." → ANSWERS D1:1's question
  - "What's new with you?" → NEW QUESTION answered by D1:3
- D1:4 contains TWO parts:
  - "Wow, that's amazing!" → RESPONDS to D1:3 (merge with previous)
  - "How was it?" → NEW QUESTION (would be answered next)

Output:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "Tom", "content": "Hey Sarah! How have you been?"}},
                {{"dia_id": "D1:2", "speaker": "Sarah", "content": "Hey Tom! I'm good, been busy with work."}}
            ],
            "summary": "Tom asks how Sarah has been, Sarah says she's good but busy"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:2", "speaker": "Sarah", "content": "What's new with you?"}},
                {{"dia_id": "D1:3", "speaker": "Tom", "content": "I just got back from a trip to Japan!"}},
                {{"dia_id": "D1:4", "speaker": "Sarah", "content": "Wow, that's amazing!"}}
            ],
            "summary": "Sarah asks what's new, Tom shares Japan trip, Sarah reacts"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:4", "speaker": "Sarah", "content": "How was it?"}}
            ],
            "summary": "Sarah asks about the Japan trip details"
        }}
    ]
}}

---

**Example 2 - Response + Follow-up Question pattern:**

Input:
[D1:1] Alice: I joined a support group yesterday, it was really powerful.
[D1:2] Bob: That's great, Alice! What happened that made it so powerful?
[D1:3] Alice: The stories people shared were so inspiring and relatable.
[D1:4] Bob: Sounds wonderful! Are you planning to go again?

**Semantic boundary analysis:**
- D1:2: "That's great, Alice!" (response) + "What happened..." (new question) → SPLIT
- D1:4: "Sounds wonderful!" (response, merge with previous) + "Are you planning..." (new question) → SPLIT

Output:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "Alice", "content": "I joined a support group yesterday, it was really powerful."}},
                {{"dia_id": "D1:2", "speaker": "Bob", "content": "That's great, Alice!"}}
            ],
            "summary": "Alice shares support group experience, Bob responds positively"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:2", "speaker": "Bob", "content": "What happened that made it so powerful?"}},
                {{"dia_id": "D1:3", "speaker": "Alice", "content": "The stories people shared were so inspiring and relatable."}},
                {{"dia_id": "D1:4", "speaker": "Bob", "content": "Sounds wonderful!"}}
            ],
            "summary": "Bob asks what made it powerful, Alice explains, Bob reacts"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:4", "speaker": "Bob", "content": "Are you planning to go again?"}}
            ],
            "summary": "Bob asks if Alice will attend again"
        }}
    ]
}}

---

**Example 3 - Simple Q→A without splitting:**
When messages contain only ONE semantic part, no splitting needed.

Input:
[D1:1] Tom: How long was the hiking trail?
[D1:2] Sarah: About 8 miles round trip.
[D1:3] Tom: Did you take any photos?
[D1:4] Sarah: Yes, I got some beautiful sunset shots!

Output:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "Tom", "content": "How long was the hiking trail?"}},
                {{"dia_id": "D1:2", "speaker": "Sarah", "content": "About 8 miles round trip."}}
            ],
            "summary": "Tom asks trail length, Sarah answers 8 miles"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:3", "speaker": "Tom", "content": "Did you take any photos?"}},
                {{"dia_id": "D1:4", "speaker": "Sarah", "content": "Yes, I got some beautiful sunset shots!"}}
            ],
            "summary": "Tom asks about photos, Sarah mentions sunset shots"
        }}
    ]
}}

---

**Example 4 - Multiple consecutive messages + reaction merged:**

Input:
[D1:1] Tom: What did you do this weekend?
[D1:2] Sarah: I went to a concert on Saturday!
[D1:3] Sarah: Then on Sunday I visited my parents.
[D1:4] Tom: Sounds like a busy weekend!

**Analysis:** D1:4 is a pure reaction/comment with no new question or topic, so it merges with the previous exchange.

Output:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "Tom", "content": "What did you do this weekend?"}},
                {{"dia_id": "D1:2", "speaker": "Sarah", "content": "I went to a concert on Saturday!"}},
                {{"dia_id": "D1:3", "speaker": "Sarah", "content": "Then on Sunday I visited my parents."}},
                {{"dia_id": "D1:4", "speaker": "Tom", "content": "Sounds like a busy weekend!"}}
            ],
            "summary": "Tom asks about weekend, Sarah describes activities, Tom reacts"
        }}
    ]
}}

Return only the JSON object, no additional text:
"""


# =============================================================================
# Batch Narrative Generation Prompt
# Based on: episode_mem_prompts.py -> EPISODE_GENERATION_PROMPT
# =============================================================================
BATCH_NARRATIVE_PROMPT = """
You are an episodic memory generation expert. Convert each conversation unit into an episodic memory.

**Conversation Units to Process:**
{units_json}

**Instructions:**
For each unit, generate a structured episodic memory with:
1. `title`: A concise, descriptive title (10-20 words) that includes key topics/activities
2. `content`: A detailed third-person narrative following ALL requirements below

**IMPORTANT TIME HANDLING:**
- Use each unit's timestamp as the exact time when that conversation/episode began
- When the conversation mentions relative times (e.g., "yesterday", "last week"), preserve both the original relative expression AND calculate the absolute date
- Format time references as: "original relative time (absolute date)" - e.g., "last week (May 7, 2023)"
- This dual format supports both absolute and relative time-based questions

**Requirements for `content`:**
1. The content must include all important information from the conversation.
2. Convert the dialogue format into a narrative description.
3. Maintain chronological order and causal relationships.
4. Use third-person unless explicitly first-person.
5. Include specific details that aid keyword search, especially concrete activities, places, and objects.
6. For time references, use the dual format: "relative time (absolute date)" to support different question types.
7. When describing decisions or actions, naturally include the reasoning or motivation behind them.
8. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
9. CRITICAL DETAIL PRESERVATION:
   - Person Names: Always include full names of people mentioned (e.g., "went to yoga with Amy's colleague, Rob" not just "went to yoga with a colleague")
   - Special Nouns & Entities: Preserve all proper nouns, brand names, place names, organization names exactly as mentioned
   - Item Names: Include specific product names, book titles, movie names, restaurant names, etc.
   - Quantities & Numbers: Record exact numbers, amounts, prices, percentages, dates, times (e.g., "ordered 3 pizzas" not "ordered pizzas")
   - Specific Activities: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")
   - Time Points: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")
10. FREQUENCY INFORMATION:
   - Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")
   - Note patterns of behavior (e.g., "mentioned calling mom three times during the conversation")
   - Include habitual actions (e.g., "usually has coffee at 8 AM before work")
   - Document repetition counts (e.g., "asked about the project status twice")

**Output Format (BATCH):**
Return a JSON object with results for ALL units:
{{
    "results": [
        {{
            "unit_id": 0,
            "title": "Descriptive title here",
            "content": "Detailed narrative content here..."
        }},
        {{
            "unit_id": 1,
            "title": "...",
            "content": "..."
        }}
    ]
}}

**Critical Rules:**
- Process ALL units in order (unit_id 0, 1, 2, ...)
- Do NOT skip any unit
- Each narrative should be self-contained and retrievable

Return only the JSON object:
"""


# =============================================================================
# Batch EventLog Generation Prompt
# Based on: event_log_prompts.py -> EVENT_LOG_PROMPT
# =============================================================================
BATCH_EVENTLOG_PROMPT = """
You are an expert episodic memory extraction analyst and information architect.
Your task is to analyze each narrative and produce an event log optimized for factual retrieval.

**Narratives to Process:**
{narratives_json}

---

### EXTRACTION RULES

#### 1. Atomicity
* Each entry in `"atomic_fact"` must express **exactly one coherent unit of meaning** — an action, emotion, reason, plan, decision, or statement.
* If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.
* Each atomic_fact must be **independent and retrievable on its own**.

#### 2. Time & Date Handling
* Do **not** add or prepend timestamps such as "On March 10, 2024" to each fact.
* The `"time"` field already represents the episode's start time.
* However, when the original text mentions **explicit or relative time expressions**, you must:
  - **Preserve** explicit dates verbatim (e.g., "October 6, 2023").
  - **Resolve** relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to the timestamp, and **append the resolved absolute date in parentheses**.
    Example: "Gina said she launched the campaign yesterday (March 9, 2024)."
  - If the exact resolution is uncertain, use a normalized vague phrase (e.g., "in early 2023", "during the summer of 2021") instead of guessing.

#### 3. Content Preservation
* Preserve **all semantically meaningful information**, including:
  - emotions, attitudes, reasons, intentions, consequences, conditions, and comparisons
* Resolve pronouns into explicit names or entities when unambiguous.
* Keep original wording as much as possible — only fix grammar for clarity.

#### 4. Expression Format
* Write each atomic_fact as a **single, complete English sentence** in **third-person** form.
  - e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024)."
* Do **not** simplify, paraphrase, or merge logically distinct ideas.

#### 5. Retrieval Clarity
* Each atomic_fact must be concise, factual, and self-contained.
* Avoid small talk or filler unless it conveys meaningful emotional or informational content.
* Ensure entities and actions are explicit and unambiguous.

---

### QUALITY CHECKS
Before returning the final output, verify that:
1. Every meaningful fact, intention, and emotion is included.
2. Each `atomic_fact` contains exactly one idea.
3. All relevant time references are preserved or normalized.
4. Wording remains faithful to the source.
5. The output JSON is valid and follows the exact schema.

---

### Output Format (BATCH)
Return a JSON object with results for ALL narratives:
{{
    "results": [
        {{
            "unit_id": 0,
            "time": "May 08, 2023(Monday) at 01:56 PM",
            "atomic_fact": [
                "Tom asked Sarah about her weekend plans.",
                "Sarah mentioned she went hiking at Mount Wilson.",
                "..."
            ]
        }},
        {{
            "unit_id": 1,
            "time": "...",
            "atomic_fact": ["...", "..."]
        }}
    ]
}}

**Critical Rules:**
- Process ALL narratives in order (unit_id must match input)
- Do NOT skip any unit
- Ensure valid JSON output

Return only the JSON object:
"""
