# Prompts for V4 fine-grained MemUnit splitting

FINE_GRAINED_SEGMENTATION_PROMPT = """
You are a conversation segmentation expert. Your task is to split a conversation into fine-grained semantic units.

**Conversation to segment:**
{conversation}

**Segmentation Principles:**

1. **Granularity**: Each unit should be a semantically complete minimal exchange, typically:
   - One question + its direct answer
   - One statement + its direct response
   - One request + its acknowledgment

2. **Semantic Binding**: Group messages by what they directly address:
   - A response belongs with the content it responds to
   - If a message contains multiple parts (e.g., "Hello Alice, how can I help?"), split it:
     - "Hello Alice" binds to the greeting
     - "How can I help?" binds to the next user request

3. **Message Splitting**: When a single message contains multiple semantic parts:
   - Split at natural boundaries (punctuation, topic shifts)
   - Each part goes to the unit where it semantically belongs
   - Preserve the original dia_id for traceability

4. **Avoid Over-grouping**: Do NOT group multiple Q-A pairs into one unit unless they are inseparable (e.g., clarification within a single exchange)

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
- `content` must contain the ACTUAL text (may be partial if message was split)
- `dia_id` must reference the original message ID
- Same `dia_id` can appear in multiple units if the message was split
- `speaker` must match the EXACT speaker name from the input (e.g., "Caroline", "Melanie")
- `summary` should be concise (under 20 words)
- Maintain chronological order within each unit
- Every part of the conversation must be included (no content loss)

**Example:**
Input conversation:
[D1:1] Alice: Hi, I'm Alice
[D1:2] Bob: Hello Alice, how can I help you today?
[D1:3] Alice: I want to book a flight to Beijing

Output:
{{
    "units": [
        {{
            "messages": [
                {{"dia_id": "D1:1", "speaker": "Alice", "content": "Hi, I'm Alice"}},
                {{"dia_id": "D1:2", "speaker": "Bob", "content": "Hello Alice"}}
            ],
            "summary": "Alice introduces herself, Bob greets"
        }},
        {{
            "messages": [
                {{"dia_id": "D1:2", "speaker": "Bob", "content": "how can I help you today?"}},
                {{"dia_id": "D1:3", "speaker": "Alice", "content": "I want to book a flight to Beijing"}}
            ],
            "summary": "Bob offers help, Alice requests flight booking"
        }}
    ]
}}

Return only the JSON object, no additional text:
"""
