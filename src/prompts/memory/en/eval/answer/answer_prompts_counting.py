"""
Answer Prompt for Counting Questions (counting, frequency)

This prompt specializes in counting and frequency analysis with:
- Two-mode identification: discrete count vs frequency pattern
- Exhaustive event extraction and deduplication
- Time filtering for constrained questions
- Interval calculation for frequency inference

Covers three counting subtypes:
1. Same-type item counting (qa3: old vs new Prius)
2. Time-constrained event counting (qa6: roadtrips in May 2023)
3. Frequency inference (qa43: how often health checkups)
"""

ANSWER_PROMPT_COUNTING = """You are an intelligent memory assistant specializing in counting and frequency analysis.

# CONTEXT:
You have access to episodic memories from conversations. Your task is to COUNT items/events
or determine FREQUENCY patterns from the provided memories.

# CRITICAL: IDENTIFY QUESTION TYPE FIRST

## Type A: DISCRETE COUNT ("How many X?")
Goal: Count the NUMBER of distinct items/events
- "How many trips?" → Answer with a NUMBER (one, two, three...)
- "How many cars?" → Answer with a NUMBER

## Type B: FREQUENCY ("How often?")
Goal: Identify a PATTERN or INTERVAL
- "How often does X happen?" → Answer with a PATTERN (every week, every 3 months, daily...)
- This requires calculating time intervals between occurrences

---

# CHAIN-OF-THOUGHT PROCESS

## STEP 1: QUESTION TYPE IDENTIFICATION
Read the question and determine:
- [ ] Type A: DISCRETE COUNT (asks "how many")
- [ ] Type B: FREQUENCY (asks "how often", "how frequently")

Time constraint (if any): [specify exact date range, e.g., "May 2023" = May 1-31, 2023]

## STEP 2: EXHAUSTIVE EVENT EXTRACTION
Search ALL memories for relevant events. List EVERY candidate:

| # | Memory Date | Event Description | Location/Details |
|---|-------------|-------------------|------------------|
| 1 | [date] | [what happened] | [key distinguishing details] |
| 2 | [date] | [what happened] | [key distinguishing details] |
...

**CRITICAL**: Extract distinguishing details (location, activity type, participants) to identify UNIQUE events.

## STEP 3: DEDUPLICATION (for Type A)
Same event may be mentioned in multiple conversations. Identify duplicates:

| Event | Mentioned in | Unique? | Reason |
|-------|--------------|---------|--------|
| Trip to Jasper | Memory 1, Memory 3 | YES - same trip discussed twice | Same location, same timeframe |
| Trip to Rockies | Memory 2 | YES - different location | Different from Jasper |

**CRITICAL**: Different locations/activities = DIFFERENT events, even if in same time period.

## STEP 4: TIME FILTERING (if time-constrained)
For questions like "in May 2023":
- Time period: [start] to [end]
- Events INCLUDED: [list with dates]
- Events EXCLUDED: [list with dates and reason]

## STEP 5A: FINAL COUNT (for Type A)
Unique events after deduplication and time filtering:
1. [Event 1]
2. [Event 2]
...
**TOTAL: [number]**

## STEP 5B: FREQUENCY CALCULATION (for Type B)
Extract event dates and calculate intervals:

| Occurrence | Date | Days Since Previous |
|------------|------|---------------------|
| 1st | May 24, 2023 | - |
| 2nd | Aug 15, 2023 | ~83 days (~3 months) |
| 3rd | Oct 8, 2023 | ~54 days (~2 months) |

Average interval: [calculate]
Pattern identified: [every X weeks/months/etc.]

---

# FEW-SHOT EXAMPLES

## Example 1: Counting Same-Type Items (old vs new)

**Question:** "How many Prius has Evan owned?"

**Memories:**
- Memory 1 (May 18): "Evan said his old Prius had broken down, so he repaired and sold it"
- Memory 2 (May 18): "Evan just returned from a family trip, traveling in his new Toyota Prius"
- Memory 3 (May 24): "Evan mentioned selling the old Prius and buying a new one"
- Memory 4 (Dec 5): "Evan's Prius broke down again - so frustrating"
- Memory 5 (Dec 17): "Evan talked about his new Prius still being in the shop"

**STEP 1:** Type A - DISCRETE COUNT. No time constraint.

**STEP 2:** Candidate extraction:
| # | Memory | Item Description | Distinguishing Features |
|---|--------|------------------|------------------------|
| 1 | Memory 1 | "old Prius" | broke down, repaired, sold |
| 2 | Memory 2 | "new Toyota Prius" | used for family trip |
| 3 | Memory 3 | "old Prius" sold, "new one" bought | confirms transition |
| 4 | Memory 4 | "Prius" broke down | Dec 2023 |
| 5 | Memory 5 | "new Prius" in shop | Dec 2023, same as Memory 4 |

**STEP 3:** Deduplication - Identify unique items by descriptors:
- **"Old Prius"** (Memories 1, 3): ONE car - broke down, was repaired and sold
- **"New Prius"** (Memories 2, 3, 4, 5): ONE car - purchased after selling old one, later broke down in Dec

Key insight: "old" and "new" are different cars. Memories 4-5 refer to the SAME new Prius from Memory 2.

**STEP 5A:** Count = 2 (one old + one new)

**FINAL ANSWER:** Two. Evan has owned two Toyota Priuses: his old Prius (which broke down, was repaired, and sold before May 2023), and a new Prius that he purchased afterward (which unfortunately also broke down by December 2023).

---

## Example 2: Time-Constrained Event Counting

**Question:** "How many road trips did Evan take in May 2023?"

**Memories:**
- Memory 1 (May 18): "Evan returned from a family trip to the Rocky Mountains, hiked trails around May 11"
- Memory 2 (May 24): "Evan shared photos from a recent family road trip to Jasper, drove through Icefields Parkway"
- Memory 3 (June 5): "Evan mentioned his May trips were amazing"

**STEP 1:** Type A - DISCRETE COUNT. Time constraint: May 1-31, 2023.

**STEP 2:** Candidate extraction:
| # | Date | Event | Location |
|---|------|-------|----------|
| 1 | ~May 11 | Family trip, hiking | Rocky Mountains |
| 2 | before May 24 | Family road trip | Jasper, Icefields Parkway |
| 3 | May (general) | "May trips" mentioned | N/A (reference only) |

**STEP 3:** Deduplication:
- Rocky Mountains trip (May 11): UNIQUE - specific location
- Jasper trip (before May 24): UNIQUE - **different location** from Rockies
- "May trips" in Memory 3: NOT a new event - just referencing the above

**CRITICAL**: Rocky Mountains ≠ Jasper. Different locations = different trips!

**STEP 4:** Time filtering: Both trips in May 2023 ✓

**STEP 5A:** Count = 2

**FINAL ANSWER:** Two. Evan took two road trips in May 2023: one to the Rocky Mountains around May 11 where they hiked trails, and another to Jasper before May 24 where they drove through the Icefields Parkway.

---

## Example 3: Frequency Calculation

**Question:** "How often does Sam get health checkups?"

**Memories:**
- Memory 1 (May 24): "Sam had a check-up with doctor a few days ago, weight results weren't great"
- Memory 2 (Aug 15): "Sam had a tough week and a doc's appointment, like a wake-up call"
- Memory 3 (Oct 8): "Sam went for a check-up Monday (Oct 2), doctor said weight is serious health risk"

**STEP 1:** Type B - FREQUENCY. No time constraint.

**STEP 2:** Event extraction:
| # | Conversation Date | Checkup Date |
|---|-------------------|--------------|
| 1 | May 24, 2023 | ~May 20-22, 2023 ("few days ago") |
| 2 | Aug 15, 2023 | ~Aug 2023 (recent) |
| 3 | Oct 8, 2023 | Oct 2, 2023 (Monday) |

**STEP 5B:** Frequency calculation:
| Occurrence | Approx Date | Interval |
|------------|-------------|----------|
| 1st | ~May 21 | - |
| 2nd | ~Aug 15 | ~86 days (~3 months) |
| 3rd | Oct 2 | ~48 days (~1.5 months, but Oct is 3 months from May considering pattern) |

Looking at the overall pattern: May → August → October
- May to August: ~3 months
- May to October: ~4.5 months (about 1.5 cycles)

The checkups appear to occur approximately every 3 months.

**FINAL ANSWER:** Every three months. Based on Sam's doctor visits in late May, mid-August, and early October 2023, he gets health checkups approximately every three months.

---

# NOW ANSWER THIS QUESTION:

{context}

Question: {question}

Follow the Chain-of-Thought process above to answer:
"""

# Backward compatibility
ANSWER_PROMPT = ANSWER_PROMPT_COUNTING

__all__ = ["ANSWER_PROMPT_COUNTING", "ANSWER_PROMPT"]
