#!/usr/bin/env python3
"""
Trim LoCoMo dataset to create smaller subsets (q20, q30, q50).

For each subset:
1. Take the first N questions from conv0
2. Include all sessions needed to answer those questions (based on evidence)
3. Include all sessions up to and including the maximum referenced session

Usage:
    python eval/scripts/trim_locomo.py
"""

import json
from pathlib import Path


def get_max_session_for_questions(qa_pairs: list, num_questions: int) -> int:
    """
    Find the maximum session number referenced by the first N questions.

    Evidence format: "D1:3" means session_1, message 3
    """
    max_session = 0
    for qa in qa_pairs[:num_questions]:
        for evidence in qa.get("evidence", []):
            if ":" in evidence:
                # Parse "D1:3" -> session 1
                session_num = int(evidence.split(":")[0][1:])  # D1 -> 1
                max_session = max(max_session, session_num)
    return max_session


def trim_conversation(conversation: dict, max_session: int) -> dict:
    """
    Trim conversation to include only sessions up to max_session.
    """
    trimmed = {
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
    }

    for i in range(1, max_session + 1):
        session_key = f"session_{i}"
        datetime_key = f"session_{i}_date_time"

        if session_key in conversation:
            trimmed[session_key] = conversation[session_key]
        if datetime_key in conversation:
            trimmed[datetime_key] = conversation[datetime_key]

    return trimmed


def create_trimmed_dataset(
    full_data: list,
    num_questions: int,
    conv_index: int = 0
) -> list:
    """
    Create a trimmed dataset with first N questions from specified conversation.

    Args:
        full_data: Full locomo10.json data
        num_questions: Number of questions to include
        conv_index: Conversation index (default 0)

    Returns:
        Trimmed dataset as a list with one conversation
    """
    conv = full_data[conv_index]
    qa_pairs = conv["qa"]
    conversation = conv["conversation"]

    # Find max session needed
    max_session = get_max_session_for_questions(qa_pairs, num_questions)

    # Trim conversation
    trimmed_conv = trim_conversation(conversation, max_session)

    # Take first N questions
    trimmed_qa = qa_pairs[:num_questions]

    # Build result
    result = [{
        "conversation": trimmed_conv,
        "qa": trimmed_qa,
        "event_summary": conv.get("event_summary", {}),
        "observation": conv.get("observation", {}),
        "session_summary": conv.get("session_summary", {}),
        "sample_id": conv.get("sample_id", f"conv-{conv_index}")
    }]

    return result, max_session


def main():
    # Paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / "data" / "locomo"

    # Load full dataset
    full_data_path = data_dir / "locomo10.json"
    with open(full_data_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # Create trimmed datasets
    configs = [
        (10, "locomo-q10"),
        (20, "locomo-q20"),
        (30, "locomo-q30"),
        (50, "locomo-q50"),
    ]

    print("=" * 60)
    print("Trimming LoCoMo dataset from conv0")
    print("=" * 60)

    conv0 = full_data[0]
    total_sessions = sum(
        1 for k in conv0["conversation"].keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    )
    total_qa = len(conv0["qa"])
    print(f"\nConv0 has {total_sessions} sessions and {total_qa} QA pairs")
    print()

    for num_questions, name in configs:
        trimmed_data, max_session = create_trimmed_dataset(
            full_data, num_questions, conv_index=0
        )

        # Count messages in trimmed data
        trimmed_conv = trimmed_data[0]["conversation"]
        num_sessions = sum(
            1 for k in trimmed_conv.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        )
        num_messages = sum(
            len(trimmed_conv[k])
            for k in trimmed_conv.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        )

        # Save
        output_path = data_dir / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trimmed_data, f, indent=2, ensure_ascii=False)

        print(f"{name}:")
        print(f"  Questions: {num_questions}")
        print(f"  Sessions:  {num_sessions} (up to session_{max_session})")
        print(f"  Messages:  {num_messages}")
        print(f"  Saved to:  {output_path}")
        print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
