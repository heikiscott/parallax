"""Classify Stage - Batch Question Classification.

This module implements the Classify stage that performs global batch classification
of all questions before the Search stage. Classification results are attached to
each QAPair's metadata for downstream stages to use.
"""

import asyncio
import json
from logging import Logger
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from eval.core.data_models import QAPair
from eval.utils.checkpoint import CheckpointManager
from src.retrieval.classification.question_classifier import (
    QuestionClassifier,
    ClassificationResult,
    QuestionType,
)
from src.prompts.memory.en.eval.classify.batch_classification_prompts import (
    get_batch_classification_prompt,
    CATEGORY_TO_QUESTION_TYPE,
    VALID_CATEGORIES,
)


async def run_classify_stage(
    qa_pairs: List[QAPair],
    output_dir: Path,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    llm_provider: Optional[Any] = None,
    use_llm_classification: bool = True,
    batch_size: int = 15,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch classify all questions from all conversations.

    Args:
        qa_pairs: List of all QA pairs to classify
        output_dir: Output directory for saving results
        checkpoint_manager: Checkpoint manager for fine-grained resume
        logger: Logger instance
        llm_provider: LLM provider for LLM-based classification
        use_llm_classification: Whether to use LLM or rule-based classifier
        batch_size: Number of questions per batch for LLM classification

    Returns:
        Dict mapping question_id to classification data
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🏷️  Classify Stage")
    logger.info(f"{'='*60}")

    output_dir = Path(output_dir)
    classification_results: Dict[str, Dict[str, Any]] = {}

    # Load checkpoint if exists
    if checkpoint_manager:
        checkpoint_data = checkpoint_manager.load_classification_progress()
        classification_results = checkpoint_data.get("results", {})
        if classification_results:
            logger.info(f"Loaded {len(classification_results)} classifications from checkpoint")

    # Determine pending questions
    pending_qa_pairs = [
        qa for qa in qa_pairs
        if qa.question_id not in classification_results
    ]

    total_count = len(qa_pairs)
    processed_count = len(classification_results)

    logger.info(f"Total questions: {total_count}")
    if processed_count > 0:
        logger.info(f"Already processed: {processed_count} (from checkpoint)")
        logger.info(f"Remaining: {len(pending_qa_pairs)}")

    if not pending_qa_pairs:
        logger.info("All questions already classified!")
        # Attach classification to QA pairs
        attach_classification_to_qa_pairs(qa_pairs, classification_results)
        return classification_results

    # Choose classification method
    if use_llm_classification and llm_provider:
        logger.info("Using LLM-based batch classifier (GPT-4o-mini)")
        classification_results = await _batch_llm_classify(
            pending_qa_pairs=pending_qa_pairs,
            existing_results=classification_results,
            llm_provider=llm_provider,
            batch_size=batch_size,
            checkpoint_manager=checkpoint_manager,
            output_dir=output_dir,
            logger=logger,
            total_count=total_count,
        )
    else:
        logger.info("Using rule-based classifier")
        classification_results = _rule_based_classify(
            pending_qa_pairs=pending_qa_pairs,
            existing_results=classification_results,
            checkpoint_manager=checkpoint_manager,
            output_dir=output_dir,
            logger=logger,
            total_count=total_count,
        )

    # Save final results
    final_results_path = output_dir / "classification_results.json"
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(classification_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved classification results to {final_results_path.name}")

    # Delete checkpoint file on success
    if checkpoint_manager:
        checkpoint_manager.delete_classification_checkpoint()

    # Print statistics
    _print_classification_stats(classification_results, logger)

    # Attach classification to QA pairs for downstream stages
    attach_classification_to_qa_pairs(qa_pairs, classification_results)

    return classification_results


async def _batch_llm_classify(
    pending_qa_pairs: List[QAPair],
    existing_results: Dict[str, Dict[str, Any]],
    llm_provider: Any,
    batch_size: int,
    checkpoint_manager: Optional[CheckpointManager],
    output_dir: Path,
    logger: Logger,
    total_count: int,
) -> Dict[str, Dict[str, Any]]:
    """Batch classify questions using LLM."""
    classification_results = existing_results.copy()
    processed_count = len(classification_results)

    pbar = tqdm(
        total=total_count,
        initial=processed_count,
        desc="🏷️  Classify Progress",
        unit="qa"
    )

    # Process in batches
    for batch_start in range(0, len(pending_qa_pairs), batch_size):
        batch = pending_qa_pairs[batch_start:batch_start + batch_size]

        # Build batch prompt using the prompt module
        questions_list = "\n".join([
            f"{i+1}. \"{qa.question}\""
            for i, qa in enumerate(batch)
        ])
        prompt = get_batch_classification_prompt(questions_list, len(batch))

        try:
            # Call LLM
            # Note: Using json_object forces OpenAI to return an object {}, not array []
            # So we ask LLM to wrap results in {"classifications": [...]}
            response = await llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=4000,  # Increased for batch response (15 questions need ~200 tokens each)
                response_format={"type": "json_object"},
            )

            # Debug: log raw response for troubleshooting
            logger.debug(f"LLM batch response (first 500 chars): {response[:500] if response else 'None'}")

            # Parse batch response
            batch_results = _parse_batch_response(response, batch, logger)

            # Store results
            for qa, result in zip(batch, batch_results):
                classification_results[qa.question_id] = result
                pbar.update(1)

        except Exception as e:
            logger.warning(f"Batch LLM classification failed: {e}, falling back to rule-based for this batch")
            # Fallback to rule-based for this batch
            classifier = QuestionClassifier()
            for qa in batch:
                result = classifier.classify(qa.question)
                result_dict = result.to_dict()
                # Remove strategy from rule-based result (V4 doesn't use it)
                result_dict.pop("strategy", None)
                classification_results[qa.question_id] = result_dict
                pbar.update(1)

        # Save checkpoint after each batch
        if checkpoint_manager:
            checkpoint_manager.save_classification_progress(
                classification_results,
                len(classification_results),
                total_count
            )

    pbar.close()
    return classification_results


def _rule_based_classify(
    pending_qa_pairs: List[QAPair],
    existing_results: Dict[str, Dict[str, Any]],
    checkpoint_manager: Optional[CheckpointManager],
    output_dir: Path,
    logger: Logger,
    total_count: int,
) -> Dict[str, Dict[str, Any]]:
    """Classify questions using rule-based classifier."""
    classification_results = existing_results.copy()
    processed_count = len(classification_results)

    classifier = QuestionClassifier()
    checkpoint_interval = 50

    pbar = tqdm(
        total=total_count,
        initial=processed_count,
        desc="🏷️  Classify Progress",
        unit="qa"
    )

    for i, qa in enumerate(pending_qa_pairs):
        result = classifier.classify(qa.question)
        result_dict = result.to_dict()
        # Remove strategy from rule-based result (V4 doesn't use it)
        result_dict.pop("strategy", None)
        classification_results[qa.question_id] = result_dict
        pbar.update(1)

        # Periodic checkpoint
        if checkpoint_manager and (i + 1) % checkpoint_interval == 0:
            checkpoint_manager.save_classification_progress(
                classification_results,
                len(classification_results),
                total_count
            )

    pbar.close()
    return classification_results


def _parse_batch_response(
    response: str,
    batch: List[QAPair],
    logger: Logger
) -> List[Dict[str, Any]]:
    """Parse batch LLM response into list of classification dicts.

    Uses CATEGORY_TO_QUESTION_TYPE from the prompts module
    to ensure consistency with the prompt definitions.
    """
    try:
        # Parse JSON response
        data = json.loads(response.strip())

        # Handle both array and object with array inside
        if isinstance(data, dict):
            # Try common keys (prioritize "classifications" as per our prompt)
            for key in ["classifications", "results", "questions", "items"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # If it's a single result wrapped in object (shouldn't happen with our prompt)
                logger.warning(f"LLM response is dict without expected array key. Keys found: {list(data.keys())}")
                data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        # Log warning if LLM returned fewer items than expected
        if len(data) < len(batch):
            logger.warning(f"LLM returned {len(data)} classifications for {len(batch)} questions")

        results = []
        for i, qa in enumerate(batch):
            if i < len(data):
                item = data[i]
                category = item.get("category", "GENERAL").upper()

                # Use imported mappings from prompts module for consistency
                question_type = CATEGORY_TO_QUESTION_TYPE.get(category, "general")

                # V4 classification only contains question_type
                # No strategy field - V4 uses cross-attention for all types
                results.append({
                    "question_type": question_type,
                    "confidence": float(item.get("confidence", 0.5)),
                    "reasoning": item.get("reasoning", "LLM batch classification"),
                    "detected_patterns": ["llm_batch_classification"],
                    "entities": [],
                })
            else:
                # Fallback for missing items - should not happen with improved prompt
                logger.warning(f"Missing classification for question {i+1}, using rule-based fallback")
                classifier = QuestionClassifier()
                result = classifier.classify(qa.question)
                result_dict = result.to_dict()
                # Remove strategy from rule-based result (V4 doesn't use it)
                result_dict.pop("strategy", None)
                results.append(result_dict)

        return results

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse batch LLM response: {e}")
        # Fallback to rule-based for entire batch
        classifier = QuestionClassifier()
        results = []
        for qa in batch:
            result_dict = classifier.classify(qa.question).to_dict()
            # Remove strategy from rule-based result (V4 doesn't use it)
            result_dict.pop("strategy", None)
            results.append(result_dict)
        return results


def _print_classification_stats(
    classification_results: Dict[str, Dict[str, Any]],
    logger: Logger
) -> None:
    """Print classification statistics."""
    type_counts: Dict[str, int] = {}

    for result in classification_results.values():
        qtype = result.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Classification Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total questions: {len(classification_results)}")
    logger.info(f"\nBy Question Type:")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {qtype}: {count}")
    logger.info(f"{'='*60}\n")


def attach_classification_to_qa_pairs(
    qa_pairs: List[QAPair],
    classification_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Attach classification results to QA pair metadata.

    This mutates qa_pairs in place for efficiency.
    After this, downstream stages can access classification via:
        qa.metadata["classification"]

    Args:
        qa_pairs: List of QA pairs
        classification_results: Dict of question_id -> classification
    """
    for qa in qa_pairs:
        if qa.question_id in classification_results:
            qa.metadata["classification"] = classification_results[qa.question_id]


def load_classification_results(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load classification results from file."""
    results_path = output_dir / "classification_results.json"
    if not results_path.exists():
        return {}

    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)
