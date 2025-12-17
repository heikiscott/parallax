"""Common utilities for evaluation workflow nodes."""

from pathlib import Path
from typing import List, Dict, Any, Optional

from eval.core.data_models import SearchResult, AnswerResult, EvaluationResult
from eval.utils.saver import ResultSaver


def is_stage_completed(state: Dict[str, Any], stage_name: str) -> bool:
    """Check if a stage is already completed."""
    completed_stages = set(state.get("completed_stages", []))
    return stage_name in completed_stages


def load_search_results(output_dir: Path) -> List[SearchResult]:
    """Load search results from checkpoint file."""
    saver = ResultSaver(output_dir)
    search_data = saver.load_json("search_results.json")

    return [
        SearchResult(
            query=item["query"],
            conversation_id=item["conversation_id"],
            results=item["results"],
            retrieval_metadata=item.get("retrieval_metadata", {}),
        )
        for item in search_data
    ]


def save_search_results(output_dir: Path, search_results: List[SearchResult], logger) -> None:
    """Save search results to file."""
    saver = ResultSaver(output_dir)
    search_data = [
        {
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": sr.results,
            "retrieval_metadata": sr.retrieval_metadata,
        }
        for sr in search_results
    ]
    saver.save_json(search_data, "search_results.json")
    logger.info("Saved search results to search_results.json")


def load_answer_results(output_dir: Path) -> List[AnswerResult]:
    """Load answer results from checkpoint file."""
    saver = ResultSaver(output_dir)
    answer_data = saver.load_json("answer_results.json")

    return [
        AnswerResult(
            question_id=item["question_id"],
            question=item["question"],
            answer=item["answer"],
            golden_answer=item["golden_answer"],
            category=item.get("category"),
            conversation_id=item.get("conversation_id", ""),
            formatted_context=item.get("formatted_context", ""),
            metadata=item.get("metadata", {}),
        )
        for item in answer_data
    ]


def save_answer_results(output_dir: Path, answer_results: List[AnswerResult], logger) -> None:
    """Save answer results to file."""
    saver = ResultSaver(output_dir)
    answer_data = [
        {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "formatted_context": ar.formatted_context,
            "metadata": ar.metadata,
        }
        for ar in answer_results
    ]
    saver.save_json(answer_data, "answer_results.json")
    logger.info("Saved answer results to answer_results.json")


def load_eval_results(output_dir: Path) -> EvaluationResult:
    """Load evaluation results from checkpoint file."""
    saver = ResultSaver(output_dir)
    eval_data = saver.load_json("eval_results.json")

    return EvaluationResult(
        total_questions=eval_data["total_questions"],
        correct=eval_data["correct"],
        accuracy=eval_data["accuracy"],
        detailed_results=eval_data["detailed_results"],
        metadata=eval_data.get("metadata", {}),
    )


def save_eval_results(output_dir: Path, eval_results: EvaluationResult, logger) -> None:
    """Save evaluation results to file."""
    saver = ResultSaver(output_dir)
    eval_data = {
        "total_questions": eval_results.total_questions,
        "correct": eval_results.correct,
        "accuracy": eval_results.accuracy,
        "detailed_results": eval_results.detailed_results,
        "metadata": eval_results.metadata,
    }
    saver.save_json(eval_data, "eval_results.json")
    logger.info("Saved evaluation results to eval_results.json")


def update_checkpoint(context, state: Dict[str, Any], stage_name: str) -> None:
    """Update checkpoint after stage completion."""
    if context.checkpoint_manager:
        current_completed = set(state.get("completed_stages", []))
        current_completed.add(stage_name)
        context.checkpoint_manager.save_checkpoint(current_completed)
        context.logger.info(f"Checkpoint updated: {sorted(list(current_completed))}")
