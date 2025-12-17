"""Evaluate Stage Node - Assess answer quality."""

from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

from eval.core.stages.evaluate_stage import run_evaluate_stage
from eval.core.nodes.common import (
    is_stage_completed,
    load_eval_results,
    save_eval_results,
)
from src.orchestration.nodes import register_node


@register_node("eval_evaluate_stage")
async def eval_evaluate_stage_node(state, context) -> Dict[str, Any]:
    """Evaluate: Assess answer quality."""
    # Check if stage already completed
    if is_stage_completed(state, "evaluate"):
        context.logger.info("Evaluate stage already completed, skipping...")

        # Load existing eval results from file
        eval_results = load_eval_results(Path(context.output_dir))

        return {
            "eval_results": eval_results,
            "completed_stages": ["evaluate"],
            "metadata": {**state.get("metadata", {}), "evaluate_completed": True}
        }

    eval_results = await run_evaluate_stage(
        evaluator=context.evaluator,
        answer_results=state.get("answer_results"),
        checkpoint_manager=context.checkpoint_manager,
        logger=context.logger,
    )

    # Save evaluation results to file
    save_eval_results(Path(context.output_dir), eval_results, context.logger)

    # Generate comprehensive report
    _generate_report(context, state, eval_results)

    return {
        "eval_results": eval_results,
        "completed_stages": ["evaluate"],
        "metadata": {**state.get("metadata", {}), "evaluate_completed": True}
    }


def _generate_report(context, state, eval_results) -> None:
    """Generate comprehensive evaluation report."""
    metadata = state.get("metadata", {})

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # ========== 1. Core Results ==========
    report_lines.append("=" * 78)
    report_lines.append("  FINAL RESULTS")
    report_lines.append("=" * 78)
    report_lines.append(f"  Accuracy: {eval_results.accuracy:.2%}")
    report_lines.append(f"  Correct: {eval_results.correct}/{eval_results.total_questions}")
    incorrect = eval_results.total_questions - eval_results.correct
    report_lines.append(f"  Incorrect: {incorrect}")
    report_lines.append("")

    # ========== 2. Results by Category ==========
    report_lines.append("-" * 80)
    report_lines.append("RESULTS BY CATEGORY")
    report_lines.append("-" * 80)

    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    # Flatten detailed_results
    detailed_list = []
    if isinstance(eval_results.detailed_results, dict):
        for conv_results in eval_results.detailed_results.values():
            if isinstance(conv_results, list):
                detailed_list.extend(conv_results)
    elif isinstance(eval_results.detailed_results, list):
        detailed_list = eval_results.detailed_results

    for detail in detailed_list:
        if isinstance(detail, dict):
            cat = detail.get("category", "Unknown")
            if cat is None:
                cat = "Unknown"
            category_stats[cat]["total"] += 1

            is_correct = _check_is_correct(detail)
            if is_correct:
                category_stats[cat]["correct"] += 1

    if category_stats:
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            report_lines.append(f"  Category {cat}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    report_lines.append("")

    # ========== 3. Time Statistics ==========
    report_lines.append("-" * 80)
    report_lines.append("TIME STATISTICS")
    report_lines.append("-" * 80)

    total_time = 0
    for stage in ["add", "cluster", "search", "answer", "evaluate"]:
        time_key = f"{stage}_time"
        if time_key in metadata:
            report_lines.append(f"  {stage.capitalize()} Stage: {_format_time(metadata[time_key])}")
            total_time += metadata[time_key]

    if total_time > 0:
        report_lines.append("  " + "-" * 40)
        report_lines.append(f"  Total Time: {_format_time(total_time)}")
    report_lines.append("")

    # ========== 4. Token Usage ==========
    if context.token_stats_collector:
        report_lines.append("-" * 80)
        report_lines.append("TOKEN USAGE")
        report_lines.append("-" * 80)

        all_summaries = context.token_stats_collector.get_all_summaries()
        total_tokens_all = 0

        for stage in ["add", "cluster", "search", "answer"]:
            if stage in all_summaries:
                summary = all_summaries[stage]
                if summary["total_calls"] > 0:
                    report_lines.append(
                        f"  {stage.capitalize():8s}: {summary['total_tokens']:,} tokens "
                        f"({summary['total_calls']} calls, avg {summary['avg_total_tokens']:.0f}/call)"
                    )
                    total_tokens_all += summary["total_tokens"]

        report_lines.append("  " + "-" * 40)
        report_lines.append(f"  Total: {total_tokens_all:,} tokens")
        report_lines.append("")

    # ========== 5. Incorrect Cases ==========
    incorrect_cases = [d for d in detailed_list if isinstance(d, dict) and not _check_is_correct(d)]

    if incorrect_cases:
        report_lines.append("-" * 80)
        report_lines.append(f"INCORRECT CASES ({len(incorrect_cases)} total, showing first 3)")
        report_lines.append("-" * 80)

        for i, case in enumerate(incorrect_cases[:3], 1):
            report_lines.append(f"  [{i}] Question ID: {case.get('question_id', 'N/A')}")
            question = case.get("question", "N/A")
            if len(question) > 60:
                question = question[:57] + "..."
            report_lines.append(f"      Question: {question}")
            report_lines.append(f"      Category: {case.get('category', 'N/A')}")
            report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("End of Report")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    report_path = Path(context.output_dir) / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    context.logger.info("Saved comprehensive report to report.txt")


def _check_is_correct(detail: dict) -> bool:
    """Check if an answer is correct."""
    is_correct = detail.get("is_correct", False)
    if not is_correct and "llm_judgments" in detail:
        judgments = detail["llm_judgments"]
        if isinstance(judgments, dict):
            true_count = sum(1 for v in judgments.values() if v)
            is_correct = true_count > len(judgments) / 2
    return is_correct


def _format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
