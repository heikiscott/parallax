"""Evaluation Workflow Nodes for LangGraph orchestration.

Wraps eval/core/stages/ functions as LangGraph nodes for YAML-driven configuration.
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add

# Add src to path if needed (for imports)
_project_root = Path(__file__).parent.parent.resolve()
_src_path = str(_project_root / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from eval.core.stages.add_stage import run_add_stage
from eval.core.stages.cluster_stage import run_cluster_stage
from eval.core.stages.search_stage import run_search_stage
from eval.core.stages.answer_stage import run_answer_stage
from eval.core.stages.evaluate_stage import run_evaluate_stage
from src.orchestration.nodes import register_node
from eval.utils.token_stats import TokenStatsCollector
from eval.utils.saver import ResultSaver
import json


# ============================================================================
# State Schema
# ============================================================================

class EvalState(TypedDict, total=False):
    """Evaluation workflow state."""
    dataset: Any
    conversations: List[Any]
    qa_pairs: List[Any]
    conv_id: Optional[int]

    index: Optional[Dict[str, Any]]
    search_results: Optional[List[Any]]
    answer_results: Optional[List[Any]]
    eval_results: Optional[Any]

    metadata: Dict[str, Any]
    completed_stages: Annotated[List[str], add]
    filter_categories: List[int]


# ============================================================================
# Nodes
# ============================================================================

@register_node("eval_add_stage")
async def eval_add_stage_node(state, context) -> Dict[str, Any]:
    """Stage 1: Add - Build indexes."""
    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("add")

    try:
        result = await run_add_stage(
            adapter=context.adapter,
            dataset=state.get("dataset"),
            output_dir=Path(context.output_dir),
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
            console=context.console,
            completed_stages=set(state.get("completed_stages", [])),
        )

        return {
            "index": result.get("index"),
            "completed_stages": ["add"],
            "metadata": {**state.get("metadata", {}), "add_completed": True}
        }
    finally:
        # Clear stage after completion
        TokenStatsCollector.set_current_stage(None)


@register_node("eval_cluster_stage")
async def eval_cluster_stage_node(state: EvalState, context) -> Dict[str, Any]:
    """Stage 1.5: Cluster - Group event clustering."""
    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("cluster")

    try:
        result = await run_cluster_stage(
            adapter=context.adapter,
            conversations=state.get("conversations"),
            output_dir=Path(context.output_dir),
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
            console=context.console,
            completed_stages=set(state.get("completed_stages", [])),
        )

        return {
            "completed_stages": ["cluster"],
            "metadata": {**state.get("metadata", {}), "cluster_completed": True}
        }
    finally:
        # Clear stage after completion
        TokenStatsCollector.set_current_stage(None)


@register_node("eval_search_stage")
async def eval_search_stage_node(state: EvalState, context) -> Dict[str, Any]:
    """Stage 2: Search - Retrieve memories."""
    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("search")

    try:
        search_results = await run_search_stage(
            adapter=context.adapter,
            qa_pairs=state.get("qa_pairs"),
            index=state.get("index"),
            conversations=state.get("conversations"),
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
        )

        # Save search results to file
        saver = ResultSaver(context.output_dir)
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
        context.logger.info(f"Saved search results to search_results.json")

        return {
            "search_results": search_results,
            "completed_stages": ["search"],
            "metadata": {**state.get("metadata", {}), "search_completed": True}
        }
    finally:
        # Clear stage after completion
        TokenStatsCollector.set_current_stage(None)


@register_node("eval_answer_stage")
async def eval_answer_stage_node(state: EvalState, context) -> Dict[str, Any]:
    """Stage 3: Answer - Generate answers."""
    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("answer")

    try:
        answer_results = await run_answer_stage(
            adapter=context.adapter,
            qa_pairs=state.get("qa_pairs"),
            search_results=state.get("search_results"),
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
        )

        # Save answer results to file
        saver = ResultSaver(context.output_dir)
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
        context.logger.info(f"Saved answer results to answer_results.json")

        return {
            "answer_results": answer_results,
            "completed_stages": ["answer"],
            "metadata": {**state.get("metadata", {}), "answer_completed": True}
        }
    finally:
        # Clear stage after completion
        TokenStatsCollector.set_current_stage(None)


@register_node("eval_evaluate_stage")
async def eval_evaluate_stage_node(state: EvalState, context) -> Dict[str, Any]:
    """Stage 4: Evaluate - Assess answer quality."""
    eval_results = await run_evaluate_stage(
        evaluator=context.evaluator,
        answer_results=state.get("answer_results"),
        checkpoint_manager=context.checkpoint_manager,
        logger=context.logger,
    )

    # Save evaluation results to file
    saver = ResultSaver(context.output_dir)
    eval_data = {
        "total_questions": eval_results.total_questions,
        "correct": eval_results.correct,
        "accuracy": eval_results.accuracy,
        "detailed_results": eval_results.detailed_results,
        "metadata": eval_results.metadata,
    }
    saver.save_json(eval_data, "eval_results.json")
    context.logger.info(f"Saved evaluation results to eval_results.json")

    # Generate comprehensive report
    metadata = state.get("metadata", {})
    answer_results = state.get("answer_results", [])

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # ========== 1. 核心结果（最醒目位置）==========
    report_lines.append("╔" + "=" * 78 + "╗")
    report_lines.append("║" + " " * 28 + "📈 FINAL RESULTS" + " " * 35 + "║")
    report_lines.append("╠" + "=" * 78 + "╣")
    report_lines.append(f"║  Accuracy: {eval_results.accuracy:.2%}" + " " * (78 - len(f"  Accuracy: {eval_results.accuracy:.2%}")) + "║")
    report_lines.append(f"║  Correct: {eval_results.correct}/{eval_results.total_questions}" + " " * (78 - len(f"  Correct: {eval_results.correct}/{eval_results.total_questions}")) + "║")
    incorrect = eval_results.total_questions - eval_results.correct
    report_lines.append(f"║  Incorrect: {incorrect}" + " " * (78 - len(f"  Incorrect: {incorrect}")) + "║")
    report_lines.append("╚" + "=" * 78 + "╝")
    report_lines.append("")

    # ========== 2. 分类详细结果 ==========
    report_lines.append("─" * 80)
    report_lines.append("📋 RESULTS BY CATEGORY")
    report_lines.append("─" * 80)

    # 统计每个类别的准确率
    from collections import defaultdict
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for detail in eval_results.detailed_results:
        cat = detail.get("category", "Unknown")
        if cat is None:
            cat = "Unknown"
        category_stats[cat]["total"] += 1
        if detail.get("is_correct", False):
            category_stats[cat]["correct"] += 1

    # 按类别排序输出
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        report_lines.append(f"  Category {cat}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    report_lines.append("")

    # ========== 3. 运行时间统计 ==========
    # Time is automatically tracked by WorkflowBuilder's node_wrapper
    report_lines.append("─" * 80)
    report_lines.append("⏱️  TIME STATISTICS")
    report_lines.append("─" * 80)

    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    total_time = 0
    # WorkflowBuilder injects timing as "{node_name}_time" in metadata
    if "add_time" in metadata:
        report_lines.append(f"  Add Stage:      {format_time(metadata['add_time'])}")
        total_time += metadata["add_time"]
    if "cluster_time" in metadata:
        report_lines.append(f"  Cluster Stage:  {format_time(metadata['cluster_time'])}")
        total_time += metadata["cluster_time"]
    if "search_time" in metadata:
        report_lines.append(f"  Search Stage:   {format_time(metadata['search_time'])}")
        total_time += metadata["search_time"]
    if "answer_time" in metadata:
        report_lines.append(f"  Answer Stage:   {format_time(metadata['answer_time'])}")
        total_time += metadata["answer_time"]
    if "evaluate_time" in metadata:
        report_lines.append(f"  Evaluate Stage: {format_time(metadata['evaluate_time'])}")
        total_time += metadata["evaluate_time"]

    if total_time > 0:
        report_lines.append(f"  " + "─" * 40)
        report_lines.append(f"  Total Time:     {format_time(total_time)}")
    report_lines.append("")

    # ========== 4. Token 使用统计 ==========
    if context.token_stats_collector:
        report_lines.append("─" * 80)
        report_lines.append("💰 TOKEN USAGE")
        report_lines.append("─" * 80)

        all_summaries = context.token_stats_collector.get_all_summaries()
        total_tokens_all = 0

        stage_order = ["add", "cluster", "search", "answer"]
        for stage in stage_order:
            if stage in all_summaries:
                summary = all_summaries[stage]
                if summary["total_calls"] > 0:
                    report_lines.append(
                        f"  {stage.capitalize():8s}: {summary['total_tokens']:,} tokens "
                        f"({summary['total_calls']} calls, avg {summary['avg_total_tokens']:.0f}/call)"
                    )
                    total_tokens_all += summary["total_tokens"]

        report_lines.append(f"  " + "─" * 40)
        report_lines.append(f"  Total:    {total_tokens_all:,} tokens")
        report_lines.append("")

    # ========== 5. 错误案例（可选：显示前几个错误） ==========
    incorrect_cases = [d for d in eval_results.detailed_results if not d.get("is_correct", False)]
    if incorrect_cases:
        report_lines.append("─" * 80)
        report_lines.append(f"❌ INCORRECT CASES ({len(incorrect_cases)} total, showing first 3)")
        report_lines.append("─" * 80)

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
    context.logger.info(f"Saved comprehensive report to report.txt")

    return {
        "eval_results": eval_results,
        "completed_stages": ["evaluate"],
        "metadata": {**state.get("metadata", {}), "evaluate_completed": True}
    }
