"""Answer Stage Node - Generate answers."""

from pathlib import Path
from typing import Dict, Any

from eval.core.stages.answer_stage import run_answer_stage
from eval.core.nodes.common import (
    is_stage_completed,
    load_answer_results,
    save_answer_results,
)
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_answer_stage")
async def eval_answer_stage_node(state, context) -> Dict[str, Any]:
    """Answer: Generate answers."""
    # Check if stage already completed
    if is_stage_completed(state, "answer"):
        context.logger.info("Answer stage already completed, skipping...")

        # Load existing answer results from file
        answer_results = load_answer_results(Path(context.output_dir))

        return {
            "answer_results": answer_results,
            "completed_stages": ["answer"],
            "metadata": {**state.get("metadata", {}), "answer_completed": True}
        }

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
        save_answer_results(Path(context.output_dir), answer_results, context.logger)

        return {
            "answer_results": answer_results,
            "completed_stages": ["answer"],
            "metadata": {**state.get("metadata", {}), "answer_completed": True}
        }
    finally:
        TokenStatsCollector.set_current_stage(None)
