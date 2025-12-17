"""Search Stage Node - V3 retrieval (Pure ColBERT)."""

from pathlib import Path
from typing import Dict, Any

from eval.core.stages.search_stage import run_search_stage_v3
from eval.core.nodes.common import (
    is_stage_completed,
    load_search_results,
    save_search_results,
    update_checkpoint,
)
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_search_stage_v3")
async def eval_search_stage_v3_node(state, context) -> Dict[str, Any]:
    """Search V3: Pure ColBERT retrieval."""
    # Check if stage already completed
    if is_stage_completed(state, "search"):
        context.logger.info("Search stage already completed, skipping...")

        # Load existing search results from file
        search_results = load_search_results(Path(context.output_dir))

        return {
            "search_results": search_results,
            "completed_stages": ["search"],
            "metadata": {**state.get("metadata", {}), "search_completed": True}
        }

    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("search")

    try:
        # Call V3-specific search stage
        search_results = await run_search_stage_v3(
            adapter=context.adapter,
            qa_pairs=state.get("qa_pairs"),
            index=state.get("index"),
            conversations=state.get("conversations"),
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
        )

        # Save search results to file
        save_search_results(Path(context.output_dir), search_results, context.logger)

        # Update checkpoint immediately (prevent re-running if later stages fail)
        update_checkpoint(context, state, "search")

        return {
            "search_results": search_results,
            "completed_stages": ["search"],
            "metadata": {**state.get("metadata", {}), "search_completed": True}
        }
    finally:
        TokenStatsCollector.set_current_stage(None)
