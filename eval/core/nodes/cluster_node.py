"""Cluster Stage Node - Group event clustering."""

from pathlib import Path
from typing import Dict, Any

from eval.core.stages.cluster_stage import run_cluster_stage
from eval.core.nodes.common import is_stage_completed
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_cluster_stage")
async def eval_cluster_stage_node(state, context) -> Dict[str, Any]:
    """Cluster: Group event clustering."""
    # Check if stage already completed
    if is_stage_completed(state, "cluster"):
        context.logger.info("Cluster stage already completed, skipping...")
        return {
            "completed_stages": ["cluster"],
            "metadata": {**state.get("metadata", {}), "cluster_completed": True}
        }

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
        TokenStatsCollector.set_current_stage(None)
