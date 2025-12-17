"""Add Stage Node - V4 fine-grained MemUnit extraction."""

from pathlib import Path
from typing import Dict, Any

from eval.core.nodes.common import is_stage_completed
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_add_stage_v4")
async def eval_add_stage_v4_node(state, context) -> Dict[str, Any]:
    """Add V4: Fine-grained MemUnit extraction.

    V4 uses fine-grained segmentation:
    - Each MemUnit corresponds to 1-2 dialogue turns (one Q-A pair)
    - One-time LLM call segments entire conversation
    - Finer granularity than V3's topic-level segmentation
    """
    # Check if stage already completed
    if is_stage_completed(state, "add"):
        context.logger.info("Add V4 stage already completed, skipping...")

        # Rebuild index metadata
        output_dir = Path(context.output_dir)
        dataset = state.get("dataset")
        conversations = dataset.conversations if dataset else []

        index = {
            "type": "lazy_load",
            "pipeline_version": "v4",
            "granularity": "fine-grained",
            "memunits_dir": str(output_dir / "memunits"),
            "v4_index_dir": str(output_dir / "v4_index"),
            "output_dir": str(output_dir),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "total_conversations": len(conversations),
        }

        return {
            "index": index,
            "completed_stages": ["add"],
            "metadata": {**state.get("metadata", {}), "add_completed": True}
        }

    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("add")

    try:
        # Call V4-specific add method
        index = await context.adapter.add_v4(
            conversations=state.get("dataset").conversations,
            output_dir=Path(context.output_dir),
            checkpoint_manager=context.checkpoint_manager,
        )

        # Add output_dir to index if not present
        if index and "output_dir" not in index:
            index["output_dir"] = str(context.output_dir)

        return {
            "index": index,
            "completed_stages": ["add"],
            "metadata": {**state.get("metadata", {}), "add_completed": True}
        }
    finally:
        TokenStatsCollector.set_current_stage(None)
