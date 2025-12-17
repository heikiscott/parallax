"""Add Stage Node - Build indexes."""

from pathlib import Path
from typing import Dict, Any

from eval.core.stages.add_stage import run_add_stage
from eval.core.nodes.common import is_stage_completed
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_add_stage")
async def eval_add_stage_node(state, context) -> Dict[str, Any]:
    """Add: Build indexes."""
    # Check if stage already completed
    if is_stage_completed(state, "add"):
        context.logger.info("Add stage already completed, skipping...")

        # Rebuild index metadata (needed by Search Stage)
        output_dir = Path(context.output_dir)
        dataset = state.get("dataset")
        conversations = dataset.conversations if dataset else []

        # Check if hybrid search is enabled
        use_hybrid = True
        if hasattr(context, 'adapter') and hasattr(context.adapter, 'config'):
            use_hybrid = context.adapter.config.get("search", {}).get("use_hybrid_search", True)

        index = {
            "type": "lazy_load",
            "memunits_dir": str(output_dir / "memunits"),
            "bm25_index_dir": str(output_dir / "bm25_index"),
            "emb_index_dir": str(output_dir / "vectors"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": use_hybrid,
            "total_conversations": len(conversations),
            "output_dir": str(output_dir),
        }

        return {
            "index": index,
            "completed_stages": ["add"],
            "metadata": {**state.get("metadata", {}), "add_completed": True}
        }

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

        # Add output_dir to index (for subsequent stages)
        index = result.get("index", {})
        if index and "output_dir" not in index:
            index["output_dir"] = str(context.output_dir)

        return {
            "index": index,
            "completed_stages": ["add"],
            "metadata": {**state.get("metadata", {}), "add_completed": True}
        }
    finally:
        TokenStatsCollector.set_current_stage(None)
