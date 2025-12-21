"""Classify Stage Node - Batch question classification."""

from pathlib import Path
from typing import Dict, Any

from eval.core.stages.classify_stage import (
    run_classify_stage,
    attach_classification_to_qa_pairs,
    load_classification_results,
)
from eval.core.nodes.common import is_stage_completed, update_checkpoint
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node


@register_node("eval_classify_stage")
async def eval_classify_stage_node(state, context) -> Dict[str, Any]:
    """Classify: Batch classify all questions before search.

    This stage performs global batch classification of all questions from all
    conversations. Classification results are attached to each QAPair's metadata
    for downstream stages to access via qa.metadata["classification"].
    """
    output_dir = Path(context.output_dir)

    # Check if stage already completed
    if is_stage_completed(state, "classify"):
        context.logger.info("Classify stage already completed, skipping...")

        # Load existing classification results
        classification_results = load_classification_results(output_dir)

        # Re-attach to QA pairs (needed after checkpoint restore)
        qa_pairs = state.get("qa_pairs", [])
        attach_classification_to_qa_pairs(qa_pairs, classification_results)

        return {
            "classification_results": classification_results,
            "qa_pairs": qa_pairs,
            "completed_stages": ["classify"],
            "metadata": {**state.get("metadata", {}), "classify_completed": True}
        }

    # Set current stage for token stats collection
    TokenStatsCollector.set_current_stage("classify")

    try:
        qa_pairs = state.get("qa_pairs", [])

        # Get classification config from system config via adapter
        use_llm_classification = True
        batch_size = 15

        if hasattr(context, 'adapter') and hasattr(context.adapter, 'config'):
            retrieval_config = context.adapter.config.get("retrieval", {})
            agentic_v4_config = retrieval_config.get("agentic_v4", {})
            use_llm_classification = agentic_v4_config.get("use_llm_classification", True)
            batch_size = agentic_v4_config.get("classification_batch_size", 15)

        # Run classification stage
        classification_results = await run_classify_stage(
            qa_pairs=qa_pairs,
            output_dir=output_dir,
            checkpoint_manager=context.checkpoint_manager,
            logger=context.logger,
            llm_provider=context.llm_provider,
            use_llm_classification=use_llm_classification,
            batch_size=batch_size,
        )

        # Update checkpoint
        update_checkpoint(context, state, "classify")

        return {
            "classification_results": classification_results,
            "qa_pairs": qa_pairs,  # Now with classification attached to metadata
            "completed_stages": ["classify"],
            "metadata": {**state.get("metadata", {}), "classify_completed": True}
        }
    finally:
        TokenStatsCollector.set_current_stage(None)
