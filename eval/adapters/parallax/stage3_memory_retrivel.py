"""Stage 3: Memory Retrieval for Evaluation.

This module is ONLY for evaluation purposes. All retrieval logic is in src/retrieval/.
This file only handles:
- Loading evaluation data
- Calling retrieval APIs from src/retrieval/
- Collecting and saving results
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional
import time
import logging

import nltk
import asyncio

logger = logging.getLogger(__name__)

from eval.adapters.parallax.config import ExperimentConfig

# Import retrieval functions from src/retrieval/ (NO duplication)
# Note: uses short paths because pytest.ini sets pythonpath = . src eval
from retrieval.pipelines.agentic import agentic_retrieval
from retrieval.pipelines.lightweight import lightweight_retrieval
from retrieval.pipelines.rerank import reranker_search
from retrieval.pipelines.search_utils import (
    search_with_bm25_index,
    search_with_emb_index,
    hybrid_search_with_rrf,
)
from retrieval.services import rerank as rerank_service

# LLM Provider
from providers.llm.llm_provider import LLMProvider

# Group Event Cluster
from memory.group_event_cluster import GroupEventClusterIndex


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)

    try:
        from nltk.corpus import stopwords
        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


# ============================================================================
# Eval-specific helper functions (NOT in src/retrieval/)
# ============================================================================

def _extract_cluster_selection_data(results_for_conv: List[dict]) -> dict:
    """Extract Cluster Selection info from retrieval results for checkpoint.

    Args:
        results_for_conv: All QA retrieval results for current conversation

    Returns:
        Cluster selection checkpoint data with question selection details
    """
    checkpoint_data = {
        "qa_count": len(results_for_conv),
        "questions": [],
    }

    for result in results_for_conv:
        if not result:
            continue

        retrieval_meta = result.get("retrieval_metadata", {})
        cluster_expansion = retrieval_meta.get("cluster_expansion", {})

        if cluster_expansion.get("strategy") == "cluster_rerank":
            question_data = {
                "query": result.get("query", ""),
                "clusters_found": cluster_expansion.get("clusters_found", []),
                "clusters_selected": cluster_expansion.get("clusters_selected", []),
                "selection_reasoning": cluster_expansion.get("selection_reasoning", ""),
                "cluster_details": cluster_expansion.get("cluster_details", {}),
                "members_per_cluster": cluster_expansion.get("members_per_cluster", {}),
                "final_count": cluster_expansion.get("final_count", 0),
                "truncated": cluster_expansion.get("truncated", False),
                "evidence_cluster_analysis": result.get("evidence_cluster_analysis"),
            }
            checkpoint_data["questions"].append(question_data)

    return checkpoint_data if checkpoint_data["questions"] else None


def _analyze_evidence_clusters(
    evidence_list: list,
    cluster_index: GroupEventClusterIndex,
    unit_ids: List[str],
) -> dict:
    """Analyze which MemUnits and Clusters correspond to ground truth evidence.

    Args:
        evidence_list: Evidence list from ground truth (with evidence_id)
        cluster_index: Cluster index
        unit_ids: Retrieved unit_ids

    Returns:
        Analysis result including:
        - evidence_units: MemUnit for each evidence (if matched)
        - evidence_clusters: Clusters for each evidence
        - coverage: How many evidence clusters are covered by retrieval
    """
    analysis = {
        "evidence_details": [],
        "unique_evidence_clusters": [],
        "clusters_in_results": [],
        "cluster_coverage": 0.0,
    }

    evidence_clusters = set()
    clusters_in_results = set()
    retrieved_unit_set = set(unit_ids)

    for evidence in evidence_list:
        evidence_id = evidence.get("evidence_id")
        if not evidence_id:
            continue

        matched_unit_id = None
        matched_cluster_ids = []

        # Direct match
        if evidence_id in cluster_index.unit_to_clusters:
            matched_unit_id = evidence_id
            matched_cluster_ids = cluster_index.unit_to_clusters.get(evidence_id, [])
        else:
            # Try extracting mu_X part for matching
            for unit_id in cluster_index.unit_to_clusters.keys():
                if evidence_id in unit_id or unit_id in evidence_id:
                    matched_unit_id = unit_id
                    matched_cluster_ids = cluster_index.unit_to_clusters.get(unit_id, [])
                    break

        detail = {
            "evidence_id": evidence_id,
            "matched_unit_id": matched_unit_id,
            "cluster_ids": matched_cluster_ids,
            "in_results": matched_unit_id in retrieved_unit_set if matched_unit_id else False,
        }
        analysis["evidence_details"].append(detail)

        for cluster_id in matched_cluster_ids:
            evidence_clusters.add(cluster_id)
            if matched_unit_id and matched_unit_id in retrieved_unit_set:
                clusters_in_results.add(cluster_id)

    analysis["unique_evidence_clusters"] = list(evidence_clusters)
    analysis["clusters_in_results"] = list(clusters_in_results)

    if evidence_clusters:
        analysis["cluster_coverage"] = len(clusters_in_results) / len(evidence_clusters)
    else:
        analysis["cluster_coverage"] = 0.0

    return analysis


# ============================================================================
# Main evaluation entry point
# ============================================================================

async def main():
    """Main function to perform batch search and save results in nemori format."""
    # --- Configuration ---
    config = ExperimentConfig()
    bm25_index_dir = (
        Path(__file__).parent / config.experiment_name / "bm25_index"
    )
    emb_index_dir = (
        Path(__file__).parent / config.experiment_name / "vectors"
    )
    save_dir = Path(__file__).parent / config.experiment_name

    dataset_path = config.datase_path
    results_output_path = save_dir / "search_results.json"

    # Checkpoint file path for resume support
    checkpoint_path = save_dir / "search_results_checkpoint.json"

    # Ensure NLTK data is ready
    ensure_nltk_data()

    # Initialize LLM Provider (for Agentic retrieval)
    llm_provider = None
    llm_config = None
    if config.use_agentic_retrieval:
        llm_config = config.llm_config.get(config.llm_service, config.llm_config["openai"])

        llm_provider = LLMProvider(
            provider_type="openai",
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=int(llm_config.get("max_tokens", 32768)),
        )
        logger.info(f"✅ LLM Provider initialized for agentic retrieval")
        logger.info(f"   Model: {llm_config['model']}")

    # Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Resume support: load existing checkpoint
    all_search_results = {}
    processed_conversations = set()

    if checkpoint_path.exists():
        logger.info(f"\n🔄 Found checkpoint file: {checkpoint_path}")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                all_search_results = json.load(f)
            processed_conversations = set(all_search_results.keys())
            logger.info(f"✅ Loaded {len(processed_conversations)} conversations from checkpoint")
            logger.debug(f"   Already processed: {sorted(processed_conversations)}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            logger.info(f"   Starting from scratch...")
            all_search_results = {}
            processed_conversations = set()
    else:
        logger.info(f"\n🆕 No checkpoint found, starting from scratch")

    # Iterate through the dataset
    for i, conversation_data in enumerate(dataset):
        conv_id = f"locomo_exp_user_{i}"

        # Skip already processed conversations
        if conv_id in processed_conversations:
            logger.info(f"\n⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            continue

        speaker_a = conversation_data["conversation"].get("speaker_a")
        speaker_b = conversation_data["conversation"].get("speaker_b")
        print(f"\n--- Processing Conversation ID: {conv_id} ({i+1}/{len(dataset)}) ---")

        if "qa" not in conversation_data:
            logger.warning(f"Warning: No 'qa' key found in conversation #{i}. Skipping.")
            continue

        # --- Load index once per conversation ---
        if config.use_hybrid_search:
            # Load Embedding index
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                logger.error(
                    f"Error: Embedding index not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)

            # Load BM25 index
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                logger.error(
                    f"Error: BM25 index not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]

            logger.debug(f"Loaded both Embedding and BM25 indexes for conversation {i} (Hybrid Search)")

        elif config.use_emb:
            # Load Embedding index only
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                logger.error(
                    f"Error: Index file not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)
            bm25 = None
            docs = None
        else:
            # Load BM25 index only
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                logger.error(
                    f"Error: Index file not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]
            emb_index = None

        # Load cluster index (if enabled)
        cluster_index = None
        if getattr(config, 'enable_group_event_cluster', False):
            cluster_index_dir = save_dir / "event_clusters"
            cluster_index_path = cluster_index_dir / f"conv_{i}.json"
            if cluster_index_path.exists():
                try:
                    cluster_index = GroupEventClusterIndex.load_from_file(cluster_index_path)
                    logger.info(f"  ✅ Loaded cluster index: {len(cluster_index.clusters)} clusters, {cluster_index.total_units} units")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to load cluster index: {e}")
                    cluster_index = None
            else:
                logger.debug(f"  📭 No cluster index found at {cluster_index_path}")

        # Parallelize per-question retrieval with bounded concurrency
        max_concurrent = int(os.getenv('EVAL_RETRIEVAL_MAX_CONCURRENT', '5'))
        sem = asyncio.Semaphore(max_concurrent)

        if config.use_agentic_retrieval:
            logger.info(f"  🚀 Agentic retrieval enabled with HIGH CONCURRENCY: {max_concurrent} concurrent requests")

        async def process_single_qa(qa_pair):
            """Process a single QA pair (supports multiple retrieval modes)."""
            question = qa_pair.get("question")
            if not question:
                return None
            if qa_pair.get("category") == 5:
                logger.debug(f"Skipping question {question} because it is category 5")
                return None

            qa_start_time = time.time()

            try:
                async with sem:
                    retrieval_metadata = {}

                    # ========== Retrieval mode selection ==========
                    if config.retrieval_mode == "agentic":
                        # Agentic multi-round retrieval
                        top_results, retrieval_metadata = await agentic_retrieval(
                            query=question,
                            config=config,
                            llm_provider=llm_provider,
                            llm_config=llm_config,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                            cluster_index=cluster_index,
                            enable_traversal_stats=True,
                        )

                    elif config.retrieval_mode == "lightweight":
                        # Lightweight fast retrieval
                        top_results, retrieval_metadata = await lightweight_retrieval(
                            query=question,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                            config=config,
                        )

                    else:
                        # Traditional retrieval (backward compatible)
                        if config.use_reranker:
                            # First stage: initial retrieval
                            if config.use_hybrid_search:
                                results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=config.emb_recall_top_n,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k
                                )
                            elif config.use_emb:
                                results = await search_with_emb_index(
                                    query=question,
                                    emb_index=emb_index,
                                    top_n=config.emb_recall_top_n
                                )
                            else:
                                results = await asyncio.to_thread(
                                    search_with_bm25_index,
                                    question,
                                    bm25,
                                    docs,
                                    config.emb_recall_top_n
                                )

                            # Second stage: Reranker
                            top_results = await reranker_search(
                                query=question,
                                results=results,
                                top_n=config.reranker_top_n,
                                reranker_instruction=config.reranker_instruction,
                                batch_size=config.reranker_batch_size,
                                max_retries=config.reranker_max_retries,
                                retry_delay=config.reranker_retry_delay,
                                timeout=config.reranker_timeout,
                                fallback_threshold=config.reranker_fallback_threshold,
                                config=config,
                            )
                        else:
                            # Single-stage retrieval (no reranker)
                            if config.use_hybrid_search:
                                top_results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=20,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k
                                )
                            elif config.use_emb:
                                top_results = await search_with_emb_index(
                                    query=question, emb_index=emb_index, top_n=20
                                )
                            else:
                                top_results = await asyncio.to_thread(
                                    search_with_bm25_index, question, bm25, docs, 20
                                )

                        retrieval_metadata = {
                            "retrieval_mode": "traditional",
                            "use_reranker": config.use_reranker,
                            "use_hybrid_search": config.use_hybrid_search,
                        }

                    # ========== Extract unit_ids ==========
                    unit_ids = []
                    if top_results:
                        for doc, score in top_results:
                            unit_id = doc.get('unit_id')
                            if unit_id:
                                unit_ids.append(unit_id)

                    qa_latency_ms = (time.time() - qa_start_time) * 1000

                    # ========== Extract Cluster info ==========
                    cluster_expansion_meta = retrieval_metadata.get("cluster_expansion", {})
                    unit_to_cluster = cluster_expansion_meta.get("unit_to_cluster", {})

                    unit_cluster_info = []
                    for unit_id in unit_ids:
                        cluster_ids = unit_to_cluster.get(unit_id, [])
                        unit_cluster_info.append({
                            "unit_id": unit_id,
                            "cluster_ids": cluster_ids,
                        })

                    # Analyze evidence clusters
                    evidence_cluster_analysis = None
                    if cluster_index and qa_pair.get("evidence"):
                        evidence_analysis = _analyze_evidence_clusters(
                            evidence_list=qa_pair.get("evidence", []),
                            cluster_index=cluster_index,
                            unit_ids=unit_ids,
                        )
                        evidence_cluster_analysis = evidence_analysis

                    result = {
                        "query": question,
                        "unit_ids": unit_ids,
                        "unit_cluster_info": unit_cluster_info,
                        "original_qa": qa_pair,
                        "evidence_cluster_analysis": evidence_cluster_analysis,
                        "retrieval_metadata": {
                            **retrieval_metadata,
                            "qa_latency_ms": qa_latency_ms,
                            "target_unit_ids_count": len(top_results) if top_results else 0,
                            "actual_unit_ids_count": len(unit_ids),
                        }
                    }

                    return result

            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                import traceback
                traceback.print_exc()
                return None

        tasks = [
            asyncio.create_task(process_single_qa(qa_pair))
            for qa_pair in conversation_data["qa"]
        ]
        results_for_conv = [
            res for res in await asyncio.gather(*tasks) if res is not None
        ]

        all_search_results[conv_id] = results_for_conv

        # Save checkpoint after each conversation
        try:
            logger.debug(f"💾 Saving checkpoint after conversation {conv_id}...")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_search_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Checkpoint saved: {len(all_search_results)} conversations")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save checkpoint: {e}")

        # Save Cluster Selection Checkpoint (separate file)
        cluster_retrieval_cfg = getattr(config, 'group_event_cluster_retrieval_config', {})
        if cluster_retrieval_cfg.get('expansion_strategy') == 'cluster_rerank':
            try:
                cluster_selection_checkpoint = _extract_cluster_selection_data(results_for_conv)
                if cluster_selection_checkpoint:
                    cluster_selection_dir = save_dir / "cluster_selection"
                    cluster_selection_dir.mkdir(parents=True, exist_ok=True)
                    cluster_selection_path = cluster_selection_dir / f"{conv_id}.json"
                    with open(cluster_selection_path, "w", encoding="utf-8") as f:
                        json.dump(cluster_selection_checkpoint, f, indent=2, ensure_ascii=False)
                    logger.debug(f"  💾 Cluster selection checkpoint saved: {cluster_selection_path}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to save cluster selection checkpoint: {e}")

    # Save all results
    print(f"\n{'='*60}")
    print(f"🎉 All conversations processed!")
    print(f"{'='*60}")
    print(f"\nSaving final results to: {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Batch search and retrieval complete!")
    print(f"   Total conversations: {len(all_search_results)}")

    # Remove checkpoint file after completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info(f"🗑️  Checkpoint file removed (task completed)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to remove checkpoint: {e}")

    # Clean up resources
    reranker = rerank_service.get_rerank_service()
    if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
        await reranker.close()


if __name__ == "__main__":
    asyncio.run(main())
