"""Reranking utilities for retrieval pipelines.

This module provides reranker search functionality:
- Batch processing with concurrency control
- Retry with exponential backoff
- Fallback to original ranking on failure
"""

import asyncio
import logging
from typing import List, Tuple, Any, Optional

from retrieval.services import rerank as rerank_service

logger = logging.getLogger(__name__)


async def reranker_search(
    query: str,
    results: List[Tuple[dict, float]],
    top_n: int = 20,
    reranker_instruction: Optional[str] = None,
    batch_size: int = 10,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0,
    fallback_threshold: float = 0.3,
    config: Any = None,
) -> List[Tuple[dict, float]]:
    """Rerank retrieval results using a reranker model.

    For documents with event_log:
    - Format as multi-line text: time + each atomic_fact on its own line

    For traditional documents:
    - Fall back to narrative field

    Stability optimizations:
    - Process documents in batches
    - Serial batch processing (avoid API rate limits)
    - Retry with exponential backoff per batch
    - Auto-fallback to original ranking on low success rate
    - Timeout protection per batch

    Args:
        query: User query
        results: Initial retrieval results
        top_n: Number of results to return (default 20)
        reranker_instruction: Reranker instruction
        batch_size: Documents per batch (default 10)
        max_retries: Max retries per batch (default 3)
        retry_delay: Base retry delay in seconds (default 2.0, exponential backoff)
        timeout: Timeout per batch in seconds (default 30)
        fallback_threshold: Fallback if success rate below this (default 0.3)
        config: Experiment config (for concurrency settings)

    Returns:
        Reranked Top-N results
    """
    if not results:
        return []

    # Step 1: Format documents
    docs_with_text = []
    doc_texts = []
    original_indices = []

    for idx, (doc, score) in enumerate(results):
        # Prefer event_log format (if exists)
        if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
            event_log = doc["event_log"]
            time_str = event_log.get("time", "")
            atomic_facts = event_log.get("atomic_fact", [])

            if isinstance(atomic_facts, list) and atomic_facts:
                formatted_lines = []
                if time_str:
                    formatted_lines.append(time_str)

                for fact in atomic_facts:
                    if isinstance(fact, dict) and "fact" in fact:
                        formatted_lines.append(fact["fact"])
                    elif isinstance(fact, str):
                        formatted_lines.append(fact)

                formatted_text = "\n".join(formatted_lines)

                docs_with_text.append(doc)
                doc_texts.append(formatted_text)
                original_indices.append(idx)
                continue

        # Fall back to narrative field
        if narrative_text := doc.get("narrative"):
            docs_with_text.append(doc)
            doc_texts.append(narrative_text)
            original_indices.append(idx)

    if not doc_texts:
        return []

    reranker = rerank_service.get_rerank_service()
    logger.info(f"Reranking {len(doc_texts)} documents in batches of {batch_size}...")

    # Step 2: Batch processing
    batches = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        batches.append((i, batch))

    async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
        """Process a single batch with retry and timeout."""
        for attempt in range(max_retries):
            try:
                batch_results = await asyncio.wait_for(
                    reranker._make_rerank_request(
                        query, batch_texts, instruction=reranker_instruction
                    ),
                    timeout=timeout,
                )

                # Adjust indices to global
                for item in batch_results["results"]:
                    item["global_index"] = start_idx + item["index"]

                return batch_results["results"]

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"  Batch at {start_idx} timeout (attempt {attempt + 1}), "
                        f"retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"  Batch at {start_idx} timeout after {max_retries} attempts")
                    return []

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"  Batch at {start_idx} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"  Batch at {start_idx} failed after {max_retries} attempts: {e}")
                    return []

    # Controlled concurrency
    max_concurrent = getattr(config, 'reranker_concurrent_batches', 2) if config else 2

    batch_results_list = []
    successful_batches = 0

    # Process in groups
    for group_start in range(0, len(batches), max_concurrent):
        group_batches = batches[group_start : group_start + max_concurrent]

        tasks = [
            process_batch_with_retry(start_idx, batch)
            for start_idx, batch in group_batches
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in group_results:
            if isinstance(result, list) and result:
                batch_results_list.append(result)
                successful_batches += 1
            else:
                batch_results_list.append([])

        # Inter-group delay
        if group_start + max_concurrent < len(batches):
            await asyncio.sleep(0.3)

    # Step 3: Merge results + fallback strategy
    all_rerank_results = []
    for batch_results in batch_results_list:
        all_rerank_results.extend(batch_results)

    success_rate = successful_batches / len(batches) if batches else 0.0
    logger.info(f"Reranker success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)")

    # Fallback 1: Complete failure
    if not all_rerank_results:
        logger.warning("Warning: All reranker batches failed, using original ranking")
        return results[:top_n]

    # Fallback 2: Low success rate
    if success_rate < fallback_threshold:
        logger.warning(
            f"Warning: Reranker success rate too low ({success_rate:.1%} < {fallback_threshold:.1%}), "
            f"using original ranking"
        )
        return results[:top_n]

    logger.info(f"Reranking complete: {len(all_rerank_results)} documents scored")

    # Step 4: Sort by reranker score and return Top-N
    sorted_results = sorted(
        all_rerank_results,
        key=lambda x: x["relevance_score"],
        reverse=True,
    )[:top_n]

    # Map back to original documents
    final_results = [
        (results[original_indices[item["global_index"]]][0], item["relevance_score"])
        for item in sorted_results
    ]

    return final_results
