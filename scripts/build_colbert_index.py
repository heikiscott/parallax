#!/usr/bin/env python
"""ColBERT Index Builder for Parallax Evaluation.

This script builds ColBERT token-level embeddings for all documents.
Unlike single-vector embeddings, ColBERT stores multi-vector per document
for late interaction (MaxSim) scoring.

Usage:
    python scripts/build_colbert_index.py --data-dir eval/results/locomo-all/memunits
    python scripts/build_colbert_index.py --data-dir eval/results/locomo-all/memunits --conv 0

Output:
    {save_dir}/colbert_index_conv_{i}.pkl

Index Format:
    [
        {
            "doc": { ... original document dict ... },
            "embeddings": np.ndarray  # [seq_len, 128]
        },
        ...
    ]
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

# Ensure core module is importable (for core.di)
import os
os.chdir(project_root)  # Change to project root for relative imports

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_searchable_text_for_colbert(doc: dict) -> str:
    """Build searchable text from a document for ColBERT encoding.

    For ColBERT, we concatenate all relevant text into one string.
    ColBERT will tokenize and create token-level embeddings.

    Priority:
    1. If event_log exists, use atomic_facts
    2. Otherwise, use subject + summary + narrative
    """
    parts = []

    # Use event_log atomic_facts if available
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            for fact in atomic_facts:
                if isinstance(fact, dict) and "fact" in fact:
                    parts.append(fact["fact"])
                elif isinstance(fact, str):
                    parts.append(fact)
            return " ".join(parts)

    # Fallback to narrative/summary/subject
    if doc.get("subject"):
        parts.append(doc["subject"])
    if doc.get("summary"):
        parts.append(doc["summary"])
    if doc.get("narrative"):
        parts.append(doc["narrative"])

    return " ".join(parts)


async def build_colbert_index_for_file(
    file_path: Path,
    save_dir: Path,
    colbert_service
) -> dict:
    """Build ColBERT index for a single memunit file.

    Args:
        file_path: Path to memunit JSON file
        save_dir: Directory to save index
        colbert_service: ColBERT service instance

    Returns:
        dict with stats: {docs_count, index_size_mb, encoding_time_s}
    """
    conv_index = file_path.stem.split('_')[-1]

    logger.info(f"\n{'='*60}")
    logger.info(f"Building ColBERT index for {file_path.name}...")
    logger.info(f"{'='*60}")

    # Load documents
    with open(file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not docs:
        logger.warning(f"No documents found in {file_path.name}")
        return {"docs_count": 0, "index_size_mb": 0, "encoding_time_s": 0}

    # Prepare texts for encoding
    texts = []
    for doc in docs:
        text = build_searchable_text_for_colbert(doc)
        texts.append(text)

    logger.info(f"Encoding {len(texts)} documents with ColBERT...")
    logger.info(f"Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")

    # Encode documents
    start_time = time.time()
    embeddings = await colbert_service.encode_documents(texts)
    encoding_time = time.time() - start_time

    logger.info(f"Encoding completed in {encoding_time:.1f}s "
                f"({encoding_time/len(texts)*1000:.0f}ms/doc)")

    # Build index structure
    colbert_index = []
    for doc, emb in zip(docs, embeddings):
        colbert_index.append({
            "doc": doc,
            "embeddings": emb,
        })

    # Calculate index size
    total_tokens = sum(emb.shape[0] for emb in embeddings)
    index_size_bytes = total_tokens * 128 * 4  # 128 dim * 4 bytes per float
    index_size_mb = index_size_bytes / (1024 * 1024)

    logger.info(f"Index stats:")
    logger.info(f"  Documents: {len(colbert_index)}")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Avg tokens/doc: {total_tokens / len(colbert_index):.0f}")
    logger.info(f"  Estimated size: {index_size_mb:.2f} MB")

    # Save index
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"colbert_index_conv_{conv_index}.pkl"

    logger.info(f"Saving ColBERT index to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(colbert_index, f)

    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Actual file size: {actual_size_mb:.2f} MB")

    return {
        "docs_count": len(colbert_index),
        "index_size_mb": actual_size_mb,
        "encoding_time_s": encoding_time,
        "total_tokens": total_tokens,
    }


async def main():
    parser = argparse.ArgumentParser(description="Build ColBERT index for evaluation")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing memunit JSON files"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save ColBERT index (default: {data-dir}/../colbert_index)"
    )
    parser.add_argument(
        "--conv",
        type=int,
        default=None,
        help="Only build index for specific conversation (e.g., --conv 0)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_dir else data_dir.parent / "colbert_index"

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Save directory: {save_dir}")

    # Initialize ColBERT service
    logger.info("Initializing ColBERT service...")
    # Import ColBERTService and ColBERTConfig classes directly
    from retrieval.services.colbert_service import ColBERTService, ColBERTConfig, get_colbert_service

    colbert_service = get_colbert_service()
    await colbert_service.initialize()

    # Find memunit files
    import glob
    if args.conv is not None:
        memunit_files = [data_dir / f"memunit_list_conv_{args.conv}.json"]
        if not memunit_files[0].exists():
            logger.error(f"File not found: {memunit_files[0]}")
            sys.exit(1)
    else:
        memunit_files = sorted(glob.glob(str(data_dir / "memunit_list_conv_*.json")))

    if not memunit_files:
        logger.error(f"No memunit files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(memunit_files)} memunit file(s)")

    # Build index for each file
    total_stats = {
        "total_docs": 0,
        "total_size_mb": 0,
        "total_time_s": 0,
        "files_processed": 0,
    }

    for file_path in memunit_files:
        file_path = Path(file_path)
        try:
            stats = await build_colbert_index_for_file(
                file_path=file_path,
                save_dir=save_dir,
                colbert_service=colbert_service,
            )
            total_stats["total_docs"] += stats["docs_count"]
            total_stats["total_size_mb"] += stats["index_size_mb"]
            total_stats["total_time_s"] += stats["encoding_time_s"]
            total_stats["files_processed"] += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("ColBERT Indexing Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {total_stats['files_processed']}")
    logger.info(f"Total documents: {total_stats['total_docs']}")
    logger.info(f"Total index size: {total_stats['total_size_mb']:.2f} MB")
    logger.info(f"Total encoding time: {total_stats['total_time_s']:.1f}s")
    if total_stats["total_docs"] > 0:
        logger.info(f"Avg time per doc: {total_stats['total_time_s']/total_stats['total_docs']*1000:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
