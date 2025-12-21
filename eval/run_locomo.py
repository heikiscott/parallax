#!/usr/bin/env python3
"""
LoCoMo Evaluation Runner

Unified script for running evaluations in different modes:
- Mini mode: Quick validation with minimal dataset
- All mode: Full evaluation on all conversations
- Conv mode: Run specific conversation by index
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 添加 src 目录到 sys.path，以便导入 core 模块
_project_root = Path(__file__).parent.parent.resolve()
_src_path = str(_project_root / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)



def find_output_dir(base_name: str, results_dir: Path, resume: bool = False) -> Path:
    """
    Find the output directory.

    If resume=True: Use the most recent existing directory
    If resume=False (default): Create a new directory

    First run: base_name (e.g., locomo-mini)
    Second run: base_name-1 (e.g., locomo-mini-1)
    Third run: base_name-2 (e.g., locomo-mini-2)
    """
    base_dir = results_dir / base_name

    if resume:
        # Resume mode: find the most recent directory
        if base_dir.exists():
            # Find all numbered directories
            numbered_dirs = []
            counter = 1
            while (results_dir / f"{base_name}-{counter}").exists():
                numbered_dirs.append((counter, results_dir / f"{base_name}-{counter}"))
                counter += 1

            # Return the highest numbered directory, or base_dir if no numbered ones
            if numbered_dirs:
                return numbered_dirs[-1][1]
            else:
                return base_dir
        else:
            # No existing directory to resume from
            print(f"⚠️  Warning: No existing directory found for {base_name}")
            print(f"   Creating new directory instead")
            return base_dir
    else:
        # Normal mode: create new directory
        if not base_dir.exists():
            return base_dir

        # Find next available number
        counter = 1
        while (results_dir / f"{base_name}-{counter}").exists():
            counter += 1

        return results_dir / f"{base_name}-{counter}"


def find_log_file(output_dir: Path) -> Path:
    """
    Find the next available log file.

    First log: run.log
    Second log: run_1.log
    Third log: run_2.log
    """
    base_log = output_dir / "run.log"
    if not base_log.exists():
        return base_log

    # Find next available number
    counter = 1
    while (output_dir / f"run_{counter}.log").exists():
        counter += 1

    return output_dir / f"run_{counter}.log"


def handle_truncation_and_find_dir(
    base_name: str,
    results_dir: Path,
    truncate_to: str,
    resume: bool
) -> tuple[Path, bool]:
    """
    Handle truncation if requested and find output directory.

    Args:
        base_name: Base directory name (e.g., "locomo-conv8")
        results_dir: Results base directory
        truncate_to: Comma-separated stages to keep (or None)
        resume: Whether to resume evaluation

    Returns:
        (output_dir, should_continue): Output directory and whether to continue evaluation
    """
    if truncate_to:
        # Parse stages to keep
        keep_stages = [s.strip() for s in truncate_to.split(',')]

        # Find the most recent directory as source
        source_dir = find_output_dir(base_name, results_dir, resume=True)

        if not source_dir.exists():
            print(f"❌ Error: No existing directory found for {base_name}")
            print(f"   Please run evaluation first before truncating")
            sys.exit(1)

        # Create a new directory for truncated results (always increment version)
        new_dir = find_output_dir(base_name, results_dir, resume=False)

        # Perform truncation (copy files to new directory)
        truncate_results(source_dir, keep_stages, new_dir)

        # If no --resume, exit after truncation
        if not resume:
            print(f"\n💡 To continue from this point, run with --resume")
            return new_dir, False

        # If --resume, continue evaluation
        print(f"\n🔄 Resuming evaluation from new directory...")
        return new_dir, True
    else:
        # Normal mode: find or create output directory
        output_dir = find_output_dir(base_name, results_dir, resume=resume)

        if resume:
            print(f"🔄 Resuming from: {output_dir}")

        return output_dir, True


def truncate_results(source_dir: Path, keep_stages: list[str], new_dir: Path):
    """
    Create a new truncated results directory by copying only specified stages.

    Stage order: add -> search -> answer -> evaluate

    Args:
        source_dir: Source results directory (e.g., locomo-conv8-5)
        keep_stages: List of stages to keep (e.g., ["add", "search"])
        new_dir: New directory to create (e.g., locomo-conv8-6)
    """
    import json
    import shutil
    from datetime import datetime

    print(f"🔧 Creating truncated copy:")
    print(f"   Source: {source_dir}")
    print(f"   Target: {new_dir}")
    print(f"   Keeping stages: {', '.join(keep_stages)}")

    all_stages = ["add", "classify", "search", "answer", "evaluate"]

    # Validate stage names
    for stage in keep_stages:
        if stage not in all_stages:
            raise ValueError(f"Invalid stage: {stage}. Valid stages: {all_stages}")

    # Create new directory
    new_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always copy: memunits and index (add stage outputs)
    if "add" in keep_stages:
        # Copy memunits directory
        memunits_src = source_dir / "memunits"
        if memunits_src.exists():
            memunits_dst = new_dir / "memunits"
            shutil.copytree(memunits_src, memunits_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: memunits/")

        # Copy v4_index if exists
        v4_index_src = source_dir / "v4_index"
        if v4_index_src.exists():
            v4_index_dst = new_dir / "v4_index"
            shutil.copytree(v4_index_src, v4_index_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: v4_index/")

        # Copy bm25_index if exists
        bm25_index_src = source_dir / "bm25_index"
        if bm25_index_src.exists():
            bm25_index_dst = new_dir / "bm25_index"
            shutil.copytree(bm25_index_src, bm25_index_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: bm25_index/")

        # Copy vectors if exists
        vectors_src = source_dir / "vectors"
        if vectors_src.exists():
            vectors_dst = new_dir / "vectors"
            shutil.copytree(vectors_src, vectors_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: vectors/")

        # Copy event_clusters if exists
        event_clusters_src = source_dir / "event_clusters"
        if event_clusters_src.exists():
            event_clusters_dst = new_dir / "event_clusters"
            shutil.copytree(event_clusters_src, event_clusters_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: event_clusters/")

    # 2. Copy classify stage output
    if "classify" in keep_stages:
        classify_file = source_dir / "classification_results.json"
        if classify_file.exists():
            shutil.copy2(classify_file, new_dir / "classification_results.json")
            print(f"   ✓ Copied: classification_results.json")

    # 3. Copy search stage output
    if "search" in keep_stages:
        search_file = source_dir / "search_results.json"
        if search_file.exists():
            shutil.copy2(search_file, new_dir / "search_results.json")
            print(f"   ✓ Copied: search_results.json")

        # Copy cluster_selection if exists
        cluster_selection_src = source_dir / "cluster_selection"
        if cluster_selection_src.exists():
            cluster_selection_dst = new_dir / "cluster_selection"
            shutil.copytree(cluster_selection_src, cluster_selection_dst, dirs_exist_ok=True)
            print(f"   ✓ Copied: cluster_selection/")

    # 4. Copy answer stage output
    if "answer" in keep_stages:
        answer_file = source_dir / "answer_results.json"
        if answer_file.exists():
            shutil.copy2(answer_file, new_dir / "answer_results.json")
            print(f"   ✓ Copied: answer_results.json")

    # 5. Copy evaluate stage output
    if "evaluate" in keep_stages:
        eval_file = source_dir / "eval_results.json"
        if eval_file.exists():
            shutil.copy2(eval_file, new_dir / "eval_results.json")
            print(f"   ✓ Copied: eval_results.json")

        report_file = source_dir / "report.txt"
        if report_file.exists():
            shutil.copy2(report_file, new_dir / "report.txt")
            print(f"   ✓ Copied: report.txt")

        token_stats_file = source_dir / "token_stats.json"
        if token_stats_file.exists():
            shutil.copy2(token_stats_file, new_dir / "token_stats.json")
            print(f"   ✓ Copied: token_stats.json")

    # 6. Create new checkpoint file with only kept stages
    checkpoint_src = source_dir / "checkpoint_default.json"
    if checkpoint_src.exists():
        with open(checkpoint_src, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        original_stages = checkpoint.get("completed_stages", [])
        filtered_stages = [s for s in original_stages if s in keep_stages]

        checkpoint["completed_stages"] = filtered_stages
        checkpoint["last_updated"] = datetime.now().isoformat()

        checkpoint_dst = new_dir / "checkpoint_default.json"
        with open(checkpoint_dst, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        print(f"   ✓ Created checkpoint:")
        print(f"     Original stages: {original_stages}")
        print(f"     Kept stages: {filtered_stages}")

    print(f"✅ Truncation complete! New directory: {new_dir.name}")


def run_evaluation(dataset: str, output_dir: Path, conv_id: int = None):
    """
    Run the evaluation with specified parameters.

    Args:
        dataset: Dataset name (locomo or locomo-mini)
        output_dir: Output directory path
        conv_id: Conversation ID (optional)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "uv", "run", "python", "-m", "eval.cli",
        "--dataset", dataset,
        "--system", "parallax",
        "--output-dir", str(output_dir)
    ]

    # Add conversation ID if specified
    if conv_id is not None:
        cmd.extend(["--conv", str(conv_id)])

    # Determine log file
    log_file = find_log_file(output_dir)

    # Print info
    if conv_id is not None:
        print(f"Running evaluation for conversation {conv_id}...")
    elif dataset == "locomo-mini":
        print(f"Running mini dataset validation...")
    elif dataset.startswith("locomo-q"):
        print(f"Running {dataset} evaluation...")
    else:
        print(f"Running full evaluation on all conversations...")

    print(f"Output directory: {output_dir}")
    print(f"Logging to: {log_file}")
    print()

    # Run command and tee output to both console and log file
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()

        process.wait()

    print()
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Log saved to: {log_file}")

    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="""
LoCoMo Evaluation Runner - Unified script for running memory evaluations

This script provides a streamlined interface for running LoCoMo (Long Context Memory)
evaluations with different dataset sizes and configurations. It supports checkpoint-based
resume and stage-level result management.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EVALUATION MODES:
  The script supports multiple evaluation modes for different testing scenarios:

  --mini, -m              Quick validation with minimal dataset (fast iteration)
  --q10/q20/q30/q50       Run first N questions from conv0 (progressive testing)
  --all, -a               Full evaluation on all conversations (complete benchmark)
  --conv N, -c N          Run specific conversation by index (targeted testing)

PIPELINE STAGES:
  Evaluations run through 4 stages in sequence:

  1. add       - Process conversations and create memory units (MemUnits)
  2. search    - Retrieve relevant memories for each question
  3. answer    - Generate answers using LLM with retrieved context
  4. evaluate  - Compare answers with ground truth and compute metrics

OUTPUT DIRECTORY STRUCTURE:
  Results are saved with automatic versioning:

  eval/results/
  ├── locomo-conv8/        # First run
  ├── locomo-conv8-1/      # Second run
  └── locomo-conv8-2/      # Third run (most recent)

  Each directory contains:
  - memunits/              - Memory units and statistics
  - v4_index/              - Vector index (if using v4 mode)
  - search_results.json    - Retrieval results
  - answer_results.json    - Generated answers
  - eval_results.json      - Evaluation metrics
  - checkpoint_default.json - Progress tracking
  - report.txt             - Human-readable summary
  - *.log                  - Execution logs

EXAMPLES:
  # Quick validation
  %(prog)s --mini
  %(prog)s -m

  # Progressive testing with different dataset sizes
  %(prog)s --q10           # First 10 questions (3 sessions)
  %(prog)s --q20           # First 20 questions (9 sessions)
  %(prog)s --q30           # First 30 questions (9 sessions)
  %(prog)s --q50           # First 50 questions (12 sessions)

  # Full evaluation
  %(prog)s --all           # All conversations
  %(prog)s -a              # Same (short form)

  # Targeted conversation testing
  %(prog)s --conv 4        # Run conversation 4
  %(prog)s -c 4            # Same (short form)

  # Resume from checkpoint (skip completed stages)
  %(prog)s -c 1 --resume
  %(prog)s -c 1 -r         # Same (short form)

  # Stage truncation workflow (advanced)
  # Useful for rerunning later stages with different configurations

  # 1. Truncate to early stages (creates new directory)
  %(prog)s -c 8 --truncate-to add,search

  # 2. Truncate and immediately resume (one-step rerun)
  %(prog)s -c 8 --truncate-to add,search --resume

  # 3. Truncate to single stage
  %(prog)s -c 8 --truncate-to add

  # Common workflow: Test configuration changes
  # Run full pipeline once
  %(prog)s -c 8

  # Truncate to search stage and rerun with new answer config
  %(prog)s -c 8 --truncate-to add,search --resume

NOTES:
  - Results are automatically versioned, previous runs are never overwritten
  - Checkpoint system enables resuming from any stage
  - Truncation creates new directories, preserving original results
  - Logs and metrics are saved for every run for reproducibility
        """
    )

    # Create mutually exclusive group for modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-m", "--mini",
        action="store_true",
        help="Run mini dataset for quick validation (fastest, for development)"
    )
    group.add_argument(
        "--q10",
        action="store_true",
        help="Run first 10 questions from conv0 (3 sessions, ~5 min)"
    )
    group.add_argument(
        "--q20",
        action="store_true",
        help="Run first 20 questions from conv0 (9 sessions, ~10 min)"
    )
    group.add_argument(
        "--q30",
        action="store_true",
        help="Run first 30 questions from conv0 (9 sessions, ~15 min)"
    )
    group.add_argument(
        "--q50",
        action="store_true",
        help="Run first 50 questions from conv0 (12 sessions, ~20 min)"
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run full evaluation on all conversations (complete benchmark)"
    )
    group.add_argument(
        "-c", "--conv",
        type=int,
        metavar="CONV_ID",
        help="Run specific conversation by index (0-based). Example: -c 8 for conv8"
    )

    # Add resume option
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint, skipping completed stages. "
             "Finds the highest-numbered directory (e.g., locomo-conv8-5) and continues from there."
    )

    # Add truncate option
    parser.add_argument(
        "--truncate-to",
        type=str,
        metavar="STAGES",
        help="Truncate results to keep only specified stages (comma-separated: add,search,answer,evaluate). "
             "Creates a new versioned directory with only the specified stage outputs, allowing you to "
             "rerun later stages with different configurations. Use with --resume to immediately continue. "
             "Example: --truncate-to add,search creates a new dir with only add+search results."
    )

    args = parser.parse_args()

    # Get evaluation root directory
    eval_root = Path(__file__).parent
    results_dir = eval_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Process based on mode
    if args.mini:
        # Mini mode: locomo-mini, locomo-mini-1, locomo-mini-2, ...
        dataset = "locomo-mini"
        base_name = "locomo-mini"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.q10:
        # Q10 mode: locomo-q10, locomo-q10-1, locomo-q10-2, ...
        dataset = "locomo-q10"
        base_name = "locomo-q10"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.q20:
        # Q20 mode: locomo-q20, locomo-q20-1, locomo-q20-2, ...
        dataset = "locomo-q20"
        base_name = "locomo-q20"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.q30:
        # Q30 mode: locomo-q30, locomo-q30-1, locomo-q30-2, ...
        dataset = "locomo-q30"
        base_name = "locomo-q30"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.q50:
        # Q50 mode: locomo-q50, locomo-q50-1, locomo-q50-2, ...
        dataset = "locomo-q50"
        base_name = "locomo-q50"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.all:
        # All mode: locomo-all, locomo-all-1, locomo-all-2, ...
        dataset = "locomo"
        base_name = "locomo-all"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir)

    elif args.conv is not None:
        # Conv mode: locomo-conv3, locomo-conv3-1, locomo-conv3-2, ...
        conv_id = args.conv
        dataset = "locomo"
        base_name = f"locomo-conv{conv_id}"

        output_dir, should_continue = handle_truncation_and_find_dir(
            base_name, results_dir, args.truncate_to, args.resume
        )

        if not should_continue:
            return 0

        return run_evaluation(dataset, output_dir, conv_id=conv_id)


if __name__ == "__main__":
    sys.exit(main())
