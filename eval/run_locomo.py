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

from core.observation.logger import setup_logger

logger = setup_logger(name="runner")


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
            logger.warning(f"⚠️  Warning: No existing directory found for {base_name}")
            logger.info(f"   Creating new directory instead")
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
        logger.info(f"Running evaluation for conversation {conv_id}...")
    elif dataset == "locomo-mini":
        logger.info(f"Running mini dataset validation...")
    elif dataset.startswith("locomo-q"):
        logger.info(f"Running {dataset} evaluation...")
    else:
        logger.info(f"Running full evaluation on all conversations...")

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logging to: {log_file}")
    logger.info("")

    # Run command and tee output to both console and log file
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()

        process.wait()

    logger.info("")
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Log saved to: {log_file}")

    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run LoCoMo evaluation in different modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mini           Run mini dataset for quick validation
  %(prog)s -m               Same as --mini (short form)
  %(prog)s --q10            Run first 10 questions from conv0
  %(prog)s --q20            Run first 20 questions from conv0
  %(prog)s --q30            Run first 30 questions from conv0
  %(prog)s --q50            Run first 50 questions from conv0
  %(prog)s --all            Run all conversations
  %(prog)s -a               Same as --all (short form)
  %(prog)s --conv 4         Run single conversation (index 4)
  %(prog)s -c 4             Same as --conv (short form)
  %(prog)s -c 1 --resume    Resume conversation 1 from last checkpoint
  %(prog)s -c 1 -r          Same as above (short form)
        """
    )

    # Create mutually exclusive group for modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-m", "--mini",
        action="store_true",
        help="Run mini dataset validation"
    )
    group.add_argument(
        "--q10",
        action="store_true",
        help="Run first 10 questions from conv0 (3 sessions)"
    )
    group.add_argument(
        "--q20",
        action="store_true",
        help="Run first 20 questions from conv0 (9 sessions)"
    )
    group.add_argument(
        "--q30",
        action="store_true",
        help="Run first 30 questions from conv0 (9 sessions)"
    )
    group.add_argument(
        "--q50",
        action="store_true",
        help="Run first 50 questions from conv0 (12 sessions)"
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run full evaluation on all conversations"
    )
    group.add_argument(
        "-c", "--conv",
        type=int,
        metavar="CONV_ID",
        help="Run specific conversation by index (0-based)"
    )

    # Add resume option
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint instead of creating a new run"
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
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.q10:
        # Q10 mode: locomo-q10, locomo-q10-1, locomo-q10-2, ...
        dataset = "locomo-q10"
        base_name = "locomo-q10"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.q20:
        # Q20 mode: locomo-q20, locomo-q20-1, locomo-q20-2, ...
        dataset = "locomo-q20"
        base_name = "locomo-q20"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.q30:
        # Q30 mode: locomo-q30, locomo-q30-1, locomo-q30-2, ...
        dataset = "locomo-q30"
        base_name = "locomo-q30"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.q50:
        # Q50 mode: locomo-q50, locomo-q50-1, locomo-q50-2, ...
        dataset = "locomo-q50"
        base_name = "locomo-q50"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.all:
        # All mode: locomo-all, locomo-all-1, locomo-all-2, ...
        dataset = "locomo"
        base_name = "locomo-all"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir)

    elif args.conv is not None:
        # Conv mode: locomo-conv3, locomo-conv3-1, locomo-conv3-2, ...
        conv_id = args.conv
        dataset = "locomo"
        base_name = f"locomo-conv{conv_id}"
        output_dir = find_output_dir(base_name, results_dir, resume=args.resume)

        if args.resume:
            logger.info(f"🔄 Resuming from: {output_dir}")
        return run_evaluation(dataset, output_dir, conv_id=conv_id)


if __name__ == "__main__":
    sys.exit(main())
