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


def find_output_dir(base_name: str, results_dir: Path) -> Path:
    """
    Find the next available output directory.

    First run: base_name (e.g., locomo-mini)
    Second run: base_name-1 (e.g., locomo-mini-1)
    Third run: base_name-2 (e.g., locomo-mini-2)
    """
    base_dir = results_dir / base_name
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
        "uv", "run", "python", "-m", "evaluation.cli",
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
        description="Run LoCoMo evaluation in different modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mini           Run mini dataset for quick validation
  %(prog)s -m               Same as --mini (short form)
  %(prog)s --all            Run all conversations
  %(prog)s -a               Same as --all (short form)
  %(prog)s --conv 4         Run single conversation (index 4)
  %(prog)s -c 4             Same as --conv (short form)
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
        output_dir = find_output_dir(base_name, results_dir)

        return run_evaluation(dataset, output_dir)

    elif args.all:
        # All mode: locomo-all, locomo-all-1, locomo-all-2, ...
        dataset = "locomo"
        base_name = "locomo-all"
        output_dir = find_output_dir(base_name, results_dir)

        return run_evaluation(dataset, output_dir)

    elif args.conv is not None:
        # Conv mode: locomo-conv3, locomo-conv3-1, locomo-conv3-2, ...
        conv_id = args.conv
        dataset = "locomo"
        base_name = f"locomo-conv{conv_id}"
        output_dir = find_output_dir(base_name, results_dir)

        return run_evaluation(dataset, output_dir, conv_id=conv_id)


if __name__ == "__main__":
    sys.exit(main())