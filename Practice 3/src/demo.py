#!/usr/bin/env python3
"""
Demo script — Regenerate all report figures with a single command.

Usage:
    python demo.py          # full run (~8 min)
    python demo.py --fast  # quick test (fewer seeds, fewer generations)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
DOCS_IMG = SRC.parent / "docs" / "img"


def run_benchmarks(fast: bool = False):
    """Regenerate benchmark plots."""
    print("=" * 60)
    print("1/2 Benchmark plots (scalability, feasibility, comparison…)")
    print("=" * 60)
    cmd = [sys.executable, str(SRC / "benchmark_plots.py")]
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)


def run_report_plots(fast: bool = False):
    """Regenerate report convergence plot."""
    print("=" * 60)
    print("2/2 Report convergence plot")
    print("=" * 60)
    cmd = [sys.executable, str(SRC / "generate_report_plots.py")]
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Regenerate all report figures")
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick run with fewer seeds and generations"
    )
    args = parser.parse_args()

    t0 = time.time()

    run_benchmarks(fast=args.fast)
    run_report_plots(fast=args.fast)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s — all figures in {DOCS_IMG}")
    print("=" * 60)


if __name__ == "__main__":
    main()