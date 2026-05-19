"""
Main entry point for Practice 3: Talk Allocation via CHC Metaheuristic.

Usage:
    python main.py --schools  <schools.csv>
                  --talks    <requested_talks.csv>
                  --researchers <proposed_talks.csv>
                  [--pop-size 50] [--generations 200] [--seed 42]
                  [--verbose]

If no CSVs are provided, an instance is generated using the professor's
data_generator_talks11f logic and the algorithm runs on it directly.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from models import School, Talk, Researcher
from data_loader import load_instance, build_valid_researchers_per_talk
from fitness import compute_fitness, DEFAULT_CONFIG
from chc import chc
from generate_instance import generate_instance


# ---------------------------------------------------------------------------
# Result display utilities
# ---------------------------------------------------------------------------

def _decode_chromosome(chrom, talks, researchers, schools):
    """Return a human-readable summary of the solution."""
    researcher_ids = list(researchers.keys())
    rows = []
    for talk_id, idx in enumerate(chrom):
        talk = talks[talk_id]
        school = schools[talk.school_id]
        r_str = researcher_ids[idx] if 0 <= idx < len(researcher_ids) else "UNASSIGNED"
        r = researchers.get(r_str)
        rows.append({
            "talk_id": talk_id,
            "school": talk.school_id,
            "location": school.location,
            "topic": talk.topic,
            "level": talk.level,
            "researcher": r_str,
            "r_topic": r.topic if r else "-",
            "r_level": r.level if r else "-",
        })
    return rows


def _print_solution_summary(rows, best_fitness, elapsed):
    """Print a formatted solution table."""
    print("\n" + "=" * 75)
    print("  FINAL SOLUTION")
    print("=" * 75)
    hdr = (f"{'Talk':>5} | {'School':<12} | {'Location':<10} | "
           f"{'Topic':<18} | {'Researcher':<14} | {'Level':<18}")
    print(hdr)
    print("-" * 75)
    for r in rows:
        print(f"{r['talk_id']:>5} | {r['school']:<12} | {r['location']:<10} | "
              f"{r['topic']:<18} | {r['researcher']:<14} | {r['level']:<18}")
    print("=" * 75)
    print(f"  Best Fitness (penalty): {best_fitness:.2f}")
    print(f"  Elapsed time:           {elapsed:.2f}s")
    print("=" * 75)


def _print_school_coverage(rows, schools):
    """Print which schools are covered and which are not."""
    covered = {r["school"] for r in rows if r["researcher"] != "UNASSIGNED"}
    print("\n  SCHOOL COVERAGE:")
    for sid, school in schools.items():
        status = "✓ ASSIGNED" if sid in covered else "✗ UNSERVED"
        print(f"    {sid:<12} | {school.location:<10} | {school.school_type:<10} | {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CHC Talk Allocation – Practice 3")
    parser.add_argument("--schools",       type=str, default=None)
    parser.add_argument("--talks",         type=str, default=None)
    parser.add_argument("--researchers",   type=str, default=None)
    parser.add_argument("--pop-size",      type=int, default=50)
    parser.add_argument("--generations",   type=int, default=200)
    parser.add_argument("--mutation-rate", type=float, default=0.50)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--verbose",       action="store_true")
    # Synthetic instance params (used when no CSVs are given)
    parser.add_argument("--num-schools",     type=int, default=10)
    parser.add_argument("--num-talks",       type=int, default=20)
    parser.add_argument("--num-researchers", type=int, default=15)
    parser.add_argument("--prob-topics",     type=float, default=0.2,
                        help="Probability a talk requests a specific topic "
                             "(0 = all 'any', default 0.2)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n========= CHC TALK ALLOCATION – Practice 3 =========")

    # --- Load or generate data ---
    if args.schools and args.talks and args.researchers:
        print("  Loading instance from CSVs…")
        schools, talks, researchers, valid_map = load_instance(
            args.schools, args.talks, args.researchers
        )
    else:
        print(f"  No CSVs provided – generating instance "
              f"({args.num_schools} schools, {args.num_talks} talks, "
              f"{args.num_researchers} researchers, "
              f"prob_topics={args.prob_topics})…")
        schools, talks, researchers = generate_instance(
            num_schools=args.num_schools,
            num_talks=args.num_talks,
            num_researchers=args.num_researchers,
            prob_topics=args.prob_topics,
            seed=args.seed,
        )
        valid_map = build_valid_researchers_per_talk(talks, researchers, schools)

    T = len(talks)
    R = len(researchers)
    print(f"  Instance: T={T} talks | E={len(schools)} schools | R={R} researchers")

    # Summarise valid researchers per talk
    coverages = [len(v) for v in valid_map.values()]
    n_infeasible = sum(1 for c in coverages if c == 0)
    print(f"  Preprocessing: avg valid researchers/talk = "
          f"{sum(coverages) / max(len(coverages), 1):.1f} "
          f"| infeasible talks (no valid researcher) = {n_infeasible}")

    # --- Run CHC ---
    print(f"\n  Running CHC (pop={args.pop_size}, gen={args.generations}, "
          f"mutation_rate={args.mutation_rate}, seed={args.seed})…\n")
    t0 = time.time()
    best_chrom, best_fitness, convergence, elite_set, restart_gens, _final_fitnesses = chc(
        talks=talks,
        schools=schools,
        researchers=researchers,
        valid_map=valid_map,
        pop_size=args.pop_size,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        config=DEFAULT_CONFIG,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # --- Display results ---
    decoded = _decode_chromosome(best_chrom, talks, researchers, schools)
    _print_solution_summary(decoded, best_fitness, elapsed)
    _print_school_coverage(decoded, schools)

    # Display elite set (top unique solutions) for user choice
    print("\n" + "=" * 75)
    print("  ELITE SET — Top alternative solutions")
    print("=" * 75)
    for rank, (chrom, fit) in enumerate(elite_set, 1):
        n_unassigned = sum(1 for g in chrom if g == -1)
        print(f"  #{rank}: fitness={fit:.2f} | unassigned talks={n_unassigned}")
    print("=" * 75)

    # Save convergence data
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    conv_path = out_dir / "convergence.json"
    elite_info = [{"rank": i + 1, "fitness": f, "unassigned": sum(1 for g in c if g == -1)}
                  for i, (c, f) in enumerate(elite_set)]
    with open(conv_path, "w") as f:
        json.dump({
            "convergence": convergence,
            "restart_generations": restart_gens,
            "best_fitness": best_fitness,
            "elite_set": elite_info,
        }, f)
    print(f"\n  Convergence curve saved to {conv_path}")

    # Generate plots
    try:
        from plot_results import plot_all
        img_dir = Path(__file__).parent.parent / "docs" / "img"
        plot_all(str(conv_path), str(img_dir))
        print(f"  Plots saved to {img_dir}")
    except ImportError as e:
        print(f"  [INFO] plot_results module not found, skipping plots ({e})")
    except Exception as e:
        print(f"  [WARN] Could not generate plots: {e}")


if __name__ == "__main__":
    main()
