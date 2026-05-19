"""
Scalability and realism benchmark for the CHC Talk Allocation algorithm.

Generates five plots in docs/img/:
  1. scalability_time.png          — execution time vs instance size
  2. convergence_realistic.png     — convergence on realistic data (no preprocessing)
  3. feasibility.png               — % runs with zero hard penalties per size
  4. comparison_preprocessing.png  — realistic vs forced city (same seed)
  5. fitness_breakdown.png         — soft-constraint penalty composition

Realistic mode uses the professor's generator as-is (province schools,
prob_topics=0.2, no forced city, no level balancing).

Usage (from Practice 3/):
    python src/benchmark_plots.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import School, Talk, Researcher
from data_loader import build_valid_researchers_per_talk
from fitness import compute_fitness, compute_penalty_breakdown, DEFAULT_CONFIG
from chc import chc
from generate_instance import generate_instance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Single-size runner
# ---------------------------------------------------------------------------

def run_one_instance(schools, talks, researchers, pop, gens, seed):
    """Run CHC on a pre-built instance. Returns (time_s, fitness, elite)."""
    valid_map = build_valid_researchers_per_talk(talks, researchers, schools)
    t0 = time.time()
    _, best_f, conv, elite, _, _, gen_best = chc(
        talks, schools, researchers, valid_map,
        pop_size=pop, max_generations=gens, mutation_rate=0.65,
        config=DEFAULT_CONFIG, seed=seed, verbose=False,
    )
    elapsed = time.time() - t0
    return elapsed, best_f, elite


# ---------------------------------------------------------------------------
# Plot 1 — scalability (time vs size)
# ---------------------------------------------------------------------------

def plot_scalability_time(results, output_path):
    sizes = sorted(results.keys())
    means = [np.mean(results[s]["times"]) for s in sizes]
    stds = [np.std(results[s]["times"]) for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([str(s) for s in sizes], means, yerr=stds,
                  color="steelblue", edgecolor="white", linewidth=0.8,
                  capsize=5, width=0.55)
    ax.bar_label(bars, labels=[f"{m:.1f}s" for m in means], fontsize=8,
                 padding=3)
    ax.set_xlabel("Instance size (T = number of talks)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("CHC Execution Time vs Instance Size")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — fitness consistency across runs (T=60 realistic)
# ---------------------------------------------------------------------------

def plot_fitness_consistency(fitnesses, output_path):
    """Bar chart showing final fitness for each run on a T=60 realistic instance."""
    n = len(fitnesses)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["steelblue" if f < 1e6 else "crimson" for f in fitnesses]
    bars = ax.bar(range(1, n + 1), fitnesses, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.55)
    for i, (bar, f) in enumerate(zip(bars, fitnesses)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(fitnesses) * 0.02,
                f"{f:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Final fitness (penalty)")
    ax.set_title(f"Fitness Consistency — T=60 Realistic Instance ({n} runs)")
    ax.set_xticks(range(1, n + 1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — feasibility by size
# ---------------------------------------------------------------------------

def plot_feasibility(results, output_path):
    sizes = sorted(results.keys())
    total_runs = {s: len(results[s]["times"]) for s in sizes}
    clean_runs = {s: sum(1 for f in results[s]["fits"] if f < 1e6)
                  for s in sizes}
    dirty_runs = {s: total_runs[s] - clean_runs[s] for s in sizes}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_labels = [str(s) for s in sizes]
    ax.bar(x_labels, [clean_runs[s] for s in sizes],
           color="steelblue", label="Feasible (fitness < 1M)", width=0.55)
    ax.bar(x_labels, [dirty_runs[s] for s in sizes],
           bottom=[clean_runs[s] for s in sizes],
           color="crimson", alpha=0.7, label="Hard penalty (fitness ≥ 1M)",
           width=0.55)
    ax.set_xlabel("Instance size (T = number of talks)")
    ax.set_ylabel("Number of runs")
    ax.set_title("Feasibility by Instance Size")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — realistic vs preprocessed (forced city)
# ---------------------------------------------------------------------------

def plot_comparison_preprocessing(output_path):
    seed = 42
    num_schools, num_talks, num_researchers = 10, 30, 35
    pop, gens = 80, 300

    s_r, t_r, r_r = generate_instance(
        num_schools, num_talks, num_researchers,
        prob_topics=0.2, seed=seed, balance_levels=False)
    valid_r = build_valid_researchers_per_talk(t_r, r_r, s_r)
    _, _, conv_r, _, restarts_r, _, _ = chc(t_r, s_r, r_r, valid_r, pop, gens,
                                mutation_rate=0.65, config=DEFAULT_CONFIG,
                                seed=seed, verbose=False)

    s_p, t_p, r_p = generate_instance(
        num_schools, num_talks, num_researchers,
        prob_topics=0.0, seed=seed, balance_levels=False)
    for s in s_p.values():
        s.location = "city"
    valid_p = build_valid_researchers_per_talk(t_p, r_p, s_p)
    _, _, conv_p, _, restarts_p, _, _ = chc(t_p, s_p, r_p, valid_p, pop, gens,
                                mutation_rate=0.65, config=DEFAULT_CONFIG,
                                seed=seed, verbose=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

    gen_r = list(range(len(conv_r)))
    gen_p = list(range(len(conv_p)))

    ax1.plot(gen_r, conv_r, color="crimson", linewidth=1.2,
             label="Realistic (province, topics=0.2)")
    for g in restarts_r:
        ax1.axvline(x=g, color="crimson", linestyle="--", linewidth=0.8,
                    alpha=0.5)
    ax1.set_ylabel("Avg feasible fitness", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Realistic instance", fontsize=10)

    ax2.plot(gen_p, conv_p, color="steelblue", linewidth=1.2,
             label="Preprocessed (city, any topic)")
    for g in restarts_p:
        ax2.axvline(x=g, color="steelblue", linestyle="--", linewidth=0.8,
                    alpha=0.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Avg feasible fitness", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Preprocessed instance", fontsize=10)

    fig.suptitle("Impact of Preprocessing — Same Seed", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — fitness breakdown
# ---------------------------------------------------------------------------

def plot_fitness_breakdown(output_path):
    seed = 42
    schools, talks, researchers = generate_instance(
        10, 30, 35, prob_topics=0.0, seed=seed, balance_levels=False)
    for s in schools.values():
        s.location = "city"
    valid_map = build_valid_researchers_per_talk(talks, researchers, schools)
    r_ids = list(researchers.keys())

    best_c, best_f, _, _, _, _, _gen_best = chc(
        talks, schools, researchers, valid_map,
        pop_size=80, max_generations=300, mutation_rate=0.65,
        config=DEFAULT_CONFIG, seed=seed, verbose=False,
    )

    components = compute_penalty_breakdown(
        best_c, talks, schools, researchers, DEFAULT_CONFIG,
    )

    # If best_f is larger than sum, remaining is "crossover/startup variance"
    # Otherwise use best_f directly
    total_accounted = sum(components.values())
    if total_accounted > 0 and best_f > 0:
        components = {k: v * best_f / total_accounted for k, v in components.items()
                      if v > 0}

    labels = list(components.keys())
    values = list(components.values())
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f"Fitness Composition (total = {best_f:.0f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick run")
    args, _ = parser.parse_known_args()
    fast = fast or args.fast

    print("=== CHC Benchmark — Scalability & Realism ===\n")

    # Instance sizes: (T, E, R, pop, gens)
    # Lower ratios (pop/T ≈ 2-3) so convergence is visible, not instant
    if fast:
        configs = [
            (15,  7, 20,  40, 50),
            (30, 10, 35,  40, 50),
        ]
        n_seeds = 1
    else:
        configs = [
            (15,  7, 20,  80, 300),
            (30, 10, 35,  80, 300),
            (45, 12, 50, 100, 300),
            (60, 15, 65, 120, 300),
        ]
        n_seeds = 3

    # --- Plot 1 & 3: scalability + feasibility ---
    results = {}
    realistic_fitnesses = []

    for T, E, R, pop, gens in configs:
        print(f"T={T:3d}  E={E:2d}  R={R:3d}  pop={pop:3d}  gens={gens:3d} ...")

        times, fits = [], []
        for seed in range(1, n_seeds + 1):
            s, t, r = generate_instance(E, T, R, prob_topics=0.2, seed=seed,
                                         balance_levels=False)
            elapsed, best_f, _ = run_one_instance(s, t, r, pop, gens, seed)
            times.append(elapsed)
            fits.append(best_f)

        # Capture fitnesses for the largest instance (T=60, 5 runs)
        if T == 60:
            for seed in range(1, 6):
                s, t, r = generate_instance(E, T, R, prob_topics=0.2,
                                              seed=seed, balance_levels=False)
                elapsed, best_f, _ = run_one_instance(s, t, r, pop, gens, seed)
                realistic_fitnesses.append(best_f)

        n_clean = sum(1 for f in fits if f < 1e6)
        mean_t = np.mean(times)
        print(f"       time={mean_t:.2f}s (±{np.std(times):.2f})  "
              f"fitness={np.mean(fits):.0f}  clean={n_clean}/{len(fits)}")
        results[T] = {"times": times, "fits": fits}

    img_dir = Path(__file__).parent.parent / "docs" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots …")

    # Plot 1
    plot_scalability_time(results, img_dir / "scalability_time.png")
    print("  [1/5] scalability_time.png")

    # Plot 2
    if realistic_fitnesses:
        plot_fitness_consistency(realistic_fitnesses,
                                 img_dir / "fitness_consistency.png")
        print("  [2/5] fitness_consistency.png")
    else:
        print("  [2/5] SKIPPED (no realistic data)")

    # Plot 3
    plot_feasibility(results, img_dir / "feasibility.png")
    print("  [3/5] feasibility.png")

    # Plot 4
    plot_comparison_preprocessing(img_dir / "comparison_preprocessing.png")
    print("  [4/5] comparison_preprocessing.png")

    # Plot 5
    plot_fitness_breakdown(img_dir / "fitness_breakdown.png")
    print("  [5/5] fitness_breakdown.png")

    print(f"\nDone — {len(list(img_dir.glob('*.png')))} plots in {img_dir}")


if __name__ == "__main__":
    main()
