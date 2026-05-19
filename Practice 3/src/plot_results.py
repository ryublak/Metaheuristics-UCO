"""
Plot generator for the Talk Allocation CHC results.

Generates two figures:
  1. convergence.png  — mean convergence ± 1σ over N runs (if available),
                        with restart markers from the best run.
  2. population.png   — sorted final-population fitness for the best run,
                        with the top-5 elite solutions highlighted.

Usage (standalone):
    python plot_results.py [convergence.json] [output_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(data: dict, output_path: str | Path) -> None:
    """Averaged convergence curve with ±1σ shaded band and restart markers."""
    generations = list(range(data.get("generations", 0) + 1))

    # Averaged data (from multi-run mode)
    avg_conv = data.get("avg_convergence")
    std_conv = data.get("std_convergence")

    if avg_conv is not None and len(avg_conv) > 1:
        avg = np.array(avg_conv[:len(generations)])
        std = np.array(std_conv[:len(generations)]) if std_conv else None

        fig, ax = plt.subplots(figsize=(9, 4.5))

        gen_vals = generations[:len(avg)]
        ax.plot(gen_vals, avg, color="steelblue", linewidth=1.8,
                label=f"Mean fitness (averaged over {data.get('num_runs', 'N')} runs)")

        if std is not None:
            ax.fill_between(gen_vals,
                            np.maximum(avg - std, 0),
                            avg + std,
                            alpha=0.18, color="steelblue",
                            label="±1σ")

        # Restart markers from the best run
        for g in data.get("best_restart_generations", []):
            if g < len(gen_vals):
                ax.axvline(x=g, color="crimson", linestyle="--",
                           linewidth=1.0, alpha=0.7)
        ax.plot([], [], color="crimson", linestyle="--", linewidth=1.0,
                alpha=0.7, label="Cataclysmic restart (best run)")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (penalty, lower is better)")
        ax.set_title("CHC Convergence — Talk Allocation")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    else:
        # Fallback: single-run convergence curve
        conv = data.get("convergence", [])
        if not conv:
            return
        generations = list(range(len(conv)))
        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.plot(generations, conv, color="steelblue", linewidth=1.5,
                marker=".", markersize=2.5, label="Best fitness")

        for g in data.get("restart_generations", []):
            ax.axvline(x=g, color="crimson", linestyle="--",
                       linewidth=1.0, alpha=0.7)
        ax.plot([], [], color="crimson", linestyle="--", linewidth=1.0,
                alpha=0.7, label="Cataclysmic restart")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (penalty)")
        ax.set_title("CHC Convergence — Talk Allocation")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_population(data: dict, output_path: str | Path) -> None:
    """
    Sorted bar chart of the final population's fitness values for the best run.
    The top 5 elite solutions are highlighted in dark blue; the rest in grey.
    """
    pop_fits = data.get("best_final_population", [])
    if not pop_fits:
        return

    sorted_fits = sorted(pop_fits)
    n = len(sorted_fits)

    fig, ax = plt.subplots(figsize=(8, 4))

    x = list(range(1, n + 1))

    # Draw all bars grey, then over-draw elite in blue
    bars = ax.bar(x, sorted_fits, color="lightgray", edgecolor="gray",
                  linewidth=0.4, zorder=2)

    for i in range(min(5, n)):
        bars[i].set_color("steelblue")
        bars[i].set_edgecolor("steelblue")
        bars[i].set_linewidth(1.0)

    # Mark elite region
    ax.axvspan(0.5, min(5, n) + 0.5, alpha=0.06, color="steelblue", zorder=1)

    ax.set_xlabel("Individual (sorted by fitness)")
    ax.set_ylabel("Fitness (penalty)")
    ax.set_title("Final Population — Best Run Diversity")
    ax.legend([plt.Rectangle((0,0),1,1, color="steelblue"),
               plt.Rectangle((0,0),1,1, color="lightgray", ec="gray")],
              ["Elite (top 5)", "Rest of population"],
              fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all(data_path: str | Path = "data/convergence.json",
             output_dir: str | Path = "docs/img") -> None:
    """Load convergence data and generate all plots."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    plot_convergence(data, output_dir / "convergence.png")
    plot_population(data, output_dir / "population.png")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/convergence.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
    plot_all(data_path, output_dir)
