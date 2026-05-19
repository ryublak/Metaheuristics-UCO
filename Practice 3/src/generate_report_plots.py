"""
Generate report-quality plots for the LaTeX report.

Runs CHC with 5 different seeds on a well-posed instance, averages the
convergence curves (mean ± std), and produces two plots:
  - convergence.png   — mean convergence with shaded standard deviation band
  - population.png    — final-population fitness distribution (best-run)

Usage (from Practice 3/):
    python src/generate_report_plots.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import build_valid_researchers_per_talk
from fitness import DEFAULT_CONFIG
from chc import chc
from generate_instance import generate_instance
from plot_results import plot_all


def main():
    num_schools = 5
    num_talks = 18
    num_researchers = 30
    prob_topics = 0.05
    pop_size = 40
    generations = 150
    mutation_rate = 0.50
    num_runs = 5

    print(f"Running {num_runs} CHC iterations for averaged convergence …")

    all_convergences = []
    all_restart_gens = []
    all_final_pops = []
    all_best = []

    # Use seeds that produce feasible instances
    seeds = [8, 20, 42, 67, 91]

    for run_idx, seed in enumerate(seeds):
        schools, talks, researchers = generate_instance(
            num_schools, num_talks, num_researchers,
            prob_topics=prob_topics, seed=seed,
        )
        # Force all schools to city so travel never blocks feasibility
        for s in schools.values():
            s.location = "city"
        valid_map = build_valid_researchers_per_talk(talks, researchers, schools)

        n_inf = sum(1 for v in valid_map.values() if not v)
        if n_inf > num_talks * 0.3:
            print(f"  Run {run_idx} (seed {seed}): {n_inf} infeasible, retrying …")
            continue

        best_c, best_f, conv, elite, restarts, final_fits = chc(
            talks, schools, researchers, valid_map,
            pop_size=pop_size, max_generations=generations,
            mutation_rate=mutation_rate, config=DEFAULT_CONFIG,
            seed=seed, verbose=False,
        )

        all_convergences.append(conv)
        all_restart_gens.append(restarts)
        all_final_pops.append(final_fits)
        all_best.append((best_f, round(n_inf), seed, elite, conv, restarts))

        improvements = sum(1 for i in range(1, len(conv)) if conv[i] < conv[i - 1])
        print(f"  Run {run_idx} (seed {seed}): "
              f"{best_f:.0f} final | {improvements} improvements | "
              f"{len(restarts)} restarts | infeasible={n_inf}")

    if not all_convergences:
        print("ERROR: no valid runs. Widen parameters.")
        sys.exit(1)

    # Filter out degenerate runs (1M+ hard penalty) and pick the best
    valid_indices = [i for i, b in enumerate(all_best) if b[0] < 1e6]
    if not valid_indices:
        print("ERROR: all runs have 1M+ hard penalty. Cannot continue.")
        sys.exit(1)

    valid_best = [all_best[i] for i in valid_indices]
    valid_best.sort(key=lambda x: x[0])
    best_fitness, n_inf, best_seed, elite, best_conv, best_restarts = valid_best[0]

    # Get final population from the best run
    best_run_idx = next(i for i, b in enumerate(all_best)
                        if b[2] == best_seed)
    best_final_pop = all_final_pops[best_run_idx]

    # Pad and average only valid convergence curves
    max_len = generations + 1
    valid_convs = [all_convergences[i] for i in valid_indices]
    num_valid = len(valid_convs)
    padded = []
    for c in valid_convs:
        p = list(c)
        while len(p) < max_len:
            p.append(p[-1])
        padded.append(p[:max_len])

    arr = np.array(padded)
    avg_conv = arr.mean(axis=0).tolist()
    std_conv = arr.std(axis=0).tolist()

    print(f"\nBest run: seed {best_seed}, fitness {best_fitness:.0f}, "
          f"infeasible={n_inf}, restarts={len(best_restarts)}")
    print(f"Average convergence ({num_valid} valid runs): "
          f"{avg_conv[0]:.0f} → {avg_conv[-1]:.0f} "
          f"(±{std_conv[-1]:.0f})")

    # Save
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    conv_path = out_dir / "convergence.json"

    elite_info = [
        {"rank": i + 1, "fitness": f,
         "unassigned": sum(1 for g in c if g == -1)}
        for i, (c, f) in enumerate(elite)
    ]

    with open(conv_path, "w") as f:
        json.dump({
            "type": "averaged",
            "num_runs": num_valid,
            "generations": max_len - 1,
            "avg_convergence": avg_conv,
            "std_convergence": std_conv,
            "best_restart_generations": best_restarts,
            "best_fitness": best_fitness,
            "best_final_population": sorted(best_final_pop),
            "elite_set": elite_info,
        }, f)

    print(f"  Saved {conv_path}")

    img_dir = Path(__file__).parent.parent / "docs" / "img"
    plot_all(str(conv_path), str(img_dir))
    print(f"  Plots saved to {img_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
