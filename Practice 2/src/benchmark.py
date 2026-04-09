"""
Performance benchmark comparing Random Search, Grid Search, and the Adaptive GA.
Generates comparative plots and persistence data for reports.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Paths and configuration
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_PATH    = SCRIPT_DIR.parent / "data" / "winequality-red.csv"
IMG_DIR      = SCRIPT_DIR.parent / "docs" / "img"
RESULTS_PATH = SCRIPT_DIR.parent / "docs" / "benchmark_results.json"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH, sep=";")
data["quality"] = (data["quality"] >= 6).astype(int)

import main as m
m.X = data.drop("quality", axis=1)
m.y = data["quality"]

# Colour palette (consistent across all plots)
C = {"rs": "#55A868", "gs": "#DD8452", "ga": "#4C72B0", "pm": "#C44E52"}

N_RUNS = 5

# ===========================================================================
# RUN THE BENCHMARK
# ===========================================================================
print("=" * 62)
print("  BENCHMARK: 5-Run Comparative Experiment")
print("=" * 62)

# --- Worker functions for parallel execution ---
def run_rs_worker(data_path):
    # Each process loads its own data to be safe
    import pandas as pd
    import main as m
    data = pd.read_csv(data_path, sep=";")
    data["quality"] = (data["quality"] >= 6).astype(int)
    m.X = data.drop("quality", axis=1)
    m.y = data["quality"]
    
    _, score, run_scores = m.random_search()
    return score, run_scores

def run_ga_worker(data_path):
    import pandas as pd
    import main as m
    data = pd.read_csv(data_path, sep=";")
    data["quality"] = (data["quality"] >= 6).astype(int)
    m.X = data.drop("quality", axis=1)
    m.y = data["quality"]
    
    _, score, history, pc_hist, pm_hist, n_evals = m.genetic_algorithm()
    return score, history, pc_hist, pm_hist, n_evals

# --- (A) Random Search (Parallel) ---
print(f"\n[RS] Running {N_RUNS} runs in parallel...", flush=True)
rs_scores     = []
all_rs_scores = []
with ProcessPoolExecutor(max_workers=min(N_RUNS, multiprocessing.cpu_count())) as executor:
    futures = [executor.submit(run_rs_worker, DATA_PATH) for _ in range(N_RUNS)]
    for i, future in enumerate(futures):
        score, run_scores = future.result()
        rs_scores.append(score)
        all_rs_scores.extend(run_scores)
        print(f"     Run {i+1} Best: {score:.6f}")

# --- (B) Grid Search (single run, deterministic) ---
print(f"\n[GS] Running Grid Search (deterministic) ...", flush=True)
_, gs_score, gs_heatmap = m.grid_search()
print(f"     Accuracy: {gs_score:.6f}")
gs_n_evals = len(gs_heatmap)

# --- (C) Genetic Algorithm (Parallel) ---
print(f"\n[GA] Running {N_RUNS} runs in parallel...", flush=True)
ga_scores      = []
ga_histories   = []
pc_histories   = []
pm_histories   = []
ga_n_evals_all = []
with ProcessPoolExecutor(max_workers=min(N_RUNS, multiprocessing.cpu_count())) as executor:
    futures = [executor.submit(run_ga_worker, DATA_PATH) for _ in range(N_RUNS)]
    for i, future in enumerate(futures):
        score, history, pc_hist, pm_hist, n_evals = future.result()
        ga_scores.append(score)
        ga_histories.append(history)
        pc_histories.append(pc_hist)
        pm_histories.append(pm_hist)
        ga_n_evals_all.append(n_evals)
        print(f"     Run {i+1} Accuracy: {score:.6f} | Evals: {n_evals}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("  SUMMARY")
print("=" * 62)
print(f"  Random Search  | Mean: {np.mean(rs_scores):.6f}  Std: {np.std(rs_scores):.6f}  Best: {max(rs_scores):.6f}")
print(f"  Grid Search    | Score: {gs_score:.6f}  (deterministic)")
print(f"  Genetic Alg.   | Mean: {np.mean(ga_scores):.6f}  Std: {np.std(ga_scores):.6f}  Best: {max(ga_scores):.6f}")

winner = max(
    [("Random Search", np.mean(rs_scores)),
     ("Grid Search",   gs_score),
     ("Genetic Alg.",  np.mean(ga_scores))],
    key=lambda x: x[1]
)
print(f"\n  WINNER (by mean accuracy): {winner[0]}  ({winner[1]:.6f})")

# Persist raw results
results = {
    "n_runs": N_RUNS,
    "random_search":     {"scores": rs_scores, "mean": np.mean(rs_scores),
                          "std": np.std(rs_scores), "all_scores": all_rs_scores},
    "grid_search":       {"score": gs_score, "heatmap": {str(k): v for k, v in gs_heatmap.items()}},
    "genetic_algorithm": {"scores": ga_scores, "mean": np.mean(ga_scores),
                          "std": np.std(ga_scores), "histories": ga_histories,
                          "pc_histories": pc_histories, "pm_histories": pm_histories,
                          "n_evals": ga_n_evals_all},
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Raw results → {RESULTS_PATH}")

# ===========================================================================
# PLOT 1 — Violin + box hybrid: final accuracy comparison
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
data_groups = [rs_scores, ga_scores]
positions   = [1, 2]
colors      = [C["rs"], C["ga"]]

parts = ax.violinplot(data_groups, positions=positions, showmedians=False, showextrema=False)
for pc, col in zip(parts["bodies"], colors):
    pc.set_facecolor(col);  pc.set_alpha(0.45)

ax.boxplot(data_groups, positions=positions, widths=0.12, patch_artist=True,
           boxprops=dict(facecolor="white", linewidth=1.5),
           medianprops=dict(color="black", linewidth=2.5),
           whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
           flierprops=dict(marker="o", markersize=5))

ax.axhline(gs_score, color=C["gs"], linestyle="--", linewidth=1.8,
           label=f"Grid Search: {gs_score:.4f} (deterministic)")
ax.set_xticks(positions)
ax.set_xticklabels(["Random Search\n(5 runs)", "Adaptive GA\n(5 runs)"], fontsize=12)
ax.set_ylabel("Mean Accuracy (5-Fold Stratified CV)", fontsize=11)
ax.set_title("Final Accuracy Distribution — 5 Independent Runs", fontsize=13, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "comparison_boxplot.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 1 (violin+box) → {p}")

# ===========================================================================
# PLOT 2 — Convergence: spaghetti lines + mean ± std band
# ===========================================================================
n_gen        = len(ga_histories[0])
hist_matrix  = np.array(ga_histories)
mean_hist    = hist_matrix.mean(axis=0);  std_hist = hist_matrix.std(axis=0)
generations  = np.arange(1, n_gen + 1)

fig, ax = plt.subplots(figsize=(11, 5))
for i, hist in enumerate(ga_histories):
    ax.plot(generations, hist, color=C["ga"], linewidth=0.9, alpha=0.35,
            label="Individual runs" if i == 0 else None)
ax.fill_between(generations, mean_hist - std_hist, mean_hist + std_hist,
                alpha=0.20, color=C["ga"])
ax.plot(generations, mean_hist, color=C["ga"], linewidth=2.5, label="GA Mean")
ax.axhline(gs_score,           color=C["gs"], linestyle="--", linewidth=1.6,
           label=f"Grid Search ({gs_score:.4f})")
ax.axhline(np.mean(rs_scores), color=C["rs"], linestyle=":",  linewidth=1.6,
           label=f"RS Mean ({np.mean(rs_scores):.4f})")
ax.set_xlabel("Generation", fontsize=11);  ax.set_ylabel("Best Accuracy (Cumulative)", fontsize=11)
ax.set_title("GA Convergence — Individual Runs & Mean ± Std (5 Runs)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "ga_convergence.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 2 (convergence spaghetti) → {p}")

# ===========================================================================
# PLOT 3 — Adaptive Pc/Pm evolution (mean ± std)
# ===========================================================================
pc_matrix = np.array(pc_histories);  pm_matrix = np.array(pm_histories)
mean_pc   = pc_matrix.mean(axis=0);  std_pc    = pc_matrix.std(axis=0)
mean_pm   = pm_matrix.mean(axis=0);  std_pm    = pm_matrix.std(axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(generations, mean_pc, color=C["ga"], linewidth=2.2, label="Mean $P_c$ (Crossover)")
ax.fill_between(generations, mean_pc - std_pc, mean_pc + std_pc, alpha=0.20, color=C["ga"])
ax.plot(generations, mean_pm, color=C["pm"], linewidth=2.2, linestyle="--",
        label="Mean $P_m$ (Mutation)")
ax.fill_between(generations, mean_pm - std_pm, mean_pm + std_pm, alpha=0.20, color=C["pm"])
ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.6, label="Equilibrium (0.5)")
ax.set_xlabel("Generation", fontsize=11);  ax.set_ylabel("Probability", fontsize=11)
ax.set_ylim(0.0, 1.0)
ax.set_title("Adaptive $P_c$ / $P_m$ Evolution — Mean ± Std over 5 Runs",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "ga_adaptive_params.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 3 (Pc/Pm evolution) → {p}")

# ===========================================================================
# PLOT 4 — Per-run bar chart: RS vs GA
# ===========================================================================
run_labels = [f"Run {i+1}" for i in range(N_RUNS)]
x = np.arange(N_RUNS);  width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_rs = ax.bar(x - width/2, rs_scores, width, label="Random Search",
                 color=C["rs"], alpha=0.8, edgecolor="white")
bars_ga = ax.bar(x + width/2, ga_scores,  width, label="Adaptive GA",
                 color=C["ga"], alpha=0.8, edgecolor="white")
ax.axhline(gs_score, color=C["gs"], linestyle="--", linewidth=1.8,
           label=f"Grid Search: {gs_score:.4f}")
for bar in bars_rs:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, color=C["rs"])
for bar in bars_ga:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, color=C["ga"])
ax.set_xticks(x);  ax.set_xticklabels(run_labels, fontsize=11)
ax.set_ylabel("Accuracy (5-Fold Stratified CV)", fontsize=11)
ax.set_title("Accuracy per Independent Run — Random Search vs Adaptive GA",
             fontsize=13, fontweight="bold")
lo = min(min(rs_scores), min(ga_scores), gs_score)
hi = max(max(rs_scores), max(ga_scores), gs_score)
ax.set_ylim(lo - 0.005, hi + 0.010)
ax.legend(fontsize=10);  ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "runs_bar_chart.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 4 (per-run bars) → {p}")

# ===========================================================================
# PLOT 5 — RS Score Histogram: distribution of all 500 sampled scores
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(all_rs_scores, bins=30, color=C["rs"], alpha=0.75, edgecolor="white",
        label=f"RS ({len(all_rs_scores)} total evaluations across {N_RUNS} runs)")
ax.axvline(np.mean(rs_scores), color=C["rs"], linestyle="--", linewidth=2,
           label=f"RS Best Mean ({np.mean(rs_scores):.4f})")
ax.axvline(gs_score,           color=C["gs"], linestyle="--", linewidth=2,
           label=f"Grid Search ({gs_score:.4f})")
ax.axvline(np.mean(ga_scores), color=C["ga"], linestyle="-",  linewidth=2,
           label=f"GA Mean ({np.mean(ga_scores):.4f})")
ax.set_xlabel("Accuracy Score", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Random Search Score Distribution — All Sampled Configurations",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "rs_score_histogram.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 5 (RS histogram) → {p}")

# ===========================================================================
# PLOT 6 — Grid Search Heatmap: n_estimators × max_depth
# ===========================================================================
n_est_vals  = [25, 75, 200]
depth_vals  = [5, 15, 25]
heat_matrix = np.array([[gs_heatmap.get((ne, md), np.nan)
                          for ne in n_est_vals] for md in depth_vals])

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heat_matrix, cmap="YlOrRd", aspect="auto",
               vmin=np.nanmin(heat_matrix), vmax=np.nanmax(heat_matrix))
ax.set_xticks(range(len(n_est_vals)));  ax.set_xticklabels(n_est_vals, fontsize=11)
ax.set_yticks(range(len(depth_vals)));  ax.set_yticklabels(depth_vals, fontsize=11)
ax.set_xlabel("n_estimators", fontsize=11);  ax.set_ylabel("max_depth", fontsize=11)
ax.set_title("Grid Search Accuracy Heatmap\n(best across other parameters)",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Accuracy")
for i in range(len(depth_vals)):
    for j in range(len(n_est_vals)):
        val = heat_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    color="black" if val < np.nanmax(heat_matrix) - 0.005 else "white",
                    fontsize=9, fontweight="bold")
fig.tight_layout()
p = IMG_DIR / "gs_heatmap.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 6 (GS heatmap) → {p}")

# ===========================================================================
# PLOT 7 — Evaluation cost comparison: unique model trainings per method
# ===========================================================================
rs_evals_per_run = 100
ga_mean_evals    = np.mean(ga_n_evals_all)

labels  = ["Random Search\n(per run)", "Grid Search\n(total, 1 run)", "Adaptive GA\n(per run, unique)"]
values  = [rs_evals_per_run, gs_n_evals, ga_mean_evals]
colors  = [C["rs"], C["gs"], C["ga"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.45)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Unique Model Evaluations", fontsize=11)
ax.set_title("Computational Cost — Unique RF Training Calls per Method\n"
             "(GA count excludes cache hits)", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = IMG_DIR / "eval_cost_comparison.png";  fig.savefig(p, dpi=150);  plt.close(fig)
print(f"  Plot 7 (eval cost) → {p}")

print("\n  ✓ Benchmark complete. 7 plots ready for the LaTeX report.")
