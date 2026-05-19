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

from data_generator_talks11f.schools_functions import generate_random_school
from data_generator_talks11f.talks_functions import (
    generate_random_topic,
    generate_random_talk_level,
)
from data_generator_talks11f.proposed_talks_functions import (
    generate_random_repeat_talk,
    generate_random_travelling,
    generate_random_first_participation,
    generate_random_previous_talk_province,
)
from models import School, Talk, Researcher
from data_loader import build_valid_researchers_per_talk
from fitness import compute_fitness, DEFAULT_CONFIG
from chc import chc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Realistic instance generator (professor's logic, no preprocessing)
# ---------------------------------------------------------------------------

def generate_realistic(num_schools, num_talks, num_researchers,
                       prob_topics=0.2, seed=None):
    """Generate using the professor's original logic — province schools,
       travel constraints, and specific topics are all active."""
    if seed is not None:
        np.random.seed(seed)

    schools = {}
    for i in range(num_schools):
        s = generate_random_school()
        sid = f"school{i + 1}"
        schools[sid] = School(
            school_id=sid, location=s["location"],
            disadvantaged_area=s["disadvantaged_area"],
            school_type=s["school_type"], first_year=s["first_year"],
        )

    school_ids = list(schools.keys())

    talks = []
    seen = set()
    # Guarantee one talk per school
    for sid in school_ids:
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        talks.append(Talk(talk_id=len(talks), topic=topic, level=level, school_id=sid))

    max_attempts = num_talks * 10
    attempts = 0
    while len(talks) < num_talks and attempts < max_attempts:
        sid = np.random.choice(school_ids)
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        key = (sid, topic, level)
        if key not in seen:
            seen.add(key)
            talks.append(Talk(talk_id=len(talks), topic=topic, level=level, school_id=sid))
        attempts += 1
    while len(talks) < num_talks:
        sid = np.random.choice(school_ids)
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        talks.append(Talk(talk_id=len(talks), topic=topic, level=level, school_id=sid))

    province_schools = [s for s in school_ids if schools[s].location == "province"]
    city_schools = [s for s in school_ids if schools[s].location == "city"]

    researchers = {}
    for i in range(num_researchers):
        repeat = generate_random_repeat_talk()
        can_travel = generate_random_travelling()
        first_part = generate_random_first_participation()
        prev_prov = generate_random_previous_talk_province()
        topic = generate_random_topic()
        level = generate_random_talk_level()

        if prev_prov and province_schools:
            prev_school = str(np.random.choice(province_schools))
        elif city_schools:
            prev_school = str(np.random.choice(city_schools))
            prev_prov = False
        else:
            prev_school = str(np.random.choice(school_ids))
            prev_prov = False

        max_t = 2 if repeat else 1
        researchers[f"researcher{i + 1}"] = Researcher(
            researcher_id=f"researcher{i + 1}", topic=topic, level=level,
            can_travel=can_travel, first_participation=first_part,
            previous_talk_province=prev_prov, previous_school=prev_school,
            max_talks=max_t,
        )

    return schools, talks, researchers


# ---------------------------------------------------------------------------
# Single-size runner
# ---------------------------------------------------------------------------

def run_one_instance(schools, talks, researchers, pop, gens, seed):
    """Run CHC on a pre-built instance. Returns (time_s, fitness, elite)."""
    valid_map = build_valid_researchers_per_talk(talks, researchers, schools)
    t0 = time.time()
    _, best_f, conv, elite, _, _ = chc(
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
# Plot 2 — realistic convergence
# ---------------------------------------------------------------------------

def plot_realistic_convergence(all_convs, output_path):
    max_len = max(len(c) for c in all_convs)
    padded = []
    for c in all_convs:
        p = list(c)
        while len(p) < max_len:
            p.append(p[-1])
        padded.append(p[:max_len])
    arr = np.array(padded)
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    gen_vals = list(range(len(avg)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(gen_vals, avg, color="crimson", linewidth=1.6,
            label=f"Mean fitness (realistic, T=30, {len(all_convs)} runs)")
    ax.fill_between(gen_vals, np.maximum(avg - std, 0), avg + std,
                    alpha=0.15, color="crimson", label="±1σ")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (penalty)")
    ax.set_title("CHC Convergence — Realistic Instance (province + travel)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
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
    pop, gens = 200, 600

    # Realistic (no preprocessing)
    s_r, t_r, r_r = generate_realistic(num_schools, num_talks,
                                       num_researchers, prob_topics=0.2,
                                       seed=seed)
    valid_r = build_valid_researchers_per_talk(t_r, r_r, s_r)
    _, _, conv_r, _, _, _ = chc(t_r, s_r, r_r, valid_r, pop, gens,
                                mutation_rate=0.65, config=DEFAULT_CONFIG,
                                seed=seed, verbose=False)

    # Preprocessed (forced city + prob_topics=0)
    s_p, t_p, r_p = generate_realistic(num_schools, num_talks,
                                       num_researchers, prob_topics=0.0,
                                       seed=seed)
    for s in s_p.values():
        s.location = "city"
    valid_p = build_valid_researchers_per_talk(t_p, r_p, s_p)
    _, _, conv_p, _, _, _ = chc(t_p, s_p, r_p, valid_p, pop, gens,
                                mutation_rate=0.65, config=DEFAULT_CONFIG,
                                seed=seed, verbose=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    gen_r = list(range(len(conv_r)))
    gen_p = list(range(len(conv_p)))
    ax.plot(gen_r, conv_r, color="crimson", linewidth=1.4,
            label="Realistic (province + travel + specific topics)")
    ax.plot(gen_p, conv_p, color="steelblue", linewidth=1.4,
            label="Preprocessed (forced city + any topic)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (penalty)")
    ax.set_title("Impact of Preprocessing — Same Instance")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — fitness breakdown
# ---------------------------------------------------------------------------

def plot_fitness_breakdown(output_path):
    seed = 42
    schools, talks, researchers = generate_realistic(10, 30, 35,
                                                     prob_topics=0.0, seed=seed)
    for s in schools.values():
        s.location = "city"
    valid_map = build_valid_researchers_per_talk(talks, researchers, schools)
    r_ids = list(researchers.keys())

    best_c, best_f, _, _, _, _ = chc(
        talks, schools, researchers, valid_map,
        pop_size=200, max_generations=600, mutation_rate=0.65,
        config=DEFAULT_CONFIG, seed=seed, verbose=False,
    )

    resolved = [r_ids[idx] if idx != -1 and idx < len(r_ids) else None
                for idx in best_c]
    n_assigned = sum(1 for r in resolved if r is not None)
    n_unassigned = sum(1 for r in resolved if r is None)

    # Count soft penalties manually
    researcher_used = set(r for r in resolved if r is not None)
    unused_penalty = 0
    for r in researchers.values():
        if r.researcher_id not in researcher_used:
            if r.first_participation:
                unused_penalty += DEFAULT_CONFIG["w_first_year_researcher"]
            if r.can_travel:
                unused_penalty += DEFAULT_CONFIG["w_travel_researcher"]

    hist_penalty = 0
    for t_id, r_str in enumerate(resolved):
        if r_str is None:
            continue
        r = researchers.get(r_str)
        if r is None:
            continue
        school = schools[talks[t_id].school_id]
        if r.previous_school and r.previous_school == talks[t_id].school_id:
            hist_penalty += DEFAULT_CONFIG["w_same_school_as_last_year"]
        if r.previous_talk_province and school.location == "province":
            hist_penalty += DEFAULT_CONFIG["w_same_province_as_last_year"]

    assigned_schools = {talks[i].school_id for i, r in enumerate(resolved)
                        if r is not None}
    unserved_penalty = 0
    for s_id, s in schools.items():
        if s_id in assigned_schools:
            continue
        if s.first_year:
            unserved_penalty += DEFAULT_CONFIG["w_first_year_school"]
        if s.location == "province":
            unserved_penalty += DEFAULT_CONFIG["w_province_school"]
        if s.disadvantaged_area:
            unserved_penalty += DEFAULT_CONFIG["w_disadvantaged"]
        if s.school_type == "public":
            unserved_penalty += DEFAULT_CONFIG["w_public_institution"]
        elif s.school_type == "concerted":
            unserved_penalty += DEFAULT_CONFIG["w_concerted"]

    unassigned_penalty = n_unassigned * DEFAULT_CONFIG["w_unassigned_talk"]

    # Ensure all components are non-negative
    components = {
        "Unused researchers": max(0, unused_penalty),
        "Historical": max(0, hist_penalty),
        "Unserved schools (soft)": max(0, unserved_penalty),
        "Unassigned talks": max(0, unassigned_penalty),
    }

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

def main():
    print("=== CHC Benchmark — Scalability & Realism ===\n")

    configs = [
        (15,  7, 22),
        (30, 10, 40),
        (60, 15, 75),
    ]
    n_seeds = 3

    # --- Plot 1 & 3: scalability + feasibility ---
    results = {}
    all_realistic_convs = []

    for T, E, R in configs:
        pop = max(200, T * 5)
        gens = min(max(500, T * 10), 1000)  # capped at 1000 for time
        print(f"T={T:3d}  E={E:2d}  R={R:3d}  pop={pop:4d}  gens={gens:4d} ...")

        times, fits = [], []
        for seed in range(1, n_seeds + 1):
            s, t, r = generate_realistic(E, T, R, prob_topics=0.2, seed=seed)
            elapsed, best_f, _ = run_one_instance(s, t, r, pop, gens, seed)
            times.append(elapsed)
            fits.append(best_f)
            if T == 30:
                # Capture one convergence for realistic plot
                valid = build_valid_researchers_per_talk(t, r, s)
                _, _, conv, _, _, _ = chc(t, s, r, valid, pop, gens,
                                          mutation_rate=0.65,
                                          config=DEFAULT_CONFIG,
                                          seed=seed, verbose=False)
                all_realistic_convs.append(conv)

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
    if all_realistic_convs:
        plot_realistic_convergence(all_realistic_convs,
                                   img_dir / "convergence_realistic.png")
        print("  [2/5] convergence_realistic.png")
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
