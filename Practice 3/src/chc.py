"""
CHC Metaheuristic for the Talk Allocation Problem
===================================================
CHC: Cross-generational elitist selection, Heterogeneous recombination (HUX),
     Cataclysmic mutation restart.

Reference: Eshelman (1991) "The CHC Adaptive Search Algorithm"
"""

import random
import math
import copy
from typing import Dict, List, Optional, Tuple

from models import School, Talk, Researcher
from fitness import compute_fitness, repair_gene, compute_infeasible_schools, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Chromosome = List[int]   # length T; gene i in {-1, 0 .. R-1}


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def _build_random_chromosome(
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    researcher_ids: List[str],
    researchers: Dict[str, Researcher],
    schools: Dict[str, School] | None = None,
) -> Chromosome:
    """Generate one random chromosome, respecting valid_map and capacity.

    Two-pass initialisation:
      1. Guarantee at least one talk per school (avoids the 1M hard penalty).
      2. Fill remaining talks greedily.
    """
    T = len(talks)
    chrom: Chromosome = [-1] * T
    usage: Dict[str, int] = {}

    def _can_take(r_str: str) -> bool:
        r = researchers.get(r_str)
        return r is not None and usage.get(r_str, 0) < r.max_talks

    # First pass: one talk per school (if schools dict provided)
    if schools is not None:
        school_ids = list(schools.keys())
        random.shuffle(school_ids)
        for sid in school_ids:
            school_talks = [t for t in talks if t.school_id == sid]
            random.shuffle(school_talks)
            for t in school_talks:
                if chrom[t.talk_id] != -1:
                    continue
                candidates = [r for r in valid_map.get(t.talk_id, [])
                              if _can_take(r)]
                if not candidates:
                    continue
                r_str = random.choice(candidates)
                chrom[t.talk_id] = researcher_ids.index(r_str)
                usage[r_str] = usage.get(r_str, 0) + 1
                break  # one talk per school is enough

    # Second pass: fill remaining talks respecting capacity
    for talk in talks:
        if chrom[talk.talk_id] != -1:
            continue
        candidates = [r for r in valid_map.get(talk.talk_id, [])
                      if _can_take(r)]
        if not candidates:
            continue
        random.shuffle(candidates)
        r_str = candidates[0]
        chrom[talk.talk_id] = researcher_ids.index(r_str)
        usage[r_str] = usage.get(r_str, 0) + 1

    return chrom


def hamming_distance(a: Chromosome, b: Chromosome) -> int:
    """Count positions where two chromosomes differ."""
    return sum(x != y for x, y in zip(a, b))


def initialise_population(
    pop_size: int,
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    researcher_ids: List[str],
    researchers: Dict[str, Researcher],
    schools: Dict[str, School] | None = None,
) -> List[Chromosome]:
    """
    Create an initial population of pop_size diverse chromosomes.
    Attempts basic diversity enforcement but does not block completeness.
    """
    population: List[Chromosome] = []
    attempts = 0
    max_attempts = pop_size * 20
    min_hamming = max(1, len(talks) // 10)  # at least 10% different genes

    while len(population) < pop_size and attempts < max_attempts:
        candidate = _build_random_chromosome(talks, valid_map, researcher_ids, researchers, schools)
        too_similar = any(hamming_distance(candidate, existing) < min_hamming
                          for existing in population)
        if not too_similar:
            population.append(candidate)
        attempts += 1

    # Fill remainder without diversity constraint if needed
    while len(population) < pop_size:
        population.append(_build_random_chromosome(talks, valid_map, researcher_ids, researchers, schools))

    return population


# ---------------------------------------------------------------------------
# Repair operator
# ---------------------------------------------------------------------------

def repair_chromosome(
    chromosome: Chromosome,
    researcher_ids: List[str],
    researchers: Dict[str, Researcher],
    valid_map: Dict[int, List[str]],
) -> Chromosome:
    """
    Repair a chromosome that may violate the overallocation hard constraint
    or leave talks unassigned when a valid researcher exists.

    Iterates until no more changes are made, with a 10-iteration cap
    to resolve cascading overallocations.
    """
    chrom = copy.copy(chromosome)

    for iteration in range(10):
        changed = False

        # --- Phase 1: fix overallocation ---
        counts: Dict[str, int] = {}
        for pos, r_idx in enumerate(chrom):
            if r_idx == -1:
                continue
            r_str = researcher_ids[r_idx]
            counts[r_str] = counts.get(r_str, 0) + 1

        for r_str, count in counts.items():
            r = researchers.get(r_str)
            if r is None or count <= r.max_talks:
                continue
            excess = count - r.max_talks
            positions = [
                i for i, idx in enumerate(chrom)
                if idx != -1 and researcher_ids[idx] == r_str
            ]
            random.shuffle(positions)
            for pos in positions[:excess]:
                candidates = [v for v in valid_map.get(pos, []) if v != r_str]
                if candidates:
                    new_r = random.choice(candidates)
                    chrom[pos] = researcher_ids.index(new_r)
                else:
                    chrom[pos] = -1
            changed = True

        # --- Phase 2: fill unassigned talks if possible ---
        usage: Dict[str, int] = {}
        for idx in chrom:
            if idx != -1:
                r_str = researcher_ids[idx]
                usage[r_str] = usage.get(r_str, 0) + 1

        for pos, r_idx in enumerate(chrom):
            if r_idx != -1:
                continue
            candidates = valid_map.get(pos, [])
            random.shuffle(candidates)
            for r_str in candidates:
                r = researchers.get(r_str)
                if r and usage.get(r_str, 0) < r.max_talks:
                    chrom[pos] = researcher_ids.index(r_str)
                    usage[r_str] = usage.get(r_str, 0) + 1
                    changed = True
                    break

        if not changed:
            break

    return chrom


def _force_desassign_overallocated(
    chromosome: Chromosome,
    researcher_ids: List[str],
    researchers: Dict[str, Researcher],
) -> Chromosome:
    """
    Force any overallocated researcher's excess talks to -1.
    Guarantees the returned chromosome has zero overallocation,
    at the cost of some talks being left unassigned.
    """
    chrom = copy.copy(chromosome)
    counts: Dict[str, int] = {}
    for idx in chrom:
        if idx == -1:
            continue
        r_str = researcher_ids[idx]
        counts[r_str] = counts.get(r_str, 0) + 1

    for r_str, count in counts.items():
        r = researchers.get(r_str)
        if r is None or count <= r.max_talks:
            continue
        excess = count - r.max_talks
        positions = [
            i for i, idx in enumerate(chrom)
            if idx != -1 and researcher_ids[idx] == r_str
        ]
        for pos in positions[-excess:]:
            chrom[pos] = -1

    return chrom


# ---------------------------------------------------------------------------
# HUX Crossover
# ---------------------------------------------------------------------------

def hux_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """
    Half-Uniform Crossover (HUX):
    Identify all differing positions; swap exactly half of them (random selection).
    Returns two offspring.
    """
    diff_positions = [i for i, (a, b) in enumerate(zip(parent1, parent2)) if a != b]

    if len(diff_positions) < 2:
        return copy.copy(parent1), copy.copy(parent2)

    half = len(diff_positions) // 2
    swap_positions = set(random.sample(diff_positions, half))

    child1 = copy.copy(parent1)
    child2 = copy.copy(parent2)
    for pos in swap_positions:
        child1[pos], child2[pos] = child2[pos], child1[pos]

    return child1, child2


# ---------------------------------------------------------------------------
# Cataclysmic Mutation (Restart)
# ---------------------------------------------------------------------------

def cataclysmic_mutation(
    elite_survivors: List[Chromosome],
    pop_size: int,
    base_mutation_rate: float,
    stale_restarts: int,
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    researcher_ids: List[str],
    researchers: Dict[str, Researcher],
) -> List[Chromosome]:
    """
    Restart: keep up to `n_survivors` elite individuals and fill the rest
    by mutating a fraction of genes. The mutation rate adapts based on
    how many consecutive restarts failed to produce improvements.

    Survivors are distributed evenly; mutants are repaired with the full
    repair operator (overallocation + fill -1).
    """
    n_survivors = min(len(elite_survivors), max(1, pop_size // 8))
    survivors = elite_survivors[:n_survivors]

    # Adaptive mutation rate: increases with stale restarts (capped at 0.8)
    adaptive_rate = min(base_mutation_rate + 0.05 * stale_restarts, 0.8)

    T = len(talks)
    n_mutate = max(1, int(adaptive_rate * T))

    new_population: List[Chromosome] = [copy.copy(s) for s in survivors]
    slots_per_survivor = (pop_size - len(survivors)) // len(survivors)
    remainder = (pop_size - len(survivors)) % len(survivors)

    for idx, survivor in enumerate(survivors):
        n_mutants = slots_per_survivor + (1 if idx < remainder else 0)
        for _ in range(n_mutants):
            mutant = copy.copy(survivor)
            positions = random.sample(range(T), min(n_mutate, T))
            for pos in positions:
                mutant[pos] = repair_gene(pos, valid_map, researcher_ids)
            mutant = repair_chromosome(mutant, researcher_ids, researchers, valid_map)
            new_population.append(mutant)

    return new_population


# ---------------------------------------------------------------------------
# CHC Main Loop
# ---------------------------------------------------------------------------

def chc(
    talks: List[Talk],
    schools: Dict[str, School],
    researchers: Dict[str, Researcher],
    valid_map: Dict[int, List[str]],
    pop_size: int = 50,
    max_generations: int = 200,
    mutation_rate: float = 0.50,
    config: Optional[Dict] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    elite_size: int = 5,
) -> Tuple[Chromosome, float, List[float], List[Tuple[Chromosome, float]], List[int], List[float], List[float]]:
    """
    Run the CHC algorithm for talk allocation.

    Args:
        talks:          ordered list of Talk objects (length T).
        schools:        dict school_id -> School.
        researchers:    dict researcher_id -> Researcher.
        valid_map:      talk_id -> [valid_researcher_id_strings].
        pop_size:       population size (even number recommended).
        max_generations: stopping criterion.
        mutation_rate:  fraction of genes mutated during restart (default 50%).
        config:         fitness weight config dict.
        seed:           random seed for reproducibility.
        verbose:        print progress.
        elite_size:     number of top unique solutions to keep in the elite set.

    Returns:
        (best_chromosome, best_fitness, avg_fitness_curve, elite_list,
         restart_generations, final_fitness_values, gen_best_fitness)
    """
    if seed is not None:
        random.seed(seed)

    if config is None:
        config = DEFAULT_CONFIG

    researcher_ids = list(researchers.keys())
    T = len(talks)

    # Initial Hamming threshold:  d0 = T / 4  (standard CHC initialisation)
    threshold = T // 4

    # Initialise population
    population = initialise_population(pop_size, talks, valid_map, researcher_ids, researchers, schools)

    # Repair initial population so all individuals are free of hard penalties
    population = [repair_chromosome(c, researcher_ids, researchers, valid_map)
                  for c in population]

    # Compute structurally infeasible schools (all their talks have V(t) = ∅)
    infeasible_schools = compute_infeasible_schools(talks, schools, valid_map)

    # Evaluate
    def evaluate(chrom: Chromosome) -> float:
        return compute_fitness(chrom, talks, schools, researchers, valid_map, config, infeasible_schools)

    fitness_values = [evaluate(c) for c in population]

    # Track best
    best_idx = min(range(pop_size), key=lambda i: fitness_values[i])
    best_chrom = copy.copy(population[best_idx])
    best_fitness = fitness_values[best_idx]

    # Elite set: top elite_size unique individuals (best solution for user choice)
    elite_set: List[Tuple[Chromosome, float]] = []
    seen: set = set()
    for idx in sorted(range(pop_size), key=lambda i: fitness_values[i]):
        t = tuple(population[idx])
        if t not in seen:
            elite_set.append((copy.copy(population[idx]), fitness_values[idx]))
            seen.add(t)
            if len(elite_set) >= elite_size:
                break

    feasible0 = [f for f in fitness_values if f < 1_000_000]
    convergence: List[float] = [(sum(feasible0) / len(feasible0)) if feasible0
                                else (sum(fitness_values) / len(fitness_values))]
    gen_best: List[float] = [best_fitness]
    restarts = 0
    stale_restarts = 0
    stagnation_count = 0
    restart_generations: List[int] = []
    stagnation_limit = max(10, T // 2)

    for gen in range(max_generations):
        # Shuffle population for pairing
        indices = list(range(pop_size))
        random.shuffle(indices)
        pairs = [(indices[i], indices[i + 1]) for i in range(0, pop_size - 1, 2)]

        new_individuals: List[Tuple[Chromosome, float]] = []

        for i1, i2 in pairs:
            p1, p2 = population[i1], population[i2]
            d = hamming_distance(p1, p2)

            # Incest prevention: only mate if Hamming distance > threshold
            if d > threshold:
                c1, c2 = hux_crossover(p1, p2)
                c1 = repair_chromosome(c1, researcher_ids, researchers, valid_map)
                c2 = repair_chromosome(c2, researcher_ids, researchers, valid_map)
                f1, f2 = evaluate(c1), evaluate(c2)

                # Child survives only if strictly better than both parents
                if f1 < fitness_values[i1] or f1 < fitness_values[i2]:
                    new_individuals.append((c1, f1))
                if f2 < fitness_values[i1] or f2 < fitness_values[i2]:
                    new_individuals.append((c2, f2))

        if new_individuals:
            # Elitist replacement: combine old + new, keep top pop_size
            combined = list(zip(population, fitness_values)) + new_individuals
            combined.sort(key=lambda x: x[1])
            population = [c for c, _ in combined[:pop_size]]
            fitness_values = [f for _, f in combined[:pop_size]]
            # Gradual threshold reset: raise gradually rather than jumping to T//4
            threshold = min(T // 4, max(threshold + 1, threshold + T // 8))
        else:
            # No children improved parents: decrease threshold
            threshold -= 1
            if verbose:
                print(f"  Gen {gen:4d} | threshold={threshold} | best_fit={best_fitness:.2f}")

        # Update global best
        gen_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
        gen_best.append(fitness_values[gen_best_idx])
        if fitness_values[gen_best_idx] < best_fitness:
            best_fitness = fitness_values[gen_best_idx]
            best_chrom = copy.copy(population[gen_best_idx])
            stagnation_count = 0
            stale_restarts = 0
        else:
            stagnation_count += 1

        # Update elite set with unique top individuals
        combined_elite = elite_set + list(zip(population, fitness_values))
        combined_elite.sort(key=lambda x: x[1])
        elite_set.clear()
        seen.clear()
        for c, f in combined_elite:
            t = tuple(c)
            if t not in seen:
                elite_set.append((copy.copy(c), f))
                seen.add(t)
                if len(elite_set) >= elite_size:
                    break

        # Cataclysmic restart when threshold reaches 0
        # Also restart when stagnated (no improvement for too long)
        trigger_restart = (threshold <= 0) or (stagnation_count >= stagnation_limit)

        if trigger_restart:
            restarts += 1
            if stagnation_count >= stagnation_limit and threshold > 0:
                stale_restarts += 1
                if verbose:
                    print(f"  [STALE RESTART #{restarts}] Gen {gen} | "
                          f"stagnation={stagnation_count} | best_fit={best_fitness:.2f}")
            else:
                if verbose:
                    print(f"  [RESTART #{restarts}] Gen {gen} | best_fit={best_fitness:.2f}")
            restart_generations.append(gen)

            # Pick the top N survivors from the current population
            # (using the full population gives more genetic variety than the elite set)
            n_surv = max(1, pop_size // 8)
            surv_indices = sorted(range(pop_size),
                                   key=lambda i: fitness_values[i])[:n_surv]
            restart_elites = [copy.copy(population[i]) for i in surv_indices]

            population = cataclysmic_mutation(
                restart_elites, pop_size, mutation_rate, stale_restarts,
                talks, valid_map, researcher_ids, researchers,
            )
            fitness_values = [evaluate(c) for c in population]
            # Ensure best is in population after restart
            fitness_values[0] = best_fitness
            population[0] = copy.copy(best_chrom)
            threshold = T // 4  # reset threshold
            stagnation_count = 0

        feasible = [f for f in fitness_values if f < 1_000_000]
        convergence.append((sum(feasible) / len(feasible)) if feasible
                           else (sum(fitness_values) / len(fitness_values)))

        if verbose and gen % 20 == 0:
            print(f"  Gen {gen:4d} | threshold={threshold} | best_fit={best_fitness:.2f} | restarts={restarts}")

    # Final aggressive repair: force-desassign any remaining overallocation
    # so the returned solution never carries a hard 1M penalty
    best_chrom = _force_desassign_overallocated(
        best_chrom, researcher_ids, researchers)
    best_fitness = evaluate(best_chrom)

    # Clean the elite set too so all displayed values are consistent
    clean_elite: List[Tuple[Chromosome, float]] = []
    seen_clean: set = set()
    for c, _ in elite_set[:elite_size]:
        c_clean = _force_desassign_overallocated(c, researcher_ids, researchers)
        t = tuple(c_clean)
        if t not in seen_clean:
            f_clean = evaluate(c_clean)
            clean_elite.append((c_clean, f_clean))
            seen_clean.add(t)

    return best_chrom, best_fitness, convergence, clean_elite, restart_generations, fitness_values, gen_best
