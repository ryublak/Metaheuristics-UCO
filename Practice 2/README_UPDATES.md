# Project Development Log: Adaptive Genetic Algorithm (v3.0)

This document tracks the major improvements and design decisions made during the development of the evolutionary algorithm for Random Forest hyperparameter tuning.

## 🐛 Critical Bug Fixes

### 1. Corrected Tournament Selection
Fixed an indexing bug where the tournament was selecting candidates based on their list position rather than their actual fitness score. The selection process is now fully competitive and fair.

### 2. Robust Evaluation with StratifiedKFold
Switched from standard 5-fold CV to `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Reason:** The wine quality dataset has a slight order to it. Sequential folds were biased.
- **Impact:** Stratification ensures class balance across folds, providing an honest "thermometer" for the GA to measure model improvement.

| Method | Before (Broken CV) | After (Correct CV) |
|--------|-----------------|----------------------|
| Random Search  | ~0.740 | 0.767 |
| Grid Search    | ~0.739 | 0.812 |
| Genetic Alg.   | ~0.747 | **0.815** |

## 🧬 Diversity & Exploration

### 3. Hamming Distance Initialization
Initial population members must differ by at least 2 genes. This ensures the algorithm starts with a wide spread instead of clustered around a single point.

### 4. Variable Impact Mutation (Roulette)
Instead of a fixed mutation rate, we use a roulette-based approach:
- 75% → 1 gene change (fine-tuning)
- 20% → 2 gene change (medium perturbation)
- 5%  → 3 gene change (drastic jump to escape local minima)

## 🎯 Adaptive Logic (Pc and Pm)

### 5. Stagnation Detection
The algorithm tracks the elite fitness mean. If it fails to improve for 5 consecutive generations, it triggers an "exploration boost" by increasing the mutation probability ($P_m$).

### 6. Safety Bounds
To prevent the algorithm from collapsing into pure random search:
- $P_c$ (Crossover) floor: **0.40**
- $P_m$ (Mutation) ceiling: **0.60**

## 🚀 Performance Optimizations

### 7. Fitness Caching (Memorization)
A global dictionary stores evaluated configurations. Combined with elitism, this significantly speeds up the later stages of evolution when the population starts to converge.

### 8. Elitism (10%)
The top 3 individuals are preserved in every generation, ensuring we never lose the best solution found so far.

## 📊 Benchmark results

The `benchmark.py` script runs each method 5 times and generates 7 plots for analysis.

**Final Standings:**
- Random Search: `0.7667`
- Grid Search: `0.8123`
- **Genetic Algorithm: `0.8154` (Winner)**
