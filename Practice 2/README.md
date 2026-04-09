# Hyperparameter Optimization: Random Forest with Adaptive GA

This repository contains the implementation of an **Adaptive Genetic Algorithm** designed to optimize hyperparameters for a `RandomForestClassifier` on the Wine Quality dataset.

## Project Structure

- `src/main.py`: Core algorithm implementation (GA, Random Search, Grid Search).
- `src/benchmark.py`: Comparative performance experiment and visualization generator.
- `docs/`: LaTeX reports (English and Spanish) and generated plots.
- `data/`: Dataset storage (`winequality-red.csv`).

## Key Features

- **Strict Diversity Filter**: Uses Hamming distance to ensure population variety.
- **Adaptive Pc/Pm**: Dynamic adjustment of crossover and mutation rates based on population stagnation.
- **Fitness Caching**: Memorization of model evaluations to reduce computational overhead.
- **Stratified Validation**: Uses 5-fold Stratified Cross-Validation for robust performance measurement.

## Usage

### Run the Optimization
To run a single execution of the Genetic Algorithm:
```bash
python3 "Practice 2/src/main.py"
```

### Run the Benchmark
To generate the full comparative report and plots:
```bash
python3 "Practice 2/src/benchmark.py"
```

The plots will be saved in `docs/img/` and the raw results in `docs/benchmark_results.json`.

## Requirements
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
