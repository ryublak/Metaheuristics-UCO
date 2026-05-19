# Metaheuristics — University of Córdoba

Coursework repository containing the three practical assignments developed for the
Metaheuristics subject.

## Projects

### Practice 1 — Local Search
Exploratory analysis of time series data using local search heuristics, implemented
as a Jupyter notebook (`P1MtH.ipynb`).

### Practice 2 — Genetic Algorithm for Hyperparameter Tuning
Adaptive Genetic Algorithm that optimises a `RandomForestClassifier` on the Wine
Quality dataset (`winequality-red.csv`). Features Hamming-distance diversity
enforcement, adaptive crossover and mutation rates, fitness caching, and
StratifiedKFold validation. Benchmarked against Grid Search and Random Search.

### Practice 3 — CHC Metaheuristic for Talk Allocation
CHC evolutionary algorithm that matches scientific outreach talks requested by
local schools to researchers from the University of Córdoba for the International
Day of Women and Girls in Science. The implementation handles hard constraints
(topic, level, travel, coverage) and a rich set of lexicographic soft priorities
(school equity, researcher rotation, historical patterns).

Key features:
- Modular penalty-based fitness with configurable weights
- 2-pass chromosome initialisation (one talk per school guaranteed)
- HUX crossover with incest prevention via dynamic Hamming threshold
- Multi-survivor cataclysmic restarts with adaptive mutation rate
- Stagnation detection and gradual threshold reset
- Two-phase repair operator (overallocation + fill unassigned)
- Complete elite set returned to the user as ranked alternatives

## Setup

```bash
git clone git@github.com:ryublak/Metaheuristics-UCO.git
cd Metaheuristics-UCO/"Practice 3"
```

Practice 2 uses `uv` for dependency management:

```bash
cd "Practice 2"
uv sync
```

Practice 3 uses standard Python (no virtual environment configured yet):

```bash
cd "Practice 3"
pip install pandas numpy matplotlib
```

## Usage

```bash
# Practice 3 — run on a generated instance
python3 "Practice 3/src/main.py" --seed 42 --verbose

# Practice 3 — run with custom instance parameters
python3 "Practice 3/src/main.py" --num-schools 10 --num-talks 20 --num-researchers 15 --prob-topics 0.2

# Practice 3 — run with CSV files
python3 "Practice 3/src/main.py" --schools data/schools.csv --talks data/talks.csv --researchers data/researchers.csv

# Practice 3 — generate report-quality plots (5-run averaged convergence)
python3 "Practice 3/src/generate_report_plots.py"

# Build the LaTeX report
cd "Practice 3/docs" && bash build.sh
```

## License

Copyright © 2026 Rafael Carlo Schoettel Vilchez, Jesús Fernández López.
All rights reserved. Unauthorised use, copying, modification, or distribution
is strictly prohibited. This code is for academic purposes only.
