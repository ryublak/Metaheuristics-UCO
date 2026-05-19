# Metaheuristics-UCO

Coursework repository with three independent Python assignments.

## Layout

- `Practice 1/` — Jupyter notebook (Google Colab), no deps config
- `Practice 2/` — GA for Random Forest hyperparameter tuning on winequality-red.csv. Has `pyproject.toml`, `uv.lock`, `.python-version`.
- `Practice 3/` — CHC metaheuristic for talk allocation. Source code only; no `pyproject.toml` yet.

## Commands

```bash
# Run from inside a practice directory:
uv run src/main.py

# Or from repo root:
python3 "Practice 2/src/main.py"
python3 "Practice 3/src/main.py"
```

Practice 2 benchmark:
```bash
python3 "Practice 2/src/benchmark.py"
```

Practice 3 CLI flags (all optional, instance generated if no CSVs given):
`--schools --talks --researchers --pop-size --generations --mutation-rate --seed --verbose --num-schools --num-talks --num-researchers --prob-topics`

Report plots (regenerates `data/convergence.json` + `docs/img/*.png` with a well-posed instance):
```bash
python3 "Practice 3/src/generate_report_plots.py"
```

LaTeX report build (in `docs/`):
```bash
bash build.sh   # compiles pdflatex × 2 and cleans auxiliary files

## Toolchain

- **Python 3.14** (`.python-version` in Practice 2/). `uv` for dependency management.
- **No tests, no CI, no linters/formatters/typecheckers** — course project.
- Only Practice 2 has a `pyproject.toml` + `uv.lock`. Practice 3 imports `pandas`/`numpy` but has no manifest yet.
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` (Practice 2), plus stdlib (Practice 3).

## Codebase notes

- All practices are **independent** — no shared code between them.
- `Practice 3/src/main.py:27` does `sys.path.insert` to find sibling modules; always run from the Practice 3 root, not from `src/`.
- Practice 3 chromosome encoding: integer indices into `researchers.keys()` list, or `-1` for unassigned.
- Hard constraints (topic, level, travel) pre-filtered in `build_valid_researchers_per_talk`; fitness uses large penalty weights (1e6) for violations.
- Instance generation (`generate_instance.py`) wraps the professor's `data_generator_talks11f/` logic — uses `numpy.random` (seed via `np.random.seed`), not the `random` module.
