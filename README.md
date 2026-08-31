# Comparative Evaluation of Concept Drift Adaptation Strategies for Streaming Classification: A Benchmark Study Across Synthetic and Real-World Data

Empirical benchmark of concept drift adaptation strategies for streaming classification, evaluated across synthetic and real-world datasets.

> Research project for Advanced Machine Learning (CE29T/CE29P)

## Overview

Real-world data streams change over time (concept drift), causing static ML models to degrade in accuracy. This project benchmarks multiple adaptation strategies — blind periodic retraining, detector-triggered retraining, incremental updates, and Adaptive Random Forest — to answer:

**Research question:** Which adaptation strategy restores model performance most effectively after concept drift, and how does the best strategy vary across drift types (abrupt, gradual, real-world) and computational budgets?

## Project Structure

```
.
├── data/                  # Cached real-world datasets (gitignored)
├── scripts/
│   └── download_data.py   # Fetches Elec2 + Airlines
├── src/
│   ├── streams.py         # Stream loaders; synthetic streams carry ground-truth drift indices
│   ├── pipeline.py        # Prequential evaluation loop, strategy configs, MLflow logging
│   └── experiments.py     # Grid runner (datasets x strategies x seeds) + summary tables
├── notebooks/
│   └── 01_sea.ipynb       # Worked example: drift validation, strategy comparison, plots
├── results/               # raw_runs.csv, summary.csv (gitignored)
├── mlflow.db              # MLflow SQLite tracking store (gitignored)
├── paper/                 # LaTeX source for the paper draft (not yet written)
├── environment.yml
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11.9 |
| Streaming ML | River (drift detectors, incremental learners, ARF) |
| Batch ML | scikit-learn, XGBoost (not yet used — River covers the current strategies) |
| Experiment tracking | MLflow |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | conda |

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/AryanKulkarni11042005/Advance-Machine-Learning-Project-Concept-Drift.git
cd Advance-Machine-Learning-Project-Concept-Drift
```

### 2. Create the environment
```bash
conda env create -f environment.yml
conda activate aml-project
```

### 3. Register the Jupyter kernel
```bash
python -m ipykernel install --user --name=aml-project --display-name="Python (aml-project)"
```

### 4. Verify installation
```bash
python -c "import river, mlflow, sklearn, xgboost; print('Setup OK')"
```

### 5. Fetch the real-world datasets
```bash
python scripts/download_data.py
```

### 6. Run the benchmark
```bash
python src/experiments.py          # full grid -> results/*.csv + mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**The full grid takes roughly 45–60 minutes**, almost entirely ARF: it costs
~2 min on Elec2 and ~5 min on Airlines per seed, against seconds for every
other strategy. Restrict the grid while iterating:

```python
from experiments import run_grid
run_grid(datasets=["sea_abrupt"], strategies=["baseline", "detector_full_retrain"])
```

Or work through `notebooks/01_sea.ipynb` for the annotated walkthrough.

## Running MLflow Tracking

No server is required. Runs log to a local SQLite store (`mlflow.db`) configured
in `src/pipeline.py`, so notebooks work whether or not a server is up.

Browse the results:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Note: MLflow 3.x rejects the plain `./mlruns` file store, which is why the
backend is SQLite rather than a directory. To use a tracking server instead,
point `TRACKING_URI` in `src/pipeline.py` at `http://localhost:5000`.

## Datasets

| Dataset | Key in `streams.STREAMS` | Drift Type | Ground-truth drift index | Status |
|---|---|---|---|---|
| SEA Concepts (abrupt) | `sea_abrupt` | Abrupt (synthetic) | Yes | Implemented |
| SEA Concepts (gradual) | `sea_gradual` | Gradual transition (synthetic) | Yes (window midpoint) | Implemented |
| Rotating Hyperplane | `hyperplane` | Continuous/incremental (synthetic) | No | Implemented |
| Electricity Market (Elec2) | — | Gradual / recurring | No | Planned |
| Airlines | — | Real-world, implicit | No | Planned |

Recovery time is only reported where a ground-truth drift index exists. On the
hyperplane and the real-world streams, comparisons rest on accuracy, macro-F1,
and detector firing counts instead.

Synthetic streams are generated on demand by `src/streams.py` — nothing to download.
Real-world datasets (Elec2, Airlines) are not yet wired in; see Roadmap.

**On synthetic drift:** SEA variants differ only in the threshold on `att1 + att2`
(`{0: 8, 1: 9, 2: 7, 3: 9.5}`). The default pairing is **(3, 2)** — a 9.5 → 7 shift.
Narrower pairings such as (0, 2) move the boundary by only 1.0 across a sum spanning
0–20 and produce **no measurable accuracy drop**, which makes recovery time undefined.
Always confirm the baseline visibly degrades before trusting a comparison.

## Methodology Summary

1. **Baseline** — static classifier trained once, evaluated prequentially with no adaptation.
2. **Drift detection** — ADWIN wraps an incremental learner; firing points logged.
   (DDM is on the roadmap, not yet implemented.)
3. **Adaptation strategies compared:**
   - Blind periodic retraining
   - Detector-triggered full retrain
   - Detector-triggered incremental update
   - Adaptive Random Forest (ARF)
4. **Evaluation metrics** — prequential accuracy/macro-F1 over time, recovery time
   post-drift, computational cost (runtime, peak memory).

`detector_incremental` is a deliberate control: ADWIN fires but nothing is reset,
isolating how much adaptation comes from the detector versus the incremental
learner's own updating. On SEA it matches the baseline exactly — an expected
result, not a bug.

**Seeds.** On synthetic streams the seed regenerates the stream, so every
configuration runs across 3 seeds — the varying stream is what makes repeats a
real sample. On the fixed real-world datasets only ARF is stochastic, so the
deterministic strategies run once (see Datasets above).

`experiments.py` reports mean ± std and a `separable` flag: `True` when the
winner's margin exceeds its own spread, `False` when the top two are
indistinguishable, and `None` when the run was not replicated and the question
cannot be answered either way.

Results and their caveats are written up in [docs/findings.md](docs/findings.md).
Full methodology detail is in `paper/methodology.tex` (or `docs/methodology.md`, once written).

## Team Workflow

- Branch per feature/experiment (`feature/adwin-detector`, `experiment/airlines-arf`), PR into `main`.
- Log every experiment run to MLflow — do not report numbers that aren't tracked.
- Keep `environment.yml` in sync — regenerate with `conda env export --no-builds > environment.yml` after adding a dependency, and note the addition in your PR description.
- Random seeds are fixed in `src/config.py` for reproducibility across machines.

## Roadmap

- [x] Synthetic stream loaders (SEA abrupt, SEA gradual, rotating hyperplane)
- [x] Baseline model + prequential evaluation loop
- [x] ADWIN drift detection wired into the loop
- [x] Implement 4 adaptation strategies + baseline control
- [x] MLflow logging integration (SQLite backend)
- [x] Recovery time, runtime, and peak-memory metrics
- [x] Multi-seed experiment grid + summary tables
- [x] Real-world loaders: Elec2, Airlines (recovery time not reportable — no ground-truth drift index)
- [ ] Base-learner sweep (Hoeffding Tree / Naive Bayes / Logistic Regression)
- [ ] ADWIN `delta` sensitivity analysis
- [ ] DDM detector alongside ADWIN
- [ ] Paper draft (Introduction → Conclusion)

## Citation

If this work informs your research, citation details will be added on paper acceptance.

## License
[![License: MIT](https://shields.io)](LICENSE.md)

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
