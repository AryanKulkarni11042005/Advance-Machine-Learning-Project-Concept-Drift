# Comparative Evaluation of Concept Drift Adaptation Strategies for Streaming Classification: A Benchmark Study Across Synthetic and Real-World Data

Empirical benchmark of concept drift adaptation strategies for streaming classification, evaluated across synthetic and real-world datasets.

> Research project for Advanced Machine Learning (CE29T/CE29P)

## Overview

Real-world data streams change over time (concept drift), causing static ML models to degrade in accuracy. This project benchmarks multiple adaptation strategies — blind periodic retraining, detector-triggered retraining, incremental updates, and Adaptive Random Forest — to answer:

**Research question:** Which adaptation strategy restores model performance most effectively after concept drift, and how does the best strategy vary across drift types (abrupt, gradual, real-world) and computational budgets?

## Project Structure

```
.
├── data/                  # Raw and processed datasets (gitignored — see Data section)
├── src/
│   ├── data_loaders/      # Stream loaders for each dataset
│   ├── models/            # Baseline models + adaptation strategy implementations
│   ├── detectors/         # ADWIN / DDM wrappers
│   └── evaluation/        # Metrics: prequential accuracy, recovery time, compute cost
├── notebooks/              # Exploratory + experiment notebooks
├── results/                 # Logged metrics, plots, exported tables (gitignored raw runs)
├── paper/                    # LaTeX source for the paper draft
├── environment.yml
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11.9 |
| Streaming ML | River (drift detectors, incremental learners, ARF) |
| Batch ML | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | conda |

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/AryanKulkarni11042005/Advance-Machine-Learning-Project-Concept-Drift.git
cd https://github.com/AryanKulkarni11042005/Advance-Machine-Learning-Project-Concept-Drift.git
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

## Running MLflow Tracking

Start the tracking server (run in a separate terminal, keep it running during experiments):
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

View the dashboard at `http://localhost:5000`.

In notebooks/scripts:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("concept-drift-adaptation")
```

## Datasets

| Dataset | Drift Type | Source |
|---|---|---|
| Electricity Market (Elec2) | Gradual / recurring | `river.datasets` |
| Rotating Hyperplane | Gradual, tunable (synthetic) | `river.datasets.synth` |
| SEA Concepts | Abrupt (synthetic) | `river.datasets.synth` |
| Airlines | Real-world, implicit | OpenML dataset id 1169 |

Datasets are not committed to the repository. Run `src/data_loaders/download_data.py` to fetch them locally (script to be added — see Roadmap).

## Methodology Summary

1. **Baseline** — static classifier trained once, evaluated prequentially with no adaptation.
2. **Drift detection** — ADWIN and DDM wrap an incremental learner; firing points logged.
3. **Adaptation strategies compared:**
   - Blind periodic retraining
   - Detector-triggered full retrain
   - Detector-triggered incremental update
   - Adaptive Random Forest (ARF)
4. **Evaluation metrics** — prequential accuracy/F1 over time, recovery time post-drift, computational cost (retrain time, memory).

Full methodology detail is in `paper/methodology.tex` (or `docs/methodology.md`, once written).

## Team Workflow

- Branch per feature/experiment (`feature/adwin-detector`, `experiment/airlines-arf`), PR into `main`.
- Log every experiment run to MLflow — do not report numbers that aren't tracked.
- Keep `environment.yml` in sync — regenerate with `conda env export --no-builds > environment.yml` after adding a dependency, and note the addition in your PR description.
- Random seeds are fixed in `src/config.py` for reproducibility across machines.

## Roadmap

- [ ] Data loaders for all 4 datasets
- [ ] Baseline model + prequential evaluation loop
- [ ] ADWIN/DDM drift detection wrappers
- [ ] Implement 4 adaptation strategies
- [ ] MLflow logging integration
- [ ] Full experiment grid run
- [ ] Results analysis + plots
- [ ] Paper draft (Introduction → Conclusion)

## Citation

If this work informs your research, citation details will be added on paper acceptance.

## License
This project is licensed under the MIT License — see LICENSE.md for details.
