#!/usr/bin/env python
"""Rebuild results/*.csv from the MLflow store.

MLflow is the source of truth: every run logs there as it completes, so an
interrupted grid still leaves its finished runs intact. Rebuilding from the
store is therefore safer than appending to a CSV that a crashed run may have
left stale or half-written.

    python scripts/export_results.py
"""
import pathlib
import sys

import mlflow
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from experiments import best_per_dataset, summarise  # noqa: E402
from pipeline import EXPERIMENT, TRACKING_URI  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# MLflow column -> our column. mean_recovery is absent on streams with no
# ground-truth drift index, so it is filled with NA rather than assumed 0.
METRICS = {
    "metrics.final_accuracy": "final_accuracy",
    "metrics.final_macro_f1": "final_macro_f1",
    "metrics.runtime_sec": "runtime_sec",
    "metrics.peak_memory_mb": "peak_memory_mb",
    "metrics.n_drift_detections": "n_drift_detections",
    "metrics.mean_recovery_samples": "mean_recovery",
    "metrics.n_recovered": "n_recovered",
}


def export():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if runs.empty:
        sys.exit(f"No runs found in {TRACKING_URI} - run src/experiments.py first.")

    df = pd.DataFrame({
        "dataset": runs["params.dataset"],
        "strategy": runs["params.strategy"],
        "seed": runs["params.seed"].astype(int),
    })
    for src, dst in METRICS.items():
        df[dst] = runs[src] if src in runs.columns else pd.NA

    # n_drifts is not logged as a metric; recover it from the logged param.
    df["n_drifts"] = (runs["params.true_drift_points"]
                      .fillna("[]").map(lambda s: len(eval(s))))
    df["n_recovered"] = df["n_recovered"].fillna(0).astype(int)

    # Guard against duplicates from re-running a config: keep the newest.
    df["_t"] = runs["start_time"]
    before = len(df)
    df = (df.sort_values("_t")
            .drop_duplicates(subset=["dataset", "strategy", "seed"], keep="last")
            .drop(columns="_t"))
    if before != len(df):
        print(f"dropped {before - len(df)} duplicate run(s), kept the most recent")

    df = df.sort_values(["dataset", "strategy", "seed"]).reset_index(drop=True)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "raw_runs.csv", index=False)
    agg = summarise(df)
    agg.to_csv(RESULTS / "summary.csv", index=False)

    print(f"{len(df)} runs -> results/raw_runs.csv")
    print(df.groupby("dataset").size().to_string())
    print("\n=== summary ===")
    print(agg.round(4).to_string(index=False))
    print("\n=== best per dataset ===")
    print(best_per_dataset(agg).round(4).to_string(index=False))
    return df, agg


if __name__ == "__main__":
    export()
