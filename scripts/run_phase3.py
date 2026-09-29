#!/usr/bin/env python
"""Run the explicitly configured Phase 3 concept-drift benchmark.

Full run: python scripts/run_phase3.py
Small infrastructure smoke: python scripts/run_phase3.py --smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import streams  # noqa: E402
from pipeline import NEURAL_STRATEGIES, STRATEGIES, run_strategy  # noqa: E402


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def environment_metadata():
    names = {"river": "river", "pytorch": "torch", "numpy": "numpy",
             "pandas": "pandas", "scikit_learn": "scikit-learn"}
    versions = {}
    for key, package in names.items():
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        status = None
    implementation_files = ("src/pipeline.py", "src/streams.py",
                            "scripts/run_phase3.py", "scripts/export_results.py",
                            "configs/phase3.yaml")
    digest = hashlib.sha256()
    for relative in implementation_files:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    hardware = {"platform": platform.platform(), "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count()}
    try:
        import torch
        hardware["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            hardware["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        hardware["cuda_available"] = False
    return {"python": platform.python_version(), **versions,
            "git_commit": commit, "git_worktree_dirty": bool(status),
            "phase3_source_sha256": digest.hexdigest(), "hardware": hardware}


def _resolved_config(path, smoke=False, pilot=False):
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["run_id"] = "phase3_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["environment"] = environment_metadata()
    if smoke:
        cfg["experiment"]["version"] = "phase3a_smoke"
        cfg["seeds"] = [42]
        cfg["strategies"] = ["baseline", "lstm", "gru"]
        cfg["datasets"] = {"sea_abrupt": {"stream": {
            "n_per_concept": 40, "variants": [3, 2]}, "label_delay": 0}}
        cfg["eligibility"] = {"sea_abrupt": cfg["strategies"]}
    elif pilot:
        # Keep the configured matrix and stream/model settings intact, but
        # constrain this execution to the first development seed only.
        cfg["seeds"] = [42]
    _validate_config(cfg)
    return cfg


def _validate_config(cfg):
    strategies = set(STRATEGIES) | set(NEURAL_STRATEGIES)
    if len(cfg["seeds"]) != len(set(cfg["seeds"])) or not cfg["seeds"]:
        raise ValueError("seeds must be a non-empty list of unique integers")
    if set(cfg["datasets"]) != set(cfg["eligibility"]):
        raise ValueError("eligibility must explicitly list every configured dataset")
    for dataset, allowed in cfg["eligibility"].items():
        if dataset not in streams.STREAMS:
            raise ValueError(f"unknown configured dataset: {dataset}")
        if len(allowed) != len(set(allowed)) or set(allowed) != set(cfg["strategies"]):
            raise ValueError(f"{dataset}: eligibility must explicitly include each configured strategy once")
    if set(cfg["strategies"]) - strategies:
        raise ValueError("configuration contains an unknown strategy")
    if cfg["model"]["sequence_length"] != 10 or cfg["model"]["update_steps"] != 1:
        raise ValueError("Phase 3A requires sequence_length=10 and update_steps=1")
    if cfg["model"]["optimizer"].lower() != "adam":
        raise ValueError("the existing NeuralAdapter currently supports Adam")
    if cfg["airlines_hashing"]["hash"] != "blake2b-64":
        raise ValueError("the existing Airlines encoder supports blake2b-64 only")
    if cfg["airlines_hashing"]["numeric_transform"] != "signed_log1p":
        raise ValueError("the existing Airlines encoder supports signed_log1p only")


def _independence_unit(dataset, strategy, seed):
    stochastic = strategy == "arf" or strategy in NEURAL_STRATEGIES
    if not streams.is_deterministic(dataset):
        return f"stream_seed:{seed}"
    if stochastic:
        return f"model_seed:{seed}"
    return "fixed_stream_deterministic_model"


def _summary(raw):
    rows = []
    for (dataset, strategy), group in raw.groupby(["dataset", "strategy"], sort=True):
        units = group.drop_duplicates("independence_unit", keep="first")
        row = {"dataset": dataset, "strategy": strategy,
               "n_runs": int(len(group)),
               "n_independent_runs": int(units["independence_unit"].nunique()),
               "n_seeds": int(group["seed"].nunique())}
        for metric in ("accuracy", "macro_f1", "runtime", "peak_memory",
                       "drift_detections", "n_valid_predictions", "mean_recovery"):
            values = pd.to_numeric(units[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        row["majority_baseline"] = float(group["majority_baseline"].iloc[0])
        row["majority_baseline_type"] = "offline_full_stream_reference"
        rows.append(row)
    return pd.DataFrame(rows)


def run(config_path=ROOT / "configs" / "phase3.yaml", smoke=False, pilot=False,
        log_to_mlflow=True):
    if smoke and pilot:
        raise ValueError("--smoke and --pilot are mutually exclusive")
    cfg = _resolved_config(pathlib.Path(config_path), smoke=smoke, pilot=pilot)
    exp = cfg["experiment"]
    output = (ROOT / exp["output_root"] / exp["run_id"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    raw_rows, event_rows, recovery_rows = [], [], []
    cache = {}
    total = sum(len(cfg["eligibility"][d]) * len(cfg["seeds"])
                for d in cfg["datasets"])
    completed = 0
    for dataset, dataset_cfg in cfg["datasets"].items():
        fixed = streams.is_deterministic(dataset)
        if fixed:
            cache[dataset] = streams.load_stream(dataset, seed=cfg["seeds"][0],
                                                **dataset_cfg["stream"])
        for strategy in cfg["eligibility"][dataset]:
            for seed in cfg["seeds"]:
                if fixed:
                    samples, drift_points = cache[dataset]
                else:
                    samples, drift_points = streams.load_stream(
                        dataset, seed=seed, **dataset_cfg["stream"])
                result = run_strategy(
                    samples, strategy, dataset, drift_points, seed=seed,
                    eval_every=int(exp["eval_every"]),
                    delay=int(dataset_cfg["label_delay"]),
                    adwin_config=cfg["adwin"], model_config=cfg["model"],
                    dataset_config=dataset_cfg["stream"],
                    run_id=exp["run_id"],
                    experiment_version=exp["version"],
                    mlflow_experiment=exp["mlflow_experiment"],
                    airline_hash_config=cfg["airlines_hashing"],
                    recovery_tolerance=float(exp["recovery_tolerance"]),
                    log_to_mlflow=log_to_mlflow,
                )
                neural = strategy in NEURAL_STRATEGIES
                raw_rows.append({
                    "run_id": exp["run_id"], "experiment_version": exp["version"],
                    "dataset": dataset, "strategy": strategy, "seed": seed,
                    "accuracy": result["final_accuracy"],
                    "macro_f1": result["final_macro_f1"],
                    "runtime": result["runtime_sec"],
                    "peak_memory": result["peak_memory_mb"],
                    "n_valid_predictions": result["n_valid_predictions"],
                    "drift_detections": result["n_drift_detections"],
                    "majority_baseline": result["majority_baseline"],
                    "majority_baseline_type": "offline_full_stream_reference",
                    "label_delay": int(dataset_cfg["label_delay"]),
                    "sequence_length": cfg["model"]["sequence_length"] if neural else None,
                    "hidden_size": cfg["model"]["hidden_size"] if neural else None,
                    "optimizer": cfg["model"]["optimizer"] if neural else None,
                    "learning_rate": cfg["model"]["learning_rate"] if neural else None,
                    "update_steps": cfg["model"]["update_steps"] if neural else None,
                    "replay_size": cfg["model"]["replay_size"] if neural else None,
                    "adwin_delta": cfg["adwin"]["delta"] if "adwin" in strategy or strategy.startswith("detector_") else None,
                    "adwin_clock": cfg["adwin"]["clock"] if "adwin" in strategy or strategy.startswith("detector_") else None,
                    "dataset_configuration": _json(dataset_cfg["stream"]),
                    "model_configuration": _json(cfg["model"] if neural else {}),
                    "airlines_hashing": _json(result["airlines_encoder"] or {}),
                    "adwin_configuration": _json(result["adwin_config"] or {}),
                    "drift_events_json": _json(result["drift_events"]),
                    "detector_feedback_json": _json(result["detector_feedback"]),
                    "recovery_records_json": _json(result["recovery_records"]),
                    "mean_recovery": result["mean_recovery"],
                    "n_recovered": result["n_recovered"],
                    "n_known_drifts": len(drift_points),
                    "independence_unit": _independence_unit(dataset, strategy, seed),
                })
                for event in result["drift_events"]:
                    event_rows.append({"run_id": exp["run_id"], "dataset": dataset,
                                       "strategy": strategy, "seed": seed,
                                       **event, "adwin_delta": cfg["adwin"]["delta"],
                                       "adwin_clock": cfg["adwin"]["clock"]})
                for recovery in result["recovery_records"]:
                    recovery_rows.append({"run_id": exp["run_id"], "dataset": dataset,
                                          "strategy": strategy, "seed": seed,
                                          **recovery})
                completed += 1
                print(f"[{completed}/{total}] {dataset} {strategy} seed={seed} "
                      f"accuracy={result['final_accuracy']:.4f} "
                      f"valid={result['n_valid_predictions']} "
                      f"detections={result['n_drift_detections']}", flush=True)

    raw = pd.DataFrame(raw_rows)
    summary = _summary(raw)
    raw.to_csv(output / "raw_runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    event_columns = ["run_id", "dataset", "strategy", "seed", "feedback_index",
                     "prediction_index", "detected_at", "adwin_delta", "adwin_clock"]
    pd.DataFrame(event_rows, columns=event_columns).to_csv(output / "drift_events.csv", index=False)
    recovery_columns = ["run_id", "dataset", "strategy", "seed", "drift_index",
                        "baseline_accuracy", "threshold", "recovery_index",
                        "recovery_time", "status"]
    pd.DataFrame(recovery_rows, columns=recovery_columns).to_csv(
        output / "recoveries.csv", index=False)
    print(f"Phase 3 outputs: {output}")
    return output, raw, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase3.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="run SEA abrupt baseline/LSTM/GRU with one seed and 80 samples")
    mode.add_argument("--pilot", action="store_true",
                      help="run the full configured dataset/strategy matrix with seed 42 only")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(args.config, smoke=args.smoke, pilot=args.pilot,
        log_to_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
