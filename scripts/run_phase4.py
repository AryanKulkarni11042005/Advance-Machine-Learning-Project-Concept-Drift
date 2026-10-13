#!/usr/bin/env python
"""Run the Phase 4 neural architecture/adaptation ablation.

Full run: python scripts/run_phase4.py
Resume:   python scripts/run_phase4.py --resume results/phase4/<run_id>
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
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase4.yaml"
PHASE4_ROOT = (ROOT / "results" / "phase4").resolve()
RAW_COLUMNS = [
    "run_id", "experiment_version", "dataset", "strategy", "seed",
    "accuracy", "macro_f1", "runtime", "peak_memory",
    "n_valid_predictions", "drift_detections", "majority_baseline",
    "majority_baseline_type", "label_delay", "sequence_length", "hidden_size",
    "optimizer", "learning_rate", "update_steps", "replay_size",
    "adwin_delta", "adwin_clock", "dataset_configuration",
    "model_configuration", "airlines_hashing", "adwin_configuration",
    "drift_events_json", "detector_feedback_json", "recovery_records_json",
    "mean_recovery", "n_recovered", "n_known_drifts", "independence_unit",
]
SUMMARY_COLUMNS = [
    "dataset", "strategy", "adwin_delta", "n_runs", "n_independent_runs",
    "n_seeds", "accuracy_mean", "accuracy_std", "macro_f1_mean",
    "macro_f1_std", "runtime_mean", "runtime_std", "peak_memory_mean",
    "peak_memory_std", "drift_detections_mean", "drift_detections_std",
    "n_valid_predictions_mean", "n_valid_predictions_std", "mean_recovery_mean",
    "mean_recovery_std", "majority_baseline", "majority_baseline_type",
]
DELTAS = (0.001, 0.002, 0.005)
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("sea_abrupt", "sea_gradual", "hyperplane")


def _runtime_imports():
    """Load project/runtime dependencies only after CLI parsing (so --help works)."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import pandas as pd
    import yaml
    import streams
    from pipeline import NEURAL_STRATEGIES, run_strategy
    from run_phase3 import _atomic_csv, _auxiliary_frames, _summary

    return pd, yaml, streams, NEURAL_STRATEGIES, run_strategy, _atomic_csv, _auxiliary_frames, _summary


def _environment_metadata():
    package_names = {"river": "river", "pytorch": "torch", "numpy": "numpy",
                     "pandas": "pandas", "scikit_learn": "scikit-learn",
                     "mlflow": "mlflow"}
    versions = {}
    for key, package in package_names.items():
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
    implementation_files = (
        "src/pipeline.py", "src/streams.py", "src/models/networks.py",
        "src/models/neural_adapter.py", "src/models/sequence.py",
        "scripts/run_phase4.py", "configs/phase4.yaml",
    )
    digest = hashlib.sha256()
    for relative in implementation_files:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return {
        "python": platform.python_version(), **versions,
        "git_commit": commit,
        "phase4_source_sha256": digest.hexdigest(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }


def _validate_config(cfg, neural_strategies, stream_registry):
    metadata = cfg["metadata"]
    exp = cfg["experiment"]
    model = cfg["model"]
    adwin = cfg["adwin"]
    if metadata["phase"] != 4:
        raise ValueError("metadata.phase must be 4")
    if exp["output_root"] != "results/phase4":
        raise ValueError("Phase 4 output_root must be results/phase4")
    if exp["mlflow_experiment"] != "concept-drift-phase4":
        raise ValueError("Phase 4 MLflow experiment must be concept-drift-phase4")
    if tuple(cfg["seeds"]) != SEEDS or tuple(metadata["seeds"]) != SEEDS:
        raise ValueError("Phase 4 must use exactly seeds 42, 43, 44, 45, 46")
    if tuple(cfg["datasets"]) != DATASETS or tuple(metadata["datasets"]) != DATASETS:
        raise ValueError("Phase 4 must use sea_abrupt, sea_gradual, and hyperplane")
    if tuple(adwin["deltas"]) != DELTAS or tuple(metadata["adwin_deltas"]) != DELTAS:
        raise ValueError("ADWIN deltas must be exactly 0.001, 0.002, and 0.005")
    if adwin["clock"] != 32:
        raise ValueError("Phase 4 ADWIN clock must remain 32")
    if model["replay_size"] != 32:
        raise ValueError("Phase 4 requires replay_size=32 (the pipeline strategy buffer is fixed at 32)")
    fixed_model = {"sequence_length": 10, "hidden_size": 16,
                   "optimizer": "Adam", "learning_rate": 0.01,
                   "update_steps": 1}
    for key, expected in fixed_model.items():
        if model[key] != expected:
            raise ValueError(f"Phase 4 requires model.{key}={expected!r}")
    if exp["eval_every"] != 200 or exp["recovery_tolerance"] != 0.02:
        raise ValueError("Phase 4 evaluation and recovery settings must match Phase 3")
    if set(cfg["datasets"]) - set(stream_registry):
        raise ValueError("Phase 4 contains a dataset not registered in src/streams.py")
    if metadata["expected_runs"] != 210:
        raise ValueError("metadata.expected_runs must be 210")

    expected = set()
    for architecture in ("lstm", "gru"):
        expected.add((architecture, architecture, "online", None))
        for policy in ("adwin_reset", "adwin_finetune"):
            strategy = f"{architecture}_{policy}"
            if strategy not in neural_strategies:
                raise ValueError(f"pipeline does not implement {strategy}")
            for delta in DELTAS:
                expected.add((strategy, architecture, policy, delta))
    conditions = cfg["conditions"]
    actual = set()
    condition_ids = set()
    for condition in conditions:
        condition_id = condition["condition_id"]
        if condition_id in condition_ids:
            raise ValueError(f"duplicate condition_id: {condition_id}")
        condition_ids.add(condition_id)
        delta = condition["adwin_delta"]
        delta = None if delta is None else float(delta)
        actual.add((condition["strategy"], condition["architecture"],
                    condition["adaptation_policy"], delta))
        is_online = condition["adaptation_policy"] == "online"
        if is_online:
            if delta is not None or condition["adwin_clock"] is not None:
                raise ValueError(f"{condition_id}: online conditions cannot configure ADWIN")
        elif delta not in DELTAS or condition["adwin_clock"] != 32:
            raise ValueError(f"{condition_id}: detector conditions require a listed delta and clock 32")
        if condition["strategy"] not in neural_strategies:
            raise ValueError(f"unknown neural strategy: {condition['strategy']}")
        effective_replay_size = neural_strategies[condition["strategy"]].get(
            "replay_buffer_size", 32
        )
        if effective_replay_size != model["replay_size"]:
            raise ValueError("configured replay_size does not match the pipeline's effective buffer size")
    if len(conditions) != 14 or actual != expected:
        raise ValueError("Phase 4 must contain exactly the 14 specified neural conditions")
    if tuple(metadata["strategies"]) != (
            "lstm", "lstm_adwin_reset", "lstm_adwin_finetune",
            "gru", "gru_adwin_reset", "gru_adwin_finetune"):
        raise ValueError("metadata.strategies does not match the Phase 4 condition matrix")
    total = len(cfg["datasets"]) * len(conditions) * len(cfg["seeds"])
    if total != metadata["expected_runs"]:
        raise ValueError(f"configured run count is {total}, expected 210")


def _independence_unit(dataset, strategy, streams, neural_strategies, seed):
    if not streams.is_deterministic(dataset):
        return f"stream_seed:{seed}"
    if strategy in neural_strategies:
        return f"model_seed:{seed}"
    return "fixed_stream_deterministic_model"


def _resume_key(row):
    delta = row["adwin_delta"]
    if delta is None or (isinstance(delta, float) and delta != delta):
        delta = None
    else:
        delta = float(delta)
    return (str(row["dataset"]), str(row["strategy"]), delta, int(row["seed"]))


def _summary_by_condition(raw, pd, phase3_summary):
    if raw.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    grouped = raw.groupby(["dataset", "strategy", "adwin_delta"],
                          sort=True, dropna=False)
    for (dataset, strategy, delta), group in grouped:
        # Reuse Phase 3's exact metric and independence-unit aggregation for
        # each Phase 4 delta stratum instead of pooling detector settings.
        row = phase3_summary(group).iloc[0].to_dict()
        row["adwin_delta"] = None if pd.isna(delta) else float(delta)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _checkpoint(output, raw, pd, phase3_summary, auxiliary_frames, atomic_csv):
    _atomic_raw = pd.DataFrame(raw, columns=RAW_COLUMNS)
    atomic_csv(_atomic_raw, output / "raw_runs.csv")
    atomic_csv(_summary_by_condition(_atomic_raw, pd, phase3_summary),
               output / "summary.csv")
    events, recoveries = auxiliary_frames(_atomic_raw)
    atomic_csv(events, output / "drift_events.csv")
    atomic_csv(recoveries, output / "recoveries.csv")


def run(resume=None):
    pd, yaml, streams, neural_strategies, run_strategy, atomic_csv, auxiliary_frames, phase3_summary = _runtime_imports()
    if resume:
        output = pathlib.Path(resume).expanduser().resolve()
        try:
            output.relative_to(PHASE4_ROOT)
        except ValueError as exc:
            raise ValueError(f"--resume must point inside {PHASE4_ROOT}") from exc
        if not output.is_dir() or not (output / "config.yaml").is_file():
            raise ValueError(f"--resume requires an existing Phase 4 directory with config.yaml: {output}")
        with (output / "config.yaml").open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg["experiment"]["run_id"] != output.name:
            raise ValueError("resume directory name does not match config.yaml run_id")
    else:
        with CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["experiment"]["run_id"] = "phase4_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (ROOT / cfg["experiment"]["output_root"] / cfg["experiment"]["run_id"]).resolve()
        try:
            output.relative_to(PHASE4_ROOT)
        except ValueError as exc:
            raise ValueError("Phase 4 output path must remain under results/phase4") from exc

    _validate_config(cfg, neural_strategies, streams.STREAMS)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        cfg["environment"] = _environment_metadata()
        with (output / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    raw_path = output / "raw_runs.csv"
    raw_frame = pd.read_csv(raw_path) if resume and raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    required_resume_columns = {"dataset", "strategy", "adwin_delta", "seed"}
    if not required_resume_columns.issubset(raw_frame.columns):
        raise ValueError("existing raw_runs.csv is missing Phase 4 resume identity columns")
    if not raw_frame.empty:
        raw_frame = raw_frame.drop_duplicates(
            subset=["dataset", "strategy", "adwin_delta", "seed"], keep="first"
        )
    raw_rows = raw_frame.reindex(columns=RAW_COLUMNS).to_dict("records")
    completed_keys = {_resume_key(row) for row in raw_rows}
    expected_keys = {
        (str(dataset), str(condition["strategy"]),
         None if condition["adwin_delta"] is None else float(condition["adwin_delta"]),
         int(seed))
        for dataset in cfg["datasets"]
        for condition in cfg["conditions"]
        for seed in cfg["seeds"]
    }
    if not completed_keys.issubset(expected_keys):
        raise ValueError("raw_runs.csv contains resume keys outside the saved Phase 4 configuration")
    total = len(expected_keys)
    completed = len(completed_keys)
    if completed > total:
        raise ValueError(f"raw_runs.csv has {completed} completed keys, exceeding expected {total}")
    if resume:
        print(f"[RESUME] Existing completed runs: {completed}", flush=True)
        print(f"[RESUME] Remaining runs: {total - completed}", flush=True)

    cache = {}
    for dataset, dataset_cfg in cfg["datasets"].items():
        fixed = streams.is_deterministic(dataset)
        if fixed:
            cache[dataset] = streams.load_stream(
                dataset, seed=cfg["seeds"][0], **dataset_cfg["stream"]
            )
        for condition in cfg["conditions"]:
            strategy = condition["strategy"]
            delta = condition["adwin_delta"]
            clock = condition["adwin_clock"]
            key_delta = None if delta is None else float(delta)
            for seed in cfg["seeds"]:
                key = (str(dataset), str(strategy), key_delta, int(seed))
                if key in completed_keys:
                    continue
                if fixed:
                    samples, drift_points = cache[dataset]
                else:
                    samples, drift_points = streams.load_stream(
                        dataset, seed=seed, **dataset_cfg["stream"]
                    )
                unit = _independence_unit(dataset, strategy, streams,
                                          neural_strategies, seed)
                adwin_config = ({"delta": float(delta), "clock": int(clock)}
                                if delta is not None else {})
                extra_mlflow_params = {
                    "phase": 4,
                    "dataset": dataset,
                    "strategy": strategy,
                    "architecture": condition["architecture"],
                    "adaptation_policy": condition["adaptation_policy"],
                    "adwin_delta": key_delta,
                    "adwin_clock": int(clock) if clock is not None else None,
                    "seed": int(seed),
                    "sequence_length": cfg["model"]["sequence_length"],
                    "hidden_size": cfg["model"]["hidden_size"],
                    "learning_rate": cfg["model"]["learning_rate"],
                    "update_steps": cfg["model"]["update_steps"],
                    "replay_size": cfg["model"]["replay_size"],
                    "independence_unit": unit,
                }
                result = run_strategy(
                    samples, strategy, dataset, drift_points, seed=seed,
                    eval_every=int(cfg["experiment"]["eval_every"]),
                    delay=int(dataset_cfg["label_delay"]),
                    adwin_config=adwin_config,
                    model_config=cfg["model"],
                    dataset_config=dataset_cfg["stream"],
                    run_id=cfg["experiment"]["run_id"],
                    experiment_version=cfg["experiment"]["version"],
                    mlflow_experiment=cfg["experiment"]["mlflow_experiment"],
                    airline_hash_config=cfg["airlines_hashing"],
                    recovery_tolerance=float(cfg["experiment"]["recovery_tolerance"]),
                    log_to_mlflow=True,
                    mlflow_extra_params=extra_mlflow_params,
                    mlflow_run_name=f"{dataset}_{condition['condition_id']}_s{seed}",
                )
                neural = strategy in neural_strategies
                raw_rows.append({
                    "run_id": cfg["experiment"]["run_id"],
                    "experiment_version": cfg["experiment"]["version"],
                    "dataset": dataset, "strategy": strategy, "seed": int(seed),
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
                    "adwin_delta": key_delta,
                    "adwin_clock": int(clock) if clock is not None else None,
                    "dataset_configuration": json.dumps(
                        dataset_cfg["stream"], sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "model_configuration": json.dumps(
                        cfg["model"], sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "airlines_hashing": "{}",
                    "adwin_configuration": json.dumps(
                        result["adwin_config"] or {}, sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "drift_events_json": json.dumps(
                        result["drift_events"], sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "detector_feedback_json": json.dumps(
                        result["detector_feedback"], sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "recovery_records_json": json.dumps(
                        result["recovery_records"], sort_keys=True,
                        separators=(",", ":"), default=str,
                    ),
                    "mean_recovery": result["mean_recovery"],
                    "n_recovered": result["n_recovered"],
                    "n_known_drifts": len(drift_points),
                    "independence_unit": unit,
                })
                completed += 1
                completed_keys.add(key)
                _checkpoint(output, raw_rows, pd, phase3_summary,
                            auxiliary_frames, atomic_csv)
                print(f"[{completed}/{total}] {dataset} {strategy} "
                      f"delta={key_delta} seed={seed} "
                      f"accuracy={result['final_accuracy']:.4f} "
                      f"valid={result['n_valid_predictions']} "
                      f"detections={result['n_drift_detections']}", flush=True)

    _checkpoint(output, raw_rows, pd, phase3_summary, auxiliary_frames, atomic_csv)
    raw = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
    summary = _summary_by_condition(raw, pd, phase3_summary)
    print(f"Phase 4 outputs: {output}")
    return output, raw, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", metavar="OUTPUT_DIRECTORY",
                        help="resume a Phase 4 output directory using its config.yaml")
    args = parser.parse_args()
    run(resume=args.resume)


if __name__ == "__main__":
    main()
