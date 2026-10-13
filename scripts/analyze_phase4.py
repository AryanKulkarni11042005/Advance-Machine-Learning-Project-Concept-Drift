#!/usr/bin/env python
"""Validate and analyze frozen Phase 4 results without running models.

Example:
    python scripts/analyze_phase4.py --input results/phase4/phase4_20260929_233350
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "phase4" / "phase4_20260929_233350"
DEFAULT_PHASE3_SUMMARY = ROOT / "results" / "phase3" / "phase3_20260929_121546" / "summary.csv"
METRICS = (
    "accuracy", "macro_f1", "runtime", "peak_memory",
    "drift_detections", "n_valid_predictions", "mean_recovery",
)
CONDITION_METRICS = (
    "accuracy", "macro_f1", "runtime", "peak_memory",
    "drift_detections", "n_valid_predictions", "mean_recovery",
)
PAIR_METRICS = CONDITION_METRICS
DELTAS = (0.001, 0.002, 0.005)
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("sea_abrupt", "sea_gradual", "hyperplane")
PHASE3_METRICS = (
    "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
    "runtime_mean", "runtime_std", "peak_memory_mean", "peak_memory_std",
    "drift_detections_mean", "drift_detections_std",
    "n_valid_predictions_mean", "n_valid_predictions_std",
    "mean_recovery_mean", "mean_recovery_std",
)


def _load_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import yaml
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError(
            "Analysis requires the existing pandas, NumPy, SciPy, and PyYAML packages."
        ) from exc
    return np, pd, scipy, stats, yaml


def _delta(value, np):
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    return float(value)


def _strategy_factors(strategy):
    architecture = "lstm" if strategy.startswith("lstm") else "gru"
    if strategy == architecture:
        policy = "online"
    elif strategy.endswith("_adwin_reset"):
        policy = "adwin_reset"
    elif strategy.endswith("_adwin_finetune"):
        policy = "adwin_finetune"
    else:
        raise ValueError(f"Unrecognized Phase 4 strategy: {strategy}")
    return architecture, policy


def _counter_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return round(float(value), 10)
    return str(value)


def _validate_artifacts(input_dir, phase3_summary_path, np, pd, yaml):
    expected_files = (
        "config.yaml", "raw_runs.csv", "summary.csv",
        "drift_events.csv", "recoveries.csv",
    )
    missing = [name for name in expected_files if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 4 run is missing required files: {missing}")
    if not phase3_summary_path.is_file():
        raise FileNotFoundError(f"Phase 3 summary not found: {phase3_summary_path}")

    with (input_dir / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    raw = pd.read_csv(input_dir / "raw_runs.csv")
    summary = pd.read_csv(input_dir / "summary.csv")
    events = pd.read_csv(input_dir / "drift_events.csv")
    recoveries = pd.read_csv(input_dir / "recoveries.csv")
    phase3 = pd.read_csv(phase3_summary_path)

    resume_columns = {"dataset", "strategy", "adwin_delta", "seed"}
    missing_identity = resume_columns - set(raw.columns)
    if missing_identity:
        raise ValueError(f"raw_runs.csv is missing run identity fields: {sorted(missing_identity)}")
    config_conditions = config.get("conditions", [])
    configured_datasets = tuple(config.get("datasets", {}))
    seeds = tuple(config.get("seeds", []))
    if config.get("metadata", {}).get("expected_runs") != len(raw):
        raise ValueError("config expected_runs does not match raw_runs.csv row count")
    if set(configured_datasets) != set(DATASETS) or set(seeds) != set(SEEDS):
        raise ValueError("config datasets/seeds do not match the declared Phase 4 design")

    expected_conditions = set()
    for condition in config_conditions:
        expected_conditions.add((
            condition["strategy"],
            None if condition.get("adwin_delta") is None
            else float(condition["adwin_delta"]),
        ))
    raw_conditions = {
        (str(row.strategy), _delta(row.adwin_delta, np))
        for row in raw[["strategy", "adwin_delta"]].drop_duplicates().itertuples(index=False)
    }
    matrix_conditions = {("lstm", None), ("gru", None)}
    for architecture in ("lstm", "gru"):
        for policy in ("adwin_reset", "adwin_finetune"):
            matrix_conditions.update(
                (f"{architecture}_{policy}", delta) for delta in DELTAS
            )
    if (len(config_conditions) != 14 or expected_conditions != matrix_conditions
            or raw_conditions != expected_conditions):
        raise ValueError("raw strategy/delta conditions do not match config.yaml")
    if len(raw) != 210:
        raise ValueError(f"Expected 210 raw rows, found {len(raw)}")

    key = ["dataset", "strategy", "adwin_delta", "seed"]
    duplicate_count = int(raw.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"raw_runs.csv contains {duplicate_count} duplicate run keys")
    if set(raw.dataset) != set(DATASETS):
        raise ValueError("Observed dataset names do not match Phase 4")
    if set(raw.strategy) != {
        "lstm", "lstm_adwin_reset", "lstm_adwin_finetune",
        "gru", "gru_adwin_reset", "gru_adwin_finetune",
    }:
        raise ValueError("Observed strategy names do not match Phase 4")
    if sorted(raw.seed.unique().tolist()) != list(SEEDS):
        raise ValueError("Observed seeds do not match 42–46")
    groups = raw.groupby(key[:3], dropna=False)["seed"].agg(["count", "nunique"])
    if len(groups) != 42 or not (groups["count"].eq(5) & groups["nunique"].eq(5)).all():
        raise ValueError("Each dataset × strategy × delta condition must have five distinct seeds")

    metrics = ["accuracy", "macro_f1", "runtime", "peak_memory",
               "n_valid_predictions", "drift_detections"]
    missing_metrics = raw[metrics].isna().sum()
    if missing_metrics.any():
        raise ValueError(f"Missing required metrics: {missing_metrics[missing_metrics.gt(0)].to_dict()}")
    if (raw["n_valid_predictions"] <= 0).any():
        raise ValueError("n_valid_predictions must be positive for all neural runs")
    expected_valid = {"sea_abrupt": 9991, "sea_gradual": 10991, "hyperplane": 9991}
    for dataset, expected in expected_valid.items():
        observed = set(raw.loc[raw.dataset.eq(dataset), "n_valid_predictions"].astype(int))
        if observed != {expected}:
            raise ValueError(f"{dataset}: unexpected valid prediction counts {observed}")

    online = raw.strategy.isin(["lstm", "gru"])
    if raw.loc[online, ["adwin_delta", "adwin_clock"]].notna().any().any():
        raise ValueError("Online strategies must have null ADWIN delta and clock")
    detector = ~online
    if set(raw.loc[detector, "adwin_delta"].unique()) != set(DELTAS):
        raise ValueError("Detector conditions must contain exactly the three configured deltas")
    if set(raw.loc[detector, "adwin_clock"].unique()) != {32}:
        raise ValueError("Detector conditions must use ADWIN clock 32")
    for column, expected in (("sequence_length", 10), ("hidden_size", 16),
                             ("learning_rate", 0.01), ("update_steps", 1),
                             ("replay_size", 32)):
        if set(raw[column].dropna().unique()) != {expected}:
            raise ValueError(f"Unexpected {column} values in raw_runs.csv")
    expected_units = {f"stream_seed:{seed}" for seed in SEEDS}
    if set(raw.independence_unit) != expected_units:
        raise ValueError("independence_unit values do not match the Phase 3 synthetic-stream convention")
    expected_model = {"sequence_length": 10, "hidden_size": 16,
                      "optimizer": "Adam", "learning_rate": 0.01,
                      "update_steps": 1, "replay_size": 32}
    if config.get("model") != expected_model:
        raise ValueError("config model settings do not match the frozen Phase 4 controls")
    for column, expected in (("sequence_length", 10), ("hidden_size", 16),
                             ("learning_rate", 0.01), ("update_steps", 1),
                             ("replay_size", 32)):
        if set(raw[column].dropna().unique()) != {expected}:
            raise ValueError(f"raw {column} values do not match config.yaml")
    delay_by_dataset = {name: int(data["label_delay"])
                        for name, data in config["datasets"].items()}
    if any(int(row.label_delay) != delay_by_dataset[row.dataset]
           for row in raw[["dataset", "label_delay"]].itertuples(index=False)):
        raise ValueError("raw label_delay values do not match config.yaml")

    summary_keys = ["dataset", "strategy", "adwin_delta"]
    if not set(summary_keys).issubset(summary.columns):
        raise ValueError("summary.csv must include dataset, strategy, and adwin_delta")
    if summary.groupby(summary_keys, dropna=False).ngroups != 42:
        raise ValueError("summary.csv collapses or omits delta-specific conditions")
    if summary.duplicated(summary_keys).any():
        raise ValueError("summary.csv contains duplicate dataset/strategy/delta groups")
    if not (summary.n_runs.eq(5).all() and summary.n_seeds.eq(5).all()
            and summary.n_independent_runs.eq(5).all()):
        raise ValueError("summary run, seed, or independence counts are inconsistent")

    # Confirm drift-event CSV records against their raw embedded source,
    # including delta because drift_events.csv carries that field.
    embedded_event_counts = Counter()
    for row in raw.to_dict("records"):
        for event in json.loads(row["drift_events_json"]):
            embedded_event_counts[(row["dataset"], row["strategy"],
                                   _delta(row["adwin_delta"], np), int(row["seed"]),
                                   int(event["feedback_index"]),
                                   int(event["prediction_index"]),
                                   int(event["detected_at"]))] += 1
    csv_event_counts = Counter(
        (row.dataset, row.strategy, _delta(row.adwin_delta, np), int(row.seed),
         int(row.feedback_index), int(row.prediction_index), int(row.detected_at))
        for row in events.itertuples(index=False)
    )
    if embedded_event_counts != csv_event_counts:
        raise ValueError("drift_events.csv does not match raw_runs.csv event JSON")

    # The frozen Phase 3-compatible recovery CSV has no delta column. Validate
    # its records as a multiset, but analysis below takes recovery data from
    # raw_runs.csv so every measurement retains its delta identity.
    recovery_fields = ["dataset", "strategy", "seed", "drift_index",
                       "baseline_accuracy", "threshold", "recovery_index",
                       "recovery_time", "status"]
    embedded_recovery_counts = Counter()
    for row in raw.to_dict("records"):
        for recovery in json.loads(row["recovery_records_json"]):
            values = [row["dataset"], row["strategy"], int(row["seed"])]
            values.extend(recovery.get(field) for field in recovery_fields[3:])
            embedded_recovery_counts[tuple(_counter_value(value) for value in values)] += 1
    csv_recovery_counts = Counter(
        tuple(_counter_value(getattr(row, field)) for field in recovery_fields)
        for row in recoveries.itertuples(index=False)
    )
    if embedded_recovery_counts != csv_recovery_counts:
        raise ValueError("recoveries.csv does not match raw recovery JSON (within CSV precision)")
    if len(recoveries) != sum(embedded_recovery_counts.values()):
        raise ValueError("recoveries.csv record count does not match raw recovery records")

    run_ids = raw.run_id.dropna().unique().tolist()
    if len(run_ids) != 1 or run_ids[0] != config["experiment"]["run_id"]:
        raise ValueError("config.yaml run_id does not match raw_runs.csv")
    if config["experiment"]["mlflow_experiment"] != "concept-drift-phase4":
        raise ValueError("config.yaml has an unexpected MLflow experiment")
    if not {"dataset", "strategy", "accuracy_mean", "macro_f1_mean"}.issubset(phase3.columns):
        raise ValueError("Phase 3 summary is missing expected comparison fields")

    audit = {
        "rows": len(raw), "unique_keys": raw[key].drop_duplicates().shape[0],
        "duplicates": duplicate_count, "datasets": raw.dataset.nunique(),
        "strategies": raw.strategy.nunique(), "conditions": len(groups),
        "seeds": raw.seed.nunique(), "missing_metrics": missing_metrics.to_dict(),
        "valid_predictions": expected_valid, "events": len(events),
        "recoveries": len(recoveries),
        "recovery_delta_column_missing": "adwin_delta" not in recoveries.columns,
    }
    return config, raw, summary, events, recoveries, phase3, audit


def _mean_std(values, pd):
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else float("nan")
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return mean, std


def _condition_summary(raw, pd, np):
    rows = []
    groups = raw.groupby(["dataset", "strategy", "adwin_delta"],
                         sort=True, dropna=False)
    fields = ["accuracy", "macro_f1", "runtime", "peak_memory",
              "drift_detections", "n_valid_predictions", "mean_recovery"]
    for (dataset, strategy, delta), group in groups:
        row = {"dataset": dataset, "strategy": strategy,
               "adwin_delta": _delta(delta, np), "n_runs": int(len(group))}
        for metric in fields:
            row[f"{metric}_mean"], row[f"{metric}_std"] = _mean_std(group[metric], pd)
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_with_factors(raw, np, pd):
    out = raw.copy()
    factors = out.strategy.map(_strategy_factors)
    out["architecture"] = factors.map(lambda x: x[0])
    out["adaptation_policy"] = factors.map(lambda x: x[1])
    out["adwin_delta"] = out.adwin_delta.map(lambda x: _delta(x, np))
    out["drift_regime"] = out.dataset.map({
        "sea_abrupt": "abrupt synthetic drift",
        "sea_gradual": "gradual synthetic drift",
        "hyperplane": "continuous synthetic drift",
    })
    return out


def _condition_lookup(raw):
    lookup = {}
    for row in raw.to_dict("records"):
        delta = row["adwin_delta"]
        if delta is not None and isinstance(delta, float) and math.isnan(delta):
            delta = None
        else:
            delta = None if delta is None else float(delta)
        lookup[(row["dataset"], row["strategy"], delta, int(row["seed"]))] = row
    return lookup


def _paired_effect(differences, stats, np):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if not n:
        return {"n_paired": 0, "mean_difference": np.nan,
                "median_difference": np.nan, "sd_difference": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
                "p_value": np.nan, "paired_rank_biserial": np.nan,
                "test_method": "Wilcoxon signed-rank, two-sided"}
    mean = float(np.mean(differences))
    median = float(np.median(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else float("nan")
    if n > 1 and np.isfinite(sd):
        margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        ci_low, ci_high = mean - margin, mean + margin
    else:
        ci_low = ci_high = np.nan
    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(
            differences, zero_method="wilcox", alternative="two-sided",
            method="auto",
        ).pvalue)
    nonzero = differences[differences != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero), method="average")
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        rank_biserial = np.nan
    return {
        "n_paired": n, "mean_difference": mean,
        "median_difference": median, "sd_difference": sd,
        "ci95_low": ci_low, "ci95_high": ci_high,
        "p_value": p_value,
        "paired_rank_biserial": float(rank_biserial),
        "test_method": "Wilcoxon signed-rank, two-sided, scipy method=auto",
    }


def _comparison_specs(datasets):
    specs = []
    for dataset in datasets:
        # Architecture contrasts keep policy, delta, and seed matched.
        specs.append(("architecture", dataset, "online", None,
                      "lstm", None, "gru", None))
        for policy in ("adwin_reset", "adwin_finetune"):
            for delta in DELTAS:
                specs.append(("architecture", dataset, policy, delta,
                              f"lstm_{policy}", delta,
                              f"gru_{policy}", delta))

        # Policy contrasts are evaluated separately at each applicable delta.
        for architecture in ("lstm", "gru"):
            online = architecture
            for delta in DELTAS:
                reset = f"{architecture}_adwin_reset"
                fine = f"{architecture}_adwin_finetune"
                specs.extend([
                    ("adaptation_policy", dataset, architecture, delta,
                     online, None, reset, delta),
                    ("adaptation_policy", dataset, architecture, delta,
                     online, None, fine, delta),
                    ("adaptation_policy", dataset, architecture, delta,
                     reset, delta, fine, delta),
                ])

            # Detector sensitivity contrasts keep architecture and policy fixed.
            for policy in ("adwin_reset", "adwin_finetune"):
                strategy = f"{architecture}_{policy}"
                for delta_a, delta_b in itertools.combinations(DELTAS, 2):
                    specs.append(("detector_sensitivity", dataset,
                                  architecture, policy, strategy, delta_a,
                                  strategy, delta_b))
    return specs


def _holm_adjust(rows, pd):
    frame = pd.DataFrame(rows)
    frame["p_adjusted"] = float("nan")
    valid = frame.p_value.notna()
    for _, indices in frame.loc[valid].groupby(["family", "metric"]).groups.items():
        ordered = frame.loc[indices].sort_values("p_value")
        m = len(ordered)
        adjusted = []
        running = 0.0
        for rank, value in enumerate(ordered.p_value.tolist(), start=1):
            running = max(running, min(1.0, (m - rank + 1) * float(value)))
            adjusted.append(running)
        frame.loc[ordered.index, "p_adjusted"] = adjusted
    return frame


def _paired_comparisons(raw, stats, np, pd):
    lookup = _condition_lookup(raw)
    rows = []
    for spec in _comparison_specs(sorted(raw.dataset.unique())):
        family, dataset, context, context_value, left, left_delta, right, right_delta = spec
        left_delta = None if left_delta is None else float(left_delta)
        right_delta = None if right_delta is None else float(right_delta)
        for metric in PAIR_METRICS:
            differences = []
            detector_metric_applicable = not (
                metric == "drift_detections"
                and (left_delta is None or right_delta is None)
            )
            if detector_metric_applicable:
                for seed in SEEDS:
                    a = lookup[(dataset, left, left_delta, seed)][metric]
                    b = lookup[(dataset, right, right_delta, seed)][metric]
                    if pd.notna(a) and pd.notna(b):
                        differences.append(float(b) - float(a))
            result = _paired_effect(differences, stats, np)
            rows.append({
                "family": family, "dataset": dataset,
                "factor_context": context,
                "context_value": context_value,
                "condition_a": left, "delta_a": left_delta,
                "condition_b": right, "delta_b": right_delta,
                "contrast_direction": "condition_b minus condition_a",
                "not_applicable_reason": (
                    "ADWIN detector absent in at least one condition"
                    if not detector_metric_applicable else None
                ),
                "metric": metric, **result,
            })
    frame = _holm_adjust(rows, pd)
    frame["holm_family"] = frame["family"] + " × " + frame["metric"]
    frame["multiple_comparison_correction"] = "Holm step-down within family × metric"
    frame["small_sample_note"] = "Five matched seeds; low power; interpret p-values cautiously."
    return frame


def _flatten_recoveries(raw, np, pd):
    rows = []
    for row in raw.to_dict("records"):
        architecture, policy = _strategy_factors(row["strategy"])
        delta = _delta(row["adwin_delta"], np)
        for item in json.loads(row["recovery_records_json"]):
            rows.append({
                "dataset": row["dataset"], "strategy": row["strategy"],
                "architecture": architecture, "adaptation_policy": policy,
                "adwin_delta": delta, "seed": int(row["seed"]),
                **item,
            })
    return pd.DataFrame(rows)


def _recovery_analysis(flat, pd, np):
    rows = []
    keys = ["dataset", "strategy", "adwin_delta"]
    for (dataset, strategy, delta), group in flat.groupby(keys, dropna=False, sort=True):
        recovered = group.loc[group.status.eq("recovered"), "recovery_time"]
        mean, std = _mean_std(recovered, pd)
        rows.append({
            "dataset": dataset, "strategy": strategy,
            "delta": _delta(delta, np), "mean_recovery": mean,
            "std_recovery": std,
            "n_recovery_observations": int(pd.to_numeric(recovered, errors="coerce").notna().sum()),
            "n_recovery_records": int(len(group)),
            "record_statuses": ";".join(sorted(group.status.dropna().unique())),
        })
    return pd.DataFrame(rows)


def _phase_comparison(phase3, condition_summary, np, pd):
    controlled = condition_summary.loc[
        condition_summary.adwin_delta.isna() | condition_summary.adwin_delta.eq(0.002)
    ].copy()
    phase3_neural = phase3.loc[
        phase3.strategy.isin({"lstm", "lstm_adwin_reset", "lstm_adwin_finetune",
                              "gru", "gru_adwin_reset", "gru_adwin_finetune"})
        & phase3.dataset.isin(DATASETS)
    ].copy()
    keep3 = ["dataset", "strategy", "n_runs", "n_independent_runs", "n_seeds",
             *PHASE3_METRICS]
    phase3_neural = phase3_neural[[column for column in keep3 if column in phase3_neural.columns]]
    phase3_neural = phase3_neural.rename(columns={
        column: f"phase3_{column}" for column in phase3_neural.columns
        if column not in {"dataset", "strategy"}
    })
    phase4_keep = ["dataset", "strategy", "adwin_delta", "n_runs",
                   "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
                   "runtime_mean", "runtime_std", "peak_memory_mean", "peak_memory_std",
                   "drift_detections_mean", "drift_detections_std",
                   "n_valid_predictions_mean", "n_valid_predictions_std",
                   "mean_recovery_mean", "mean_recovery_std"]
    controlled = controlled[phase4_keep].rename(columns={
        column: f"phase4_{column}" for column in phase4_keep
        if column not in {"dataset", "strategy"}
    })
    merged = phase3_neural.merge(controlled, on=["dataset", "strategy"], how="outer")
    merged.insert(0, "phase3_scope", "Phase 3 broad benchmark")
    merged.insert(1, "phase4_scope", "Phase 4 controlled neural ablation")
    merged["comparison_note"] = (
        "Related settings; separate experiment phases and scopes, not the same experiment."
    )
    return merged


def _write_observations(output, condition_summary, comparisons, adwin, recovery, pd):
    lines = [
        "Phase 4 descriptive observations",
        "",
        "Descriptive results",
        "Condition means are calculated over the five matched seed runs.",
    ]
    for row in condition_summary.sort_values(
            ["dataset", "strategy", "adwin_delta"], na_position="first").itertuples(index=False):
        delta_text = "not applicable" if pd.isna(row.adwin_delta) else f"{row.adwin_delta:g}"
        lines.append(
            f"On {row.dataset}, {row.strategy} (delta {delta_text}) had mean accuracy "
            f"{row.accuracy_mean:.6f}, mean macro-F1 {row.macro_f1_mean:.6f}, "
            f"mean runtime {row.runtime_mean:.3f} seconds, and mean ADWIN detections "
            f"{row.drift_detections_mean:.3f}."
        )
    for row in recovery.sort_values(["dataset", "strategy", "delta"],
                                    na_position="first").itertuples(index=False):
        delta_text = "not applicable" if pd.isna(row.delta) else f"{row.delta:g}"
        if row.dataset == "hyperplane":
            lines.append(
                f"On hyperplane, recovery is not applicable for {row.strategy} "
                f"(delta {delta_text}); the stream has no discrete known drift index."
            )
        elif row.n_recovery_observations:
            lines.append(
                f"On {row.dataset}, {row.strategy} (delta {delta_text}) had mean "
                f"recovery time {row.mean_recovery:.1f} samples across "
                f"{row.n_recovery_observations} recovered drift observations."
            )
        else:
            lines.append(
                f"On {row.dataset}, {row.strategy} (delta {delta_text}) had no "
                "recovered drift observations in the recorded recovery metric."
            )
    lines.extend([
        "",
        "Statistical results",
        "Paired test estimates, 95% paired-mean confidence intervals, raw p-values, "
        "and Holm-adjusted p-values are in paired_comparisons.csv. Tests use matched "
        "seeds and report n=5 when all values are available. Five pairs provide low "
        "power; p-values are reported without broad significance claims. The 95% "
        "interval is a two-sided t interval over paired differences. Effect size is "
        "paired rank-biserial correlation: (positive signed-rank sum minus negative "
        "signed-rank sum) divided by the sum of nonzero absolute-difference ranks. "
        "Holm step-down correction is applied separately within each comparison-family "
        "and metric combination.",
    ])
    for family in ("architecture", "adaptation_policy", "detector_sensitivity"):
        for metric in ("accuracy", "macro_f1"):
            subset = comparisons.loc[
                comparisons.family.eq(family) & comparisons.metric.eq(metric)
                & comparisons.p_value.notna()
            ]
            raw_below = int(subset.p_value.lt(0.05).sum())
            adjusted_below = int(subset.p_adjusted.lt(0.05).sum())
            lines.append(
                f"{family} / {metric}: {raw_below} of {len(subset)} paired tests have "
                f"raw p < 0.05; {adjusted_below} have Holm-adjusted p < 0.05."
            )
    lines.extend([
        "",
        "Interpretation",
        "These are descriptive and paired comparisons within the configured benchmark. "
        "They do not establish universal rankings or causal claims. SEA gradual recovery "
        "uses the existing 200-sample evaluation resolution and is coarse. Hyperplane "
        "has no discrete drift index, so recovery is not applicable.",
        "",
        "Recovery values in this analysis are read from raw_runs.csv recovery_records_json "
        "because the frozen recoveries.csv schema omits adwin_delta.",
    ])
    (output / "observations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_dir, output_dir, phase3_summary_path):
    np, pd, scipy, stats, yaml = _load_dependencies()
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    phase3_summary_path = pathlib.Path(phase3_summary_path).expanduser().resolve()
    if output_dir == input_dir:
        raise ValueError("Analysis output must be separate from the frozen Phase 4 run directory")
    config, raw_source, summary_source, events, frozen_recoveries, phase3, audit = _validate_artifacts(
        input_dir, phase3_summary_path, np, pd, yaml
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _raw_with_factors(raw_source, np, pd)
    condition_summary = _condition_summary(raw, pd, np)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)

    dataset_summary = condition_summary.copy()
    factors = dataset_summary.strategy.map(_strategy_factors)
    dataset_summary["architecture"] = factors.map(lambda x: x[0])
    dataset_summary["adaptation_policy"] = factors.map(lambda x: x[1])
    dataset_summary["drift_regime"] = dataset_summary.dataset.map(raw.set_index("dataset").drift_regime.to_dict())
    dataset_summary["dataset_run_count"] = dataset_summary.dataset.map(
        raw.groupby("dataset").size().to_dict()
    )
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    paired = _paired_comparisons(raw_source, stats, np, pd)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    paired.loc[paired.family.eq("architecture")].to_csv(
        output_dir / "architecture_effect.csv", index=False
    )
    paired.loc[paired.family.eq("adaptation_policy")].to_csv(
        output_dir / "adaptation_policy_effect.csv", index=False
    )
    paired.loc[paired.family.eq("detector_sensitivity")].to_csv(
        output_dir / "detector_sensitivity_effect.csv", index=False
    )

    drift_regime = dataset_summary.copy()
    drift_regime.to_csv(output_dir / "drift_regime_effect.csv", index=False)

    adwin_source = raw.copy()
    adwin_source["detector_detections"] = adwin_source.drift_detections.where(
        adwin_source.adwin_delta.notna()
    )
    adwin = (adwin_source.groupby(["dataset", "architecture", "adaptation_policy",
                                   "adwin_delta", "adwin_clock"],
                                  dropna=False, sort=True)
             .agg(n_runs=("seed", "count"),
                  n_detector_runs=("detector_detections", "count"),
                  drift_detections_mean=("detector_detections", "mean"),
                  drift_detections_std=("detector_detections", "std"),
                  drift_detections_total=("detector_detections", "sum"))
             .reset_index()
             .assign(detector_applicable=lambda frame: frame.adwin_delta.notna()))
    adwin.loc[~adwin.detector_applicable, "drift_detections_total"] = np.nan
    adwin.to_csv(output_dir / "adwin_sensitivity.csv", index=False)

    flat_recoveries = _flatten_recoveries(raw_source, np, pd)
    recovery = _recovery_analysis(flat_recoveries, pd, np)
    recovery.to_csv(output_dir / "recovery_analysis.csv", index=False)
    flat_recoveries.to_csv(output_dir / "plot_recovery.csv", index=False)

    computational = (raw.groupby(["dataset", "architecture", "adaptation_policy",
                                   "strategy", "adwin_delta"], dropna=False, sort=True)
                      .agg(n_runs=("seed", "count"),
                           runtime_mean=("runtime", "mean"),
                           runtime_std=("runtime", "std"),
                           peak_memory_mean=("peak_memory", "mean"),
                           peak_memory_std=("peak_memory", "std"))
                      .reset_index())
    computational.to_csv(output_dir / "computational_cost.csv", index=False)

    comparison = _phase_comparison(phase3, condition_summary, np, pd)
    comparison.to_csv(output_dir / "phase3_phase4_comparison.csv", index=False)

    for metric, filename in (("accuracy", "plot_accuracy.csv"),
                             ("macro_f1", "plot_macro_f1.csv"),
                             ("runtime", "plot_runtime.csv")):
        plot_data = raw[["dataset", "drift_regime", "strategy", "architecture",
                         "adaptation_policy", "adwin_delta", "seed", metric]].rename(
                             columns={metric: "value"})
        plot_data.to_csv(output_dir / filename, index=False)
    raw[["dataset", "strategy", "architecture", "adaptation_policy",
         "adwin_delta", "adwin_clock", "seed", "drift_detections"]].assign(
             detector_applicable=lambda frame: frame.adwin_delta.notna(),
             drift_detections=lambda frame: frame.drift_detections.where(
                 frame.adwin_delta.notna()
             ),
         ).to_csv(
             output_dir / "plot_adwin.csv", index=False
         )

    _write_observations(output_dir, condition_summary, paired, adwin, recovery, pd)
    (output_dir / "integrity_audit.json").write_text(
        json.dumps({**audit, "phase3_summary": str(phase3_summary_path),
                    "analysis_script": "scripts/analyze_phase4.py",
                    "scipy_version": scipy.__version__}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis outputs written to: {output_dir}")
    print(f"Integrity: {audit['rows']} runs; {audit['unique_keys']} unique keys; "
          f"{audit['duplicates']} duplicates; {len(events)} drift events; "
          f"{len(frozen_recoveries)} recovery records")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="frozen Phase 4 run directory")
    parser.add_argument("--output", default=None,
                        help="analysis output directory (default: <input>/analysis)")
    parser.add_argument("--phase3-summary", default=str(DEFAULT_PHASE3_SUMMARY),
                        help="frozen Phase 3 summary.csv for contextual comparison")
    args = parser.parse_args()
    output = args.output or str(pathlib.Path(args.input) / "analysis")
    analyze(args.input, output, args.phase3_summary)


if __name__ == "__main__":
    main()
