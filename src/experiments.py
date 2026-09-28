# src/experiments.py
"""Grid runner: datasets x strategies x seeds, all logged to MLflow."""
import statistics
import sys
from collections import Counter

import pandas as pd

import streams
from pipeline import STRATEGIES, run_strategy

SEEDS = (42, 43, 44)

# Strategies whose only randomness is the model's own. On a fixed real-world
# dataset every other strategy is fully deterministic, so repeating it across
# seeds yields byte-identical runs.
STOCHASTIC_STRATEGIES = {"arf", "lstm", "gru"}

# Streams whose label is not knowable at prediction time. stock_sp500's target
# is next-day direction, so learning it at t would train on the future; delay=1
# holds each sample until its label would genuinely have arrived.
STREAM_DELAY = {"stock_sp500": 1}

# Minimum lift over the majority class for a result to count as having learned
# anything. One percentage point; below this the "win" is noise.
MIN_LIFT = 0.01


def majority_rate(samples):
    """Accuracy of always predicting the most common label."""
    counts = Counter(y for _, y in samples)
    return max(counts.values()) / sum(counts.values())


def _seeds_for(ds_name, strat, seeds):
    """Seeds to run for one (dataset, strategy) pair.

    Synthetic streams regenerate from the seed, so every strategy gets the full
    set - the varying stream is what makes repeats a real sample.

    Fixed real-world datasets (Elec2, Airlines) are the same rows in the same
    order every time. There, only a stochastic model varies; a deterministic
    strategy would produce identical numbers on all three seeds, and reporting
    that as "mean +/- 0.000 over 3 seeds" overstates a single run as replicated
    evidence. So those run once.
    """
    if streams.is_deterministic(ds_name) and strat not in STOCHASTIC_STRATEGIES:
        return seeds[:1]
    return seeds


def run_grid(datasets=None, strategies=None, seeds=SEEDS, log_to_mlflow=True,
             verbose=True):
    """Run every (dataset, strategy, seed) combination.

    Returns a tidy DataFrame, one row per run. Seed counts vary by dataset -
    see _seeds_for.
    """
    datasets = datasets or list(streams.STREAMS)
    strategies = strategies or list(STRATEGIES)
    rows = []

    for ds_name in datasets:
        # Fixed datasets are loaded once and reused; regenerating per seed
        # would re-read the same rows and, for Airlines, re-parse 100k of them.
        cache = None
        if streams.is_deterministic(ds_name):
            cache = streams.STREAMS[ds_name]()

        for strat in strategies:
            for seed in _seeds_for(ds_name, strat, seeds):
                samples, drift = (cache if cache is not None
                                  else streams.STREAMS[ds_name](seed=seed))
                r = run_strategy(samples, strat, ds_name, drift, seed=seed,
                                 log_to_mlflow=log_to_mlflow,
                                 delay=STREAM_DELAY.get(ds_name, 0))
                measured = [x for x in r["recoveries"] if x is not None]
                rows.append({
                    "dataset": ds_name,
                    "strategy": strat,
                    "seed": seed,
                    "final_accuracy": r["final_accuracy"],
                    "final_macro_f1": r["final_macro_f1"],
                    "runtime_sec": r["runtime_sec"],
                    "peak_memory_mb": r["peak_memory_mb"],
                    "n_drift_detections": r["n_drift_detections"],
                    "mean_recovery": (statistics.mean(measured)
                                      if measured else None),
                    "n_drifts": len(drift),
                    "n_recovered": len(measured),
                    # Always-predict-the-majority-class rate. A strategy that
                    # does not clear this has learned only the class prior;
                    # on near-unpredictable streams every strategy lands here,
                    # and a ranking among them is a ranking of noise.
                    "majority_baseline": majority_rate(samples),
                })
                if verbose:
                    print(f"{ds_name:14s} {strat:22s} seed={seed} "
                          f"acc={r['final_accuracy']:.4f} "
                          f"t={r['runtime_sec']:5.1f}s", flush=True)

    return pd.DataFrame(rows)


def summarise(df):
    """Mean +/- std across seeds, per (dataset, strategy).

    Report the std alongside the mean - with 3 seeds, differences smaller than
    the spread are not results.
    """
    agg = df.groupby(["dataset", "strategy"]).agg(
        acc_mean=("final_accuracy", "mean"),
        acc_std=("final_accuracy", "std"),
        f1_mean=("final_macro_f1", "mean"),
        recovery_mean=("mean_recovery", "mean"),
        runtime_mean=("runtime_sec", "mean"),
        memory_mean=("peak_memory_mb", "mean"),
        n_recovered=("n_recovered", "sum"),
        n_drifts=("n_drifts", "sum"),
        n_runs=("seed", "count"),
        majority_baseline=("majority_baseline", "first"),
    ).reset_index()
    # Negative means the strategy is worse than always guessing the majority
    # class - it has learned the prior and nothing more.
    agg["lift_over_majority"] = agg["acc_mean"] - agg["majority_baseline"]
    return agg.sort_values(["dataset", "acc_mean"], ascending=[True, False])


def best_per_dataset(agg):
    """Highest mean accuracy per dataset, with whether the win is separable.

    separable is True only when the margin over the runner-up exceeds the
    winner's own spread across seeds. It is None - not False - when either
    side ran once and has no spread to compare against: an unreplicated
    single run cannot support a separability claim in either direction, and
    collapsing that to False would read as "we checked, they are tied".
    """
    out = []
    for ds, grp in agg.groupby("dataset"):
        grp = grp.sort_values("acc_mean", ascending=False)
        top = grp.iloc[0]
        row = {"dataset": ds, "best_strategy": top["strategy"],
               "acc_mean": top["acc_mean"], "acc_std": top["acc_std"],
               "n_runs": top.get("n_runs")}
        if len(grp) > 1:
            second = grp.iloc[1]
            margin = top["acc_mean"] - second["acc_mean"]
            row["runner_up"] = second["strategy"]
            row["margin"] = margin
            std = top["acc_std"]
            row["separable"] = (None if pd.isna(std)
                                else bool(margin > std))
        # A winner that does not clear the majority class has learned the prior
        # and nothing more; the ranking above it is noise. The threshold is not
        # zero: a lift of +0.0008 is arithmetically positive and practically
        # indistinguishable from guessing, so require at least MIN_LIFT.
        lift = top.get("lift_over_majority")
        if lift is not None and not pd.isna(lift):
            row["lift_over_majority"] = lift
            row["beats_majority"] = bool(lift > MIN_LIFT)
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import pathlib

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    df = run_grid()
    df.to_csv("results/raw_runs.csv", index=False)

    agg = summarise(df)
    agg.to_csv("results/summary.csv", index=False)

    print("\n=== summary ===")
    print(agg.to_string(index=False))

    print("\n=== best per dataset ===")
    print(best_per_dataset(agg).to_string(index=False))
