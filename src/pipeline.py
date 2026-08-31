# src/pipeline.py
"""Prequential (test-then-train) evaluation of drift adaptation strategies."""
import pathlib
import tracemalloc
from collections import Counter, deque
from time import perf_counter

import mlflow
from river import forest, metrics, tree
from river.drift import ADWIN

# Local SQLite backend: notebooks work whether or not a tracking server is up.
# Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
# MLflow >= 3.x rejects the plain './mlruns' file store, so SQLite is the
# serverless option; point this at http://localhost:5000 to use a server.
TRACKING_URI = f"sqlite:///{pathlib.Path(__file__).resolve().parent.parent / 'mlflow.db'}"
EXPERIMENT = "concept-drift-adaptation"


def base_learner(seed=None):
    # Hoeffding Trees are deterministic; seed is accepted so every model
    # factory shares one signature.
    return tree.HoeffdingTreeClassifier()


# --- strategy configurations -------------------------------------------------
# Each strategy is data, not a branch inside the loop: the loop reads these
# flags, so adding a strategy never means editing the evaluation code.
#
#   reset_on_drift  - detector fires -> discard model
#   refit_on_reset  - after a reset, refit on the buffered recent samples
#                     (a real retrain, rather than starting from nothing)
#   retrain_every   - blind periodic reset, ignoring any detector
#   window_size     - how many recent samples the buffer holds
STRATEGIES = {
    "baseline": dict(),
    "blind_periodic": dict(retrain_every=1000, refit_on_reset=True),
    "detector_full_retrain": dict(detector=True, reset_on_drift=True,
                                  refit_on_reset=True),
    "detector_incremental": dict(detector=True, reset_on_drift=False),
    # ARF's seed flows through from the run so its across-seed variance is
    # real; hardcoding it would make the ensemble identical on every seed and
    # understate its spread.
    "arf": dict(model_fn=lambda seed=None: forest.ARFClassifier(seed=seed)),
}


def recovery_time(curve, drift_idx, tolerance=0.02):
    """Samples from a drift until windowed accuracy returns to its pre-drift level.

    curve is [(index, windowed_accuracy), ...]. Pre-drift level is the last
    windowed reading strictly before the drift. Returns None when accuracy
    never recovers within the stream - which is itself a finding, and must be
    reported as non-recovery rather than folded into a mean.
    """
    pre = [a for i, a in curve if i < drift_idx]
    if not pre:
        return None
    target = pre[-1] - tolerance
    for i, a in curve:
        if i > drift_idx and a >= target:
            return i - drift_idx
    return None


def run_strategy(samples, strategy_name, dataset_name, drift_points=(),
                 seed=42, window=500, eval_every=200, log_to_mlflow=True,
                 delay=0):
    """Run one (dataset, strategy, seed) configuration prequentially.

    delay: how many steps to hold a sample before learning from it. Use this
    when the label is not knowable at prediction time - a forecast of the next
    period's outcome, for instance, whose true value only arrives later.
    delay=0 (the default) is standard prequential: predict, then immediately
    learn. delay=1 predicts at t but learns (x_t, y_t) only at t+1, which is
    what a next-step forecast actually permits. Learning a forward-looking
    label at t trains the model on the future and inflates accuracy.

    Returns a dict of results; also logs params and metrics to MLflow.
    """
    cfg = STRATEGIES[strategy_name]
    _factory = cfg.get("model_fn", base_learner)
    def model_fn():
        return _factory(seed=seed)
    detector = ADWIN() if cfg.get("detector") else None
    reset_on_drift = cfg.get("reset_on_drift", False)
    refit_on_reset = cfg.get("refit_on_reset", False)
    retrain_every = cfg.get("retrain_every")

    model = model_fn()
    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()
    buffer = deque(maxlen=window)

    # Always-predict-majority rate: the floor any strategy must clear to have
    # learned anything. Logged with every run so a near-chance result cannot be
    # mistaken for a real one.
    _labels = Counter(y for _, y in samples)
    majority_baseline = max(_labels.values()) / sum(_labels.values())

    detected_drifts = []
    curve = []          # (index, windowed accuracy) - windowed, so a local
    win_correct = win_n = 0   # dip at the drift is visible instead of averaged away
    pending = deque()   # samples awaiting their label under `delay`

    tracemalloc.start()
    start = perf_counter()

    for i, (x, y) in enumerate(samples):
        y_pred = model.predict_one(x)

        if y_pred is not None:
            acc.update(y, y_pred)
            f1.update(y, y_pred)
            win_correct += int(y_pred == y)
            win_n += 1
            # Feed the detector only on real predictions. Counting the warmup
            # (y_pred is None) as an error would inject a false error signal
            # and can trip the detector before any drift exists.
            if detector is not None:
                detector.update(int(y_pred != y))
                if detector.drift_detected:
                    detected_drifts.append(i)
                    if reset_on_drift:
                        model = model_fn()
                        if refit_on_reset:
                            for bx, by in buffer:
                                model.learn_one(bx, by)

        if retrain_every and i > 0 and i % retrain_every == 0:
            model = model_fn()
            if refit_on_reset:
                for bx, by in buffer:
                    model.learn_one(bx, by)

        # Under delay>0 the pair (x, y) is not learnable yet: at time i its
        # label still lies in the future. Hold it and learn it once it would
        # genuinely be known.
        if delay:
            pending.append((x, y))
            if len(pending) > delay:
                lx, ly = pending.popleft()
                model.learn_one(lx, ly)
                buffer.append((lx, ly))
        else:
            model.learn_one(x, y)
            buffer.append((x, y))

        if win_n and (i + 1) % eval_every == 0:
            curve.append((i + 1, win_correct / win_n))
            win_correct = win_n = 0

    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recoveries = [recovery_time(curve, d) for d in drift_points]
    measured = [r for r in recoveries if r is not None]

    results = {
        "dataset": dataset_name,
        "strategy": strategy_name,
        "seed": seed,
        "final_accuracy": acc.get(),
        "final_macro_f1": f1.get(),
        "runtime_sec": elapsed,
        "peak_memory_mb": peak / 1e6,
        "n_drift_detections": len(detected_drifts),
        "detected_drifts": detected_drifts,
        "majority_baseline": majority_baseline,
        "lift_over_majority": acc.get() - majority_baseline,
        "recoveries": recoveries,
        "curve": curve,
    }

    if log_to_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"{dataset_name}_{strategy_name}_s{seed}"):
            mlflow.log_params({
                "dataset": dataset_name,
                "strategy": strategy_name,
                "base_learner": model_fn().__class__.__name__,
                "seed": seed,
                "window": window,
                "retrain_every": retrain_every,
                "detector": "ADWIN" if detector is not None else None,
                "n_samples": len(samples),
                "delay": delay,
                "true_drift_points": list(drift_points),
            })
            for idx, a in curve:
                mlflow.log_metric("windowed_accuracy", a, step=idx)
            mlflow.log_metrics({
                "final_accuracy": results["final_accuracy"],
                "final_macro_f1": results["final_macro_f1"],
                "runtime_sec": results["runtime_sec"],
                "peak_memory_mb": results["peak_memory_mb"],
                "n_drift_detections": results["n_drift_detections"],
                "majority_baseline": majority_baseline,
                "lift_over_majority": results["lift_over_majority"],
            })
            # Logged only when measurable: absent on gradual/real-world streams
            # with no ground-truth drift index, and when a strategy never
            # recovers. n_recovered exposes which of those two it was.
            if measured:
                mlflow.log_metric("mean_recovery_samples",
                                  sum(measured) / len(measured))
            if drift_points:
                mlflow.log_metric("n_recovered", len(measured))

    return results
