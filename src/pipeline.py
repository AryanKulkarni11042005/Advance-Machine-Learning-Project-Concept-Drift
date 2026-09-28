# src/pipeline.py
"""Prequential (test-then-train) evaluation of drift adaptation strategies."""
import pathlib
import tracemalloc
from collections import Counter, deque
from time import perf_counter

import mlflow
from river import forest, metrics, tree
from river.drift import ADWIN
from models.neural_adapter import NeuralAdapter
from models.sequence import RollingSequence

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

# Kept separate so the existing full benchmark grid remains unchanged. Neural
# strategies are opt-in until categorical preprocessing and adaptation policy
# are designed in a later phase.
NEURAL_STRATEGIES = {
    "lstm": dict(architecture="lstm", adaptation="online"),
    "gru": dict(architecture="gru", adaptation="online"),
    "lstm_adwin_reset": dict(architecture="lstm", adaptation="adwin_reset",
                              detector=True),
    "gru_adwin_reset": dict(architecture="gru", adaptation="adwin_reset",
                             detector=True),
    "lstm_adwin_finetune": dict(architecture="lstm",
                                 adaptation="adwin_finetune", detector=True,
                                 replay_buffer_size=32),
    "gru_adwin_finetune": dict(architecture="gru",
                                adaptation="adwin_finetune", detector=True,
                                replay_buffer_size=32),
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
    neural_cfg = NEURAL_STRATEGIES.get(strategy_name)
    if neural_cfg:
        first_features = samples[0][0]
        feature_names = tuple(first_features.keys())
        sequence_builder = RollingSequence(sequence_length=10,
                                           feature_names=feature_names)
        model = NeuralAdapter(
            architecture=neural_cfg["architecture"],
            input_size=len(feature_names), seed=seed, sequence_length=10,
        )
        neural_buffer = deque(maxlen=neural_cfg.get("replay_buffer_size", 32))
        model_fn = None
        cfg = {}
    else:
        cfg = STRATEGIES[strategy_name]
        _factory = cfg.get("model_fn", base_learner)
        def model_fn():
            return _factory(seed=seed)
        model = model_fn()
        sequence_builder = None
    detector_enabled = (neural_cfg.get("detector", False) if neural_cfg
                        else cfg.get("detector", False))
    detector = ADWIN() if detector_enabled else None
    reset_on_drift = cfg.get("reset_on_drift", False)
    refit_on_reset = cfg.get("refit_on_reset", False)
    retrain_every = cfg.get("retrain_every")

    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()
    buffer = deque(maxlen=window)

    # Always-predict-majority rate: the floor any strategy must clear to have
    # learned anything. Logged with every run so a near-chance result cannot be
    # mistaken for a real one.
    _labels = Counter(y for _, y in samples)
    majority_baseline = max(_labels.values()) / sum(_labels.values())

    detected_drifts = []
    valid_predictions = 0
    curve = []          # (index, windowed accuracy) - windowed, so a local
    win_correct = win_n = 0   # dip at the drift is visible instead of averaged away
    pending = deque()   # samples awaiting their label under `delay`

    def adapt_neural_from_error(error, detection_index):
        if detector is None or error is None:
            return
        detector.update(error)
        if detector.drift_detected:
            detected_drifts.append(detection_index)
            if neural_cfg["adaptation"] == "adwin_reset":
                model.reset()
            elif neural_cfg["adaptation"] == "adwin_finetune":
                # Replay only labels already revealed before this detection.
                for buffered_sequence, buffered_label in neural_buffer:
                    model.update(buffered_sequence, buffered_label)

    tracemalloc.start()
    start = perf_counter()

    for i, (x, y) in enumerate(samples):
        if sequence_builder:
            sequence = sequence_builder.append(x)
            y_pred = model.predict(sequence) if sequence is not None else None
        else:
            sequence = None
            y_pred = model.predict_one(x)

        if y_pred is not None:
            valid_predictions += 1
            acc.update(y, y_pred)
            f1.update(y, y_pred)
            win_correct += int(y_pred == y)
            win_n += 1
            # Feed the detector only on real predictions. Counting the warmup
            # (y_pred is None) as an error would inject a false error signal
            # and can trip the detector before any drift exists.
            if detector is not None and not sequence_builder:
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
            if sequence_builder:
                if sequence is not None:
                    # Score immediately as the existing prequential loop does,
                    # but withhold detector feedback and training until label
                    # arrival. This prevents delayed labels from leaking.
                    pending.append((sequence, y,
                                    int(y_pred != y) if y_pred is not None else None))
            else:
                pending.append((x, y))
            if len(pending) > delay:
                item = pending.popleft()
                if sequence_builder:
                    lx, ly, error = item
                    adapt_neural_from_error(error, i)
                    model.update(lx, ly)
                    neural_buffer.append((lx, ly))
                else:
                    lx, ly = item
                    model.learn_one(lx, ly)
                    buffer.append((lx, ly))
        else:
            if sequence_builder:
                if sequence is not None:
                    if detector is not None:
                        adapt_neural_from_error(int(y_pred != y), i)
                    model.update(sequence, y)
                    neural_buffer.append((sequence, y))
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
        "n_valid_predictions": valid_predictions,
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
                "base_learner": (model.metadata["model_type"] if sequence_builder
                                 else model_fn().__class__.__name__),
                "seed": seed,
                "window": window,
                "retrain_every": retrain_every,
                "detector": "ADWIN" if detector is not None else None,
                "n_samples": len(samples),
                "delay": delay,
                "true_drift_points": list(drift_points),
            })
            if sequence_builder:
                mlflow.log_params({
                    "model_family": model.metadata["model_type"],
                    "sequence_length": model.metadata["sequence_length"],
                    "hidden_size": model.metadata["hidden_size"],
                    "adaptation_strategy": neural_cfg["adaptation"],
                    "optimizer": model.metadata["optimizer"],
                    "learning_rate": model.metadata["learning_rate"],
                    "update_steps_per_label": model.metadata["updates_per_label"],
                    "replay_buffer_size": neural_cfg.get("replay_buffer_size", 0),
                })
            for idx, a in curve:
                mlflow.log_metric("windowed_accuracy", a, step=idx)
            mlflow.log_metrics({
                "final_accuracy": results["final_accuracy"],
                "final_macro_f1": results["final_macro_f1"],
                "runtime_sec": results["runtime_sec"],
                "peak_memory_mb": results["peak_memory_mb"],
                "n_drift_detections": results["n_drift_detections"],
                "n_valid_predictions": results["n_valid_predictions"],
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
