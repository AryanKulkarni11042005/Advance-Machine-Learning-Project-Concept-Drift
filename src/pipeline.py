# src/pipeline.py
"""Prequential (test-then-train) evaluation of drift adaptation strategies."""
import pathlib
import hashlib
import math
import tracemalloc
from collections import Counter, deque
from numbers import Real
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

# Kept separate so the existing default benchmark grid remains unchanged.
# The dedicated Phase 3 runner opts into these strategies explicitly.
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


class AirlinesNeuralEncoder:
    """Fixed-width, causal representation for Airlines neural models only.

    Categories are hashed independently within feature-specific bucket blocks;
    numeric features use a fixed signed-log transform. No vocabulary or scale
    is fitted from the stream, so later observations cannot affect earlier rows.
    """

    def __init__(self, feature_names, categorical_features, buckets_per_feature=32):
        self.feature_names = tuple(feature_names)
        self.categorical_features = tuple(
            name for name in self.feature_names if name in categorical_features
        )
        self.numeric_features = tuple(
            name for name in self.feature_names if name not in categorical_features
        )
        self.buckets_per_feature = int(buckets_per_feature)
        if self.buckets_per_feature < 1:
            raise ValueError("buckets_per_feature must be positive")
        self.input_size = len(self.numeric_features) + (
            len(self.categorical_features) * self.buckets_per_feature
        )
        self.metadata = {
            "type": "causal_feature_hash",
            "hash": "blake2b-64",
            "buckets_per_categorical_feature": self.buckets_per_feature,
            "categorical_features": list(self.categorical_features),
            "numeric_features": list(self.numeric_features),
            "numeric_transform": "signed_log1p",
            "vocabulary_fitted": False,
            "label_dependent": False,
        }

    def transform(self, features):
        out = []
        for name in self.numeric_features:
            value = features[name]
            if not isinstance(value, Real):
                raise TypeError(f"numeric feature {name!r} is not numeric")
            value = float(value)
            out.append(math.copysign(math.log1p(abs(value)), value))
        for name in self.categorical_features:
            value = str(features[name])
            token = f"{name}\0{value}".encode("utf-8")
            bucket = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(),
                                    "big") % self.buckets_per_feature
            block = [0.0] * self.buckets_per_feature
            block[bucket] = 1.0
            out.extend(block)
        return {f"encoded_{i}": value for i, value in enumerate(out)}


def run_strategy(samples, strategy_name, dataset_name, drift_points=(),
                 seed=42, window=500, eval_every=200, log_to_mlflow=True,
                 delay=0, adwin_config=None, model_config=None,
                 dataset_config=None, run_id=None, experiment_version=None,
                 mlflow_experiment=None, airline_hash_config=None,
                 recovery_tolerance=0.02):
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
    model_config = dict(model_config or {})
    adwin_config = dict(adwin_config or {})
    airline_hash_config = dict(airline_hash_config or {})
    encoder = None
    if neural_cfg:
        first_features = samples[0][0]
        if dataset_name == "airlines":
            categorical = airline_hash_config.get(
                "categorical_features",
                ("Airline", "AirportFrom", "AirportTo", "DayOfWeek"),
            )
            encoder = AirlinesNeuralEncoder(
                first_features.keys(), categorical,
                airline_hash_config.get("buckets_per_feature", 32),
            )
            first_features = encoder.transform(first_features)
        feature_names = tuple(first_features.keys())
        sequence_length = int(model_config.get("sequence_length", 10))
        sequence_builder = RollingSequence(sequence_length=sequence_length,
                                           feature_names=feature_names)
        model = NeuralAdapter(
            architecture=neural_cfg["architecture"],
            input_size=len(feature_names), seed=seed,
            hidden_size=int(model_config.get("hidden_size", 16)),
            learning_rate=float(model_config.get("learning_rate", 0.01)),
            sequence_length=sequence_length,
        )
        if int(model_config.get("update_steps", 1)) != 1:
            raise ValueError("Phase 3 currently supports exactly one update step per label")
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
    detector = (ADWIN(delta=float(adwin_config.get("delta", 0.002)),
                      clock=int(adwin_config.get("clock", 32)))
                if detector_enabled else None)
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
    drift_events = []
    detector_feedback = []
    valid_predictions = 0
    curve = []          # (index, windowed accuracy) - windowed, so a local
    win_correct = win_n = 0   # dip at the drift is visible instead of averaged away
    pending = deque()   # samples awaiting their label under `delay`

    def adapt_from_error(error, feedback_index, prediction_index):
        nonlocal model
        if detector is None or error is None:
            return
        detector_feedback.append({"feedback_index": int(feedback_index),
                                  "prediction_index": int(prediction_index),
                                  "error": int(error)})
        detector.update(error)
        if detector.drift_detected:
            detected_drifts.append(int(feedback_index))
            drift_events.append({
                "feedback_index": int(feedback_index),
                "prediction_index": int(prediction_index),
                "detected_at": int(feedback_index),
            })
            if neural_cfg:
                if neural_cfg["adaptation"] == "adwin_reset":
                    model.reset()
                elif neural_cfg["adaptation"] == "adwin_finetune":
                    # Only examples whose labels were revealed earlier.
                    for buffered_sequence, buffered_label in neural_buffer:
                        model.update(buffered_sequence, buffered_label)
            elif reset_on_drift:
                model = model_fn()
                if refit_on_reset:
                    for bx, by in buffer:
                        model.learn_one(bx, by)

    tracemalloc.start()
    start = perf_counter()

    for i, (x, y) in enumerate(samples):
        if sequence_builder:
            neural_x = encoder.transform(x) if encoder else x
            sequence = sequence_builder.append(neural_x)
            y_pred = model.predict(sequence) if sequence is not None else None
        else:
            sequence = None
            y_pred = model.predict_one(x)

        if retrain_every and i > 0 and i % retrain_every == 0:
            model = model_fn()
            if refit_on_reset:
                for bx, by in buffer:
                    model.learn_one(bx, by)

        # Queue every learnable sample, even when River cannot yet predict a
        # class. Its label must still be revealed and learned; only scoring
        # and ADWIN feedback require a valid prediction.
        if not sequence_builder or sequence is not None:
            pending.append({"x": x, "sequence": sequence, "label": y,
                            "prediction": y_pred, "prediction_index": i})

        # delay=0 reveals the current item after its prediction. For delayed
        # streams, predict the current item first, then resolve prior labels.
        while len(pending) > delay:
            item = pending.popleft()
            label = item["label"]
            prediction = item["prediction"]
            if prediction is not None:
                valid_predictions += 1
                acc.update(label, prediction)
                f1.update(label, prediction)
                win_correct += int(prediction == label)
                win_n += 1
                adapt_from_error(int(prediction != label), i,
                                 item["prediction_index"])
            if sequence_builder:
                model.update(item["sequence"], label)
                neural_buffer.append((item["sequence"], label))
            else:
                model.learn_one(item["x"], label)
                buffer.append((item["x"], label))

        if win_n and (i + 1) % eval_every == 0:
            curve.append((i + 1, win_correct / win_n))
            win_correct = win_n = 0

    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recoveries = []
    for drift_idx in drift_points:
        pre = [a for idx, a in curve if idx < drift_idx]
        if not pre:
            recoveries.append({"drift_index": int(drift_idx),
                               "baseline_accuracy": None, "threshold": None,
                               "recovery_index": None, "recovery_time": None,
                               "status": "no_pre_drift_baseline"})
            continue
        baseline = pre[-1]
        threshold = baseline - recovery_tolerance
        recovered_idx = next((idx for idx, value in curve
                              if idx > drift_idx and value >= threshold), None)
        recoveries.append({
            "drift_index": int(drift_idx),
            "baseline_accuracy": float(baseline),
            "threshold": float(threshold),
            "recovery_index": (int(recovered_idx) if recovered_idx is not None else None),
            "recovery_time": (int(recovered_idx - drift_idx)
                              if recovered_idx is not None else None),
            "status": "recovered" if recovered_idx is not None else "no_recovery",
        })
    if not drift_points:
        recoveries.append({"drift_index": None, "baseline_accuracy": None,
                           "threshold": None, "recovery_index": None,
                           "recovery_time": None, "status": "not_applicable"})
    measured = [r["recovery_time"] for r in recoveries
                if r["status"] == "recovered"]

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
        "drift_events": drift_events,
        "detector_feedback": detector_feedback,
        "majority_baseline": majority_baseline,
        "lift_over_majority": acc.get() - majority_baseline,
        "recoveries": [r["recovery_time"] for r in recoveries
                       if r["status"] != "not_applicable"],
        "recovery_records": recoveries,
        "curve": curve,
        "mean_recovery": (sum(measured) / len(measured) if measured else None),
        "n_recovered": len(measured),
        "dataset_config": dataset_config or {},
        "airlines_encoder": encoder.metadata if encoder else None,
        "adwin_config": (dict(adwin_config) if detector is not None else None),
        "model_config": model_config if sequence_builder else {},
    }

    if log_to_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(mlflow_experiment or EXPERIMENT)
        with mlflow.start_run(run_name=f"{dataset_name}_{strategy_name}_s{seed}"):
            mlflow.log_params({
                "run_id": run_id or "legacy",
                "experiment_version": experiment_version or "legacy",
                "dataset": dataset_name,
                "strategy": strategy_name,
                "base_learner": (model.metadata["model_type"] if sequence_builder
                                 else model_fn().__class__.__name__),
                "seed": seed,
                "window": window,
                "retrain_every": retrain_every,
                "detector": "ADWIN" if detector is not None else None,
                "adwin_delta": adwin_config.get("delta", 0.002) if detector is not None else None,
                "adwin_clock": adwin_config.get("clock", 32) if detector is not None else None,
                "n_samples": len(samples),
                "delay": delay,
                "true_drift_points": list(drift_points),
                "dataset_configuration": str(dataset_config or {})[:450],
                "recovery_tolerance": recovery_tolerance,
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
                if encoder:
                    mlflow.log_params({"airlines_encoder": str(encoder.metadata)[:450]})
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
            if run_id:
                mlflow.set_tags({"phase3_run_id": run_id,
                                 "experiment_version": experiment_version or "phase3"})

    return results
