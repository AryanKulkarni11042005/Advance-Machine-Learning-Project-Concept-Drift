# src/pipeline.py
import mlflow
from river import metrics, forest
from river.drift import ADWIN
from time import perf_counter

def run_strategy(stream, model_fn, strategy_name, dataset_name, detector_fn=None, retrain_every=None):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("concept-drift-adaptation")

    model = model_fn()
    detector = detector_fn() if detector_fn else None
    acc = metrics.Accuracy()
    f1 = metrics.F1()
    drift_points = []
    accuracy_over_time = []

    with mlflow.start_run(run_name=f"{dataset_name}_{strategy_name}"):
        mlflow.log_params({
            "dataset": dataset_name,
            "strategy": strategy_name,
            "retrain_every": retrain_every,
        })
        start = perf_counter()
        for i, (x, y) in enumerate(stream):
            y_pred = model.predict_one(x)
            if y_pred is not None:
                acc.update(y, y_pred)
                f1.update(y, y_pred)

            if detector is not None:
                detector.update(int(y_pred == y) if y_pred is not None else 0)
                if detector.drift_detected:
                    drift_points.append(i)
                    if strategy_name == "detector_retrain":
                        model = model_fn()  # reset

            if retrain_every and i % retrain_every == 0:
                model = model_fn()  # blind periodic retrain

            model.learn_one(x, y)

            if i % 1000 == 0:
                accuracy_over_time.append((i, acc.get()))
                mlflow.log_metric("accuracy", acc.get(), step=i)

        elapsed = perf_counter() - start
        mlflow.log_metrics({
            "final_accuracy": acc.get(),
            "final_f1": f1.get(),
            "runtime_sec": elapsed,
            "n_drift_points": len(drift_points),
        })

    return {"accuracy_over_time": accuracy_over_time, "drift_points": drift_points}