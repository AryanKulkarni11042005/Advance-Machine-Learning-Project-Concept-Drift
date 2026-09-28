#!/usr/bin/env python
"""Small SEA abrupt integration experiment for neural adaptation strategies."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streams  # noqa: E402
from pipeline import run_strategy  # noqa: E402


def main():
    samples, drift_points = streams.sea(
        n_per_concept=1000, variants=(3, 2), seed=42
    )
    strategies = (
        "lstm", "lstm_adwin_reset", "lstm_adwin_finetune",
        "gru", "gru_adwin_reset", "gru_adwin_finetune",
    )
    for strategy in strategies:
        result = run_strategy(
            samples, strategy, "sea_abrupt_neural_smoke", drift_points,
            seed=42, eval_every=50, log_to_mlflow=False,
        )
        print(
            f"{strategy}: accuracy={result['final_accuracy']:.4f} "
            f"macro_f1={result['final_macro_f1']:.4f} "
            f"valid_predictions={result['n_valid_predictions']} "
            f"drift_detections={result['n_drift_detections']} "
            f"runtime={result['runtime_sec']:.2f}s"
        )


if __name__ == "__main__":
    main()
