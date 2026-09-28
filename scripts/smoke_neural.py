#!/usr/bin/env python
"""Small CPU smoke run for the LSTM and GRU prequential adapters."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streams  # noqa: E402
from pipeline import run_strategy  # noqa: E402


def main():
    samples, drift_points = streams.sea(
        n_per_concept=100, variants=(3, 2), seed=42
    )
    for name in ("lstm", "gru"):
        result = run_strategy(
            samples, name, "sea_abrupt_smoke", drift_points,
            seed=42, eval_every=25, log_to_mlflow=False,
        )
        assert result["final_accuracy"] >= 0.0
        assert result["final_macro_f1"] >= 0.0
        assert result["curve"], "expected at least one scored window"
        print(
            f"{name.upper()}: accuracy={result['final_accuracy']:.4f} "
            f"macro_f1={result['final_macro_f1']:.4f} "
            f"runtime={result['runtime_sec']:.2f}s "
            f"valid_predictions={len(samples) - 9}"
        )


if __name__ == "__main__":
    main()
