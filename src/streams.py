# src/streams.py
"""Stream loaders.

Every loader returns (samples, drift_points):
  samples     - list of (x, y), x a dict of features, y the label
  drift_points - list of sample indices where the concept changes.
                 Empty for real-world streams, where true drift is unknown.

Synthetic streams are built by concatenating stationary generators, so the
drift indices are ground truth and recovery time is measurable against them.
"""
import pathlib

from river import datasets
from river.datasets import synth

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# Real-world streams: fixed rows in a fixed order, so a seed changes nothing
# about them. The grid runner uses this to avoid reporting repeated identical
# runs as replicated evidence.
DETERMINISTIC = {"elec2", "airlines"}


def is_deterministic(name):
    return name in DETERMINISTIC


def sea(n_per_concept=5000, variants=(3, 2), seed=42):
    """SEA concepts - abrupt drift at each variant boundary.

    Each SEA variant uses a different threshold on the sum of the first two
    features, so switching variants relabels the space without changing its
    distribution: a clean abrupt concept drift.

    Variant thresholds are {0: 8, 1: 9, 2: 7, 3: 9.5}. The default (3, 2) is
    the widest available gap (9.5 -> 7). Narrower gaps such as (0, 2) shift the
    boundary by only 1.0 on a sum spanning 0-20, relabelling ~7% of the space -
    a Hoeffding Tree absorbs that without a measurable accuracy drop, which
    makes recovery time unmeasurable. Verified empirically: (0, 2) produces no
    drop at all, (3, 2) produces ~13 points.
    """
    samples, drift_points = [], []
    for k, variant in enumerate(variants):
        if k > 0:
            drift_points.append(len(samples))
        gen = synth.SEA(variant=variant, seed=seed + k)
        samples.extend(gen.take(n_per_concept))
    return samples, drift_points


def hyperplane(n_samples=10000, n_features=10, n_drift_features=5,
               mag_change=0.001, seed=42):
    """Rotating hyperplane - continuous gradual drift.

    The decision boundary rotates every sample, so drift is incremental and
    has no single boundary index. drift_points is empty by design; recovery
    time is not defined for this stream.
    """
    gen = synth.Hyperplane(
        seed=seed,
        n_features=n_features,
        n_drift_features=n_drift_features,
        mag_change=mag_change,
    )
    return list(gen.take(n_samples)), []


def sea_gradual(n_per_concept=5000, transition=1000, variants=(3, 2), seed=42):
    """SEA with a gradual transition: over `transition` samples the source
    mixes probabilistically from one variant to the next.

    The recorded drift point is the midpoint of the transition window, which
    is the convention used when reporting recovery time for gradual drift.
    """
    import random
    rng = random.Random(seed)
    gens = [synth.SEA(variant=v, seed=seed + k).take(10**9)
            for k, v in enumerate(variants)]
    gens = [iter(g) for g in gens]

    samples, drift_points = [], []
    for k in range(len(variants)):
        if k > 0:
            # transition window: probability of drawing from the new concept
            # ramps 0 -> 1 across the window.
            start = len(samples)
            for t in range(transition):
                p_new = (t + 1) / transition
                src = gens[k] if rng.random() < p_new else gens[k - 1]
                samples.append(next(src))
            drift_points.append(start + transition // 2)
        # stationary stretch of the current concept
        for _ in range(n_per_concept):
            samples.append(next(gens[k]))
    return samples, drift_points


def elec2(n_samples=45_312, seed=None):
    """Electricity market (NSW), 45,312 samples, 8 numeric features.

    Real-world stream with gradual and recurring drift from demand cycles.
    True drift points are unknown, so drift_points is empty and recovery time
    is not reported for this stream. `seed` is accepted and ignored - the rows
    are fixed and always in the same temporal order.
    """
    samples = list(datasets.Elec2())
    return samples[:n_samples], []


def airlines(n_samples=100_000, seed=None):
    """Airlines delay prediction (OpenML id 1169), capped at n_samples.

    539,383 rows total; the default takes the first 100,000 **in order**.
    Order matters: the drift is temporal, so shuffling or random-sampling
    would destroy the very thing under study. Report the cap in the paper.

    Mixed feature types - Airline / AirportFrom / AirportTo / DayOfWeek are
    categorical, Flight / Time / Length numeric. River trees split on both
    natively, so no encoding is applied; encoding categoricals into numbers
    would impose a false ordering on airport codes.

    Requires data/airlines.parquet (see scripts/download_data.py).
    """
    import pandas as pd

    path = DATA_DIR / "airlines.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/download_data.py` first."
        )
    df = pd.read_parquet(path).head(n_samples)
    target = "Delay"
    feature_cols = [c for c in df.columns if c != target]

    samples = []
    for row in df.itertuples(index=False):
        d = dict(zip(df.columns, row))
        y = int(d.pop(target))
        x = {k: (v if isinstance(v, (int, float)) else str(v))
             for k, v in d.items() if k in feature_cols}
        samples.append((x, y))
    return samples, []


# Registry consumed by the experiment grid.
STREAMS = {
    "sea_abrupt": sea,
    "sea_gradual": sea_gradual,
    "hyperplane": hyperplane,
    "elec2": elec2,
    "airlines": airlines,
}


def load_stream(name, seed=42, **stream_config):
    """Build a registered stream with explicit, recorded generator arguments."""
    try:
        generator = STREAMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {name}") from exc
    return generator(seed=seed, **stream_config)


def stock_sp500(csv="stock_market_cleaned.csv", seed=None, scale_free=True,
                n_samples=None):
    """S&P 500 daily direction, 2005-2024. Fixed rows in temporal order.

    Real-world stream with no ground-truth drift index, so drift_points is
    empty and recovery time is not reported. `seed` is accepted and ignored.

    scale_free=True replaces raw price levels (ma_5/20/50, macd) with ratios
    to the current close. Raw levels drift ~4.8x across this period, so a tree
    splitting on `ma_50 < 1200` learns a rule that stops firing after ~2013.
    That is price inflation masquerading as concept drift, and it contaminates
    the very thing this benchmark measures.

    NOTE ON LABELS: the target is next-day direction, so the label for row t is
    only known after t+1 closes. Feeding it to learn_one at time t trains the
    model on the future. Use `delay=1` in run_strategy to hold each sample back
    one step - see the delayed-label handling there.
    """
    import numpy as np
    import pandas as pd

    path = pathlib.Path(csv)
    if not path.is_absolute():
        for cand in (DATA_DIR / csv, DATA_DIR.parent / "notebooks" / csv, path):
            if cand.exists():
                path = cand
                break
    if not path.exists():
        raise FileNotFoundError(f"{csv} not found - build it in the stock notebook first.")

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)

    if scale_free and "ma_50" in df.columns:
        close_proxy = df["ma_5"] * (1 + df["return_1d"])  # close is not stored
        df["ma_5_ratio"] = close_proxy / df["ma_5"] - 1
        df["ma_20_ratio"] = df["ma_5"] / df["ma_20"] - 1
        df["ma_50_ratio"] = df["ma_20"] / df["ma_50"] - 1
        df["macd_norm"] = df["macd"] / df["ma_20"]
        df["macd_sig_norm"] = df["macd_signal"] / df["ma_20"]
        feats = ["return_1d", "volatility_10d", "volume_change", "rsi_14",
                 "ma_5_ratio", "ma_20_ratio", "ma_50_ratio",
                 "macd_norm", "macd_sig_norm"]
    else:
        feats = [c for c in df.columns if c not in ("target", "Date")]

    df = df.dropna(subset=feats + ["target"]).reset_index(drop=True)
    samples = [({f: float(r[f]) for f in feats}, int(r["target"]))
               for _, r in df.iterrows()]
    if n_samples is not None:
        samples = samples[:n_samples]
    return samples, []


DETERMINISTIC.add("stock_sp500")
STREAMS["stock_sp500"] = stock_sp500
