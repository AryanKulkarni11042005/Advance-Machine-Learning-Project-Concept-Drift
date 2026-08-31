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


def elec2(seed=None):
    """Electricity market (NSW), 45,312 samples, 8 numeric features.

    Real-world stream with gradual and recurring drift from demand cycles.
    True drift points are unknown, so drift_points is empty and recovery time
    is not reported for this stream. `seed` is accepted and ignored - the rows
    are fixed and always in the same temporal order.
    """
    return list(datasets.Elec2()), []


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
