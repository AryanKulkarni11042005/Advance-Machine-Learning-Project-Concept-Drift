#!/usr/bin/env python
"""Fetch the real-world datasets. Synthetic streams need nothing - they are
generated on demand by src/streams.py.

    python scripts/download_data.py

Elec2 goes to River's own cache (~/river_data). Airlines is cached here as
data/airlines.parquet, which is gitignored - each machine fetches its own.
"""
import pathlib
import sys

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def download_elec2():
    from river import datasets
    d = datasets.Elec2()
    d.download()
    print(f"Elec2 ready: {d.n_samples:,} samples")


def download_airlines():
    from sklearn.datasets import fetch_openml
    out = DATA_DIR / "airlines.parquet"
    if out.exists():
        print(f"Airlines already cached at {out}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print("Fetching OpenML 1169 (airlines, ~540k rows) - this takes a minute...")
    df = fetch_openml(data_id=1169, as_frame=True, parser="auto").frame
    df.to_parquet(out)
    print(f"Airlines ready: {len(df):,} rows -> {out}")


if __name__ == "__main__":
    download_elec2()
    download_airlines()
    print("\nAll datasets ready.")
    sys.exit(0)
