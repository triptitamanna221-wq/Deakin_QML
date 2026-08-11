"""Macro stress factors C_t. Real FRED series exist in data/raw/fred/
(BRIEF.md Sec 2.1's real-data path). BAMLH0A0HYM2 (ICE BofA High Yield
spread) only starts 2023-08-11 and does not cover the 2020-2023 network
anchor window at all, so it is excluded from the PCA -- using it would mean
either fabricating pre-2023 values or silently degrading to near-constant
imputation, both of which this project rules out. The other 5 series (DFF,
NFCI, STLFSI4, T10Y2Y, VIXCLS) all predate 2020 and are used as-is.

The T panel weeks are a SYNTHETIC calendar (see cqgt/data/temporal.py):
week t is linearly mapped onto a real date between the first and last
network anchor, and the FRED value looked up for that date is real,
as-of/forward-filled from the most recent actual observation. So the macro
values fed into the model at any given t are genuine FRED data, but the
date they're attached to is a synthetic interpolation point, not a real
weekly observation date.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fred"

USABLE_SERIES = ["DFF", "NFCI", "STLFSI4", "T10Y2Y", "VIXCLS"]
EXCLUDED_SERIES = {"BAMLH0A0HYM2": "starts 2023-08-11, does not cover the 2020-2023 anchor window"}

K_FACTORS = 3


def load_fred_series(series=USABLE_SERIES):
    frames = {}
    for name in series:
        df = pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=["observation_date"])
        frames[name] = df.set_index("observation_date")[name].sort_index()
    return frames


def dates_for_panel(T, anchor_start, anchor_end):
    """Synthetic calendar: linearly space T dates between the first and last
    real network anchor date."""
    start, end = pd.Timestamp(anchor_start), pd.Timestamp(anchor_end)
    frac = np.linspace(0, 1, T)
    return pd.DatetimeIndex([start + f * (end - start) for f in frac])


def asof_lookup(series, dates):
    """Most recent real observation on/before each date (forward fill)."""
    return series.reindex(series.index.union(dates)).ffill().reindex(dates)


def build_macro_factors(T, anchor_start, anchor_end, k=K_FACTORS, train_frac=0.6):
    """Returns (factors (T,k), loadings (n_series,k), dates, report). PCA is
    fit on the train-split portion only (first `train_frac` of T) to avoid
    leaking future macro information into factor directions, consistent
    with the no-future-information rule for the rest of the pipeline."""
    dates = dates_for_panel(T, anchor_start, anchor_end)
    raw = load_fred_series()
    X = np.stack([asof_lookup(s, dates).values for s in raw.values()], axis=1).astype(float)

    n_train = int(round(T * train_frac))
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    U, S, Vt = np.linalg.svd(Xz[:n_train], full_matrices=False)
    loadings = Vt[:k].T  # (n_series, k)
    factors = Xz @ loadings  # (T, k), full panel projected onto train-fit loadings

    explained_var_ratio = (S[:k] ** 2) / (S ** 2).sum()
    report = {
        "series_used": list(raw.keys()),
        "series_excluded": EXCLUDED_SERIES,
        "explained_variance_ratio": explained_var_ratio,
        "date_range": (dates[0], dates[-1]),
    }
    return factors, loadings, dates, report
