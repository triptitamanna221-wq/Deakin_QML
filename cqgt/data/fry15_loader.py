"""Loader for real FFIEC FR Y-15 'Snapshot Indicators' exports.

Each yearly export reports every one of ~13 G-SIB indicator items three
times, under prefixes RISK*, RISI*, RISO* -- one block per reporting
population (top-tier US BHC, US intermediate holding company of a foreign
banking organization, or the foreign parent itself). For any given
institution-year, exactly one of the three blocks is populated and the
other two are entirely NaN; `_coalesce_block` picks whichever is non-null.

Column names, column order, and even which indicator items are present
drift year to year (see BRIEF_MNEMONIC_MAP below and NOTES.md's mnemonic
table). Selection here is always by column name, never position, per
explicit project constraint.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fry15"

# Explicit per-year mapping of canonical field -> source column base name
# (before the RISK/RISI/RISO prefix for the three indicator blocks).
# `m362` (Intra-Financial System Assets) is None for 2024: FFIEC's 2024
# export structurally omits that field (confirmed: 0 of 45 columns match
# "M362" in that file, vs 3 of N in every other year). See NOTES.md.
# Extra G-SIB indicator items pulled alongside m362/m370/y832, used only to
# build honestly-labeled feature PROXIES (cqgt/data/features.py) for the
# quantities BRIEF.md's feature list wants (leverage, Tier-1 ratio, CDS-
# proxy, liquidity ratio, macro sensitivity) that FR Y-15 does not itself
# report -- it has no capital/equity or market-price data. Present under
# the same base name in all 5 years (verified directly against each header),
# so no per-year override is needed for these.
_EXTRA_FIELDS = ("M376", "M390", "M405", "M408", "M411", "M422", "M426")

YEAR_COLUMN_MAP = {
    2020: {
        "file": "20201231_20210801_FRY15 Snapshot Indicators.csv",
        "id": "ID_RSSD", "name": "NAME", "date": "DT",
        "m362": "M362", "m370": "M370", "y832": "Y832",
    },
    2021: {
        "file": "20211231_20220722_FRY15 Snapshot Indicators.csv",
        "id": "ID_RSSD", "name": "Name (Legal)", "date": "DT",
        "m362": "M362", "m370": "M370", "y832": "Y832",
    },
    2022: {
        "file": "20221231_20230804_FRY15 Snapshot Indicators.csv",
        "id": "ID_RSSD", "name": "Name (Legal)", "date": "DT",
        "m362": "M362", "m370": "M370", "y832": "Y832",
    },
    2023: {
        "file": "20231231_20240724_FRY15 Snapshot Indicators.csv",
        "id": "ID_RSSD", "name": "NAME", "date": "AsOfDate",
        "m362": "M362", "m370": "M370", "y832": "Y832",
    },
    2024: {
        "file": "20241231_20250722_FRY15 Snapshot Indicators.csv",
        "id": "ID_RSSD", "name": "Name", "date": "As of Date",
        "m362": None, "m370": "M370", "y832": "Y832",
    },
}
for _spec in YEAR_COLUMN_MAP.values():
    for _f in _EXTRA_FIELDS:
        _spec[_f.lower()] = _f

BLOCK_PREFIXES = ("RISK", "RISI", "RISO")

# Years used as real network-reconstruction anchors. 2024 is loaded for its
# available fields (m370, y832) but is NOT a network anchor -- see
# NOTES.md / BRIEF.md Sec 6 Gate 1 log for why (M362 does not exist in 2024's
# export; the user chose to drop 2024 as an anchor rather than carry forward
# or relax the completeness rule).
NETWORK_ANCHOR_YEARS = (2020, 2021, 2022, 2023)
ALL_LOADED_YEARS = (2020, 2021, 2022, 2023, 2024)


def _strip_bom_prefix(df):
    """Two of the five source files carry a leading BOM baked into the first
    header cell as literal mojibake characters ('﻿ï»¿'-style, from a
    UTF-8-as-Latin-1 round trip during the original export/copy) rather than
    a true UTF-8 BOM byte, so encoding="utf-8-sig" does not strip it. Strip
    any such prefix from column names so the rest of the loader can use
    clean, consistent names regardless of which file it's reading."""
    rename = {c: c.lstrip("﻿ï»¿") for c in df.columns}
    return df.rename(columns=rename)


def _coalesce_block(df, base_col):
    """Pick whichever of RISK<base>/RISI<base>/RISO<base> is the populated
    block per row, in that priority order. Exactly one population reports
    each institution, but at least one filer (Discover Financial Services,
    2023) explicitly entered 0 rather than leaving the non-applicable blocks
    blank -- so "populated" means non-null AND non-zero, not just non-null,
    when deciding whether more than one block is genuinely in conflict.
    Raises only on a true conflict: two blocks both non-null and non-zero."""
    cols = [f"{p}{base_col}" for p in BLOCK_PREFIXES if f"{p}{base_col}" in df.columns]
    block = df[cols]
    n_real = ((block.notna()) & (block != 0)).sum(axis=1)
    if (n_real > 1).any():
        bad = df.loc[n_real > 1, "ID_RSSD_canonical"].tolist()
        raise ValueError(f"More than one RISK/RISI/RISO block genuinely populated for "
                          f"{base_col} on RSSD IDs {bad}; coalescing assumption violated.")
    return block.bfill(axis=1).iloc[:, 0]


def load_year(year):
    """Return a DataFrame with columns [id_rssd, name, date, m362, m370, y832]
    for one FR Y-15 snapshot year. m362 is all-NaN for years where FFIEC's
    export doesn't carry that field (2024)."""
    spec = YEAR_COLUMN_MAP[year]
    path = RAW_DIR / spec["file"]
    df = pd.read_csv(path, na_values=["NULL", ""], encoding="utf-8-sig")
    df = _strip_bom_prefix(df)

    out = pd.DataFrame({
        "id_rssd": df[spec["id"]].astype("Int64"),
        "name": df[spec["name"]],
        "date": df[spec["date"]],
    })
    out["ID_RSSD_canonical"] = out["id_rssd"]  # for error messages in _coalesce_block
    df = pd.concat([df, out[["ID_RSSD_canonical"]]], axis=1)

    fields = [("m362", spec["m362"]), ("m370", spec["m370"]), ("y832", spec["y832"])]
    fields += [(f.lower(), spec[f.lower()]) for f in _EXTRA_FIELDS]
    for canon, base in fields:
        out[canon] = _coalesce_block(df, base) if base is not None else np.nan

    out["year"] = year
    out = out.dropna(subset=["id_rssd"]).drop_duplicates(subset=["id_rssd"])
    return out.drop(columns=[]).reset_index(drop=True)


def load_all_years(years=ALL_LOADED_YEARS):
    return {y: load_year(y) for y in years}
