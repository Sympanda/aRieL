"""
Load the Ariel MCS catalogue CSV and return a clean, renamed DataFrame.

The output contains only the columns defined in schemas.RAW_COL_MAP,
renamed to the canonical internal names.  No derived columns are added
here — that happens in preprocess_targets.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aRieL.data.schemas import RAW_COL_MAP, TARGET_DTYPES
from aRieL.utils.paths import data_search_dirs, find_data_csv

# Try the canonical name first, then fall back to any MCS CSV in data/raw/
# so the script works regardless of what the file was named locally.
_CANDIDATE_NAMES = [
    "Ariel_MCS_Known_2025-08-18.csv",
    "MCS.csv",
]


def mcs_not_found_message() -> str:
    """Explain how to supply the MCS. The package does not ship the file."""
    return (
        "Mission Candidate Sample CSV not found. Download "
        "Ariel_MCS_Known_2025-08-18.csv from "
        "https://github.com/arielmission-space/Mission_Candidate_Sample/blob/main/target_lists/Ariel_MCS_Known_2025-08-18.csv, "
        "then set ARIEL_DATA to that directory or place the file at "
        "./data/raw/MCS.csv. aRieL does not redistribute this catalogue. "
        "See docs/CATALOGUE.md."
    )


def find_mcs_csv() -> Path | None:
    """Return a user-supplied MCS CSV, or None when none is on the search path."""
    return find_data_csv(_CANDIDATE_NAMES, extra_globs=True)


def _find_default_csv() -> Path:
    found = find_mcs_csv()
    if found is not None:
        return found
    hint_dirs = data_search_dirs()
    hint = hint_dirs[0] if hint_dirs else Path("data/raw")
    return hint / _CANDIDATE_NAMES[0]


def load_mcs(path: str | Path | None = None) -> pd.DataFrame:
    """Load the MCS CSV and return a tidy DataFrame with canonical column names.

    Parameters
    ----------
    path:
        Path to the raw CSV.  Defaults to the first MCS file found via
        ``$ARIEL_DATA`` or ``./data/raw``.

    Returns
    -------
    pd.DataFrame
        One row per target, columns from ``schemas.RAW_COL_MAP`` values.
        Rows with missing ``period`` or ``epoch`` are dropped (shouldn't
        happen per the data, but guards against future catalogue updates).
    """
    csv_path = Path(path) if path is not None else _find_default_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MCS catalogue not found at {csv_path}. {mcs_not_found_message()}"
        )

    raw = pd.read_csv(csv_path, low_memory=False)

    # TPC exports use slightly different metallicity column names than MCS.
    if "Star Metallicity" not in raw.columns and "Star Metallicity [Fe/H]" in raw.columns:
        raw = raw.rename(columns={"Star Metallicity [Fe/H]": "Star Metallicity"})

    # Keep only mapped columns
    available = [c for c in RAW_COL_MAP if c in raw.columns]
    missing = [c for c in RAW_COL_MAP if c not in raw.columns]
    if missing:
        import warnings
        warnings.warn(f"Columns not found in CSV (will be NaN): {missing}", stacklevel=2)

    df = raw[available].rename(columns=RAW_COL_MAP).copy()

    # Add any columns that were in the map but missing from the CSV as NaN
    for new_name in RAW_COL_MAP.values():
        if new_name not in df.columns:
            df[new_name] = np.nan

    # ---------- type coercion ----------
    for col, dtype in TARGET_DTYPES.items():
        if col not in df.columns:
            continue
        try:
            if dtype in ("string",):
                df[col] = df[col].astype("string")
            elif dtype in ("Int64",):
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            elif dtype in ("float64",):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype in ("bool",):
                df[col] = df[col].astype("boolean")
        except Exception:
            pass  # leave as-is; caller will see NaN / object

    # ---------- essential rows ----------
    before = len(df)
    df = df.dropna(subset=["period", "epoch"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        import warnings
        warnings.warn(f"Dropped {dropped} rows with missing period/epoch.", stacklevel=2)

    # ---------- convenience: integer target index ----------
    df.insert(0, "target_idx", np.arange(len(df), dtype=np.int32))

    return df


def load_mcs_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Return the full raw CSV with no column selection or renaming."""
    csv_path = Path(path) if path is not None else _find_default_csv()
    return pd.read_csv(csv_path, low_memory=False)
