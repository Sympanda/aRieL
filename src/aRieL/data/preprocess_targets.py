"""
Full preprocessing pipeline: CSV → clean target DataFrame ready for the env.

Usage
-----
    from aRieL.data.preprocess_targets import build_target_table

    targets = build_target_table()          # uses default CSV path
    targets.to_parquet("data/processed/targets.parquet")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aRieL.data.load_catalogue import load_mcs
from aRieL.data.observation_requirements import add_observation_costs
from aRieL.data.population_bins import assign_population_bins


def build_target_table(
    csv_path: str | Path | None = None,
    min_available_obs: int = 1,
    science_weight_floor: float = 0.3,
) -> pd.DataFrame:
    """Load, clean, and enrich the MCS catalogue.

    Steps
    -----
    1. Load raw CSV → canonical column names.
    2. Drop targets where the preferred method is impossible
       (e.g. eclipse-only target with no available eclipses).
    3. Assign population bins + science weights.
    4. Add per-observation time costs.

    Parameters
    ----------
    csv_path:
        Path to the raw MCS CSV.  Defaults to ``data/raw/...``.
    min_available_obs:
        Drop targets with fewer than this many available observations
        (transits or eclipses depending on preferred_method).

    Returns
    -------
    pd.DataFrame
        Fully processed target table, indexed 0..N-1.
    """
    df = load_mcs(csv_path)

    # ------------------------------------------------------------------
    # Filter: must have at least one available observation of the right type
    # ------------------------------------------------------------------
    def _has_obs(row: pd.Series) -> bool:
        method = str(row.get("preferred_method") or "Transit")
        if method == "Eclipse":
            n = row.get("available_eclipses", 0) or 0
        else:
            n = row.get("available_transits", 0) or 0
        return int(n) >= min_available_obs

    mask = df.apply(_has_obs, axis=1)
    dropped = (~mask).sum()
    if dropped:
        import warnings
        warnings.warn(
            f"Dropped {dropped} targets with < {min_available_obs} available observations.",
            stacklevel=2,
        )
    df = df[mask].reset_index(drop=True)
    # Keep target_idx consistent after filter
    df["target_idx"] = range(len(df))

    # ------------------------------------------------------------------
    # Population bins + science weights
    # ------------------------------------------------------------------
    df = assign_population_bins(df, science_weight_floor=science_weight_floor)

    # ------------------------------------------------------------------
    # Observation costs
    # ------------------------------------------------------------------
    df = add_observation_costs(df)

    return df


def load_or_build(
    parquet_path: str | Path = "data/processed/targets.parquet",
    csv_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load processed targets from Parquet if available, else build and save.

    Parameters
    ----------
    parquet_path:
        Where to cache the processed table.
    csv_path:
        Source CSV (only used if rebuilding).
    force_rebuild:
        If True, always rebuild even if Parquet exists.
    """
    p = Path(parquet_path)
    if p.exists() and not force_rebuild:
        return pd.read_parquet(p)

    df = build_target_table(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
