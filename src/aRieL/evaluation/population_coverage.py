"""
Population coverage analysis: how well did a schedule cover the
astrophysical diversity of the Ariel target catalogue?

The key questions are:
  - Which population bins were reached at Tier 1 / Tier 2 / Tier 3?
  - Which bins were completely ignored?
  - Is coverage uniform across radius and temperature classes?
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState


def coverage_table(state: "MissionState") -> pd.DataFrame:
    """Per-bin coverage summary.

    Returns
    -------
    pd.DataFrame with columns:
        population_bin, n_targets, n_tier1, n_tier2, n_tier3,
        tier1_rate, tier2_rate, tier3_rate, n_eligible_t2, n_eligible_t3
    Sorted by tier1_rate descending.
    """
    targets  = state.targets
    progress = state.progress

    # Join progress back to targets
    merged = targets.set_index("target_id").join(
        progress[["tier1_done", "tier2_done", "tier3_done"]]
    ).reset_index()

    rows = []
    for bin_label, group in merged.groupby("population_bin"):
        n   = len(group)
        t1  = int(group["tier1_done"].sum())
        t2  = int(group["tier2_done"].sum())
        t3  = int(group["tier3_done"].sum())
        t2_elig = int((group["max_tier"] >= 2).sum())
        t3_elig = int((group["max_tier"] >= 3).sum())

        rows.append({
            "population_bin": bin_label,
            "n_targets":      n,
            "n_tier1":        t1,
            "n_tier2":        t2,
            "n_tier3":        t3,
            "n_eligible_t2":  t2_elig,
            "n_eligible_t3":  t3_elig,
            "tier1_rate":     t1 / n if n > 0 else 0.0,
            "tier2_rate":     t2 / t2_elig if t2_elig > 0 else 0.0,
            "tier3_rate":     t3 / t3_elig if t3_elig > 0 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("tier1_rate", ascending=False).reset_index(drop=True)
    return df


def coverage_matrix(state: "MissionState", tier: int = 1) -> pd.DataFrame:
    """Radius × temperature completion matrix for a given tier.

    Returns a DataFrame where rows = radius class, columns = temperature
    class, values = fraction of targets in that cell at the given tier.
    Useful for spotting systematic gaps in population coverage.
    """
    targets  = state.targets
    progress = state.progress
    done_col = f"tier{tier}_done"

    merged = targets.set_index("target_id").join(progress[[done_col]]).reset_index()

    # Parse population_bin into (radius_class, temp_class).
    # Bin format: {radius}_{temp}_{stellar}  where radius and temp may themselves
    # contain underscores (e.g. "sub_earth", "very_hot", "ultra_hot").
    # We match greedily against the canonical ordered lists (longest first for temp).
    def _parse_bin(b: str) -> tuple[str, str]:
        b = str(b)
        radius = "unknown"
        for r in _RADIUS_ORDER:
            if b.startswith(r + "_") or b == r:
                radius = r
                break
        remaining = b[len(radius) + 1:] if radius != "unknown" else b
        temp = "unknown"
        for t in _TEMP_ORDER:   # ultra_hot / very_hot checked before hot/warm/cold
            if remaining.startswith(t):
                temp = t
                break
        return radius, temp

    merged[["radius_cls", "temp_cls"]] = pd.DataFrame(
        merged["population_bin"].map(_parse_bin).tolist(),
        index=merged.index,
    )

    temp_display = list(reversed(_TEMP_ORDER))

    pivot = merged.pivot_table(
        values=done_col,
        index="radius_cls",
        columns="temp_cls",
        aggfunc="mean",
        fill_value=0.0,
    )
    pivot = pivot.reindex(
        index=[r for r in _RADIUS_ORDER if r in pivot.index],
        columns=[t for t in temp_display if t in pivot.columns],
        fill_value=0.0,
    )
    return pivot


def coverage_counts_matrix(
    state: "MissionState", tier: int = 1
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return (completed, total) count DataFrames matching ``coverage_matrix`` shape.

    Both DataFrames have the same radius × temperature layout as ``coverage_matrix``.
    ``completed[r, t]`` = number of targets in that cell that reached *tier*.
    ``total[r, t]``     = total number of targets in that cell (regardless of tier).
    """
    import pandas as pd

    targets  = state.targets
    progress = state.progress
    done_col = f"tier{tier}_done"

    merged = targets.set_index("target_id").join(progress[[done_col]]).reset_index()

    def _parse_bin(b: str) -> tuple[str, str]:
        b = str(b)
        radius = "unknown"
        for r in _RADIUS_ORDER:
            if b.startswith(r + "_") or b == r:
                radius = r
                break
        remaining = b[len(radius) + 1:] if radius != "unknown" else b
        temp = "unknown"
        for t in _TEMP_ORDER:
            if remaining.startswith(t):
                temp = t
                break
        return radius, temp

    merged[["radius_cls", "temp_cls"]] = pd.DataFrame(
        merged["population_bin"].map(_parse_bin).tolist(),
        index=merged.index,
    )

    temp_display = list(reversed(_TEMP_ORDER))
    idx  = [r for r in _RADIUS_ORDER if r in merged["radius_cls"].values]
    cols = [t for t in temp_display  if t in merged["temp_cls"].values]

    completed = merged.pivot_table(
        values=done_col, index="radius_cls", columns="temp_cls",
        aggfunc="sum", fill_value=0,
    ).reindex(index=idx, columns=cols, fill_value=0)

    total = merged.pivot_table(
        values=done_col, index="radius_cls", columns="temp_cls",
        aggfunc="count", fill_value=0,
    ).reindex(index=idx, columns=cols, fill_value=0)

    return completed, total


# Canonical ordered lists used by both coverage_matrix and the heatmap parser
_RADIUS_ORDER = ["sub_earth", "super_earth", "mini_neptune", "neptune", "saturn", "jupiter"]
# Longer multi-word names listed first so prefix matching works correctly
_TEMP_ORDER   = ["ultra_hot", "very_hot", "hot", "warm", "cold"]


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.

    Returns 0 (perfect equality) to 1 (maximum inequality).
    Used as a diversity measure: lower Gini = more uniform bin coverage.
    """
    v = np.sort(np.abs(values.astype(float)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)


def coverage_gini(state: "MissionState", tier: int = 1) -> float:
    """Gini coefficient of per-bin tier-completion counts.

    Lower is better — means Ariel covered the target population uniformly.
    """
    tbl = coverage_table(state)
    col = f"n_tier{tier}"
    return gini_coefficient(tbl[col].to_numpy())
