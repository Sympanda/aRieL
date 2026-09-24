"""
Tier observation requirements and per-target observation cost computation.

Key design:
  - Tier obs counts in the MCS are *cumulative* (e.g. T1=3, T2=7, T3=12 means
    you need 3 total obs for Tier 1, 7 for Tier 2, 12 for Tier 3).
  - Cost per observation: cost_days = COST_FACTOR * duration_s / 86400
    where duration_s is the T14 (transit) or E14 (eclipse) duration.
  - The preferred_method column determines which duration to use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aRieL.data.schemas import COST_FACTOR, METHOD_ECLIPSE, METHOD_TRANSIT


def add_observation_costs(targets: pd.DataFrame) -> pd.DataFrame:
    """Add ``obs_cost_days_t1/t2/t3`` columns to *targets*.

    The cost model is the same for all tiers — it's the wall-clock time
    required for a single observation.  Tier differences come from the
    *number* of required observations, not the per-obs duration.

    Cost = COST_FACTOR * observation_duration_seconds / 86400

    The observation duration is:
      - transit_duration (T14) for "Transit" and "Either" targets
      - eclipse_duration (E14) for "Eclipse" targets
      - max(transit, eclipse) if either is missing

    Returns
    -------
    pd.DataFrame
        Same rows, with three new cost columns filled.
    """
    targets = targets.copy()

    def _single_obs_cost(row: pd.Series) -> float:
        method = str(row.get("preferred_method", "Transit") or "Transit")
        t14 = row.get("transit_duration", np.nan)
        e14 = row.get("eclipse_duration", np.nan)

        if method == METHOD_ECLIPSE:
            dur = e14 if pd.notna(e14) else t14
        else:
            dur = t14 if pd.notna(t14) else e14

        if pd.isna(dur):
            return np.nan

        return float(COST_FACTOR * dur / 86400.0)

    costs = targets.apply(_single_obs_cost, axis=1)
    targets["obs_cost_days_t1"] = costs
    targets["obs_cost_days_t2"] = costs   # per-obs cost identical; tier ≠ cost schedule
    targets["obs_cost_days_t3"] = costs

    return targets


# ---------------------------------------------------------------------------
# Progress computation  (called during episode, not just at startup)
# ---------------------------------------------------------------------------

def compute_progress(obs_completed: float, target_row: pd.Series) -> dict:
    """Compute tier progress state for a single target given obs_completed.

    Parameters
    ----------
    obs_completed:
        Equivalent observations executed so far for this target.  This is a
        *float* to support partial-observation credit (e.g. 0.6 equivalent obs
        when the telescope arrives mid-block and captures 60 % of the window).
        Tier thresholds are still integers; progress crosses a tier boundary
        when ``obs_completed`` first meets or exceeds the threshold.
    target_row:
        A row from the processed target DataFrame (must have
        tier1_required_obs, tier2_required_obs, tier3_required_obs, max_tier).

    Returns
    -------
    dict with keys:
        obs_completed, current_tier, tier1_done, tier2_done, tier3_done,
        progress_in_tier, obs_remaining_next_tier
    """
    t1 = int(target_row["tier1_required_obs"]) if pd.notna(target_row["tier1_required_obs"]) else 1
    t2 = int(target_row["tier2_required_obs"]) if pd.notna(target_row["tier2_required_obs"]) else t1
    t3 = int(target_row["tier3_required_obs"]) if pd.notna(target_row["tier3_required_obs"]) else t2
    max_tier = int(target_row["max_tier"]) if pd.notna(target_row["max_tier"]) else 1

    # Clamp to what's reachable
    reachable_t1 = t1
    reachable_t2 = t2 if max_tier >= 2 else t1
    reachable_t3 = t3 if max_tier >= 3 else reachable_t2

    tier1_done = obs_completed >= reachable_t1
    tier2_done = obs_completed >= reachable_t2 and max_tier >= 2
    tier3_done = obs_completed >= reachable_t3 and max_tier >= 3

    if tier3_done:
        current_tier = 3
        progress_in_tier = 1.0
        obs_remaining = 0
    elif tier2_done:
        current_tier = 2
        if max_tier >= 3:
            span = reachable_t3 - reachable_t2
            progress_in_tier = (obs_completed - reachable_t2) / span if span > 0 else 1.0
            obs_remaining = max(0, reachable_t3 - obs_completed)
        else:
            progress_in_tier = 1.0
            obs_remaining = 0
    elif tier1_done:
        current_tier = 1
        if max_tier >= 2:
            span = reachable_t2 - reachable_t1
            progress_in_tier = (obs_completed - reachable_t1) / span if span > 0 else 1.0
            obs_remaining = max(0, reachable_t2 - obs_completed)
        else:
            progress_in_tier = 1.0
            obs_remaining = 0
    else:
        current_tier = 0
        progress_in_tier = obs_completed / reachable_t1 if reachable_t1 > 0 else 0.0
        obs_remaining = max(0, reachable_t1 - obs_completed)

    return {
        "obs_completed":           obs_completed,
        "current_tier":            current_tier,
        "tier1_done":              tier1_done,
        "tier2_done":              tier2_done,
        "tier3_done":              tier3_done,
        "progress_in_tier":        float(np.clip(progress_in_tier, 0.0, 1.0)),
        "obs_remaining_next_tier": obs_remaining,
    }


def initialise_progress_table(targets: pd.DataFrame) -> pd.DataFrame:
    """Build a zeroed progress table from the target table.

    Returns
    -------
    pd.DataFrame
        One row per target, all obs_completed=0.0, current_tier=0.
    """
    rows = []
    for _, row in targets.iterrows():
        p = compute_progress(0.0, row)
        p["target_id"] = row["target_id"]
        p["max_tier"] = int(row["max_tier"]) if pd.notna(row["max_tier"]) else 1
        rows.append(p)

    col_order = [
        "target_id", "obs_completed", "current_tier",
        "tier1_done", "tier2_done", "tier3_done",
        "progress_in_tier", "obs_remaining_next_tier", "max_tier",
    ]
    return pd.DataFrame(rows)[col_order].set_index("target_id")
