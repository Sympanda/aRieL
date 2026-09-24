"""
Build the static event DataFrame from the processed target table.

One call to ``generate_events`` produces a table of every transit and eclipse
window that falls within the mission lifetime.  This table is computed once
at the start of an episode (or cached) and used throughout.

Event window definition
-----------------------
For each event:
  window_mid   = ephemeris mid-time (BJD)
  window_start = window_mid - duration_days / 2
  window_end   = window_mid + duration_days / 2

No additional overhead is added here; scheduling overhead (slew, settle)
is accounted for in the mission_state / cost calculation at action time.

Visibility
----------
Ariel field-of-regard and solar-exclusion constraints are not implemented.
``_check_visibility`` marks every event visible.

Science value
-------------
``base_science_value`` is a static score on [0, 1] combining the target's
``science_weight`` (population rarity) and a simple SNR proxy.  It is *not*
the reward — the reward function conditions on mission state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aRieL.data.schemas import (
    COST_FACTOR,
    METHOD_ECLIPSE,
    METHOD_EITHER,
    METHOD_TRANSIT,
    MISSION_END_BJD,
    MISSION_START_BJD,
)
from aRieL.simulator.ephemeris import propagate


def generate_events(
    targets: pd.DataFrame,
    mission_start: float = MISSION_START_BJD,
    mission_end: float = MISSION_END_BJD,
) -> pd.DataFrame:
    """Generate the full event table for an episode.

    Parameters
    ----------
    targets:
        Processed target DataFrame from ``preprocess_targets.build_target_table``.
    mission_start, mission_end:
        BJD bounds of the mission window.

    Returns
    -------
    pd.DataFrame
        One row per observable transit/eclipse, sorted by ``window_mid``.
        Columns: see ``schemas.EVENT_COLS``.
    """
    records: list[dict] = []
    event_id = 0

    for _, row in targets.iterrows():
        target_id = row["target_id"]
        period = float(row["period"])
        epoch = float(row["epoch"])

        transit_dur_s = float(row["transit_duration"]) if pd.notna(row.get("transit_duration")) else np.nan
        eclipse_dur_s = float(row["eclipse_duration"]) if pd.notna(row.get("eclipse_duration")) else np.nan

        method = str(row.get("preferred_method") or METHOD_TRANSIT)
        eccentricity = float(row["eccentricity"]) if pd.notna(row.get("eccentricity")) else 0.0
        inclination = float(row["inclination"]) if pd.notna(row.get("inclination")) else 90.0
        # argument of periastron not in our target schema directly; default 0
        omega_deg = 0.0

        sigma_epoch = float(row["epoch_uncertainty"]) if pd.notna(row.get("epoch_uncertainty")) else 0.0
        sigma_period = 0.0  # period uncertainty not separately stored in schema

        science_weight = float(row["science_weight"]) if pd.notna(row.get("science_weight")) else 0.5
        max_tier = int(row["max_tier"]) if pd.notna(row.get("max_tier")) else 1

        # Which event types to generate for this target?
        generate_transit = method in (METHOD_TRANSIT, METHOD_EITHER) and not np.isnan(transit_dur_s)
        generate_eclipse = method in (METHOD_ECLIPSE, METHOD_EITHER) and not np.isnan(eclipse_dur_s)

        # If eclipse method but no eclipse duration, fall back to transit duration
        if method == METHOD_ECLIPSE and np.isnan(eclipse_dur_s) and not np.isnan(transit_dur_s):
            eclipse_dur_s = transit_dur_s
            generate_eclipse = True

        for etype, dur_s in [
            ("transit", transit_dur_s),
            ("eclipse", eclipse_dur_s),
        ]:
            if etype == "transit" and not generate_transit:
                continue
            if etype == "eclipse" and not generate_eclipse:
                continue
            if np.isnan(dur_s):
                continue

            result = propagate(
                target_id=target_id,
                epoch=epoch,
                period=period,
                t_start=mission_start,
                t_end=mission_end,
                event_type=etype,
                eccentricity=eccentricity,
                omega_deg=omega_deg,
                sigma_epoch_days=sigma_epoch,
                sigma_period_days=sigma_period,
            )

            if len(result.mid_times) == 0:
                continue

            dur_days = dur_s / 86400.0
            base_val = _base_science_value(row, science_weight, dur_s)
            valid = _check_visibility(result.mid_times, dur_days)

            for i, (n, t_mid, sigma_s) in enumerate(
                zip(result.indices, result.mid_times, result.uncertainties)
            ):
                records.append(
                    {
                        "event_id":              event_id,
                        "target_id":             target_id,
                        "event_type":            etype,
                        "window_start":          t_mid - dur_days / 2.0,
                        "window_mid":            t_mid,
                        "window_end":            t_mid + dur_days / 2.0,
                        "duration":              dur_s,
                        "duration_days":         dur_days,
                        "block_duration_days":   COST_FACTOR * dur_days,
                        "tier_goal":             max_tier,
                        "base_science_value":    base_val,
                        "visibility_valid":      bool(valid[i]),
                        "ephemeris_uncertainty": float(sigma_s),
                        "event_index":           int(n),
                    }
                )
                event_id += 1

    if not records:
        return pd.DataFrame(columns=[
            "event_id", "target_id", "event_type",
            "window_start", "window_mid", "window_end",
            "duration", "duration_days", "block_duration_days", "tier_goal",
            "base_science_value", "visibility_valid",
            "ephemeris_uncertainty", "event_index",
        ])

    events = pd.DataFrame(records)
    events = events.sort_values("window_mid").reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_science_value(row: pd.Series, science_weight: float, dur_s: float) -> float:
    """Static science value in [0, 1] for a single observation.

    Combines population rarity (science_weight) and a rough SNR proxy
    (shorter transits = higher SNR per unit time for a fixed photon budget).

    This is intentionally simple — the reward function adds context.
    """
    # Penalise very long observations relative to mission budget
    # T14 median ~8000 s; normalise around that
    dur_norm = np.clip(8000.0 / max(dur_s, 100.0), 0.1, 2.0) / 2.0  # 0.05 – 1
    return float(np.clip(0.6 * science_weight + 0.4 * dur_norm, 0.0, 1.0))


def _check_visibility(mid_times: np.ndarray, dur_days: float) -> np.ndarray:
    """Return visibility for each event.

    Ariel field-of-regard and solar-exclusion geometry are not implemented.
    Every event is marked visible.
    """
    return np.ones(len(mid_times), dtype=bool)


def save_events(events: pd.DataFrame, path: str | Path = "data/processed/events.parquet") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(p, index=False)


def load_events(path: str | Path = "data/processed/events.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
