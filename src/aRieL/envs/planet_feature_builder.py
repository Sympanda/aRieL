"""
Per-planet feature builder for the full-set action space.

Computes a dense feature vector for every target in the catalogue at each
step.  This is the authoritative feature specification for Phase 3 (full
target-set observation space), where the agent sees all N planets rather
than a top-K slice.

Single source-of-truth
----------------------
Dynamic event timing (dt_next_event, block_duration, slew, slack,
capture_fraction, …) is derived from the same event that would be
*executed* if this planet were selected.  When ``per_target_events`` is
provided (from ``_candidates_target()``), those pre-computed events are
used directly.  When omitted (e.g. in tests), ephemeris is computed
independently from orbital parameters — but this path is slower and may
differ from what the backend returns if a partial block is still ongoing.

Feature schema (per planet)
---------------------------
Static (time-invariant)
  planet_radius_norm              planet radius / 20 Re
  planet_mass_norm                planet mass / 4000 Me
  planet_temperature_norm         equilibrium temp / 3000 K
  period_norm                     orbital period / 365.25 days
  stellar_temperature_norm        stellar Teff / 10000 K
  stellar_metallicity             [Fe/H], clipped to [-3, 3]
  distance_norm                   distance / 1000 pc
  tier_goal_norm                  max_tier / 3
  science_weight                  catalogue priority weight [0, 1]
  event_type_binary               0=transit, 1=eclipse, 0.5=either
  host_multiplicity_norm          number of Ariel targets in same system / 5

Dynamic (updated each step)
  obs_completed_norm              obs_completed / tier3_required_obs
  progress_in_tier                current progress toward next tier [0, 1]
  current_tier_norm               current_tier / 3
  obs_remaining_norm              obs_remaining_next_tier / tier3_required_obs

Immediate action-quality (from the event that would be executed right now)
  capture_fraction_now            fraction of the observation block capturable
  block_currently_active          1 if the observation block is open right now
  time_to_block_end_norm          (block_end - t_now) / 10 days
  idle_if_selected_norm           idle wait if selected now / (2hr/24)
  slew_norm                       slew time from current pointing / (2hr/24)
  scheduling_slack_norm           (block_start - t_arrive) / block_duration

Future opportunity features (from next 1–3 events)
  dt_next_event_norm              Δt to next event / 365.25 days
  dt_second_event_norm            Δt to second event / 365.25 days
  dt_third_event_norm             Δt to third event / 365.25 days
  block_duration_norm             block_duration_days (next event) / 3.0
  ephemeris_uncertainty_norm      σ_timing / period (fractional uncertainty)
  remaining_opps_mission_norm     remaining opportunities in mission / available_total
  remaining_opps_season_norm      remaining opportunities in next 90 days / 10

The vector is padded to a fixed length (``n_features``) regardless of
which features are enabled.  Missing values (e.g. no upcoming event) are
filled with 0.

Public API
----------
    from aRieL.envs.planet_feature_builder import (
        build_planet_features,
        PLANET_FEATURE_NAMES,
        N_PLANET_FEATURES,
    )

Changing features
-----------------
Edit ``STATIC_FEATURE_NAMES`` / ``DYNAMIC_FEATURE_NAMES`` and the builders
below.  ``FullSetISABPolicy`` sizes itself from ``observation_space`` so it
adapts to a new feature width after you rebuild the env — then retrain.
See ``docs/CUSTOMISING.md`` and ``docs/CATALOGUE.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState


# ---------------------------------------------------------------------------
# Static feature names (time-invariant per planet)
# ---------------------------------------------------------------------------

STATIC_FEATURE_NAMES: list[str] = [
    "planet_radius_norm",
    "planet_mass_norm",
    "planet_temperature_norm",
    "period_norm",
    "stellar_temperature_norm",
    "stellar_metallicity",
    "distance_norm",
    "tier_goal_norm",
    "science_weight",
    "event_type_binary",
    "host_multiplicity_norm",
]

# Dynamic feature names (computed from MissionState each step)
DYNAMIC_FEATURE_NAMES: list[str] = [
    # Progress state
    "obs_completed_norm",
    "progress_in_tier",
    "current_tier_norm",
    "obs_remaining_norm",
    # Immediate action-quality (same event that would execute if selected now)
    "capture_fraction_now",
    "block_currently_active",
    "time_to_block_end_norm",
    "idle_if_selected_norm",
    "slew_norm",
    "scheduling_slack_norm",
    # Future opportunity features
    "dt_next_event_norm",
    "dt_second_event_norm",
    "dt_third_event_norm",
    "block_duration_norm",
    "ephemeris_uncertainty_norm",
    "remaining_opps_mission_norm",
    "remaining_opps_season_norm",
]

PLANET_FEATURE_NAMES: list[str] = STATIC_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES
N_PLANET_FEATURES: int = len(PLANET_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

_NORM = {
    "planet_radius_norm":        20.0,
    "planet_mass_norm":          4000.0,
    "planet_temperature_norm":   3000.0,
    "period_norm":               365.25,
    "stellar_temperature_norm":  10000.0,
    "stellar_metallicity":       1.5,       # typical range ±1 dex → clip at ±3
    "distance_norm":             1000.0,
    "tier_goal_norm":            3.0,
    "time_to_block_end_norm":    10.0,      # block_end within 10-day lookahead
    "idle_if_selected_norm":     2.0 / 24,  # 2-hr idle cap in days
    "slew_norm":                 2.0 / 24,  # 2-hr slew cap in days
    "dt_next_event_norm":        365.25,
    "dt_second_event_norm":      365.25,
    "dt_third_event_norm":       365.25,
    "block_duration_norm":       3.0,       # 2.5 × T14; max ≈ 2.5 d for long transits
    "host_multiplicity_norm":    5.0,
    "remaining_opps_season_norm": 10.0,
}

_SEASON_WINDOW_DAYS: float = 90.0   # "season" = next 90 days for remaining_opps_season


# ---------------------------------------------------------------------------
# Static table (pre-computed once per episode; host multiplicity included)
# ---------------------------------------------------------------------------

def build_static_features(state: "MissionState") -> np.ndarray:
    """Pre-compute static features for all N targets.

    Returns a float32 array of shape (N, len(STATIC_FEATURE_NAMES)).
    Call this once after reset(); cache the result on the env.
    """
    targets = state.targets
    n = len(targets)
    arr = np.zeros((n, len(STATIC_FEATURE_NAMES)), dtype=np.float32)

    # Pre-compute host multiplicity: number of targets sharing a host_id.
    host_counts: dict[str, int] = {}
    if "host_id" in targets.columns:
        for hid in targets["host_id"]:
            if hid and str(hid) != "nan":
                host_counts[str(hid)] = host_counts.get(str(hid), 0) + 1

    for i in range(n):
        row = targets.iloc[i]
        hid = str(row.get("host_id", ""))
        pref = str(row.get("preferred_method", "transit")).lower()

        arr[i, 0] = float(row.get("planet_radius", 0.0))           / _NORM["planet_radius_norm"]
        arr[i, 1] = float(row.get("planet_mass", 0.0))             / _NORM["planet_mass_norm"]
        arr[i, 2] = float(row.get("planet_temperature", 0.0))      / _NORM["planet_temperature_norm"]
        arr[i, 3] = float(row.get("period", 0.0))                  / _NORM["period_norm"]
        arr[i, 4] = float(row.get("stellar_temperature", 0.0))     / _NORM["stellar_temperature_norm"]
        arr[i, 5] = float(row.get("stellar_metallicity", 0.0))
        arr[i, 6] = float(row.get("distance_pc", 0.0))             / _NORM["distance_norm"]
        arr[i, 7] = float(row.get("max_tier", 1))                  / _NORM["tier_goal_norm"]
        arr[i, 8] = float(row.get("science_weight", 0.5))
        # 0=transit, 1=eclipse, 0.5=either
        arr[i, 9] = 1.0 if "eclipse" in pref else (0.5 if "either" in pref else 0.0)
        arr[i, 10] = min(host_counts.get(hid, 1), 5) / _NORM["host_multiplicity_norm"]

    # Clip stellar metallicity to [-3, 3] range then normalise
    arr[:, 5] = np.clip(arr[:, 5], -3.0, 3.0) / _NORM["stellar_metallicity"]

    return arr


def build_planet_features(
    state: "MissionState",
    static_features: np.ndarray | None = None,
    per_target_events: dict[str, dict] | None = None,
    target_ids: list[str] | None = None,
) -> np.ndarray:
    """Compute the full per-planet feature matrix for the current step.

    Parameters
    ----------
    state:
        Current MissionState (read-only).
    static_features:
        Pre-computed static features.  Shape must be ``(N, n_static)`` where
        N matches the number of targets being built (either ``len(target_ids)``
        when that argument is provided, or ``len(state.targets)`` otherwise).
        If None, static features are computed on the fly from ``state``.
    per_target_events:
        Mapping from ``target_id`` to the event dict that would be executed
        if that planet were selected right now.
    target_ids:
        Optional ordered list of target IDs to build features for.  Only
        these targets are included in the output (in this order).  Useful for
        the dynamic active-set where completed planets have been removed.
        When None (default), all targets in ``state.targets`` are used.

    Returns
    -------
    float32 array of shape (N, N_PLANET_FEATURES) where N = len(target_ids)
    if provided, else len(state.targets).
    """
    from aRieL.simulator.slew import slew_time_days
    from aRieL.data.schemas import COST_FACTOR, MISSION_LIFETIME_DAYS

    if target_ids is not None:
        target_list = target_ids
        n = len(target_list)
    else:
        target_list = [str(r["target_id"]) for _, r in state.targets.iterrows()]
        n = len(state.targets)

    n_static = len(STATIC_FEATURE_NAMES)
    n_dynamic = len(DYNAMIC_FEATURE_NAMES)

    arr = np.zeros((n, N_PLANET_FEATURES), dtype=np.float32)

    if static_features is not None:
        arr[:, :n_static] = static_features
    else:
        # Compute static features only for the target subset
        if target_ids is not None:
            # Build static features for all targets first, then filter
            all_static = build_static_features(state)
            all_tids = [str(r["target_id"]) for _, r in state.targets.iterrows()]
            tid_to_idx = {tid: i for i, tid in enumerate(all_tids)}
            for j, tid in enumerate(target_list):
                si = tid_to_idx.get(tid)
                if si is not None:
                    arr[j, :n_static] = all_static[si]
        else:
            arr[:, :n_static] = build_static_features(state)

    t_now = state.clock.current_time
    mission_end = state.clock.mission_end
    season_end = t_now + _SEASON_WINDOW_DAYS

    for i, tid in enumerate(target_list):
        row = state._target_lookup.get(tid)
        if row is None:
            continue
        prog = state._progress_dict.get(tid, {})

        # ---- progress features ----
        t3_req = int(row.get("tier3_required_obs", 1)) or 1
        obs_done = float(prog.get("obs_completed", 0.0))
        t_prog = float(prog.get("progress_in_tier", 0.0))
        t_tier = int(prog.get("current_tier", 0))
        obs_rem = float(prog.get("obs_remaining_next_tier", float(t3_req)))

        arr[i, n_static + 0] = obs_done / t3_req
        arr[i, n_static + 1] = t_prog
        arr[i, n_static + 2] = t_tier / 3.0
        arr[i, n_static + 3] = obs_rem / t3_req

        # ---- immediate action-quality features ----
        # Use the pre-computed event from _candidates_target when available
        # so that the features describe exactly the event that would execute.
        ev = per_target_events.get(tid) if per_target_events is not None else None

        if ev is not None:
            # Trusted path: event from the same backend call used for execution.
            block_dur   = float(ev.get("block_duration_days", 0.0))
            window_mid  = float(ev.get("window_mid", t_now))
            block_start = window_mid - block_dur / 2.0
            block_end   = window_mid + block_dur / 2.0
            ev_ra       = float(row.get("ra", 0.0))
            ev_dec      = float(row.get("dec", 0.0))
        else:
            # Fallback: compute from orbital parameters.
            period      = float(row.get("period", 1.0)) or 1.0
            epoch       = float(row.get("epoch", t_now))
            tr_dur_days = float(row.get("transit_duration", 0.0)) / 86400.0
            block_dur   = COST_FACTOR * tr_dur_days
            phase       = (t_now - epoch) % period
            half_block  = block_dur * COST_FACTOR / 2.0
            in_block    = phase < (COST_FACTOR * tr_dur_days / 2.0)
            if in_block:
                window_mid = t_now - phase
            else:
                window_mid = t_now + (period - phase)
            block_start = window_mid - block_dur / 2.0
            block_end   = window_mid + block_dur / 2.0
            ev_ra       = float(row.get("ra", 0.0))
            ev_dec      = float(row.get("dec", 0.0))

        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=ev_ra, dec2=ev_dec,
        )
        t_arrive = t_now + slew

        # capture_fraction: mirrors execute_observation's three cases
        if block_dur <= 0.0 or t_arrive >= block_end:
            capture_fraction = 0.0
        elif t_arrive <= block_start:
            capture_fraction = 1.0
        else:
            capture_fraction = (block_end - t_arrive) / block_dur

        block_active  = 1.0 if (t_now > block_start and t_now < block_end) else 0.0
        time_to_bend  = max(0.0, block_end - t_now)
        idle_now      = max(0.0, block_start - t_arrive)

        # scheduling slack: how much margin before block_start if selected now
        slack = (block_start - t_arrive) / max(block_dur, 1e-6)

        arr[i, n_static + 4] = float(np.clip(capture_fraction, 0.0, 1.0))
        arr[i, n_static + 5] = block_active
        arr[i, n_static + 6] = time_to_bend  / _NORM["time_to_block_end_norm"]
        arr[i, n_static + 7] = idle_now      / _NORM["idle_if_selected_norm"]
        arr[i, n_static + 8] = slew          / _NORM["slew_norm"]
        arr[i, n_static + 9] = float(np.clip(slack, -1.0, 10.0)) / 10.0

        # ---- future opportunity features (via backend — single source of truth) ----
        # The future-event sequence MUST be anchored to the same event that
        # would execute if the agent selects this planet.  When per_target_events
        # provides the first-reachable event (ev), we start the lookahead from
        # that event's window_mid so that:
        #   future_events[0] = ev  (= event_1, the action event)
        #   future_events[1] = next occurrence after ev  (= event_2)
        #   future_events[2] = two occurrences after ev  (= event_3)
        # Without this anchor, an unreachable earlier event could contaminate
        # dt_next_event and make the immediate features describe different events.
        period = float(row.get("period", 1.0)) or 1.0
        future_anchor = float(ev["window_mid"]) if ev is not None else t_now
        future_events = state._backend.events_for_target(tid, future_anchor, n=3)

        # dt features: use backend events when available
        for k_ev, slot in enumerate([10, 11, 12]):
            if k_ev < len(future_events):
                dt = max(0.0, float(future_events[k_ev]["window_mid"]) - t_now)
            else:
                dt = 365.25  # no further events → clip at normalisation ceiling
            arr[i, n_static + slot] = min(dt, 365.25) / _NORM[
                ["dt_next_event_norm", "dt_second_event_norm", "dt_third_event_norm"][k_ev]
            ]

        # block_duration: from the first backend event (or from immediate action ev)
        if future_events:
            bd_future = float(future_events[0].get("block_duration_days", block_dur))
        else:
            bd_future = block_dur
        arr[i, n_static + 13] = min(bd_future, 3.0) / _NORM["block_duration_norm"]

        # Ephemeris uncertainty (fractional: σ_timing / period)
        unc_days = float(row.get("epoch_uncertainty", 0.0)) or 0.0
        arr[i, n_static + 14] = min(unc_days / period, 1.0)

        # Remaining opportunities: count future mid-times within mission / season.
        # Use backend to count (estimate from period spacing when exact count unknown).
        avail_total = float(row.get("available_transits", 0)) or 1.0
        if future_events:
            first_future_mid = float(future_events[0]["window_mid"])
        else:
            phase = (t_now - float(row.get("epoch", t_now))) % period
            first_future_mid = t_now + (period - phase)

        remaining_mission = max(0.0, (mission_end - first_future_mid) / period)
        arr[i, n_static + 15] = min(remaining_mission / avail_total, 1.0)

        remaining_season = max(0.0, (season_end - first_future_mid) / period)
        arr[i, n_static + 16] = min(remaining_season, 10.0) / _NORM["remaining_opps_season_norm"]

    # Global clip to reasonable range
    arr = np.clip(arr, -3.0, 3.0)

    return arr
