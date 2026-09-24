"""
Observation builder: converts MissionState + candidate events into the
numpy arrays the agent actually sees.

The observation is a dict with two arrays:

  "events"  : float32 array of shape (K, 18)
              One row per candidate event, in candidate order.  Rows beyond
              the number of real events are zero-padded (corresponding to
              invalid / masked actions).

  "global"  : float32 array of shape (G,)
              Mission-level state: 8 named features + one feature per
              population bin that has ≥ cfg.min_bin_targets targets.

Both arrays are normalised to roughly [0, 1] when cfg.normalise=True.
Event features that can be negative (e.g. stellar_metallicity) are clipped
to [−3, 3].  Global features are clipped to [0, 1].

The builder is **stateless**: call ``build(state, candidates, cfg)`` at
every step.  It does not mutate state.

Feature audit (audited on 1 500 random-valid-action steps, 60-day episodes):
  - No event feature is constant; window_urgency_norm replaces the mostly-zero
    wait_time_days; days_to_window_end_norm replaces the constant-1 is_valid.
  - obs_remaining_next_tier_norm is normalised per-target (by the target's own
    tier3_required_obs) rather than by the catalogue-wide maximum, giving a
    proper [0, 1] fraction of total observation budget remaining.
  - Global bin fractions are normalised per-bin rather than by total-catalogue
    size, so all bins are on the same [0, 1] scale regardless of bin size.
  - Bins with < cfg.min_bin_targets targets are excluded; they almost never
    appear in the k-nearest candidates during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState
    from aRieL.utils.config import ObservationConfig

# ---------------------------------------------------------------------------
# Normalisation constants  (used when cfg.normalise=True)
# ---------------------------------------------------------------------------

_NORM = {
    # event-level
    "slew_time_days":               2.0 / 24,   # 2-hr slew cap in days
    "duration_days":                1.0,         # raw T14 ≤ 1 day
    "block_duration_days":          1.0,         # 2.5 × T14 ≤ ~2.5 days; clip keeps in [0,1]
    "total_time_cost_days":         3.0,         # slew + idle + block; max ~3 days
    "capture_fraction":             1.0,         # already in [0, 1]
    "obs_remaining_next_tier_norm": 1.0,         # already normalised
    "days_to_block_end_norm":       5.0,         # block_end within ~5-day lookahead for k=50
    "planet_radius_norm":           20.0,        # Re (Jupiter ~11 Re)
    "planet_temperature_norm":      3000.0,      # K
    "planet_mass_norm":             4000.0,      # Earth masses (10 MJup)
    "stellar_temperature_norm":     10000.0,     # K
    "stellar_metallicity":          1.5,         # typical range ±1 dex
    "tier_goal_norm":               1.0,         # already /3
    # global
    "n_observations_norm":          5000.0,      # reasonable upper bound
    "used_idle_fraction":           1.0,         # already a fraction
}


def build(
    state: "MissionState",
    candidate_events: pd.DataFrame,
    cfg: "ObservationConfig",
) -> dict[str, np.ndarray]:
    """Build the observation dict for the current step.

    Parameters
    ----------
    state:
        Current MissionState (read-only).
    candidate_events:
        Subset of the event table representing the current action candidates.
        For ``topk`` this is the next K events; for ``target`` it is one row
        per target (the next event for each target, or a zero-padded dummy).
    cfg:
        ObservationConfig controlling which features to include.

    Returns
    -------
    dict with keys ``"events"`` (shape ``(K, 16)``) and ``"global"``
    (shape ``(8 + n_large_bins,)``).  See module docstring for full
    feature list.
    """
    n = len(candidate_events)
    n_ef = len(cfg.event_features)
    n_gf = _global_feature_count(state, cfg)

    events_arr = np.zeros((n, n_ef), dtype=np.float32)
    global_arr = np.zeros(n_gf, dtype=np.float32)

    # Pre-compute shared constants once per call (not once per row).
    max_obs_rem = state._max_obs_rem_val
    t_now = state.clock.current_time

    # ---- per-event features ----
    for i, (_, ev) in enumerate(candidate_events.iterrows()):
        events_arr[i] = _build_event_row(ev, state, cfg, t_now, max_obs_rem)

    # ---- global features ----
    global_arr[:] = _build_global(state, cfg)

    return {"events": events_arr, "global": global_arr}


def _build_event_row(
    ev: pd.Series,
    state: "MissionState",
    cfg: "ObservationConfig",
    t_now: float,
    max_obs_rem: int,
) -> np.ndarray:
    target_id = ev["target_id"]
    target = state._target_lookup.get(target_id)

    from aRieL.data.schemas import COST_FACTOR
    # Compute dynamic event costs
    slew_days    = _slew_to_event(ev, state)
    window_start = float(ev["window_start"])
    window_end   = float(ev["window_end"])
    window_mid   = float(ev["window_mid"])
    dur_days     = float(ev["duration_days"])
    block_dur    = float(ev["block_duration_days"]) if "block_duration_days" in ev.index else COST_FACTOR * dur_days
    # Idle time if arrived early (consistent with execute_observation)
    t_arrive     = t_now + slew_days
    block_start  = window_mid - block_dur / 2.0
    block_end    = window_mid + block_dur / 2.0
    idle_days    = max(0.0, block_start - t_arrive)

    # Fraction of the observation block that would be captured if chosen now.
    # Mirrors the three cases in MissionState.execute_observation:
    #   Case A: t_arrive ≤ block_start  → 1.0
    #   Case B: block_start < t_arrive < block_end → (block_end - t_arrive) / block_dur
    #   Case C: t_arrive ≥ block_end    → 0.0
    if t_arrive >= block_end:
        capture_fraction = 0.0
    elif t_arrive <= block_start:
        capture_fraction = 1.0
    else:
        capture_fraction = (block_end - t_arrive) / block_dur

    # Window urgency: fraction of the raw transit window already elapsed.
    # 0 = window just opened, approaching 1 = window nearly closed.
    window_dur = max(window_end - window_start, 1e-6)
    window_urgency = max(0.0, (t_now - window_start) / window_dur)

    # Time remaining until the observation *block* closes (block_end, not window_end).
    # This is the correct scheduling deadline: the agent can still partially
    # capture the observation as long as block_end > t_now.
    days_to_block_end = max(0.0, block_end - t_now)

    # Progress for this target — use fast dict instead of pandas .loc
    prog_row = state._progress_dict.get(target_id)
    progress_in_tier = float(prog_row["progress_in_tier"]) if prog_row is not None else 0.0
    obs_rem = float(prog_row["obs_remaining_next_tier"]) if prog_row is not None else 1.0

    # Effective capture = what the agent would actually receive from this event,
    # capped at the tier boundary (mirrors execute_observation's tier-scoped logic).
    # When obs_rem is large (target far from tier completion), this equals capture_fraction.
    # As the tier nears completion, effective_capture shrinks, reducing predicted cost.
    effective_capture = min(capture_fraction, obs_rem) if obs_rem > 0.0 else 0.0

    # Total time cost uses effective duration so the agent sees the real cost,
    # not the maximum possible window cost.
    total_cost   = slew_days + idle_days + effective_capture * block_dur

    # Normalize obs_remaining by the target's *own* tier-3 requirement so
    # "fraction of work left" is in [0, 1] independent of catalogue-wide scale.
    target_max_obs = (
        int(target["tier3_required_obs"])
        if target is not None and pd.notna(target.get("tier3_required_obs"))
        else max(max_obs_rem, 1)
    )
    obs_rem_norm = obs_rem / max(target_max_obs, 1)

    values: dict[str, float] = {
        "slew_time_days":               slew_days,
        "window_urgency_norm":          window_urgency,        # already in [0, 1]
        "duration_days":                dur_days,
        "block_duration_days":          block_dur,
        "total_time_cost_days":         total_cost,
        "capture_fraction":             capture_fraction,      # already in [0, 1]
        "progress_in_tier":             progress_in_tier,
        "obs_remaining_next_tier_norm": obs_rem_norm,
        "base_science_value":           float(ev.get("base_science_value", 0.0)),
        "science_weight":               float(target["science_weight"]) if target is not None else 0.0,
        "planet_radius_norm":           float(target["planet_radius"]) if target is not None else 0.0,
        "planet_temperature_norm":      float(target["planet_temperature"]) if target is not None else 0.0,
        "planet_mass_norm":             float(target["planet_mass"]) if target is not None else 0.0,
        "stellar_temperature_norm":     float(target["stellar_temperature"]) if target is not None else 0.0,
        "stellar_metallicity":          float(target["stellar_metallicity"]) if target is not None and pd.notna(target["stellar_metallicity"]) else 0.0,
        "tier_goal_norm":               float(ev.get("tier_goal", 1)) / 3.0,
        "event_type_binary":            1.0 if ev.get("event_type") == "eclipse" else 0.0,
        "days_to_block_end_norm":       days_to_block_end,
    }

    row = np.array([values.get(f, 0.0) for f in cfg.event_features], dtype=np.float32)

    if cfg.normalise:
        for j, fname in enumerate(cfg.event_features):
            if fname in _NORM:
                row[j] = row[j] / _NORM[fname]
        row = np.clip(row, -3.0, 3.0)

    return row


def _build_global(
    state: "MissionState",
    cfg: "ObservationConfig",
) -> np.ndarray:
    clk = state.clock
    n_total = state.total_targets

    mission_len = max(clk.mission_end - clk.mission_start, 1)

    # Fraction of targets fully completed (current_tier >= max_tier).
    n_completed = 0
    for tid, prog in state._progress_dict.items():
        target_row = state._target_lookup.get(tid)
        max_tier = int(target_row["max_tier"]) if target_row is not None else 99
        if int(prog["current_tier"]) >= max_tier:
            n_completed += 1

    values: dict[str, float] = {
        "fraction_elapsed":           clk.fraction_elapsed,
        "tier1_fraction":             state.tier1_completed / max(n_total, 1),
        "tier2_fraction":             state.tier2_completed / max(n_total, 1),
        "tier3_fraction":             state.tier3_completed / max(n_total, 1),
        "used_science_fraction":      clk.used_science_time / mission_len,
        "used_slew_fraction":         clk.used_slew_time / mission_len,
        "used_idle_fraction":         clk.used_idle_time / mission_len,
        "n_observations_norm":        clk.n_observations,
        "n_completed_targets_norm":   n_completed / max(n_total, 1),
    }

    base = [values.get(f, 0.0) for f in cfg.global_features]

    if cfg.include_population_bin_fractions:
        bin_obs    = state.population_bin_counts   # observations made per bin
        bin_totals = state._bin_totals             # live targets per bin
        # Prefer a frozen vocabulary (MCS ∪ TPC) so the global vector length
        # does not change when planets are injected mid-episode.
        bin_totals_gate = getattr(state, "_bin_totals_obs", None) or bin_totals
        vocab = getattr(state, "_bin_vocab", None) or ()
        if vocab:
            all_bins = sorted(
                b for b in vocab
                if bin_totals_gate.get(b, 0) >= cfg.min_bin_targets
            )
        else:
            all_bins = sorted(
                b for b in state.targets["population_bin"].unique()
                if bin_totals_gate.get(b, 0) >= cfg.min_bin_targets
            )
        for b in all_bins:
            # Fraction of targets in this bin that have been observed at least once.
            # Normalised per-bin (not by total catalogue size) so rare bins are
            # distinguishable from common ones at any coverage level.
            # Denominator is the *live* bin total so the feature stays in [0, 1].
            base.append(bin_obs.get(b, 0) / max(bin_totals.get(b, 1), 1))

    arr = np.array(base, dtype=np.float32)

    if cfg.normalise:
        for j, fname in enumerate(cfg.global_features):
            if fname in _NORM:
                arr[j] = arr[j] / _NORM[fname]
        arr = np.clip(arr, 0.0, 1.0)

    return arr


def _global_feature_count(state: "MissionState", cfg: "ObservationConfig") -> int:
    n = len(cfg.global_features)
    if cfg.include_population_bin_fractions:
        bin_totals_gate = getattr(state, "_bin_totals_obs", None) or state._bin_totals
        vocab = getattr(state, "_bin_vocab", None) or ()
        if vocab:
            n += sum(
                1 for b in vocab
                if bin_totals_gate.get(b, 0) >= cfg.min_bin_targets
            )
        else:
            n += sum(
                1 for b in state.targets["population_bin"].unique()
                if bin_totals_gate.get(b, 0) >= cfg.min_bin_targets
            )
    return n


def _slew_to_event(ev: pd.Series, state: "MissionState") -> float:
    """Slew time in days from current pointing to the event's target."""
    target_id = ev["target_id"]
    target = state._target_lookup.get(target_id)
    if target is None:
        return 0.0

    from aRieL.simulator.slew import slew_time_days
    return slew_time_days(
        ra1=state.current_ra,
        dec1=state.current_dec,
        ra2=float(target["ra"]),
        dec2=float(target["dec"]),
    )


def observation_shapes(
    state: "MissionState",
    cfg: "ObservationConfig",
    n_candidates: int,
) -> dict[str, tuple[int, ...]]:
    """Return the shapes of each array in the observation dict."""
    return {
        "events": (n_candidates, len(cfg.event_features)),
        "global": (_global_feature_count(state, cfg),),
    }
