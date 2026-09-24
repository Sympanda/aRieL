"""
Action mask computation for both action space types.

A valid action is one where:
  1. visibility_valid is True on the event
  2. The observation block hasn't already fully elapsed (block_end > t_now)
  3. The telescope can arrive before the block ends (t_arrive < block_end)
     giving a non-zero captured_fraction
  4. The actual time cost (slew + idle + captured_duration + overhead) fits
     in the remaining mission time — captured_duration uses the tier-capped
     effective fraction, so a near-completion observation is cheaper than a
     full block
  5. (optional) The target hasn't already reached max_tier

Condition 4 applies to ALL action modes (topk, target, full_set).  In
``full_set`` mode we still allow observations that require long idle waits,
but we always reject observations whose actual time cost would push past
mission_end.  Possible-but-inefficient choices are left available for the
agent to weigh via learned value/reward.

Returns a boolean numpy array of shape (n_candidates,).
True = agent may choose this action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState
    from aRieL.utils.config import ActionConfig


def compute_mask(
    state: "MissionState",
    candidate_events: pd.DataFrame,
    cfg: "ActionConfig",
) -> np.ndarray:
    """Compute the action validity mask for the current candidates.

    Parameters
    ----------
    state:
        Current MissionState.
    candidate_events:
        DataFrame of K events (topk) or N next-events-per-target (target mode).
    cfg:
        ActionConfig controlling which checks to apply.

    Returns
    -------
    np.ndarray of shape (len(candidate_events),), dtype bool.
    """
    if cfg.type == "topk":
        # Started-block exclusion is done in candidate selection
        # (``unstarted_only``), not by zeroing slots after the fact.
        return _mask_topk(state, candidate_events)
    elif cfg.type == "target":
        return _mask_target(
            state, candidate_events,
            include_completed=cfg.target.include_completed,
            permissive=False,
        )
    elif cfg.type == "full_set":
        return _mask_target(
            state, candidate_events,
            include_completed=cfg.full_set.include_completed,
            exclude_started_blocks=cfg.full_set.exclude_started_blocks,
            # permissive=False: can_fit is always checked even for full_set.
            # Long-idle actions are still allowed — the check only rejects
            # observations whose total cost physically exceeds mission_end.
            permissive=False,
        )
    else:
        raise ValueError(f"Unknown action space type: {cfg.type!r}")


def _mask_topk(
    state: "MissionState",
    events: pd.DataFrame,
) -> np.ndarray:
    """Mask for the top-K upcoming event list.

    Timing model (consistent with MissionState.execute_observation):
        slew immediately  →  idle if arrived before block_start  →  observe block

    An action is valid when:
      1. visibility_valid is True
      2. Observation block hasn't fully elapsed (block_end > t_now)
      3. Telescope arrives before block_end (any non-zero capture possible)
      4. Actual time cost (slew + idle + *captured* duration + overhead) fits in
         remaining mission time — NOT full block duration
      5. Target has not yet reached max_tier

    The miss threshold is ``block_end`` (not ``window_end``) consistent with
    MissionState.execute_observation's partial-observation model.  The can_fit
    check uses the *captured* duration (i.e. ``capture_fraction × block_dur``)
    so that a valid partial observation is never rejected simply because the
    full 2.5×T14 block would not fit.

    Note: excluding already-started blocks is handled upstream in
    ``ArielEnv._candidates_topk`` via ``unstarted_only`` candidate selection,
    not here.
    """
    from aRieL.simulator.slew import slew_time_days
    from aRieL.data.schemas import COST_FACTOR

    n = len(events)
    if n == 0:
        return np.zeros(0, dtype=bool)

    t_now    = state.clock.current_time
    remaining = state.clock.remaining_time
    overhead = getattr(state, "overhead_days_per_obs", 0.0)

    # Extract columns once (avoids pandas per-row overhead)
    vis  = events["visibility_valid"].to_numpy(dtype=bool)
    wmid = events["window_mid"].to_numpy(dtype=float)
    tids = events["target_id"].to_numpy()
    # block_duration_days is populated by DynamicBackend; fall back for legacy rows
    if "block_duration_days" in events.columns:
        bldur = events["block_duration_days"].to_numpy(dtype=float)
    else:
        bldur = COST_FACTOR * events["duration_days"].to_numpy(dtype=float)

    bend = wmid + bldur / 2.0   # block_end for each event

    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not vis[i]:
            continue
        if bend[i] <= t_now:
            continue
        target = state._target_lookup.get(tids[i])
        if target is None:
            continue
        prog = state._progress_dict.get(tids[i])
        if prog is not None and int(prog["current_tier"]) >= int(target["max_tier"]):
            continue
        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(target["ra"]), dec2=float(target["dec"]),
        )
        t_arrive = t_now + slew
        # Telescope must arrive before block_end to capture anything.
        if t_arrive >= bend[i]:
            continue
        # Idle: wait for block_start if arrived early
        block_start = wmid[i] - bldur[i] / 2.0
        idle = max(0.0, block_start - t_arrive)
        # Captured duration: mirrors execute_observation's three cases
        if t_arrive <= block_start:
            cap_frac = 1.0
        else:
            cap_frac = (bend[i] - t_arrive) / bldur[i]
        captured_dur = cap_frac * bldur[i]
        # can_fit uses actual captured duration + overhead, not full block
        total_cost = slew + idle + captured_dur + overhead
        if total_cost > remaining:
            continue
        mask[i] = True

    return mask


def _mask_target(
    state: "MissionState",
    next_events: pd.DataFrame,
    include_completed: bool,
    permissive: bool = False,
    exclude_started_blocks: bool = False,
) -> np.ndarray:
    """Mask for the full-target and full_set action spaces.

    ``next_events`` has one row per target (the next available event for that
    target, or a sentinel row with ``visibility_valid=False`` if none).

    Parameters
    ----------
    include_completed:
        If False, targets already at max_tier are masked out.
    permissive:
        Retained for call-site compatibility; the can_fit check is now always
        applied (using tier-capped captured duration) for all modes.  Actions
        that require long idle waits are still allowed — only observations
        whose actual time cost physically exceeds mission_end are rejected.
    exclude_started_blocks:
        If True (``full_set`` only), reject events whose observation block has
        already opened (``t_now >= block_start``).  Partial mid-block captures
        are not offered to the policy.
    """
    from aRieL.simulator.slew import slew_time_days_vec
    from aRieL.data.schemas import COST_FACTOR

    n = len(next_events)
    if n == 0:
        return np.zeros(0, dtype=bool)

    t_now     = state.clock.current_time
    remaining = state.clock.remaining_time
    overhead  = getattr(state, "overhead_days_per_obs", 0.0)

    # ------------------------------------------------------------------
    # Extract all columns as numpy arrays once (avoids per-row Series creation
    # from iterrows(), which costs ~20 ms on 814-row DataFrames).
    # ------------------------------------------------------------------
    tids      = next_events["target_id"].to_numpy()        # object array of str
    vis_valid = next_events["visibility_valid"].to_numpy(dtype=bool)
    wmid      = next_events["window_mid"].to_numpy(dtype=float)
    raw_dur   = next_events["duration_days"].to_numpy(dtype=float)
    block_dur = next_events["block_duration_days"].to_numpy(dtype=float)
    # Fall back to COST_FACTOR × duration_days for rows with no block_duration
    zero_bd   = block_dur <= 0.0
    block_dur = np.where(zero_bd, COST_FACTOR * raw_dur, block_dur)

    block_end   = wmid + block_dur / 2.0
    block_start = wmid - block_dur / 2.0

    # ------------------------------------------------------------------
    # Progress / tier lookups (still needs Python dict access per row, but
    # a plain comprehension over object arrays is faster than iterrows()).
    # Use explicit None checks (not `trow or {}`) because _target_lookup may
    # return pandas Series objects whose truth value raises ValueError.
    # ------------------------------------------------------------------
    tids_str = [str(t) if t is not None else "" for t in tids]
    progs    = [state._progress_dict.get(tid_s, {}) for tid_s in tids_str]
    trows    = [state._target_lookup.get(tid_s) for tid_s in tids_str]

    def _trow_get(i: int, key: str, default):
        tr = trows[i]
        if tr is None:
            return default
        try:
            return tr[key]
        except (KeyError, TypeError, IndexError):
            return default

    # Tier filter
    if not include_completed:
        tier_ok = np.array([
            int(progs[i].get("current_tier", 0)) < int(_trow_get(i, "max_tier", 1))
            for i in range(n)
        ], dtype=bool)
    else:
        tier_ok = np.ones(n, dtype=bool)

    # ------------------------------------------------------------------
    # Vectorised slew → t_arrive (one numpy call instead of N scalar calls)
    # ------------------------------------------------------------------
    ra2  = np.array([float(_trow_get(i, "ra",  0.0)) for i in range(n)])
    dec2 = np.array([float(_trow_get(i, "dec", 0.0)) for i in range(n)])
    slews    = slew_time_days_vec(state.current_ra, state.current_dec, ra2, dec2)
    t_arrive = t_now + slews

    # ------------------------------------------------------------------
    # Tier-capped captured duration for can_fit check
    # ------------------------------------------------------------------
    obs_rem  = np.array([float(progs[i].get("obs_remaining_next_tier", 1.0))
                         for i in range(n)])
    idle     = np.maximum(0.0, block_start - t_arrive)
    safe_bd  = np.where(block_dur > 0, block_dur, 1e-9)
    cap_frac = np.where(t_arrive <= block_start, 1.0,
                        (block_end - t_arrive) / safe_bd)
    eff_frac     = np.minimum(cap_frac, obs_rem)
    captured_dur = eff_frac * block_dur
    total_cost   = slews + idle + captured_dur + overhead

    # ------------------------------------------------------------------
    # Combine all conditions (fully vectorised boolean operations)
    # ------------------------------------------------------------------
    has_tid  = np.array([bool(s) for s in tids_str], dtype=bool)
    has_trow = np.array([t is not None for t in trows],  dtype=bool)

    mask = (
        has_tid
        & has_trow
        & tier_ok
        & vis_valid
        & (block_end > t_now)       # block not yet expired
        & (t_arrive < block_end)    # telescope can arrive in time
        & (total_cost <= remaining) # fits before mission_end
    )
    if exclude_started_blocks:
        # Must arrive in time for a full block, not merely before it expires.
        mask &= (t_arrive <= block_start)
    return mask


def any_valid(mask: np.ndarray) -> bool:
    """Return True if at least one action is valid."""
    return bool(mask.any())
