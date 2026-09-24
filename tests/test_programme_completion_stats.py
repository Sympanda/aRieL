"""Programme-depth and fully-complete counts in compute_stats."""

from __future__ import annotations

import pandas as pd

from aRieL.data.observation_requirements import initialise_progress_table
from aRieL.evaluation.metrics import compute_stats
from aRieL.simulator.mission_clock import MissionClock
from aRieL.simulator.mission_state import MissionState


def _minimal_state(max_tiers, current_tiers) -> MissionState:
    n = len(max_tiers)
    tids = [f"t{i}" for i in range(n)]
    targets = pd.DataFrame({
        "target_id": tids,
        "max_tier": max_tiers,
        "ra": [0.0] * n,
        "dec": [0.0] * n,
        "population_bin": ["A"] * n,
        "tier1_required_obs": [1] * n,
        "tier2_required_obs": [1] * n,
        "tier3_required_obs": [1] * n,
        "obs_cost_days": [1.0] * n,
    })
    progress = initialise_progress_table(targets)
    for tid, ct in zip(tids, current_tiers):
        progress.at[tid, "current_tier"] = int(ct)
        progress.at[tid, "tier1_done"] = ct >= 1
        progress.at[tid, "tier2_done"] = ct >= 2
        progress.at[tid, "tier3_done"] = ct >= 3
    clock = MissionClock(mission_start=0.0, mission_end=10.0)
    clock.current_time = 10.0
    state = MissionState(
        targets=targets,
        events=pd.DataFrame(),
        clock=clock,
        progress=progress,
        _backend=object(),  # skip DynamicBackend (needs full ephemeris cols)
    )
    state.n_initial_targets = n
    return state


def test_nested_vs_programme_and_total():
    # max_tier: 1,1,2,2,3,3
    # current:  1,0,2,1,3,2
    # nested T1/T2/T3 = 5 / 3 / 1
    # programme done: T1:1/2, T2:1/2, T3:1/2 → fully=3
    state = _minimal_state(
        max_tiers=[1, 1, 2, 2, 3, 3],
        current_tiers=[1, 0, 2, 1, 3, 2],
    )
    s = compute_stats(state)
    assert s.tier1_completed == 5
    assert s.tier2_completed == 3
    assert s.tier3_completed == 1
    assert s.prog_t1_completed == 1
    assert s.prog_t2_completed == 1
    assert s.prog_t3_completed == 1
    assert s.prog_t1_eligible == 2
    assert s.prog_t2_eligible == 2
    assert s.prog_t3_eligible == 2
    assert s.fully_completed == 3
    assert s.fully_completed == (
        s.prog_t1_completed + s.prog_t2_completed + s.prog_t3_completed
    )
