"""Annual Tier-3 progress reset (yearly revisit eval scenario)."""

from __future__ import annotations

import dataclasses

import pytest

from aRieL.data.observation_requirements import compute_progress
from aRieL.simulator.mission_state import MissionState
from aRieL.utils.config import AnnualT3RevisitConfig, load_env_config


def test_annual_t3_config_loads_from_yaml():
    cfg = load_env_config("annual_t3")
    assert cfg.annual_t3_revisit.enabled is True
    assert cfg.annual_t3_revisit.period_days == pytest.approx(365.25)
    assert cfg.annual_t3_revisit.min_max_tier == 3


def test_reset_progress_zeros_completed_target(tmp_path):
    """MissionState.reset_progress puts a finished planet back at T0."""
    from aRieL.data.demo import make_demo_targets
    from aRieL.simulator.event_backend import DynamicBackend

    targets = make_demo_targets(12)
    # Force at least one max_tier=3 row if the head slice has none.
    if (targets["max_tier"].astype(int) >= 3).sum() == 0:
        targets.loc[targets.index[0], "max_tier"] = 3
        targets.loc[targets.index[0], "tier1_required_obs"] = 1
        targets.loc[targets.index[0], "tier2_required_obs"] = 1
        targets.loc[targets.index[0], "tier3_required_obs"] = 1

    backend = DynamicBackend(targets)
    state = MissionState.from_backend(
        targets,
        backend=backend,
        mission_start=2462867.5,
        mission_end=2462867.5 + 100.0,
    )
    t3 = targets[targets["max_tier"].astype(int) >= 3].iloc[0]
    tid = str(t3["target_id"])
    # Simulate full completion.
    done = compute_progress(float(t3["tier3_required_obs"]), t3)
    state._progress_dict[tid].update(done)
    for k, v in done.items():
        state.progress.at[tid, k] = v
    assert int(state.progress.loc[tid, "current_tier"]) >= 3

    reset = state.reset_progress([tid])
    assert reset == [tid]
    assert float(state.progress.loc[tid, "obs_completed"]) == 0.0
    assert int(state.progress.loc[tid, "current_tier"]) == 0
    assert bool(state.progress.loc[tid, "tier1_done"]) is False


def test_env_fires_annual_reset_at_year_boundary():
    """Crossing day 365.25 resets max_tier≥3 progress and reopens them."""
    import pandas as pd

    from aRieL.data.demo import make_demo_targets
    from aRieL.envs.env import ArielEnv
    from aRieL.simulator.event_backend import DynamicBackend
    from aRieL.utils.config import (
        ActionConfig,
        AnnualT3RevisitConfig,
        TopKActionConfig,
        default_env_config,
    )

    targets = make_demo_targets(12)
    t3_ids = set(
        targets.loc[targets["max_tier"].astype(int) >= 3, "target_id"].astype(str)
    )
    assert len(t3_ids) > 0

    cfg = default_env_config()
    cfg = dataclasses.replace(
        cfg,
        mission=dataclasses.replace(cfg.mission, lifetime_days=400.0),
        action=ActionConfig(
            type="topk",
            topk=TopKActionConfig(k=32, exclude_started_blocks=True),
        ),
        annual_t3_revisit=AnnualT3RevisitConfig(
            enabled=True, period_days=365.25, min_max_tier=3,
        ),
    )
    env = ArielEnv(config=cfg, targets=targets, backend=DynamicBackend(targets))
    obs, info = env.reset(seed=0)

    # Manually mark all T3 planets as fully complete before the year boundary.
    for tid in t3_ids:
        row = env.state._target_lookup[tid]
        done = compute_progress(float(row["tier3_required_obs"]), row)
        env.state._progress_dict[tid].update(done)
        for k, v in done.items():
            env.state.progress.at[tid, k] = v
    assert env.state.tier3_completed == len(t3_ids)

    # Fast-forward the clock just past the first year boundary and apply.
    env.state.clock.current_time = cfg.mission.start_bjd + 365.25 + 0.01
    reset = env._apply_annual_t3_resets()
    assert len(reset) == len(t3_ids)
    assert env._n_t3_revisits_fired == 1
    assert env.state.tier3_completed == 0
    # Non-T3 planets were not touched — leave their progress alone (none set).
    assert env.state.tier1_completed == 0
    assert len(env._year_snapshots) == 1
    assert env._year_snapshots[0]["snapshot_kind"] == "year_end"
    assert env._year_snapshots[0]["year_index"] == 1
    assert env._year_snapshots[0]["tier3_completed"] == len(t3_ids)
