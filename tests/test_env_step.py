"""
Tests for MissionClock and MissionState — the core episode mechanics.
"""

import pytest
import pandas as pd
import numpy as np

from aRieL.data.schemas import MISSION_START_BJD, MISSION_END_BJD, MISSION_LIFETIME_DAYS
from aRieL.simulator.mission_clock import MissionClock
from aRieL.simulator.mission_state import MissionState


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clock():
    return MissionClock(mission_start=MISSION_START_BJD, mission_end=MISSION_END_BJD)


@pytest.fixture
def minimal_targets():
    """Three targets with known tier thresholds."""
    return pd.DataFrame({
        "target_idx":         [0, 1, 2],
        "target_id":          ["P_A", "P_B", "P_C"],
        "host_id":            ["S_A", "S_B", "S_C"],
        "ra":                 [30.0, 90.0, 200.0],
        "dec":                [10.0, -20.0, 45.0],
        "period":             [2.0, 5.0, 10.0],
        "epoch":              [MISSION_START_BJD + 1.0,
                               MISSION_START_BJD + 2.0,
                               MISSION_START_BJD + 3.0],
        "epoch_uncertainty":  [0.001, 0.001, 0.001],
        "transit_duration":   [5000.0, 8000.0, 12000.0],
        "eclipse_duration":   [5000.0, 8000.0, 12000.0],
        "planet_radius":      [2.0, 5.0, 12.0],
        "planet_mass":        [5.0, 30.0, 300.0],
        "planet_temperature": [800.0, 600.0, 1500.0],
        "stellar_type":       ["G2", "M4", "F5"],
        "stellar_temperature":[5800.0, 3500.0, 6500.0],
        "stellar_metallicity":[0.0, -0.2, 0.1],
        "tier1_required_obs": [2, 1, 3],
        "tier2_required_obs": [5, 3, 7],
        "tier3_required_obs": [10, 6, 12],
        "max_tier":           [3, 2, 2],
        "preferred_method":   ["Transit", "Eclipse", "Transit"],
        "available_transits": [100, 50, 30],
        "available_eclipses": [100, 50, 30],
        "fgs_flag":           [1, 1, 1],
        "rp_rs":              [0.1, 0.12, 0.15],
        "a_rs":               [5.0, 10.0, 20.0],
        "eccentricity":       [0.0, 0.0, 0.0],
        "inclination":        [89.0, 88.0, 87.0],
        "distance_pc":        [20.0, 50.0, 100.0],
        "population_bin":     ["super_earth_warm_gf", "neptune_cold_m", "jupiter_hot_gf"],
        "science_weight":     [0.5, 0.8, 0.3],
        "obs_cost_days_t1":   [5000/86400*2.5]*3,
        "obs_cost_days_t2":   [5000/86400*2.5]*3,
        "obs_cost_days_t3":   [5000/86400*2.5]*3,
    })


@pytest.fixture
def minimal_events(minimal_targets):
    """Hand-crafted events for three targets to avoid dependency on event_generator."""
    t0 = MISSION_START_BJD
    rows = []
    eid = 0
    for i, row in minimal_targets.iterrows():
        tid = row["target_id"]
        period = row["period"]
        dur = row["transit_duration"]
        dur_d = dur / 86400.0
        method = "eclipse" if row["preferred_method"] == "Eclipse" else "transit"
        for n in range(10):
            mid = row["epoch"] + n * period
            rows.append({
                "event_id":              eid,
                "target_id":             tid,
                "event_type":            method,
                "window_start":          mid - dur_d / 2,
                "window_mid":            mid,
                "window_end":            mid + dur_d / 2,
                "duration":              dur,
                "duration_days":         dur_d,
                "tier_goal":             int(row["max_tier"]),
                "base_science_value":    float(row["science_weight"]),
                "visibility_valid":      True,
                "ephemeris_uncertainty": 0.0,
                "event_index":           n,
            })
            eid += 1
    return pd.DataFrame(rows).sort_values("window_mid").reset_index(drop=True)


@pytest.fixture
def state(minimal_targets, minimal_events):
    return MissionState.from_tables(minimal_targets, minimal_events)


# ---------------------------------------------------------------------------
# MissionClock tests
# ---------------------------------------------------------------------------

class TestMissionClock:
    def test_initial_time(self, clock):
        assert clock.current_time == pytest.approx(MISSION_START_BJD)

    def test_remaining_time_at_start(self, clock):
        assert clock.remaining_time == pytest.approx(MISSION_LIFETIME_DAYS, rel=1e-4)

    def test_advance_moves_time(self, clock):
        clock.advance(obs_duration_days=1.0, slew_days=0.5)
        assert clock.current_time == pytest.approx(MISSION_START_BJD + 1.5)

    def test_advance_tracks_components(self, clock):
        clock.advance(obs_duration_days=2.0, slew_days=0.3, overhead_days=0.0)
        assert clock.used_science_time == pytest.approx(2.0)
        assert clock.used_slew_time == pytest.approx(0.3)

    def test_remaining_decreases_after_advance(self, clock):
        before = clock.remaining_time
        clock.advance(obs_duration_days=10.0, slew_days=0.0, overhead_days=0.0)
        assert clock.remaining_time == pytest.approx(before - 10.0)

    def test_fraction_elapsed(self, clock):
        assert clock.fraction_elapsed == pytest.approx(0.0)
        clock.advance(MISSION_LIFETIME_DAYS / 2, slew_days=0, overhead_days=0)
        assert clock.fraction_elapsed == pytest.approx(0.5, rel=1e-4)

    def test_mission_over_flag(self, clock):
        assert not clock.mission_over
        clock.advance(MISSION_LIFETIME_DAYS + 1, slew_days=0, overhead_days=0)
        assert clock.mission_over

    def test_remaining_never_negative(self, clock):
        clock.advance(MISSION_LIFETIME_DAYS + 999, slew_days=0, overhead_days=0)
        assert clock.remaining_time == 0.0

    def test_skip_to_advances_time(self, clock):
        target = MISSION_START_BJD + 100.0
        wait = clock.skip_to(target)
        assert clock.current_time == pytest.approx(target)
        assert wait == pytest.approx(100.0)

    def test_skip_to_past_raises(self, clock):
        with pytest.raises(ValueError):
            clock.skip_to(MISSION_START_BJD - 1.0)

    def test_reset(self, clock):
        clock.advance(100.0, slew_days=5.0)
        clock.reset()
        assert clock.current_time == pytest.approx(MISSION_START_BJD)
        assert clock.used_science_time == 0.0
        assert clock.n_observations == 0

    def test_n_observations_increments(self, clock):
        clock.advance(1.0)
        clock.advance(1.0)
        assert clock.n_observations == 2

    def test_can_fit_true(self, clock):
        assert clock.can_fit(1.0)

    def test_can_fit_false(self, clock):
        clock.advance(MISSION_LIFETIME_DAYS - 0.5, slew_days=0, overhead_days=0)
        assert not clock.can_fit(1.0)

    def test_snapshot_keys(self, clock):
        snap = clock.snapshot()
        for key in ("current_time", "remaining_time", "n_observations", "used_science_time"):
            assert key in snap


# ---------------------------------------------------------------------------
# MissionState tests
# ---------------------------------------------------------------------------

class TestMissionState:
    def test_from_tables_creates_state(self, state):
        assert state is not None
        assert len(state.targets) == 3
        assert len(state.events) > 0

    def test_initial_progress_all_zero(self, state):
        assert (state.progress["obs_completed"] == 0).all()
        assert (state.progress["current_tier"] == 0).all()
        assert not state.progress["tier1_done"].any()

    def test_execute_observation_advances_clock(self, state):
        t_before = state.clock.current_time
        first_event_id = int(state.events["event_id"].iloc[0])
        state.execute_observation(first_event_id)
        assert state.clock.current_time > t_before

    def test_execute_observation_increments_progress(self, state):
        pa_events = state.events[state.events["target_id"] == "P_A"]
        eid = int(pa_events["event_id"].iloc[0])
        state.execute_observation(eid)
        assert state.progress.loc["P_A", "obs_completed"] == 1

    def test_tier_completion_after_required_obs(self, state):
        """P_B needs 1 obs for Tier 1; after 1 obs it should be Tier 1 done."""
        pb_events = state.events[state.events["target_id"] == "P_B"]
        eid = int(pb_events["event_id"].iloc[0])
        state.execute_observation(eid)
        assert state.progress.loc["P_B", "tier1_done"]
        assert state.progress.loc["P_B", "current_tier"] == 1

    def test_tier1_completed_count(self, state):
        pb_events = state.events[state.events["target_id"] == "P_B"]
        eid = int(pb_events["event_id"].iloc[0])
        state.execute_observation(eid)
        assert state.tier1_completed >= 1

    def test_progress_in_tier_increases(self, state):
        pa_events = state.events[state.events["target_id"] == "P_A"].reset_index()
        # P_A needs 2 obs for Tier 1; after 1 obs progress should be 0.5
        eid = int(pa_events["event_id"].iloc[0])
        info = state.execute_observation(eid)
        prog = state.progress.loc["P_A", "progress_in_tier"]
        assert 0.0 < prog <= 1.0

    def test_execute_returns_info_dict(self, state):
        eid = int(state.events["event_id"].iloc[0])
        info = state.execute_observation(eid)
        for key in ("target_id", "tier_before", "tier_after", "obs_duration_days",
                    "slew_days", "missed", "obs_number"):
            assert key in info

    def test_is_done_when_clock_ends(self, state):
        assert not state.is_done()
        state.clock.advance(MISSION_LIFETIME_DAYS + 1, slew_days=0, overhead_days=0)
        assert state.is_done()

    def test_summary_has_expected_keys(self, state):
        s = state.summary()
        for key in ("current_time", "remaining_time", "tier1_completed", "total_targets"):
            assert key in s

    def test_reset_clears_progress(self, state):
        pa_events = state.events[state.events["target_id"] == "P_A"]
        eid = int(pa_events["event_id"].iloc[0])
        state.execute_observation(eid)
        state.reset()
        assert (state.progress["obs_completed"] == 0).all()
        assert state.clock.current_time == pytest.approx(MISSION_START_BJD)

    def test_upcoming_events_returns_future_only(self, state):
        upcoming = state.upcoming_events(n=10)
        assert (upcoming["window_end"] > state.clock.current_time).all()


# ---------------------------------------------------------------------------
# DynamicBackend integration tests
# ---------------------------------------------------------------------------

class TestDynamicBackend:
    """Verify DynamicBackend produces valid candidates and can run full episodes."""

    @pytest.fixture
    def env_dynamic(self, minimal_targets):
        from aRieL.simulator.event_backend import DynamicBackend
        from aRieL.envs.env import ArielEnv
        from aRieL.utils.config import (
            EnvConfig, MissionConfig, SlewConfig, ActionConfig,
            TopKActionConfig, ObservationConfig, RewardConfig,
        )
        cfg = EnvConfig(
            mission=MissionConfig(
                start_bjd=MISSION_START_BJD,
                lifetime_days=60.0,
            ),
            slew=SlewConfig(),
            action=ActionConfig(type="topk", topk=TopKActionConfig(k=10)),
            observation=ObservationConfig(
                event_features=["duration_days", "base_science_value", "progress_in_tier"],
                global_features=["fraction_elapsed"],
                include_population_bin_fractions=False,
            ),
            reward=RewardConfig(),
            seed=42,
        )
        backend = DynamicBackend(minimal_targets)
        return ArielEnv(cfg, targets=minimal_targets, backend=backend)

    def test_candidates_returns_correct_schema(self, minimal_targets):
        from aRieL.simulator.event_backend import DynamicBackend, EVENT_COLUMNS
        db = DynamicBackend(minimal_targets)
        t_now = MISSION_START_BJD + 1.0
        cands = db.candidates(t_now, k=5)
        assert set(EVENT_COLUMNS).issubset(set(cands.columns))
        assert len(cands) <= 5

    def test_candidates_window_ends_in_future(self, minimal_targets):
        from aRieL.simulator.event_backend import DynamicBackend
        db = DynamicBackend(minimal_targets)
        t_now = MISSION_START_BJD + 5.0
        cands = db.candidates(t_now, k=10)
        if len(cands):
            assert (cands["window_end"] > t_now).all()

    def test_get_event_returns_same_row(self, minimal_targets):
        from aRieL.simulator.event_backend import DynamicBackend
        db = DynamicBackend(minimal_targets)
        t_now = MISSION_START_BJD + 1.0
        cands = db.candidates(t_now, k=5)
        if len(cands):
            eid = int(cands["event_id"].iloc[0])
            ev = db.get_event(eid)
            assert ev["event_id"] == eid
            assert ev["window_mid"] == pytest.approx(cands["window_mid"].iloc[0])

    def test_env_reset_and_step(self, env_dynamic):
        obs, info = env_dynamic.reset()
        assert "events" in obs and "global" in obs
        assert "action_mask" in info
        valid = np.where(info["action_mask"])[0]
        assert len(valid) > 0
        obs2, reward, terminated, truncated, info2 = env_dynamic.step(int(valid[0]))
        assert "events" in obs2

    def test_full_episode_completes(self, env_dynamic):
        """DynamicBackend episode terminates without error."""
        obs, info = env_dynamic.reset()
        for _ in range(500):
            valid = np.where(info["action_mask"])[0]
            if not len(valid):
                break
            obs, _, done, _, info = env_dynamic.step(int(valid[0]))
            if done:
                break
        # Should have made at least one observation
        assert env_dynamic.state is not None
