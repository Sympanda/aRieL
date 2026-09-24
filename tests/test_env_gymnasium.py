"""
Tests for ArielEnv — Gymnasium compliance and core env mechanics.

Uses a small synthetic dataset (3 targets, hand-built events) to keep
tests fast and deterministic without touching the real CSV.
"""

import numpy as np
import pandas as pd
import pytest
import gymnasium as gym

from aRieL.data.schemas import MISSION_START_BJD
from aRieL.envs.env import ArielEnv
from aRieL.utils.config import (
    ActionConfig, EnvConfig, MissionConfig, ObservationConfig,
    RewardConfig, SlewConfig, TargetActionConfig, TopKActionConfig,
    default_env_config,
)


# ---------------------------------------------------------------------------
# Shared fixture: tiny synthetic dataset
# ---------------------------------------------------------------------------

T0 = MISSION_START_BJD
MISSION_DAYS = 50.0   # short episode for fast tests


def _make_targets():
    return pd.DataFrame({
        "target_idx":         [0, 1, 2],
        "target_id":          ["P_A", "P_B", "P_C"],
        "host_id":            ["S_A", "S_B", "S_C"],
        "ra":                 [30.0, 90.0, 200.0],
        "dec":                [10.0, -20.0, 45.0],
        "period":             [2.0, 3.0, 5.0],
        "epoch":              [T0 + 1.0, T0 + 1.5, T0 + 2.0],
        "epoch_uncertainty":  [0.001, 0.001, 0.001],
        "transit_duration":   [5000.0, 8000.0, 6000.0],
        "eclipse_duration":   [5000.0, 8000.0, 6000.0],
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
        "preferred_method":   ["Transit", "Transit", "Transit"],
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
        "obs_cost_days_t1":   [0.15, 0.23, 0.18],
        "obs_cost_days_t2":   [0.15, 0.23, 0.18],
        "obs_cost_days_t3":   [0.15, 0.23, 0.18],
    })


def _make_events(targets):
    rows = []
    eid = 0
    for _, row in targets.iterrows():
        for n in range(20):
            mid = float(row["epoch"]) + n * float(row["period"])
            if mid > T0 + MISSION_DAYS:
                break
            dur = float(row["transit_duration"]) / 86400
            rows.append({
                "event_id": eid, "target_id": row["target_id"],
                "event_type": "transit",
                "window_start": mid - dur/2, "window_mid": mid, "window_end": mid + dur/2,
                "duration": float(row["transit_duration"]), "duration_days": dur,
                "tier_goal": int(row["max_tier"]),
                "base_science_value": float(row["science_weight"]),
                "visibility_valid": True, "ephemeris_uncertainty": 0.0, "event_index": n,
            })
            eid += 1
    return pd.DataFrame(rows).sort_values("window_mid").reset_index(drop=True)


@pytest.fixture
def targets():
    return _make_targets()


@pytest.fixture
def events(targets):
    return _make_events(targets)


def _make_cfg(action_type="topk", k=10) -> EnvConfig:
    return EnvConfig(
        mission=MissionConfig(
            start_bjd=T0,
            lifetime_days=MISSION_DAYS,
            cost_factor=2.5,
        ),
        slew=SlewConfig(rate_deg_per_min=1.0, min_slew_seconds=120.0, max_slew_seconds=7200.0),
        action=ActionConfig(
            type=action_type,
            topk=TopKActionConfig(k=k, sort_by="window_mid"),
            target=TargetActionConfig(include_completed=False),
        ),
        observation=ObservationConfig(
            event_features=["slew_time_days", "wait_time_days", "duration_days",
                            "progress_in_tier", "base_science_value", "is_valid"],
            global_features=["fraction_elapsed", "tier1_fraction", "tier2_fraction"],
            include_population_bin_fractions=False,
            normalise=True,
        ),
        reward=RewardConfig(),
        seed=0,
    )


@pytest.fixture
def topk_env(targets, events):
    return ArielEnv(config=_make_cfg("topk", k=10), targets=targets, events=events)


@pytest.fixture
def target_env(targets, events):
    return ArielEnv(config=_make_cfg("target"), targets=targets, events=events)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_default_config_loads(self):
        cfg = default_env_config()
        assert cfg.action.type == "topk"
        assert cfg.slew.rate_deg_per_min == 1.0

    def test_yaml_config_loads(self):
        from aRieL.utils.config import load_env_config
        cfg = load_env_config("simple")
        assert cfg.action.type == "topk"
        assert cfg.action.topk.k == 50

    def test_config_is_frozen(self):
        cfg = default_env_config()
        with pytest.raises((TypeError, AttributeError)):
            cfg.seed = 99   # frozen dataclass should raise

    def test_to_dict(self):
        from aRieL.utils.config import env_config_to_dict
        d = env_config_to_dict(default_env_config())
        assert "mission" in d
        assert "slew" in d
        assert "action" in d


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

class TestEnvCreation:
    def test_topk_env_creates(self, topk_env):
        assert topk_env is not None

    def test_target_env_creates(self, target_env):
        assert target_env is not None

    def test_topk_action_space_size(self, topk_env):
        assert topk_env.action_space.n == 10

    def test_target_action_space_size(self, target_env, targets):
        assert target_env.action_space.n == len(targets)

    def test_observation_space_is_dict(self, topk_env):
        assert isinstance(topk_env.observation_space, gym.spaces.Dict)

    def test_observation_space_has_events_and_global(self, topk_env):
        assert "events" in topk_env.observation_space.spaces
        assert "global" in topk_env.observation_space.spaces

    def test_events_obs_shape(self, topk_env):
        k = topk_env.cfg.action.topk.k
        n_ef = len(topk_env.cfg.observation.event_features)
        assert topk_env.observation_space["events"].shape == (k, n_ef)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, topk_env):
        obs, info = topk_env.reset()
        assert "events" in obs
        assert "global" in obs

    def test_obs_shapes_match_space(self, topk_env):
        obs, _ = topk_env.reset()
        assert obs["events"].shape == topk_env.observation_space["events"].shape
        assert obs["global"].shape == topk_env.observation_space["global"].shape

    def test_obs_dtypes_float32(self, topk_env):
        obs, _ = topk_env.reset()
        assert obs["events"].dtype == np.float32
        assert obs["global"].dtype == np.float32

    def test_info_has_action_mask(self, topk_env):
        _, info = topk_env.reset()
        assert "action_mask" in info

    def test_action_mask_shape(self, topk_env):
        _, info = topk_env.reset()
        assert info["action_mask"].shape == (topk_env.n_actions,)

    def test_action_mask_is_bool(self, topk_env):
        _, info = topk_env.reset()
        assert info["action_mask"].dtype == bool

    def test_at_least_one_valid_action_after_reset(self, topk_env):
        _, info = topk_env.reset()
        assert info["action_mask"].any()

    def test_reset_clears_state(self, topk_env):
        topk_env.reset()
        # Take some steps
        for _ in range(3):
            valid = np.where(topk_env.action_mask)[0]
            if len(valid) == 0:
                break
            topk_env.step(int(valid[0]))
        # Reset should put clock back to start
        topk_env.reset()
        assert topk_env.state.clock.current_time == pytest.approx(T0)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_5_tuple(self, topk_env):
        topk_env.reset()
        valid = np.where(topk_env.action_mask)[0]
        obs, reward, terminated, truncated, info = topk_env.step(int(valid[0]))
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_valid_step_advances_clock(self, topk_env):
        topk_env.reset()
        t_before = topk_env.state.clock.current_time
        valid = np.where(topk_env.action_mask)[0]
        topk_env.step(int(valid[0]))
        assert topk_env.state.clock.current_time > t_before

    def test_invalid_action_returns_negative_reward(self, topk_env):
        topk_env.reset()
        # Force an invalid action by picking a padded/masked slot
        mask = topk_env.action_mask
        invalid_indices = np.where(~mask)[0]
        if len(invalid_indices) == 0:
            pytest.skip("No invalid actions available in this episode")
        _, reward, _, _, info = topk_env.step(int(invalid_indices[0]))
        assert reward < 0
        assert info["invalid_action"]

    def test_step_count_increments(self, topk_env):
        topk_env.reset()
        valid = np.where(topk_env.action_mask)[0]
        _, _, _, _, info = topk_env.step(int(valid[0]))
        assert info["step_count"] == 1

    def test_step_info_has_mission_summary(self, topk_env):
        topk_env.reset()
        valid = np.where(topk_env.action_mask)[0]
        _, _, _, _, info = topk_env.step(int(valid[0]))
        assert "mission_summary" in info

    def test_episode_terminates(self, topk_env):
        """Run a full episode greedily and confirm it terminates."""
        topk_env.reset()
        terminated = False
        max_steps = 2000
        for _ in range(max_steps):
            if terminated:
                break
            valid = np.where(topk_env.action_mask)[0]
            if len(valid) == 0:
                break
            _, _, terminated, _, _ = topk_env.step(int(valid[0]))
        assert terminated or topk_env.state.is_done()

    def test_obs_in_observation_space_after_step(self, topk_env):
        topk_env.reset()
        valid = np.where(topk_env.action_mask)[0]
        obs, _, _, _, _ = topk_env.step(int(valid[0]))
        assert topk_env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Target action space
# ---------------------------------------------------------------------------

class TestTargetActionSpace:
    def test_target_reset_returns_n_actions(self, target_env, targets):
        _, info = target_env.reset()
        assert info["action_mask"].shape == (len(targets),)

    def test_target_step_works(self, target_env):
        target_env.reset()
        valid = np.where(target_env.action_mask)[0]
        if len(valid) == 0:
            pytest.skip("No valid target actions")
        obs, reward, terminated, _, info = target_env.step(int(valid[0]))
        assert "events" in obs
