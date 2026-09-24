"""
Tests for baseline agents and the evaluation framework.
"""

import numpy as np
import pandas as pd
import pytest

from aRieL.baselines import (
    ALL_BASELINES,
    EarliestDeadline,
    GreedyBalanced,
    GreedyValue,
    RandomValid,
)
from aRieL.data.schemas import MISSION_START_BJD
from aRieL.envs.env import ArielEnv
from aRieL.evaluation.compare_runs import compare_baselines, run_episode, summary_table
from aRieL.evaluation.metrics import EpisodeStats, compute_stats
from aRieL.evaluation.population_coverage import (
    coverage_gini,
    coverage_matrix,
    coverage_table,
    gini_coefficient,
)
from aRieL.utils.config import (
    ActionConfig, EnvConfig, MissionConfig, ObservationConfig,
    RewardConfig, SlewConfig, TopKActionConfig,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T0 = MISSION_START_BJD
MISSION_DAYS = 60.0


def _make_targets():
    return pd.DataFrame({
        "target_idx":         list(range(6)),
        "target_id":          [f"P_{i}" for i in range(6)],
        "host_id":            [f"S_{i}" for i in range(6)],
        "ra":                 [10.0, 50.0, 100.0, 180.0, 250.0, 320.0],
        "dec":                [10.0, -20.0, 40.0, -5.0, 30.0, -40.0],
        "period":             [1.5, 2.0, 3.0, 5.0, 7.0, 10.0],
        "epoch":              [T0 + 1.0] * 6,
        "epoch_uncertainty":  [0.001] * 6,
        "transit_duration":   [4000.0, 6000.0, 8000.0, 5000.0, 7000.0, 9000.0],
        "eclipse_duration":   [4000.0, 6000.0, 8000.0, 5000.0, 7000.0, 9000.0],
        "planet_radius":      [1.5, 3.0, 5.0, 10.0, 2.0, 4.0],
        "planet_mass":        [4.0, 15.0, 30.0, 200.0, 8.0, 20.0],
        "planet_temperature": [700.0, 1000.0, 500.0, 1800.0, 400.0, 1200.0],
        "stellar_type":       ["G2", "M4", "K5", "F5", "M2", "G8"],
        "stellar_temperature":[5800.0, 3500.0, 4800.0, 6500.0, 3600.0, 5600.0],
        "stellar_metallicity":[0.0, -0.2, 0.1, 0.3, -0.1, 0.0],
        "tier1_required_obs": [1, 2, 1, 3, 2, 1],
        "tier2_required_obs": [3, 5, 4, 8, 5, 3],
        "tier3_required_obs": [6, 10, 8, 15, 10, 6],
        "max_tier":           [3, 2, 3, 2, 2, 3],
        "preferred_method":   ["Transit"] * 6,
        "available_transits": [100] * 6,
        "available_eclipses": [100] * 6,
        "fgs_flag":           [1] * 6,
        "rp_rs":              [0.1, 0.12, 0.13, 0.15, 0.11, 0.14],
        "a_rs":               [5.0, 8.0, 12.0, 20.0, 6.0, 10.0],
        "eccentricity":       [0.0] * 6,
        "inclination":        [89.0] * 6,
        "distance_pc":        [20.0, 50.0, 80.0, 100.0, 30.0, 60.0],
        "population_bin":     [
            "super_earth_warm_gf", "mini_neptune_hot_m",
            "neptune_warm_k", "jupiter_very_hot_gf",
            "super_earth_cold_m", "mini_neptune_hot_gf",
        ],
        "science_weight":     [0.5, 0.9, 0.6, 0.3, 0.8, 0.7],
        "obs_cost_days_t1":   [0.12, 0.17, 0.23, 0.14, 0.20, 0.26],
        "obs_cost_days_t2":   [0.12, 0.17, 0.23, 0.14, 0.20, 0.26],
        "obs_cost_days_t3":   [0.12, 0.17, 0.23, 0.14, 0.20, 0.26],
    })


def _make_events(targets):
    rows = []
    eid = 0
    for _, row in targets.iterrows():
        for n in range(40):
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


@pytest.fixture(scope="module")
def shared_targets():
    return _make_targets()


@pytest.fixture(scope="module")
def shared_events(shared_targets):
    return _make_events(shared_targets)


def _make_cfg(k=20) -> EnvConfig:
    return EnvConfig(
        mission=MissionConfig(start_bjd=T0, lifetime_days=MISSION_DAYS),
        slew=SlewConfig(),
        action=ActionConfig(type="topk", topk=TopKActionConfig(k=k)),
        observation=ObservationConfig(
            event_features=[
                "slew_time_days", "wait_time_days", "duration_days",
                "total_time_cost_days", "progress_in_tier",
                "obs_remaining_next_tier_norm", "base_science_value",
                "science_weight", "tier_goal_norm", "is_valid",
            ],
            global_features=["fraction_elapsed", "tier1_fraction", "tier2_fraction"],
            include_population_bin_fractions=False,
            normalise=True,
        ),
        reward=RewardConfig(),
        seed=0,
    )


@pytest.fixture(scope="module")
def env(shared_targets, shared_events):
    return ArielEnv(config=_make_cfg(), targets=shared_targets, events=shared_events)


# ---------------------------------------------------------------------------
# Baseline agent tests
# ---------------------------------------------------------------------------

class TestRandomValid:
    def test_always_returns_valid(self, env):
        obs, info = env.reset()
        agent = RandomValid(seed=42)
        for _ in range(20):
            action = agent.act(obs, info)
            assert 0 <= action < env.n_actions
            assert info["action_mask"][action], "RandomValid returned an invalid action"
            obs, _, terminated, _, info = env.step(action)
            if terminated:
                break

    def test_different_seeds_give_different_episodes(self, env):
        actions_a, actions_b = [], []
        for seed, store in [(0, actions_a), (99, actions_b)]:
            obs, info = env.reset(seed=seed)
            agent = RandomValid(seed=seed)
            for _ in range(10):
                a = agent.act(obs, info)
                store.append(a)
                obs, _, done, _, info = env.step(a)
                if done:
                    break
        # With different seeds at least some actions should differ
        assert actions_a != actions_b or len(actions_a) == 0


class TestGreedyValue:
    def test_always_returns_valid(self, env):
        cfg = _make_cfg()
        agent = GreedyValue(obs_cfg=cfg.observation, seed=0)
        obs, info = env.reset()
        for _ in range(20):
            action = agent.act(obs, info)
            assert info["action_mask"][action]
            obs, _, done, _, info = env.step(action)
            if done:
                break

    def test_picks_high_science_value(self, env):
        """GreedyValue should consistently pick the highest science value event."""
        cfg = _make_cfg()
        agent = GreedyValue(obs_cfg=cfg.observation, seed=0)
        obs, info = env.reset()
        action = agent.act(obs, info)
        mask = info["action_mask"]
        events_arr = obs["events"]
        sv_idx = cfg.observation.event_features.index("base_science_value")
        # The chosen action should have the max valid science value
        valid_svs = events_arr[mask, sv_idx]
        chosen_sv = events_arr[action, sv_idx]
        assert chosen_sv == pytest.approx(valid_svs.max(), rel=1e-4)


class TestGreedyBalanced:
    def test_always_returns_valid(self, env):
        cfg = _make_cfg()
        agent = GreedyBalanced(obs_cfg=cfg.observation, alpha=1.0, seed=0)
        obs, info = env.reset()
        for _ in range(20):
            action = agent.act(obs, info)
            assert info["action_mask"][action]
            obs, _, done, _, info = env.step(action)
            if done:
                break

    def test_alpha_zero_matches_science_weight(self, env):
        """With alpha=0 GreedyBalanced should score by science_weight alone."""
        cfg = _make_cfg()
        agent = GreedyBalanced(obs_cfg=cfg.observation, alpha=0.0, seed=0)
        obs, info = env.reset()
        action = agent.act(obs, info)
        mask = info["action_mask"]
        events_arr = obs["events"]
        sw_idx = cfg.observation.event_features.index("science_weight")
        valid_sws = events_arr[mask, sw_idx]
        chosen_sw = events_arr[action, sw_idx]
        assert chosen_sw == pytest.approx(valid_sws.max(), rel=1e-4)


class TestEarliestDeadline:
    def test_always_returns_valid(self, env):
        cfg = _make_cfg()
        agent = EarliestDeadline(obs_cfg=cfg.observation, seed=0)
        obs, info = env.reset()
        for _ in range(20):
            action = agent.act(obs, info)
            assert info["action_mask"][action]
            obs, _, done, _, info = env.step(action)
            if done:
                break

    def test_picks_soonest_event(self, env):
        """EarliestDeadline should pick the valid event with smallest wait_time."""
        cfg = _make_cfg()
        agent = EarliestDeadline(obs_cfg=cfg.observation, seed=0)
        obs, info = env.reset()
        action = agent.act(obs, info)
        mask = info["action_mask"]
        events_arr = obs["events"]
        wait_idx = cfg.observation.event_features.index("wait_time_days")
        valid_waits = events_arr[mask, wait_idx]
        chosen_wait = events_arr[action, wait_idx]
        assert chosen_wait == pytest.approx(valid_waits.min(), rel=1e-4)


class TestAllBaselinesRegistry:
    def test_all_baselines_in_registry(self):
        for name, cls in ALL_BASELINES.items():
            assert hasattr(cls, "act"), f"{name} missing .act()"

    def test_all_baselines_return_valid_action(self, env):
        cfg = _make_cfg()
        obs, info = env.reset()
        for name, cls in ALL_BASELINES.items():
            if name == "hill_climbing":
                continue  # needs a live env + optimisation loop
            try:
                agent = cls(obs_cfg=cfg.observation, seed=0)
            except TypeError:
                agent = cls(seed=0)
            action = agent.act(obs, info)
            assert 0 <= action < env.n_actions, f"{name} returned out-of-range action"
            assert info["action_mask"][action], f"{name} returned an invalid action"


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestEpisodeStats:
    @pytest.fixture
    def finished_state(self, env):
        agent = GreedyValue(obs_cfg=_make_cfg().observation, seed=0)
        run_episode(env, agent, seed=0)
        return env.state

    def test_compute_stats_returns_dataclass(self, finished_state):
        stats = compute_stats(finished_state)
        assert isinstance(stats, EpisodeStats)

    def test_tier_rates_in_range(self, finished_state):
        stats = compute_stats(finished_state)
        assert 0.0 <= stats.tier1_rate <= 1.0
        assert 0.0 <= stats.tier2_rate <= 1.0
        assert 0.0 <= stats.tier3_rate <= 1.0

    def test_tier_counts_ordered(self, finished_state):
        stats = compute_stats(finished_state)
        # More T1 than T2, more T2 than T3
        assert stats.tier1_completed >= stats.tier2_completed >= stats.tier3_completed

    def test_miss_rate_in_range(self, finished_state):
        stats = compute_stats(finished_state)
        assert 0.0 <= stats.miss_rate <= 1.0

    def test_science_efficiency_in_range(self, finished_state):
        stats = compute_stats(finished_state)
        assert 0.0 <= stats.science_efficiency <= 1.0

    def test_bin_coverage_in_range(self, finished_state):
        stats = compute_stats(finished_state)
        assert 0.0 <= stats.bin_coverage <= 1.0

    def test_summary_str_is_string(self, finished_state):
        stats = compute_stats(finished_state)
        s = stats.summary_str()
        assert isinstance(s, str) and len(s) > 0

    def test_to_dict_has_no_bin_counts(self, finished_state):
        stats = compute_stats(finished_state)
        d = stats.to_dict()
        assert "bin_counts" not in d
        assert "tier1_completed" in d


# ---------------------------------------------------------------------------
# Population coverage tests
# ---------------------------------------------------------------------------

class TestPopulationCoverage:
    @pytest.fixture
    def state_after_run(self, env):
        agent = GreedyBalanced(obs_cfg=_make_cfg().observation, seed=0)
        run_episode(env, agent, seed=0)
        return env.state

    def test_coverage_table_returns_dataframe(self, state_after_run):
        df = coverage_table(state_after_run)
        assert isinstance(df, pd.DataFrame)
        assert "population_bin" in df.columns
        assert "tier1_rate" in df.columns

    def test_coverage_table_has_all_bins(self, state_after_run, shared_targets):
        df = coverage_table(state_after_run)
        expected_bins = set(shared_targets["population_bin"].unique())
        actual_bins   = set(df["population_bin"].unique())
        assert expected_bins == actual_bins

    def test_coverage_matrix_is_dataframe(self, state_after_run):
        m = coverage_matrix(state_after_run, tier=1)
        assert isinstance(m, pd.DataFrame)

    def test_coverage_matrix_values_in_range(self, state_after_run):
        m = coverage_matrix(state_after_run, tier=1)
        assert (m.values >= 0).all()
        assert (m.values <= 1.0 + 1e-6).all()

    def test_gini_in_range(self, state_after_run):
        g = coverage_gini(state_after_run, tier=1)
        assert 0.0 <= g <= 1.0

    def test_gini_coefficient_zero_for_uniform(self):
        import numpy as np
        assert gini_coefficient(np.ones(10)) == pytest.approx(0.0, abs=1e-6)

    def test_gini_coefficient_one_for_monopoly(self):
        import numpy as np
        v = np.zeros(10)
        v[0] = 1.0
        g = gini_coefficient(v)
        assert g > 0.8   # near 1 but not exactly due to discrete definition


# ---------------------------------------------------------------------------
# Full comparison run tests
# ---------------------------------------------------------------------------

class TestCompareBaselines:
    def test_run_episode_returns_stats(self, env):
        agent = RandomValid(seed=0)
        stats = run_episode(env, agent, seed=0)
        assert isinstance(stats, EpisodeStats)
        assert stats.n_observations > 0

    def test_compare_baselines_returns_dataframe(self, env):
        cfg = _make_cfg()
        agents = {
            "random": RandomValid(seed=0),
            "greedy": GreedyValue(obs_cfg=cfg.observation, seed=0),
        }
        df = compare_baselines(env, agents, n_episodes=1)
        assert isinstance(df, pd.DataFrame)
        assert "agent" in df.columns
        assert set(df["agent"].unique()) == {"random", "greedy"}

    def test_compare_baselines_episode_count(self, env):
        cfg = _make_cfg()
        agents = {"random": RandomValid(seed=0)}
        df = compare_baselines(env, agents, n_episodes=3)
        assert len(df) == 3

    def test_summary_table_aggregates(self, env):
        cfg = _make_cfg()
        agents = {
            "random": RandomValid(seed=0),
            "greedy": GreedyValue(obs_cfg=cfg.observation, seed=0),
        }
        df = compare_baselines(env, agents, n_episodes=2)
        tbl = summary_table(df)
        assert "agent" in tbl.columns
        assert len(tbl) == 2   # one row per agent

    def test_comparison_runs_multiple_agents_and_episodes(self, env):
        """compare_baselines produces one row per (agent, episode) pair."""
        cfg = _make_cfg()
        n_ep = 3
        agents = {
            "random":  RandomValid(seed=0),
            "balanced": GreedyBalanced(obs_cfg=cfg.observation, alpha=1.0, seed=0),
        }
        df = compare_baselines(env, agents, n_episodes=n_ep, seed_start=0)
        assert set(df["agent"].unique()) == {"random", "balanced"}
        assert len(df) == n_ep * len(agents)   # 3 × 2 = 6 rows
        assert df["n_observations"].min() > 0  # every agent observes something
        # All numeric columns should be finite
        numeric = df.select_dtypes(include="number")
        assert numeric.notna().all().all()
