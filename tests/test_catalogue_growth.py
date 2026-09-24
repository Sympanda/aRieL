"""Tests for catalogue growth schedules, overflow milestones, and injection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aRieL.rewards.compute_reward import check_milestone_reward
from aRieL.simulator.catalogue_growth import (
    build_injection_schedule,
    expand_milestone_fractions,
)
from aRieL.utils.config import RewardConfig


class TestExpandMilestones:
    def test_extends_every_10pct_above_one(self):
        fracs = expand_milestone_fractions(
            (0.5, 1.0), max_fraction=1.35, step_above_one=0.1
        )
        assert 1.0 in fracs
        assert 1.1 in fracs
        assert 1.2 in fracs
        assert 1.3 in fracs
        assert 1.4 not in fracs

    def test_disabled_when_step_zero(self):
        fracs = expand_milestone_fractions(
            (0.5, 1.0), max_fraction=2.0, step_above_one=0.0
        )
        assert fracs == (0.5, 1.0)


class TestMilestoneOverflowBonus:
    def test_exact_100_uses_final_bonus(self):
        cfg = RewardConfig(
            t1_milestone_fractions=(0.5, 1.0, 1.1),
            t1_milestone_bonus=1.0,
            t1_final_milestone_bonus=30.0,
            t1_overflow_milestone_bonus=2.0,
        )
        bonus, hit = check_milestone_reward(
            n_completed=10,
            total_reachable=10,  # exactly 100%
            milestones_hit=set(),
            milestone_fractions=cfg.t1_milestone_fractions,
            intermediate_bonus=cfg.t1_milestone_bonus,
            final_bonus=cfg.t1_final_milestone_bonus,
            overflow_bonus=cfg.t1_overflow_milestone_bonus,
        )
        assert 1.0 in hit
        assert bonus == pytest.approx(31.0)  # 0.5 + 1.0 final

    def test_overflow_uses_overflow_bonus(self):
        cfg = RewardConfig(
            t1_milestone_fractions=(1.0, 1.1, 1.2),
            t1_milestone_bonus=1.0,
            t1_final_milestone_bonus=30.0,
            t1_overflow_milestone_bonus=2.0,
        )
        bonus, hit = check_milestone_reward(
            n_completed=12,
            total_reachable=10,  # 120% of initial
            milestones_hit={1.0},
            milestone_fractions=cfg.t1_milestone_fractions,
            intermediate_bonus=cfg.t1_milestone_bonus,
            final_bonus=cfg.t1_final_milestone_bonus,
            overflow_bonus=cfg.t1_overflow_milestone_bonus,
        )
        assert 1.1 in hit and 1.2 in hit
        assert bonus == pytest.approx(4.0)


class TestInjectionScheduleDeterminism:
    def _pool(self, n: int = 200) -> pd.DataFrame:
        return pd.DataFrame({"target_id": [f"TOI-{i}" for i in range(n)]})

    def test_same_seed_same_schedule(self):
        kwargs = dict(
            pool=self._pool(),
            mission_start=1000.0,
            mission_end=1000.0 + 120.0,  # ~4 months
            mean_per_month=15.0,
            std_per_month=5.0,
            month_days=30.0,
        )
        a = build_injection_schedule(**kwargs, rng=np.random.default_rng(7))
        b = build_injection_schedule(**kwargs, rng=np.random.default_rng(7))
        assert [(e.bjd, e.target_id) for e in a] == [(e.bjd, e.target_id) for e in b]
        assert len(a) > 0

    def test_different_seed_diverges(self):
        pool = self._pool()
        a = build_injection_schedule(
            pool, mission_start=0, mission_end=90, rng=np.random.default_rng(1),
            mean_per_month=15, std_per_month=5, month_days=30,
        )
        b = build_injection_schedule(
            pool, mission_start=0, mission_end=90, rng=np.random.default_rng(2),
            mean_per_month=15, std_per_month=5, month_days=30,
        )
        assert [(e.bjd, e.target_id) for e in a] != [(e.bjd, e.target_id) for e in b]

    def test_no_replacement(self):
        pool = self._pool(50)
        sched = build_injection_schedule(
            pool, mission_start=0, mission_end=365, rng=np.random.default_rng(0),
            mean_per_month=20, std_per_month=1, month_days=30,
        )
        ids = [e.target_id for e in sched]
        assert len(ids) == len(set(ids))


class TestFixedVsPerEpisodeMode:
    """Env-level schedule_mode: fixed is identical across reset seeds."""

    @pytest.fixture
    def tpc_csv(self, tmp_path):
        from aRieL.utils.paths import find_data_csv
        found = find_data_csv(["TPC.csv"], extra_globs=False)
        if found is None:
            pytest.skip("TPC.csv not available; env-level growth tests need a real TPC file")
        return found

    def test_fixed_mode_ignores_reset_seed(self, tpc_csv):
        from aRieL.data.demo import make_demo_targets
        from aRieL.envs.env import ArielEnv
        from aRieL.utils.config import CatalogueGrowthConfig, default_env_config
        import dataclasses

        base = default_env_config()
        growth = CatalogueGrowthConfig(
            enabled=True,
            tpc_csv_path=str(tpc_csv),
            schedule_mode="fixed",
            schedule_seed=123,
            mean_per_month=15.0,
            std_per_month=5.0,
        )
        cfg = dataclasses.replace(base, catalogue_growth=growth, seed=0)
        env = ArielEnv(config=cfg, targets=make_demo_targets(8))
        env.reset(seed=1)
        s1 = [(e.bjd, e.target_id) for e in env._injection_schedule]
        env.reset(seed=999)
        s2 = [(e.bjd, e.target_id) for e in env._injection_schedule]
        assert s1 == s2
        assert len(s1) > 0

    def test_per_episode_mode_varies_with_seed(self, tpc_csv):
        from aRieL.data.demo import make_demo_targets
        from aRieL.envs.env import ArielEnv
        from aRieL.utils.config import CatalogueGrowthConfig, default_env_config
        import dataclasses

        base = default_env_config()
        growth = CatalogueGrowthConfig(
            enabled=True,
            tpc_csv_path=str(tpc_csv),
            schedule_mode="per_episode",
            mean_per_month=15.0,
            std_per_month=5.0,
        )
        cfg = dataclasses.replace(base, catalogue_growth=growth, seed=0)
        env = ArielEnv(config=cfg, targets=make_demo_targets(8))
        env.reset(seed=1)
        s1 = [(e.bjd, e.target_id) for e in env._injection_schedule]
        env.reset(seed=2)
        s2 = [(e.bjd, e.target_id) for e in env._injection_schedule]
        assert s1 != s2
