"""
Tests for the observation requirements, tier progress mechanics, and the
multi-component reward function.
"""

import pytest
import pandas as pd
import numpy as np

from aRieL.data.observation_requirements import (
    compute_progress,
    initialise_progress_table,
)
from aRieL.rewards.compute_reward import compute_reward, _diversity_multiplier
from aRieL.utils.config import RewardConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_target(t1=2, t2=5, t3=10, max_tier=3) -> pd.Series:
    return pd.Series({
        "tier1_required_obs": t1,
        "tier2_required_obs": t2,
        "tier3_required_obs": t3,
        "max_tier":           max_tier,
    })


# ---------------------------------------------------------------------------
# compute_progress
# ---------------------------------------------------------------------------

class TestComputeProgress:
    def test_zero_obs_at_start(self):
        p = compute_progress(0, make_target())
        assert p["obs_completed"] == 0
        assert p["current_tier"] == 0
        assert not p["tier1_done"]
        assert not p["tier2_done"]
        assert not p["tier3_done"]
        assert p["progress_in_tier"] == pytest.approx(0.0)

    def test_progress_in_tier_before_t1(self):
        # 1 of 2 required for Tier 1 → 50%
        p = compute_progress(1, make_target(t1=2, t2=5, t3=10))
        assert p["progress_in_tier"] == pytest.approx(0.5)
        assert p["current_tier"] == 0

    def test_tier1_completion(self):
        p = compute_progress(2, make_target(t1=2, t2=5, t3=10))
        assert p["tier1_done"]
        assert p["current_tier"] == 1
        assert p["progress_in_tier"] == pytest.approx(0.0)

    def test_progress_between_t1_and_t2(self):
        # 3 of 5 total obs, T1=2 already done, 1 of 3 incremental obs toward T2
        p = compute_progress(3, make_target(t1=2, t2=5, t3=10))
        assert p["current_tier"] == 1
        assert not p["tier2_done"]
        expected_progress = (3 - 2) / (5 - 2)  # 1/3
        assert p["progress_in_tier"] == pytest.approx(expected_progress, rel=1e-6)

    def test_tier2_completion(self):
        p = compute_progress(5, make_target(t1=2, t2=5, t3=10))
        assert p["tier2_done"]
        assert p["current_tier"] == 2

    def test_tier3_completion(self):
        p = compute_progress(10, make_target(t1=2, t2=5, t3=10))
        assert p["tier3_done"]
        assert p["current_tier"] == 3
        assert p["progress_in_tier"] == pytest.approx(1.0)
        assert p["obs_remaining_next_tier"] == 0

    def test_max_tier_1_caps_at_tier1(self):
        """Targets with max_tier=1 should never show tier2_done or tier3_done."""
        p = compute_progress(99, make_target(t1=2, t2=5, t3=10, max_tier=1))
        assert p["tier1_done"]
        assert not p["tier2_done"]
        assert not p["tier3_done"]
        assert p["current_tier"] == 1
        assert p["obs_remaining_next_tier"] == 0

    def test_max_tier_2_caps_at_tier2(self):
        p = compute_progress(99, make_target(t1=2, t2=5, t3=10, max_tier=2))
        assert p["tier2_done"]
        assert not p["tier3_done"]
        assert p["current_tier"] == 2

    def test_obs_remaining_decreases(self):
        p0 = compute_progress(0, make_target())
        p1 = compute_progress(1, make_target())
        assert p1["obs_remaining_next_tier"] < p0["obs_remaining_next_tier"]

    def test_progress_clipped_to_1(self):
        # Over-observing should not push progress > 1
        p = compute_progress(100, make_target(t1=2, t2=5, t3=10, max_tier=3))
        assert p["progress_in_tier"] <= 1.0

    def test_monotone_progress_sequence(self):
        """progress_in_tier should be non-decreasing within a tier."""
        target = make_target(t1=3, t2=8, t3=15)
        progresses = [compute_progress(n, target)["progress_in_tier"]
                      for n in range(16)]
        # Within tier 0 (n=0..2), progress increases
        assert progresses[0] < progresses[1] < progresses[2]


# ---------------------------------------------------------------------------
# initialise_progress_table
# ---------------------------------------------------------------------------

class TestInitialiseProgressTable:
    @pytest.fixture
    def targets_df(self):
        return pd.DataFrame({
            "target_id":          ["P_A", "P_B"],
            "tier1_required_obs": [2, 1],
            "tier2_required_obs": [5, 4],
            "tier3_required_obs": [10, 8],
            "max_tier":           [3, 2],
        })

    def test_returns_dataframe(self, targets_df):
        df = initialise_progress_table(targets_df)
        assert isinstance(df, pd.DataFrame)

    def test_indexed_by_target_id(self, targets_df):
        df = initialise_progress_table(targets_df)
        assert "P_A" in df.index
        assert "P_B" in df.index

    def test_all_obs_completed_zero(self, targets_df):
        df = initialise_progress_table(targets_df)
        assert (df["obs_completed"] == 0).all()

    def test_no_tiers_done(self, targets_df):
        df = initialise_progress_table(targets_df)
        assert not df["tier1_done"].any()
        assert not df["tier2_done"].any()
        assert not df["tier3_done"].any()

    def test_required_columns_present(self, targets_df):
        df = initialise_progress_table(targets_df)
        for col in ("obs_completed", "current_tier", "tier1_done", "tier2_done",
                    "tier3_done", "progress_in_tier", "obs_remaining_next_tier"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Diversity multiplier
# ---------------------------------------------------------------------------

class TestDiversityMultiplier:
    def test_unseen_bin_returns_max_multiplier(self):
        """A bin with no observations yet should give multiplier = max_multiplier (default 5.0)."""
        mult = _diversity_multiplier(
            "hot_jupiter",
            bin_totals={"hot_jupiter": 10},
            bin_observed={},
        )
        assert mult == pytest.approx(5.0)

    def test_custom_max_multiplier(self):
        """Passing max_multiplier=2.0 reproduces the old transformer_v1 behaviour."""
        mult = _diversity_multiplier(
            "hot_jupiter",
            bin_totals={"hot_jupiter": 10},
            bin_observed={},
            max_multiplier=2.0,
        )
        assert mult == pytest.approx(2.0)

    def test_fully_observed_bin_returns_one(self):
        """A bin where all targets are T1+ complete gives multiplier = 1.0 regardless of max."""
        mult = _diversity_multiplier(
            "hot_jupiter",
            bin_totals={"hot_jupiter": 5},
            bin_observed={"hot_jupiter": 5},
        )
        assert mult == pytest.approx(1.0)

    def test_half_observed_with_default_max(self):
        """Half-observed bin: 1 + (5-1)*0.5 = 3.0 with default max_multiplier=5.0."""
        mult = _diversity_multiplier(
            "warm_neptune",
            bin_totals={"warm_neptune": 4},
            bin_observed={"warm_neptune": 2},
        )
        assert mult == pytest.approx(3.0)

    def test_half_observed_legacy_max(self):
        """Explicitly passing max_multiplier=2.0 gives 1.5 (old behaviour)."""
        mult = _diversity_multiplier(
            "warm_neptune",
            bin_totals={"warm_neptune": 4},
            bin_observed={"warm_neptune": 2},
            max_multiplier=2.0,
        )
        assert mult == pytest.approx(1.5)

    def test_unknown_bin_defaults_to_max_multiplier(self):
        """A population_bin not in bin_totals should not crash and give the max boost."""
        mult = _diversity_multiplier(
            "exotic_world",
            bin_totals={"hot_jupiter": 10},
            bin_observed={},
        )
        assert mult == pytest.approx(5.0)

    def test_multiplier_never_below_one(self):
        """Over-observed bins (count > total due to rounding) still give >= 1.0."""
        mult = _diversity_multiplier(
            "hot_jupiter",
            bin_totals={"hot_jupiter": 5},
            bin_observed={"hot_jupiter": 100},
        )
        assert mult >= 1.0

    def test_multiplier_linear_between_one_and_max(self):
        """Multiplier should decrease linearly as the bin fills."""
        totals = {"bin": 10}
        for n_observed in range(0, 11):
            mult = _diversity_multiplier("bin", totals, {"bin": n_observed})
            expected = 1.0 + (5.0 - 1.0) * (1.0 - n_observed / 10.0)
            assert mult == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    return RewardConfig()


@pytest.fixture
def bin_totals():
    return {"hot_jupiter": 10, "warm_neptune": 4, "cool_rocky": 2}


def _step(
    missed=False,
    science_weight=1.0,
    population_bin="hot_jupiter",
    tier_before=0,
    tier_after=0,
    progress_before=0.0,
    progress_after=0.2,
    obs_duration_days=0.05,
    slew_days=0.01,
    idle_days=0.0,
    total_cost_days=None,
):
    total = total_cost_days if total_cost_days is not None else (obs_duration_days + slew_days + idle_days)
    return dict(
        missed=missed,
        science_weight=science_weight,
        population_bin=population_bin,
        tier_before=tier_before,
        tier_after=tier_after,
        progress_before=progress_before,
        progress_after=progress_after,
        obs_duration_days=obs_duration_days,
        slew_days=slew_days,
        idle_days=idle_days,
        total_cost_days=total,
    )


def _reward(step, cfg, bin_totals, bin_observed=None):
    """Helper that calls compute_reward with the new signature (same dict for before/after)."""
    obs = bin_observed if bin_observed is not None else {}
    return compute_reward(step, cfg, bin_totals, bin_observed_before=obs, bin_observed_after=obs)


class TestComputeReward:
    def test_missed_returns_negative_miss_penalty(self, default_cfg, bin_totals):
        r = _reward(_step(missed=True), default_cfg, bin_totals, {})
        assert r == pytest.approx(-default_cfg.miss_penalty)

    def test_missed_does_not_add_other_components(self, default_cfg, bin_totals):
        """A missed event should only incur the miss penalty, nothing else."""
        r = _reward(
            _step(missed=True, tier_before=0, tier_after=1),
            default_cfg, bin_totals, {},
        )
        assert r == pytest.approx(-default_cfg.miss_penalty)

    def test_no_tier_completion_no_tier_bonus(self, default_cfg, bin_totals):
        r = _reward(
            _step(tier_before=0, tier_after=0, progress_before=0.0, progress_after=0.2),
            default_cfg, bin_totals, {},
        )
        assert r > 0.0
        # Should not include tier bonus — only progress + efficiency.
        # The tier-1 bonus for an unseen bin would be:
        #   tier1_completion × science_weight × diversity_multiplier_max
        # which is well above what progress + efficiency can produce.
        expected_tier_bonus = (
            default_cfg.tier1_completion * 1.0 * default_cfg.diversity_multiplier_max
        )
        assert r < expected_tier_bonus

    def test_tier1_completion_bonus_positive(self, default_cfg, bin_totals):
        r = _reward(
            _step(tier_before=0, tier_after=1, progress_before=0.8, progress_after=0.0),
            default_cfg, bin_totals, {},
        )
        # T1 bonus = tier1_completion * science_weight * div_mult + efficiency
        assert r > 0.0

    def test_tier3_completion_bonus_largest(self, default_cfg, bin_totals):
        r_t1 = _reward(_step(tier_before=0, tier_after=1), default_cfg, bin_totals, {})
        r_t3 = _reward(_step(tier_before=2, tier_after=3), default_cfg, bin_totals, {})
        assert r_t3 > r_t1

    def test_higher_science_weight_gives_higher_reward(self, default_cfg, bin_totals):
        r_low  = _reward(_step(science_weight=0.2), default_cfg, bin_totals, {})
        r_high = _reward(_step(science_weight=1.0), default_cfg, bin_totals, {})
        assert r_high > r_low

    def test_efficiency_reward_penalises_long_slew(self, default_cfg, bin_totals):
        """Observation with short slew should score higher than long slew."""
        r_short = _reward(
            _step(obs_duration_days=0.05, slew_days=0.001),
            default_cfg, bin_totals, {},
        )
        r_long = _reward(
            _step(obs_duration_days=0.05, slew_days=1.0),
            default_cfg, bin_totals, {},
        )
        assert r_short > r_long

    def test_rare_bin_gets_higher_reward(self, default_cfg, bin_totals):
        """An under-observed bin should produce a higher reward than a saturated one."""
        bin_obs_saturated = {"hot_jupiter": 10}
        bin_obs_empty     = {}
        r_saturated = _reward(
            _step(population_bin="hot_jupiter"), default_cfg, bin_totals, bin_obs_saturated
        )
        r_rare = _reward(
            _step(population_bin="hot_jupiter"), default_cfg, bin_totals, bin_obs_empty
        )
        assert r_rare > r_saturated

    def test_no_progress_shaping_on_tier_completion(self, default_cfg, bin_totals):
        """When a tier is crossed, progress shaping is NOT added."""
        r_with_crossing = _reward(
            _step(tier_before=0, tier_after=1, progress_before=0.9, progress_after=0.0),
            default_cfg, bin_totals, {},
        )
        r_without_crossing = _reward(
            _step(tier_before=0, tier_after=0, progress_before=0.7, progress_after=0.9),
            default_cfg, bin_totals, {},
        )
        assert r_with_crossing > r_without_crossing

    def test_reward_is_float(self, default_cfg, bin_totals):
        r = _reward(_step(), default_cfg, bin_totals, {})
        assert isinstance(r, float)

    def test_zero_efficiency_weight_removes_efficiency_bonus(self, bin_totals):
        cfg_no_eff = RewardConfig(efficiency_weight=0.0, progress_weight=0.0)
        r = _reward(
            _step(tier_before=0, tier_after=0, progress_before=0.1, progress_after=0.1),
            cfg_no_eff, bin_totals, {},
        )
        assert r == pytest.approx(0.0)
