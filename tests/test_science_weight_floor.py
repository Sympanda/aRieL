"""
Tests for science_weight_floor propagation through population-bin weighting.

Verifies that the configured floor actually changes the resulting science weights
(Item 1 regression: the floor must be wired all the way from RewardConfig →
build_target_table → assign_population_bins → _compute_weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aRieL.data.population_bins import assign_population_bins, _compute_weights


# ---------------------------------------------------------------------------
# _compute_weights tests (unit)
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def _bins(self) -> pd.Series:
        """Return a Series with three bins of different frequencies."""
        return pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 5)

    def test_floor_zero_most_common_bin_gets_zero(self):
        weights = _compute_weights(self._bins(), floor=0.0)
        # Bin A is most common → weight should be 0 when floor=0
        assert float(weights[self._bins() == "A"].iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_floor_nonzero_minimum_weight_at_least_floor(self):
        floor = 0.3
        weights = _compute_weights(self._bins(), floor=floor)
        assert float(weights.min()) >= floor - 1e-9

    def test_max_weight_always_one(self):
        for floor in [0.0, 0.1, 0.5]:
            weights = _compute_weights(self._bins(), floor=floor)
            assert float(weights.max()) == pytest.approx(1.0, rel=1e-6)

    def test_changing_floor_changes_weights(self):
        bins = self._bins()
        w_low  = _compute_weights(bins, floor=0.1)
        w_high = _compute_weights(bins, floor=0.5)
        # Minimum weight must differ
        assert float(w_low.min()) < float(w_high.min()) - 1e-6


# ---------------------------------------------------------------------------
# assign_population_bins integration test
# ---------------------------------------------------------------------------

class TestScienceWeightFloorPropagation:
    def _minimal_targets(self, n: int = 60) -> pd.DataFrame:
        """Minimal target DataFrame with enough columns for assign_population_bins."""
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "target_id":          [f"T{i}" for i in range(n)],
            "planet_radius":      rng.uniform(1, 15, n),
            "planet_mass":        rng.uniform(1, 300, n),
            "planet_temperature": rng.uniform(300, 2000, n),
            "period":             rng.uniform(0.5, 30, n),
            "stellar_temperature":rng.uniform(3500, 7000, n),
            "max_tier":           rng.integers(1, 4, n),
        })

    def test_different_floor_produces_different_weights(self):
        targets = self._minimal_targets()
        df_low  = assign_population_bins(targets.copy(), science_weight_floor=0.05)
        df_high = assign_population_bins(targets.copy(), science_weight_floor=0.5)
        # science_weight values must differ (not all the same)
        assert not np.allclose(
            df_low["science_weight"].values,
            df_high["science_weight"].values,
        ), "Changing science_weight_floor must change science_weight values"

    def test_floor_respected_as_minimum(self):
        for floor in [0.0, 0.2, 0.5, 0.8]:
            targets = self._minimal_targets()
            df = assign_population_bins(targets.copy(), science_weight_floor=floor)
            assert float(df["science_weight"].min()) >= floor - 1e-9, (
                f"science_weight floor={floor} violated: min={df['science_weight'].min()}"
            )

    def test_max_weight_always_one(self):
        targets = self._minimal_targets()
        for floor in [0.0, 0.3, 0.7]:
            df = assign_population_bins(targets.copy(), science_weight_floor=floor)
            assert float(df["science_weight"].max()) == pytest.approx(1.0, rel=1e-4)
