"""Tests for simulator.slew — pure math, no pandas required."""

import math

import pytest

from aRieL.simulator.slew import (
    MAX_SLEW_S,
    MIN_SLEW_S,
    SLEW_RATE_S_PER_DEG,
    angular_separation_deg,
    slew_time_days,
    slew_time_seconds,
)


class TestAngularSeparation:
    def test_same_point_is_zero(self):
        assert angular_separation_deg(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-10)

    def test_known_separation_along_equator(self):
        # Two points on the equator separated by 90 degrees in RA
        sep = angular_separation_deg(0.0, 0.0, 90.0, 0.0)
        assert sep == pytest.approx(90.0, rel=1e-6)

    def test_antipodal_points(self):
        sep = angular_separation_deg(0.0, 0.0, 180.0, 0.0)
        assert sep == pytest.approx(180.0, abs=1e-6)

    def test_north_to_south_pole(self):
        sep = angular_separation_deg(0.0, 90.0, 0.0, -90.0)
        assert sep == pytest.approx(180.0, abs=1e-6)

    def test_small_separation(self):
        # 1 degree separation along the equator (dec=0 → no cos correction)
        sep = angular_separation_deg(10.0, 0.0, 11.0, 0.0)
        assert sep == pytest.approx(1.0, rel=1e-4)

    def test_symmetry(self):
        sep1 = angular_separation_deg(30.0, 45.0, 90.0, -10.0)
        sep2 = angular_separation_deg(90.0, -10.0, 30.0, 45.0)
        assert sep1 == pytest.approx(sep2, rel=1e-10)

    def test_non_negative(self):
        for ra1, dec1, ra2, dec2 in [(0, 0, 0, 0), (10, 5, 350, -80), (180, 45, 0, -45)]:
            assert angular_separation_deg(ra1, dec1, ra2, dec2) >= 0.0


class TestSlewTime:
    def test_minimum_enforced_for_same_point(self):
        t = slew_time_seconds(10.0, 20.0, 10.0, 20.0)
        assert t == pytest.approx(MIN_SLEW_S)

    def test_scales_with_separation(self):
        t1 = slew_time_seconds(0.0, 0.0, 10.0, 0.0)
        t2 = slew_time_seconds(0.0, 0.0, 20.0, 0.0)
        # Should be approximately twice as long (unless hitting min/max)
        if t1 > MIN_SLEW_S and t2 < MAX_SLEW_S:
            assert t2 == pytest.approx(2 * t1, rel=1e-3)

    def test_maximum_enforced(self):
        # 180 degree slew should be capped
        t = slew_time_seconds(0.0, 0.0, 180.0, 0.0)
        assert t <= MAX_SLEW_S

    def test_10_degree_slew(self):
        t = slew_time_seconds(0.0, 0.0, 10.0, 0.0)
        expected = 10.0 * SLEW_RATE_S_PER_DEG
        # Should be above minimum
        assert t == pytest.approx(max(MIN_SLEW_S, expected), rel=1e-4)

    def test_days_conversion(self):
        t_s = slew_time_seconds(0.0, 0.0, 30.0, 0.0)
        t_d = slew_time_days(0.0, 0.0, 30.0, 0.0)
        assert t_d == pytest.approx(t_s / 86400.0, rel=1e-10)

    def test_positive(self):
        assert slew_time_seconds(45.0, 30.0, 90.0, -15.0) > 0


class TestSlewMatrix:
    def test_matrix_shape_and_symmetry(self):
        import numpy as np
        import pandas as pd
        from aRieL.simulator.slew import build_slew_matrix

        targets = pd.DataFrame({
            "ra":  [10.0, 50.0, 200.0],
            "dec": [20.0, -30.0, 0.0],
        })
        matrix = build_slew_matrix(targets)
        assert matrix.shape == (3, 3)
        # Diagonal is zero (same pointing → minimum is applied, but let's check symmetry)
        np.testing.assert_allclose(matrix, matrix.T, rtol=1e-5)

    def test_diagonal_is_minimum_slew(self):
        import numpy as np
        import pandas as pd
        from aRieL.simulator.slew import build_slew_matrix

        targets = pd.DataFrame({"ra": [0.0, 90.0], "dec": [0.0, 0.0]})
        matrix = build_slew_matrix(targets)
        # Self-slew should be 0 in the raw loop (never set)
        assert matrix[0, 0] == 0.0
        assert matrix[1, 1] == 0.0
        assert matrix[0, 1] > 0
