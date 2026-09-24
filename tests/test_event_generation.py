"""Tests for ephemeris propagation and event table generation."""

import math

import numpy as np
import pytest

from aRieL.simulator.ephemeris import eclipse_offset_days, propagate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPOCH = 2460000.0    # arbitrary BJD reference
PERIOD = 3.0         # days


class TestEphemerisPropagation:
    def test_single_transit_at_epoch(self):
        """If t_start=t_end=epoch the epoch itself should be returned."""
        result = propagate("target_A", EPOCH, PERIOD, EPOCH, EPOCH)
        assert len(result.mid_times) == 1
        assert result.mid_times[0] == pytest.approx(EPOCH)

    def test_no_events_outside_window(self):
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH - 0.1, t_end=EPOCH - 0.01)
        assert len(result.mid_times) == 0

    def test_correct_count_over_mission(self):
        """Over 10 days with period=3, expect floor(10/3)+1 = 4 transits."""
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH, t_end=EPOCH + 10.0)
        # n=0,1,2,3 → t=0,3,6,9 all within [0,10]
        assert len(result.mid_times) == 4

    def test_mid_times_spaced_by_period(self):
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH, t_end=EPOCH + 20.0)
        diffs = np.diff(result.mid_times)
        np.testing.assert_allclose(diffs, PERIOD, rtol=1e-10)

    def test_transit_vs_eclipse_offset(self):
        """Eclipse mid-times should be offset from transit mid-times by ~period/2."""
        t_result = propagate("target_A", EPOCH, PERIOD,
                             t_start=EPOCH, t_end=EPOCH + 30.0, event_type="transit")
        e_result = propagate("target_A", EPOCH, PERIOD,
                             t_start=EPOCH, t_end=EPOCH + 30.0, event_type="eclipse")
        # Eclipse times should be offset by period/2
        offset = PERIOD / 2.0
        # Each eclipse should be approximately transit + offset
        for et in e_result.mid_times:
            # Find nearest transit
            nearest = min(t_result.mid_times, key=lambda tt: abs(tt - (et - offset)))
            assert abs((et - offset) - nearest) < 1e-8

    def test_uncertainty_grows_with_n(self):
        """Uncertainty should be monotonically non-decreasing with |n|."""
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH, t_end=EPOCH + 30.0,
                           sigma_epoch_days=0.001, sigma_period_days=0.0001)
        # uncertainties are in seconds; indices grow → uncertainty grows
        assert all(result.uncertainties[i] <= result.uncertainties[i + 1]
                   for i in range(len(result.uncertainties) - 1))

    def test_zero_uncertainty_when_no_sigma(self):
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH, t_end=EPOCH + 10.0,
                           sigma_epoch_days=0.0, sigma_period_days=0.0)
        np.testing.assert_array_equal(result.uncertainties, 0.0)

    def test_indices_are_consecutive(self):
        result = propagate("target_A", EPOCH, PERIOD,
                           t_start=EPOCH, t_end=EPOCH + 10.0)
        expected_indices = np.arange(0, 4, dtype=np.int64)
        np.testing.assert_array_equal(result.indices, expected_indices)

    def test_event_type_stored(self):
        r = propagate("X", EPOCH, PERIOD, EPOCH, EPOCH + 5, event_type="eclipse")
        assert r.event_type == "eclipse"

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            propagate("X", EPOCH, 0.0, EPOCH, EPOCH + 10)


class TestEclipseOffset:
    def test_circular_orbit_is_half_period(self):
        assert eclipse_offset_days(PERIOD, eccentricity=0.0) == pytest.approx(PERIOD / 2.0)

    def test_eccentric_orbit_differs(self):
        offset_circ = eclipse_offset_days(PERIOD, eccentricity=0.0)
        # omega=0 → cos(omega)=1 → maximum correction term
        offset_ecc = eclipse_offset_days(PERIOD, eccentricity=0.3, omega_deg=0.0)
        assert offset_circ != pytest.approx(offset_ecc, rel=1e-3)


class TestGenerateEvents:
    """Integration tests for the full event table generator."""

    @pytest.fixture
    def small_targets(self):
        """Minimal target DataFrame with 3 targets."""
        import pandas as pd
        from aRieL.data.schemas import MISSION_START_BJD

        data = {
            "target_idx":        [0, 1, 2],
            "target_id":         ["PlanetA", "PlanetB", "PlanetC"],
            "host_id":           ["StarA", "StarB", "StarC"],
            "ra":                [30.0, 90.0, 200.0],
            "dec":               [10.0, -20.0, 45.0],
            "period":            [1.0, 5.0, 10.0],
            "epoch":             [MISSION_START_BJD + 0.5, MISSION_START_BJD + 1.0, MISSION_START_BJD + 2.0],
            "epoch_uncertainty": [0.001, 0.002, 0.001],
            "transit_duration":  [3600.0, 7200.0, 5400.0],   # seconds
            "eclipse_duration":  [3600.0, 7200.0, 5400.0],
            "planet_radius":     [2.0, 4.0, 12.0],
            "planet_mass":       [5.0, 20.0, 300.0],
            "planet_temperature":[800.0, 500.0, 1800.0],
            "stellar_type":      ["G2", "M4", "F5"],
            "stellar_temperature":[5800.0, 3500.0, 6500.0],
            "stellar_metallicity":[0.0, -0.2, 0.1],
            "tier1_required_obs":[2, 1, 3],
            "tier2_required_obs":[5, 4, 8],
            "tier3_required_obs":[10, 8, 15],
            "max_tier":          [3, 2, 2],
            "preferred_method":  ["Transit", "Eclipse", "Transit"],
            "available_transits":[100, 50, 30],
            "available_eclipses":[100, 50, 30],
            "fgs_flag":          [1, 1, 1],
            "rp_rs":             [0.1, 0.12, 0.15],
            "a_rs":              [5.0, 10.0, 20.0],
            "eccentricity":      [0.0, 0.05, 0.1],
            "inclination":       [89.0, 88.0, 87.0],
            "distance_pc":       [20.0, 50.0, 100.0],
            "population_bin":    ["super_earth_warm_gf", "neptune_warm_m", "jupiter_very_hot_gf"],
            "science_weight":    [0.5, 0.8, 0.3],
            "obs_cost_days_t1":  [0.104, 0.208, 0.156],
            "obs_cost_days_t2":  [0.104, 0.208, 0.156],
            "obs_cost_days_t3":  [0.104, 0.208, 0.156],
        }
        return pd.DataFrame(data)

    def test_returns_dataframe(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD, MISSION_END_BJD
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        import pandas as pd
        assert isinstance(events, pd.DataFrame)

    def test_events_within_window(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD
        m_start = MISSION_START_BJD
        m_end = m_start + 30.0
        events = generate_events(small_targets, mission_start=m_start, mission_end=m_end)
        assert (events["window_mid"] >= m_start).all()
        assert (events["window_mid"] <= m_end).all()

    def test_sorted_by_window_mid(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        assert (events["window_mid"].diff().dropna() >= 0).all()

    def test_eclipse_target_generates_eclipses(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        planet_b_events = events[events["target_id"] == "PlanetB"]
        assert (planet_b_events["event_type"] == "eclipse").all()

    def test_required_columns_present(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD, EVENT_COLS
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        for col in EVENT_COLS:
            assert col in events.columns, f"Missing column: {col}"

    def test_window_start_before_mid_before_end(self, small_targets):
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        assert (events["window_start"] < events["window_mid"]).all()
        assert (events["window_mid"] < events["window_end"]).all()

    def test_fast_planet_has_more_events(self, small_targets):
        """PlanetA (period=1d) should have more events than PlanetC (period=10d)."""
        from aRieL.simulator.event_generator import generate_events
        from aRieL.data.schemas import MISSION_START_BJD
        events = generate_events(small_targets,
                                  mission_start=MISSION_START_BJD,
                                  mission_end=MISSION_START_BJD + 30.0)
        n_a = len(events[events["target_id"] == "PlanetA"])
        n_c = len(events[events["target_id"] == "PlanetC"])
        assert n_a > n_c
