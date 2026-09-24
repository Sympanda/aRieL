"""
Tests for action validity logic.

The action_mask module is not yet implemented, so these tests cover the
validity concepts currently embedded in MissionState and the event table:
  - an event is valid if visibility_valid=True
  - an event is valid if it fits within remaining mission time
  - an event is valid if we can arrive before window_end
"""

import pytest
import pandas as pd
import numpy as np

from aRieL.data.schemas import MISSION_START_BJD
from aRieL.simulator.mission_clock import MissionClock
from aRieL.simulator.mission_state import MissionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_id: int,
    target_id: str,
    window_mid: float,
    duration_days: float = 0.1,
    visibility_valid: bool = True,
    event_type: str = "transit",
) -> dict:
    from aRieL.data.schemas import COST_FACTOR
    return {
        "event_id":              event_id,
        "target_id":             target_id,
        "event_type":            event_type,
        "window_start":          window_mid - duration_days / 2,
        "window_mid":            window_mid,
        "window_end":            window_mid + duration_days / 2,
        "duration":              duration_days * 86400,
        "duration_days":         duration_days,
        "block_duration_days":   COST_FACTOR * duration_days,
        "tier_goal":             2,
        "base_science_value":    0.5,
        "visibility_valid":      visibility_valid,
        "ephemeris_uncertainty": 0.0,
        "event_index":           0,
    }


def make_state_with_events(events: list[dict]) -> MissionState:
    targets = pd.DataFrame({
        "target_idx":         [0],
        "target_id":          ["P_A"],
        "host_id":            ["S_A"],
        "ra":                 [30.0],
        "dec":                [10.0],
        "period":             [2.0],
        "epoch":              [MISSION_START_BJD + 1.0],
        "epoch_uncertainty":  [0.001],
        "transit_duration":   [5000.0],
        "eclipse_duration":   [5000.0],
        "planet_radius":      [2.0],
        "planet_mass":        [5.0],
        "planet_temperature": [800.0],
        "stellar_type":       ["G2"],
        "stellar_temperature":[5800.0],
        "stellar_metallicity":[0.0],
        "tier1_required_obs": [2],
        "tier2_required_obs": [5],
        "tier3_required_obs": [10],
        "max_tier":           [2],
        "preferred_method":   ["Transit"],
        "available_transits": [50],
        "available_eclipses": [50],
        "fgs_flag":           [1],
        "rp_rs":              [0.1],
        "a_rs":               [5.0],
        "eccentricity":       [0.0],
        "inclination":        [89.0],
        "distance_pc":        [20.0],
        "population_bin":     ["super_earth_warm_gf"],
        "science_weight":     [0.5],
        "obs_cost_days_t1":   [0.15],
        "obs_cost_days_t2":   [0.15],
        "obs_cost_days_t3":   [0.15],
    })
    events_df = pd.DataFrame(events).sort_values("window_mid").reset_index(drop=True)
    return MissionState.from_tables(targets, events_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpcomingEvents:
    def test_only_future_events_returned(self):
        t0 = MISSION_START_BJD
        events = [
            make_event(0, "P_A", t0 + 1.0),   # future
            make_event(1, "P_A", t0 + 5.0),   # future
        ]
        state = make_state_with_events(events)
        upcoming = state.upcoming_events(n=10)
        assert len(upcoming) == 2
        assert (upcoming["window_end"] > state.clock.current_time).all()

    def test_past_events_excluded(self):
        t0 = MISSION_START_BJD
        events = [
            make_event(0, "P_A", t0 - 10.0),  # in the past
            make_event(1, "P_A", t0 + 5.0),   # future
        ]
        state = make_state_with_events(events)
        upcoming = state.upcoming_events(n=10)
        assert len(upcoming) == 1
        assert upcoming.iloc[0]["event_id"] == 1

    def test_invisible_events_excluded(self):
        t0 = MISSION_START_BJD
        events = [
            make_event(0, "P_A", t0 + 1.0, visibility_valid=True),
            make_event(1, "P_A", t0 + 2.0, visibility_valid=False),
        ]
        state = make_state_with_events(events)
        upcoming = state.upcoming_events(n=10)
        assert len(upcoming) == 1
        assert upcoming.iloc[0]["event_id"] == 0

    def test_n_limits_results(self):
        t0 = MISSION_START_BJD
        events = [make_event(i, "P_A", t0 + i + 1.0) for i in range(20)]
        state = make_state_with_events(events)
        upcoming = state.upcoming_events(n=5)
        assert len(upcoming) <= 5


class TestMissedEvents:
    def test_missed_when_clock_past_window_end(self):
        """If the clock is already past window_end, the observation is missed."""
        t0 = MISSION_START_BJD
        dur = 0.05  # days
        # Event window: [t0+1, t0+1.05]
        events = [make_event(0, "P_A", t0 + 1.0, duration_days=dur)]
        state = make_state_with_events(events)
        # Advance clock well past the window
        state.clock.skip_to(t0 + 2.0)
        info = state.execute_observation(0)
        assert info["missed"]

    def test_not_missed_when_clock_before_window(self):
        t0 = MISSION_START_BJD
        events = [make_event(0, "P_A", t0 + 10.0, duration_days=0.5)]
        state = make_state_with_events(events)
        info = state.execute_observation(0)
        assert not info["missed"]
