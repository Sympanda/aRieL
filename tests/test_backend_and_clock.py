"""
Regression tests for:
  - DynamicBackend event validity using block_end (Item 1)
  - DynamicBackend preferred-method filtering (Item 4)
  - MissionClock.fraction_elapsed uses actual mission length (Item 6)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from aRieL.data.schemas import COST_FACTOR, MISSION_START_BJD
from aRieL.simulator.event_backend import DynamicBackend
from aRieL.simulator.mission_clock import MissionClock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_targets(
    n: int = 1,
    period: float = 10.0,
    transit_duration_s: float = 7200.0,   # 2 hours
    preferred_method: str = "Transit",
) -> pd.DataFrame:
    """Minimal target DataFrame for DynamicBackend."""
    return pd.DataFrame({
        "target_id":          [f"T{i}" for i in range(n)],
        "epoch":              [MISSION_START_BJD] * n,
        "period":             [period] * n,
        "transit_duration":   [transit_duration_s] * n,
        "eclipse_duration":   [transit_duration_s] * n,
        "preferred_method":   [preferred_method] * n,
        "science_weight":     [1.0] * n,
        "max_tier":           [3] * n,
        # columns needed by _base_science_value
        "planet_radius":      [5.0] * n,
        "planet_mass":        [20.0] * n,
        "planet_temperature": [800.0] * n,
        "stellar_temperature":[5000.0] * n,
        "stellar_metallicity":[0.0] * n,
        "distance_pc":        [100.0] * n,
        "tier3_required_obs": [8] * n,
    })


# ---------------------------------------------------------------------------
# Item 1: DynamicBackend event validity uses block_end
# ---------------------------------------------------------------------------

class TestDynamicBackendBlockEnd:
    """DynamicBackend must keep an event valid until block_end, not window_end."""

    def _transit_params(self, transit_duration_s: float = 7200.0, period: float = 10.0):
        """Return half_T14 in days and block_half in days."""
        half_t14 = transit_duration_s / 86400.0 / 2.0
        block_half = COST_FACTOR * (transit_duration_s / 86400.0) / 2.0
        return half_t14, block_half

    def test_event_present_before_window_end(self):
        """t_now well inside raw window → event returned."""
        targets = _make_targets(transit_duration_s=7200.0)
        backend = DynamicBackend(targets)

        half_t14, _ = self._transit_params()
        # Place clock just before window_end (= epoch + half_t14)
        t_now = MISSION_START_BJD + half_t14 - 0.001
        candidates = backend.candidates(t_now, k=10)
        assert len(candidates) >= 1, "Expected event still valid before window_end"

    def test_event_present_after_window_end_but_before_block_end(self):
        """t_now is past window_end but still inside block_end → event must be returned.

        Old code (using window_end) would drop the event here;
        corrected code (using block_end) must keep it.
        """
        targets = _make_targets(transit_duration_s=7200.0)
        backend = DynamicBackend(targets)

        half_t14, block_half = self._transit_params()
        # Place clock midway between window_end and block_end.
        # window_end = epoch + half_t14
        # block_end  = epoch + block_half  (= epoch + 1.25 × T14/2)
        t_now = MISSION_START_BJD + half_t14 + (block_half - half_t14) / 2.0
        candidates = backend.candidates(t_now, k=10)
        assert len(candidates) >= 1, (
            "Event should still be valid: t_now is past window_end but before block_end"
        )
        # The event returned should be the *current* occurrence (mid = epoch = MISSION_START_BJD)
        event_mids = candidates["window_mid"].values
        assert any(abs(m - MISSION_START_BJD) < 1e-6 for m in event_mids), (
            "Returned event should be the current occurrence (mid ≈ epoch)"
        )

    def test_event_gone_after_block_end(self):
        """t_now is past block_end → event must NOT be returned (next occ is future)."""
        targets = _make_targets(transit_duration_s=7200.0, period=10.0)
        backend = DynamicBackend(targets)

        _, block_half = self._transit_params()
        # Place clock just after block_end
        t_now = MISSION_START_BJD + block_half + 0.001
        candidates = backend.candidates(t_now, k=10)
        # The current occurrence has expired; the next one is ~10 days away
        if len(candidates) >= 1:
            event_mids = candidates["window_mid"].values
            assert not any(abs(m - MISSION_START_BJD) < 1e-6 for m in event_mids), (
                "Expired event must not appear after block_end"
            )


# ---------------------------------------------------------------------------
# Item 4: DynamicBackend preferred method filtering
# ---------------------------------------------------------------------------

class TestDynamicBackendPreferredMethod:
    """Transit-only targets should produce no eclipse candidates and vice versa."""

    def test_transit_only_no_eclipses(self):
        """preferred_method='Transit' → only transit events returned."""
        targets = _make_targets(preferred_method="Transit")
        backend = DynamicBackend(targets)
        t_now = MISSION_START_BJD + 5.0   # somewhere in the middle of the mission
        candidates = backend.candidates(t_now, k=50)
        if len(candidates) > 0:
            assert (candidates["event_type"] == "transit").all(), (
                "Transit-preferred targets must not produce eclipse candidates"
            )

    def test_eclipse_only_no_transits(self):
        """preferred_method='Eclipse' → only eclipse events returned."""
        targets = _make_targets(preferred_method="Eclipse")
        backend = DynamicBackend(targets)
        t_now = MISSION_START_BJD + 5.0
        candidates = backend.candidates(t_now, k=50)
        if len(candidates) > 0:
            assert (candidates["event_type"] == "eclipse").all(), (
                "Eclipse-preferred targets must not produce transit candidates"
            )

    def test_either_produces_both(self):
        """preferred_method='Either' → both transit and eclipse events returned."""
        targets = _make_targets(preferred_method="Either")
        backend = DynamicBackend(targets)
        t_now = MISSION_START_BJD + 5.0
        candidates = backend.candidates(t_now, k=50)
        # With one target there should be exactly one transit and one eclipse candidate
        event_types = set(candidates["event_type"].tolist())
        assert "transit" in event_types and "eclipse" in event_types, (
            "Either-preferred target should produce both transit and eclipse events"
        )

    def test_eclipse_preferred_transit_not_in_candidates(self):
        """Synthetic ID for eclipse-preferred target is eid=1; transit eid=0 absent."""
        targets = _make_targets(preferred_method="Eclipse")
        backend = DynamicBackend(targets)
        t_now = MISSION_START_BJD + 5.0
        candidates = backend.candidates(t_now, k=50)
        # Transit event id for target index 0 is 0*2=0; eclipse is 0*2+1=1
        assert 0 not in candidates["event_id"].values, (
            "Transit event_id=0 must not appear for an eclipse-preferred target"
        )


# ---------------------------------------------------------------------------
# Item 6: MissionClock.fraction_elapsed uses actual mission length
# ---------------------------------------------------------------------------

class TestFractionElapsed:
    """fraction_elapsed must use mission_end - mission_start, not a global constant."""

    def test_full_mission_reaches_one(self):
        clock = MissionClock(mission_start=2460000.0, mission_end=2460365.0)
        clock.current_time = clock.mission_end
        assert clock.fraction_elapsed == pytest.approx(1.0, rel=1e-6)

    def test_half_mission(self):
        clock = MissionClock(mission_start=2460000.0, mission_end=2460100.0)
        clock.current_time = 2460050.0   # exactly halfway
        assert clock.fraction_elapsed == pytest.approx(0.5, rel=1e-6)

    def test_short_episode_does_not_exceed_one(self):
        """A 30-day curriculum episode should give fraction in [0, 1], not overrun."""
        clock = MissionClock(mission_start=2460000.0, mission_end=2460030.0)  # 30 days
        clock.current_time = 2460025.0   # 25 days in
        assert 0.0 <= clock.fraction_elapsed <= 1.0

    def test_one_year_mission(self):
        """365-day mission: fraction at day 100 ≈ 0.274."""
        clock = MissionClock(mission_start=2460000.0, mission_end=2460365.0)
        clock.current_time = 2460100.0
        assert clock.fraction_elapsed == pytest.approx(100.0 / 365.0, rel=1e-5)

    def test_fraction_independent_of_global_constant(self):
        """Two clocks with different lengths both give correct fractions at midpoint."""
        for length in [30.0, 90.0, 365.25, 1278.0]:
            start = MISSION_START_BJD
            clock = MissionClock(mission_start=start, mission_end=start + length)
            clock.current_time = start + length / 2.0
            assert clock.fraction_elapsed == pytest.approx(0.5, rel=1e-5), (
                f"Failed for mission length {length} days"
            )
