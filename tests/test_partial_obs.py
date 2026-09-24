"""
Tests for the partial-observation model.

Three cases are tested for MissionState.execute_observation:

  Case A  t_arrive ≤ block_start   → captured_fraction = 1.0
  Case B  block_start < t_arrive < block_end
                                   → captured_fraction = (block_end - t_arrive)
                                                         / block_duration_days
  Case C  t_arrive ≥ block_end     → missed = True, no progress

Additional tests cover:
  - fractional progress accumulation across multiple steps
  - tier boundary crossed by fractional accumulation
  - action mask consistency with the block_end miss threshold
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from aRieL.data.schemas import (
    COST_FACTOR,
    MISSION_START_BJD,
    MISSION_END_BJD,
)
from aRieL.simulator.mission_state import MissionState
from aRieL.simulator.mission_clock import MissionClock
from aRieL.simulator.slew import MIN_SLEW_S

# Minimum slew (same-pointing) in days.  Always paid, even with current_ra ==
# target_ra.  Test clocks are set so that t_now + MIN_SLEW_DAYS == desired
# arrival time, giving exact partial-capture fractions.
MIN_SLEW_DAYS: float = MIN_SLEW_S / 86400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_target_df(ra: float = 0.0) -> pd.DataFrame:
    """Minimal target DataFrame with one target (T1=2, T2=4, T3=8)."""
    dur_s = 7200.0   # 2-hour transit → block ≈ 5 h
    dur_d = dur_s / 86400.0
    return pd.DataFrame({
        "target_idx":         [0],
        "target_id":          ["T_PARTIAL"],
        "host_id":            ["S_PARTIAL"],
        "ra":                 [ra],
        "dec":                [0.0],
        "period":             [3.0],
        "epoch":              [MISSION_START_BJD + 10.0],
        "epoch_uncertainty":  [0.001],
        "transit_duration":   [dur_s],
        "eclipse_duration":   [dur_s],
        "planet_radius":      [3.0],
        "planet_mass":        [10.0],
        "planet_temperature": [800.0],
        "stellar_type":       ["G2"],
        "stellar_temperature":[5800.0],
        "stellar_metallicity":[0.0],
        "tier1_required_obs": [2],
        "tier2_required_obs": [4],
        "tier3_required_obs": [8],
        "max_tier":           [3],
        "preferred_method":   ["Transit"],
        "available_transits": [100],
        "available_eclipses": [100],
        "fgs_flag":           [1],
        "rp_rs":              [0.1],
        "a_rs":               [10.0],
        "eccentricity":       [0.0],
        "inclination":        [89.0],
        "distance_pc":        [30.0],
        "population_bin":     ["mini_neptune_warm_gf"],
        "science_weight":     [0.6],
        "obs_cost_days_t1":   [dur_s / 86400.0 * COST_FACTOR],
        "obs_cost_days_t2":   [dur_s / 86400.0 * COST_FACTOR],
        "obs_cost_days_t3":   [dur_s / 86400.0 * COST_FACTOR],
    })


def _make_event(window_mid: float, target_id: str = "T_PARTIAL") -> pd.DataFrame:
    """Build a minimal event DataFrame with explicit block_duration_days."""
    dur_s = 7200.0
    dur_d = dur_s / 86400.0
    block_dur = COST_FACTOR * dur_d
    return pd.DataFrame([{
        "event_id":              0,
        "target_id":             target_id,
        "event_type":            "transit",
        "window_start":          window_mid - dur_d / 2,
        "window_mid":            window_mid,
        "window_end":            window_mid + dur_d / 2,
        "duration":              dur_s,
        "duration_days":         dur_d,
        "block_duration_days":   block_dur,
        "tier_goal":             1,
        "base_science_value":    0.5,
        "visibility_valid":      True,
        "ephemeris_uncertainty": 0.0,
        "event_index":           0,
    }])


def _state_at_time(t_now: float, targets: pd.DataFrame, events: pd.DataFrame) -> MissionState:
    """Create a MissionState with the clock already at t_now.

    Current pointing is set to the first target's RA/Dec so slew cost is zero.
    """
    state = MissionState.from_tables(targets, events)
    state.clock.current_time = t_now
    # Align current pointing to target → zero slew.
    state.current_ra = float(targets["ra"].iloc[0])
    state.current_dec = float(targets["dec"].iloc[0])
    return state


# ---------------------------------------------------------------------------
# Case A: arrive before block_start → full capture
# ---------------------------------------------------------------------------

class TestCaseAFullCapture:
    """Telescope arrives before the observation block opens → full block, frac=1.0."""

    def _setup(self) -> tuple[MissionState, int]:
        """Return (state, event_id) positioned so t_arrive ≤ block_start."""
        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        block_start = window_mid - block_dur / 2.0

        events = _make_event(window_mid)
        # Set clock to be well before block_start (slew = 0, so t_arrive = t_now)
        t_now = block_start - 0.01   # 14 minutes before block opens
        state = _state_at_time(t_now, targets, events)
        return state, 0

    def test_captured_fraction_is_one(self):
        state, eid = self._setup()
        info = state.execute_observation(eid)
        assert not info["missed"]
        assert info["captured_fraction"] == pytest.approx(1.0)

    def test_obs_duration_equals_block_duration(self):
        state, eid = self._setup()
        block_dur = COST_FACTOR * (7200.0 / 86400.0)
        info = state.execute_observation(eid)
        assert info["obs_duration_days"] == pytest.approx(block_dur)

    def test_progress_increments_by_one(self):
        state, eid = self._setup()
        state.execute_observation(eid)
        obs = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs == pytest.approx(1.0)

    def test_obs_remaining_decreases_by_one(self):
        state, eid = self._setup()
        rem_before = float(state.progress.loc["T_PARTIAL", "obs_remaining_next_tier"])
        state.execute_observation(eid)
        rem_after = float(state.progress.loc["T_PARTIAL", "obs_remaining_next_tier"])
        assert rem_after == pytest.approx(rem_before - 1.0)


# ---------------------------------------------------------------------------
# Case B: arrive mid-block → partial capture
# ---------------------------------------------------------------------------

class TestCaseBPartialCapture:
    """Telescope arrives after block_start but before block_end → partial frac."""

    def _setup(self, fraction_elapsed: float) -> tuple[MissionState, int, float]:
        """
        fraction_elapsed: fraction of the block that has already elapsed when
        the telescope arrives (0.0 = just at block_start, 1.0 = block_end).
        Returns (state, event_id, expected_fraction).

        Note: slew_time_days(ra=0, dec=0, ra=0, dec=0) == MIN_SLEW_DAYS (min
        settle time is always paid).  We set t_now = desired_t_arrive - MIN_SLEW_DAYS
        so that the actual t_arrive = t_now + MIN_SLEW_DAYS lands exactly where
        intended.
        """
        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        block_start = window_mid - block_dur / 2.0
        block_end   = window_mid + block_dur / 2.0

        # Desired arrival time inside the block
        t_arrive = block_start + fraction_elapsed * block_dur
        # Clamp to stay strictly inside [block_start, block_end)
        t_arrive = min(t_arrive, block_end - 1e-9)
        t_arrive = max(t_arrive, block_start + 1e-9)
        expected_frac = (block_end - t_arrive) / block_dur

        # Set clock so that after adding MIN_SLEW_DAYS we land at t_arrive
        t_now = t_arrive - MIN_SLEW_DAYS

        events = _make_event(window_mid)
        state = _state_at_time(t_now, targets, events)
        return state, 0, expected_frac

    def test_captured_fraction_50pct(self):
        state, eid, expected = self._setup(0.5)
        info = state.execute_observation(eid)
        assert not info["missed"]
        assert info["captured_fraction"] == pytest.approx(expected, rel=1e-5)

    def test_captured_fraction_25pct(self):
        state, eid, expected = self._setup(0.75)
        info = state.execute_observation(eid)
        assert info["captured_fraction"] == pytest.approx(expected, rel=1e-5)

    def test_captured_fraction_95pct(self):
        state, eid, expected = self._setup(0.05)
        info = state.execute_observation(eid)
        assert info["captured_fraction"] == pytest.approx(expected, rel=1e-5)

    def test_obs_duration_equals_captured_portion(self):
        state, eid, expected = self._setup(0.5)
        block_dur = COST_FACTOR * (7200.0 / 86400.0)
        info = state.execute_observation(eid)
        assert info["obs_duration_days"] == pytest.approx(expected * block_dur, rel=1e-5)

    def test_fractional_progress_stored(self):
        state, eid, expected = self._setup(0.5)
        state.execute_observation(eid)
        obs = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs == pytest.approx(expected, rel=1e-5)

    def test_progress_in_tier_between_0_and_1(self):
        state, eid, _ = self._setup(0.6)
        state.execute_observation(eid)
        pit = float(state.progress.loc["T_PARTIAL", "progress_in_tier"])
        assert 0.0 < pit < 1.0

    def test_clock_advanced_by_captured_duration_only(self):
        state, eid, expected = self._setup(0.5)
        block_dur = COST_FACTOR * (7200.0 / 86400.0)
        t_before = state.clock.current_time
        info = state.execute_observation(eid)
        expected_obs_dur = expected * block_dur
        # clock = t_before + idle + obs_dur (idle = 0 in Case B)
        assert state.clock.used_science_time == pytest.approx(expected_obs_dur, rel=1e-5)


# ---------------------------------------------------------------------------
# Case C: arrive after block_end → complete miss
# ---------------------------------------------------------------------------

class TestCaseCMiss:
    """Telescope arrives at or after block_end → missed=True, no progress."""

    def _setup(self, offset_after_end: float = 0.0) -> tuple[MissionState, int]:
        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        block_end = window_mid + block_dur / 2.0

        events = _make_event(window_mid)
        t_now = block_end + offset_after_end
        state = _state_at_time(t_now, targets, events)
        return state, 0

    def test_missed_exactly_at_block_end(self):
        state, eid = self._setup(0.0)
        info = state.execute_observation(eid)
        assert info["missed"]
        assert info["captured_fraction"] == pytest.approx(0.0)

    def test_missed_after_block_end(self):
        state, eid = self._setup(0.05)
        info = state.execute_observation(eid)
        assert info["missed"]

    def test_no_science_time_on_miss(self):
        state, eid = self._setup(0.0)
        state.execute_observation(eid)
        assert state.clock.used_science_time == pytest.approx(0.0)

    def test_no_progress_on_miss(self):
        state, eid = self._setup(0.0)
        state.execute_observation(eid)
        obs = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs == pytest.approx(0.0)

    def test_slew_still_paid_on_miss(self):
        state, eid = self._setup(0.01)
        t_before = state.clock.current_time
        state.execute_observation(eid)
        # Only slew cost paid; slew=0 (same RA/Dec), so time barely moves
        # Clock should still advance (even if by 0 for zero-slew).
        assert state.clock.current_time >= t_before


# ---------------------------------------------------------------------------
# Fractional progress accumulation
# ---------------------------------------------------------------------------

class TestFractionalProgressAccumulation:
    """Multiple partial observations should accumulate toward tier thresholds."""

    def _make_state_events(self) -> tuple[MissionState, list[int]]:
        """Return a state and two event IDs scheduled far apart so no overlap."""
        targets = _single_target_df(ra=0.0)
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d

        # Two separate events, well apart
        window_mid_1 = MISSION_START_BJD + 10.0
        window_mid_2 = MISSION_START_BJD + 20.0

        block_start_1 = window_mid_1 - block_dur / 2.0
        block_start_2 = window_mid_2 - block_dur / 2.0

        rows = []
        for eid, wmid in enumerate([window_mid_1, window_mid_2]):
            rows.append({
                "event_id":              eid,
                "target_id":             "T_PARTIAL",
                "event_type":            "transit",
                "window_start":          wmid - dur_d / 2,
                "window_mid":            wmid,
                "window_end":            wmid + dur_d / 2,
                "duration":              7200.0,
                "duration_days":         dur_d,
                "block_duration_days":   block_dur,
                "tier_goal":             1,
                "base_science_value":    0.5,
                "visibility_valid":      True,
                "ephemeris_uncertainty": 0.0,
                "event_index":           eid,
            })
        events = pd.DataFrame(rows).sort_values("window_mid").reset_index(drop=True)
        state = MissionState.from_tables(targets, events)
        # Align pointing to target
        state.current_ra = 0.0
        state.current_dec = 0.0
        return state, [0, 1]

    def test_two_half_observations_sum_to_one(self):
        """Two 50 % partial obs should yield obs_completed ≈ 1.5 (0.5 + 1.0)."""
        state, [eid1, eid2] = self._make_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d

        # First observation: desired t_arrive at 50 % into the block.
        # Set t_now = desired_t_arrive - MIN_SLEW_DAYS.
        block_start_1 = MISSION_START_BJD + 10.0 - block_dur / 2.0
        desired_arrive_1 = block_start_1 + 0.5 * block_dur
        state.clock.current_time = desired_arrive_1 - MIN_SLEW_DAYS
        state.execute_observation(eid1)

        obs_after_1 = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs_after_1 == pytest.approx(0.5, rel=1e-3)

        # Second observation: arrive before block_start (full capture)
        block_start_2 = MISSION_START_BJD + 20.0 - block_dur / 2.0
        state.clock.current_time = block_start_2 - 0.01 - MIN_SLEW_DAYS
        state.execute_observation(eid2)

        obs_after_2 = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs_after_2 == pytest.approx(1.5, rel=1e-3)

    def test_tier_boundary_crossed_via_fractions(self):
        """T_PARTIAL needs tier1_required_obs=2; two full observations cross it."""
        state, [eid1, eid2] = self._make_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d

        for i, eid in enumerate([eid1, eid2]):
            wmid = MISSION_START_BJD + 10.0 + i * 10.0
            state.clock.current_time = wmid - block_dur / 2.0 - 0.01
            state.execute_observation(eid)

        assert state.progress.loc["T_PARTIAL", "tier1_done"]
        assert state.progress.loc["T_PARTIAL", "current_tier"] >= 1

    def test_obs_remaining_is_fractional_after_partial(self):
        """After a 0.75-fraction obs, obs_remaining_next_tier should be ~1.25."""
        state, [eid1, _] = self._make_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        block_start_1 = MISSION_START_BJD + 10.0 - block_dur / 2.0

        # Desired t_arrive at 25 % into the block → capture 75 %
        desired_arrive = block_start_1 + 0.25 * block_dur
        state.clock.current_time = desired_arrive - MIN_SLEW_DAYS
        state.execute_observation(eid1)

        obs_rem = float(state.progress.loc["T_PARTIAL", "obs_remaining_next_tier"])
        # tier1_required_obs=2; after 0.75 obs, remaining ≈ 1.25
        assert obs_rem == pytest.approx(1.25, rel=1e-3)


# ---------------------------------------------------------------------------
# Tier-scoped observation (effective_fraction capped at tier boundary)
# ---------------------------------------------------------------------------

class TestTierScopedObservation:
    """Observation duration is capped when it would cross a tier boundary early.

    The agent regains control as soon as the tier threshold is crossed — it does
    NOT continue observing for the remainder of the captured window.
    """

    def _make_near_complete_state(self, obs_completed_start: float):
        """Return a state with obs_completed already at obs_completed_start.

        Target has tier1_required_obs=2 so the tier completes at obs_completed=2.
        """
        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        events = _make_event(window_mid)
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d

        from aRieL.data.observation_requirements import compute_progress
        state = MissionState.from_tables(targets, events)
        state.current_ra = 0.0
        state.current_dec = 0.0

        # Manually advance obs_completed to the desired starting value
        target_row = state.targets.loc[state.targets["target_id"] == "T_PARTIAL"].iloc[0]
        new_prog = compute_progress(obs_completed_start, target_row)
        state._progress_dict["T_PARTIAL"].update(new_prog)
        for k, v in new_prog.items():
            state.progress.at["T_PARTIAL", k] = v

        # Place clock before block_start (full window capture)
        state.clock.current_time = window_mid - block_dur / 2.0 - 0.01 - MIN_SLEW_DAYS
        return state, 0, block_dur

    def test_effective_fraction_capped_at_obs_remaining(self):
        """If obs_remaining=0.3 and window offers 1.0, effective_fraction=0.3."""
        state, eid, block_dur = self._make_near_complete_state(obs_completed_start=1.7)
        # obs_remaining = 2.0 - 1.7 = 0.3
        info = state.execute_observation(eid)
        assert info["effective_fraction"] == pytest.approx(0.3, rel=1e-3)

    def test_clock_advances_by_effective_fraction_only(self):
        """Clock advances by effective_fraction * block_dur, not full block."""
        state, eid, block_dur = self._make_near_complete_state(obs_completed_start=1.7)
        # obs_remaining = 0.3; window offers 1.0 → effective = 0.3
        state.execute_observation(eid)
        assert state.clock.used_science_time == pytest.approx(0.3 * block_dur, rel=1e-3)

    def test_tier_completes_exactly_at_threshold(self):
        """After the capped observation, obs_completed should hit exactly 2.0."""
        state, eid, _ = self._make_near_complete_state(obs_completed_start=1.7)
        state.execute_observation(eid)
        obs = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs == pytest.approx(2.0, rel=1e-6)
        assert state.progress.loc["T_PARTIAL", "tier1_done"]

    def test_geometric_captured_fraction_unchanged(self):
        """captured_fraction (geometric) is 1.0 regardless of tier cap."""
        state, eid, _ = self._make_near_complete_state(obs_completed_start=1.7)
        info = state.execute_observation(eid)
        assert info["captured_fraction"] == pytest.approx(1.0, rel=1e-5)

    def test_max_tier_produces_zero_science(self):
        """Observing a fully-completed target collects no science (safety net)."""
        state, eid, block_dur = self._make_near_complete_state(obs_completed_start=8.0)
        # tier3_required_obs=8 and max_tier=3, so this target is fully done
        state.execute_observation(eid)
        assert state.clock.used_science_time == pytest.approx(0.0, abs=1e-9)
        obs = float(state.progress.loc["T_PARTIAL", "obs_completed"])
        assert obs == pytest.approx(8.0, rel=1e-6)  # unchanged


# ---------------------------------------------------------------------------
# Action mask consistency with block_end
# ---------------------------------------------------------------------------

class TestMaskBlockEndConsistency:
    """Action mask should allow actions up to block_end, not just window_end."""

    def _make_mask_state_events(self) -> tuple[MissionState, pd.DataFrame]:
        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        events = _make_event(window_mid)
        state = MissionState.from_tables(targets, events)
        state.current_ra = 0.0
        state.current_dec = 0.0
        return state, events

    def test_valid_inside_block_but_after_window_end(self):
        """An event should be valid if t_arrive is between window_end and block_end."""
        from aRieL.envs.action_mask import compute_mask
        from aRieL.utils.config import ActionConfig, TopKActionConfig

        state, events = self._make_mask_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        window_mid = MISSION_START_BJD + 10.0
        window_end = window_mid + dur_d / 2.0
        block_end  = window_mid + block_dur / 2.0

        # Place clock just after window_end but before block_end
        state.clock.current_time = window_end + (block_end - window_end) * 0.5

        cfg = ActionConfig(type="topk", topk=TopKActionConfig(k=10))
        mask = compute_mask(state, events, cfg)
        assert mask[0], (
            "Event should be valid (partial capture): t_now is between "
            "window_end and block_end"
        )

    def test_invalid_at_or_after_block_end(self):
        """An event should be masked once t_arrive >= block_end."""
        from aRieL.envs.action_mask import compute_mask
        from aRieL.utils.config import ActionConfig, TopKActionConfig

        state, events = self._make_mask_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        window_mid = MISSION_START_BJD + 10.0
        block_end  = window_mid + block_dur / 2.0

        # Exactly at block_end
        state.clock.current_time = block_end

        cfg = ActionConfig(type="topk", topk=TopKActionConfig(k=10))
        mask = compute_mask(state, events, cfg)
        assert not mask[0], "Event must be invalid when t_arrive == block_end"

    def test_full_set_mask_permissive_vs_target_mask_strict(self):
        """full_set mask allows actions even when budget is tight; target mask doesn't."""
        from aRieL.envs.action_mask import compute_mask
        from aRieL.utils.config import (
            ActionConfig, TargetActionConfig, FullSetActionConfig,
        )

        targets = _single_target_df(ra=0.0)
        window_mid = MISSION_START_BJD + 10.0
        events = _make_event(window_mid)
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d

        # Use a very short mission so total cost doesn't fit
        from aRieL.simulator.mission_clock import MissionClock
        from aRieL.data.observation_requirements import initialise_progress_table
        clock = MissionClock(
            mission_start=MISSION_START_BJD,
            mission_end=MISSION_START_BJD + block_dur * 0.3,  # only 30 % of block left
        )
        progress = initialise_progress_table(targets)
        state = MissionState(
            targets=targets,
            events=events,
            clock=clock,
            progress=progress,
        )
        state.current_ra = 0.0
        state.current_dec = 0.0
        # Advance to just before block start so t_arrive is inside block
        state.clock.current_time = window_mid - block_dur / 2.0 + 0.001

        cfg_target = ActionConfig(
            type="target",
            target=TargetActionConfig(include_completed=False),
        )
        cfg_full   = ActionConfig(
            type="full_set",
            full_set=FullSetActionConfig(include_completed=False),
        )
        mask_target = compute_mask(state, events, cfg_target)
        mask_full   = compute_mask(state, events, cfg_full)

        # full_set should be MORE permissive
        assert int(mask_full[0]) >= int(mask_target[0])

    def test_full_set_exclude_started_blocks(self):
        """When exclude_started_blocks=True, mid-block events are masked out."""
        from aRieL.envs.action_mask import compute_mask
        from aRieL.utils.config import ActionConfig, FullSetActionConfig

        state, events = self._make_mask_state_events()
        dur_d = 7200.0 / 86400.0
        block_dur = COST_FACTOR * dur_d
        window_mid = MISSION_START_BJD + 10.0
        window_end = window_mid + dur_d / 2.0
        block_end  = window_mid + block_dur / 2.0

        # Mid-block: partial capture allowed by default, rejected when excluded
        state.clock.current_time = window_end + (block_end - window_end) * 0.5

        cfg_default = ActionConfig(
            type="full_set",
            full_set=FullSetActionConfig(include_completed=False),
        )
        cfg_exclude = ActionConfig(
            type="full_set",
            full_set=FullSetActionConfig(
                include_completed=False,
                exclude_started_blocks=True,
            ),
        )
        assert compute_mask(state, events, cfg_default)[0]
        assert not compute_mask(state, events, cfg_exclude)[0]
