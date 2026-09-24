"""
Mission clock: tracks the current BJD time and remaining budget.

The clock advances when an observation is executed.  It accounts for:
  - Observation duration (T14 or E14)
  - Slew time from previous pointing to new target
  - Any fixed overhead (settle, guide-star acquisition)

The clock does *not* know about target science values or tier progress —
those are tracked in mission_state.MissionState.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aRieL.data.schemas import (
    MISSION_END_BJD,
    MISSION_LIFETIME_DAYS,
    MISSION_START_BJD,
    OBS_OVERHEAD_DAYS_BASE,
)


@dataclass
class MissionClock:
    """Stateful mission clock.

    Attributes
    ----------
    current_time:
        Current BJD time.
    mission_start:
        BJD of mission start.
    mission_end:
        BJD of mission end (fixed).
    used_science_time:
        Total days spent on observations (excluding slew).
    used_slew_time:
        Total days spent slewing.
    used_overhead_time:
        Total days spent on fixed per-obs overhead.
    """

    mission_start: float = field(default=MISSION_START_BJD)
    mission_end: float = field(default=MISSION_END_BJD)
    current_time: float = field(init=False)
    used_science_time: float = field(default=0.0, init=False)
    used_slew_time: float = field(default=0.0, init=False)
    used_idle_time: float = field(default=0.0, init=False)
    used_overhead_time: float = field(default=0.0, init=False)
    n_observations: int = field(default=0, init=False)
    n_missed: int = field(default=0, init=False)  # events we tried but arrived late

    def __post_init__(self) -> None:
        self.current_time = self.mission_start

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def remaining_time(self) -> float:
        """Days remaining in the mission."""
        return max(0.0, self.mission_end - self.current_time)

    @property
    def elapsed_time(self) -> float:
        """Days elapsed since mission start."""
        return self.current_time - self.mission_start

    @property
    def fraction_elapsed(self) -> float:
        """Fraction of mission lifetime consumed (0–1).

        Uses the *actual* ``mission_end – mission_start`` rather than a global
        constant so curriculum episodes with shorter horizons are handled
        correctly.
        """
        mission_length = self.mission_end - self.mission_start
        if mission_length <= 0:
            return 1.0
        return min(1.0, float(self.elapsed_time / mission_length))

    @property
    def mission_over(self) -> bool:
        return self.current_time >= self.mission_end

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def advance(
        self,
        obs_duration_days: float,
        slew_days: float = 0.0,
        idle_days: float = 0.0,
        overhead_days: float = OBS_OVERHEAD_DAYS_BASE,
        n_obs: int = 1,
    ) -> float:
        """Advance the clock by one observation campaign.

        The canonical campaign structure is:
            slew  →  idle (wait for block_start)  →  N × [observe + inter-transit gap]

        For a single-transit action n_obs=1 and this is identical to the old
        behaviour.  For a tier-completion campaign n_obs counts the number of
        consecutive transits executed.

        Parameters
        ----------
        obs_duration_days:
            Total science time = n_obs × block_duration_days.
        slew_days:
            Time to slew from current pointing to target.
        idle_days:
            All non-science time after slewing: initial wait for first block +
            inter-transit gaps between subsequent blocks.
        overhead_days:
            Fixed per-observation overhead (settle, guide-star, etc.).  Applied
            once per *action*, not per transit.
        n_obs:
            Number of individual transit/eclipse observations in this campaign.

        Returns
        -------
        float
            Total time consumed in days.
        """
        total = slew_days + idle_days + obs_duration_days + overhead_days
        self.current_time += total
        self.used_science_time += obs_duration_days
        self.used_slew_time += slew_days
        self.used_idle_time += idle_days
        self.used_overhead_time += overhead_days
        self.n_observations += max(n_obs, 1)
        return total

    def skip_to(self, bjd: float) -> float:
        """Jump the clock forward to *bjd* (e.g. wait for next event window).

        Returns the wait time in days.  Raises if *bjd* < current_time.
        """
        if bjd < self.current_time:
            raise ValueError(
                f"Cannot skip backwards: bjd={bjd:.4f} < current={self.current_time:.4f}"
            )
        wait = bjd - self.current_time
        self.current_time = bjd
        return wait

    def record_miss(self) -> None:
        """Register that the agent tried to observe but arrived after window_end."""
        self.n_missed += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def can_fit(self, total_cost_days: float) -> bool:
        """Return True if *total_cost_days* fits in remaining time."""
        return total_cost_days <= self.remaining_time

    def time_until(self, bjd: float) -> float:
        """Days from now until *bjd* (can be negative if in the past)."""
        return bjd - self.current_time

    def reset(self) -> None:
        """Reset clock to mission start."""
        self.current_time = self.mission_start
        self.used_science_time = 0.0
        self.used_slew_time = 0.0
        self.used_idle_time = 0.0
        self.used_overhead_time = 0.0
        self.n_observations = 0
        self.n_missed = 0

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of clock state."""
        return {
            "current_time":       self.current_time,
            "mission_start":      self.mission_start,
            "mission_end":        self.mission_end,
            "remaining_time":     self.remaining_time,
            "elapsed_time":       self.elapsed_time,
            "fraction_elapsed":   self.fraction_elapsed,
            "used_science_time":  self.used_science_time,
            "used_slew_time":     self.used_slew_time,
            "used_idle_time":     self.used_idle_time,
            "used_overhead_time": self.used_overhead_time,
            "n_observations":     self.n_observations,
            "n_missed":           self.n_missed,
        }
