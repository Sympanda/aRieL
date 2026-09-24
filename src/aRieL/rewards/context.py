"""Simple reward interface for customisation / tutorials.

Design
------
The stock ``compute_reward`` is powerful but heavy for someone who just wants
to try ``"give +1 on Tier-1 completion"``.

Custom rewards should take a single :class:`RewardInfo` built by the env after
each observation.  That keeps the contract small and avoids digging into
``MissionState``.

    def my_reward(info: RewardInfo) -> float:
        if info.tier_after > info.tier_before:
            return 1.0
        return -info.idle_days

Pass it as ``ArielEnv(..., reward_fn=my_reward)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class RewardInfo:
    """Flat snapshot of what happened on one env step.

    Enough for most custom rewards; nothing here requires pandas.
    """

    # Outcome
    missed: bool
    target_id: str
    host_id: str
    population_bin: str
    science_weight: float
    period_days: float

    # Tier / progress
    tier_before: int
    tier_after: int
    progress_before: float
    progress_after: float
    captured_fraction: float

    # Time costs (days)
    slew_days: float
    idle_days: float
    obs_duration_days: float
    total_cost_days: float

    # Light catalogue context
    bin_totals: Mapping[str, int]
    bin_observed_before: Mapping[str, int]
    bin_observed_after: Mapping[str, int]
    host_tier1_before: Mapping[str, int]

    @property
    def tier_completed(self) -> int | None:
        """Tier number just completed, or ``None`` if no tier boundary crossed."""
        if self.tier_after > self.tier_before:
            return int(self.tier_after)
        return None

    @property
    def overhead_days(self) -> float:
        return float(self.slew_days + self.idle_days)


RewardFn = Callable[[RewardInfo], float]


def reward_info_from_step(
    step_result: Mapping[str, Any],
    *,
    bin_totals: Mapping[str, int],
    bin_observed_before: Mapping[str, int],
    bin_observed_after: Mapping[str, int],
    host_tier1_before: Mapping[str, int] | None = None,
) -> RewardInfo:
    """Build a :class:`RewardInfo` from a simulator ``step_result`` dict."""
    return RewardInfo(
        missed=bool(step_result.get("missed", False)),
        target_id=str(step_result.get("target_id", "")),
        host_id=str(step_result.get("host_id", "")),
        population_bin=str(step_result.get("population_bin", "")),
        science_weight=float(step_result.get("science_weight", 0.5)),
        period_days=float(step_result.get("period", 0.0)),
        tier_before=int(step_result.get("tier_before", 0)),
        tier_after=int(step_result.get("tier_after", 0)),
        progress_before=float(step_result.get("progress_before", 0.0)),
        progress_after=float(step_result.get("progress_after", 0.0)),
        captured_fraction=float(step_result.get("captured_fraction", 0.0)),
        slew_days=float(step_result.get("slew_days", 0.0)),
        idle_days=float(step_result.get("idle_days", 0.0)),
        obs_duration_days=float(step_result.get("obs_duration_days", 0.0)),
        total_cost_days=float(
            step_result.get(
                "total_cost_days",
                float(step_result.get("obs_duration_days", 0.0))
                + float(step_result.get("slew_days", 0.0))
                + float(step_result.get("idle_days", 0.0)),
            )
        ),
        bin_totals=dict(bin_totals),
        bin_observed_before=dict(bin_observed_before),
        bin_observed_after=dict(bin_observed_after),
        host_tier1_before=dict(host_tier1_before or {}),
    )
