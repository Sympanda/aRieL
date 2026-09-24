"""Tiny example rewards for tutorials — deliberately basic.

Use these as templates, not as science-grade objectives.
"""

from __future__ import annotations

from aRieL.rewards.context import RewardFn, RewardInfo


def t1_bonus(info: RewardInfo) -> float:
    """+1 when a planet newly reaches Tier 1; small miss penalty."""
    if info.missed:
        return -0.1
    if info.tier_completed == 1:
        return 1.0
    return 0.0


def progress_minus_idle(info: RewardInfo) -> float:
    """Reward progress; gently penalise waiting."""
    if info.missed:
        return -0.2
    gained = max(0.0, info.progress_after - info.progress_before)
    return gained - 0.05 * info.idle_days


def science_weighted_t1(info: RewardInfo) -> float:
    """Tier-1 completion scaled by catalogue ``science_weight``."""
    if info.missed:
        return -0.1
    if info.tier_completed == 1:
        return float(info.science_weight)
    return 0.0


SIMPLE_REWARDS: dict[str, RewardFn] = {
    "t1_bonus": t1_bonus,
    "progress_minus_idle": progress_minus_idle,
    "science_weighted_t1": science_weighted_t1,
}
