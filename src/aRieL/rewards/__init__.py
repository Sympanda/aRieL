"""Reward computation for the Ariel RL environment."""

from aRieL.rewards.compute_reward import (
    check_milestone_reward,
    check_t1_milestone_reward,
    check_t2_milestone_reward,
    compute_reward,
    compute_terminal_reward,
)
from aRieL.rewards.context import RewardFn, RewardInfo, reward_info_from_step
from aRieL.rewards.simple import (
    SIMPLE_REWARDS,
    progress_minus_idle,
    science_weighted_t1,
    t1_bonus,
)

# Backward-compatible alias: older call sites used check_milestone_reward for T1.
check_milestone_reward_t1 = check_t1_milestone_reward

__all__ = [
    "SIMPLE_REWARDS",
    "RewardFn",
    "RewardInfo",
    "check_milestone_reward",
    "check_milestone_reward_t1",
    "check_t1_milestone_reward",
    "check_t2_milestone_reward",
    "compute_reward",
    "compute_terminal_reward",
    "progress_minus_idle",
    "reward_info_from_step",
    "science_weighted_t1",
    "t1_bonus",
]
