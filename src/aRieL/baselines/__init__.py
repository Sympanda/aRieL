"""
Baseline schedulers for the Ariel RL environment.

All baselines share the same interface as an RL agent: they receive
``(obs, info)`` from the env and return a scalar action index.  This
makes comparisons with learned agents apples-to-apples.

Available baselines
-------------------
RandomValid         — uniform random over valid actions
GreedyValue         — highest base_science_value among valid actions
GreedyBalanced      — highest science_weight × (1 + progress_in_tier)
EarliestDeadline    — earliest window_end among valid actions
SmartGreedy         — highest science_weight × progress_bonus / (slew + duration)
HillClimbingGreedy  — linear scoring, weights optimised by greedy hill-climbing
"""

from aRieL.baselines.random_valid      import RandomValid
from aRieL.baselines.greedy_value      import GreedyValue
from aRieL.baselines.greedy_balanced   import GreedyBalanced
from aRieL.baselines.earliest_deadline import EarliestDeadline
from aRieL.baselines.smart_greedy      import SmartGreedy
from aRieL.baselines.hill_climbing     import HillClimbingGreedy
from aRieL.baselines.base              import BaselineAgent

ALL_BASELINES: dict[str, type] = {
    "random":            RandomValid,
    "greedy_value":      GreedyValue,
    "greedy_balanced":   GreedyBalanced,
    "earliest_deadline": EarliestDeadline,
    "smart_greedy":      SmartGreedy,
    "hill_climbing":     HillClimbingGreedy,
}

__all__ = [
    "BaselineAgent",
    "RandomValid",
    "GreedyValue",
    "GreedyBalanced",
    "EarliestDeadline",
    "SmartGreedy",
    "HillClimbingGreedy",
    "ALL_BASELINES",
]
