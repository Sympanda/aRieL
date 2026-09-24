"""Synthetic ``step_result`` fixtures and the required-key contract.

``compute_reward`` (and any custom drop-in) reads keys from the dict returned
by ``MissionState.execute_observation``.  Custom rewards should tolerate
missing optional keys via ``.get`` defaults, but must not crash on the
canonical set below.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Keys the stock reward reads (required for a full success path).
REQUIRED_STEP_RESULT_KEYS: tuple[str, ...] = (
    "missed",
    "idle_days",
    "obs_duration_days",
    "slew_days",
    "total_cost_days",
    "science_weight",
    "population_bin",
    "tier_before",
    "tier_after",
    "progress_before",
    "progress_after",
    "period",
    "host_id",
)

# Common extras produced by the simulator (reward may ignore these).
OPTIONAL_STEP_RESULT_KEYS: tuple[str, ...] = (
    "target_id",
    "event_id",
    "event_type",
    "captured_fraction",
    "effective_fraction",
    "block_duration_days",
    "window_mid",
)

STEP_RESULT_REQUIREMENTS = """
step_result contract
--------------------
A reward callable is invoked roughly as:

    reward = fn(
        step_result,          # dict from MissionState.execute_observation
        cfg,                  # RewardConfig
        bin_totals,           # dict[str, int]
        bin_observed_before,  # dict[str, int]
        bin_observed_after,   # dict[str, int]
        host_tier1_counts,    # dict[str, int] | None
    ) -> float

Must return a finite Python float (or 0-dim numpy scalar).
Must be fast: default budget is tens of microseconds per call — see
validate_reward_speed().  Never allocate large arrays or touch disk/network.

Required keys on a successful observation (missed=False):
""" + "\n".join(f"  - {k}" for k in REQUIRED_STEP_RESULT_KEYS)


def make_step_result(**overrides: Any) -> dict:
    """Return a canonical successful-observation step_result."""
    base = {
        "missed": False,
        "idle_days": 0.05,
        "obs_duration_days": 0.20,
        "slew_days": 0.02,
        "total_cost_days": 0.27,
        "science_weight": 0.55,
        "population_bin": "demo_bin_a",
        "tier_before": 0,
        "tier_after": 1,
        "progress_before": 0.0,
        "progress_after": 1.0,
        "period": 12.0,
        "host_id": "HOST-01",
        "target_id": "DEMO-01",
        "event_id": 0,
        "event_type": "transit",
        "captured_fraction": 1.0,
        "effective_fraction": 1.0,
        "block_duration_days": 0.20,
    }
    base.update(overrides)
    return base


def make_missed_step_result(**overrides: Any) -> dict:
    base = make_step_result(
        missed=True,
        tier_after=0,
        progress_after=0.0,
        captured_fraction=0.0,
        effective_fraction=0.0,
    )
    base.update(overrides)
    return base


def default_bin_context() -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """(bin_totals, before, after, host_tier1_counts) for a T1 completion."""
    totals = {"demo_bin_a": 8, "demo_bin_b": 4}
    before = {"demo_bin_a": 0, "demo_bin_b": 1}
    after = {"demo_bin_a": 1, "demo_bin_b": 1}
    hosts = {"HOST-01": 0, "HOST-02": 1}
    return totals, before, after, hosts


def step_result_battery() -> list[dict]:
    """Diverse cases for correctness + timing loops."""
    cases = [
        make_step_result(),
        make_missed_step_result(),
        make_step_result(tier_before=1, tier_after=2, progress_before=0.8, progress_after=1.0),
        make_step_result(tier_before=2, tier_after=3, period=365.0, science_weight=0.9),
        make_step_result(idle_days=1.5, slew_days=0.5, total_cost_days=2.2, tier_after=0, progress_after=0.2),
        make_step_result(missed=False, tier_before=0, tier_after=0, progress_before=0.1, progress_after=0.4),
    ]
    return [deepcopy(c) for c in cases]
