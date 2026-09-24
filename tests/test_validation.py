"""Tests for aRieL.validation."""

from __future__ import annotations

import pytest

from aRieL import ArielEnv, make_demo_targets
from aRieL.rewards import compute_reward
from aRieL.utils.config import ObservationConfig, RewardConfig
from aRieL.validation import (
    validate_env_observation,
    validate_observation_config,
    validate_reward,
    validate_reward_fn,
)


def test_stock_reward_passes_including_speed():
    report = validate_reward(cfg="default", fn=compute_reward, n_timing=500)
    assert report.ok, report.summary_str()
    speed = next(c for c in report.checks if c.name == "speed")
    assert speed.status == "ok"


def test_reward_config_rejects_unknown_mode():
    cfg = RewardConfig(overhead_penalty_mode="nope")
    report = validate_reward(cfg=cfg)
    assert not report.ok
    assert any(c.name == "overhead_mode" and c.status == "fail" for c in report.checks)


def test_slow_reward_fails_speed_gate():
    def slow(info):
        s = 0.0
        for i in range(80_000):
            s += i * 1e-12
        return s

    report = validate_reward_fn(slow, n_timing=30, mean_budget_us=50.0, p95_budget_us=100.0)
    assert not report.ok
    assert any(c.name == "speed" and c.status == "fail" for c in report.checks)


def test_non_finite_reward_fails():
    def bad(info):
        return float("nan")

    report = validate_reward_fn(bad, check_speed=False)
    assert not report.ok
    assert any(c.name == "battery" and c.status == "fail" for c in report.checks)


def test_unknown_event_feature_fails():
    cfg = ObservationConfig(event_features=["totally_fake_feature"])
    report = validate_observation_config(cfg)
    assert not report.ok
    assert any("unknown event" in c.message for c in report.checks if c.status == "fail")


def test_live_demo_env_obs_ok():
    env = ArielEnv.from_preset("demo", targets=make_demo_targets(6))
    report = validate_env_observation(env, n_steps=4)
    env.close()
    assert report.ok, report.summary_str()


def test_crashing_reward_gives_hint():
    def brittle(step_result, cfg, bin_totals, before, after, hosts=None):
        return float(step_result["this_key_does_not_exist"])

    report = validate_reward_fn(brittle, check_speed=False)
    assert not report.ok
    crash = next(c for c in report.checks if c.name == "sparse_step_result")
    assert crash.status == "fail"
    assert "get" in crash.hint.lower() or "Required keys" in crash.hint
