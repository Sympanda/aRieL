"""Inference checks for the published ISAB checkpoint.

The short test always runs (with the train extra) on the synthetic demo
catalogue, using the paper observation layout. Tests that need the real
Mission Candidate Sample skip when that CSV is not on the search path.
The full 3.5-year reproduction is marked ``slow``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from aRieL.agents.pretrained import (  # noqa: E402
    PAPER_LIFETIME_DAYS,
    PAPER_N_TARGETS,
    PAPER_SEED,
    PAPER_TIER1_COMPLETED,
    PAPER_TIER2_COMPLETED,
    PAPER_TIER3_COMPLETED,
    load_default_model,
    make_checkpoint_smoke_env,
    make_paper_env,
    run_inference_episode,
)
from aRieL.agents.rl_agent import RLAgentWrapper  # noqa: E402
from aRieL.data.load_catalogue import find_mcs_csv  # noqa: E402


def _require_mcs():
    path = find_mcs_csv()
    if path is None:
        pytest.skip(
            "MCS CSV not on the search path. Set ARIEL_DATA to run paper-catalogue tests."
        )
    return path


def test_legacy_checkpoint_import_resolves_to_ariel():
    import aRieL
    import importlib

    legacy = importlib.import_module(
        "ariel_rl.agents.policies.full_set_isab_policy"
    )
    assert legacy.FullSetISABPolicy is aRieL.agents.FullSetISABPolicy


def test_published_checkpoint_exists():
    from aRieL.agents.pretrained import default_model_path

    assert default_model_path().is_file()
    assert default_model_path().stat().st_size > 1_000_000


def test_default_model_inference_respects_mask():
    env = make_checkpoint_smoke_env()
    model = load_default_model()
    agent = RLAgentWrapper(model, deterministic=True, name="ISAB")
    try:
        obs, info = env.reset(seed=PAPER_SEED)
        assert "planets" in obs
        actions = []
        for _ in range(12):
            action = agent.act(obs, info)
            mask = info["action_mask"]
            assert 0 <= action < len(mask)
            assert bool(mask[action]), "policy chose a masked action"
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert len(actions) >= 1

        env2 = make_checkpoint_smoke_env()
        obs_a, info_a = env2.reset(seed=PAPER_SEED)
        obs_b, info_b = env.reset(seed=PAPER_SEED)
        a = agent.act(obs_a, info_a)
        b = agent.act(obs_b, info_b)
        assert a == b
        env2.close()
    finally:
        env.close()


def test_paper_env_loads_user_catalogue():
    _require_mcs()
    env = make_paper_env(lifetime_days=30.0)
    try:
        assert len(env._targets) == PAPER_N_TARGETS
        assert env.cfg.action.type == "full_set"
        assert env.cfg.action.full_set.k_filter == 72
        obs, info = env.reset(seed=PAPER_SEED)
        assert "planets" in obs
        assert info["action_mask"].shape[0] == env.n_actions
        assert int(info["action_mask"].sum()) > 0
    finally:
        env.close()


@pytest.mark.slow
def test_reproduces_paper_tier_counts():
    """Full 3.5-year rollout. Enable with ``ARIEL_RUN_PAPER=1 pytest -m slow``."""
    if os.environ.get("ARIEL_RUN_PAPER") != "1" and not os.environ.get("PYTEST_SLOW"):
        pytest.skip("set ARIEL_RUN_PAPER=1 to roll out the full paper mission")
    _require_mcs()

    stats = run_inference_episode(lifetime_days=PAPER_LIFETIME_DAYS, seed=PAPER_SEED)
    assert stats.n_initial_targets == PAPER_N_TARGETS
    assert stats.tier1_completed == PAPER_TIER1_COMPLETED
    assert stats.tier2_completed == PAPER_TIER2_COMPLETED
    assert stats.tier3_completed == PAPER_TIER3_COMPLETED
