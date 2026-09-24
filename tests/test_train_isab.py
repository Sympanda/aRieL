"""Training stack imports (skipped if [train] deps are missing)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")


def test_agents_package_exports_isab():
    from aRieL.agents import FullSetISABPolicy, make_training_envs

    assert FullSetISABPolicy is not None
    assert make_training_envs is not None


def test_isab_policy_matches_planet_feature_width():
    from dataclasses import replace

    from sb3_contrib import MaskablePPO

    from aRieL import make_demo_targets
    from aRieL.agents import FullSetISABPolicy, make_masked_env
    from aRieL.envs.planet_feature_builder import N_PLANET_FEATURES
    from aRieL.utils.config import ActionConfig, FullSetActionConfig, load_preset

    cfg = load_preset("demo", reward="default")
    cfg = replace(
        cfg,
        action=ActionConfig(
            type="full_set",
            full_set=FullSetActionConfig(k_filter=6, n_max=8),
        ),
    )
    env = make_masked_env(cfg, seed=0, targets=make_demo_targets(8))
    model = MaskablePPO(
        FullSetISABPolicy,
        env,
        policy_kwargs={
            "d_model": 32,
            "n_heads": 2,
            "n_isab_layers": 1,
            "n_inducing": 2,
        },
        n_steps=8,
        batch_size=8,
        verbose=0,
    )
    n_pf = env.observation_space["planets"].shape[-1]
    assert n_pf == N_PLANET_FEATURES
    assert model.policy.isab_net.planet_proj.in_features == n_pf
    env.close()
