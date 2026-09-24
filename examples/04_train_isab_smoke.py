"""Short MaskablePPO + ISAB smoke train on the demo catalogue (no MCS needed).

Requires: pip install -e ".[train]"

This is intentionally tiny (few thousand steps) to verify the training stack.
For real runs use a full MCS catalogue, ``full_set`` preset, and 1e5–1e6+ steps.
"""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO

from aRieL import make_demo_targets
from aRieL.agents import FullSetISABPolicy, make_training_envs
from aRieL.utils.config import load_preset


def main() -> None:
    targets = make_demo_targets(12)
    cfg = load_preset("demo", reward="default")
    # demo preset is top-k; switch to a small full_set action space for ISAB
    from dataclasses import replace
    from aRieL.utils.config import ActionConfig, FullSetActionConfig

    cfg = replace(
        cfg,
        action=ActionConfig(
            type="full_set",
            full_set=FullSetActionConfig(
                include_completed=False,
                cache_static=True,
                k_filter=8,
                n_max=12,
            ),
        ),
    )

    env = make_training_envs(cfg, n_envs=1, seed=0, targets=targets)
    out = Path("outputs/isab_smoke")
    out.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO(
        FullSetISABPolicy,
        env,
        policy_kwargs={
            "d_model": 64,
            "n_heads": 4,
            "n_isab_layers": 1,
            "n_inducing": 4,
        },
        n_steps=64,
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=1_024)
    model.save(out / "final_model")
    print(f"saved → {out / 'final_model.zip'}")
    env.close()


if __name__ == "__main__":
    main()
