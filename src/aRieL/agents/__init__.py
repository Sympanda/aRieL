"""RL agents and training helpers (optional ``[train]`` extra).

Install::

    pip install -e ".[train]"

Then::

    from aRieL.agents import FullSetISABPolicy, make_training_envs
"""

from __future__ import annotations


def _require_train_deps() -> None:
    try:
        import torch  # noqa: F401
        import stable_baselines3  # noqa: F401
        import sb3_contrib  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training / policy imports require the optional train extra.\n"
            '  pip install -e ".[train]"\n'
            f"Original error: {exc}"
        ) from exc


_require_train_deps()

from aRieL.agents.policies.event_attention_policy import (  # noqa: E402
    ArielTransformerNet,
    ArielTransformerPolicy,
)
from aRieL.agents.policies.full_set_attention_policy import (  # noqa: E402
    FullSetSelfAttentionPolicy,
)
from aRieL.agents.policies.full_set_isab_policy import (  # noqa: E402
    FullSetISABNet,
    FullSetISABPolicy,
)
from aRieL.agents.policies.mlp_scorer import ArielMlpPolicy  # noqa: E402
from aRieL.agents.ppo_masked import make_masked_env, make_training_envs  # noqa: E402
from aRieL.agents.pretrained import (  # noqa: E402
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

__all__ = [
    "ArielMlpPolicy",
    "ArielTransformerNet",
    "ArielTransformerPolicy",
    "FullSetISABNet",
    "FullSetISABPolicy",
    "FullSetSelfAttentionPolicy",
    "RLAgentWrapper",
    "PAPER_N_TARGETS",
    "PAPER_SEED",
    "PAPER_TIER1_COMPLETED",
    "PAPER_TIER2_COMPLETED",
    "PAPER_TIER3_COMPLETED",
    "load_default_model",
    "make_checkpoint_smoke_env",
    "make_masked_env",
    "make_paper_env",
    "make_training_envs",
    "run_inference_episode",
]
