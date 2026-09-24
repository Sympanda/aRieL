"""Reinforcement-learning environment for Ariel observation scheduling."""

from aRieL._checkpoint_alias import install as _install_checkpoint_alias

_install_checkpoint_alias()

from aRieL.envs import ArielEnv
from aRieL.data import load_mcs, build_target_table
from aRieL.data.demo import make_demo_targets
from aRieL.simulator.event_backend import DynamicBackend
from aRieL.utils.config import (
    EnvConfig,
    default_env_config,
    load_env_config,
    load_preset,
    load_reward_config,
    list_env_presets,
    list_reward_presets,
)

__version__ = "0.1.0b1"

__all__ = [
    "ArielEnv",
    "DynamicBackend",
    "EnvConfig",
    "build_target_table",
    "default_env_config",
    "list_env_presets",
    "list_reward_presets",
    "load_env_config",
    "load_mcs",
    "load_preset",
    "load_reward_config",
    "make_demo_targets",
    "__version__",
]
