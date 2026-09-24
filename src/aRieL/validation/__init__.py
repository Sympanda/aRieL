"""User-facing validators for custom rewards and observation spaces.

Typical usage
-------------
    from aRieL.validation import validate_reward, validate_env_observation
    from aRieL.rewards import compute_reward

    print(validate_reward(cfg="default", fn=compute_reward).summary_str())

    env = ArielEnv.from_preset("demo", targets=make_demo_targets(8))
    print(validate_env_observation(env).summary_str())

See ``docs/CUSTOMISING.md`` for the reward and observation contracts.
"""

from aRieL.validation.obs_checks import (
    OBS_REQUIREMENTS,
    validate_env_observation,
    validate_obs_builder,
    validate_observation_config,
    validate_observation_dict,
)
from aRieL.validation.report import CheckResult, ValidationError, ValidationReport
from aRieL.validation.reward_checks import (
    DEFAULT_REWARD_MEAN_US,
    DEFAULT_REWARD_P95_US,
    DEFAULT_REWARD_WARN_MEAN_US,
    validate_reward,
    validate_reward_config,
    validate_reward_fn,
)
from aRieL.validation.step_result import (
    REQUIRED_STEP_RESULT_KEYS,
    STEP_RESULT_REQUIREMENTS,
    make_missed_step_result,
    make_step_result,
)

__all__ = [
    "CheckResult",
    "DEFAULT_REWARD_MEAN_US",
    "DEFAULT_REWARD_P95_US",
    "DEFAULT_REWARD_WARN_MEAN_US",
    "OBS_REQUIREMENTS",
    "REQUIRED_STEP_RESULT_KEYS",
    "STEP_RESULT_REQUIREMENTS",
    "ValidationError",
    "ValidationReport",
    "make_missed_step_result",
    "make_step_result",
    "validate_env_observation",
    "validate_obs_builder",
    "validate_observation_config",
    "validate_observation_dict",
    "validate_reward",
    "validate_reward_config",
    "validate_reward_fn",
]
