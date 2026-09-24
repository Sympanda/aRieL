"""Validate stock + deliberately broken custom reward/obs setups."""

from aRieL import ArielEnv, make_demo_targets
from aRieL.rewards import compute_reward
from aRieL.utils.config import ObservationConfig, default_env_config
from aRieL.validation import (
    validate_env_observation,
    validate_observation_config,
    validate_reward,
)


def main() -> None:
    print("1) Stock reward preset + stock compute_reward (includes SPEED gate)\n")
    report = validate_reward(cfg="default", fn=compute_reward)
    print(report.summary_str())
    print()

    print("2) Live demo env observation / action masks\n")
    env = ArielEnv.from_preset("demo", targets=make_demo_targets(8))
    print(validate_env_observation(env, n_steps=5).summary_str())
    env.close()
    print()

    print("3) ObservationConfig with an unknown feature (expect FAIL)\n")
    bad_obs = ObservationConfig(event_features=["slew_time_days", "not_a_real_feature"])
    print(validate_observation_config(bad_obs).summary_str())
    print()

    print("4) Intentionally slow reward (expect SPEED FAIL)\n")

    def slow_reward(info):
        # Artificial stall — do not do this in real rewards.
        total = 0.0
        for i in range(50_000):
            total += i * 1e-12
        return float(total) + float(info.science_weight)

    print(validate_reward(fn=slow_reward, n_timing=50).summary_str())
    print()

    print("5) Simple RewardInfo toy reward\n")
    from aRieL.rewards import t1_bonus
    print(validate_reward(fn=t1_bonus).summary_str())
    print()

    print("6) Default ObservationConfig from EnvConfig\n")
    print(validate_observation_config(default_env_config().observation).summary_str())


if __name__ == "__main__":
    main()
