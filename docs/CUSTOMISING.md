# Customising aRieL

Edit or subclass the existing modules. There is no plugin registry.

After a change, check it with `aRieL.validation` before a long training run. Changing feature width or order changes the observation space, so an existing checkpoint will not match and the policy must be retrained.

## Custom reward

`reward_fn` replaces the stock per-step reward. It receives one `RewardInfo` and returns a finite float.

`RewardInfo` (`aRieL.rewards.context`) includes:

- `missed`, `target_id`, `host_id`, `population_bin`, `science_weight`, `period_days`
- `tier_before`, `tier_after`, `progress_before`, `progress_after`, `captured_fraction`
- `slew_days`, `idle_days`, `obs_duration_days`, `total_cost_days`
- `tier_completed` and `overhead_days`

```python
from aRieL import ArielEnv, make_demo_targets
from aRieL.rewards import RewardInfo

def t1_bonus(info: RewardInfo) -> float:
    if info.tier_completed == 1:
        return 1.0
    return 0.0

env = ArielEnv.from_preset("demo", targets=make_demo_targets(8), reward_fn=t1_bonus)
```

A custom per-step function does not remove configured milestone or terminal rewards. `ArielEnv.step` still adds the Tier 1 and Tier 2 milestone bonuses and the terminal bonus from the reward configuration. Set those bonuses to zero in the reward YAML if the custom function should be the whole return.

Keep the function cheap. It runs on every step. Avoid pandas, disk access, and loops over the catalogue inside it.

```python
from aRieL.validation import validate_reward

report = validate_reward(cfg="default", fn=t1_bonus)
print(report.summary_str())
report.raise_if_failed()
```

`validate_reward` fails when the mean call exceeds 100 µs or the 95th percentile exceeds 250 µs. A mean above 50 µs is a warning. The stock per-step reward is about 1–2 µs.

## Custom observation

### Full-set / ISAB

Edit `src/aRieL/envs/planet_feature_builder.py`.

`PLANET_FEATURE_NAMES` is `STATIC_FEATURE_NAMES` then `DYNAMIC_FEATURE_NAMES`. That order is the last axis of `obs["planets"]`, with shape `(n_actions, N_PLANET_FEATURES)`. Mission-level inputs are `obs["global"]`, shape `(n_global,)`.

Add a name to the static or dynamic list, write the value in the matching builder, and keep `PLANET_FEATURE_NAMES` as the concatenation of those lists.

### Top-K

Edit `src/aRieL/envs/observation_builder.py` and the feature lists on `ObservationConfig` (or `observation.event_features` / `observation.global_features` in the environment YAML).

`obs["events"]` has shape `(k, n_event_features)`. Columns follow `event_features` in listed order. `obs["global"]` follows `global_features`, then any population-bin fractions when `include_population_bin_fractions` is set. New names must be computed in the builder and listed in `ALL_EVENT_FEATURES` or `ALL_GLOBAL_FEATURES` in `src/aRieL/utils/config.py`.

### Check

```python
from aRieL import ArielEnv, make_demo_targets
from aRieL.validation import validate_env_observation

env = ArielEnv.from_preset("full_set", targets=make_demo_targets(8))
print(validate_env_observation(env).summary_str())
```

The check requires float32 finite arrays whose shapes match `observation_space`, and a 1-D boolean `info["action_mask"]` of length `n_actions`.

Feature position is part of the policy input. Changing width or order requires retraining.

## Custom policy

Start from `src/aRieL/agents/policies/tiny_linear_policy.py`. Subclass `MaskableActorCriticPolicy` and implement `forward`, `evaluate_actions`, and `predict_values`.

- Full-set policies read `obs["planets"]`.
- Top-K policies read `obs["events"]`.
- Mission-level inputs are `obs["global"]`.
- Illegal actions must be masked, typically by filling those logits with `-inf` using `action_masks`.

Pass the class directly to `MaskablePPO`:

```python
from sb3_contrib import MaskablePPO
from aRieL.agents.policies.tiny_linear_policy import TinyFullSetPolicy

model = MaskablePPO(TinyFullSetPolicy, train_env, n_steps=64, batch_size=64)
```

The released architecture is `FullSetISABPolicy` in `src/aRieL/agents/policies/full_set_isab_policy.py`. It reads its feature widths from `observation_space`, so a newly constructed network follows a changed feature width. A checkpoint trained on the old width does not.
