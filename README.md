# aRieL

aRieL is a reinforcement-learning environment for scheduling Ariel transit and eclipse observations from a supplied target catalogue.

This is unofficial research software. It is not an ESA product and it is not an Ariel Mission product.

Ariel field-of-regard and solar-exclusion constraints are not implemented. Every generated event is treated as visible.

## What is included

- Gymnasium scheduling environment (`aRieL.ArielEnv`)
- Synthetic demo catalogue (`make_demo_targets()`)
- Reward and environment configuration used for the published model
- Published ISAB checkpoint
- Baseline policies and evaluation utilities

The Ariel Mission Candidate Sample is not included. The published checkpoint used [`Ariel_MCS_Known_2025-08-18.csv`](https://github.com/arielmission-space/Mission_Candidate_Sample/blob/main/target_lists/Ariel_MCS_Known_2025-08-18.csv). Place that file at `./data/raw/MCS.csv`, or set `ARIEL_DATA` to its directory. See [docs/CATALOGUE.md](docs/CATALOGUE.md).

## Installation

From this directory:

```bash
conda env create -f environment.yml
conda activate ariel-rl
```

Or, with pip:

```bash
pip install -e ".[train]"
```

`pip install -e .` installs the environment only. The `train` extra adds PyTorch, Stable-Baselines3, and sb3-contrib, which are required to load or train a policy. `plot` adds matplotlib. `dev` adds pytest together with the `plot` and `train` libraries.

## Quick start

```python
from aRieL import ArielEnv, make_demo_targets

env = ArielEnv.from_preset("demo", targets=make_demo_targets(8))
obs, info = env.reset(seed=0)
action = int(info["action_mask"].nonzero()[0][0])
obs, reward, terminated, truncated, info = env.step(action)
print(reward, info["mission_summary"]["tier1_completed"])
env.close()
```

## Published model

The released checkpoint is ISAB + MaskablePPO, trained for 3,106,752 steps with the frozen configuration in `src/aRieL/models/env_config.yaml` and the `default` reward. Deterministic evaluation uses seed 42. Details, tier counts, and limitations are in [MODEL_CARD.md](MODEL_CARD.md).

Loading the checkpoint does not need the Mission Candidate Sample. Reproducing the paper episode does. `make_paper_env()` reads a local MCS via `ARIEL_DATA` or `./data/raw/MCS.csv`.

```python
from aRieL.agents import RLAgentWrapper, load_default_model, make_paper_env

agent = RLAgentWrapper(load_default_model(), deterministic=True)
env = make_paper_env()
obs, info = env.reset(seed=42)
action = agent.act(obs, info)
obs, reward, terminated, truncated, info = env.step(action)
```

## Customising aRieL

These are source-level extension points. aRieL does not provide a separate plugin or registration framework. Changes to policy input features normally require retraining.

| Change | Where |
| --- | --- |
| Reward | Pass `reward_fn(info: RewardInfo) -> float`. See `aRieL/rewards/`. Configured milestone and terminal bonuses still apply unless they are disabled in the reward configuration. |
| Policy | Pass a custom `MaskableActorCriticPolicy` to `MaskablePPO`. `src/aRieL/agents/policies/tiny_linear_policy.py` is the minimal template. The policy must respect the action mask. |
| Full-set observations | Edit `src/aRieL/envs/planet_feature_builder.py`. |
| Top-K observations | Edit `src/aRieL/envs/observation_builder.py` and the feature lists in the environment configuration. |

See [docs/CUSTOMISING.md](docs/CUSTOMISING.md).

## Target catalogues

```python
from aRieL import build_target_table

targets = build_target_table("path/to/MCS.csv")
```

With no path, this searches `ARIEL_DATA`, then `./data/raw/`, then `./data/`. Column format and attribution are in [docs/CATALOGUE.md](docs/CATALOGUE.md).

## Tests

```bash
pytest
```

The full 3.5-year checkpoint reproduction is optional and is not part of the default run:

```bash
ARIEL_RUN_PAPER=1 pytest -m slow
```

## Citation

Cite this software with [CITATION.cff](CITATION.cff).

The paper that reports the published checkpoint does not yet have a DOI recorded in this repository. Add that identifier to `CITATION.cff` when it is available.

Publications that use the Mission Candidate Sample should also cite the sources listed in [docs/CATALOGUE.md](docs/CATALOGUE.md). The MIT licence applies to the code. It does not cover that catalogue.
