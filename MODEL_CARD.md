# Model card

## Model

- Name: `isab_default`
- File: `src/aRieL/models/isab_default.zip`
- Architecture: ISAB actor-critic (`FullSetISABPolicy`) trained with MaskablePPO
- Widths stored in the checkpoint: `d_model=128`, `n_heads=4`, `n_isab_layers=3`, `n_inducing=24`
- Training step: 3,106,752 environment steps
- Package version required to load it: aRieL 0.1.0b1, with the `train` extra (PyTorch and sb3-contrib). The checkpoint records Stable-Baselines3 2.9.0, PyTorch 2.12.1, and Python 3.11.15.

## Environment

Frozen configuration: `src/aRieL/models/env_config.yaml`, loaded by `make_paper_env()`.

- Action space: `full_set`, `k_filter=72`, `n_max=1000`, `exclude_started_blocks=true`
- Mission: start BJD 2462867.5, lifetime 1278.375 days, `cost_factor=2.5`
- Slew: 1 deg/min, minimum 120 s, maximum 7200 s
- Catalogue growth: disabled
- Reward configuration: the `default` preset (per-step Tier 1/2 completion 0.5, Tier 3 completion 0.6, Tier 1 milestone and terminal bonuses as in that file)
- Target catalogue: user-supplied `Ariel_MCS_Known_2025-08-18` MCS, 814 targets after preprocessing (`science_weight_floor=0.3`). The CSV is not bundled; see `docs/CATALOGUE.md`.

## Reported result

Deterministic evaluation, seed 42, one full mission:

| Tier | Completed |
| --- | --- |
| 1 | 696 / 814 |
| 2 | 306 |
| 3 | 129 |

Reproduce with `ARIEL_RUN_PAPER=1 pytest -m slow`.

## Training hardware

Training was performed on a MacBook Pro with an Apple M3 processor using PyTorch's Metal Performance Shaders (MPS) backend.

## Intended use

Load the checkpoint with `load_default_model()` and roll it out in `make_paper_env()` to reproduce the published scheduling policy on this catalogue and configuration. `make_paper_env()` needs a local copy of the Mission Candidate Sample.

## Known limitations

- Ariel field-of-regard and solar-exclusion geometry are not implemented. `_check_visibility` marks every event visible.
- The policy matches this observation layout only. Changing feature width or order requires retraining.
- The result above is one deterministic episode on the paper catalogue. It is not a claim about on-orbit Ariel operations.
- The package schedules observations for a catalogue you supply. It does not build the Ariel Mission Candidate Sample.
