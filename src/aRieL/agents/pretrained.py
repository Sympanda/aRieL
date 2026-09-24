"""Published ISAB checkpoint and the environment it was trained in.

The weights are the fixed-MCS run used in the paper
(``PAPER_isab_survey_k72_m24_v1``, checkpoint at 3,106,752 environment steps).
Action space: full-set, ``k_filter=72``, 24 inducing points.
Reward: the ``default`` preset (survey-first).
Catalogue: the user-supplied Ariel MCS (814 targets after preprocessing in the paper run).
The CSV is not bundled. Set ``ARIEL_DATA`` or place it at ``./data/raw/MCS.csv``.

Full 3.5-year mission, one episode, seed 42, reported tier completions:

* Tier 1: 696 / 814 (85.5% of the catalogue)
* Tier 2: 306
* Tier 3: 129

Loading the policy needs the ``[train]`` extra (PyTorch + sb3-contrib).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aRieL.utils.paths import bundled_model_dir

# Paper table (fixed MCS, 3.5 years, seed 42).
PAPER_TIER1_COMPLETED = 696
PAPER_TIER2_COMPLETED = 306
PAPER_TIER3_COMPLETED = 129
PAPER_N_TARGETS = 814
PAPER_SEED = 42
PAPER_LIFETIME_DAYS = 1278.375


def default_model_path() -> Path:
    return bundled_model_dir() / "isab_default.zip"


def paper_env_config_path() -> Path:
    return bundled_model_dir() / "env_config.yaml"


def load_paper_env_config(lifetime_days: float | None = None):
    """Env config saved with the published checkpoint."""
    from aRieL.utils.config import load_env_config

    cfg = load_env_config(paper_env_config_path())
    if lifetime_days is not None:
        cfg = replace(
            cfg,
            mission=replace(cfg.mission, lifetime_days=float(lifetime_days)),
        )
    return cfg


def load_default_model():
    """Load the published MaskablePPO / ISAB policy.

    Requires ``pip install -e \".[train]\"``.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading the published model needs the train extra:\n"
            '  pip install -e ".[train]"'
        ) from exc

    path = default_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Bundled model not found at {path}")
    return MaskablePPO.load(str(path))


def make_paper_env(lifetime_days: float | None = None):
    """ArielEnv on a user-supplied MCS with the checkpoint's env config.

    Raises ``FileNotFoundError`` when the catalogue is not on the search path.
    """
    from aRieL.data.load_catalogue import find_mcs_csv, mcs_not_found_message
    from aRieL.data.preprocess_targets import build_target_table
    from aRieL.envs.env import ArielEnv

    csv_path = find_mcs_csv()
    if csv_path is None:
        raise FileNotFoundError(mcs_not_found_message())
    cfg = load_paper_env_config(lifetime_days=lifetime_days)
    targets = build_target_table(
        csv_path,
        science_weight_floor=cfg.reward.science_weight_floor,
    )
    return ArielEnv(config=cfg, targets=targets)


def make_checkpoint_smoke_env(lifetime_days: float = 30.0):
    """Paper observation layout on a synthetic catalogue.

    Does not need the Mission Candidate Sample. The published checkpoint's
    global vector has length 26: the nine configured global features, plus
    one fraction for each population bin that contains at least
    ``min_bin_targets`` planets. This helper builds 17 such bins so a forward
    pass matches that width. Tier counts from this environment are not the
    published result.
    """
    from aRieL.data.demo import make_demo_targets
    from aRieL.envs.env import ArielEnv

    cfg = load_paper_env_config(lifetime_days=lifetime_days)
    n_base = len(cfg.observation.global_features)
    n_bins = 26 - n_base
    per_bin = max(int(cfg.observation.min_bin_targets), 1)
    targets = make_demo_targets(n_bins * per_bin)
    targets["population_bin"] = [
        f"bin_{i // per_bin:02d}" for i in range(len(targets))
    ]
    return ArielEnv(config=cfg, targets=targets)


def run_inference_episode(lifetime_days: float | None = None, seed: int = PAPER_SEED):
    """Roll out the published policy for one episode. Returns ``EpisodeStats``."""
    from aRieL.agents.rl_agent import RLAgentWrapper
    from aRieL.evaluation.compare_runs import run_episode

    env = make_paper_env(lifetime_days=lifetime_days)
    model = load_default_model()
    agent = RLAgentWrapper(model, deterministic=True, name="ISAB")
    stats = run_episode(env, agent, seed=seed)
    env.close()
    return stats
