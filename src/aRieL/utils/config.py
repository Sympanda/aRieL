"""
Configuration system: YAML files → frozen dataclass hierarchy.

All tuneable constants live here.  Hardcoded values in the simulator
modules serve only as fallback defaults; the env always passes explicit
values derived from the loaded config.

Usage
-----
    from aRieL.utils.config import load_env_config, EnvConfig

    cfg = load_env_config("configs/env/simple.yaml")
    print(cfg.slew.rate_deg_per_min)   # 1.0
    print(cfg.action.topk.k)           # 50

Structure
---------
    EnvConfig
    ├── MissionConfig          mission timing and budget
    ├── SlewConfig             telescope slew model
    ├── ActionConfig           action space type + type-specific sub-config
    │   ├── TopKActionConfig   K, sort order
    │   └── TargetActionConfig N-target full-set config
    ├── ObservationConfig      which features to include, normalisation
    └── RewardConfig           weights per component (populated later)
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from aRieL.data.schemas import (
    MISSION_END_BJD,
    MISSION_LIFETIME_DAYS,
    MISSION_START_BJD,
    COST_FACTOR,
)
from aRieL.simulator.slew import (
    MAX_SLEW_S,
    MIN_SLEW_S,
    SLEW_RATE_DEG_PER_MIN,
)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionConfig:
    """Mission-level constants."""
    start_bjd: float = MISSION_START_BJD
    lifetime_days: float = MISSION_LIFETIME_DAYS
    cost_factor: float = COST_FACTOR
    overhead_days_per_obs: float = 0.0  # fixed per-obs overhead beyond T14

    # Global cap on the maximum tier any target can reach, regardless of what
    # the catalogue says.  Useful for ablation studies:
    #   max_tier_cap=1 → T1-only run (baseline feasibility)
    #   max_tier_cap=2 → ignore T3 targets
    #   max_tier_cap=3 → use catalogue values (default)
    max_tier_cap: int = 3


@dataclass(frozen=True)
class CatalogueGrowthConfig:
    """Mid-episode injection of newly confirmed planets (TPC → MCS).

    When ``enabled``, planets are drawn from the TPC pool and inserted at
    mission BJDs according to ``schedule_mode``:

    * ``"fixed"`` (default) — one schedule, built once from ``schedule_seed``
      (or ``EnvConfig.seed``).  Every episode sees the **same** planets at the
      **same** times.  Recommended starting point: no extra non-stationarity.
    * ``"per_episode"`` — rebuild the schedule from ``reset(seed)`` each
      episode.  Adds catalogue stochasticity (useful later as domain
      randomisation); expect to need more training samples.

    Coverage milestones / terminal / reported tier rates use the **initial**
    catalogue size as denominator, so completing injected planets can push
    coverage above 100 %.  Live observation features still use the growing
    catalogue (fraction ≤ 1).
    """
    enabled: bool = False
    #: Path to TPC (or Ariel_MCS_TPCs) CSV.  Empty → auto-discover under data/raw/.
    tpc_csv_path: str = ""
    mean_per_month: float = 15.0
    std_per_month: float = 5.0
    month_days: float = 30.4375  # mean Gregorian month length
    #: ``"fixed"`` | ``"per_episode"`` — see class docstring.
    schedule_mode: str = "fixed"
    #: RNG seed for the injection schedule.  ``-1`` → use ``EnvConfig.seed``.
    #: In ``fixed`` mode this is the sole source of schedule randomness.
    #: In ``per_episode`` mode it is ignored (``reset(seed)`` is used instead).
    schedule_seed: int = -1


@dataclass(frozen=True)
class AnnualT3RevisitConfig:
    """Eval-only: undo progress on deep-programme planets each mission year.

    When ``enabled``, at every multiple of ``period_days`` after mission start
    (365.25 d, 730.5 d, …) every target with ``max_tier >= min_max_tier`` has
    its observation progress reset to zero (T0).  They re-enter the action
    mask / active set so the scheduler must re-observe them.

    Intended for counterfactual “yearly T3 revisit” figures with a frozen
    policy — not for training.
    """
    enabled: bool = False
    period_days: float = 365.25
    #: Reset planets whose catalogue ``max_tier`` is at least this value
    #: (default 3 → Tier-3 programme targets only).
    min_max_tier: int = 3


@dataclass(frozen=True)
class SlewConfig:
    """Telescope slew model parameters.

    Note: the true Ariel slew performance is not yet published.
    ``rate_deg_per_min`` is the key uncertain parameter — keep it in config.
    """
    rate_deg_per_min: float = SLEW_RATE_DEG_PER_MIN
    min_slew_seconds: float = MIN_SLEW_S
    max_slew_seconds: float = MAX_SLEW_S


@dataclass(frozen=True)
class TopKActionConfig:
    """Config for the ``topk`` action space.

    The agent chooses from the next *k* upcoming events sorted by
    ``sort_by`` (e.g. chronological order).
    """
    k: int = 50
    sort_by: str = "window_mid"   # column in the event table to sort by
    #: When True, top-K is built from unstarted, fully catchable events only
    #: (``DynamicBackend.candidates(..., unstarted_only=True)`` + slew filter).
    #: Started / mid-block events never enter the action set.
    exclude_started_blocks: bool = False


@dataclass(frozen=True)
class TargetActionConfig:
    """Config for the ``target`` action space.

    The agent picks one of the N targets directly; the env automatically
    schedules the next available event for that target.  Targets with no
    remaining events are masked out.
    """
    include_completed: bool = False   # mask out fully-completed targets?


@dataclass(frozen=True)
class FullSetActionConfig:
    """Config for the ``full_set`` action space (Phase 3).

    The agent sees all N targets simultaneously, each described by the full
    per-planet feature vector from ``planet_feature_builder``.  The action
    is a target index 0…N-1; the env schedules the next available event for
    that target.

    This replaces top-K filtering with a learned ranking / attention over
    the full catalogue.  It requires a set-based policy architecture
    (e.g. transformer).

    Note: computing dynamic features for all ~800 targets at every step is
    O(N), typically < 5 ms.  Enable ``cache_static`` to pre-compute static
    features at reset() and only recompute dynamic features each step.
    """
    include_completed: bool = False
    cache_static: bool = True   # pre-compute static features at reset()
    #: Fixed action-space size.  The observation is padded to (n_max, F) with
    #: zero tokens so the policy sees a constant-shape input independent of
    #: how many targets remain active.  Set to 0 to use len(targets) as-is
    #: (backward-compatible default; no padding added).
    #: For the full Ariel catalogue (~2000 targets) set n_max=2000.
    n_max: int = 0
    #: Fast pre-filter: at each step, keep only the K planets whose
    #: first-reachable event has the soonest window_mid (same principle as
    #: top-K event mode — "these windows are closing soonest").  The ISAB
    #: then decides which of the K to actually observe.
    #: Reduces ISAB token count from N_max (~814) to k_filter, giving a
    #: ~(N/K)× GPU speedup with minimal quality loss for k_filter ≥ 64.
    #: Set to 0 to disable (default: pass all active planets up to n_max).
    #: When k_filter > 0 it overrides n_max as the action-space size.
    k_filter: int = 0
    #: When True, planets whose first-reachable observation block has already
    #: opened (``t_now >= block_start``) are omitted from the ISAB token set
    #: and masked out — no partial mid-block observations.  Candidate selection
    #: skips to the next future event for that target when possible.
    exclude_started_blocks: bool = False


@dataclass(frozen=True)
class ActionConfig:
    """Selects which action space to use and carries its sub-config."""
    type: str = "topk"                            # "topk" | "target" | "full_set"
    topk: TopKActionConfig = field(default_factory=TopKActionConfig)
    target: TargetActionConfig = field(default_factory=TargetActionConfig)
    full_set: FullSetActionConfig = field(default_factory=FullSetActionConfig)


# Per-event feature names the observation builder understands.
# Listed here as a reference; the YAML can select a subset.
ALL_EVENT_FEATURES: list[str] = [
    "slew_time_days",                 # angular slew cost in days
    "window_urgency_norm",            # fraction of window already elapsed (0=fresh, 1=closing)
    "duration_days",                  # raw transit / eclipse duration (T14)
    "block_duration_days",            # full observation block = 2.5 × T14
    "total_time_cost_days",           # slew + idle + block_duration
    "capture_fraction",               # fraction of block capturable if chosen now (0–1)
    "progress_in_tier",               # fraction of obs completed toward next tier
    "obs_remaining_next_tier_norm",   # equivalent obs still needed, normalised by max possible
    "base_science_value",             # catalogue SNR-derived value [0, 1]
    "science_weight",                 # catalogue priority weight [0, 1]
    "planet_radius_norm",             # planet radius / 20 Re
    "planet_temperature_norm",        # equilibrium temp / 3000 K
    "planet_mass_norm",               # planet mass / 4000 Me
    "stellar_temperature_norm",       # stellar Teff / 10000 K
    "stellar_metallicity",            # [Fe/H] (negative values allowed)
    "tier_goal_norm",                 # tier_goal / 3
    "event_type_binary",              # 0 = transit, 1 = eclipse
    "days_to_block_end_norm",         # (block_end - t_now) days; block_end = mid + 1.25×T14
]

ALL_GLOBAL_FEATURES: list[str] = [
    "fraction_elapsed",               # mission time consumed
    "tier1_fraction",                 # T1-complete targets / total
    "tier2_fraction",                 # T2-complete targets / total
    "tier3_fraction",                 # T3-complete targets / total
    "used_science_fraction",          # science time / mission length
    "used_slew_fraction",             # slew time / mission length
    "used_idle_fraction",             # idle/wait time / mission length
    "n_observations_norm",            # cumulative obs count / 5000
    "n_completed_targets_norm",       # fraction of targets fully completed (at max tier)
]


@dataclass(frozen=True)
class ObservationConfig:
    """Controls which features appear in the agent observation."""
    event_features: list[str] = field(default_factory=lambda: list(ALL_EVENT_FEATURES))
    global_features: list[str] = field(default_factory=lambda: list(ALL_GLOBAL_FEATURES))
    include_population_bin_fractions: bool = True
    #: Only include population bins with at least this many targets.
    #: Tiny bins (< threshold) are almost never observed in a single episode
    #: and would be constant-zero features that waste model capacity.
    min_bin_targets: int = 10
    normalise: bool = True


@dataclass(frozen=True)
class RewardConfig:
    """Reward component weights.

    Tier completion bonuses (sparse, per-step)
    ------------------------------------------
    ``tier1/2/3_completion`` — base reward when a target reaches that tier.
    Scaled by ``science_weight × (1 + diversity)`` at runtime so rare,
    under-represented targets are worth proportionally more.

    Suggested starting ratio: T1=1, T2=3, T3=10.  Since tier progression is
    *cumulative* (T2 requires finishing T1 first), a single target taken all
    the way to T3 earns 1+3+10=14 × scale.

    Progress shaping (dense, per-step)
    -----------------------------------
    ``progress_weight`` — small reward proportional to the increase in
    ``progress_in_tier`` on each valid observation.  Provides a learning
    signal in the long stretches between tier completions.  Also scaled by
    ``science_weight × (1 + diversity)``.

    ``near_completion_scale`` — multiplier applied to the progress reward
    when ``progress_in_tier > near_completion_threshold``.  Encourages the
    agent to finish targets that are almost complete rather than abandoning
    them.  E.g., scale=3.0 makes the last 30% of a tier worth 3× more per
    step.

    Efficiency reward (dense, per-step)
    ------------------------------------
    ``efficiency_weight`` — reward proportional to
    ``obs_duration / (obs_duration + slew_duration)``.  Penalises wasted
    slew time without needing a separate slew penalty weight.

    Coverage milestone bonuses (sparse, one-shot)
    ----------------------------------------------
    ``t1_milestone_fractions`` — list of T1-coverage fractions at which a
    one-shot bonus is fired.  Defaults to [0.25, 0.5, 0.75, 0.90, 1.0].
    Fractions are relative to the **initial** catalogue size, so with
    catalogue growth they may exceed 1.0.  ``t1_milestone_step_above_one``
    auto-extends the list every 10 % above 100 % up to the scheduled ceiling.

    ``t1_milestone_bonus`` — bonus for each intermediate milestone (fractions
    other than exactly 1.0, including overflow >100% unless
    ``t1_overflow_milestone_bonus`` is set).

    ``t1_final_milestone_bonus`` — bonus for the 100% milestone specifically.
    If not set (0.0), falls back to ``2 × t1_milestone_bonus`` for backward
    compatibility.  Set explicitly to make the final T1 completion the
    standout reward signal.

    Terminal episode bonus
    ----------------------
    ``t1_terminal_weight`` — applied at episode end (or mission completion)
    as ``t1_terminal_weight × (tier1_fraction)^t1_terminal_power``.  The
    quadratic default (power=2) means the last 10 % of T1 coverage is
    disproportionately valuable, pushing the agent toward full coverage.

    Missed-event penalty
    --------------------
    ``miss_penalty`` — subtracted when the agent arrives after ``window_end``.

    Invalid-action penalty
    ----------------------
    ``invalid_action_penalty`` — applied immediately by ArielEnv when the
    agent picks a masked action (clock does not advance).
    """
    # --- sparse tier completion bonuses ---
    tier1_completion: float = 1.0
    tier2_completion: float = 3.0
    tier3_completion: float = 10.0

    # --- dense per-step shaping ---
    progress_weight:            float = 0.3
    efficiency_weight:          float = 0.5
    near_completion_threshold:  float = 0.7   # progress_in_tier above which boost applies
    near_completion_scale:      float = 3.0   # multiplier on progress reward near a tier boundary

    # --- diversity multiplier ---
    # Scales how much more attractive an under-observed population bin is vs a saturated one.
    # At max_multiplier=5.0 a fully unseen bin is 5× more rewarding than a full bin.
    # Was hardcoded at 2.0 in transformer_v1; increase to 5.0 to close the coverage gap.
    diversity_multiplier_max: float = 5.0

    # --- rarity / difficulty bonus ---
    # Added on every successful observation: rarity_weight × (period/period_ref)² / tier_worked
    # Rewards long-period targets that are costly to miss (they won't come around again soon).
    # The quadratic squashing keeps short-period targets near-zero while pushing yearly orbits
    # up to ~rarity_weight in magnitude.  Dividing by tier_worked (1/2/3) softens the bonus
    # for higher tiers which are already well-rewarded by tier_completion weights.
    rarity_weight:          float = 0.5    # scale; at period=period_ref this equals rarity_weight/tier
    rarity_period_ref_days: float = 365.0  # period [days] that maps to difficulty = 1.0

    # --- coverage milestone bonuses (one-shot per episode) ---
    # Fractions are relative to the **initial** catalogue size (MCS at reset).
    # With catalogue growth, completing injected planets can exceed 1.0.
    t1_milestone_fractions:       tuple = (0.25, 0.5, 0.75, 0.90, 1.0)
    t1_milestone_bonus:           float = 20.0   # intermediate milestones (!= 100%)
    t1_final_milestone_bonus:     float = 0.0    # exactly 100%; 0 → use 2×t1_milestone_bonus
    #: Auto-extend milestones every this step above 1.0 up to the scheduled
    #: catalogue ceiling (e.g. 1.1, 1.2, …).  Set 0 to disable overflow milestones.
    t1_milestone_step_above_one:  float = 0.1
    #: Bonus for each milestone strictly above 100%.  0 → use t1_milestone_bonus.
    t1_overflow_milestone_bonus:  float = 0.0

    # --- terminal episode bonus ---
    # Also uses initial-catalogue denominator (can exceed 1.0 with growth).
    t1_terminal_weight: float = 50.0   # scale of end-of-episode T1-coverage bonus
    t1_terminal_power:  float = 2.0    # exponent; >1 rewards near-full coverage most

    # --- random-baseline subtraction ---
    # When True, ``random_baseline_per_step`` is subtracted from the reward on
    # every valid (non-missed) observation.  This centres the reward around zero
    # for average random-agent behaviour, making the advantage signal proportional
    # to improvement *over* random rather than absolute reward magnitude.
    # Calibrate ``random_baseline_per_step`` from a quick random-agent run:
    #   total_random_reward / n_steps  (typically ~4.0 for the default reward config).
    subtract_random_baseline:  bool  = False
    random_baseline_per_step:  float = 4.0

    # --- overhead penalty (slew + idle) ---
    # Two modes controlled by ``overhead_penalty_mode``:
    #
    # "fraction" (original):
    #   overhead_fraction = (slew + idle) / total_step_days
    #   penalty = weight × fraction^power
    #   Problem: long observations dilute the fraction, making deep T2/T3 obs
    #   appear "efficient" even when the idle wait is identical in absolute time.
    #
    # "absolute" (recommended):
    #   penalty = weight × (slew_days + idle_days)^power
    #   Correct opportunity-cost interpretation: every wasted day costs the same
    #   regardless of how long the subsequent observation is.
    #   At weight=5, power=2:
    #     0.1 d overhead → 0.05   (negligible — normal slew)
    #     0.5 d overhead → 1.25   (¼ T1 completion — noticeable)
    #     1.0 d overhead → 5.0    (= 1 T1 completion — expensive)
    #     2.0 d overhead → 20.0   (= 4 T1 completions — very painful)
    #
    # Set overhead_penalty_weight=0.0 to disable entirely.
    overhead_penalty_mode:   str   = "fraction"  # "absolute" | "fraction"
    overhead_penalty_weight: float = 0.0   # disabled by default; set in reward YAMLs
    overhead_penalty_power:  float = 2.0   # quadratic by default
    # Legacy: kept for backward compatibility with old configs; prefer overhead_penalty.
    idle_penalty_per_day: float = 0.005

    # --- population coverage potential (U_pop) ---
    # Replaces the per-step diversity multiplier with a marginal coverage signal:
    #   r_coverage = U(s_{t+1}) − U(s_t)
    #   U(s) = sum_b  coverage_bin_weight_b * min(q_b / n_b, 1)
    # where q_b = observed-Tier-1+ count in bin b, n_b = quota for bin b.
    # Once a bin reaches its quota, extra observations stop contributing.
    # coverage_quota_per_bin: desired number of T1+ observations per bin.
    # coverage_weight: scale applied to the marginal U_pop signal.
    coverage_quota_per_bin: int   = 5      # target coverage per population bin
    coverage_weight:        float = 2.0    # scale on marginal coverage reward

    # --- science weight floor ---
    # Minimum science_weight for any target after inverse-frequency reweighting.
    # Prevents the most-common bin from receiving exactly 0 science weight.
    # Weights are remapped as: w' = floor + (1 − floor) * w_normalised
    # Typical range: 0.25–0.5.
    science_weight_floor: float = 0.3

    # --- science_weight scaling scope ---
    # Which tier-completion bonuses multiply by science_weight (diversity still
    # applies to all).  Default (1,2,3) = legacy behaviour.  Use (1,) so T2/T3
    # completions are not re-boosted by the same catalogue priority that already
    # paid out at T1 — stops easy high-weight T3 targets from compounding.
    # Empty tuple → never multiply tier bonuses by science_weight.
    science_weight_tiers: tuple = (1, 2, 3)
    # When True, dense progress shaping also multiplies by science_weight.
    # Set False to use diversity alone on progress (same anti-compounding idea).
    science_weight_on_progress: bool = True

    # --- T2 coverage milestones (optional; empty fractions = disabled) ---
    # Analogous to T1 milestones, but the denominator is chosen by
    # ``t2_milestone_pool`` so you can target hard T2s rather than the cheap
    # T3-capable subset:
    #   "max_tier_eq_2"     — initial targets with max_tier == 2 (~421)
    #   "max_tier_ge_2"     — initial targets with max_tier >= 2 (~550)
    #   "initial_catalogue" — all initial targets (same denom as T1)
    t2_milestone_fractions:       tuple = ()
    t2_milestone_bonus:           float = 0.0
    t2_final_milestone_bonus:     float = 0.0
    t2_milestone_step_above_one:  float = 0.0
    t2_overflow_milestone_bonus:  float = 0.0
    t2_milestone_pool:             str   = "max_tier_eq_2"

    # --- unique host diversity bonus ---
    # Fired the first time a new planetary *system* (host star) has any target
    # reach Tier 1.  Rewards breadth across stellar systems, not just bins.
    # Set to 0.0 to disable.
    unique_host_weight: float = 0.5

    # --- comparative planetology bonus ---
    # Fired when a target reaches Tier 1 and the same host already has at least
    # one other Tier-1+ target.  Rewards completing scientifically useful pairs
    # or triples within multi-planet systems (Ariel comparative planetology).
    # Scales with the number of Tier-1+ siblings already in the system.
    # Set to 0.0 to disable.
    comparative_weight: float = 0.3

    # --- penalties ---
    miss_penalty:           float = 0.1
    invalid_action_penalty: float = 0.5

    # ---------------------------------------------------------------------------
    # Relative reward mode
    # ---------------------------------------------------------------------------
    # Set ``reward_mode = "relative"`` to replace the per-step absolute reward with
    # a checkpoint-based signal that measures how much better the agent is doing
    # compared to a pre-recorded baseline policy trajectory.
    #
    # How it works
    # ------------
    # The underlying absolute reward is always computed internally (same formula as
    # "absolute" mode).  In relative mode the agent is NOT given the raw per-step
    # reward; instead it receives two types of bonus/penalty at fixed mission-time
    # checkpoints:
    #
    # Short-interval (weekly) comparison:
    #   Every ``comparison_interval_days`` of elapsed mission time the agent's
    #   accumulated absolute reward for that interval is compared to the stored
    #   baseline mean reward for the same interval:
    #       reward += comparison_scale × (agent_interval_reward − baseline_interval_reward)
    #
    # Compound (monthly) comparison:
    #   Every ``compound_interval_days`` the agent's TOTAL cumulative absolute
    #   reward so far is compared to the baseline total at that same mission-time
    #   checkpoint:
    #       reward += compound_scale × (agent_total_reward − baseline_total_at_checkpoint)
    #   This compounds: a consistently-better agent sees a growing bonus each month.
    #
    # Setup
    # -----
    # 1. Generate the baseline trajectory once (BEFORE training):
    #       python scripts/generate_baseline_trajectory.py \
    #           --policy smart_greedy --n-episodes 20 \
    #           --config configs/env/simple.yaml \
    #           --out data/baselines/smart_greedy_trajectory.json
    # 2. Point ``baseline_trajectory_path`` at the resulting JSON.
    # 3. Set ``reward_mode: relative`` in your reward YAML.
    #
    # The ``info["abs_reward"]`` key always carries the raw absolute reward for
    # monitoring/debugging regardless of which mode is active.
    # ---------------------------------------------------------------------------
    reward_mode: str = "absolute"          # "absolute" | "relative"

    # Interval for the weekly-style marginal comparison (mission days).
    comparison_interval_days: float = 7.0

    # Interval for the monthly-style cumulative comparison (mission days).
    # Conventionally a multiple of comparison_interval_days.
    compound_interval_days: float = 28.0

    # Scale factors applied to each checkpoint signal.
    # Keep comparison_scale ~ 1 and compound_scale < 1 so the compound bonus
    # is a supplement rather than the dominant signal.
    comparison_scale: float = 1.0
    compound_scale:   float = 0.1

    # Path to a baseline trajectory JSON produced by generate_baseline_trajectory.py.
    # Required when reward_mode = "relative".  Supports absolute and cwd-relative paths.
    baseline_trajectory_path: str = ""


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """Full environment configuration."""
    mission: MissionConfig = field(default_factory=MissionConfig)
    slew: SlewConfig = field(default_factory=SlewConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    catalogue_growth: CatalogueGrowthConfig = field(
        default_factory=CatalogueGrowthConfig
    )
    annual_t3_revisit: AnnualT3RevisitConfig = field(
        default_factory=AnnualT3RevisitConfig
    )
    seed: int = 42


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Recursively convert a nested dict to a (frozen) dataclass instance."""
    if data is None:
        return cls()

    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}

    for fname, fld in field_map.items():
        if fname not in data:
            continue
        val = data[fname]
        ftype = fld.type

        # Resolve string annotations
        if isinstance(ftype, str):
            import sys
            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # noqa: S307

        # Recurse into nested dataclasses
        if hasattr(ftype, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[fname] = _dict_to_dataclass(ftype, val)
        else:
            kwargs[fname] = val

    return cls(**kwargs)


def load_env_config(path: str | Path) -> EnvConfig:
    """Load an EnvConfig from a YAML file or bundled preset name.

    Unknown keys are silently ignored; missing keys fall back to defaults.

    Parameters
    ----------
    path:
        Path to a YAML file, or a bundled env preset name such as
        ``"simple"`` / ``"default"`` / ``"demo"``.
    """
    p = resolve_env_preset(path)

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    return _dict_to_dataclass(EnvConfig, raw)


def _default_env_dir() -> Path:
    from aRieL.utils.paths import bundled_config_dir
    return bundled_config_dir() / "env"


def _default_reward_dir() -> Path:
    from aRieL.utils.paths import bundled_config_dir
    return bundled_config_dir() / "reward"


def list_env_presets(env_dir: str | Path | None = None) -> list[Path]:
    """Return sorted bundled (or *env_dir*) environment YAML paths."""
    root = Path(env_dir) if env_dir is not None else _default_env_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.yaml"))


def list_reward_presets(reward_dir: str | Path | None = None) -> list[Path]:
    """Return sorted ``*.yaml`` paths under the reward presets directory."""
    root = Path(reward_dir) if reward_dir is not None else _default_reward_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.yaml"))


def _resolve_named_yaml(name: str | Path, root: Path, kind: str) -> Path:
    p = Path(name)
    if p.suffix in {".yaml", ".yml"} and p.is_file():
        return p
    stem = p.name.removesuffix(".yaml").removesuffix(".yml")
    for candidate in (root / stem, root / f"{stem}.yaml", root / f"{stem}.yml"):
        if candidate.is_file():
            return candidate
    available = ", ".join(f.stem for f in sorted(root.glob("*.yaml"))) or "(none)"
    raise FileNotFoundError(
        f"{kind} preset {name!r} not found in {root}. Available: {available}"
    )


def resolve_env_preset(name: str | Path) -> Path:
    """Resolve a filesystem path or bundled env preset name to a YAML file."""
    return _resolve_named_yaml(name, _default_env_dir(), "Env")


_REWARD_ALIASES = {
    "survey_first": "default",
}


def resolve_reward_preset(name: str | Path) -> Path:
    """Resolve a filesystem path or bundled reward preset name to a YAML file."""
    p = Path(name)
    if p.suffix not in {".yaml", ".yml"}:
        aliased = _REWARD_ALIASES.get(p.name, p.name)
        name = aliased
    return _resolve_named_yaml(name, _default_reward_dir(), "Reward")


def load_preset(env: str = "default", reward: str | None = None) -> EnvConfig:
    """Load a bundled env preset, optionally overlaying a reward preset.

    Examples
    --------
    >>> cfg = load_preset("demo")
    >>> cfg = load_preset("simple", reward="default")
    """
    cfg = load_env_config(resolve_env_preset(env))
    if reward is None:
        return cfg
    reward_cfg = load_reward_config(resolve_reward_preset(reward), strict=False)
    return replace(cfg, reward=reward_cfg)


def _format_reward_preset_help(reward_dir: str | Path | None = None) -> str:
    """Human-readable list of available reward presets for error messages."""
    root = Path(reward_dir) if reward_dir is not None else _default_reward_dir()
    presets = list_reward_presets(root)
    if not presets:
        return (
            f"No reward YAML files found in {root}.\n"
            "Add a complete preset under configs/reward/ and pass it via --reward-config."
        )
    lines = [f"Available reward presets in {root}:"]
    for p in presets:
        lines.append(f"  --reward-config {p}")
    lines.append(
        "Pick one of the files above (or another complete reward YAML). "
        "There is no default reward config."
    )
    return "\n".join(lines)


def load_reward_config(path: str | Path, *, strict: bool = True) -> RewardConfig:
    """Load a ``RewardConfig`` from YAML.

    The file must exist, be non-empty, and either:

    * contain a top-level ``reward:`` mapping, or
    * be a flat mapping of ``RewardConfig`` fields.

    ``strict=True`` (default, used at training time)
        Every current field must be present; unknown keys are an error.
        Missing keys are **not** filled from dataclass defaults.

    ``strict=False`` (saved run snapshots at evaluation time)
        Fields added after the run was trained are filled from current
        ``RewardConfig`` defaults.  Keys that have since been removed are
        ignored.  A short warning is printed for each case.

    On a strict failure the error message lists the presets under
    ``configs/reward/``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the YAML is empty, or (when ``strict``) missing/unknown keys.
    """
    p = Path(path)
    help_txt = _format_reward_preset_help()

    if not p.exists():
        raise FileNotFoundError(
            f"Reward config file not found: {p}\n"
            "A --reward-config path is required; nothing is loaded by default.\n"
            f"{help_txt}"
        )

    with open(p) as f:
        raw = yaml.safe_load(f)

    if raw is None or raw == {}:
        raise ValueError(
            f"Reward config file is empty or contains no valid YAML: {p}\n"
            f"{help_txt}"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"Reward config must be a YAML mapping, got {type(raw).__name__}: {p}\n"
            f"{help_txt}"
        )

    if "reward" in raw:
        if not isinstance(raw["reward"], dict):
            raise ValueError(
                f"'reward' section must be a mapping in {p}, "
                f"got {type(raw['reward']).__name__}\n{help_txt}"
            )
        other = {k for k in raw if k != "reward"}
        if other:
            raise ValueError(
                f"Reward config {p} has a 'reward:' section but also top-level "
                f"keys {sorted(other)}. Put all reward fields under 'reward:'.\n"
                f"{help_txt}"
            )
        data = raw["reward"]
    else:
        data = raw

    if not data:
        raise ValueError(
            f"Reward config has no fields under 'reward:' (or at top level): {p}\n"
            f"{help_txt}"
        )

    required = {f.name for f in fields(RewardConfig)}
    provided = set(data.keys())
    unknown = sorted(provided - required)
    missing = sorted(required - provided)
    if unknown or missing:
        if strict:
            parts = [
                f"Invalid reward config: {p}",
                "Every RewardConfig field must be set explicitly — "
                "missing keys are not filled from defaults.",
            ]
            if missing:
                parts.append(
                    "Missing required keys:\n  - " + "\n  - ".join(missing)
                )
            if unknown:
                parts.append(
                    "Unknown keys (typo or outdated field?):\n  - "
                    + "\n  - ".join(unknown)
                )
            parts.append(help_txt)
            raise ValueError("\n".join(parts))
        if missing:
            print(
                f"  Reward snapshot {p.name}: filling {len(missing)} newer "
                f"field(s) from current defaults:\n    "
                + ", ".join(missing)
            )
        if unknown:
            print(
                f"  Reward snapshot {p.name}: ignoring {len(unknown)} "
                f"removed/unknown key(s):\n    " + ", ".join(unknown)
            )

    _tuple_fields = {
        "t1_milestone_fractions",
        "t2_milestone_fractions",
        "science_weight_tiers",
    }
    _pool_values = {"max_tier_eq_2", "max_tier_ge_2", "initial_catalogue"}

    defaults = RewardConfig()
    kwargs: dict = {}
    for f in fields(RewardConfig):
        if f.name in data:
            val = data[f.name]
        else:
            kwargs[f.name] = getattr(defaults, f.name)
            continue
        if f.name in _tuple_fields and isinstance(val, list):
            val = tuple(val)
        if f.name == "science_weight_tiers":
            val = tuple(int(x) for x in val)
            bad = [t for t in val if t not in (1, 2, 3)]
            if bad:
                raise ValueError(
                    f"science_weight_tiers entries must be in {{1,2,3}}, got {bad} in {p}"
                )
        if f.name == "t2_milestone_pool" and val not in _pool_values:
            raise ValueError(
                f"t2_milestone_pool must be one of {sorted(_pool_values)}, "
                f"got {val!r} in {p}"
            )
        kwargs[f.name] = val

    return RewardConfig(**kwargs)


def default_env_config() -> EnvConfig:
    """Return an EnvConfig with all defaults (no YAML needed)."""
    return EnvConfig()


def env_config_to_dict(cfg: EnvConfig) -> dict:
    """Serialise an EnvConfig back to a plain dict (for logging/saving)."""
    return asdict(cfg)
