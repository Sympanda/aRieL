"""
ArielEnv: Gymnasium environment for Ariel observation scheduling.

Episode flow
------------
    obs, info = env.reset()
    while True:
        action = agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Action spaces
-------------
``topk`` (default):
    Discrete(K).  The agent picks an index 0…K-1 into the K upcoming events.
    ``info["action_mask"]`` is a boolean array of shape (K,).

``target``:
    Discrete(N).  The agent picks a target index 0…N-1.
    The env schedules the next available event for that target.
    ``info["action_mask"]`` is a boolean array of shape (N,).

Observation space
-----------------
Dict with two Box spaces:
    "events"  Box(shape=(K_or_N, n_event_features), dtype=float32)
    "global"  Box(shape=(n_global_features,),        dtype=float32)

Reward
------
Per-step reward is computed by ``rewards.compute_reward`` and includes:

* Sparse tier-completion bonuses (T1=1, T2=3, T3=10, scaled by science_weight × diversity_mult)
* Dense progress shaping (proportional to Δprogress_in_tier; 3× boost when near a tier boundary)
* Dense efficiency reward (obs_duration / total_cost; penalises long slews)
* Missed-event penalty (if agent arrives after window_end)

In addition, ``check_milestone_reward`` fires one-shot bonuses when T1 coverage
crosses 25/50/75/90/100 % of the catalogue, and ``compute_terminal_reward`` fires
a quadratic end-of-episode bonus based on final T1 coverage fraction.

The ``info`` dict always includes the raw ``step_result`` from
``execute_observation`` (tier changes, slew cost, etc.) for external analysis.

Configuration
-------------
Pass an EnvConfig (or a path to a YAML) to the constructor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from aRieL.data.preprocess_targets import build_target_table
from aRieL.envs.action_mask import any_valid, compute_mask
from aRieL.envs.observation_builder import build as build_obs, observation_shapes
from aRieL.envs.planet_feature_builder import (
    build_planet_features,
    build_static_features,
    N_PLANET_FEATURES,
    PLANET_FEATURE_NAMES,
)
from aRieL.simulator.event_backend import EventBackend, DynamicBackend
from aRieL.simulator.mission_state import MissionState
from aRieL.simulator.slew import SLEW_RATE_DEG_PER_MIN, MIN_SLEW_S, MAX_SLEW_S
from aRieL.simulator.catalogue_growth import (
    InjectionEvent,
    build_injection_schedule,
    expand_milestone_fractions,
    resolve_tpc_path,
)
from aRieL.rewards.compute_reward import (
    compute_reward,
    check_t1_milestone_reward,
    check_t2_milestone_reward,
    compute_terminal_reward,
)
from aRieL.rewards.context import RewardFn, reward_info_from_step
from aRieL.utils.config import (
    EnvConfig,
    default_env_config,
    load_env_config,
)


class ArielEnv(gym.Env):
    """Gymnasium environment for Ariel exoplanet target scheduling.

    Parameters
    ----------
    config:
        An ``EnvConfig`` instance, a YAML path, a bundled preset name
        (``"demo"``, ``"simple"``, …), or ``None`` for dataclass defaults.
    csv_path:
        Path to the raw MCS CSV.  Only used when *targets* is not provided.
    targets:
        Pre-built target DataFrame (skips CSV loading if provided).
    events:
        Unused — retained for backward-compatibility only.  The environment
        now defaults to ``DynamicBackend`` which computes events on-the-fly.
    backend:
        Optional pre-constructed ``EventBackend`` instance.  Defaults to
        ``DynamicBackend(targets)`` when *None*.
    reward_fn:
        Optional custom per-step reward ``fn(RewardInfo) -> float``.
        When set, replaces stock ``compute_reward`` for the step signal.
        Milestone / terminal bonuses from the reward config still apply
        (zero those weights in YAML if you want a pure custom signal).
    """

    metadata = {"render_modes": []}

    @classmethod
    def from_preset(
        cls,
        env: str = "default",
        reward: str | None = None,
        **kwargs,
    ) -> "ArielEnv":
        """Build an environment from bundled YAML presets.

        Parameters
        ----------
        env:
            Bundled env preset name (``demo``, ``simple``, ``full_set``, …)
            or a path to a YAML file.
        reward:
            Optional bundled reward preset (``default``, ``t1_boost``, …).
        **kwargs:
            Forwarded to the constructor (``targets``, ``csv_path``,
            ``reward_fn``, …).
        """
        from aRieL.utils.config import load_preset
        return cls(config=load_preset(env, reward=reward), **kwargs)

    def __init__(
        self,
        config: EnvConfig | str | Path | None = None,
        csv_path: str | Path | None = None,
        targets: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
        backend: Optional[EventBackend] = None,
        reward_fn: Optional[RewardFn] = None,
    ) -> None:
        super().__init__()

        # ---- config ----
        if config is None:
            self.cfg = default_env_config()
        elif isinstance(config, (str, Path)):
            self.cfg = load_env_config(config)
        else:
            self.cfg = config

        self._reward_fn = reward_fn

        self._slew_rate = self.cfg.slew.rate_deg_per_min
        self._min_slew_s = self.cfg.slew.min_slew_seconds
        self._max_slew_s = self.cfg.slew.max_slew_seconds

        # ---- static tables ----
        if targets is not None:
            self._targets = targets.copy()
        else:
            self._targets = build_target_table(
                csv_path,
                science_weight_floor=self.cfg.reward.science_weight_floor,
            )

        # Apply global max_tier_cap: clip each target's max_tier downward.
        cap = self.cfg.mission.max_tier_cap
        if cap < 3 and "max_tier" in self._targets.columns:
            self._targets["max_tier"] = self._targets["max_tier"].clip(upper=cap)

        # ---- catalogue growth (TPC injection pool) ----
        self._tpc_pool: pd.DataFrame = pd.DataFrame()
        self._tpc_by_id: dict[str, pd.Series] = {}
        self._injection_schedule: list[InjectionEvent] = []
        self._injection_cursor: int = 0
        self._episode_milestones: tuple[float, ...] = tuple(
            self.cfg.reward.t1_milestone_fractions
        )
        self._fixed_injection_schedule: list[InjectionEvent] | None = None
        self._fixed_episode_milestones: tuple[float, ...] | None = None
        self._bin_vocab: tuple[str, ...] = tuple()
        # Annual T3 revisit bookkeeping (eval scenario).
        self._next_t3_revisit_bjd: float | None = None
        self._n_t3_revisits_fired: int = 0
        self._last_t3_reset_ids: list[str] = []
        self._year_snapshots: list[dict] = []
        if self.cfg.catalogue_growth.enabled:
            self._prepare_catalogue_growth()
        elif "population_bin" in self._targets.columns:
            self._bin_vocab = tuple(sorted(self._targets["population_bin"].unique()))

        # ---- event backend ----
        self._events: pd.DataFrame = events if events is not None else pd.DataFrame()
        if backend is not None:
            self._backend: EventBackend = backend
        else:
            self._backend = DynamicBackend(self._targets)

        # ---- determine action space size ----
        if self.cfg.action.type == "topk":
            self._n_actions = self.cfg.action.topk.k
        elif self.cfg.action.type == "target":
            # Pad to final possible size when growth is enabled.
            n_final = len(self._targets) + len(self._tpc_pool)
            self._n_actions = n_final if self.cfg.catalogue_growth.enabled else len(self._targets)
        elif self.cfg.action.type == "full_set":
            cfg_k_filter = self.cfg.action.full_set.k_filter
            cfg_n_max    = self.cfg.action.full_set.n_max
            n_final = len(self._targets) + len(self._tpc_pool)
            if cfg_k_filter > 0:
                # k_filter: only top-K planets reach the policy → action space = K
                self._n_actions = cfg_k_filter
            elif cfg_n_max > 0:
                # n_max is a hard ceiling — the catalogue must fit inside it.
                if n_final > cfg_n_max:
                    raise ValueError(
                        f"Catalogue may grow to {n_final} targets but "
                        f"action.full_set.n_max={cfg_n_max}.  "
                        "Increase n_max (or reduce the TPC pool)."
                    )
                self._n_actions = cfg_n_max
            else:
                # n_max=0 → use final size when growth on, else catalogue size
                self._n_actions = n_final if self.cfg.catalogue_growth.enabled else len(self._targets)
        else:
            raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

        # Number of actual catalogue targets at episode start (may be < _n_actions)
        self._n_targets = len(self._targets)

        # Dynamic active-planet set (full_set mode only; all targets start active).
        # Maintained as an ordered list so action_index i maps to _active_target_ids[i].
        # Populated in reset(); updated after each step when targets complete.
        self._active_target_ids: list[str] = []
        # Reverse index: target_id → position in _active_target_ids (for O(1) lookup)
        self._active_tid_to_idx: dict[str, int] = {}
        # target_id → row-index in the full static-feature cache (built at reset)
        self._tid_to_cache_idx: dict[str, int] = {}

        # ---- static per-planet feature cache (full_set mode) ----
        self._static_planet_features: np.ndarray | None = None

        # ---- bootstrap a dummy state to measure observation shapes ----
        _dummy_state = MissionState.from_backend(
            self._targets,
            backend=self._backend,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
        )
        if self._bin_vocab:
            _dummy_state._bin_vocab = self._bin_vocab
        if hasattr(self, "_bin_totals_obs") and self._bin_totals_obs:
            _dummy_state._bin_totals_obs = dict(self._bin_totals_obs)

        # ---- Gymnasium spaces ----
        if self.cfg.action.type == "full_set":
            # Observation: per-planet features (N × F) + global
            shapes = observation_shapes(_dummy_state, self.cfg.observation, self._n_actions)
            self.observation_space = spaces.Dict({
                "planets": spaces.Box(
                    low=-3.0, high=3.0,
                    shape=(self._n_actions, N_PLANET_FEATURES),
                    dtype=np.float32,
                ),
                "global": spaces.Box(
                    low=0.0, high=1.0,
                    shape=shapes["global"],
                    dtype=np.float32,
                ),
            })
        else:
            shapes = observation_shapes(_dummy_state, self.cfg.observation, self._n_actions)
            self.observation_space = spaces.Dict({
                "events": spaces.Box(
                    low=-3.0, high=3.0,
                    shape=shapes["events"],
                    dtype=np.float32,
                ),
                "global": spaces.Box(
                    low=0.0, high=1.0,
                    shape=shapes["global"],
                    dtype=np.float32,
                ),
            })
        self.action_space = spaces.Discrete(self._n_actions)

        # ---- episode state (initialised in reset) ----
        self._state: Optional[MissionState] = None
        self._candidates: Optional[pd.DataFrame] = None
        self._action_mask: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._milestones_hit: set[float] = set()      # one-shot T1-coverage bonuses
        self._t2_milestones_hit: set[float] = set()   # one-shot T2-coverage bonuses
        self._t2_milestone_pool_ids: set[str] = set()
        self._n_t2_milestone_denom: int = 0
        self._t2_episode_milestones: tuple[float, ...] = tuple(
            self.cfg.reward.t2_milestone_fractions
        )

        # ---- relative reward mode: baseline trajectory ----
        self._baseline_traj: dict = {}
        if self.cfg.reward.reward_mode == "relative":
            traj_path_str = self.cfg.reward.baseline_trajectory_path
            if not traj_path_str:
                raise ValueError(
                    "reward_mode='relative' requires baseline_trajectory_path to be set. "
                    "Run scripts/generate_baseline_trajectory.py first, then point "
                    "baseline_trajectory_path at the resulting JSON."
                )
            traj_path = Path(traj_path_str)
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"Baseline trajectory not found: {traj_path}. "
                    "Run scripts/generate_baseline_trajectory.py to generate it."
                )
            with open(traj_path) as _f:
                self._baseline_traj = json.load(_f)

        # Relative reward episode accumulators (initialised in reset)
        self._rel_interval_acc: float = 0.0
        self._rel_total_acc: float = 0.0
        self._rel_comparison_idx: int = 0
        self._rel_compound_idx: int = 0
        self._next_comparison_bjd: float = 0.0
        self._next_compound_bjd: float = 0.0

    # ------------------------------------------------------------------
    # Catalogue growth helpers
    # ------------------------------------------------------------------

    def _prepare_catalogue_growth(self) -> None:
        """Load TPC pool, drop MCS overlap, harmonise science weights / bins."""
        from aRieL.data.population_bins import assign_population_bins

        growth = self.cfg.catalogue_growth
        tpc_path = resolve_tpc_path(growth.tpc_csv_path)
        if tpc_path is None:
            raise FileNotFoundError(
                "catalogue_growth.enabled=True but no TPC CSV found.  "
                "Place TPC.csv under $ARIEL_DATA or ./data/raw/, "
                "or set catalogue_growth.tpc_csv_path."
            )

        tpc = build_target_table(
            tpc_path,
            science_weight_floor=self.cfg.reward.science_weight_floor,
        )
        cap = self.cfg.mission.max_tier_cap
        if cap < 3 and "max_tier" in tpc.columns:
            tpc["max_tier"] = tpc["max_tier"].clip(upper=cap)

        mcs_ids = set(self._targets["target_id"].astype(str))
        tpc = tpc[~tpc["target_id"].astype(str).isin(mcs_ids)].reset_index(drop=True)
        if len(tpc) == 0:
            import warnings
            warnings.warn(
                "TPC pool is empty after removing MCS overlap; catalogue growth "
                "will inject nothing.",
                stacklevel=2,
            )
            self._tpc_pool = tpc
            self._mcs_targets = self._targets.copy()
            self._bin_vocab = tuple(sorted(self._targets["population_bin"].unique()))
            self._bin_totals_obs = (
                self._targets["population_bin"].value_counts().to_dict()
                if "population_bin" in self._targets.columns else {}
            )
            return

        # Recompute population bins + science weights on MCS ∪ TPC so injected
        # planets share the same inverse-frequency prior as the starting set.
        n_mcs = len(self._targets)
        combined = pd.concat([self._targets, tpc], ignore_index=True)
        combined = assign_population_bins(
            combined, science_weight_floor=self.cfg.reward.science_weight_floor
        )
        self._targets = combined.iloc[:n_mcs].reset_index(drop=True)
        self._tpc_pool = combined.iloc[n_mcs:].reset_index(drop=True)
        self._mcs_targets = self._targets.copy()
        self._bin_vocab = tuple(sorted(combined["population_bin"].unique()))
        # Combined bin counts — used to decide which bins appear in the global
        # obs vector (stable across the episode even before injections).
        self._bin_totals_obs: dict[str, int] = (
            combined["population_bin"].value_counts().to_dict()
        )

        # Lookups keyed by target_id for schedule application
        self._tpc_by_id = {
            str(r["target_id"]): r for _, r in self._tpc_pool.iterrows()
        }

        # Pre-build the fixed schedule once so every episode reuses it.
        self._fixed_injection_schedule: list[InjectionEvent] | None = None
        self._fixed_episode_milestones: tuple[float, ...] | None = None
        if self.cfg.catalogue_growth.schedule_mode == "fixed":
            self._fixed_injection_schedule, self._fixed_episode_milestones = (
                self._make_injection_schedule(seed=None)
            )

    def _schedule_rng_seed(self, episode_seed: Optional[int]) -> int:
        """Resolve the RNG seed used to build an injection schedule."""
        growth = self.cfg.catalogue_growth
        mode = str(growth.schedule_mode).lower()
        if mode == "fixed":
            if growth.schedule_seed >= 0:
                return int(growth.schedule_seed)
            return int(self.cfg.seed)
        # per_episode (and any unrecognised value → per_episode behaviour)
        if episode_seed is not None:
            return int(episode_seed)
        return int(self.cfg.seed)

    def _make_injection_schedule(
        self, seed: Optional[int]
    ) -> tuple[list[InjectionEvent], tuple[float, ...]]:
        """Sample an injection schedule and the matching milestone ladder."""
        growth = self.cfg.catalogue_growth
        base_milestones = tuple(self.cfg.reward.t1_milestone_fractions)
        if not growth.enabled or len(self._tpc_pool) == 0:
            return [], base_milestones

        mode = str(growth.schedule_mode).lower()
        if mode not in ("fixed", "per_episode"):
            raise ValueError(
                f"catalogue_growth.schedule_mode must be 'fixed' or "
                f"'per_episode', got {growth.schedule_mode!r}"
            )

        rng_seed = self._schedule_rng_seed(seed)
        rng = np.random.default_rng(rng_seed ^ 0xCA7A106E)

        # When called from _prepare (targets = MCS only), use len(_targets).
        # Prefer _mcs_targets if already stored.
        n_initial = len(getattr(self, "_mcs_targets", self._targets))
        mission_start = self.cfg.mission.start_bjd
        mission_end = mission_start + self.cfg.mission.lifetime_days
        schedule = build_injection_schedule(
            self._tpc_pool,
            mission_start=mission_start,
            mission_end=mission_end,
            rng=rng,
            mean_per_month=growth.mean_per_month,
            std_per_month=growth.std_per_month,
            month_days=growth.month_days,
        )
        max_frac = (n_initial + len(schedule)) / max(n_initial, 1)
        milestones = expand_milestone_fractions(
            base_milestones,
            max_fraction=max_frac,
            step_above_one=self.cfg.reward.t1_milestone_step_above_one,
        )
        return schedule, milestones

    def _build_episode_injection_schedule(self, seed: Optional[int]) -> None:
        """Install the injection schedule for this episode and reset the cursor."""
        growth = self.cfg.catalogue_growth
        self._injection_cursor = 0
        self._episode_milestones = tuple(self.cfg.reward.t1_milestone_fractions)
        self._injection_schedule = []

        if not growth.enabled or len(self._tpc_pool) == 0:
            return

        if (
            str(growth.schedule_mode).lower() == "fixed"
            and self._fixed_injection_schedule is not None
        ):
            # Same planets / times every episode — no extra training noise.
            self._injection_schedule = list(self._fixed_injection_schedule)
            self._episode_milestones = self._fixed_episode_milestones or self._episode_milestones
            return

        self._injection_schedule, self._episode_milestones = (
            self._make_injection_schedule(seed)
        )

    def _apply_due_injections(self) -> list[str]:
        """Insert any scheduled planets whose BJD ≤ current mission time."""
        if not self._injection_schedule or self._state is None:
            return []

        t_now = self._state.clock.current_time
        due_rows = []
        while self._injection_cursor < len(self._injection_schedule):
            ev = self._injection_schedule[self._injection_cursor]
            if ev.bjd > t_now:
                break
            row = getattr(self, "_tpc_by_id", {}).get(ev.target_id)
            if row is not None:
                due_rows.append(row)
            self._injection_cursor += 1

        if not due_rows:
            return []

        new_df = pd.DataFrame(due_rows)
        added = self._state.add_targets(new_df)
        if not added:
            return []

        # Keep env-level target table in sync (diagnostics / host counts).
        self._targets = self._state.targets
        self._n_targets = len(self._targets)

        if self.cfg.action.type == "full_set":
            for tid in added:
                if tid not in self._active_tid_to_idx:
                    self._active_tid_to_idx[tid] = len(self._active_target_ids)
                    self._active_target_ids.append(tid)

            # Extend static feature cache for the newcomers.
            if (
                self.cfg.action.full_set.cache_static
                and self._static_planet_features is not None
            ):
                # Rebuild cache from the full current state (simplest + correct).
                self._static_planet_features = build_static_features(self._state)
                all_tids = list(self._state.targets["target_id"].astype(str))
                self._tid_to_cache_idx = {tid: i for i, tid in enumerate(all_tids)}

        return added

    def _record_year_snapshot(self, *, kind: str, year_index: int) -> dict:
        """Store tier counts / rates at a year boundary or mission end."""
        from aRieL.evaluation.metrics import compute_stats

        assert self._state is not None
        stats = compute_stats(self._state)
        start = self.cfg.mission.start_bjd
        day = float(self._state.clock.current_time) - start
        rec = stats.to_dict()
        rec.update({
            "snapshot_kind": kind,
            "year_index": int(year_index),
            "mission_day": day,
            "n_t3_revisits_fired": self._n_t3_revisits_fired,
        })
        self._year_snapshots.append(rec)
        return rec

    def _schedule_next_t3_revisit(self) -> None:
        """Arm the next year-boundary progress reset (or disable if unused)."""
        rev = self.cfg.annual_t3_revisit
        if not rev.enabled or rev.period_days <= 0:
            self._next_t3_revisit_bjd = None
            return
        start = self.cfg.mission.start_bjd
        # First reset at end of year 1 (not at t=0 — progress is already zero).
        self._next_t3_revisit_bjd = start + float(rev.period_days)

    def _apply_annual_t3_resets(self) -> list[str]:
        """Undo progress on deep-programme planets when a year boundary is crossed.

        Snapshots tier stats *before* each reset so yearly figures can show
        what was achieved in that year.  May fire more than once if the clock
        jumped across several periods in a single step.
        """
        rev = self.cfg.annual_t3_revisit
        self._last_t3_reset_ids = []
        if (
            not rev.enabled
            or self._next_t3_revisit_bjd is None
            or self._state is None
        ):
            return []

        t_now = float(self._state.clock.current_time)
        mission_end = self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days
        reset_all: list[str] = []
        min_tier = int(rev.min_max_tier)

        while (
            self._next_t3_revisit_bjd is not None
            and t_now >= self._next_t3_revisit_bjd
            and self._next_t3_revisit_bjd <= mission_end + 1e-9
        ):
            year_index = self._n_t3_revisits_fired + 1
            self._record_year_snapshot(kind="year_end", year_index=year_index)

            tids = [
                str(row["target_id"])
                for _, row in self._state.targets.iterrows()
                if int(row["max_tier"]) >= min_tier
            ]
            reset = self._state.reset_progress(tids)
            reset_all.extend(reset)
            self._n_t3_revisits_fired += 1

            # Re-open completed planets in the full_set active set.
            if self.cfg.action.type == "full_set" and reset:
                active = set(self._active_target_ids)
                for tid in reset:
                    if tid not in active:
                        self._active_target_ids.append(tid)
                        active.add(tid)
                self._active_tid_to_idx = {
                    tid: i for i, tid in enumerate(self._active_target_ids)
                }

            self._next_t3_revisit_bjd += float(rev.period_days)

        # De-dupe while preserving order (multi-year jump).
        seen: set[str] = set()
        uniq: list[str] = []
        for tid in reset_all:
            if tid not in seen:
                seen.add(tid)
                uniq.append(tid)
        self._last_t3_reset_ids = uniq
        return uniq

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Rebuild backend from the starting MCS table (growth is re-applied
        # during the episode from the seeded schedule).
        if self.cfg.catalogue_growth.enabled and hasattr(self, "_mcs_targets"):
            self._targets = self._mcs_targets.copy()
            self._backend = DynamicBackend(self._targets)

        self._backend.reset()
        self._build_episode_injection_schedule(seed)

        self._state = MissionState.from_backend(
            self._targets,
            backend=self._backend,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
            overhead_days_per_obs=self.cfg.mission.overhead_days_per_obs,
        )
        # Freeze reference denominators + bin vocabulary for obs dim stability.
        self._state.n_initial_targets = len(self._targets)
        self._state._bin_totals_reward = dict(self._state._bin_totals)
        if self._bin_vocab:
            self._state._bin_vocab = self._bin_vocab
        if hasattr(self, "_bin_totals_obs") and self._bin_totals_obs:
            self._state._bin_totals_obs = dict(self._bin_totals_obs)

        self._step_count = 0
        self._milestones_hit = set()
        self._t2_milestones_hit = set()
        self._init_t2_milestone_pool()
        self._n_targets = len(self._targets)
        self._n_t3_revisits_fired = 0
        self._last_t3_reset_ids = []
        self._year_snapshots = []
        self._schedule_next_t3_revisit()

        if self.cfg.action.type == "full_set":
            # Initialise dynamic active set — at reset all targets begin at tier 0
            # so every target is active.
            self._active_target_ids = list(self._targets["target_id"].astype(str))
            self._active_tid_to_idx = {tid: i for i, tid in enumerate(self._active_target_ids)}

            # Pre-compute static planet features (time-invariant for the episode).
            if self.cfg.action.full_set.cache_static:
                self._static_planet_features = build_static_features(self._state)
                # Build reverse index: target_id → row in the (N, n_static) cache
                all_tids = list(self._targets["target_id"].astype(str))
                self._tid_to_cache_idx = {tid: i for i, tid in enumerate(all_tids)}

        # Reset relative-reward accumulators
        self._rel_interval_acc = 0.0
        self._rel_total_acc = 0.0
        self._rel_comparison_idx = 0
        self._rel_compound_idx = 0
        self._next_comparison_bjd = (
            self.cfg.mission.start_bjd + self.cfg.reward.comparison_interval_days
        )
        self._next_compound_bjd = (
            self.cfg.mission.start_bjd + self.cfg.reward.compound_interval_days
        )

        # Apply any injections scheduled at/before mission start (rare).
        self._apply_due_injections()

        self._candidates, self._action_mask = self._get_candidates_and_mask()

        obs = self._build_observation()
        info = self._make_info(step_result=None)
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        mask = self._action_mask
        if not mask[action]:
            # Invalid action — penalise and don't advance the clock
            penalty = -self.cfg.reward.invalid_action_penalty
            obs = self._build_observation()
            info = self._make_info(step_result=None)
            info["invalid_action"] = True
            info["abs_reward"] = penalty
            return obs, penalty, False, False, info

        # Map action index → event_id
        event_id = self._action_to_event_id(action)

        # Snapshot per-bin and per-host counts BEFORE executing the observation
        # (needed to compute the marginal coverage and host-diversity rewards).
        bin_observed_before = dict(self._state.population_bin_counts)
        host_tier1_before   = self._host_tier1_counts()

        # Execute the observation in the simulator
        step_result = self._state.execute_observation(event_id)
        self._step_count += 1

        # Inject any TPC planets whose confirmation BJD has been crossed.
        injected = self._apply_due_injections()

        # ---- compute full absolute reward for this step ----
        # (before annual T3 resets so this observation's tier transitions still count)
        abs_reward = self._compute_reward(step_result, bin_observed_before, host_tier1_before)

        # One-shot milestone bonuses — denominators are frozen at episode start
        # so coverage can exceed 100 % when injected planets are completed.
        n_ref = self._state.reference_total_targets
        milestone_bonus, self._milestones_hit = check_t1_milestone_reward(
            tier1_completed=self._state.tier1_completed,
            total_reachable=n_ref,
            milestones_hit=self._milestones_hit,
            cfg=self.cfg.reward,
            milestone_fractions=self._episode_milestones,
        )
        abs_reward += milestone_bonus

        if self._n_t2_milestone_denom > 0 and self._t2_episode_milestones:
            t2_bonus, self._t2_milestones_hit = check_t2_milestone_reward(
                tier2_completed=self._count_t2_in_milestone_pool(),
                total_reachable=self._n_t2_milestone_denom,
                milestones_hit=self._t2_milestones_hit,
                cfg=self.cfg.reward,
                milestone_fractions=self._t2_episode_milestones,
            )
            abs_reward += t2_bonus

        # Yearly T3 revisit: undo progress on max_tier≥3 planets after reward.
        t3_reset = self._apply_annual_t3_resets()

        # Update dynamic active set: remove planets that just completed max_tier.
        if self.cfg.action.type == "full_set":
            self._update_active_set()

        # Check episode termination
        terminated = self._state.is_done()
        self._candidates, self._action_mask = self._get_candidates_and_mask()

        # If no valid actions remain, try to recover a feasible action set.
        # For topk: ask the backend for a larger window (the K nearest events
        #   may all be expired; looking further ahead usually finds a valid one).
        # For target/full_set: candidates are fixed (one per target) — if none
        #   are valid the episode terminates; no lookahead is attempted.
        if not terminated and not any_valid(self._action_mask):
            if self.cfg.action.type == "topk":
                self._candidates, self._action_mask = self._skip_to_next_feasible_topk()

        if not terminated and not any_valid(self._action_mask):
            terminated = True

        # Terminal bonus: fired once at episode end (initial-catalogue denominator)
        if terminated:
            abs_reward += compute_terminal_reward(
                tier1_completed=self._state.tier1_completed,
                total_reachable=n_ref,
                cfg=self.cfg.reward,
            )
            if self.cfg.annual_t3_revisit.enabled:
                # Final partial year (after last reset, or full mission if none).
                year_index = self._n_t3_revisits_fired + 1
                self._record_year_snapshot(kind="mission_end", year_index=year_index)

        # ---- apply reward mode ----
        if self.cfg.reward.reward_mode == "relative":
            reward = self._apply_relative_reward(abs_reward, terminated)
        else:
            reward = abs_reward

        obs = self._build_observation()
        info = self._make_info(step_result=step_result)
        info["abs_reward"] = abs_reward
        if injected:
            info["injected_targets"] = injected
        if t3_reset:
            info["t3_revisit_reset"] = t3_reset
            info["n_t3_revisits_fired"] = self._n_t3_revisits_fired
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Candidate selection helpers
    # ------------------------------------------------------------------

    def _get_candidates_and_mask(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (candidate_events, action_mask) for the current state."""
        if self.cfg.action.type == "topk":
            return self._candidates_topk()
        elif self.cfg.action.type == "target":
            return self._candidates_target()
        elif self.cfg.action.type == "full_set":
            return self._candidates_full_set()
        else:
            raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

    def _candidates_topk(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Top-K upcoming events, delegated to the active EventBackend.

        When ``exclude_started_blocks`` is on, the pool is built from
        **unstarted** occurrences only (``unstarted_only=True``), then trimmed
        to events the telescope can fully catch (``t_arrive ≤ block_start``).
        Started / mid-block events never enter the action set — there is no
        post-hoc mask that zeroes them out.
        """
        k = self.cfg.action.topk.k
        t_now = self._state.clock.current_time
        exclude = bool(getattr(self.cfg.action.topk, "exclude_started_blocks", False))

        if exclude:
            # Oversample unstarted events, keep fully catchable, take K.
            pool = self._backend.candidates(t_now, max(k * 16, k), unstarted_only=True)
            candidates = self._select_fully_catchable_topk(pool, k)
            if len(candidates) == 0:
                return self._skip_to_next_feasible_topk()
        else:
            candidates = self._backend.candidates(t_now, k)

        mask = compute_mask(self._state, candidates, self.cfg.action)

        if len(candidates) < k:
            cols = candidates.columns if len(candidates) else pd.Index(
                ["event_id", "target_id", "event_type",
                 "window_start", "window_mid", "window_end",
                 "duration", "duration_days", "block_duration_days", "tier_goal",
                 "base_science_value", "visibility_valid",
                 "ephemeris_uncertainty", "event_index"]
            )
            padding = _make_padding_rows(k - len(candidates), cols)
            candidates = pd.concat([candidates, padding], ignore_index=True)
            mask = np.concatenate([mask, np.zeros(k - len(mask), dtype=bool)])

        return candidates.reset_index(drop=True), mask

    def _select_fully_catchable_topk(
        self, pool: pd.DataFrame, k: int,
    ) -> pd.DataFrame:
        """Keep the soonest *k* events we can arrive for before ``block_start``."""
        from aRieL.simulator.slew import slew_time_days_vec
        from aRieL.data.schemas import COST_FACTOR

        if len(pool) == 0:
            return pool

        t_now = self._state.clock.current_time
        tids = pool["target_id"].to_numpy()
        wmid = pool["window_mid"].to_numpy(dtype=float)
        if "block_duration_days" in pool.columns:
            bldur = pool["block_duration_days"].to_numpy(dtype=float)
        else:
            bldur = COST_FACTOR * pool["duration_days"].to_numpy(dtype=float)
        block_start = wmid - bldur / 2.0
        vis = pool["visibility_valid"].to_numpy(dtype=bool)

        ra = np.zeros(len(pool), dtype=float)
        dec = np.zeros(len(pool), dtype=float)
        tier_ok = np.ones(len(pool), dtype=bool)
        for i, tid in enumerate(tids):
            tid_s = str(tid) if tid is not None else ""
            target = self._state._target_lookup.get(tid_s)
            if target is None:
                tier_ok[i] = False
                continue
            ra[i] = float(target["ra"])
            dec[i] = float(target["dec"])
            prog = self._state._progress_dict.get(tid_s)
            if prog is not None and int(prog["current_tier"]) >= int(target["max_tier"]):
                tier_ok[i] = False

        slews = slew_time_days_vec(
            self._state.current_ra, self._state.current_dec, ra, dec,
        )
        t_arrive = t_now + slews
        keep = vis & tier_ok & (t_arrive <= block_start)
        return pool.iloc[np.flatnonzero(keep)].head(k).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dynamic active set management (full_set mode)
    # ------------------------------------------------------------------

    def _update_active_set(self) -> None:
        """Remove targets that have reached max_tier from the active set.

        Called after each observation step in ``full_set`` mode.  After removal
        the active set index is rebuilt so that ``_active_tid_to_idx`` remains
        consistent.
        """
        if not self._active_target_ids:
            return
        new_active: list[str] = []
        for tid in self._active_target_ids:
            prog   = self._state._progress_dict.get(tid)
            target = self._state._target_lookup.get(tid)
            if prog is None or target is None:
                continue
            if int(prog["current_tier"]) < int(target["max_tier"]):
                new_active.append(tid)
        self._active_target_ids = new_active
        self._active_tid_to_idx = {tid: i for i, tid in enumerate(new_active)}

    # ------------------------------------------------------------------
    # Candidate selection helpers (continued)
    # ------------------------------------------------------------------

    def _candidates_full_set(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Full-set mode: one *first-reachable* event per active target + padding.

        Each active planet token is associated with the first upcoming event
        that the telescope can reach from its current position (i.e.
        ``t_now + slew < block_end``).  If the current block has expired or
        the slew would miss it, we look further ahead — so the agent sees a
        real, meaningful action for every active planet rather than a
        routinely-masked stale event.

        When ``exclude_started_blocks`` is enabled, events whose observation
        block has already opened (``t_now >= block_start``) are skipped and
        those planets are omitted from the token set passed to the set
        transformer (no partial mid-block scheduling).

        Candidate row ordering matches ``_active_target_ids`` so that
        ``action_index i → _active_target_ids[i]``.

        Padding rows (indices ``n_active … n_max-1``) are always masked False.

        Implementation note
        -------------------
        Previously this method ran two nested Python loops (one over 16 k pool
        rows via ``iterrows()``, one over all active planets).  It now uses:

        1. A vectorised NumPy slew computation over all active targets at once.
        2. A pandas merge → filter → groupby to find the first-reachable event
           per target — no Python loop at all for the common case.
        3. A small Python fallback loop only for the rare long-period targets
           not covered by the pool.

        This gives ~10–50× speedup on the env-step bottleneck.
        """
        from aRieL.simulator.slew import slew_time_days_vec
        from aRieL.data.schemas import COST_FACTOR

        t_now    = self._state.clock.current_time
        n_active = len(self._active_target_ids)
        exclude_started = self.cfg.action.full_set.exclude_started_blocks

        _FALLBACK_COLS = pd.Index([
            "event_id", "target_id", "event_type",
            "window_start", "window_mid", "window_end",
            "duration", "duration_days", "block_duration_days", "tier_goal",
            "base_science_value", "visibility_valid",
            "ephemeris_uncertainty", "event_index",
        ])

        if n_active == 0:
            pad  = _make_padding_rows(self._n_actions, _FALLBACK_COLS)
            mask = np.zeros(self._n_actions, dtype=bool)
            return pad, mask

        # ------------------------------------------------------------------
        # Step 1: vectorised slew → t_arrive for every active target
        # ------------------------------------------------------------------
        active_tids = self._active_target_ids          # list[str], len = n_active
        target_rows = [self._state._target_lookup.get(tid) for tid in active_tids]
        valid_flags = np.array([r is not None for r in target_rows], dtype=bool)

        ra_arr  = np.array([float(r["ra"])  if r is not None else 0.0 for r in target_rows])
        dec_arr = np.array([float(r["dec"]) if r is not None else 0.0 for r in target_rows])

        slews     = slew_time_days_vec(
            self._state.current_ra, self._state.current_dec, ra_arr, dec_arr
        )
        t_arrives = t_now + slews          # shape (n_active,)

        # ------------------------------------------------------------------
        # Step 2: populate the backend step-cache with a large pool
        # ------------------------------------------------------------------
        pool_size = max(n_active * 20, 200)
        pool_df   = self._backend.candidates(t_now, pool_size)

        fallback_cols = pool_df.columns if len(pool_df) > 0 else _FALLBACK_COLS

        # ------------------------------------------------------------------
        # Step 3: vectorised first-reachable-event selection via merge+groupby
        # ------------------------------------------------------------------
        best_dict: dict[str, dict] = {}   # tid → best event row dict

        if len(pool_df) > 0:
            # Build a t_arrive lookup table (one row per active target)
            t_arr_df = pd.DataFrame({
                "target_id": active_tids,
                "t_arrive":  t_arrives,
            })

            # Add block_end to pool (vectorised)
            p = pool_df.copy()
            bd_arr = p["block_duration_days"].to_numpy(float)
            dd_arr = p["duration_days"].to_numpy(float)
            zero_bd = bd_arr <= 0.0
            bd_arr = np.where(zero_bd, COST_FACTOR * dd_arr, bd_arr)
            wm_arr = p["window_mid"].to_numpy(float)
            p["block_end"] = wm_arr + bd_arr / 2.0
            p["block_start"] = wm_arr - bd_arr / 2.0
            p["target_id"] = p["target_id"].astype(str)

            # Merge: each pool event gets its target's t_arrive
            p = p.merge(t_arr_df, on="target_id", how="inner")

            # Filter to reachable events, then take the earliest per target
            reachable = p[p["block_end"] > p["t_arrive"]]
            if exclude_started:
                reachable = reachable[reachable["block_start"] > t_now]
            if len(reachable) > 0:
                best_rows = (
                    reachable
                    .sort_values("window_mid")
                    .groupby("target_id", sort=False)
                    .first()
                    .reset_index()
                )
                # to_dict('records') is ~8× faster than iterrows() + to_dict()
                for rec in best_rows.to_dict("records"):
                    best_dict[str(rec["target_id"])] = rec

        # ------------------------------------------------------------------
        # Step 4: fallback for long-period targets not found in the pool
        # ------------------------------------------------------------------
        tid_to_idx = {tid: i for i, tid in enumerate(active_tids)}
        for i, tid in enumerate(active_tids):
            if not valid_flags[i] or tid in best_dict:
                continue
            t_arrive = t_arrives[i]
            future = self._backend.events_for_target(tid, t_now, n=20)
            for fev in future:
                bd    = float(fev.get("block_duration_days", 0.0))
                if bd <= 0.0:
                    bd = COST_FACTOR * float(fev.get("duration_days", 0.0))
                wm    = float(fev.get("window_mid", 0.0))
                block_start = wm - bd / 2.0
                if exclude_started and block_start <= t_now:
                    continue
                if wm + bd / 2.0 > t_arrive:
                    eid = self._backend.register_event({**fev, "target_id": tid})
                    if eid >= 0:
                        best_ev = {**fev, "event_id": eid, "target_id": tid}
                        best_ev.setdefault("tier_goal",            1)
                        best_ev.setdefault("base_science_value",   1.0)
                        best_ev.setdefault("visibility_valid",     True)
                        best_ev.setdefault("ephemeris_uncertainty", 0.0)
                        best_ev.setdefault("event_index",          -1)
                        best_ev.setdefault("duration",
                                           best_ev.get("duration_days", 0.0) * 86400.0)
                        best_dict[tid] = best_ev
                    break

        # ------------------------------------------------------------------
        # Step 5: assemble rows in _active_target_ids order → DataFrame
        # ------------------------------------------------------------------
        rows: list[dict] = []
        for i, tid in enumerate(active_tids):
            if not valid_flags[i]:
                rows.append(_sentinel_event(tid, fallback_cols))
            elif tid in best_dict:
                rows.append(best_dict[tid])
            else:
                rows.append(_sentinel_event(tid, fallback_cols))

        candidates = pd.DataFrame(rows)
        for col in fallback_cols:
            if col not in candidates.columns:
                candidates[col] = 0 if col != "target_id" else ""
        candidates = candidates.reindex(columns=fallback_cols, fill_value=0)

        if exclude_started and len(candidates) > 0:
            wm = candidates["window_mid"].to_numpy(float)
            bd = candidates["block_duration_days"].to_numpy(float)
            dd = candidates["duration_days"].to_numpy(float)
            zero_bd = bd <= 0.0
            bd = np.where(zero_bd, COST_FACTOR * dd, bd)
            block_start = wm - bd / 2.0
            eid = candidates["event_id"].to_numpy()
            keep = (eid >= 0) & (block_start > t_now)
            candidates = candidates.iloc[keep].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 5.5 (optional): fast top-K pre-filter by event urgency
        # ------------------------------------------------------------------
        # When k_filter > 0, keep only the K planets whose first-reachable event
        # has the soonest window_mid — i.e. the K most urgent opportunities.
        # This is the same principle as top-K event mode: "these windows are
        # closing soon, so the agent must reason about them right now."
        # The ISAB then decides which of the K to actually observe.
        # Reduces ISAB token count from N_max (~814) → K for ~(N/K)× GPU speedup.
        k_filter = self.cfg.action.full_set.k_filter
        if k_filter > 0 and len(candidates) > k_filter:
            wm = candidates["window_mid"].to_numpy(float)
            # Sentinel events (no valid event found) have window_mid=0, which
            # is in the past.  Sorting naively by window_mid would place them
            # first — exactly backwards.  Replace past/sentinel window_mids
            # with np.inf so they sort LAST, matching top-K semantics:
            # "K planets with the soonest FUTURE event windows."
            wm_sort = np.where(wm > t_now, wm, np.inf)
            # argpartition finds K smallest (= K soonest future windows) in O(N)
            top_k_idx = np.argpartition(wm_sort, k_filter)[:k_filter]
            # Sort ascending so slot 0 = most urgent
            top_k_idx = top_k_idx[np.argsort(wm_sort[top_k_idx])]
            candidates = candidates.iloc[top_k_idx].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 6: mask + pad to _n_actions
        # ------------------------------------------------------------------
        mask  = compute_mask(self._state, candidates, self.cfg.action)
        n_real = len(candidates)
        if n_real < self._n_actions:
            extra = _make_padding_rows(self._n_actions - n_real, candidates.columns)
            candidates = pd.concat([candidates, extra], ignore_index=True)
            mask = np.concatenate([mask, np.zeros(self._n_actions - n_real, dtype=bool)])

        return candidates.reset_index(drop=True), mask

    def _candidates_target(self) -> tuple[pd.DataFrame, np.ndarray]:
        """One next-event per target, ordered to match target table index.

        Works with both TableBackend (queries self.events) and DynamicBackend
        (calls backend.candidates for a large window, then picks first per target).
        """
        from aRieL.simulator.event_backend import TableBackend
        t_now = self._state.clock.current_time
        n_targets = len(self._targets)

        if isinstance(self._backend, TableBackend) and len(self._events) > 0:
            # Original table-based path
            rows = []
            fallback_cols = self._events.columns
            for _, trow in self._targets.iterrows():
                tid = trow["target_id"]
                nxt = self._state.next_event_for_target(tid)
                if nxt is not None:
                    rows.append(nxt.to_dict())
                else:
                    rows.append(_sentinel_event(tid, fallback_cols))
        else:
            # DynamicBackend path: fetch a large window and pick first per target
            # Fetch 3× the target count to ensure coverage of all targets
            pool = self._backend.candidates(t_now, n_targets * 3)
            # Build target_id → first upcoming event mapping
            seen: dict[str, dict] = {}
            for _, ev in pool.iterrows():
                tid = ev["target_id"]
                if tid not in seen:
                    seen[tid] = ev.to_dict()

            fallback_cols = pd.Index(
                ["event_id", "target_id", "event_type",
                 "window_start", "window_mid", "window_end",
                 "duration", "duration_days", "block_duration_days", "tier_goal",
                 "base_science_value", "visibility_valid",
                 "ephemeris_uncertainty", "event_index"]
            )
            rows = []
            for _, trow in self._targets.iterrows():
                tid = trow["target_id"]
                if tid in seen:
                    rows.append(seen[tid])
                else:
                    rows.append(_sentinel_event(tid, fallback_cols))

        candidates = pd.DataFrame(rows)
        mask = compute_mask(self._state, candidates, self.cfg.action)
        # Pad to fixed action-space size when catalogue growth reserved slots.
        n_real = len(candidates)
        if n_real < self._n_actions:
            extra = _make_padding_rows(self._n_actions - n_real, candidates.columns)
            candidates = pd.concat([candidates, extra], ignore_index=True)
            mask = np.concatenate([mask, np.zeros(self._n_actions - n_real, dtype=bool)])
        return candidates.reset_index(drop=True), mask

    def _skip_to_next_feasible_topk(
        self, max_lookahead: int = 10
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Top-K mode only: look further ahead when all K candidates are expired.

        Progressively asks the backend for larger candidate windows until a
        valid action is found or ``max_lookahead`` attempts are exhausted.
        The clock does NOT advance here — it advances only when the agent
        executes the chosen action via ``execute_observation``.

        This method must NOT be called in ``target`` or ``full_set`` modes
        because those modes return exactly one candidate per target and there
        is no "larger window" concept.  An all-invalid mask in those modes
        means the episode should terminate.
        """
        k = self.cfg.action.topk.k
        t_now = self._state.clock.current_time
        exclude = bool(getattr(self.cfg.action.topk, "exclude_started_blocks", False))

        for multiplier in range(2, 48):
            bigger_k = k * multiplier
            candidates = self._backend.candidates(
                t_now, bigger_k, unstarted_only=exclude,
            )
            if len(candidates) == 0:
                break
            if exclude:
                candidates = self._select_fully_catchable_topk(candidates, k)
                if len(candidates) == 0:
                    continue  # try a larger unstarted pool
            if len(candidates) == 0:
                break
            # Pad to k for mask computation when we already trimmed
            if len(candidates) < k:
                cols = candidates.columns
                padding = _make_padding_rows(k - len(candidates), cols)
                candidates = pd.concat([candidates, padding], ignore_index=True)

            from aRieL.envs.action_mask import compute_mask
            mask = compute_mask(self._state, candidates, self.cfg.action)

            if mask.any():
                if not exclude:
                    # Found feasible events — trim back to k, keeping the valid ones first
                    valid_idx = np.where(mask)[0][:k]
                    invalid_idx = np.where(~mask)[0]
                    keep = np.concatenate([valid_idx, invalid_idx])[:k]
                    candidates = candidates.iloc[keep].reset_index(drop=True)
                    mask = mask[keep]
                    # Pad back to k if needed
                    if len(candidates) < k:
                        cols = candidates.columns
                        padding = _make_padding_rows(k - len(candidates), cols)
                        candidates = pd.concat([candidates, padding], ignore_index=True)
                        mask = np.concatenate([mask, np.zeros(k - len(mask), dtype=bool)])
                return candidates, mask

        # Nothing found — return current (all invalid) candidates
        return self._candidates, self._action_mask

    def _action_to_event_id(self, action: int) -> int:
        """Convert an action index to an event_id in the event table."""
        if self.cfg.action.type in ("topk", "target", "full_set"):
            row = self._candidates.iloc[action]
            return int(row["event_id"])
        raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict:
        """Build the agent observation dict for the current state.

        In ``full_set`` mode the observation contains ``"planets"`` (N × F)
        and ``"global"``.  In ``topk`` / ``target`` modes it contains
        ``"events"`` (K × E) and ``"global"``.
        """
        from aRieL.envs.observation_builder import _build_global
        if self.cfg.action.type == "full_set":
            # When k_filter is active, _candidates already contains only the
            # top-K filtered rows (real + padding).  Derive active_ids from
            # those rows so that the planet feature array matches the reduced
            # observation space shape (_n_actions = k_filter, not N_max).
            # Without k_filter, use the full _active_target_ids list as before.
            _k_filter = self.cfg.action.full_set.k_filter
            _exclude_started = self.cfg.action.full_set.exclude_started_blocks
            if (
                (_k_filter > 0 or _exclude_started)
                and self._candidates is not None
                and len(self._candidates) > 0
            ):
                active_ids = [
                    str(r["target_id"])
                    for r in self._candidates.to_dict("records")
                    if r.get("target_id") and str(r["target_id"]) not in ("", "0")
                ]
            else:
                active_ids = self._active_target_ids   # full set (no pre-filter)

            # Build target_id → event dict from the pre-computed candidates
            # (candidates are in active-target order so rows 0…n_active-1 are real).
            per_target_events: dict[str, dict] | None = None
            if self._candidates is not None and len(self._candidates) > 0:
                n_real = len(active_ids)
                # Only look at real (non-padding) rows; to_dict('records') is
                # ~8× faster than iterrows() + to_dict() on 814-row DataFrames.
                per_target_events = {
                    str(rec["target_id"]): rec
                    for rec in self._candidates.iloc[:n_real].to_dict("records")
                    if rec.get("target_id")
                }

            # Prepare correctly-shaped static feature slice for active targets only.
            static_for_active: np.ndarray | None = None
            if self._static_planet_features is not None and self._tid_to_cache_idx:
                indices = [
                    self._tid_to_cache_idx[tid]
                    for tid in active_ids
                    if tid in self._tid_to_cache_idx
                ]
                if indices:
                    static_for_active = self._static_planet_features[indices]

            if active_ids:
                planet_arr = build_planet_features(
                    self._state,
                    static_features=static_for_active,
                    per_target_events=per_target_events,
                    target_ids=active_ids,
                )
            else:
                planet_arr = np.zeros((0, N_PLANET_FEATURES), dtype=np.float32)

            # Pad to _n_actions rows with zeros (padding positions)
            n_real = planet_arr.shape[0]
            if n_real < self._n_actions:
                padding = np.zeros(
                    (self._n_actions - n_real, N_PLANET_FEATURES), dtype=np.float32
                )
                planet_arr = np.concatenate([planet_arr, padding], axis=0)
            global_arr = _build_global(self._state, self.cfg.observation)
            return {"planets": planet_arr, "global": global_arr}
        return build_obs(self._state, self._candidates, self.cfg.observation)

    def _compute_reward(
        self,
        step_result: dict,
        bin_observed_before: dict[str, int] | None = None,
        host_tier1_before: dict[str, int] | None = None,
    ) -> float:
        """Compute the per-step reward for the current step."""
        if step_result is None:
            return 0.0
        bin_observed_after = self._state.population_bin_counts
        bin_totals = self._state._bin_totals_reward or self._state._bin_totals
        before = bin_observed_before or bin_observed_after

        if self._reward_fn is not None:
            info = reward_info_from_step(
                step_result,
                bin_totals=bin_totals,
                bin_observed_before=before,
                bin_observed_after=bin_observed_after,
                host_tier1_before=host_tier1_before,
            )
            return float(self._reward_fn(info))

        return compute_reward(
            step_result=step_result,
            cfg=self.cfg.reward,
            bin_totals=bin_totals,
            bin_observed_before=before,
            bin_observed_after=bin_observed_after,
            host_tier1_counts=host_tier1_before,
        )

    def _host_tier1_counts(self) -> dict[str, int]:
        """Return a dict of {host_id: n_tier1_completed_targets} for the current state."""
        counts: dict[str, int] = {}
        if "host_id" not in self._targets.columns:
            return counts
        for tid, prog in self._state._progress_dict.items():
            if int(prog.get("current_tier", 0)) >= 1:
                target_row = self._state._target_lookup.get(tid)
                if target_row is not None:
                    hid = str(target_row.get("host_id", ""))
                    if hid:
                        counts[hid] = counts.get(hid, 0) + 1
        return counts

    def _init_t2_milestone_pool(self) -> None:
        """Freeze the T2-milestone eligible set and denominator at episode start.

        Pool selection is controlled by ``reward.t2_milestone_pool``:
          * ``max_tier_eq_2`` — hard T2s only (excludes cheap T3-capable targets)
          * ``max_tier_ge_2`` — all T2-capable targets
          * ``initial_catalogue`` — every initial target (same denom style as T1)
        """
        fracs = tuple(self.cfg.reward.t2_milestone_fractions)
        if not fracs:
            self._t2_milestone_pool_ids = set()
            self._n_t2_milestone_denom = 0
            self._t2_episode_milestones = ()
            return

        pool = self.cfg.reward.t2_milestone_pool
        tids = self._targets["target_id"].astype(str)
        max_tier = self._targets["max_tier"].astype(int)
        if pool == "max_tier_eq_2":
            mask = max_tier == 2
        elif pool == "max_tier_ge_2":
            mask = max_tier >= 2
        elif pool == "initial_catalogue":
            mask = max_tier >= 0  # all initial targets
        else:
            raise ValueError(
                f"Unknown t2_milestone_pool={pool!r}; "
                "expected 'max_tier_eq_2', 'max_tier_ge_2', or 'initial_catalogue'."
            )

        self._t2_milestone_pool_ids = set(tids[mask].tolist())
        self._n_t2_milestone_denom = len(self._t2_milestone_pool_ids)

        step = float(self.cfg.reward.t2_milestone_step_above_one)
        if step > 0.0 and self._n_t2_milestone_denom > 0:
            # Allow overflow milestones when catalogue growth can add more
            # eligible targets than the frozen initial denominator.
            n_extra = 0
            if self.cfg.catalogue_growth.enabled and len(self._tpc_pool) > 0:
                if "max_tier" in self._tpc_pool.columns:
                    mt = self._tpc_pool["max_tier"].astype(int)
                    if pool == "max_tier_eq_2":
                        n_extra = int((mt == 2).sum())
                    elif pool == "max_tier_ge_2":
                        n_extra = int((mt >= 2).sum())
                    else:
                        n_extra = len(self._tpc_pool)
            max_frac = (self._n_t2_milestone_denom + n_extra) / self._n_t2_milestone_denom
            self._t2_episode_milestones = tuple(
                expand_milestone_fractions(
                    fracs, max_fraction=max_frac, step_above_one=step
                )
            )
        else:
            self._t2_episode_milestones = fracs

    def _count_t2_in_milestone_pool(self) -> int:
        """How many pool targets currently have Tier 2 complete."""
        if not self._t2_milestone_pool_ids or self._state is None:
            return 0
        n = 0
        for tid in self._t2_milestone_pool_ids:
            prog = self._state._progress_dict.get(tid)
            if prog is not None and bool(prog.get("tier2_done", False)):
                n += 1
            else:
                # Injected pool members may only exist in the live progress table.
                row = self._state.progress.loc[tid] if tid in self._state.progress.index else None
                if row is not None and bool(row.get("tier2_done", False)):
                    n += 1
        # Also count injected targets that match the pool rule and have T2 done,
        # so overflow milestones (>100%) can fire when growth is enabled.
        if self.cfg.catalogue_growth.enabled and self._n_t2_milestone_denom > 0:
            pool = self.cfg.reward.t2_milestone_pool
            for tid, prog in self._state._progress_dict.items():
                if tid in self._t2_milestone_pool_ids:
                    continue
                if not bool(prog.get("tier2_done", False)):
                    continue
                trow = self._state._target_lookup.get(tid)
                if trow is None:
                    continue
                mt = int(trow.get("max_tier", 0))
                if pool == "max_tier_eq_2" and mt == 2:
                    n += 1
                elif pool == "max_tier_ge_2" and mt >= 2:
                    n += 1
                elif pool == "initial_catalogue":
                    n += 1
        return n

    def _apply_relative_reward(self, abs_reward: float, terminated: bool) -> float:
        """Convert an absolute reward into a checkpoint-based relative reward.

        Accumulates ``abs_reward`` internally and emits rewards only at two
        types of mission-time checkpoints:

        * **Comparison intervals** (every ``comparison_interval_days``):
          ``comparison_scale × (agent_interval_acc − baseline_interval_mean)``
          Measures how much better the agent did in this short window compared
          to the baseline.

        * **Compound checkpoints** (every ``compound_interval_days``):
          ``compound_scale × (agent_total_acc − baseline_cumulative_at_checkpoint)``
          Measures cumulative advantage over the baseline so far — compounds as
          the agent consistently outperforms.

        At episode termination any remaining partial comparison interval is
        flushed so the agent always receives a signal for the final stretch.
        """
        cfg = self.cfg.reward
        traj = self._baseline_traj

        self._rel_interval_acc += abs_reward
        self._rel_total_acc += abs_reward

        reward = 0.0
        t_now = self._state.clock.current_time

        interval_rewards: list = traj.get("interval_rewards", [])
        compound_rewards: list = traj.get("compound_cumulative_rewards", [])

        # ---- emit comparison intervals that the clock has crossed ----
        while t_now >= self._next_comparison_bjd:
            baseline_iv = (
                float(interval_rewards[self._rel_comparison_idx])
                if self._rel_comparison_idx < len(interval_rewards)
                else 0.0
            )
            reward += cfg.comparison_scale * (self._rel_interval_acc - baseline_iv)
            self._rel_interval_acc = 0.0
            self._rel_comparison_idx += 1
            self._next_comparison_bjd += cfg.comparison_interval_days

        # ---- emit compound checkpoints that the clock has crossed ----
        while t_now >= self._next_compound_bjd:
            baseline_cum = (
                float(compound_rewards[self._rel_compound_idx])
                if self._rel_compound_idx < len(compound_rewards)
                else float(traj.get("total_mean_reward", 0.0))
            )
            reward += cfg.compound_scale * (self._rel_total_acc - baseline_cum)
            self._rel_compound_idx += 1
            self._next_compound_bjd += cfg.compound_interval_days

        # ---- at termination flush the remaining partial comparison interval ----
        if terminated and self._rel_interval_acc != 0.0:
            baseline_iv = (
                float(interval_rewards[self._rel_comparison_idx])
                if self._rel_comparison_idx < len(interval_rewards)
                else 0.0
            )
            reward += cfg.comparison_scale * (self._rel_interval_acc - baseline_iv)
            self._rel_interval_acc = 0.0

        return reward

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _make_info(self, step_result: Optional[dict]) -> dict:
        info: dict[str, Any] = {
            "action_mask":    self._action_mask,
            "step_count":     self._step_count,
            "mission_summary": self._state.summary() if self._state else {},
            "invalid_action": False,
            "n_scheduled_injections": len(self._injection_schedule),
            "n_injected": self._injection_cursor,
            "n_t3_revisits_fired": self._n_t3_revisits_fired,
            "n_initial_targets": (
                self._state.reference_total_targets if self._state else len(self._targets)
            ),
        }
        if step_result is not None:
            info["step_result"] = step_result
        return info

    def export_events(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> pd.DataFrame:
        """Export a DynamicBackend event table for analysis / plotting.

        Defaults to the configured mission window.  The RL loop does not
        use this table.
        """
        if not isinstance(self._backend, DynamicBackend):
            raise TypeError(
                "export_events() requires DynamicBackend "
                f"(got {type(self._backend).__name__})"
            )
        start = (
            float(t_start)
            if t_start is not None
            else float(self.cfg.mission.start_bjd)
        )
        end = (
            float(t_end)
            if t_end is not None
            else start + float(self.cfg.mission.lifetime_days)
        )
        return self._backend.export_events(start, end)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def state(self) -> Optional[MissionState]:
        """Direct access to the simulator state (useful for debugging)."""
        return self._state

    @property
    def action_mask(self) -> Optional[np.ndarray]:
        return self._action_mask


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_padding_rows(n: int, columns: pd.Index) -> pd.DataFrame:
    """Create n zero-filled dummy rows to pad the candidate table to K."""
    dummy = {col: [0] * n for col in columns}
    dummy["visibility_valid"] = [False] * n
    dummy["event_id"] = [-1] * n
    dummy["target_id"] = [""] * n
    dummy["window_end"] = [0.0] * n
    dummy["window_mid"] = [0.0] * n
    dummy["duration_days"] = [0.0] * n
    dummy["block_duration_days"] = [0.0] * n
    dummy["duration"] = [0.0] * n
    return pd.DataFrame(dummy)[columns]


def _sentinel_event(target_id: str, columns: pd.Index) -> dict:
    """A dummy event row for a target with no upcoming events."""
    row = {col: 0 for col in columns}
    row["target_id"] = target_id
    row["event_id"] = -1
    row["visibility_valid"] = False
    row["window_end"] = 0.0
    row["window_mid"] = 0.0
    row["duration_days"] = 0.0
    row["block_duration_days"] = 0.0
    row["duration"] = 0.0
    return row
