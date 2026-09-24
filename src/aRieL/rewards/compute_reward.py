"""
Multi-component reward function for the Ariel mission scheduling environment.

Per-step reward components
--------------------------
1. **Tier completion bonus** (sparse)
   Fired once when a target crosses a tier boundary.  Scaled by
   ``science_weight × diversity_multiplier``.  Weights: T1=1, T2=3, T3=10.

2. **Progress shaping** (dense)
   Reward proportional to Δprogress_in_tier.  A near-completion multiplier
   (``near_completion_scale``) kicks in when ``progress_in_tier > threshold``
   so the agent is incentivised to *finish* targets rather than abandoning
   them just before a tier boundary.

3. **Efficiency reward** (dense)
   ``block_duration / total_cost`` — penalises long slews and idle waits
   implicitly.

4. **Rarity / difficulty bonus** (dense)
   ``rarity_weight × (period / period_ref)² / tier_worked``
   Rewards observations of long-period targets that are costly to miss.

5. **Coverage potential reward** (dense, marginal)
   ``coverage_weight × [U_pop(s_{t+1}) − U_pop(s_t)]``
   where ``U_pop(s) = Σ_b w_b · min(q_b / n_b, 1)`` and ``q_b`` is the
   desired quota per bin.  Fires only when a Tier-1 completion advances
   bin coverage; saturates once the quota is met.

6. **Unique host bonus** (sparse)
   Fired the first time a new planetary *system* has any target reach Tier 1.
   Rewards breadth across stellar systems.

7. **Comparative planetology bonus** (sparse)
   Fired when a Tier-1 target shares a host with at least one existing
   Tier-1+ sibling.  Rewards completing scientifically useful pairs/triples.

8. **Idle time penalty**
   Per-day cost for time the telescope spends waiting for an observation
   block to start after arriving on target early.

9. **Missed-event penalty**
   Subtracted when the agent arrives after ``window_end``.

One-shot per-episode components
--------------------------------
10. **Coverage milestone bonuses**
    Large one-shot bonuses when T1 completions cross coverage thresholds.
    Fired by ``ArielEnv`` via ``check_milestone_reward``.

11. **Terminal episode bonus**
    Fired at the end of each episode.  ``t1_terminal_weight × (t1_fraction)^power``
    — quadratic by default so near-complete T1 coverage is disproportionately
    valuable.

Public API
----------
    from aRieL.rewards.compute_reward import (
        compute_reward,
        check_milestone_reward,
        compute_terminal_reward,
        compute_coverage_potential,
    )
"""

from __future__ import annotations

from aRieL.utils.config import RewardConfig


# ---------------------------------------------------------------------------
# Coverage potential U_pop
# ---------------------------------------------------------------------------

def compute_coverage_potential(
    bin_observed: dict[str, int],
    bin_totals: dict[str, int],
    quota_per_bin: int,
) -> float:
    """Compute the population coverage potential U_pop(s).

    U_pop(s) = Σ_b  min(q_b / quota, 1)

    where q_b is the current number of Tier-1+ completions in bin b
    and ``quota`` is the desired number per bin.  Each bin contributes at
    most 1.0 to the sum (saturation), incentivising breadth without
    over-rewarding already-covered bins.

    Parameters
    ----------
    bin_observed:
        Current Tier-1+ completion count per bin (from MissionState).
    bin_totals:
        Total target count per bin (from MissionState._bin_totals).
    quota_per_bin:
        Desired number of Tier-1+ observations per bin (``coverage_quota_per_bin``).

    Returns
    -------
    float — sum over all bins, each in [0, 1].
    """
    if quota_per_bin <= 0:
        return 0.0
    total = 0.0
    for b in bin_totals:
        observed = bin_observed.get(b, 0)
        total += min(observed / quota_per_bin, 1.0)
    return total


def _diversity_multiplier(
    population_bin: str,
    bin_totals: dict[str, int],
    bin_observed: dict[str, int],
    max_multiplier: float = 5.0,
) -> float:
    """Return a coverage-diversity multiplier for *population_bin*.

    Ranges from ``max_multiplier`` (bin completely unseen) down to ``1.0``
    (bin fully saturated).  Used to scale tier-completion and progress rewards.
    """
    total = bin_totals.get(population_bin, 1)
    observed = bin_observed.get(population_bin, 0)
    fraction = observed / total if total > 0 else 0.0
    return 1.0 + (max_multiplier - 1.0) * max(0.0, 1.0 - fraction)


def compute_reward(
    step_result: dict,
    cfg: RewardConfig,
    bin_totals: dict[str, int],
    bin_observed_before: dict[str, int],
    bin_observed_after: dict[str, int],
    host_tier1_counts: dict[str, int] | None = None,
) -> float:
    """Compute the scalar reward for a completed environment step.

    Parameters
    ----------
    step_result:
        The dict returned by ``MissionState.execute_observation``.
    cfg:
        ``RewardConfig`` holding all weight hyper-parameters.
    bin_totals:
        Static per-bin target counts (``MissionState._bin_totals``).
    bin_observed_before:
        Per-bin Tier-1+ counts *before* this observation was executed.
    bin_observed_after:
        Per-bin Tier-1+ counts *after* this observation was executed
        (``MissionState.population_bin_counts``).
    host_tier1_counts:
        Optional dict mapping ``host_id → number of Tier-1+ completed targets
        in that system`` *before* this observation.  Used for unique-host and
        comparative rewards.  Pass ``None`` to disable those components.

    Returns
    -------
    float
        Scalar reward for this step.
    """
    missed: bool          = step_result.get("missed", False)
    idle_days: float      = float(step_result.get("idle_days", 0.0))
    obs_dur: float        = float(step_result.get("obs_duration_days", 0.0))
    slew_dur: float       = float(step_result.get("slew_days", 0.0))
    total_cost: float     = float(step_result.get("total_cost_days", obs_dur + slew_dur + idle_days))

    # ------------------------------------------------------------------ #
    # 0. Overhead penalty — fires on every step regardless of miss/success.
    #
    # Two modes (set by cfg.overhead_penalty_mode):
    #
    # "absolute"  penalty = weight × (slew + idle)^power
    #   Directly proportional to mission time lost.  A day wasted is a day
    #   that could have observed another planet, regardless of obs duration.
    #   No dilution: deep T3 observations get the same penalty for the same
    #   idle wait as a quick T1 observation.
    #
    # "fraction"  penalty = weight × ((slew + idle) / total_step)^power
    #   Legacy mode.  Gets diluted by long observations (deep T3 commits
    #   reduce overhead_frac even though absolute waste is unchanged).
    #
    # Legacy idle_penalty_per_day is additive on top if still set.
    # ------------------------------------------------------------------ #
    overhead_days = slew_dur + idle_days
    if cfg.overhead_penalty_weight > 0.0 and overhead_days > 0.0:
        mode = getattr(cfg, "overhead_penalty_mode", "fraction")
        if mode == "absolute":
            basis = overhead_days
        else:
            basis = overhead_days / total_cost if total_cost > 0.0 else 0.0
        reward = -(cfg.overhead_penalty_weight * (basis ** cfg.overhead_penalty_power))
    else:
        reward = 0.0

    # Legacy per-day idle penalty (kept for backward compat; set to 0 in new configs).
    reward -= cfg.idle_penalty_per_day * idle_days

    # Missed-event: pay the slew, get no science, small additional penalty.
    if missed:
        reward -= cfg.miss_penalty
        return reward

    science_weight: float  = float(step_result.get("science_weight", 0.5))
    population_bin: str    = str(step_result.get("population_bin", ""))
    tier_before: int       = int(step_result.get("tier_before", 0))
    tier_after: int        = int(step_result.get("tier_after", 0))
    progress_before: float = float(step_result.get("progress_before", 0.0))
    progress_after: float  = float(step_result.get("progress_after", 0.0))

    # Diversity multiplier — scales tier/progress rewards for under-observed bins.
    div_mult = _diversity_multiplier(
        population_bin, bin_totals, bin_observed_after, cfg.diversity_multiplier_max
    )
    # science_weight_tiers / science_weight_on_progress control whether catalogue
    # priority compounds on every tier (legacy) or only on selected tiers / progress.
    sw_tiers = set(getattr(cfg, "science_weight_tiers", (1, 2, 3)))
    sw_on_progress = bool(getattr(cfg, "science_weight_on_progress", True))

    # ------------------------------------------------------------------ #
    # 1. Tier completion bonus (sparse)
    # ------------------------------------------------------------------ #
    if tier_after > tier_before:
        _tier_weights = {
            1: cfg.tier1_completion,
            2: cfg.tier2_completion,
            3: cfg.tier3_completion,
        }
        for completed_tier in range(tier_before + 1, tier_after + 1):
            sw = science_weight if completed_tier in sw_tiers else 1.0
            reward += _tier_weights.get(completed_tier, 0.0) * sw * div_mult

    # ------------------------------------------------------------------ #
    # 2. Progress shaping (dense, only when no tier boundary is crossed)
    # ------------------------------------------------------------------ #
    if tier_before == tier_after:
        delta = progress_after - progress_before
        if delta > 0.0:
            near_boost = (
                cfg.near_completion_scale
                if progress_after >= cfg.near_completion_threshold
                else 1.0
            )
            sw = science_weight if sw_on_progress else 1.0
            reward += cfg.progress_weight * delta * sw * div_mult * near_boost

    # ------------------------------------------------------------------ #
    # 3. Efficiency reward (dense, legacy — prefer overhead_penalty instead)
    # Uses obs_duration / total_cost (linear, not power-law).
    # ------------------------------------------------------------------ #
    if cfg.efficiency_weight > 0.0 and total_cost > 0.0:
        efficiency = obs_dur / total_cost
        reward += cfg.efficiency_weight * efficiency

    # ------------------------------------------------------------------ #
    # 4. Rarity / difficulty bonus (dense)
    # ------------------------------------------------------------------ #
    if cfg.rarity_weight > 0.0:
        period: float = float(step_result.get("period", 0.0))
        if period > 0.0 and cfg.rarity_period_ref_days > 0.0:
            difficulty  = min(period / cfg.rarity_period_ref_days, 1.0)
            tier_worked = max(tier_before + 1, 1)
            reward += cfg.rarity_weight * (difficulty ** 2) / tier_worked

    # ------------------------------------------------------------------ #
    # 5. Coverage potential reward (marginal U_pop, fires on T1 completion)
    # ------------------------------------------------------------------ #
    if cfg.coverage_weight > 0.0 and tier_after >= 1 and tier_before < 1:
        u_before = compute_coverage_potential(
            bin_observed_before, bin_totals, cfg.coverage_quota_per_bin
        )
        u_after = compute_coverage_potential(
            bin_observed_after, bin_totals, cfg.coverage_quota_per_bin
        )
        reward += cfg.coverage_weight * max(0.0, u_after - u_before)

    # ------------------------------------------------------------------ #
    # 6 & 7. Host diversity rewards (unique host + comparative planetology)
    # Only apply when a new Tier-1 is reached and host info is available.
    # ------------------------------------------------------------------ #
    if host_tier1_counts is not None and tier_after >= 1 and tier_before < 1:
        host_id: str = str(step_result.get("host_id", ""))
        if host_id:
            n_siblings_before = host_tier1_counts.get(host_id, 0)
            if n_siblings_before == 0:
                # First Tier-1 in this planetary system — unique host bonus.
                reward += cfg.unique_host_weight
            else:
                # A sibling already exists — comparative planetology bonus.
                # Scale gently with the number of existing siblings.
                reward += cfg.comparative_weight * min(n_siblings_before, 3)

    # ------------------------------------------------------------------ #
    # 8. Random-baseline subtraction (optional)
    # ------------------------------------------------------------------ #
    if cfg.subtract_random_baseline:
        reward -= cfg.random_baseline_per_step

    return reward


# ---------------------------------------------------------------------------
# One-shot milestone bonuses (called by ArielEnv, not per-step)
# ---------------------------------------------------------------------------

def check_milestone_reward(
    n_completed: int,
    total_reachable: int,
    milestones_hit: set[float],
    milestone_fractions: tuple[float, ...] | list[float],
    *,
    intermediate_bonus: float,
    final_bonus: float = 0.0,
    overflow_bonus: float = 0.0,
) -> tuple[float, set[float]]:
    """Return a one-shot bonus if a new coverage milestone has been crossed.

    Generic helper used for both T1 and T2 coverage ladders.

    Coverage fraction uses ``total_reachable`` as denominator.  With catalogue
    growth this should be the **initial** eligible-pool size so fractions can
    exceed 1.0.

    Milestone bonuses
    -----------------
    * fraction < 1.0  → ``intermediate_bonus``
    * fraction == 1.0 → ``final_bonus`` if > 0, else ``2 × intermediate_bonus``
    * fraction > 1.0  → ``overflow_bonus`` if > 0, else ``intermediate_bonus``
    """
    if total_reachable == 0 or not milestone_fractions:
        return 0.0, milestones_hit

    fraction = n_completed / total_reachable
    bonus = 0.0

    for milestone in milestone_fractions:
        if milestone not in milestones_hit and fraction >= milestone:
            milestones_hit.add(milestone)
            if abs(milestone - 1.0) < 1e-12:
                bonus += final_bonus if final_bonus > 0.0 else intermediate_bonus * 2.0
            elif milestone > 1.0:
                bonus += overflow_bonus if overflow_bonus > 0.0 else intermediate_bonus
            else:
                bonus += intermediate_bonus

    return bonus, milestones_hit


def check_t1_milestone_reward(
    tier1_completed: int,
    total_reachable: int,
    milestones_hit: set[float],
    cfg: RewardConfig,
    milestone_fractions: tuple[float, ...] | list[float] | None = None,
) -> tuple[float, set[float]]:
    """T1 coverage milestone wrapper (keeps existing call sites working)."""
    fracs = (
        milestone_fractions
        if milestone_fractions is not None
        else cfg.t1_milestone_fractions
    )
    return check_milestone_reward(
        n_completed=tier1_completed,
        total_reachable=total_reachable,
        milestones_hit=milestones_hit,
        milestone_fractions=fracs,
        intermediate_bonus=cfg.t1_milestone_bonus,
        final_bonus=getattr(cfg, "t1_final_milestone_bonus", 0.0),
        overflow_bonus=getattr(cfg, "t1_overflow_milestone_bonus", 0.0),
    )


def check_t2_milestone_reward(
    tier2_completed: int,
    total_reachable: int,
    milestones_hit: set[float],
    cfg: RewardConfig,
    milestone_fractions: tuple[float, ...] | list[float] | None = None,
) -> tuple[float, set[float]]:
    """T2 coverage milestone wrapper.  No-op when ``t2_milestone_fractions`` is empty."""
    fracs = (
        milestone_fractions
        if milestone_fractions is not None
        else getattr(cfg, "t2_milestone_fractions", ())
    )
    if not fracs:
        return 0.0, milestones_hit
    return check_milestone_reward(
        n_completed=tier2_completed,
        total_reachable=total_reachable,
        milestones_hit=milestones_hit,
        milestone_fractions=fracs,
        intermediate_bonus=float(getattr(cfg, "t2_milestone_bonus", 0.0)),
        final_bonus=float(getattr(cfg, "t2_final_milestone_bonus", 0.0)),
        overflow_bonus=float(getattr(cfg, "t2_overflow_milestone_bonus", 0.0)),
    )


def compute_terminal_reward(
    tier1_completed: int,
    total_reachable: int,
    cfg: RewardConfig,
) -> float:
    """Terminal bonus fired once at the end of each episode.

    Scales as ``t1_terminal_weight × (t1_fraction)^t1_terminal_power``.
    ``total_reachable`` should be the **initial** catalogue size so that
    completing injected planets can yield a fraction > 1.
    """
    if total_reachable == 0:
        return 0.0
    t1_fraction = tier1_completed / total_reachable
    return cfg.t1_terminal_weight * (t1_fraction ** cfg.t1_terminal_power)
