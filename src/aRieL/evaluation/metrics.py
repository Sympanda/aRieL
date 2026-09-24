"""
Episode statistics: everything you want to know after a completed episode.

``EpisodeStats`` is a frozen dataclass.  ``compute_stats(state)`` builds
it from a finished ``MissionState``.  The result can be logged, printed,
or collected across multiple runs for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState

# Imported lazily inside compute_stats to avoid circular imports at module level
_coverage_gini = None


def _get_coverage_gini():
    global _coverage_gini
    if _coverage_gini is None:
        from aRieL.evaluation.population_coverage import coverage_gini
        _coverage_gini = coverage_gini
    return _coverage_gini


@dataclass(frozen=True)
class EpisodeStats:
    """Complete statistics for one finished episode.

    Tier completion
    ---------------
    tier1/2/3_completed : int
        Number of targets that reached each tier.
    tier1/2/3_rate : float
        Fraction of the **initial** catalogue that reached each tier.
        With mid-episode catalogue growth this can exceed 1.0.
    catalogue_ceiling_rate : float
        ``total_targets / n_initial_targets`` — the theoretical maximum
        coverage rate if every planet present by episode end were completed.
        Equals 1.0 when growth is disabled.  Use as a dotted reference line
        on coverage plots.
    tier1/2/3_of_eligible : float
        Fraction of targets *eligible* for each tier that completed it
        (among the final live catalogue).
    """

    # Tier completion (counts)
    tier1_completed: int
    tier2_completed: int
    tier3_completed: int
    total_targets: int
    n_initial_targets: int

    # Tier completion (rates vs initial catalogue — may exceed 1.0)
    tier1_rate: float
    tier2_rate: float
    tier3_rate: float
    catalogue_ceiling_rate: float

    # Tier completion among eligible targets only (live catalogue)
    tier1_eligible: int
    tier2_eligible: int
    tier3_eligible: int
    tier1_of_eligible: float
    tier2_of_eligible: float
    tier3_of_eligible: float

    # Schedule quality
    n_observations: int
    n_missed: int
    miss_rate: float
    used_science_days: float
    used_slew_days: float
    used_idle_days: float       # time waiting on target before window opens
    science_efficiency: float   # science / (science + slew + idle)
    fraction_elapsed: float

    # Population coverage
    n_bins_total: int
    n_bins_with_t1: int
    bin_coverage: float
    coverage_gini_t1: float   # Gini over per-bin T1 counts (0=uniform, 1=monopoly)
    coverage_gini_t2: float
    bin_counts: dict = field(compare=False)

    # Programme-depth completion (disjoint max_tier buckets; each planet once)
    fully_completed: int = 0          # current_tier >= max_tier
    fully_completed_rate: float = 0.0  # / total_targets (final catalogue)
    prog_t1_completed: int = 0        # max_tier==1 and fully done
    prog_t2_completed: int = 0
    prog_t3_completed: int = 0
    prog_t1_eligible: int = 0         # count with max_tier==1
    prog_t2_eligible: int = 0
    prog_t3_eligible: int = 0
    prog_t1_rate: float = 0.0
    prog_t2_rate: float = 0.0
    prog_t3_rate: float = 0.0

    def summary_str(self) -> str:
        """One-paragraph human-readable summary."""
        ceil = self.catalogue_ceiling_rate
        lines = [
            f"Tier completion  : T1 {self.tier1_completed}/{self.n_initial_targets} "
            f"({self.tier1_rate:.1%} of initial),  "
            f"T2 {self.tier2_completed} ({self.tier2_rate:.1%}),  "
            f"T3 {self.tier3_completed} ({self.tier3_rate:.1%})",
            f"Catalogue        : {self.n_initial_targets} initial → "
            f"{self.total_targets} final  (ceiling {ceil:.1%})",
            f"Eligible rates   : T1 {self.tier1_of_eligible:.1%}  "
            f"T2 {self.tier2_of_eligible:.1%}  T3 {self.tier3_of_eligible:.1%}",
            f"Observations     : {self.n_observations} executed, {self.n_missed} missed "
            f"(miss rate {self.miss_rate:.1%})",
            f"Time             : {self.used_science_days:.1f}d science, "
            f"{self.used_slew_days:.1f}d slew  "
            f"(efficiency {self.science_efficiency:.1%})",
            f"Population bins  : {self.n_bins_with_t1}/{self.n_bins_total} covered "
            f"({self.bin_coverage:.1%})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("bin_counts", None)   # exclude large nested dict by default
        return d


def compute_stats(state: "MissionState") -> EpisodeStats:
    """Build an EpisodeStats from a finished (or mid-episode) MissionState.

    Parameters
    ----------
    state:
        The MissionState after ``env.step()`` returned ``terminated=True``,
        or at any point during an episode for intermediate inspection.

    Returns
    -------
    EpisodeStats
    """
    clk = state.clock
    targets = state.targets
    progress = state.progress

    n_total = len(targets)
    n_initial = int(getattr(state, "n_initial_targets", 0) or n_total)
    if n_initial <= 0:
        n_initial = n_total

    # ---- tier counts ----
    t1_done = int(progress["tier1_done"].sum())
    t2_done = int(progress["tier2_done"].sum())
    t3_done = int(progress["tier3_done"].sum())

    # ---- eligible counts ----
    t1_elig = n_total                                        # everyone is eligible for T1
    t2_elig = int((targets["max_tier"] >= 2).sum())
    t3_elig = int((targets["max_tier"] >= 3).sum())

    def safe_rate(n: int, d: int) -> float:
        return n / d if d > 0 else 0.0

    # ---- schedule quality ----
    n_obs  = clk.n_observations
    n_miss = clk.n_missed
    total_attempts = n_obs + n_miss
    miss_rate = safe_rate(n_miss, total_attempts)

    sci_days  = clk.used_science_time
    slew_days = clk.used_slew_time
    idle_days = clk.used_idle_time          # time waiting on target before window opens
    # Efficiency = science / (science + slew + idle).
    # Idle time is included: arriving early and waiting is a real cost.
    active  = sci_days + slew_days + idle_days
    sci_eff = safe_rate(sci_days, active)

    # ---- population coverage ----
    bin_counts = state.population_bin_counts      # bins with ≥1 T1 completed
    all_bins   = sorted(targets["population_bin"].unique())
    n_bins     = len(all_bins)
    n_covered  = sum(1 for b in all_bins if bin_counts.get(b, 0) > 0)

    cov_gini = _get_coverage_gini()

    # ---- programme-depth (disjoint) + fully complete ----
    # Each planet contributes to exactly one programme bucket (its max_tier).
    t_ids = targets["target_id"].astype(str)
    mx = (
        targets.set_index(t_ids)["max_tier"]
        .fillna(1)
        .astype(int)
    )
    ct = progress["current_tier"].copy()
    ct.index = ct.index.astype(str)
    ct = ct.reindex(mx.index).fillna(0).astype(int)
    fully = ct >= mx
    n_fully = int(fully.sum())
    prog_n = {
        1: int(((mx == 1) & fully).sum()),
        2: int(((mx == 2) & fully).sum()),
        3: int(((mx == 3) & fully).sum()),
    }
    prog_e = {
        1: int((mx == 1).sum()),
        2: int((mx == 2).sum()),
        3: int((mx == 3).sum()),
    }

    return EpisodeStats(
        tier1_completed=t1_done,
        tier2_completed=t2_done,
        tier3_completed=t3_done,
        total_targets=n_total,
        n_initial_targets=n_initial,
        tier1_rate=safe_rate(t1_done, n_initial),
        tier2_rate=safe_rate(t2_done, n_initial),
        tier3_rate=safe_rate(t3_done, n_initial),
        catalogue_ceiling_rate=safe_rate(n_total, n_initial),
        tier1_eligible=t1_elig,
        tier2_eligible=t2_elig,
        tier3_eligible=t3_elig,
        tier1_of_eligible=safe_rate(t1_done, t1_elig),
        tier2_of_eligible=safe_rate(t2_done, t2_elig),
        tier3_of_eligible=safe_rate(t3_done, t3_elig),
        n_observations=n_obs,
        n_missed=n_miss,
        miss_rate=miss_rate,
        used_science_days=sci_days,
        used_slew_days=slew_days,
        used_idle_days=idle_days,
        science_efficiency=sci_eff,
        fraction_elapsed=clk.fraction_elapsed,
        n_bins_total=n_bins,
        n_bins_with_t1=n_covered,
        bin_coverage=safe_rate(n_covered, n_bins),
        coverage_gini_t1=cov_gini(state, tier=1),
        coverage_gini_t2=cov_gini(state, tier=2),
        bin_counts=bin_counts,
        fully_completed=n_fully,
        fully_completed_rate=safe_rate(n_fully, n_total),
        prog_t1_completed=prog_n[1],
        prog_t2_completed=prog_n[2],
        prog_t3_completed=prog_n[3],
        prog_t1_eligible=prog_e[1],
        prog_t2_eligible=prog_e[2],
        prog_t3_eligible=prog_e[3],
        prog_t1_rate=safe_rate(prog_n[1], prog_e[1]),
        prog_t2_rate=safe_rate(prog_n[2], prog_e[2]),
        prog_t3_rate=safe_rate(prog_n[3], prog_e[3]),
    )
