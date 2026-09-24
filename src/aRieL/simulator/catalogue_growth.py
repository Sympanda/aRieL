"""
Deterministic mid-episode catalogue growth (TESS → TPC confirmations).

At ``reset(seed)`` the environment builds an exogenous injection schedule
keyed by mission BJD.  The same seed always yields the same
``(injection_bjd, target_id)`` pairs, independent of the agent's actions,
so models can be compared fairly under identical catalogue growth.

Injections are applied in ``ArielEnv.step`` when the mission clock crosses
each scheduled BJD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InjectionEvent:
    """One planet becoming available at a mission time."""
    bjd: float
    target_id: str


def build_injection_schedule(
    pool: pd.DataFrame,
    *,
    mission_start: float,
    mission_end: float,
    rng: np.random.Generator,
    mean_per_month: float = 15.0,
    std_per_month: float = 5.0,
    month_days: float = 30.4375,
) -> list[InjectionEvent]:
    """Sample a deterministic injection schedule from a TPC candidate pool.

    Parameters
    ----------
    pool:
        Preprocessed target rows not already in the active MCS catalogue.
        Must contain a ``target_id`` column.
    mission_start, mission_end:
        Mission window in BJD.
    rng:
        Seeded ``np.random.Generator`` — sole source of randomness.
    mean_per_month, std_per_month:
        Monthly confirmation rate ~ ``max(0, round(N(μ, σ)))``.
    month_days:
        Length of a "month" in mission days (default ≈ mean Gregorian month).

    Returns
    -------
    list[InjectionEvent]
        Sorted by ``bjd``.  Target IDs are sampled without replacement.
    """
    if pool is None or len(pool) == 0:
        return []

    ids = pool["target_id"].astype(str).tolist()
    rng.shuffle(ids)
    id_iter = iter(ids)

    events: list[InjectionEvent] = []
    month_start = float(mission_start)
    while month_start < mission_end and ids:
        month_end = min(month_start + month_days, mission_end)
        n_raw = rng.normal(mean_per_month, std_per_month)
        n = int(max(0, round(n_raw)))
        # Uniform arrival times within the month (mission time).
        if n > 0 and month_end > month_start:
            times = np.sort(rng.uniform(month_start, month_end, size=n))
            for t in times:
                try:
                    tid = next(id_iter)
                except StopIteration:
                    break
                events.append(InjectionEvent(bjd=float(t), target_id=str(tid)))
            else:
                month_start = month_end
                continue
            break  # pool exhausted
        month_start = month_end

    events.sort(key=lambda e: e.bjd)
    return events


def expand_milestone_fractions(
    base: tuple[float, ...] | list[float],
    max_fraction: float,
    step_above_one: float = 0.1,
) -> tuple[float, ...]:
    """Extend milestone list every ``step_above_one`` above 1.0 up to *max_fraction*.

    Example: base ends at 1.0, max=1.47, step=0.1 → …, 1.0, 1.1, 1.2, 1.3, 1.4.
    """
    fracs = sorted({float(f) for f in base})
    if step_above_one <= 0.0 or max_fraction <= 1.0:
        return tuple(fracs)

    # Start at the first step strictly above 1.0 (or above the largest base ≥1).
    f = 1.0 + step_above_one
    # Quantize to avoid 1.2000000001 drift.
    n_steps = int(np.floor((max_fraction - 1.0) / step_above_one + 1e-9))
    for i in range(1, n_steps + 1):
        f = round(1.0 + i * step_above_one, 10)
        if f <= max_fraction + 1e-9:
            fracs.append(f)
    return tuple(sorted(set(fracs)))


def resolve_tpc_path(path: str | Path | None) -> Optional[Path]:
    """Resolve a TPC CSV path; ``None`` / empty → search ``$ARIEL_DATA`` / ``./data/raw``."""
    from aRieL.utils.paths import find_data_csv

    if path is None or str(path).strip() == "":
        return find_data_csv(
            [
                "TPC.csv",
                "Ariel_MCS_TPCs_2025-08-18.csv",
                "Ariel_MCS_TPCs_2025-08-18 copy.csv",
            ],
            extra_globs=False,
        )
    p = Path(path)
    if p.is_file():
        return p
    cwd_p = (Path.cwd() / p).resolve()
    return cwd_p if cwd_p.is_file() else None
