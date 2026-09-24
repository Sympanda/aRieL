"""
Ephemeris propagation: given a reference epoch and orbital period, generate
all transit (or eclipse) mid-times that fall within a BJD time window.

For transits, the mid-time sequence is:
    t_n = epoch + n * period,   n = 0, ±1, ±2, …

For eclipses, the mid-time relative to transit is offset by half the period
plus a small correction for eccentricity (ω):
    t_ecl = t_transit + period/2 * (1 + e*cos(omega) * 4/π)
When eccentricity is unknown or zero, offset = period / 2 exactly.

The ephemeris uncertainty grows linearly with elapsed time:
    σ(t_n) = sqrt(σ_epoch² + (n * σ_period)²)   [days]

Both σ_epoch and σ_period are taken from the target table when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EphemerisResult:
    """Output of :func:`propagate`."""

    target_id: str
    event_type: str                   # "transit" or "eclipse"
    indices: np.ndarray               # integer n for each event
    mid_times: np.ndarray             # BJD float64
    uncertainties: np.ndarray         # 1-sigma, seconds


def propagate(
    target_id: str,
    epoch: float,
    period: float,
    t_start: float,
    t_end: float,
    event_type: str = "transit",
    eccentricity: float = 0.0,
    omega_deg: float = 0.0,
    sigma_epoch_days: float = 0.0,
    sigma_period_days: float = 0.0,
) -> EphemerisResult:
    """Generate all event mid-times within [t_start, t_end].

    Parameters
    ----------
    target_id:
        Identifier (for labelling purposes only).
    epoch:
        Reference transit mid-time in BJD.
    period:
        Orbital period in days.
    t_start, t_end:
        Mission window in BJD.
    event_type:
        ``"transit"`` or ``"eclipse"``.
    eccentricity, omega_deg:
        Orbital eccentricity and argument of periastron (degrees).
        Used to compute the transit–eclipse offset.
    sigma_epoch_days:
        1-sigma uncertainty on the reference epoch (days).
    sigma_period_days:
        1-sigma uncertainty on the period (days).

    Returns
    -------
    EphemerisResult
    """
    if period <= 0:
        raise ValueError(f"Period must be positive; got {period}")

    # ------------------------------------------------------------------ #
    # Eclipse offset: t_ecl = t_transit + half_period_correction
    # ------------------------------------------------------------------ #
    if event_type == "eclipse":
        # Approximate correction for eccentricity
        e = eccentricity if eccentricity is not None and not math.isnan(eccentricity) else 0.0
        w = math.radians(omega_deg if omega_deg is not None and not math.isnan(omega_deg) else 0.0)
        # Primary → secondary time offset
        eclipse_offset = (period / 2.0) * (1.0 + (4.0 / math.pi) * e * math.cos(w))
        base_epoch = epoch + eclipse_offset
    else:
        base_epoch = epoch

    # ------------------------------------------------------------------ #
    # Find integer range of n
    # ------------------------------------------------------------------ #
    n_min = math.ceil((t_start - base_epoch) / period)
    n_max = math.floor((t_end - base_epoch) / period)

    if n_min > n_max:
        return EphemerisResult(
            target_id=target_id,
            event_type=event_type,
            indices=np.array([], dtype=np.int64),
            mid_times=np.array([], dtype=np.float64),
            uncertainties=np.array([], dtype=np.float64),
        )

    indices = np.arange(n_min, n_max + 1, dtype=np.int64)
    mid_times = base_epoch + indices * period

    # Keep only events strictly inside the window (guard against float rounding)
    mask = (mid_times >= t_start) & (mid_times <= t_end)
    indices = indices[mask]
    mid_times = mid_times[mask]

    # ------------------------------------------------------------------ #
    # Timing uncertainties in seconds
    # ------------------------------------------------------------------ #
    se = sigma_epoch_days * 86400.0
    sp = sigma_period_days * 86400.0
    # σ(t_n)² = σ_epoch² + (n * σ_period)²
    uncertainties = np.sqrt(se**2 + (np.abs(indices) * sp) ** 2)

    return EphemerisResult(
        target_id=target_id,
        event_type=event_type,
        indices=indices,
        mid_times=mid_times,
        uncertainties=uncertainties,
    )


def eclipse_offset_days(period: float, eccentricity: float = 0.0, omega_deg: float = 0.0) -> float:
    """Return the transit → eclipse time offset in days."""
    e = eccentricity if eccentricity is not None and not math.isnan(eccentricity) else 0.0
    w = math.radians(omega_deg if omega_deg is not None and not math.isnan(omega_deg) else 0.0)
    return (period / 2.0) * (1.0 + (4.0 / math.pi) * e * math.cos(w))
