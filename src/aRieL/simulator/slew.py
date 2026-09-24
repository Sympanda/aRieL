"""
Slew time model for Ariel.

Ariel can slew at roughly 1 degree/minute (0.0167 deg/s), but the
settled, guide-star-locked slew rate for science is more conservatively
modelled as:

  slew_time_seconds = max(MIN_SLEW_S, angular_separation_deg * SLEW_RATE_S_PER_DEG)

Angular separation is the great-circle distance between two (RA, Dec) pairs.

References
----------
Ariel payload performance documentation (ESA-SCI-TN-0001 and related),
Tinetti et al. 2018.
"""

from __future__ import annotations

import math


# Ariel slew performance (conservative, science-mode slewing)
SLEW_RATE_DEG_PER_MIN: float = 1.0                    # degrees per minute
SLEW_RATE_S_PER_DEG: float = 60.0 / SLEW_RATE_DEG_PER_MIN
MIN_SLEW_S: float = 120.0    # minimum settle/guide-star acquisition (2 min)
MAX_SLEW_S: float = 7200.0   # cap at 2 hours; unrealistically large slews clipped


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular separation in degrees.

    Uses the haversine formula for numerical stability at small angles.

    Parameters
    ----------
    ra1, dec1:
        First pointing in degrees.
    ra2, dec2:
        Second pointing in degrees.
    """
    ra1_r = math.radians(ra1)
    ra2_r = math.radians(ra2)
    dec1_r = math.radians(dec1)
    dec2_r = math.radians(dec2)

    d_ra = (ra2_r - ra1_r) / 2.0
    d_dec = (dec2_r - dec1_r) / 2.0

    a = math.sin(d_dec) ** 2 + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(d_ra) ** 2
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return math.degrees(c)


def slew_time_seconds(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Return slew time in seconds between two sky positions.

    Parameters
    ----------
    ra1, dec1:
        Current pointing (degrees).
    ra2, dec2:
        Target pointing (degrees).
    """
    sep = angular_separation_deg(ra1, dec1, ra2, dec2)
    raw = sep * SLEW_RATE_S_PER_DEG
    return float(min(MAX_SLEW_S, max(MIN_SLEW_S, raw)))


def slew_time_days(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Return slew time in days between two sky positions."""
    return slew_time_seconds(ra1, dec1, ra2, dec2) / 86400.0


def slew_time_days_vec(
    ra1: float,
    dec1: float,
    ra2: "np.ndarray",
    dec2: "np.ndarray",
) -> "np.ndarray":
    """Vectorised version of :func:`slew_time_days`.

    Computes slew times from a single source position to an array of target
    positions using NumPy broadcasting.  ~100× faster than calling
    :func:`slew_time_days` in a Python loop for large catalogues.

    Parameters
    ----------
    ra1, dec1:
        Current pointing (degrees), scalars.
    ra2, dec2:
        Target pointings (degrees), shape (N,).

    Returns
    -------
    numpy.ndarray of shape (N,), values in days.
    """
    import numpy as np

    ra1_r  = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r  = np.radians(ra2)
    dec2_r = np.radians(dec2)

    d_ra  = (ra2_r  - ra1_r)  / 2.0
    d_dec = (dec2_r - dec1_r) / 2.0

    a = (np.sin(d_dec) ** 2
         + math.cos(dec1_r) * np.cos(dec2_r) * np.sin(d_ra) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    sep_deg = np.degrees(c)

    raw_s = sep_deg * SLEW_RATE_S_PER_DEG
    slew_s = np.clip(raw_s, MIN_SLEW_S, MAX_SLEW_S)
    return slew_s / 86400.0


def build_slew_matrix(targets: "pd.DataFrame") -> "np.ndarray":  # type: ignore[name-defined]
    """Pre-compute an (N x N) slew-time matrix in seconds.

    Useful for fast repeated lookups during agent training.

    Parameters
    ----------
    targets:
        Target DataFrame with ``ra`` and ``dec`` columns.

    Returns
    -------
    numpy.ndarray of shape (N, N), dtype float32, values in seconds.
    """
    import numpy as np

    ra = targets["ra"].to_numpy(dtype=float)
    dec = targets["dec"].to_numpy(dtype=float)
    n = len(ra)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            s = slew_time_seconds(ra[i], dec[i], ra[j], dec[j])
            matrix[i, j] = s
            matrix[j, i] = s

    return matrix
