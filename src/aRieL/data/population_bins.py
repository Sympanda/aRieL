"""
Assign each target a population bin and an initial science weight.

Population bins capture the astrophysical diversity Ariel is designed to
characterise.  Each target gets a short string label like:

    "super_earth_warm_m"
    "jupiter_ultra_hot_gk"

built from three dimensions:

  1. Planet radius class   (sub_earth / super_earth / mini_neptune / neptune / saturn / jupiter)
  2. Planet temperature class (cold / warm / hot / very_hot / ultra_hot)
  3. Stellar spectral class   (m / k / gf / af_hot)

Science weights are inversely proportional to bin population so the
agent is nudged toward underrepresented targets.  Weights are re-normalised
to [0, 1] per call; they can be recomputed mid-episode if desired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aRieL.data.schemas import RADIUS_BINS, TEMPERATURE_BINS


# ---------------------------------------------------------------------------
# Radius classification
# ---------------------------------------------------------------------------

def _radius_label(rp_re: float) -> str:
    for label, lo, hi in RADIUS_BINS:
        if lo <= rp_re < hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Temperature classification
# ---------------------------------------------------------------------------

def _temperature_label(teq_k: float) -> str:
    for label, lo, hi in TEMPERATURE_BINS:
        if lo <= teq_k < hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Stellar spectral class (coarse, from spectral type string or Teff)
# ---------------------------------------------------------------------------

def _stellar_label(spectral_type: str | None, teff: float | None) -> str:
    """Coarse classification: m / k / gf / af_hot.

    Uses spectral_type string if available, falls back to Teff.
    """
    if spectral_type is not None and isinstance(spectral_type, str) and spectral_type.strip():
        first = spectral_type.strip()[0].upper()
        if first == "M":
            return "m"
        if first == "K":
            return "k"
        if first in ("G", "F"):
            return "gf"
        if first in ("A", "B", "O"):
            return "af_hot"

    # Fallback: use stellar Teff
    if teff is not None and not np.isnan(teff):
        if teff < 3900:
            return "m"
        if teff < 5200:
            return "k"
        if teff < 7500:
            return "gf"
        return "af_hot"

    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_population_bins(
    targets: pd.DataFrame,
    science_weight_floor: float = 0.3,
) -> pd.DataFrame:
    """Add ``population_bin`` and ``science_weight`` columns to *targets*.

    Parameters
    ----------
    targets:
        DataFrame produced by ``load_catalogue.load_mcs``.
    science_weight_floor:
        Minimum science weight after normalisation.  The most common bin would
        otherwise receive exactly 0; setting a floor (0.25–0.5 recommended)
        ensures common populations remain scientifically meaningful.
        Weights are remapped as: ``w' = floor + (1 − floor) * w_normalised``.

    Returns
    -------
    pd.DataFrame
        Same rows, with ``population_bin`` (string) and ``science_weight``
        (float, floor–1) columns filled in.
    """
    targets = targets.copy()

    r_labels = targets["planet_radius"].map(
        lambda x: _radius_label(x) if pd.notna(x) else "unknown"
    )
    t_labels = targets["planet_temperature"].map(
        lambda x: _temperature_label(x) if pd.notna(x) else "unknown"
    )
    s_labels = [
        _stellar_label(
            row["stellar_type"] if "stellar_type" in targets.columns else None,
            row["stellar_temperature"] if "stellar_temperature" in targets.columns else None,
        )
        for _, row in targets.iterrows()
    ]

    targets["population_bin"] = [
        f"{r}_{t}_{s}" for r, t, s in zip(r_labels, t_labels, s_labels)
    ]

    targets["science_weight"] = _compute_weights(
        targets["population_bin"], floor=science_weight_floor
    )

    return targets


def _compute_weights(bins: pd.Series, floor: float = 0.3) -> pd.Series:
    """Inverse-frequency weights with a floor, normalised to [floor, 1].

    Parameters
    ----------
    bins:
        Series of population bin labels.
    floor:
        Minimum weight for the most common bin.  Prevents any target from
        receiving a science weight of exactly zero.

    Returns
    -------
    pd.Series of float in [floor, 1].
    """
    counts = bins.value_counts()
    raw_weights = bins.map(lambda b: 1.0 / counts.get(b, 1))
    w_min, w_max = raw_weights.min(), raw_weights.max()
    if w_max == w_min:
        return pd.Series(
            np.full(len(bins), fill_value=1.0),
            index=bins.index,
            dtype="float64",
        )
    normalised = (raw_weights - w_min) / (w_max - w_min)
    return (floor + (1.0 - floor) * normalised).astype("float64")


def bin_summary(targets: pd.DataFrame) -> pd.DataFrame:
    """Return a count table of targets per population bin, sorted descending."""
    return (
        targets["population_bin"]
        .value_counts()
        .rename_axis("population_bin")
        .reset_index(name="count")
    )
