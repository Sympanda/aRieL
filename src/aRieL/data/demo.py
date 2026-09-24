"""Tiny synthetic catalogue for examples and smoke tests (no MCS download)."""

from __future__ import annotations

import pandas as pd

from aRieL.data.schemas import MISSION_START_BJD


def make_demo_targets(n: int = 8) -> pd.DataFrame:
    """Build a small but valid target table the environment can run on.

    Periods, coordinates, and tier requirements are fictional.  Use this for
    tutorials and CI; pass a real MCS table for science runs.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    t0 = MISSION_START_BJD
    rows = []
    bins = (
        "super_earth_warm_gf",
        "neptune_cold_m",
        "jupiter_hot_gf",
        "super_earth_hot_m",
    )
    for i in range(n):
        rows.append({
            "target_idx": i,
            "target_id": f"DEMO-{i + 1:02d}",
            "host_id": f"HOST-{i + 1:02d}",
            "ra": (30.0 * i) % 360.0,
            "dec": -40.0 + 10.0 * (i % 8),
            "period": 1.5 + 0.7 * i,
            "epoch": t0 + 0.4 * (i + 1),
            "epoch_uncertainty": 0.001,
            "transit_duration": 4000.0 + 400.0 * i,
            "eclipse_duration": 4000.0 + 400.0 * i,
            "planet_radius": 1.5 + 0.8 * (i % 5),
            "planet_mass": 4.0 + 8.0 * (i % 4),
            "planet_temperature": 500.0 + 80.0 * i,
            "stellar_type": ("G2", "K5", "M3", "F8")[i % 4],
            "stellar_temperature": 3500.0 + 400.0 * (i % 5),
            "stellar_metallicity": -0.2 + 0.05 * (i % 6),
            "tier1_required_obs": 1 + (i % 3),
            "tier2_required_obs": 3 + (i % 3),
            "tier3_required_obs": 6 + (i % 3),
            "max_tier": 1 + (i % 3),
            "preferred_method": "Transit",
            "available_transits": 80,
            "available_eclipses": 80,
            "fgs_flag": 1,
            "rp_rs": 0.08 + 0.01 * (i % 4),
            "a_rs": 6.0 + i,
            "eccentricity": 0.0,
            "inclination": 88.0,
            "distance_pc": 20.0 + 5.0 * i,
            "population_bin": bins[i % len(bins)],
            "science_weight": 0.3 + 0.08 * (i % 7),
            "obs_cost_days_t1": 0.12,
            "obs_cost_days_t2": 0.12,
            "obs_cost_days_t3": 0.12,
        })
    return pd.DataFrame(rows)
