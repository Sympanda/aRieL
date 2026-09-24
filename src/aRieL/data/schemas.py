"""
Column name constants and dtype definitions for all core tables.

Three main tables flow through the system:

  TARGET_COLS   — one row per target, static physical properties
  EVENT_COLS    — one row per transit/eclipse window, derived from targets
  PROGRESS_COLS — one row per target, mutable observation progress state

All times are in BJD (Barycentric Julian Date) days unless noted.
Durations are in seconds unless noted.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw CSV → target column mapping
# ---------------------------------------------------------------------------

RAW_COL_MAP: dict[str, str] = {
    "Planet Name":                  "target_id",
    "Star Name":                    "host_id",
    "Star RA":                      "ra",
    "Star Dec":                     "dec",
    "Planet Period [days]":         "period",
    "Transit Mid Time":             "epoch",           # BJD reference transit
    "Transit Duration T14 [s]":     "transit_duration", # seconds
    "Eclipse Duration E14 [s]":     "eclipse_duration", # seconds
    "Planet Radius [Re]":           "planet_radius",   # Earth radii
    "Planet Mass [Me]":             "planet_mass",     # Earth masses
    "Planet Temperature [K]":       "planet_temperature",
    "Star Spectral Type":           "stellar_type",
    "Star Temperature [K]":         "stellar_temperature",
    "Star Metallicity":             "stellar_metallicity",
    "Tier 1 Observations":          "tier1_required_obs",  # cumulative count
    "Tier 2 Observations":          "tier2_required_obs",
    "Tier 3 Observations":          "tier3_required_obs",
    "Preferred Method":             "preferred_method",  # "Transit"/"Eclipse"/"Either"
    "Available Transits":           "available_transits",
    "Available Eclipses":           "available_eclipses",
    "Max Tier":                     "max_tier",          # 1, 2, or 3
    "FGS_Flag":                     "fgs_flag",
    "Star Distance [pc]":           "distance_pc",
    "Planet Mass [Mjup]":           "planet_mass_mjup",  # for reference
    "Rp/Rs":                        "rp_rs",
    "a/Rs":                         "a_rs",
    "Eccentricity":                 "eccentricity",
    "Inclination":                  "inclination",
    "Transit Mid Time Error Upper [days]": "epoch_uncertainty",
}

# ---------------------------------------------------------------------------
# Target table
# ---------------------------------------------------------------------------

TARGET_COLS = [
    # Identifiers
    "target_id",          # str  — planet name (e.g. "55Cnce")
    "host_id",            # str  — star name
    # Coordinates
    "ra",                 # float — degrees
    "dec",                # float — degrees
    # Orbital / timing
    "period",             # float — days
    "epoch",              # float — BJD reference transit mid-time
    "epoch_uncertainty",  # float — days (1-sigma)
    "transit_duration",   # float — seconds (T14)
    "eclipse_duration",   # float — seconds (E14)
    # Planet physical
    "planet_radius",      # float — Earth radii
    "planet_mass",        # float — Earth masses
    "planet_temperature", # float — K (equilibrium)
    # Stellar
    "stellar_type",       # str   — spectral type (may be empty)
    "stellar_temperature",# float — K
    "stellar_metallicity",# float — [Fe/H]
    # Ariel observation requirements
    "tier1_required_obs", # int   — cumulative transits/eclipses for Tier 1
    "tier2_required_obs", # int   — cumulative for Tier 2
    "tier3_required_obs", # int   — cumulative for Tier 3
    "max_tier",           # int   — highest reachable tier (1/2/3)
    "preferred_method",   # str   — "Transit"/"Eclipse"/"Either"
    "available_transits", # int   — total transits over mission lifetime
    "available_eclipses", # int   — total eclipses over mission lifetime
    "fgs_flag",           # int   — fine guidance sensor flag
    # Geometry
    "rp_rs",              # float — planet/star radius ratio
    "a_rs",               # float — semi-major axis / star radius
    "eccentricity",       # float
    "inclination",        # float — degrees
    "distance_pc",        # float — parsecs
    # Derived (added by preprocessing)
    "population_bin",     # str   — e.g. "super_earth_warm_gk"
    "science_weight",     # float — 0–1, emphasises underrepresented bins
    "obs_cost_days_t1",   # float — single-obs wall-clock cost for tier 1 obs
    "obs_cost_days_t2",   # float — single-obs cost for tier 2 obs
    "obs_cost_days_t3",   # float — single-obs cost for tier 3 obs
]

TARGET_DTYPES: dict[str, str] = {
    "target_id":           "string",
    "host_id":             "string",
    "ra":                  "float64",
    "dec":                 "float64",
    "period":              "float64",
    "epoch":               "float64",
    "epoch_uncertainty":   "float64",
    "transit_duration":    "float64",
    "eclipse_duration":    "float64",
    "planet_radius":       "float64",
    "planet_mass":         "float64",
    "planet_temperature":  "float64",
    "stellar_type":        "string",
    "stellar_temperature": "float64",
    "stellar_metallicity": "float64",
    "tier1_required_obs":  "Int64",
    "tier2_required_obs":  "Int64",
    "tier3_required_obs":  "Int64",
    "max_tier":            "Int64",
    "preferred_method":    "string",
    "available_transits":  "Int64",
    "available_eclipses":  "Int64",
    "fgs_flag":            "Int64",
    "rp_rs":               "float64",
    "a_rs":                "float64",
    "eccentricity":        "float64",
    "inclination":         "float64",
    "distance_pc":         "float64",
    "population_bin":      "string",
    "science_weight":      "float64",
    "obs_cost_days_t1":    "float64",
    "obs_cost_days_t2":    "float64",
    "obs_cost_days_t3":    "float64",
}

# ---------------------------------------------------------------------------
# Event table
# ---------------------------------------------------------------------------

EVENT_COLS = [
    "event_id",              # int   — unique sequential id
    "target_id",             # str   — foreign key → target table
    "event_type",            # str   — "transit" / "eclipse"
    "window_start",          # float — BJD, start of transit/eclipse (mid − T14/2)
    "window_mid",            # float — BJD, predicted mid-time
    "window_end",            # float — BJD, end of transit/eclipse (mid + T14/2)
    "duration",              # float — seconds (raw T14 or E14)
    "duration_days",         # float — days (raw T14 or E14, for SNR purposes)
    "block_duration_days",   # float — full observation block = COST_FACTOR * duration_days
    "tier_goal",             # int   — tier this observation is counted toward
    "base_science_value",    # float — 0–1, intrinsic value before context
    "visibility_valid",      # bool  — within satellite pointing constraints
    "ephemeris_uncertainty", # float — seconds, 1-sigma timing uncertainty at event
    "event_index",           # int   — transit/eclipse number (0-based from epoch)
]

EVENT_DTYPES: dict[str, str] = {
    "event_id":              "int64",
    "target_id":             "string",
    "event_type":            "string",
    "window_start":          "float64",
    "window_mid":            "float64",
    "window_end":            "float64",
    "duration":              "float64",
    "duration_days":         "float64",
    "block_duration_days":   "float64",
    "tier_goal":             "Int64",
    "base_science_value":    "float64",
    "visibility_valid":      "bool",
    "ephemeris_uncertainty": "float64",
    "event_index":           "int64",
}

# ---------------------------------------------------------------------------
# Target progress table  (mutable during episode)
# ---------------------------------------------------------------------------

PROGRESS_COLS = [
    "target_id",             # str   — primary key
    "obs_completed",         # float — equivalent observations executed (fractional with partial obs)
    "current_tier",          # int   — highest completed tier (0 = none)
    "tier1_done",            # bool
    "tier2_done",            # bool
    "tier3_done",            # bool
    "progress_in_tier",      # float — 0–1, fraction toward next tier
    "obs_remaining_next_tier",# float — equivalent obs to reach next tier (0 if maxed; fractional)
    "max_tier",              # int   — copy from target table for convenience
]

PROGRESS_DTYPES: dict[str, str] = {
    "target_id":              "string",
    "obs_completed":          "float64",
    "current_tier":           "int64",
    "tier1_done":             "bool",
    "tier2_done":             "bool",
    "tier3_done":             "bool",
    "progress_in_tier":       "float64",
    "obs_remaining_next_tier":"float64",
    "max_tier":               "int64",
}

# ---------------------------------------------------------------------------
# Mission constants
# ---------------------------------------------------------------------------

MISSION_LIFETIME_DAYS: float = 3.5 * 365.25        # ~1278 days
MISSION_START_BJD: float = 2462867.5                # ~2029-01-01
MISSION_END_BJD: float = MISSION_START_BJD + MISSION_LIFETIME_DAYS

OBS_OVERHEAD_FACTOR: float = 2.5   # multiply T14 to get total time cost (slew + settle etc.)
OBS_OVERHEAD_DAYS_BASE: float = 0.0  # fixed overhead per observation in days (TBD)

# Ariel photon-noise SNR scale: cost in days = factor * T14[s] / 86400
# Tier 1 uses this directly; higher tiers scale by the number of required stacks.
COST_FACTOR: float = 2.5

# Tier identifiers
TIER_NONE = 0
TIER_1 = 1
TIER_2 = 2
TIER_3 = 3

# Preferred method labels
METHOD_TRANSIT = "Transit"
METHOD_ECLIPSE = "Eclipse"
METHOD_EITHER  = "Either"

# Population bin dimension labels (used in population_bins.py)
RADIUS_BINS = [
    ("sub_earth",  0.0,  1.5),
    ("super_earth", 1.5, 2.5),
    ("mini_neptune", 2.5, 4.0),
    ("neptune",    4.0,  6.0),
    ("saturn",     6.0, 10.0),
    ("jupiter",   10.0, float("inf")),
]

TEMPERATURE_BINS = [
    ("cold",      0,    400),
    ("warm",    400,    900),
    ("hot",     900,   1400),
    ("very_hot", 1400, 2000),
    ("ultra_hot", 2000, float("inf")),
]
