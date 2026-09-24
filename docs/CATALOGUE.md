# Target catalogues

`build_target_table` expects Ariel Mission Candidate Sample (MCS) column names. Rename columns, or pass a DataFrame that already uses the internal names, before constructing `ArielEnv`.

The package runs without the MCS. `make_demo_targets()` builds a synthetic list for examples, tests, and a checkpoint load check. Science runs and `make_paper_env()` need the real catalogue, which you download yourself.

```python
from aRieL import build_target_table

targets = build_target_table("path/to/Ariel_MCS_Known_2025-08-18.csv")
```

With no path, `build_target_table()` uses the search order below.

## Obtaining the Mission Candidate Sample

The published checkpoint was trained on [`Ariel_MCS_Known_2025-08-18.csv`](https://github.com/arielmission-space/Mission_Candidate_Sample/blob/main/target_lists/Ariel_MCS_Known_2025-08-18.csv) (MD5 `714735ab7a3ee8a0ea459faf307973da`), in the Ariel Mission Candidate Sample repository:

https://github.com/arielmission-space/Mission_Candidate_Sample

That repository has no licence file. The MIT licence in this package covers the code only. It does not grant rights to the CSV, so the file is not included here.

Place your copy in one of these locations:

1. `$ARIEL_DATA` (a directory containing the CSV)
2. `./data/raw/`
3. `./data/`

Filenames tried first: `Ariel_MCS_Known_2025-08-18.csv`, `MCS.csv`, then any `*.csv` in those folders.

If you use the catalogue in a publication, the MCS authors ask you to cite:

- Edwards & Tinetti (2022), https://ui.adsabs.harvard.edu/abs/2022AJ....164...15E/abstract
- Mugnai et al. (2020), https://ui.adsabs.harvard.edu/abs/2020ExA....50..303M/abstract

## Columns

Mapped in `aRieL.data.schemas.RAW_COL_MAP`.

| MCS column | Internal name | Role |
| --- | --- | --- |
| Planet Name | target_id | planet id |
| Star Name | host_id | host id |
| Star RA / Star Dec | ra, dec | slew geometry (degrees) |
| Planet Period [days] | period | ephemeris |
| Transit Mid Time | epoch | reference mid-time (BJD) |
| Transit Duration T14 [s] | transit_duration | block length |
| Eclipse Duration E14 [s] | eclipse_duration | optional; falls back to T14 |
| Tier 1/2/3 Observations | tier*_required_obs | cumulative observation counts |
| Max Tier | max_tier | 1, 2, or 3 |
| Preferred Method | preferred_method | Transit / Eclipse / Either |

Useful extra columns: Planet Radius [Re], Planet Mass [Me], Planet Temperature [K], Star Temperature [K], Star Metallicity, Star Distance [pc], Available Transits, Available Eclipses, Transit Mid Time Error Upper [days].

Rows that cannot be scheduled (no period or epoch, or zero available observations) are dropped. Missing optional columns become NaN.

To build a table with internal names instead of MCS headers, start from `make_demo_targets()` and replace the rows. The column set is `TARGET_COLS` in `schemas.py`.

## Catalogue growth

Mid-episode injection expects a second CSV (`TPC.csv` or similar) on the same search path, with the same column layout. This release does not include a TPC file. The published checkpoint was trained with catalogue growth disabled.
