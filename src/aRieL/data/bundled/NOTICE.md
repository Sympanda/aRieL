# Mission Candidate Sample

This directory does not contain the Ariel Mission Candidate Sample.

The published checkpoint was trained on `Ariel_MCS_Known_2025-08-18.csv`
(MD5 `714735ab7a3ee8a0ea459faf307973da`):

https://github.com/arielmission-space/Mission_Candidate_Sample/blob/main/target_lists/Ariel_MCS_Known_2025-08-18.csv

Source repository: https://github.com/arielmission-space/Mission_Candidate_Sample

That repository has no licence file. The MIT licence of aRieL covers the code
only. It does not grant rights to redistribute the CSV, so the file is not
included in this package.

Download the catalogue yourself and either:

- set `ARIEL_DATA` to the directory that contains it, or
- place it at `./data/raw/MCS.csv`

The MCS authors ask publications that use the list to cite Edwards & Tinetti
(2022) and Mugnai et al. (2020). See `docs/CATALOGUE.md`.

Synthetic targets for examples and tests are built by `make_demo_targets()`
and do not use this catalogue.
