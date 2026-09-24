"""Filesystem helpers that do not assume a research-repo checkout."""

from __future__ import annotations

import os
from pathlib import Path


def package_dir() -> Path:
    """Return the installed ``aRieL`` package directory."""
    return Path(__file__).resolve().parents[1]


def bundled_config_dir() -> Path:
    return package_dir() / "configs"


def bundled_data_dir() -> Path:
    """Data directory inside the package. The MCS CSV is not shipped here."""
    return package_dir() / "data" / "bundled"


def bundled_model_dir() -> Path:
    return package_dir() / "models"


def data_search_dirs() -> list[Path]:
    """Places to look for MCS / TPC CSVs.

    Order:
    1. ``$ARIEL_DATA`` if set (directory containing the CSV files)
    2. ``./data/raw`` relative to the current working directory
    3. ``./data`` relative to the current working directory
    4. ``data/bundled`` inside this package (no MCS file is shipped)
    """
    dirs: list[Path] = []
    env = os.environ.get("ARIEL_DATA")
    if env:
        dirs.append(Path(env).expanduser().resolve())
    cwd = Path.cwd()
    dirs.append(cwd / "data" / "raw")
    dirs.append(cwd / "data")
    dirs.append(bundled_data_dir())
    return dirs


def find_data_csv(
    names: list[str],
    *,
    extra_globs: bool = True,
) -> Path | None:
    """Return the first matching CSV in :func:`data_search_dirs`, or ``None``."""
    for directory in data_search_dirs():
        if not directory.is_dir():
            continue
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        if extra_globs:
            csvs = sorted(p for p in directory.glob("*.csv") if p.is_file())
            if csvs:
                return csvs[0]
    return None
