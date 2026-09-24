"""Strict reward-config loading: no silent defaults; list presets on error."""

from __future__ import annotations

from dataclasses import asdict, fields as dc_fields
from pathlib import Path

import pytest
import yaml

from aRieL.utils.config import (
    RewardConfig,
    list_reward_presets,
    load_reward_config,
)
from aRieL.utils.paths import bundled_config_dir


REWARD_DIR = bundled_config_dir() / "reward"


def _complete_reward_dict(**overrides) -> dict:
    """Build a full reward mapping from dataclass defaults, then override."""
    base = asdict(RewardConfig())
    for f in dc_fields(RewardConfig):
        if isinstance(base[f.name], tuple):
            base[f.name] = list(base[f.name])
    base.update(overrides)
    return base


def test_preset_yamls_load_strictly():
    presets = list_reward_presets(REWARD_DIR)
    assert presets, "configs/reward/ should contain at least one preset"
    for path in presets:
        cfg = load_reward_config(path)
        assert isinstance(cfg, RewardConfig)
        assert cfg.tier1_completion is not None


def test_list_reward_presets_matches_directory():
    listed = {p.name for p in list_reward_presets(REWARD_DIR)}
    on_disk = {p.name for p in REWARD_DIR.glob("*.yaml")}
    assert listed == on_disk


def test_missing_file_lists_presets():
    with pytest.raises(FileNotFoundError, match="Available reward presets"):
        load_reward_config(REWARD_DIR / "does_not_exist.yaml")


def test_empty_file_raises(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_reward_config(p)


def test_lenient_fills_missing_keys_from_defaults(tmp_path: Path):
    data = _complete_reward_dict()
    del data["science_weight_on_progress"]
    del data["t2_milestone_pool"]
    p = tmp_path / "old_snapshot.yaml"
    p.write_text(yaml.dump({"reward": data}))
    cfg = load_reward_config(p, strict=False)
    assert cfg.science_weight_on_progress is RewardConfig().science_weight_on_progress
    assert cfg.t2_milestone_pool == RewardConfig().t2_milestone_pool
    # Fields that *were* in the file are kept.
    assert cfg.tier1_completion == data["tier1_completion"]


def test_lenient_ignores_unknown_keys(tmp_path: Path):
    data = _complete_reward_dict()
    data["retired_field"] = 1.0
    p = tmp_path / "legacy_key.yaml"
    p.write_text(yaml.dump({"reward": data}))
    cfg = load_reward_config(p, strict=False)
    assert cfg.tier1_completion == data["tier1_completion"]


def test_missing_key_raises_no_silent_default(tmp_path: Path):
    data = _complete_reward_dict()
    del data["t1_milestone_step_above_one"]
    p = tmp_path / "incomplete.yaml"
    p.write_text(yaml.dump({"reward": data}))
    with pytest.raises(ValueError, match="Missing required keys"):
        load_reward_config(p)


def test_unknown_key_raises(tmp_path: Path):
    data = _complete_reward_dict()
    data["typo_tier1_completioon"] = 1.0
    p = tmp_path / "typo.yaml"
    p.write_text(yaml.dump({"reward": data}))
    with pytest.raises(ValueError, match="Unknown keys"):
        load_reward_config(p)


def test_science_weight_tiers_and_t2_milestones_roundtrip(tmp_path: Path):
    data = _complete_reward_dict(
        science_weight_tiers=[1],
        science_weight_on_progress=False,
        t2_milestone_fractions=[0.25, 0.5, 1.0],
        t2_milestone_bonus=2.0,
        t2_final_milestone_bonus=20.0,
        t2_milestone_pool="max_tier_eq_2",
    )
    p = tmp_path / "t2ish.yaml"
    p.write_text(yaml.dump({"reward": data}))
    cfg = load_reward_config(p)
    assert cfg.science_weight_tiers == (1,)
    assert cfg.science_weight_on_progress is False
    assert cfg.t2_milestone_fractions == (0.25, 0.5, 1.0)
    assert cfg.t2_milestone_pool == "max_tier_eq_2"


def test_invalid_t2_pool_raises(tmp_path: Path):
    data = _complete_reward_dict(t2_milestone_pool="all_planets")
    p = tmp_path / "bad_pool.yaml"
    p.write_text(yaml.dump({"reward": data}))
    with pytest.raises(ValueError, match="t2_milestone_pool"):
        load_reward_config(p)


def test_flat_mapping_without_reward_key(tmp_path: Path):
    data = _complete_reward_dict(tier1_completion=9.0)
    p = tmp_path / "flat.yaml"
    p.write_text(yaml.dump(data))
    cfg = load_reward_config(p)
    assert cfg.tier1_completion == 9.0
