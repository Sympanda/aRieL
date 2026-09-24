"""Smoke tests for the public package surface."""

from aRieL import ArielEnv, __version__, list_env_presets, make_demo_targets
from aRieL.utils.config import load_preset


def test_version_is_set():
    assert __version__


def test_bundled_env_presets_exist():
    names = {p.stem for p in list_env_presets()}
    assert {"default", "demo", "simple", "full_set"} <= names


def test_from_preset_demo_episode():
    env = ArielEnv.from_preset("demo", targets=make_demo_targets(6))
    obs, info = env.reset(seed=0)
    assert "action_mask" in info
    action = int(info["action_mask"].nonzero()[0][0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert terminated in (True, False)
    env.close()


def test_load_preset_reward_overlay():
    cfg = load_preset("demo", reward="default")
    assert cfg.mission.lifetime_days == 60.0
    assert cfg.reward.t1_final_milestone_bonus == 30.0


def test_export_events_from_dynamic_backend():
    env = ArielEnv.from_preset("demo", targets=make_demo_targets(4))
    table = env.export_events()
    assert len(table) > 0
    assert {"target_id", "window_mid", "event_type"} <= set(table.columns)
    assert table["window_mid"].is_monotonic_increasing
    env.close()
