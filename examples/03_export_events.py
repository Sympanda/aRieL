"""Turn DynamicBackend into an event table for analysis (not used by the RL loop)."""

from aRieL import ArielEnv, make_demo_targets

env = ArielEnv.from_preset("demo", targets=make_demo_targets(5))
table = env.export_events()

print(f"{len(table)} events over {env.cfg.mission.lifetime_days:.0f} days")
print(table[["target_id", "event_type", "window_mid", "block_duration_days"]].head(10))
env.close()
