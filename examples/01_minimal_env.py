"""Minimal Gymnasium loop — no MCS catalogue required."""

from aRieL import ArielEnv, make_demo_targets

env = ArielEnv.from_preset("demo", targets=make_demo_targets(8))
obs, info = env.reset(seed=0)

for step in range(20):
    valid = info["action_mask"].nonzero()[0]
    if len(valid) == 0:
        break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={step:02d}  reward={reward:7.3f}  done={terminated or truncated}")
    if terminated or truncated:
        break

env.close()
