"""Run a heuristic scheduler on the demo catalogue and print episode stats."""

from aRieL import ArielEnv, make_demo_targets
from aRieL.baselines import SmartGreedy
from aRieL.evaluation import run_episode

env = ArielEnv.from_preset("demo", reward="default", targets=make_demo_targets(8))
stats = run_episode(env, SmartGreedy(), seed=0)

print(stats.summary_str())
env.close()
