"""Custom RewardInfo rewards plug into ArielEnv."""

from aRieL import ArielEnv, make_demo_targets
from aRieL.rewards import RewardInfo, t1_bonus
from aRieL.validation import validate_reward


def test_simple_reward_validates_and_is_fast():
    report = validate_reward(fn=t1_bonus, n_timing=500)
    assert report.ok, report.summary_str()


def test_env_accepts_reward_fn():
    def my_reward(info: RewardInfo) -> float:
        if info.missed:
            return -0.1
        if info.tier_completed is not None:
            return float(info.tier_completed)
        return 0.01 * info.progress_after

    env = ArielEnv.from_preset(
        "demo",
        targets=make_demo_targets(6),
        reward_fn=my_reward,
    )
    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(10):
        valid = info["action_mask"].nonzero()[0]
        if len(valid) == 0:
            break
        obs, reward, term, trunc, info = env.step(int(valid[0]))
        assert reward == reward  # not NaN
        total += float(reward)
        if term or trunc:
            break
    env.close()
    assert total == total
