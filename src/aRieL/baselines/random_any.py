"""Pure-random baseline: pick uniformly from *all* actions, ignoring the mask.

Unlike ``RandomValid``, this agent does not filter out completed targets or
other invalid events.  It occasionally wastes steps on masked actions and
receives the corresponding penalties.  This makes it a weaker floor than
``RandomValid`` and therefore a more conservative normalisation baseline.
"""

from __future__ import annotations

from aRieL.baselines.base import BaselineAgent


class RandomAny(BaselineAgent):
    """Uniform random over the entire action space (mask ignored)."""

    def act(self, obs: dict, info: dict) -> int:
        n_actions = len(info["action_mask"])
        return int(self.rng.integers(0, n_actions))
