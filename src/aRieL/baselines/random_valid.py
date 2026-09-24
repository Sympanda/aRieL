"""Random-valid baseline: pick uniformly from valid actions."""

from __future__ import annotations

import numpy as np

from aRieL.baselines.base import BaselineAgent


class RandomValid(BaselineAgent):
    """Uniform random over the valid action set.

    This is the weakest meaningful baseline — it tells you the floor
    of what pure random scheduling achieves under the mission constraints.
    """

    def act(self, obs: dict, info: dict) -> int:
        valid = self._valid_indices(info)
        if len(valid) == 0:
            return 0
        return int(self.rng.choice(valid))
