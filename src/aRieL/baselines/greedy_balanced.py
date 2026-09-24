"""
Greedy-balanced baseline: score each valid action by combining population
rarity (science_weight) with proximity to the next tier completion
(progress_in_tier).

Score = science_weight × (1 + α × progress_in_tier)

where α controls how strongly near-completion targets are prioritised.
When α=0 this reduces to greedy-by-science-weight.
When α is large, the baseline becomes a "finish what you started" policy.

This baseline is more scientifically motivated than greedy-value because
it simultaneously:
  - prioritises rare / underrepresented targets (science_weight)
  - avoids wasting partial observations on targets that are close to
    a tier boundary (progress_in_tier)
"""

from __future__ import annotations

import numpy as np

from aRieL.baselines.base import BaselineAgent
from aRieL.utils.config import ObservationConfig


class GreedyBalanced(BaselineAgent):
    """Pick the valid candidate maximising science_weight × (1 + α × progress_in_tier).

    Parameters
    ----------
    obs_cfg:
        ObservationConfig from the env (to locate feature columns).
    alpha:
        Weight on near-completion bonus.  Default 1.0.
    """

    def __init__(
        self,
        obs_cfg: ObservationConfig | None = None,
        alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.alpha = alpha
        self._obs_cfg = obs_cfg

        self._sw_idx: int | None = None
        self._prog_idx: int | None = None

        if obs_cfg is not None:
            feats = obs_cfg.event_features
            self._sw_idx   = feats.index("science_weight")       if "science_weight"    in feats else None
            self._prog_idx = feats.index("progress_in_tier")     if "progress_in_tier"  in feats else None

    def act(self, obs: dict, info: dict) -> int:
        valid = self._valid_indices(info)
        if len(valid) == 0:
            return 0

        from aRieL.baselines.base import (
            candidate_matrix, feature_column, feature_names_for_obs,
        )

        events = candidate_matrix(obs)
        names = feature_names_for_obs(obs, self._obs_cfg)
        K = events.shape[0]

        sw   = feature_column(events, names, "science_weight", fill="ones")
        prog = feature_column(events, names, "progress_in_tier", fill="zeros")

        scores = sw * (1.0 + self.alpha * prog)

        masked = np.full(K, -np.inf)
        masked[valid] = scores[valid]

        return int(np.argmax(masked))
