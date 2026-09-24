"""
Greedy-value baseline: always pick the valid action with the highest
``base_science_value`` in the observation.

``base_science_value`` is a static score combining population rarity
(science_weight) and a rough SNR proxy.  This baseline ignores timing,
tier progress, and diversity — it just chases the individually best target.
"""

from __future__ import annotations

import numpy as np

from aRieL.baselines.base import BaselineAgent
from aRieL.utils.config import ObservationConfig


class GreedyValue(BaselineAgent):
    """Pick the valid candidate with the highest base_science_value.

    Parameters
    ----------
    obs_cfg:
        The ``ObservationConfig`` used by the env, so we know which
        column of ``obs["events"]`` holds ``base_science_value``.
        Falls back to a heuristic column search if not provided.
    """

    def __init__(self, obs_cfg: ObservationConfig | None = None, seed: int = 0) -> None:
        super().__init__(seed)
        self._obs_cfg = obs_cfg
        self._feature_idx: int | None = None

        if obs_cfg is not None:
            try:
                self._feature_idx = obs_cfg.event_features.index("base_science_value")
            except ValueError:
                self._feature_idx = None

    def act(self, obs: dict, info: dict) -> int:
        valid = self._valid_indices(info)
        if len(valid) == 0:
            return 0

        from aRieL.baselines.base import candidate_matrix, feature_names_for_obs

        events = candidate_matrix(obs)
        names = feature_names_for_obs(obs, self._obs_cfg)

        if "base_science_value" in names:
            scores = events[:, names.index("base_science_value")].copy()
        elif "science_weight" in names:
            scores = events[:, names.index("science_weight")].copy()
        elif self._feature_idx is not None:
            scores = events[:, self._feature_idx].copy()
        else:
            scores = events[:, -1].copy()

        # Mask out invalid actions
        mask = np.full(len(scores), -np.inf)
        mask[valid] = scores[valid]

        return int(np.argmax(mask))
