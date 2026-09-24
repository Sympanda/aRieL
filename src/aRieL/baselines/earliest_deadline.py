"""
Earliest-deadline baseline: always observe the valid event whose window
closes soonest.

This is a classic scheduling heuristic (EDF — Earliest Deadline First).
It minimises missed events by always prioritising the most time-critical
opportunity.  It knows nothing about science value or tier progress.

The "deadline" here is ``block_end`` — the BJD time after which the
observation block is no longer capturable.
"""

from __future__ import annotations

import numpy as np

from aRieL.baselines.base import BaselineAgent
from aRieL.utils.config import ObservationConfig


class EarliestDeadline(BaselineAgent):
    """Pick the valid candidate with the earliest block deadline.

    Prefers ``days_to_block_end_norm`` (current default feature set).  Falls
    back to legacy ``wait_time_days`` when that column is present (older tests /
    configs).

    Parameters
    ----------
    obs_cfg:
        ObservationConfig from the env (to locate feature columns).
    use_total_cost:
        If True, rank by ``total_time_cost_days`` (slew + wait + duration)
        rather than deadline alone.  This accounts for slew overhead and
        avoids preferring distant urgent targets.
    """

    def __init__(
        self,
        obs_cfg: ObservationConfig | None = None,
        use_total_cost: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self._obs_cfg = obs_cfg
        self.use_total_cost = use_total_cost

        self._deadline_idx: int | None = None
        self._cost_idx: int | None = None

        if obs_cfg is not None:
            feats = obs_cfg.event_features
            # Prefer current deadline feature; keep wait_time_days for legacy obs.
            for name in ("days_to_block_end_norm", "wait_time_days"):
                if name in feats:
                    self._deadline_idx = feats.index(name)
                    break
            self._cost_idx = (
                feats.index("total_time_cost_days")
                if "total_time_cost_days" in feats else None
            )

    def act(self, obs: dict, info: dict) -> int:
        valid = self._valid_indices(info)
        if len(valid) == 0:
            return 0

        events: np.ndarray = obs["events"]   # (K, D)
        K = events.shape[0]

        if self.use_total_cost and self._cost_idx is not None:
            urgency = events[:, self._cost_idx]
        elif self._deadline_idx is not None:
            urgency = events[:, self._deadline_idx]
        else:
            # No timing info in obs; fall back to random valid
            return int(self.rng.choice(valid))

        # Pick minimum urgency (soonest deadline) among valid
        masked = np.full(K, np.inf)
        masked[valid] = urgency[valid]

        return int(np.argmin(masked))
