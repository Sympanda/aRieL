"""
Abstract base class for all baseline schedulers.

The interface mirrors what an RL agent receives:
  act(obs, info) → int action index

``obs``  is the dict from ArielEnv: either ``{"events": (K×D)}`` (top-K)
or ``{"planets": (N×F)}`` (full-set), plus ``"global"``.
``info`` is the dict from ArielEnv: {"action_mask": bool (K,), "step_result": …, …}

Baselines may also accept an optional ``state`` argument (the raw MissionState)
for baselines that need richer information than the obs arrays provide.
However, any baseline that relies on ``state`` is not directly comparable to a
policy-gradient agent that only sees obs/info.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def candidate_matrix(obs: dict) -> np.ndarray:
    """Return the per-action feature matrix (top-K events or full-set planets)."""
    if "events" in obs:
        return np.asarray(obs["events"])
    if "planets" in obs:
        return np.asarray(obs["planets"])
    raise KeyError("obs must contain 'events' or 'planets'")


def feature_names_for_obs(obs: dict, obs_cfg=None) -> list[str]:
    """Column names matching ``candidate_matrix(obs)``."""
    if "planets" in obs:
        from aRieL.envs.planet_feature_builder import PLANET_FEATURE_NAMES
        return list(PLANET_FEATURE_NAMES)
    if obs_cfg is not None:
        return list(obs_cfg.event_features)
    return [f"f{i}" for i in range(int(obs["events"].shape[1]))]


def feature_column(
    matrix: np.ndarray,
    names: list[str],
    *keys: str,
    fill: str | None = None,
) -> np.ndarray:
    """First matching column, or zeros/ones if ``fill`` is set."""
    for key in keys:
        try:
            return matrix[:, names.index(key)]
        except ValueError:
            continue
    n = matrix.shape[0]
    if fill == "ones":
        return np.ones(n, dtype=matrix.dtype)
    if fill == "zeros":
        return np.zeros(n, dtype=matrix.dtype)
    raise KeyError(f"None of {keys!r} found in feature names {names}")


class BaselineAgent(ABC):
    """Abstract baseline agent."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def act(self, obs: dict, info: dict) -> int:
        """Choose an action given the current observation and info dict.

        Parameters
        ----------
        obs:
            ``{"events": float32 (K, D), "global": float32 (G,)}``
        info:
            Env info dict containing at minimum ``"action_mask": bool (K,)``.

        Returns
        -------
        int
            Action index.  Must be within [0, K) and should be valid
            (i.e. ``info["action_mask"][action] == True``).
        """

    def reset(self) -> None:
        """Called at the start of each episode.  Override if stateful."""

    def _valid_indices(self, info: dict) -> np.ndarray:
        """Return indices of valid actions from the action mask."""
        mask: np.ndarray = info["action_mask"]
        return np.where(mask)[0]

    def _fallback(self, info: dict) -> int:
        """Return the first valid action, or 0 if none exist."""
        valid = self._valid_indices(info)
        return int(valid[0]) if len(valid) > 0 else 0
