"""Minimal full-set policy you can copy and replace.

This is a template, not a competitive agent.  It scores each planet with one
linear layer and a small value head.  Swap the ``score`` / ``value`` modules
for your own network; keep ``forward``, ``evaluate_actions``, and
``predict_values`` so MaskablePPO can train it.

The observation is the full-set dict:

    obs["planets"]  (batch, n_actions, n_features)
    obs["global"]   (batch, n_global)

Feature widths come from ``observation_space``, so the layer sizes follow
changes in ``planet_feature_builder`` after you rebuild the env.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class TinyFullSetPolicy(MaskableActorCriticPolicy):
    """One linear score per planet. Edit ``_build`` to insert your own net."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )
        n_pf = int(observation_space["planets"].shape[-1])
        n_gf = int(observation_space["global"].shape[0])
        self.score = nn.Linear(n_pf + n_gf, 1)
        self.value_head = nn.Linear(n_gf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _logits_and_values(
        self,
        obs: dict,
        action_masks: Optional[th.Tensor],
    ) -> Tuple[th.Tensor, th.Tensor]:
        planets = th.nan_to_num(obs["planets"].float(), nan=0.0)
        global_feat = th.nan_to_num(obs["global"].float(), nan=0.0)
        g = global_feat.unsqueeze(1).expand(-1, planets.shape[1], -1)
        logits = self.score(th.cat([planets, g], dim=-1)).squeeze(-1)
        pad = planets.abs().sum(dim=-1) == 0.0
        logits = logits.masked_fill(pad, float("-inf"))
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))
        values = self.value_head(global_feat).squeeze(-1)
        return logits, values

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        mask_t = (
            th.as_tensor(action_masks, dtype=th.bool).to(self.device)
            if action_masks is not None else None
        )
        logits, values = self._logits_and_values(obs_t, mask_t)
        all_inf = (logits == float("-inf")).all(dim=-1, keepdim=True)
        if all_inf.any():
            logits = logits.masked_fill(all_inf, 0.0)
        dist = th.distributions.Categorical(logits=logits, validate_args=False)
        actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(
        self,
        obs: dict,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        obs_t = {
            k: th.as_tensor(v, dtype=th.float32).to(self.device)
            for k, v in obs.items()
        }
        logits, values = self._logits_and_values(obs_t, action_masks)
        all_inf = (logits == float("-inf")).all(dim=-1, keepdim=True)
        if all_inf.any():
            logits = logits.masked_fill(all_inf, 0.0)
        dist = th.distributions.Categorical(logits=logits, validate_args=False)
        return values, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs: dict) -> th.Tensor:
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        _, values = self._logits_and_values(obs_t, None)
        return values.unsqueeze(-1)

    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions
