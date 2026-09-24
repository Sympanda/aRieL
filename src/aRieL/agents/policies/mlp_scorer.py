"""
Flat-MLP actor-critic policy for ArielEnv — sanity-check baseline.

This policy flattens the entire observation dict into a single vector and
passes it through a standard MLP.  It loses the "each row is a target"
structure that the transformer exploits, but is:

  * Fast to implement and train
  * A useful lower-bound benchmark: if this can't beat SmartGreedy, the
    reward signal or observation space needs fixing before adding complexity
  * A direct drop-in replacement for ArielTransformerPolicy — same interface,
    same training loop

Usage
-----
    from sb3_contrib import MaskablePPO
    from aRieL.agents.policies.mlp_scorer import ArielMlpPolicy

    model = MaskablePPO(ArielMlpPolicy, env, verbose=1)
    model.learn(total_timesteps=500_000)

    # Then swap to the transformer with the same training script:
    # model = MaskablePPO(ArielTransformerPolicy, env, ...)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class ArielMlpNet(nn.Module):
    """Flat MLP that scores all N actions from a concatenated observation."""

    def __init__(
        self,
        obs_flat_dim: int,
        n_actions: int,
        hidden_sizes: List[int] = (256, 256),
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = obs_flat_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h

        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head  = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=th.nn.init.calculate_gain("tanh"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(
        self, events: th.Tensor, global_feat: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        B = events.shape[0]
        flat = th.cat([events.reshape(B, -1), global_feat], dim=-1)
        h = self.shared(flat)
        return self.policy_head(h), self.value_head(h)


class ArielMlpPolicy(MaskableActorCriticPolicy):
    """
    Flat MLP actor-critic policy.  Same interface as ArielTransformerPolicy.

    Parameters (via ``policy_kwargs``)
    ------------------------------------
    hidden_sizes : list[int]   Hidden layer widths (default [256, 256])
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        hidden_sizes: List[int] = (256, 256),
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch=[], **kwargs
        )

        n_ef = observation_space["events"].shape[0] * observation_space["events"].shape[1]
        n_gf = observation_space["global"].shape[0]
        n_actions = int(action_space.n)

        self.mlp_net = ArielMlpNet(
            obs_flat_dim=n_ef + n_gf,
            n_actions=n_actions,
            hidden_sizes=list(hidden_sizes),
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _obs_to_tensors(self, obs: dict) -> Tuple[th.Tensor, th.Tensor]:
        def _to(x):
            if isinstance(x, th.Tensor):
                return x.to(device=self.device, dtype=th.float32)
            return th.as_tensor(x, device=self.device, dtype=th.float32)
        return _to(obs["events"]), _to(obs["global"])

    def _apply_logit_mask(self, logits: th.Tensor, action_masks) -> th.Tensor:
        if action_masks is None:
            return logits
        if isinstance(action_masks, np.ndarray):
            valid = th.tensor(action_masks, dtype=th.bool, device=self.device)
        else:
            valid = action_masks.to(device=self.device, dtype=th.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand_as(logits)
        return logits.masked_fill(~valid, float("-inf"))

    def _build_dist(self, logits: th.Tensor) -> CategoricalDistribution:
        dist = CategoricalDistribution(action_dim=logits.shape[-1])
        dist.proba_distribution(action_logits=logits)
        return dist

    def forward(
        self, obs: dict, deterministic: bool = False, action_masks=None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        events, global_f = self._obs_to_tensors(obs)
        logits, values   = self.mlp_net(events, global_f)
        logits = self._apply_logit_mask(logits, action_masks)
        dist     = self._build_dist(logits)
        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions.reshape(-1), values, log_prob

    def evaluate_actions(
        self, obs: dict, actions: th.Tensor, action_masks=None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        events, global_f = self._obs_to_tensors(obs)
        logits, values   = self.mlp_net(events, global_f)
        logits = self._apply_logit_mask(logits, action_masks)
        dist     = self._build_dist(logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: dict) -> th.Tensor:
        events, global_f = self._obs_to_tensors(obs)
        _, values = self.mlp_net(events, global_f)
        return values

    def _predict(
        self, observation: dict, deterministic: bool = False, action_masks=None
    ) -> th.Tensor:
        actions, _, _ = self.forward(
            observation, deterministic=deterministic, action_masks=action_masks
        )
        return actions
