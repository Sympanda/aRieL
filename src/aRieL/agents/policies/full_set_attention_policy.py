"""
FullSetSelfAttentionPolicy — full self-attention ablation for the full-set space.

Architecture
------------
Identical to ``FullSetISABPolicy`` EXCEPT that it uses standard full self-attention
(O(N²)) instead of ISAB's induced attention (O(N·m)).

This is an ablation to separate:
    benefit of seeing all N planets  (vs. top-K filtering)
FROM:
    effect of induced attention       (ISAB vs. full self-attention)

    planets (N_max, n_pf)  ──► Linear ──► planet tokens (N_max, d)
                                                │
                                    TransformerEncoder  (n_layers × self-attn + FFN)
                                    Pre-LN, no positional encoding
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        │                                                 │
                   policy_head                                        PMA (k=1)
                  (per-token linear)                           ──► global summary
                        │                                         + global features
                   logits (N_max,)                                      │
                   + action mask                                    value_head
                        │                                               │
                  π(a|s)                                           V(s) scalar

Compare against:
    Top-K full attention         (ArielTransformerPolicy,      input = top-K events)
    Full-set full attention      (FullSetSelfAttentionPolicy,  input = all N planets)
    Full-set induced attention   (FullSetISABPolicy,           input = all N planets)

This comparison isolates:
    Full-set vs. Top-K          → does seeing all planets help?
    ISAB vs. full attn          → does O(N·m) induced attention match O(N²)?

Usage
-----
    from sb3_contrib import MaskablePPO
    from aRieL.agents.policies.full_set_attention_policy import FullSetSelfAttentionPolicy

    model = MaskablePPO(
        FullSetSelfAttentionPolicy,
        env,   # must use action_type="full_set"
        policy_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 2},
    )
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from aRieL.agents.policies.isab_modules import PMA


# ---------------------------------------------------------------------------
# Core network module
# ---------------------------------------------------------------------------

class FullSetSelfAttentionNet(nn.Module):
    """Full self-attention set encoder with PMA critic.

    Same interface as FullSetISABNet but uses nn.TransformerEncoder (O(N²))
    instead of ISAB (O(N·m)).  For N_max ≤ ~800 the memory difference is
    negligible; for N_max ≈ 2000 ISAB is preferred.
    """

    def __init__(
        self,
        n_planet_features: int,
        n_global_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.planet_proj = nn.Linear(n_planet_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor: per-token logit conditioned on both token and global mission state
        self.global_proj_actor = nn.Linear(n_global_features, d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # PMA critic (consistent with ISAB policy for fair comparison)
        self.pma = PMA(d_model, n_heads, k=1)
        self.global_proj_critic = nn.Linear(n_global_features, d_model)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_mlp[-1].weight, gain=1.0)

    def forward(
        self,
        planets: th.Tensor,
        global_feat: th.Tensor,
        padding_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        # Sanitise inputs — MPS can silently produce NaN from degenerate floats.
        planets = th.nan_to_num(planets.float(), nan=0.0, posinf=0.0, neginf=0.0)

        tokens = self.planet_proj(planets)                              # (B, N, d)

        # Convert boolean padding mask to a float additive mask so that
        # nn.TransformerEncoder never receives a boolean key_padding_mask.
        # Boolean masks trigger a broken MPS kernel that produces all-NaN output.
        # A float mask with -1e9 for padded positions is numerically equivalent
        # and uses a different (working) code path.
        if padding_mask is not None and padding_mask.any():
            float_pad_mask = padding_mask.float().masked_fill(padding_mask, -1e9)
            tokens = self.encoder(tokens, src_key_padding_mask=float_pad_mask)
        else:
            tokens = self.encoder(tokens)

        tokens = th.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        # Actor: per-token logit conditioned on global mission state
        N = tokens.shape[1]
        g_actor  = self.global_proj_actor(global_feat)                  # (B, d)
        g_expand = g_actor.unsqueeze(1).expand(-1, N, -1)               # (B, N, d)
        logits   = self.actor_head(th.cat([tokens, g_expand], dim=-1)).squeeze(-1)  # (B, N)

        # Critic: PMA + global
        summary  = self.pma(tokens, key_padding_mask=padding_mask).squeeze(1)  # (B, d)
        g_critic = self.global_proj_critic(global_feat)                  # (B, d)
        values   = self.value_mlp(th.cat([summary, g_critic], dim=-1)).squeeze(-1)

        logits = th.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        values = th.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return logits, values


# ---------------------------------------------------------------------------
# SB3 policy wrapper (mirrors FullSetISABPolicy)
# ---------------------------------------------------------------------------

class FullSetSelfAttentionPolicy(MaskableActorCriticPolicy):
    """Full self-attention ablation on the full-set planet observation space.

    Use this to compare against FullSetISABPolicy:
        Full-set + full attention  (this class)
        Full-set + ISAB            (FullSetISABPolicy)

    Keep ArielTransformerPolicy unchanged as the Top-K baseline.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        n_pf = observation_space["planets"].shape[-1]
        n_gf = observation_space["global"].shape[0]

        self.attn_net = FullSetSelfAttentionNet(
            n_planet_features=n_pf,
            n_global_features=n_gf,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _obs_to_tensors(self, obs):
        planets     = obs["planets"]
        global_feat = obs["global"]
        pad_mask    = (planets.abs().sum(dim=-1) == 0.0)
        return planets, global_feat, pad_mask

    def _predict_logits_and_values(self, obs, action_masks):
        planets, global_feat, pad_mask = self._obs_to_tensors(obs)
        logits, values = self.attn_net(planets, global_feat, padding_mask=pad_mask)
        if pad_mask.any():
            logits = logits.masked_fill(pad_mask, float("-inf"))
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))
        return logits, values

    def forward(self, obs, deterministic=False, action_masks=None):
        obs_t  = {k: th.as_tensor(v, dtype=th.float32).to(self.device)
                  for k, v in obs.items()}
        mask_t = (
            th.as_tensor(action_masks, dtype=th.bool).to(self.device)
            if action_masks is not None else None
        )
        logits, values = self._predict_logits_and_values(obs_t, mask_t)
        dist = th.distributions.Categorical(logits=logits, validate_args=False)
        actions   = dist.sample() if not deterministic else logits.argmax(dim=-1)
        log_probs = dist.log_prob(actions)
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions, action_masks=None):
        # Ensure obs tensors are float32 on the correct device.
        obs = {k: th.as_tensor(v, dtype=th.float32).to(self.device)
               for k, v in obs.items()}
        logits, values = self._predict_logits_and_values(obs, action_masks)

        # Guard rows where every logit is -inf (fully masked terminal steps).
        # Replacing with zeros gives a uniform distribution so log_prob / entropy
        # remain finite and gradients don't propagate through these steps.
        all_masked = (logits == float("-inf")).all(dim=-1, keepdim=True)
        logits = th.where(all_masked, th.zeros_like(logits), logits)

        dist = th.distributions.Categorical(logits=logits, validate_args=False)
        return values, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        planets, global_feat, pad_mask = self._obs_to_tensors(obs_t)
        _, values = self.attn_net(planets, global_feat, padding_mask=pad_mask)
        return values.unsqueeze(-1)

    def _predict(self, observation, deterministic=False, action_masks=None):
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions
