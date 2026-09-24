"""
FullSetISABPolicy — Set Transformer actor-critic for the full-set action space.

Architecture
------------
The observation is a Dict with two arrays:

    "planets"  (N_max, n_pf)  — one row per target (padded to N_max)
    "global"   (n_gf,)        — mission-level state vector

Flow:

    planets (N_max, n_pf)  ──► Linear ──► planet tokens  (N_max, d_model)
                                                │
                                          ISAB × n_isab_layers    O(N·m)
                                                │
                              ┌─────────────────┴──────────────────┐
                              │                                     │
                         policy_head                           PMA (k=1)
                        (per-token linear)              ──► global summary (d_model)
                              │                              + global features
                         logits (N_max,)                        │
                         + action mask                       value_head
                              │                                  │
                        π(a|s)                              V(s) scalar

Distinct from ``ArielTransformerPolicy`` (Top-K full self-attention):
- Input tokens are planet features, not event features.
- ISAB replaces full self-attention → O(N·m) complexity instead of O(N²).
- Critic uses PMA (permutation-invariant set pooling) instead of [CLS] token.
- Designed for N_max ≈ 2000 planets without quadratic memory blowup.

Usage
-----
    from sb3_contrib import MaskablePPO
    from aRieL.agents.policies.full_set_isab_policy import FullSetISABPolicy

    model = MaskablePPO(
        FullSetISABPolicy,
        env,   # must use action_type="full_set"
        policy_kwargs={
            "d_model":     128,
            "n_heads":     4,
            "n_isab_layers": 2,
            "n_inducing":  32,
        },
    )

Note: keep ``ArielTransformerPolicy`` for the Top-K baseline.  Do not merge
or replace — they serve different action spaces and answer different questions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from aRieL.agents.policies.isab_modules import ISAB, PMA


# ---------------------------------------------------------------------------
# Core network module
# ---------------------------------------------------------------------------

class FullSetISABNet(nn.Module):
    """
    ISAB-based actor-critic network for the full-set action space.

    Parameters
    ----------
    n_planet_features : int
        Number of features per planet token (N_PLANET_FEATURES).
    n_global_features : int
        Dimension of the global state vector.
    d_model : int
        Internal hidden dimension for all attention layers.
    n_heads : int
        Number of attention heads (must divide d_model).
    n_isab_layers : int
        Number of stacked ISAB blocks (2 recommended).
    n_inducing : int
        Number of inducing points per ISAB layer (32–64 typical).
    dropout : float
        Currently unused (attention layers don't apply dropout in on-policy RL).
    """

    def __init__(
        self,
        n_planet_features: int,
        n_global_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projection: planet features → d_model
        self.planet_proj = nn.Linear(n_planet_features, d_model)
        # No positional encoding — planet tokens are permutation-invariant.

        # ISAB stack
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing)
            for _ in range(n_isab_layers)
        ])

        # Actor head: per-token score conditioned on both the token AND the global
        # mission state.  This allows logits to depend on mission-wide information
        # (elapsed time, coverage, accumulated science, etc.) in addition to
        # per-planet features.
        #   contextualised_token (d)  ‖  global_embedding (d)  →  (2d)  →  logit
        self.global_proj_actor = nn.Linear(n_global_features, d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Critic: PMA reduces the set to a single summary vector,
        # concatenated with global features before the value MLP.
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
        planets: th.Tensor,                         # (B, N_max, n_pf)
        global_feat: th.Tensor,                      # (B, n_gf)
        padding_mask: Optional[th.Tensor] = None,   # (B, N_max) bool, True = pad
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns
        -------
        logits : (B, N_max) — raw per-planet action scores (before masking)
        values : (B,)       — scalar state-value estimate
        """
        # Sanitize inputs: MPS can produce NaN from upstream computations.
        # nan_to_num is unconditional — the conditional `.all()` check is itself
        # unreliable on MPS when the tensor already contains NaN.
        planets = th.nan_to_num(planets, nan=0.0, posinf=3.0, neginf=-3.0)

        # --- Embed planet tokens ---
        tokens = self.planet_proj(planets)          # (B, N, d)

        # Zero out padding rows before ISAB so they contribute zero keys/values.
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # --- ISAB layers ---
        for isab in self.isab_layers:
            tokens = isab(tokens, key_padding_mask=padding_mask)   # (B, N, d)
            # Unconditional NaN scrub after every ISAB layer.
            # MPS attention kernels can produce NaN even with the float attn_mask
            # fix; scrubbing here prevents one bad layer from poisoning the rest.
            tokens = th.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- Actor: per-token logit conditioned on global mission state ---
        N = tokens.shape[1]
        g_actor  = self.global_proj_actor(global_feat)             # (B, d)
        g_expand = g_actor.unsqueeze(1).expand(-1, N, -1)          # (B, N, d)
        actor_in = th.cat([tokens, g_expand], dim=-1)              # (B, N, 2d)
        logits   = self.actor_head(actor_in).squeeze(-1)           # (B, N)
        logits   = th.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- Critic: PMA → global embed → value ---
        summary   = self.pma(tokens, key_padding_mask=padding_mask).squeeze(1)  # (B, d)
        g_critic  = self.global_proj_critic(global_feat)           # (B, d)
        critic_in = th.cat([summary, g_critic], dim=-1)            # (B, 2d)
        values    = self.value_mlp(critic_in).squeeze(-1)          # (B,)
        values    = th.nan_to_num(values, nan=0.0)

        return logits, values


# ---------------------------------------------------------------------------
# SB3 MaskableActorCriticPolicy wrapper
# ---------------------------------------------------------------------------

class FullSetISABPolicy(MaskableActorCriticPolicy):
    """
    SB3 MaskablePPO-compatible policy using the ISAB set transformer.

    Expects observation_space to be a Dict with:
        "planets" : Box(N_max, n_planet_features)
        "global"  : Box(n_global_features,)

    Uses the same action-mask interface as ``ArielTransformerPolicy``.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        n_inducing: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        # Pass net_arch=[] so SB3 builds a minimal default MLP that we never use.
        # We override forward/evaluate_actions/predict_values to use isab_net.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        # Build the ISAB network AFTER super().__init__ (which creates the nn.Module)
        n_pf = observation_space["planets"].shape[-1]
        n_gf = observation_space["global"].shape[0]

        self.isab_net = FullSetISABNet(
            n_planet_features=n_pf,
            n_global_features=n_gf,
            d_model=d_model,
            n_heads=n_heads,
            n_isab_layers=n_isab_layers,
            n_inducing=n_inducing,
            dropout=dropout,
        )

        # Rebuild optimizer to include isab_net parameters
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(
        self, obs: dict
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Extract planet/global tensors and build the padding mask."""
        planets     = obs["planets"]   # (B, N_max, n_pf)
        global_feat = obs["global"]    # (B, n_gf)

        # Padding mask: a row is padding if *all* features are exactly zero.
        # This matches the zero-padding added by ArielEnv._candidates_full_set.
        pad_mask = (planets.abs().sum(dim=-1) == 0.0)   # (B, N_max) bool

        return planets, global_feat, pad_mask

    def _predict_logits_and_values(
        self,
        obs: dict,
        action_masks: Optional[th.Tensor],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Run forward pass; return (logits, values) with masks applied."""
        planets, global_feat, pad_mask = self._obs_to_tensors(obs)
        logits, values = self.isab_net(planets, global_feat, padding_mask=pad_mask)

        # Belt-and-suspenders NaN scrub on raw logits before any masking.
        # isab_net already applies nan_to_num internally, but an unconditional
        # guard here ensures nothing slips through to masked_fill.
        logits = th.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Force padding positions to -inf so they can never be sampled.
        if pad_mask.any():
            logits = logits.masked_fill(pad_mask, float("-inf"))

        # Apply SB3 action mask (completed targets, infeasible events, etc.)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))

        return logits, values

    # ------------------------------------------------------------------
    # MaskableActorCriticPolicy interface
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Returns (actions, values, log_probs)."""
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}

        mask_t = (
            th.as_tensor(action_masks, dtype=th.bool).to(self.device)
            if action_masks is not None else None
        )

        logits, values = self._predict_logits_and_values(obs_t, mask_t)
        dist = CategoricalDistribution(int(self.action_space.n))
        dist = dist.proba_distribution(action_logits=logits)

        actions   = dist.get_actions(deterministic=deterministic)
        log_probs = dist.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: dict,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Returns (values, log_probs, entropy) for PPO update."""
        obs_t = {k: th.as_tensor(v, dtype=th.float32).to(self.device)
                 for k, v in obs.items()}
        logits, values = self._predict_logits_and_values(obs_t, action_masks)

        # Guard: if all actions in a row are masked to -inf (e.g. the last step
        # of an episode that terminated with no valid actions), Categorical's
        # softmax computes 0/0 = NaN.  Replace all-inf rows with zeros so the
        # distribution is uniform — the log_prob for the actual action taken is
        # still effectively undefined, but the loss is finite and training continues.
        all_inf = (logits == float("-inf")).all(dim=-1, keepdim=True)  # (B, 1)
        if all_inf.any():
            logits = logits.masked_fill(all_inf, 0.0)

        # Use torch.distributions.Categorical directly with validate_args=False.
        # CategoricalDistribution (SB3 wrapper) may use the global PyTorch
        # validate_args=True setting, which rejects -inf logits even though they
        # are perfectly valid for masked policies.
        dist = th.distributions.Categorical(logits=logits, validate_args=False)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        return values, log_probs, entropy

    def predict_values(self, obs: dict) -> th.Tensor:
        """Returns scalar state values (B,)."""
        obs_t = {k: th.as_tensor(v).to(self.device) for k, v in obs.items()}
        planets, global_feat, pad_mask = self._obs_to_tensors(obs_t)
        _, values = self.isab_net(planets, global_feat, padding_mask=pad_mask)
        return values.unsqueeze(-1)   # (B, 1) as expected by SB3

    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """Greedy/stochastic prediction (inference only)."""
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions
