"""
Transformer-based actor-critic policy for ArielEnv.

Architecture
------------
The observation is a Dict with two arrays:

    "events"  (N, n_ef)  — one row per candidate target/event
    "global"  (n_gf,)    — mission-level state vector

A [CLS] token is built from the global features and prepended to the event
tokens.  The transformer encoder lets every event token attend to every other
event token, so the model can reason about interactions (e.g. spatially
co-located cheap targets, approaching deadlines on others).

    global (n_gf,)  ──► Linear ──► [CLS] token  (d_model,)
                                         │
    events (N, n_ef) ──► Linear ──► event tokens (N, d_model)
                                         │
                           cat([CLS], event_tokens) → (N+1, d_model)
                                         │
                        TransformerEncoder  (n_layers × self-attention + FFN)
                        src_key_padding_mask: True for completed/invalid targets
                                         │
                ┌────────────────────────┤
                │                        │
          [CLS] output             event outputs
          value_head                 policy_head
             │                          │
         V(s) scalar             logits (N,) → mask → softmax → π(a|s)

No positional encoding is used — the ordering of target rows is arbitrary
and the model should be permutation-equivariant.

Usage
-----
    from sb3_contrib import MaskablePPO
    from aRieL.agents.policies.event_attention_policy import ArielTransformerPolicy

    model = MaskablePPO(
        ArielTransformerPolicy,
        env,
        policy_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 2},
        verbose=1,
    )
    model.learn(total_timesteps=1_000_000)

References
----------
Kool et al. (2019), "Attention, Learn to Solve Routing Problems!" — pointer
network approach for combinatorial optimisation that directly inspired this.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# ---------------------------------------------------------------------------
# Transformer network module
# ---------------------------------------------------------------------------

class ArielTransformerNet(nn.Module):
    """
    Core transformer encoder that maps (events, global) → (action_logits, value).

    Parameters
    ----------
    n_event_features : int
        Number of features per event token (16 with the default config).
    n_global_features : int
        Dimension of the mission-level global state vector (25 default).
    d_model : int
        Internal transformer hidden dimension.
    n_heads : int
        Number of attention heads (must divide d_model evenly).
    n_layers : int
        Number of TransformerEncoderLayer blocks.
    d_ff_mult : int
        FFN inner dimension = d_model × d_ff_mult.
    dropout : float
        Attention / FFN dropout.  Typically 0.0 for on-policy RL where each
        transition is only used once.
    """

    def __init__(
        self,
        n_event_features: int,
        n_global_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projections — no positional encoding; target ordering is arbitrary
        self.event_proj = nn.Linear(n_event_features, d_model)
        self.global_proj = nn.Linear(n_global_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * d_ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable gradient flow than post-LN
        )
        # Pre-LN prevents PyTorch's nested-tensor fast-path; suppress the
        # informational warning it generates (no effect on correctness).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.policy_head = nn.Linear(d_model, 1)  # one score per event token
        self.value_head  = nn.Linear(d_model, 1)  # scalar value from [CLS]

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small gain → near-uniform initial policy; large gain → confident value
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self,
        events: th.Tensor,                         # (B, N, n_ef)
        global_feat: th.Tensor,                    # (B, n_gf)
        padding_mask: Optional[th.Tensor] = None,  # (B, N) bool, True = ignore token
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Parameters
        ----------
        events : Tensor (B, N, n_event_features)
        global_feat : Tensor (B, n_global_features)
        padding_mask : Tensor (B, N) bool, optional
            True where a target token should be masked out of attention
            (completed targets or padded rows).  Derived from ~action_mask.

        Returns
        -------
        logits : Tensor (B, N) — raw (unmasked) per-token action logits
        value  : Tensor (B, 1) — scalar state-value estimate from [CLS]
        """
        B = events.shape[0]

        tok = self.event_proj(events)                      # (B, N, d)
        cls = self.global_proj(global_feat).unsqueeze(1)   # (B, 1, d)
        x   = th.cat([cls, tok], dim=1)                    # (B, N+1, d)

        # Build the padding mask for the encoder: CLS token (position 0) always
        # participates in attention; invalid/completed targets are masked out.
        if padding_mask is not None:
            cls_col   = th.zeros(B, 1, dtype=th.bool, device=x.device)
            full_mask = th.cat([cls_col, padding_mask], dim=1)  # (B, N+1)
        else:
            full_mask = None

        x = self.encoder(x, src_key_padding_mask=full_mask)   # (B, N+1, d)

        logits = self.policy_head(x[:, 1:, :]).squeeze(-1)    # (B, N)
        value  = self.value_head(x[:, 0, :])                   # (B, 1)

        return logits, value


# ---------------------------------------------------------------------------
# SB3-compatible maskable policy wrapper
# ---------------------------------------------------------------------------

class ArielTransformerPolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy backed by ArielTransformerNet.

    The standard SB3 feature-extractor → MLP-extractor pipeline is
    bypassed.  ``forward``, ``evaluate_actions``, ``predict_values``, and
    ``_predict`` are fully overridden to drive the transformer directly from
    the raw ``{"events", "global"}`` observation dict.

    Parameters (pass via ``policy_kwargs`` in MaskablePPO)
    -------------------------------------------------------
    d_model : int        Transformer hidden dim          (default 128)
    n_heads : int        Number of attention heads       (default 4)
    n_layers : int       Encoder depth                   (default 2)
    dropout : float      Attention dropout               (default 0.0)
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
        # net_arch=[] → minimal default SB3 MLP (never called in practice;
        # we override forward/evaluate_actions to use transformer_net instead)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        # Build transformer after nn.Module is initialised by super().__init__
        n_ef = observation_space["events"].shape[-1]
        n_gf = observation_space["global"].shape[0]

        self.transformer_net = ArielTransformerNet(
            n_event_features=n_ef,
            n_global_features=n_gf,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Rebuild optimiser so transformer_net parameters are included
        # (super().__init__ created one before transformer_net existed)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensors(
        self, obs: dict
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Return (events, global_feat) as float32 tensors on the policy device.

        Handles both numpy arrays (during env rollout collection) and tensors
        (during PPO training update when SB3 has already converted the buffer).
        """
        def _to(x):
            if isinstance(x, th.Tensor):
                return x.to(device=self.device, dtype=th.float32)
            return th.as_tensor(x, device=self.device, dtype=th.float32)

        return _to(obs["events"]), _to(obs["global"])

    def _to_padding_mask(
        self,
        action_masks,
        batch_size: int,
    ) -> Optional[th.Tensor]:
        """
        Convert action_masks (True = valid) to a padding mask (True = ignore)
        for use as ``src_key_padding_mask`` in the transformer encoder.
        """
        if action_masks is None:
            return None
        if isinstance(action_masks, np.ndarray):
            valid = th.tensor(action_masks, dtype=th.bool, device=self.device)
        else:
            valid = action_masks.to(device=self.device, dtype=th.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand(batch_size, -1)
        return ~valid   # True = mask out (ignore in attention)

    def _apply_logit_mask(
        self,
        logits: th.Tensor,
        action_masks,
    ) -> th.Tensor:
        """Set logits for invalid actions to -inf so they get zero probability."""
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

    # ------------------------------------------------------------------
    # Core SB3 policy methods
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
        action_masks=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        events, global_f = self._obs_to_tensors(obs)
        B = events.shape[0]

        padding_mask = self._to_padding_mask(action_masks, B)
        logits, values = self.transformer_net(events, global_f, padding_mask)
        logits = self._apply_logit_mask(logits, action_masks)

        dist    = self._build_dist(logits)
        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions.reshape(-1), values, log_prob

    def evaluate_actions(
        self,
        obs: dict,
        actions: th.Tensor,
        action_masks=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        events, global_f = self._obs_to_tensors(obs)
        B = events.shape[0]

        padding_mask = self._to_padding_mask(action_masks, B)
        logits, values = self.transformer_net(events, global_f, padding_mask)
        logits = self._apply_logit_mask(logits, action_masks)

        dist     = self._build_dist(logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: dict) -> th.Tensor:
        events, global_f = self._obs_to_tensors(obs)
        _, values = self.transformer_net(events, global_f)
        return values

    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
        action_masks=None,
    ) -> th.Tensor:
        actions, _, _ = self.forward(
            observation,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        return actions
