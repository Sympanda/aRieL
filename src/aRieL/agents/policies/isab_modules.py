"""
Set Transformer building blocks: MAB, ISAB, PMA.

Reference: Lee et al. (2019) "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks" — https://arxiv.org/abs/1810.00825

Three modules are provided:

    MAB (Multihead Attention Block)
        Cross-attention between a query set X and key-value set Y.
        Output shape matches X.

    ISAB (Induced Set Attention Block)
        O(N·m) set self-attention via m learned inducing points.
        ISAB_m(X) = MAB(X, MAB(I, X))  where I is a (m, d) learnable matrix.

    PMA (Pooling by Multihead Attention)
        Reduces a set X of N tokens to k summary vectors using k learned seeds.
        PMA_k(X) = MAB(S, rFF(X))  where S is a (k, d) learnable matrix.

Intended use
------------
    from aRieL.agents.policies.isab_modules import MAB, ISAB, PMA

    # Actor:
    isab1 = ISAB(d_model, n_heads, n_inducing)   # (B, N, d) → (B, N, d)
    isab2 = ISAB(d_model, n_heads, n_inducing)
    logit_head = nn.Linear(d_model, 1)            # (B, N, d) → (B, N, 1)

    # Critic:
    pma = PMA(d_model, n_heads, k=1)              # (B, N, d) → (B, 1, d)
    value_head = nn.Linear(d_model, 1)            # (B, 1, d) → (B, 1)

Implementation notes
--------------------
- rFF (row-wise FFN) is a single hidden-layer MLP applied independently per token.
- Pre-LN (normalise before attention) is used throughout for training stability.
- Padding positions are suppressed via a float additive attn_mask (-1e9) rather
  than a bool key_padding_mask, which has a known NaN bug on MPS (Apple Silicon).
"""

from __future__ import annotations

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class rFF(nn.Module):
    """Row-wise Feed-Forward network applied independently to each token."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class MAB(nn.Module):
    """Multihead Attention Block.

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + MultiheadAttention(X, Y, Y))

    X is the query set (B, Nx, d); Y is the key-value set (B, Ny, d).
    Output has shape (B, Nx, d) — same as X.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = rFF(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: th.Tensor,                           # (B, Nx, d) — query
        y: th.Tensor,                           # (B, Ny, d) — key / value
        key_padding_mask: th.Tensor | None = None,  # (B, Ny) bool, True = pad
    ) -> th.Tensor:
        """
        Parameters
        ----------
        x : (B, Nx, d) — query set
        y : (B, Ny, d) — key/value set
        key_padding_mask : (B, Ny) bool, optional
            True at positions that should be ignored (padding tokens in Y).
        """
        # MPS (Apple Silicon) has a known bug where nn.MultiheadAttention with a
        # bool key_padding_mask produces NaN attention weights — even with partial
        # masking, not only the all-masked case.  The fix is to bypass
        # key_padding_mask entirely and instead pass a float additive attn_mask
        # with shape (B*n_heads, L, S).  We use -1e9 rather than -inf so that
        # softmax over all-masked positions produces a uniform distribution
        # (~0 output) instead of 0/0 = NaN.
        attn_mask = None
        if key_padding_mask is not None:
            B  = x.shape[0]
            L  = x.shape[1]          # number of query positions
            S  = y.shape[1]          # number of key positions
            H  = self.attn.num_heads
            # (B, S) bool → (B, S) float, True=pad → -1e9, False=valid → 0.0
            float_mask = key_padding_mask.to(dtype=x.dtype) * -1e9   # (B, S)
            # Broadcast over query positions and heads → (B*H, L, S)
            float_mask = (
                float_mask
                .unsqueeze(1)           # (B, 1, S)
                .unsqueeze(1)           # (B, 1, 1, S)
                .expand(B, H, L, S)     # (B, H, L, S)
                .reshape(B * H, L, S)   # (B*H, L, S)  — required by MHA
            )
            attn_mask = float_mask

        # Pre-LN attention: norm before attending
        h, _ = self.attn(
            query=self.norm1(x),
            key=self.norm1(y),
            value=self.norm1(y),
            attn_mask=attn_mask,        # float additive mask, no bool key_padding_mask
        )
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class ISAB(nn.Module):
    """Induced Set Attention Block.

    ISAB_m(X) = MAB(X, MAB(I, X))

    I is a (m, d) matrix of learned inducing points.  The inner MAB reduces
    X to m summary vectors; the outer MAB broadcasts back to N tokens.
    Complexity: O(N·m) rather than O(N²) for full self-attention.
    """

    def __init__(self, d_model: int, n_heads: int, n_inducing: int = 32) -> None:
        super().__init__()
        self.inducing = nn.Parameter(th.empty(1, n_inducing, d_model))
        nn.init.xavier_uniform_(self.inducing)
        self.mab1 = MAB(d_model, n_heads)   # I ← X  (inducing ← tokens)
        self.mab2 = MAB(d_model, n_heads)   # X ← H  (tokens ← induced summary)

    def forward(
        self,
        x: th.Tensor,                           # (B, N, d)
        key_padding_mask: th.Tensor | None = None,  # (B, N) bool, True = pad
    ) -> th.Tensor:
        """
        Parameters
        ----------
        x : (B, N, d) — input token set
        key_padding_mask : (B, N) bool — True for padding positions in X.

        Returns
        -------
        (B, N, d) — contextualised token set (same shape as input)
        """
        b = x.size(0)
        I = self.inducing.expand(b, -1, -1)   # (B, m, d)
        # Inner: condense X → H via inducing points
        H = self.mab1(I, x, key_padding_mask=key_padding_mask)  # (B, m, d)
        # Outer: broadcast H → contextualised X
        return self.mab2(x, H)                                    # (B, N, d)


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    PMA_k(X) = MAB(S, rFF(X))

    S is a (k, d) matrix of learned seed vectors.  Reduces N tokens to k
    summary vectors.  For a scalar value function use k=1.
    """

    def __init__(self, d_model: int, n_heads: int, k: int = 1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(th.empty(1, k, d_model))
        nn.init.xavier_uniform_(self.seeds)
        self.ff  = rFF(d_model)   # row-wise FFN applied to X before pooling
        self.mab = MAB(d_model, n_heads)

    def forward(
        self,
        x: th.Tensor,                           # (B, N, d)
        key_padding_mask: th.Tensor | None = None,  # (B, N) bool, True = pad
    ) -> th.Tensor:
        """
        Parameters
        ----------
        x : (B, N, d) — token set to pool
        key_padding_mask : (B, N) bool — True for padding positions.

        Returns
        -------
        (B, k, d) — k pooled summary vectors
        """
        b = x.size(0)
        S = self.seeds.expand(b, -1, -1)    # (B, k, d)
        z = self.ff(x)                       # (B, N, d)  row-wise FFN
        return self.mab(S, z, key_padding_mask=key_padding_mask)  # (B, k, d)
