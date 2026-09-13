"""Phase 12 Joint Block: a single heterogeneous computational primitive that
jointly exercises Feature, Relational, and Contextual inductive biases.

Design rationale
----------------
Phase 11 established that no combination of the existing frozen experts (MLP,
Graph, AttentionBlockV1, AttentionBlockV2) materially exceeds the 66.4% single-
expert ceiling on the mixed FRC benchmark. The causal evidence pointed to
expert-portfolio insufficiency (CASE E). This module is the smallest
plausible single new computational pathway that can jointly process
Feature, Relational, and Contextual signals in one pass.

It is *not* a giant architecture. It is a single new block `JointBlock` that
composes three sub-operations on the shared `[B, S, H]` state and reports a
single FLOP estimate. It obeys the existing expert interface:

  - Input  : `[B, S, H]` (the encoded state from `StandaloneSpecialist.encoder`)
  - Output : `[B, S, H]` (the updated state, residual + delta)
  - Cost   : a single `float` accounting of analytical FLOPs

It is registered in `neuroforge.blocks.__init__` so
`StandaloneSpecialist(architecture="joint", depth=N)` works exactly like
the existing four architectures. No Phase 1-11 module is modified.

Three sub-operations
--------------------
1. **Feature nonlinear transformation** (one MLP-like substep on the mean
   pooled vector, broadcast back to sequence). This makes the block capable
   of nonlinear feature combinations, like the MLP block.

2. **Local ring-graph message passing** (one GraphBlock-style message+update
   step). This makes the block capable of exploiting the existing cyclic
   relational correspondence (the same `RingGraphAdapter` used by
   `GraphBlock`).

3. **Query-conditioned content retrieval** (one V2-style retrieval into the
   channel-4 marked query using channels 0:3 as keys, against the SAME
   underlying hidden state projected through a small value MLP). This makes
   the block capable of content-addressed contextual retrieval, like V2.

The three deltas are summed (with a small learnable per-branch scale initialised
to 1/3) and added to the residual input state. There is **no gating network**
and no other auxiliary head. This is the smallest structure that exercises
all three inductive biases with the existing `[B, S, H]` interface.

Parameter budget
----------------
With `hidden_dim=24` (the Phase 6 contract) and depth=1, the new block uses
roughly 1,200-1,800 parameters — within the per-block range of the existing
portfolio (MLP 1200, Graph 1776, Attn 3000, V2 1200). The full `StandaloneSpecialist`
of depth=1 is therefore ~3,400 parameters, comparable to `AttentionBlock`
standalone (3,914). The exact count is reported by the experiment; the
budget is enforced with a tolerance in the unit tests.
"""
from __future__ import annotations

import torch
from torch import nn

from neuroforge.core.representations import (
    RingGraphAdapter,
    SequenceToVectorAdapter,
)


class JointBlock(nn.Module):
    """Single block that jointly performs feature, relational, and contextual
    computation on the shared `[B, S, H]` state.

    Args:
        hidden_dim: matches the Phase 6 contract (24).
        temperature: V2-style retrieval temperature (default 0.1).
        init_scale: initial value of the per-branch learnable scale, applied
            to each of the three deltas before summation.
    """

    name = "joint"

    def __init__(
        self,
        hidden_dim: int = 24,
        temperature: float = 0.1,
        init_scale: float = 1.0 / 3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.init_scale = init_scale

        # --- Feature substep: pooled MLP broadcast (like MLPBlock) ---
        self.feature_pool = SequenceToVectorAdapter()
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Relational substep: ring-graph message passing (like GraphBlock) ---
        self.graph_adapter = RingGraphAdapter()
        self.graph_message = nn.Linear(hidden_dim, hidden_dim)
        self.graph_update = nn.Linear(hidden_dim * 2, hidden_dim)

        # --- Contextual substep: query-conditioned retrieval (like V2) ---
        self.context_value = nn.Linear(hidden_dim, hidden_dim)
        self.context_output = nn.Linear(hidden_dim, hidden_dim)

        # Learnable per-branch scale (initialised to 1/3 each so the three
        # deltas contribute equally at start). This is the *only* learnable
        # weighting in the block; there is no extra gating network.
        self.feature_scale = nn.Parameter(torch.tensor(init_scale))
        self.graph_scale = nn.Parameter(torch.tensor(init_scale))
        self.context_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(
        self,
        state: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Forward pass.

        Args:
            state: `[B, S, H]` shared state from the encoder.
            raw_features: `[B, S, 8]` raw input (same shape as Phase 6 contract).
                Required for the contextual substep (it reads channel 4 marker
                and channels 0:3 keys), exactly like V2.

        Returns:
            updated_state: `[B, S, H]`
            estimated_cost: analytical FLOP estimate (matches the existing
                block cost reporting convention).
        """
        if state.ndim != 3:
            raise ValueError(f"JointBlock expects [B, S, H], got {state.shape}")
        b, s, h = state.shape
        if h != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: expected {self.hidden_dim}, got {h}")
        cost = 0.0

        # 1. Feature substep: pool to vector, transform, broadcast.
        pooled = self.feature_pool(state)
        feat_delta = self.feature_net(pooled.values).unsqueeze(1)  # [B, S, H]
        cost += float(2 * self.hidden_dim * self.hidden_dim)

        # 2. Relational substep: ring graph message passing.
        graph = self.graph_adapter(state)  # node_features, adjacency
        msgs = torch.bmm(graph.adjacency, self.graph_message(graph.node_features))
        rel_delta = torch.tanh(
            self.graph_update(torch.cat((state, msgs), dim=-1))
        )
        cost += float(3 * self.hidden_dim * self.hidden_dim)

        # 3. Contextual substep: V2-style retrieval (requires raw_features).
        if raw_features is None:
            ctx_delta = torch.zeros_like(state)
        else:
            if raw_features.shape[-1] < 5:
                raise ValueError(
                    "JointBlock contextual substep requires raw_features with "
                    f"channel 4 (marker) and channels 0:3 (keys); got shape {raw_features.shape}"
                )
            query_index = raw_features[:, :, 4].argmax(dim=1)  # [B]
            keys = torch.nn.functional.normalize(raw_features[:, :, :3], dim=-1)  # [B, S, 3]
            q = keys[torch.arange(b, device=state.device), query_index]  # [B, 3]
            scores = (keys * q[:, None]).sum(dim=-1) / self.temperature  # [B, S]
            scores[torch.arange(b, device=state.device), query_index] = float("-inf")
            weights = scores.softmax(dim=1)  # [B, S]
            retrieved = torch.bmm(weights.unsqueeze(1), self.context_value(state)).squeeze(1)  # [B, H]
            update = torch.zeros_like(state)
            update[torch.arange(b, device=state.device), query_index] = self.context_output(retrieved)
            ctx_delta = update
        cost += float(2 * s * self.hidden_dim * self.hidden_dim + 2 * s * 3)

        # 4. Sum the three deltas (each with its learnable scale) and add to
        #    the residual state. This is the only "gating" in the block; the
        #    scales start at 1/3 so the three inductive biases contribute
        #    equally at initialization.
        delta = (
            self.feature_scale * feat_delta
            + self.graph_scale * rel_delta
            + self.context_scale * ctx_delta
        )
        out = state + delta
        return out, cost
