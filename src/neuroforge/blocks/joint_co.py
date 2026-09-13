"""Phase 13 JointCo Block: the Joint expert with a small inter-branch fusion.

This is a strictly additive module. It reuses the existing `JointBlock` design
but adds a single small inter-branch fusion step: the three branch deltas are
concatenated, passed through a single `H*3 -> H` linear, and the fusion
output is added to the existing sum. This is the smallest single intervention
that makes the branches cross-talk before the residual sum, distinct from
ordinary end-to-end joint training of the Phase 12 JointBlock.

Parameter overhead: a single `Linear(3*H, H)` and `Linear(3*H, 3)` (gate).
For H=24: 72*24 + 24 = 1752 + 24*3 + 3 = 1779 extra parameters.

Architecture:
    feat_delta, rel_delta, ctx_delta (computed identically to JointBlock)
        |
        v
    concat (3H) -> Linear (3H -> 3) -> sigmoid -> per-branch gate
    concat (3H) -> Linear (3H -> H) -> fused_delta
        |
        v
    gated_fused = gate_f * fused_delta
    delta = sum(branch_scale * branch_delta) + gated_fused
        |
        v
    state + delta

The cross-talk (fused_delta) is gated by a learned per-branch sigmoid so the
fusion contribution can be annealed to zero if the branches are not yet
co-adapted.
"""
from __future__ import annotations

import torch
from torch import nn

from neuroforge.core.representations import (
    RingGraphAdapter,
    SequenceToVectorAdapter,
)


class JointCoBlock(nn.Module):
    """JointBlock + a small inter-branch fusion layer (H*3 -> H).

    Behaves like `JointBlock` for the three branch deltas, then mixes them
    through a small linear fusion before adding to the residual state.

    The learnable per-branch scales are kept from JointBlock and the
    fusion contribution is gated by a per-branch sigmoid (initialised at
    0 so the gate starts as zero contribution; training can turn it on).
    """

    name = "joint_co"

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

        # --- Feature substep: pooled MLP broadcast (like JointBlock) ---
        self.feature_pool = SequenceToVectorAdapter()
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Relational substep: ring-graph message passing (like JointBlock) ---
        self.graph_adapter = RingGraphAdapter()
        self.graph_message = nn.Linear(hidden_dim, hidden_dim)
        self.graph_update = nn.Linear(hidden_dim * 2, hidden_dim)

        # --- Contextual substep: V2-style retrieval (like JointBlock) ---
        self.context_value = nn.Linear(hidden_dim, hidden_dim)
        self.context_output = nn.Linear(hidden_dim, hidden_dim)

        # --- Per-branch learnable scales (same as JointBlock) ---
        self.feature_scale = nn.Parameter(torch.tensor(init_scale))
        self.graph_scale = nn.Parameter(torch.tensor(init_scale))
        self.context_scale = nn.Parameter(torch.tensor(init_scale))

        # --- Inter-branch fusion: H*3 -> H linear (single small layer) ---
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        # Per-branch gate: H*3 -> 3 (sigmoid), initialised so the bias
        # is negative so the gate starts near zero (intervention is OFF at
        # init; training must turn it on).
        self.fusion_gate = nn.Linear(hidden_dim * 3, 3)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -3.0)
        nn.init.zeros_(self.fusion.weight)

    def forward(
        self,
        state: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        if state.ndim != 3:
            raise ValueError(f"JointCoBlock expects [B, S, H], got {state.shape}")
        b, s, h = state.shape
        if h != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: expected {self.hidden_dim}, got {h}")
        cost = 0.0

        # 1. Feature substep.
        pooled = self.feature_pool(state)
        feat_delta = self.feature_net(pooled.values)  # [B, H]
        feat_delta = feat_delta.unsqueeze(1).expand(b, s, h)  # [B, S, H]
        cost += float(2 * self.hidden_dim * self.hidden_dim)

        # 2. Relational substep.
        graph = self.graph_adapter(state)
        msgs = torch.bmm(graph.adjacency, self.graph_message(graph.node_features))
        rel_delta = torch.tanh(
            self.graph_update(torch.cat((state, msgs), dim=-1))
        )
        cost += float(3 * self.hidden_dim * self.hidden_dim)

        # 3. Contextual substep.
        if raw_features is None:
            ctx_delta = torch.zeros_like(state)
        else:
            if raw_features.shape[-1] < 5:
                raise ValueError(
                    "JointCoBlock contextual substep requires raw_features with "
                    f"channel 4 (marker) and channels 0:3 (keys); got shape {raw_features.shape}"
                )
            query_index = raw_features[:, :, 4].argmax(dim=1)
            keys = torch.nn.functional.normalize(raw_features[:, :, :3], dim=-1)
            q = keys[torch.arange(b, device=state.device), query_index]
            scores = (keys * q[:, None]).sum(dim=-1) / self.temperature
            scores[torch.arange(b, device=state.device), query_index] = float("-inf")
            weights = scores.softmax(dim=1)
            retrieved = torch.bmm(weights.unsqueeze(1), self.context_value(state)).squeeze(1)
            update = torch.zeros_like(state)
            update[torch.arange(b, device=state.device), query_index] = self.context_output(retrieved)
            ctx_delta = update
        cost += float(2 * s * self.hidden_dim * self.hidden_dim + 2 * s * 3)

        # 4. Per-branch scale.
        scaled = (
            self.feature_scale * feat_delta
            + self.graph_scale * rel_delta
            + self.context_scale * ctx_delta
        )

        # 5. Inter-branch fusion (gated).
        stacked = torch.stack([feat_delta, rel_delta, ctx_delta], dim=-1)  # [B, S, H, 3]
        b_, s_, h_, k_ = stacked.shape
        flat = stacked.reshape(b_ * s_, h_ * k_)  # [B*S, 3H]
        gate = torch.sigmoid(self.fusion_gate(flat))
        fused = self.fusion(flat)
        gate_scalar = gate.mean(dim=-1, keepdim=True)
        fused_gated = fused * gate_scalar
        fused_gated = fused_gated.reshape(b_, s_, h_)
        cost += float(2 * 3 * self.hidden_dim * 3 * self.hidden_dim)

        # 6. Sum and residual.
        delta = scaled + fused_gated
        out = state + delta
        return out, cost

    def forward_with_intermediates(
        self,
        state: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all internal states for Phase 14 readout diagnosis.

        Returns a dict with keys:
          - "input": the encoder state (before the block)
          - "feat_delta": per-position feature-branch delta `[B, S, H]`
          - "rel_delta": per-position relational-branch delta `[B, S, H]`
          - "ctx_delta": per-position contextual-branch delta `[B, S, H]`
          - "scaled_sum": the per-branch-scaled sum `[B, S, H]`
          - "fused_delta": the inter-branch fusion delta `[B, S, H]`
          - "block_output": the state after the residual update `[B, S, H]`

        This is additive (does NOT change the existing forward signature).
        Used by Phase 14's pooling/head diagnosis.
        """
        if state.ndim != 3:
            raise ValueError(f"JointCoBlock expects [B, S, H], got {state.shape}")
        b, s, h = state.shape
        if h != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: expected {self.hidden_dim}, got {h}")
        # 1. Feature substep.
        pooled = self.feature_pool(state)
        feat_delta = self.feature_net(pooled.values)  # [B, H]
        feat_delta = feat_delta.unsqueeze(1).expand(b, s, h)  # [B, S, H]
        # 2. Relational substep.
        graph = self.graph_adapter(state)
        msgs = torch.bmm(graph.adjacency, self.graph_message(graph.node_features))
        rel_delta = torch.tanh(
            self.graph_update(torch.cat((state, msgs), dim=-1))
        )
        # 3. Contextual substep.
        if raw_features is None:
            ctx_delta = torch.zeros_like(state)
        else:
            if raw_features.shape[-1] < 5:
                raise ValueError(
                    "JointCoBlock contextual substep requires raw_features with "
                    f"channel 4 (marker) and channels 0:3 (keys); got shape {raw_features.shape}"
                )
            query_index = raw_features[:, :, 4].argmax(dim=1)
            keys = torch.nn.functional.normalize(raw_features[:, :, :3], dim=-1)
            q = keys[torch.arange(b, device=state.device), query_index]
            scores = (keys * q[:, None]).sum(dim=-1) / self.temperature
            scores[torch.arange(b, device=state.device), query_index] = float("-inf")
            weights = scores.softmax(dim=1)
            retrieved = torch.bmm(weights.unsqueeze(1), self.context_value(state)).squeeze(1)
            update = torch.zeros_like(state)
            update[torch.arange(b, device=state.device), query_index] = self.context_output(retrieved)
            ctx_delta = update
        # 4. Per-branch scale.
        scaled = (
            self.feature_scale * feat_delta
            + self.graph_scale * rel_delta
            + self.context_scale * ctx_delta
        )
        # 5. Fusion.
        stacked = torch.stack([feat_delta, rel_delta, ctx_delta], dim=-1)
        b_, s_, h_, k_ = stacked.shape
        flat = stacked.reshape(b_ * s_, h_ * k_)
        gate = torch.sigmoid(self.fusion_gate(flat))
        fused = self.fusion(flat)
        gate_scalar = gate.mean(dim=-1, keepdim=True)
        fused_gated = fused * gate_scalar
        fused_gated = fused_gated.reshape(b_, s_, h_)
        # 6. Residual.
        delta = scaled + fused_gated
        out = state + delta
        return {
            "input": state,
            "feat_delta": feat_delta,
            "rel_delta": rel_delta,
            "ctx_delta": ctx_delta,
            "scaled_sum": scaled,
            "fused_delta": fused_gated,
            "block_output": out,
        }
