"""Phase 15 relational-substep variants of the JointCo block.

Strictly additive: does NOT modify `JointBlock`, `JointCoBlock`, `GraphBlock`,
or any other Phase 1-14 module. It reuses the exact Feature/Contextual/Fusion
machinery of `JointCoBlock` and only parameterises the *relational* substep
along three orthogonal axes:

  - ``rel_depth`` (1, 2, 3, ...): number of message-passing rounds.
  - ``aggregation`` ("mean" | "sum" | "max"): neighbourhood aggregation.
  - ``rel_update`` ("linear" | "mlp"): relational update transformation.

The default configuration (``rel_depth=1``, ``aggregation="mean"``,
``rel_update="linear"``) performs the same mathematics as the relational
substep of `JointCoBlock` (row-normalised ring adjacency, single
``Linear(2H, H)`` + ``tanh``), so depth-1/mean/linear is the exact-capacity
replica used as the Phase 15 control for "everything else identical".

Relational computation (per round r, with ``h_0 = state``)::

    m_r       = W_msg_r(h_r)                          # [B, S, H]
    msgs_r    = Aggregate(m_r) over ring neighbours   # [B, S, H]
    h_{r+1}   = tanh(W_upd_r([h_r ; msgs_r]))         # linear update
    h_{r+1}   = tanh(W2_r(GELU(W1_r([h_r ; msgs_r]))))# mlp update
    rel_delta = h_K                                    # [B, S, H]

Aggregations (fixed ring topology: self + left + right neighbour):

  - "mean": row-stochastic adjacency (self + neighbours averaged).
    This is exactly the existing `RingGraphAdapter` behaviour.
  - "sum": unnormalised adjacency (self + neighbours summed).
  - "max": element-wise maximum over {self, left, right} messages.
"""
from __future__ import annotations

import torch
from torch import nn

from neuroforge.core.representations import SequenceToVectorAdapter


VALID_AGGREGATIONS = ("mean", "sum", "max")
VALID_UPDATES = ("linear", "mlp")


def ring_neighbourhood_max(messages: torch.Tensor) -> torch.Tensor:
    """Element-wise max over {self, left, right} ring neighbours.

    Args:
        messages: `[B, S, H]` per-node messages.

    Returns:
        `[B, S, H]` pooled neighbourhood representation.
    """
    left = torch.roll(messages, shifts=1, dims=1)
    right = torch.roll(messages, shifts=-1, dims=1)
    return torch.maximum(torch.maximum(messages, left), right)


class JointCoRelationalBlock(nn.Module):
    """JointCoBlock with a configurable relational substep.

    All non-relational machinery (feature branch, contextual branch,
    per-branch scales, gated inter-branch fusion, residual connection) is
    identical to `JointCoBlock`. Only the relational delta computation is
    parameterised by depth / aggregation / update capacity.

    The per-branch learnable scales keep the names `feature_scale`,
    `graph_scale`, `context_scale` so Phase 13 branch-ablation utilities
    (`expert_with_branch_ablated`, `cross_evaluation_with_ablation`) work
    unchanged on experts using this block.
    """

    name = "joint_co_relational"

    def __init__(
        self,
        hidden_dim: int = 24,
        temperature: float = 0.1,
        init_scale: float = 1.0 / 3.0,
        rel_depth: int = 1,
        aggregation: str = "mean",
        rel_update: str = "linear",
        feature_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if rel_depth < 1:
            raise ValueError(f"rel_depth must be >= 1, got {rel_depth}")
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r}; valid: {VALID_AGGREGATIONS}")
        if rel_update not in VALID_UPDATES:
            raise ValueError(f"unknown rel_update {rel_update!r}; valid: {VALID_UPDATES}")
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.init_scale = init_scale
        self.rel_depth = rel_depth
        self.aggregation = aggregation
        self.rel_update = rel_update
        self.feature_hidden = feature_hidden if feature_hidden is not None else hidden_dim

        # --- Feature substep (identical to JointCoBlock when feature_hidden == hidden_dim) ---
        self.feature_pool = SequenceToVectorAdapter()
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, self.feature_hidden),
            nn.GELU(),
            nn.Linear(self.feature_hidden, hidden_dim),
        )

        # --- Relational substep (parameterised) ---
        self.graph_message = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(rel_depth)]
        )
        if rel_update == "linear":
            self.graph_update = nn.ModuleList(
                [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(rel_depth)]
            )
        else:  # "mlp": modestly wider two-layer update
            self.graph_update = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(rel_depth)
                ]
            )

        # --- Contextual substep (identical to JointCoBlock) ---
        self.context_value = nn.Linear(hidden_dim, hidden_dim)
        self.context_output = nn.Linear(hidden_dim, hidden_dim)

        # --- Per-branch learnable scales (identical to JointCoBlock) ---
        self.feature_scale = nn.Parameter(torch.tensor(init_scale))
        self.graph_scale = nn.Parameter(torch.tensor(init_scale))
        self.context_scale = nn.Parameter(torch.tensor(init_scale))

        # --- Inter-branch fusion (identical to JointCoBlock) ---
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim * 3, 3)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -3.0)
        nn.init.zeros_(self.fusion.weight)

    # ------------------------------------------------------------------
    # Relational helpers
    # ------------------------------------------------------------------
    def _neighbourhood_aggregate(self, messages: torch.Tensor) -> torch.Tensor:
        """Aggregate per-node messages over the fixed ring neighbourhood."""
        b, s, h = messages.shape
        if self.aggregation == "max":
            return ring_neighbourhood_max(messages)
        # Build the ring adjacency explicitly (same topology as RingGraphAdapter).
        adjacency = torch.zeros(s, s, device=messages.device, dtype=messages.dtype)
        indices = torch.arange(s, device=messages.device)
        adjacency[indices, (indices + 1) % s] = 1.0
        adjacency[indices, (indices - 1) % s] = 1.0
        adjacency.fill_diagonal_(1.0)
        if self.aggregation == "mean":
            adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        elif self.aggregation == "sum":
            pass  # unnormalised: self + neighbours summed
        return torch.bmm(adjacency.expand(b, -1, -1), messages)

    def _relational_delta(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Run K message-passing rounds starting from `state`.

        Returns (rel_delta, analytical_cost).
        """
        h = state
        cost = 0.0
        for r in range(self.rel_depth):
            msgs = self._neighbourhood_aggregate(self.graph_message[r](h))
            update_in = torch.cat((h, msgs), dim=-1)
            if self.rel_update == "linear":
                h = torch.tanh(self.graph_update[r](update_in))
                cost += float(3 * self.hidden_dim * self.hidden_dim)
            else:
                h = torch.tanh(self.graph_update[r](update_in))
                cost += float(4 * self.hidden_dim * self.hidden_dim)
        return h, cost

    def relational_param_count(self) -> int:
        """Number of parameters in the relational substep only."""
        return sum(p.numel() for p in self.graph_message.parameters()) + sum(
            p.numel() for p in self.graph_update.parameters()
        )

    # ------------------------------------------------------------------
    # Forward (same outer structure as JointCoBlock)
    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        if state.ndim != 3:
            raise ValueError(f"JointCoRelationalBlock expects [B, S, H], got {state.shape}")
        b, s, h = state.shape
        if h != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: expected {self.hidden_dim}, got {h}")
        cost = 0.0

        # 1. Feature substep (identical to JointCoBlock).
        pooled = self.feature_pool(state)
        feat_delta = self.feature_net(pooled.values)  # [B, H]
        feat_delta = feat_delta.unsqueeze(1).expand(b, s, h)  # [B, S, H]
        cost += float(2 * self.hidden_dim * self.hidden_dim)

        # 2. Relational substep (parameterised depth/aggregation/update).
        rel_delta, rel_cost = self._relational_delta(state)
        cost += rel_cost

        # 3. Contextual substep (identical to JointCoBlock).
        if raw_features is None:
            ctx_delta = torch.zeros_like(state)
        else:
            if raw_features.shape[-1] < 5:
                raise ValueError(
                    "JointCoRelationalBlock contextual substep requires raw_features with "
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

        # 4. Per-branch scale (identical to JointCoBlock).
        scaled = (
            self.feature_scale * feat_delta
            + self.graph_scale * rel_delta
            + self.context_scale * ctx_delta
        )

        # 5. Inter-branch fusion (identical to JointCoBlock).
        stacked = torch.stack([feat_delta, rel_delta, ctx_delta], dim=-1)  # [B, S, H, 3]
        b_, s_, h_, k_ = stacked.shape
        flat = stacked.reshape(b_ * s_, h_ * k_)  # [B*S, 3H]
        gate = torch.sigmoid(self.fusion_gate(flat))
        fused = self.fusion(flat)
        gate_scalar = gate.mean(dim=-1, keepdim=True)
        fused_gated = fused * gate_scalar
        fused_gated = fused_gated.reshape(b_, s_, h_)
        cost += float(2 * 3 * self.hidden_dim * 3 * self.hidden_dim)

        # 6. Sum and residual (identical to JointCoBlock).
        delta = scaled + fused_gated
        out = state + delta
        return out, cost

    def forward_with_intermediates(
        self,
        state: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Expose per-branch deltas (additive; mirrors JointCoBlock's method).

        Returns input / feat_delta / rel_delta (final round h_K) /
        rel_rounds (list of per-round states) / ctx_delta / scaled_sum /
        fused_delta / block_output. Added for Phase 17 branch-level diagnostics;
        does not alter forward behaviour.
        """
        if state.ndim != 3:
            raise ValueError(f"JointCoRelationalBlock expects [B, S, H], got {state.shape}")
        b, s, h = state.shape
        pooled = self.feature_pool(state)
        feat_delta = self.feature_net(pooled.values).unsqueeze(1).expand(b, s, h)
        rounds: list[torch.Tensor] = []
        hr = state
        for r in range(self.rel_depth):
            msgs = self._neighbourhood_aggregate(self.graph_message[r](hr))
            if self.rel_update == "linear":
                hr = torch.tanh(self.graph_update[r](torch.cat((hr, msgs), dim=-1)))
            else:
                hr = torch.tanh(self.graph_update[r](torch.cat((hr, msgs), dim=-1)))
            rounds.append(hr)
        rel_delta = hr
        if raw_features is None:
            ctx_delta = torch.zeros_like(state)
        else:
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
        scaled = (
            self.feature_scale * feat_delta
            + self.graph_scale * rel_delta
            + self.context_scale * ctx_delta
        )
        stacked = torch.stack([feat_delta, rel_delta, ctx_delta], dim=-1)
        b_, s_, h_, k_ = stacked.shape
        flat = stacked.reshape(b_ * s_, h_ * k_)
        gate = torch.sigmoid(self.fusion_gate(flat))
        fused = self.fusion(flat)
        fused_gated = (fused * gate.mean(dim=-1, keepdim=True)).reshape(b_, s_, h_)
        out = state + scaled + fused_gated
        return {
            "input": state,
            "feat_delta": feat_delta,
            "rel_delta": rel_delta,
            "rel_rounds": rounds,
            "ctx_delta": ctx_delta,
            "scaled_sum": scaled,
            "fused_delta": fused_gated,
            "block_output": out,
        }

    # ------------------------------------------------------------------
    # 15A machine-readable audit of the relational computation
    # ------------------------------------------------------------------
    def describe_relational_computation(self) -> dict[str, object]:
        """Return a machine-readable audit of the relational substep.

        Used by Phase 15A to document WHAT operation is performed without
        inferring behaviour from class names.
        """
        h = self.hidden_dim
        msg_per_round = h * h + h
        if self.rel_update == "linear":
            upd_per_round = (2 * h) * h + h
            update_desc = "tanh(Linear(2H -> H)([h ; msgs]))"
        else:
            upd_per_round = ((2 * h) * h + h) + (h * h + h)
            update_desc = "tanh(Linear(H -> H)(GELU(Linear(2H -> H)([h ; msgs]))))"
        return {
            "input_shape": "[B, S, H]",
            "graph_construction": "fixed ring: each node connects to self + left + right neighbour",
            "topology": "ring (circulant, degree 3 including self-loop)",
            "message_passing_rounds": self.rel_depth,
            "neighbourhood_aggregation": self.aggregation,
            "aggregation_semantics": {
                "mean": "row-normalised adjacency: average of self + 2 neighbours",
                "sum": "unnormalised adjacency: sum of self + 2 neighbours",
                "max": "element-wise maximum over {self, left, right} messages",
            }[self.aggregation],
            "self_feature_handling": "self features concatenated with neighbourhood messages ([h ; msgs])",
            "update_transformation": update_desc,
            "activation": "tanh",
            "normalisation": "none inside relational substep (LayerNorm only in final classifier head)",
            "residual_connection": "block-level residual: out = state + scaled_sum + gated_fusion",
            "output_shape": "[B, S, H]",
            "relational_param_count": self.relational_param_count(),
            "relational_params_per_round": {
                "message": msg_per_round,
                "update": upd_per_round,
            },
            "analytical_flops_per_round": (
                3 * h * h if self.rel_update == "linear" else 4 * h * h
            ),
            "when_applied": (
                "inside JointCo block, after the shared encoder linear, "
                "in parallel with the feature and contextual branches, "
                "before per-branch scaling, gated fusion and the residual sum"
            ),
        }
