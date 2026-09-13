"""Heterogeneous computational primitives with compatible shared-state outputs."""

from __future__ import annotations

import torch
from torch import nn

from neuroforge.core.representations import RingGraphAdapter, SequenceToVectorAdapter


class MLPBlock(nn.Module):
    """Feature-oriented block: pool sequence, transform features, then broadcast."""

    name = "mlp"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.adapter = SequenceToVectorAdapter()
        self.network = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        vector = self.adapter(state)
        return state + self.network(vector.values).unsqueeze(1), vector.estimated_cost + float(2 * self.hidden_dim**2)


class GraphBlock(nn.Module):
    """Local relational block over an explicit ring-graph representation."""

    name = "graph"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.adapter = RingGraphAdapter()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        graph = self.adapter(state)
        messages = torch.bmm(graph.adjacency, self.message(graph.node_features))
        return state + torch.tanh(self.update(torch.cat((state, messages), dim=-1))), graph.estimated_cost + float(3 * self.hidden_dim**2)


class AttentionBlock(nn.Module):
    """Contextual block: every sequence position can attend to every other position."""

    name = "attention"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        attended, _ = self.attention(state, state, state, need_weights=False)
        sequence = state.shape[1]
        return state + self.projection(attended), float(4 * sequence * self.hidden_dim**2 + 2 * sequence**2 * self.hidden_dim)
