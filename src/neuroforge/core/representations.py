"""Explicit representation adapters; their costs are part of every route estimate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VectorRepresentation:
    values: torch.Tensor
    estimated_cost: float


@dataclass(frozen=True)
class SequenceRepresentation:
    values: torch.Tensor
    estimated_cost: float = 0.0


@dataclass(frozen=True)
class GraphRepresentation:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    estimated_cost: float


class SequenceToVectorAdapter:
    """Mean-pools [B,S,D] to [B,D]; temporal/order information is discarded."""

    def __call__(self, state: torch.Tensor) -> VectorRepresentation:
        if state.ndim != 3:
            raise ValueError("SequenceToVectorAdapter expects [batch, sequence, features]")
        _, sequence, features = state.shape
        return VectorRepresentation(state.mean(dim=1), float(sequence * features))


class RingGraphAdapter:
    """Maps a sequence to a ring graph, retaining node features but imposing local topology."""

    def __call__(self, state: torch.Tensor) -> GraphRepresentation:
        if state.ndim != 3:
            raise ValueError("RingGraphAdapter expects [batch, sequence, features]")
        batch, sequence, _ = state.shape
        adjacency = torch.zeros(sequence, sequence, device=state.device, dtype=state.dtype)
        indices = torch.arange(sequence, device=state.device)
        adjacency[indices, (indices + 1) % sequence] = 1
        adjacency[indices, (indices - 1) % sequence] = 1
        adjacency.fill_diagonal_(1)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        return GraphRepresentation(state, adjacency.expand(batch, -1, -1), float(sequence * 2))
