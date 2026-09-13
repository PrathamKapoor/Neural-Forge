"""Learned routing over heterogeneous block identities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RoutingDecision:
    weights: torch.Tensor
    logits: torch.Tensor
    entropy: torch.Tensor

    def expected_module_cost(self, costs: torch.Tensor) -> torch.Tensor:
        if costs.shape != (self.weights.shape[-1],):
            raise ValueError("costs must provide one value per module")
        return self.weights @ costs.to(self.weights)


class SoftRouter(nn.Module):
    """A state-conditioned, temperature-scaled softmax router.

    It pools the current sequence state, so route selection adds O(SD) pooling
    plus a small MLP. This overhead is reported independently by evaluation.
    """

    def __init__(self, hidden_dim: int, module_names: tuple[str, ...], temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.module_names = module_names
        self.temperature = temperature
        self.network = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(module_names)))

    def forward(self, state: torch.Tensor) -> RoutingDecision:
        if state.ndim != 3:
            raise ValueError("router expects shared sequence state [B,S,D]")
        logits = self.network(state.mean(dim=1))
        weights = torch.softmax(logits / self.temperature, dim=-1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return RoutingDecision(weights=weights, logits=logits, entropy=entropy)
