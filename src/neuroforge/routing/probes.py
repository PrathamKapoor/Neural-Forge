"""Minimal descriptor-conditioned routers for identifiability diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ProbeDecision:
    logits: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor


class DescriptorRouter(nn.Module):
    """A router over structural descriptors, with soft or top-1 hard execution views."""

    def __init__(self, input_dim: int, hidden_dim: int = 12, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 3))

    def forward(self, descriptor: torch.Tensor, mode: str = "soft") -> ProbeDecision:
        if mode not in {"soft", "hard", "straight_through"}: raise ValueError("mode must be 'soft', 'hard', or 'straight_through'")
        logits = self.network(descriptor)
        soft = torch.softmax(logits / self.temperature, dim=-1)
        hard = torch.nn.functional.one_hot(soft.argmax(-1), 3).to(soft.dtype)
        weights = soft if mode == "soft" else hard if mode == "hard" else hard + soft - soft.detach()
        return ProbeDecision(logits, weights, -(soft * soft.clamp_min(1e-8).log()).sum(-1))
