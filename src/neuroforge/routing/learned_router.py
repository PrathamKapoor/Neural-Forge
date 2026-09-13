"""Sample-level learned router using permutation-invariant observable input statistics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnedRoutingDecision:
    logits: torch.Tensor
    weights: torch.Tensor
    selected_experts: torch.Tensor
    entropy: torch.Tensor
    soft_probabilities: torch.Tensor | None = None


class SampleLevelRouter(nn.Module):
    """Minimal learned router operating on observable Phase 6 input tensors [B, S, D].

    Uses a strictly permutation-invariant representation: sequence mean and standard
    deviation across tokens, preserving sample content without position-based shortcuts.
    Employs straight-through hard selection during training so gradients flow through
    the chosen routing paths.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        num_experts: int = 4,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.num_experts = num_experts
        # Input representation: mean [B, D] concatenated with std [B, D] = 2 * D dimensions
        self.rep_dim = input_dim * 2
        self.network = nn.Sequential(
            nn.Linear(self.rep_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts),
        )

    def extract_representation(self, features: torch.Tensor) -> torch.Tensor:
        """Extract permutation-invariant mean and standard deviation across tokens."""
        if features.ndim != 3 or features.shape[-1] != (self.rep_dim // 2):
            raise ValueError(
                f"features must have shape [B, S, {self.rep_dim // 2}], got {features.shape}"
            )
        mean = features.mean(dim=1)
        std = features.std(dim=1)
        return torch.cat([mean, std], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        mode: str = "straight_through",
    ) -> LearnedRoutingDecision:
        if mode not in {"straight_through", "hard", "soft"}:
            raise ValueError("mode must be 'straight_through', 'hard', or 'soft'")

        rep = self.extract_representation(features)
        logits = self.network(rep)
        soft = torch.softmax(logits / self.temperature, dim=-1)
        selected = logits.argmax(dim=-1)
        hard = F.one_hot(selected, num_classes=self.num_experts).to(soft.dtype)

        if mode == "hard":
            weights = hard
        elif mode == "straight_through":
            weights = hard + soft - soft.detach()
        else:
            weights = soft

        entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1)
        return LearnedRoutingDecision(
            logits=logits,
            weights=weights,
            selected_experts=selected,
            entropy=entropy,
            soft_probabilities=soft,
        )
