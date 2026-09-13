"""Phase 10 Multi-Expert Routing: Top-K and Adaptive-K mechanisms with parallel aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiExpertRoutingDecision:
    """Decision structure for multi-expert sample-level routing."""
    logits: torch.Tensor
    soft_probabilities: torch.Tensor
    selected_indices: list[list[int]]
    weights: torch.Tensor
    k_values: torch.Tensor
    entropy: torch.Tensor
    k_logits: torch.Tensor | None = None
    k_soft_probabilities: torch.Tensor | None = None


def aggregate_parallel_uniform(
    expert_logits: list[torch.Tensor],
) -> torch.Tensor:
    """C1 — Uniform Averaging: simple arithmetic mean of expert logits.
    
    expert_logits: list of [B, num_classes] tensors.
    """
    if not expert_logits:
        raise ValueError("expert_logits cannot be empty")
    stacked = torch.stack(expert_logits, dim=0)  # [K, B, num_classes]
    return stacked.mean(dim=0)


def aggregate_parallel_weighted(
    expert_logits: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """C2 — Router-Weighted Averaging: weights expert logits by routing confidence.
    
    expert_logits: list of K tensors of shape [B, num_classes]
    weights: [B, K] normalized weights across the K selected experts.
    """
    if not expert_logits:
        raise ValueError("expert_logits cannot be empty")
    stacked = torch.stack(expert_logits, dim=1)  # [B, K, num_classes]
    # weights: [B, K] -> [B, K, 1]
    norm_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (stacked * norm_weights.unsqueeze(-1)).sum(dim=1)


def aggregate_parallel_normalized(
    expert_logits: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """C3 — Normalized Weighted Aggregation: transforms logits to probabilities before weighted combination.
    
    Addresses uncalibrated or disparate logit scales between diverse architectures.
    expert_logits: list of K tensors of shape [B, num_classes]
    weights: [B, K] normalized weights across the K selected experts.
    """
    if not expert_logits:
        raise ValueError("expert_logits cannot be empty")
    stacked_probs = torch.stack([F.softmax(z, dim=-1) for z in expert_logits], dim=1)  # [B, K, num_classes]
    norm_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    combined_probs = (stacked_probs * norm_weights.unsqueeze(-1)).sum(dim=1)  # [B, num_classes]
    return torch.log(combined_probs.clamp_min(1e-8))


class TopKRouter(nn.Module):
    """Router supporting fixed top-k hard and straight-through selection."""

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
        k: int = 2,
        mode: str = "hard",
    ) -> MultiExpertRoutingDecision:
        if mode not in {"straight_through", "hard", "soft"}:
            raise ValueError("mode must be 'straight_through', 'hard', or 'soft'")
        if k < 1 or k > self.num_experts:
            raise ValueError(f"k must be between 1 and {self.num_experts}, got {k}")

        rep = self.extract_representation(features)
        logits = self.network(rep)
        soft = torch.softmax(logits / self.temperature, dim=-1)
        b_size = len(features)

        # Top-k selection
        _, topk_indices = torch.topk(logits, k=k, dim=-1)  # [B, k]
        multi_hot = torch.zeros_like(logits).scatter_(1, topk_indices, 1.0)  # [B, num_experts]

        if mode == "hard":
            weights = multi_hot / float(k)
        elif mode == "straight_through":
            topk_soft = soft * multi_hot
            norm_topk_soft = topk_soft / topk_soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = (multi_hot / float(k)) + norm_topk_soft - norm_topk_soft.detach()
        else:  # soft
            weights = soft

        entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1)
        k_tensor = torch.full((b_size,), k, dtype=torch.long, device=features.device)

        return MultiExpertRoutingDecision(
            logits=logits,
            soft_probabilities=soft,
            selected_indices=topk_indices.tolist(),
            weights=weights,
            k_values=k_tensor,
            entropy=entropy,
        )


class AdaptiveKRouter(nn.Module):
    """Router supporting sample-level adaptive expert count selection (k in {1, 2, 3})."""

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
        self.rep_dim = input_dim * 2

        # Routing decision network: score each expert
        self.expert_network = nn.Sequential(
            nn.Linear(self.rep_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Adaptive k-prediction head: predict k in {1, 2, 3} (3 classes)
        self.k_head = nn.Sequential(
            nn.Linear(self.rep_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
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
        mode: str = "hard",
        fixed_k: int | None = None,
        strategy: str = "learned",
        entropy_thresholds: tuple[float, float] = (0.6, 1.1),
        margin_thresholds: tuple[float, float] = (0.4, 0.15),
    ) -> MultiExpertRoutingDecision:
        if mode not in {"straight_through", "hard", "soft"}:
            raise ValueError("mode must be 'straight_through', 'hard', or 'soft'")

        rep = self.extract_representation(features)
        expert_logits = self.expert_network(rep)
        soft = torch.softmax(expert_logits / self.temperature, dim=-1)
        entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1)
        b_size = len(features)

        # Decide k for each sample
        k_logits = self.k_head(rep)
        k_soft = torch.softmax(k_logits, dim=-1)

        if fixed_k is not None:
            chosen_k = torch.full((b_size,), fixed_k, dtype=torch.long, device=features.device)
        elif strategy == "learned":
            if mode == "hard":
                chosen_k = k_logits.argmax(dim=-1) + 1  # 0->1, 1->2, 2->3
            elif mode == "straight_through":
                hard_idx = k_logits.argmax(dim=-1)
                chosen_k = hard_idx + 1
            else:
                chosen_k = k_logits.argmax(dim=-1) + 1
        elif strategy == "entropy":
            t1, t2 = entropy_thresholds
            k_list = []
            for h in entropy:
                h_val = float(h.item())
                if h_val < t1:
                    k_list.append(1)
                elif h_val < t2:
                    k_list.append(2)
                else:
                    k_list.append(3)
            chosen_k = torch.tensor(k_list, dtype=torch.long, device=features.device)
        elif strategy == "margin":
            m1, m2 = margin_thresholds
            sorted_soft, _ = torch.sort(soft, descending=True, dim=-1)
            margins = sorted_soft[:, 0] - sorted_soft[:, 1]
            k_list = []
            for m in margins:
                m_val = float(m.item())
                if m_val > m1:
                    k_list.append(1)
                elif m_val > m2:
                    k_list.append(2)
                else:
                    k_list.append(3)
            chosen_k = torch.tensor(k_list, dtype=torch.long, device=features.device)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Dispatch selected indices per sample according to its specific k_i
        selected_indices: list[list[int]] = []
        multi_hot = torch.zeros_like(expert_logits)

        for i in range(b_size):
            ki = int(chosen_k[i].item())
            _, top_idx = torch.topk(expert_logits[i], k=ki)
            selected_indices.append(top_idx.tolist())
            multi_hot[i, top_idx] = 1.0

        if mode == "hard":
            weights = multi_hot / chosen_k.unsqueeze(-1).to(multi_hot.dtype)
        elif mode == "straight_through":
            topk_soft = soft * multi_hot
            norm_topk_soft = topk_soft / topk_soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            norm_hard = multi_hot / chosen_k.unsqueeze(-1).to(multi_hot.dtype)
            weights = norm_hard + norm_topk_soft - norm_topk_soft.detach()
        else:
            weights = soft

        return MultiExpertRoutingDecision(
            logits=expert_logits,
            soft_probabilities=soft,
            selected_indices=selected_indices,
            weights=weights,
            k_values=chosen_k,
            entropy=entropy,
            k_logits=k_logits,
            k_soft_probabilities=k_soft,
        )
