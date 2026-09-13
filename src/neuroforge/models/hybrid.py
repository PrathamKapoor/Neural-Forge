"""Fixed and adaptive hybrid classifiers sharing the same computational blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neuroforge.blocks import AttentionBlock, GraphBlock, MLPBlock
from neuroforge.routing import SoftRouter


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    routing_weights: torch.Tensor
    expected_compute: torch.Tensor
    block_costs: torch.Tensor


class _HybridBase(nn.Module):
    module_names = ("mlp", "graph", "attention")

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList((MLPBlock(hidden_dim), GraphBlock(hidden_dim), AttentionBlock(hidden_dim)))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def _block_outputs(self, state: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        outputs, costs = [], []
        for block in self.blocks:
            output, cost = block(state)
            outputs.append(output)
            costs.append(cost)
        return outputs, state.new_tensor(costs)

    def _classify(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(state.mean(dim=1))


class AdaptiveHybridClassifier(_HybridBase):
    """Learns a module utility distribution; hard mode is an inference approximation.

    Soft training evaluates every block, so it is optimization-friendly but does not
    by itself deliver physical branch skipping. Hard mode uses a one-hot state mix;
    evaluation reports its latency separately to avoid confusing the two regimes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__(input_dim, hidden_dim, num_classes)
        self.router = SoftRouter(hidden_dim, self.module_names, temperature)

    def forward(self, features: torch.Tensor, routing_mode: str = "soft") -> ModelOutput:
        if routing_mode not in {"soft", "hard"}:
            raise ValueError("routing_mode must be 'soft' or 'hard'")
        state = self.encoder(features)
        decision = self.router(state)
        if routing_mode == "soft":
            weights = decision.weights
            outputs, costs = self._block_outputs(state)
            mixed = (torch.stack(outputs, dim=1) * weights[:, :, None, None]).sum(dim=1)
        else:
            selected = decision.weights.argmax(dim=-1)
            weights = torch.nn.functional.one_hot(selected, 3).to(state.dtype)
            costs = state.new_tensor([block(state[:1])[1] for block in self.blocks])
            mixed = torch.empty_like(state)
            # True conditional dispatch: only selected samples enter each block.
            for index, block in enumerate(self.blocks):
                mask = selected == index
                if mask.any():
                    mixed[mask], _ = block(state[mask])
        return ModelOutput(self._classify(mixed), weights, weights @ costs, costs)


class FixedHybridClassifier(_HybridBase):
    """Comparable fixed baseline using an explicit constant module policy."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, active_modules: tuple[str, ...] = ("mlp", "graph", "attention")) -> None:
        super().__init__(input_dim, hidden_dim, num_classes)
        if not active_modules or any(name not in self.module_names for name in active_modules):
            raise ValueError("active_modules must be a non-empty subset of supported module names")
        self.active_modules = active_modules

    def forward(self, features: torch.Tensor, routing_mode: str = "soft") -> ModelOutput:
        state = self.encoder(features)
        raw = state.new_tensor([float(name in self.active_modules) for name in self.module_names])
        weights = (raw / raw.sum()).expand(features.shape[0], -1)
        active_outputs, costs = [], []
        for index, block in enumerate(self.blocks):
            if raw[index]:
                output, cost = block(state)
                active_outputs.append(output)
                costs.append(cost)
            else:
                costs.append(block(state[:1])[1])
        mixed = torch.stack(active_outputs, dim=1).mean(dim=1)
        cost_tensor = state.new_tensor(costs)
        return ModelOutput(self._classify(mixed), weights, weights @ cost_tensor, cost_tensor)
