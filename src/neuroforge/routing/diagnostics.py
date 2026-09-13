"""Automated checks that stop routing metrics from hiding common failure modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingDiagnostics:
    utilization: dict[str, float]
    mean_entropy: float
    single_module_collapse: bool
    all_module_like: bool
    low_diversity: bool


def routing_diagnostics(weights: torch.Tensor, module_names: tuple[str, ...] = ("mlp", "graph", "attention")) -> RoutingDiagnostics:
    if weights.ndim != 2 or weights.shape[1] != len(module_names):
        raise ValueError("weights must be [samples, modules] matching module_names")
    mean_weights = weights.mean(dim=0)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
    uniform_entropy = float(torch.log(torch.tensor(float(len(module_names)))).item())
    return RoutingDiagnostics(
        utilization={name: float(value) for name, value in zip(module_names, mean_weights)},
        mean_entropy=entropy,
        single_module_collapse=bool(mean_weights.max() > 0.98),
        all_module_like=bool(entropy > 0.95 * uniform_entropy),
        low_diversity=bool(weights.argmax(dim=-1).unique().numel() == 1),
    )
