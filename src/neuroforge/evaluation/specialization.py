"""Capacity selection and transparent Phase 3 specialization statistics."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import statistics
import torch
from neuroforge.models.specialists import StandaloneSpecialist

@dataclass(frozen=True)
class CapacitySelection:
    depths: dict[str, int]
    parameter_counts: dict[str, int]
    max_relative_gap: float

def parameter_count(model: torch.nn.Module) -> int: return sum(p.numel() for p in model.parameters() if p.requires_grad)

def select_capacity(input_dim: int = 8, hidden_dim: int = 24, max_depth: int = 4) -> CapacitySelection:
    names = ("mlp", "graph", "attention")
    candidates = []
    for depths in product(range(1, max_depth + 1), repeat=3):
        counts = {name: parameter_count(StandaloneSpecialist(name, input_dim, hidden_dim, depth)) for name, depth in zip(names, depths)}
        gap = (max(counts.values()) - min(counts.values())) / max(counts.values())
        candidates.append((gap, sum(depths), depths, counts))
    gap, _, depths, counts = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return CapacitySelection(dict(zip(names, depths)), counts, gap)

def select_capacity_phase6(input_dim: int = 8, hidden_dim: int = 24, max_depth: int = 4) -> CapacitySelection:
    names = ("mlp", "graph", "attention", "attention_v2")
    candidates = []
    for depths in product(range(1, max_depth + 1), repeat=4):
        counts = {name: parameter_count(StandaloneSpecialist(name, input_dim, hidden_dim, depth)) for name, depth in zip(names, depths)}
        gap = (max(counts.values()) - min(counts.values())) / max(counts.values())
        candidates.append((gap, sum(depths), depths, counts))
    gap, _, depths, counts = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return CapacitySelection(dict(zip(names, depths)), counts, gap)

def specialization_metrics(per_model_by_family: dict[str, list[float]]) -> dict[str, object]:
    names = list(per_model_by_family)
    families = ("feature", "relational", "contextual")
    winners, margins = [], {}
    for index, family in enumerate(families):
        ordered = sorted(((per_model_by_family[name][index], name) for name in names), reverse=True)
        winners.append(ordered[0][1]); margins[family] = round(ordered[0][0] - ordered[1][0], 12)
    return {"winners": winners, "margins": margins, "means": {name: statistics.mean(values) for name, values in per_model_by_family.items()}}
