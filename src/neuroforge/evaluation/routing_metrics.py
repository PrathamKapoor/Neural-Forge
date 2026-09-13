"""Route-label metrics are distinct from downstream prediction metrics."""

from __future__ import annotations

import torch


def routing_metrics(predicted: torch.Tensor, oracle: torch.Tensor, names: tuple[str, ...] = ("mlp", "graph", "attention")) -> dict[str, object]:
    if predicted.shape != oracle.shape or predicted.ndim != 1:
        raise ValueError("predicted and oracle routes must be equally shaped vectors")
    confusion = torch.zeros(len(names), len(names), dtype=torch.long)
    for actual, chosen in zip(oracle.tolist(), predicted.tolist()): confusion[actual, chosen] += 1
    per_route = {name: float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else float("nan") for index, name in enumerate(names)}
    return {"route_accuracy": float((predicted == oracle).float().mean()), "per_route_accuracy": per_route,
            "confusion_matrix": confusion.tolist(), "utilization": {name: float((predicted == index).float().mean()) for index, name in enumerate(names)}}
