"""Metrics distinguish estimated cost from observed wall-clock behavior."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuroforge.routing import routing_diagnostics


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, routing_mode: str = "soft") -> dict[str, Any]:
    model.eval()
    logits, targets, routes, oracle, costs = [], [], [], [], []
    for features, labels, oracle_routes in loader:
        output = model(features, routing_mode=routing_mode)
        logits.append(output.logits.cpu()); targets.append(labels.cpu()); routes.append(output.routing_weights.cpu())
        oracle.append(oracle_routes.cpu()); costs.append(output.expected_compute.cpu())
    all_logits, all_targets = torch.cat(logits), torch.cat(targets)
    all_routes, all_oracle = torch.cat(routes), torch.cat(oracle)
    diagnostics = routing_diagnostics(all_routes)
    predicted_routes = all_routes.argmax(dim=-1)
    return {
        "accuracy": float((all_logits.argmax(dim=-1) == all_targets).float().mean()),
        "estimated_compute": float(torch.cat(costs).mean()),
        "average_active_modules": float((all_routes > 1e-6).sum(dim=-1).float().mean()),
        "oracle_route_agreement": float((predicted_routes == all_oracle).float().mean()),
        "routing": {"utilization": diagnostics.utilization, "entropy": diagnostics.mean_entropy,
                    "single_module_collapse": diagnostics.single_module_collapse,
                    "all_module_like": diagnostics.all_module_like, "low_diversity": diagnostics.low_diversity},
    }


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, batch: torch.Tensor, routing_mode: str, warmup: int, iterations: int) -> dict[str, float]:
    model.eval()
    for _ in range(warmup): model(batch, routing_mode=routing_mode)
    started = time.perf_counter()
    for _ in range(iterations): model(batch, routing_mode=routing_mode)
    elapsed = time.perf_counter() - started
    return {"batch_latency_ms": 1000 * elapsed / iterations, "samples_per_second": batch.shape[0] * iterations / elapsed}
