"""Multi-seed controlled routing diagnosis; deliberately separate from Phase 1."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from neuroforge.datasets import ControlledRoutingDataset
from neuroforge.evaluation.routing_metrics import routing_metrics
from neuroforge.routing.probes import DescriptorRouter

_COSTS = torch.tensor([1.0, 2.0, 3.0])


def _expert_logits(features: torch.Tensor) -> torch.Tensor:
    """Transparent family-specialist prediction instruments used only in diagnosis.

    Scores correspond exactly to the generated label relations. They establish
    whether conditional selection, not expert capacity, is the limiting factor.
    """
    scores = torch.stack((features[:, :, 1].mean(1), (features[:, :-1, 1] * features[:, 1:, 1]).mean(1), features[:, 0, 2] - features[:, -1, 2]), dim=1)
    return torch.stack((-scores, scores), dim=-1) * 6.0


def _evaluate(weights: torch.Tensor, data: ControlledRoutingDataset, router_ms: float = 0.0) -> dict[str, Any]:
    selected = weights.argmax(1)
    logits = (_expert_logits(data.features) * weights[:, :, None]).sum(1)
    result = routing_metrics(selected.cpu(), data.oracle_routes.cpu())
    result.update({"prediction_accuracy": float((logits.argmax(1) == data.targets).float().mean()),
                   "estimated_compute": float((weights @ _COSTS).mean()), "average_active_modules": float((weights > 1e-6).sum(1).float().mean()),
                   "entropy": float((-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)).mean()), "router_batch_ms": router_ms})
    return result


def _fit(descriptor: torch.Tensor, routes: torch.Tensor, features: torch.Tensor, targets: torch.Tensor, *, hidden_dim: int, temperature: float, epochs: int, supervision: str, compute_penalty: float, seed: int) -> DescriptorRouter:
    torch.manual_seed(seed)
    router = DescriptorRouter(descriptor.shape[1], hidden_dim, temperature)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.03)
    expert_logits = _expert_logits(features)
    for _ in range(epochs):
        optimizer.zero_grad()
        decision = router(descriptor, "soft" if supervision == "oracle" else "straight_through")
        if supervision == "oracle": loss = F.cross_entropy(decision.logits, routes)
        else: loss = F.cross_entropy((expert_logits * decision.weights[:, :, None]).sum(1), targets)
        loss = loss + compute_penalty * (decision.weights @ _COSTS).mean()
        loss.backward(); optimizer.step()
    return router


def _router_weights(router: DescriptorRouter, descriptor: torch.Tensor, mode: str) -> tuple[torch.Tensor, float]:
    for _ in range(3): router(descriptor, mode)
    started = time.perf_counter(); decision = router(descriptor, mode); elapsed = (time.perf_counter() - started) * 1000
    return decision.weights.detach(), elapsed


def _aggregate(per_seed: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    names = next(iter(per_seed.values())).keys()
    for name in names:
        output[name] = {}
        for metric in ("route_accuracy", "prediction_accuracy", "estimated_compute", "router_batch_ms"):
            values = [entry[name][metric] for entry in per_seed.values()]
            output[name][f"{metric}_mean"] = statistics.mean(values)
            output[name][f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return output


def run_routing_diagnosis(output_dir: str | Path, seeds: tuple[int, ...] = (11, 23, 37), epochs: int = 40, samples: int = 360, batch_size: int = 120) -> dict[str, Any]:
    """Run random/static/oracle and learned conditions on unseen generator samples."""
    del batch_size  # whole-tensor training avoids DataLoader order confounds in this small diagnostic.
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    per_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        train = ControlledRoutingDataset(samples, seed=seed)
        test = ControlledRoutingDataset(samples, seed=seed + 10_000)
        condition: dict[str, dict[str, Any]] = {}
        random_weights = F.one_hot(torch.randint(0, 3, (len(test),), generator=torch.Generator().manual_seed(seed)), 3).float()
        condition["random"] = _evaluate(random_weights, test)
        condition["static_mlp"] = _evaluate(F.one_hot(torch.zeros(len(test), dtype=torch.long), 3).float(), test)
        condition["oracle"] = _evaluate(test.explicit_metadata, test)
        specifications = {
            "learned_natural_supervised_hard": (train.natural_descriptors, test.natural_descriptors, "oracle", "hard", 12, 1.0, 0.0),
            "learned_explicit_supervised_hard": (train.explicit_metadata, test.explicit_metadata, "oracle", "hard", 12, 1.0, 0.0),
            "learned_natural_indirect_soft": (train.natural_descriptors, test.natural_descriptors, "indirect", "soft", 12, 1.0, 0.0),
            "learned_natural_indirect_hard": (train.natural_descriptors, test.natural_descriptors, "indirect", "hard", 12, 1.0, 0.0),
            "small_router_supervised": (train.natural_descriptors, test.natural_descriptors, "oracle", "hard", 4, 1.0, 0.0),
            "temperature_half_supervised": (train.natural_descriptors, test.natural_descriptors, "oracle", "hard", 12, 0.5, 0.0),
            "compute_aware_indirect_hard": (train.natural_descriptors, test.natural_descriptors, "indirect", "hard", 12, 1.0, 0.02),
        }
        for name, (train_descriptor, test_descriptor, supervision, mode, hidden, temperature, penalty) in specifications.items():
            router = _fit(train_descriptor, train.oracle_routes, train.features, train.targets, hidden_dim=hidden, temperature=temperature, epochs=epochs, supervision=supervision, compute_penalty=penalty, seed=seed)
            weights, latency = _router_weights(router, test_descriptor, mode)
            condition[name] = _evaluate(weights, test, latency)
        per_seed[str(seed)] = condition
    result: dict[str, Any] = {"per_seed": per_seed, "aggregate": _aggregate(per_seed),
        "manifest": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds, "samples_train": samples, "samples_test": samples,
                     "epochs": epochs, "python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu",
                     "oracle": "generator family 0/1/2", "natural_descriptor": "sequence mean,std,adjacent product,endpoint product", "gradient_estimators": {"supervised": "cross-entropy logits", "indirect": "straight-through hard forward / soft backward"}}}
    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(result["manifest"], indent=2), encoding="utf-8")
    return result
