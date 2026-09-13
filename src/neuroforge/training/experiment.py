"""Seeded training and artifact creation for reproducible local experiments."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neuroforge.config import NeuroForgeConfig
from neuroforge.datasets import MixedStructureDataset
from neuroforge.evaluation import benchmark_latency, evaluate_classifier
from neuroforge.models import AdaptiveHybridClassifier, FixedHybridClassifier


def _loader(samples: int, config: NeuroForgeConfig, seed_offset: int, shuffle: bool) -> DataLoader:
    data = config.data
    return DataLoader(MixedStructureDataset(samples, data.sequence_length, data.input_dim, config.seed + seed_offset), batch_size=config.training.batch_size, shuffle=shuffle)


def _train(model: torch.nn.Module, loader: DataLoader, config: NeuroForgeConfig) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    history = []
    model.train()
    for epoch in range(config.training.epochs):
        total_loss = total_correct = total = 0.0
        for features, targets, _ in loader:
            optimizer.zero_grad()
            output = model(features, routing_mode="soft")
            loss = F.cross_entropy(output.logits, targets) + config.router.compute_penalty * output.expected_compute.mean() / 1000
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(targets); total_correct += float((output.logits.argmax(-1) == targets).sum()); total += len(targets)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / total, "train_accuracy": total_correct / total})
    return history


def run_experiment(config: NeuroForgeConfig, output_dir: str | Path, include_baselines: bool = True) -> dict[str, Any]:
    """Run a small reproducible comparison and persist summary, history, and manifest."""
    torch.manual_seed(config.seed)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(config.data.train_samples, config, 0, True)
    test_loader = _loader(config.data.test_samples, config, 2, False)
    adaptive = AdaptiveHybridClassifier(config.data.input_dim, config.model.hidden_dim, config.model.num_classes, config.router.temperature)
    history = _train(adaptive, train_loader, config)
    summary: dict[str, Any] = {"adaptive_soft": evaluate_classifier(adaptive, test_loader, "soft"), "adaptive_hard": evaluate_classifier(adaptive, test_loader, "hard")}
    example_batch = next(iter(test_loader))[0]
    summary["adaptive_soft"]["latency"] = benchmark_latency(adaptive, example_batch, "soft", config.evaluation.latency_warmup, config.evaluation.latency_iterations)
    summary["adaptive_hard"]["latency"] = benchmark_latency(adaptive, example_batch, "hard", config.evaluation.latency_warmup, config.evaluation.latency_iterations)
    if include_baselines:
        for name, modules in {"mlp": ("mlp",), "graph": ("graph",), "attention": ("attention",), "fixed_hybrid": ("mlp", "graph", "attention")}.items():
            baseline = FixedHybridClassifier(config.data.input_dim, config.model.hidden_dim, config.model.num_classes, modules)
            _train(baseline, train_loader, config)
            summary[name] = evaluate_classifier(baseline, test_loader)
            summary[name]["latency"] = benchmark_latency(baseline, example_batch, "soft", config.evaluation.latency_warmup, config.evaluation.latency_iterations)
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": config.seed, "python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "cuda_available": torch.cuda.is_available(), "config": config.__dict__}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda value: value.__dict__), encoding="utf-8")
    return summary
