"""Phase 3 standalone-expert experiment; no learned router is involved."""
from __future__ import annotations
import csv, json, platform, statistics, sys, time, tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from neuroforge.datasets import ExpertSpecializationDataset, apply_structural_control
from neuroforge.evaluation.specialization import parameter_count, select_capacity, specialization_metrics
from neuroforge.models import StandaloneSpecialist

def _loader(family: str, split: str, samples: int, seed: int, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ExpertSpecializationDataset(family, split, samples, seed), batch_size=batch, shuffle=shuffle)

def _score(model: torch.nn.Module, loader: DataLoader, controlled: str | None = None, seed: int = 0) -> tuple[float, float]:
    model.eval(); correct = total = 0; losses = []
    with torch.no_grad():
        for item in loader:
            x = apply_structural_control(item["features"], controlled, seed) if controlled else item["features"]
            logits = model(x); losses.append(float(F.cross_entropy(logits, item["target"])))
            correct += int((logits.argmax(1) == item["target"]).sum()); total += len(x)
    return correct / total, statistics.mean(losses)

def _train(model: torch.nn.Module, train: DataLoader, valid: DataLoader, epochs: int) -> tuple[float, float, list[float]]:
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    best, best_state, history = -1.0, None, []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for item in train:
            optim.zero_grad(); loss = F.cross_entropy(model(item["features"]), item["target"]); loss.backward(); optim.step()
        accuracy, _ = _score(model, valid); history.append(accuracy)
        if accuracy > best: best, best_state = accuracy, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return best, time.perf_counter() - started, history

def _instrument(model: StandaloneSpecialist, batch: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        state = model.encoder(batch[:1]); costs = [float(2 * batch.shape[1] * 8 * state.shape[-1])]
        for block in model.blocks: state, cost = block(state); costs.append(float(cost))
    for _ in range(3): model(batch)
    tracemalloc.start(); started = time.perf_counter()
    for _ in range(10): model(batch)
    elapsed = time.perf_counter() - started; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {"parameters": parameter_count(model), "estimated_forward_flops": sum(costs), "activation_elements": float(batch.shape[1] * state.shape[-1] * (model.depth + 1)), "batch_latency_ms": 1000 * elapsed / 10, "samples_per_second": len(batch) * 10 / elapsed, "peak_python_memory_bytes": float(peak)}

def _linear_shortcut(family: str, seed: int, train_samples: int, test_samples: int) -> float:
    train = ExpertSpecializationDataset(family, "train", train_samples, seed); test = ExpertSpecializationDataset(family, "test", test_samples, seed)
    model = torch.nn.Linear(8, 2); optim = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(20): optim.zero_grad(); loss = F.cross_entropy(model(train.features.mean(1)), train.targets); loss.backward(); optim.step()
    return float((model(test.features.mean(1)).argmax(1) == test.targets).float().mean())

def run_expert_specialization(output_dir: str | Path, seeds: tuple[int, ...] = (11,23,37), epochs: int = 25, train_samples: int = 360, validation_samples: int = 120, test_samples: int = 240, batch_size: int = 60) -> dict[str, Any]:
    """Run the full 3x3 comparison. Capacity selection is independent of scores."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    selected = select_capacity(); cells: dict[str, dict[str, Any]] = {}; controls: dict[str, dict[str, Any]] = {}; history: dict[str, Any] = {}; shortcuts = {}
    for family in ExpertSpecializationDataset.families:
        shortcuts[family] = []
        for architecture in ("mlp", "graph", "attention"):
            key = f"{architecture}__{family}"; entries = []; controls[key] = []
            for seed in seeds:
                torch.manual_seed(seed)
                model = StandaloneSpecialist(architecture, depth=selected.depths[architecture])
                train = _loader(family, "train", train_samples, seed, batch_size, True); valid = _loader(family, "validation", validation_samples, seed, batch_size, False); test = _loader(family, "test", test_samples, seed, batch_size, False)
                val, duration, curve = _train(model, train, valid, epochs); test_acc, test_loss = _score(model, test)
                entry = {"seed": seed, "validation_accuracy": val, "test_accuracy": test_acc, "test_loss": test_loss, "training_seconds": duration, **_instrument(model, next(iter(test))["features"])}
                entries.append(entry); history[f"{key}__{seed}"] = curve
                if family in {"relational", "contextual"}: controls[key].append({"seed": seed, "accuracy": _score(model, test, family, seed + 90_000)[0]})
            cells[key] = {"architecture": architecture, "family": family, "per_seed": entries, "mean_test_accuracy": statistics.mean(x["test_accuracy"] for x in entries), "std_test_accuracy": statistics.stdev(x["test_accuracy"] for x in entries) if len(entries)>1 else 0.0}
        for seed in seeds: shortcuts[family].append(_linear_shortcut(family, seed, train_samples, test_samples))
    matrix = {architecture: [cells[f"{architecture}__{family}"]["mean_test_accuracy"] for family in ExpertSpecializationDataset.families] for architecture in ("mlp","graph","attention")}
    metrics = specialization_metrics(matrix)
    for key, values in controls.items():
        if values: controls[key] = {"original_accuracy": cells[key]["mean_test_accuracy"], "controlled_accuracy": statistics.mean(x["accuracy"] for x in values), "per_seed": values}
    result: dict[str, Any] = {"capacity": {"depths": selected.depths, "parameter_counts": selected.parameter_counts, "max_relative_gap": selected.max_relative_gap}, "cells": cells, "matrix": matrix, "specialization": metrics, "controls": {"relational_node_feature_permutation": {k:v for k,v in controls.items() if "relational" in k}, "contextual_query_key_negation": {k:v for k,v in controls.items() if "contextual" in k}}, "shortcut_linear_pooled": {k:{"mean_accuracy":statistics.mean(v),"std_accuracy":statistics.stdev(v) if len(v)>1 else 0.0,"per_seed":v} for k,v in shortcuts.items()}, "oracle_selection": {"executed": False, "reason": "conditional gate evaluated in report after control evidence"}}
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds, "python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device":"cpu", "git_commit": None, "shape":[12,8], "control":"node-feature permutation relative to fixed ring topology; GraphBlock external edge randomization unavailable without semantic change", "training":{"optimizer":"AdamW","lr":0.003,"epochs":epochs,"batch_size":batch_size,"early_stopping":"best validation accuracy"}, "capacity":result["capacity"]}
    result["manifest"] = manifest
    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8"); (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (destination / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell","architecture","family",*next(iter(cells.values()))["per_seed"][0].keys()]); writer.writeheader()
        for key, cell in cells.items():
            for row in cell["per_seed"]: writer.writerow({"cell":key,"architecture":cell["architecture"],"family":cell["family"],**row})
    return result
