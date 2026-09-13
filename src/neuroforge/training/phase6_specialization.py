"""Phase 6 common-input-contract expert revalidation across 4 architectures and 3 task families."""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neuroforge.datasets import (
    Phase6ExpertSpecializationDataset,
    apply_phase6_marker_ablation,
    apply_phase6_random_marker,
    apply_phase6_structural_control,
    apply_phase6_token_permutation,
)
from neuroforge.evaluation.specialization import (
    parameter_count,
    select_capacity_phase6,
    specialization_metrics,
)
from neuroforge.models import StandaloneSpecialist


def _loader(family: str, split: str, samples: int, seed: int, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        Phase6ExpertSpecializationDataset(family, split, samples, seed),
        batch_size=batch,
        shuffle=shuffle,
    )


def _score(
    model: torch.nn.Module,
    loader: DataLoader,
    control_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for item in loader:
            x = control_fn(item["features"]) if control_fn else item["features"]
            logits = model(x)
            losses.append(float(F.cross_entropy(logits, item["target"])))
            correct += int((logits.argmax(1) == item["target"]).sum())
            total += len(x)
    return correct / total, statistics.mean(losses)


def _train(
    model: torch.nn.Module,
    train: DataLoader,
    valid: DataLoader,
    epochs: int,
) -> tuple[float, float, list[float]]:
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    best, best_state, history = -1.0, None, []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for item in train:
            optim.zero_grad()
            loss = F.cross_entropy(model(item["features"]), item["target"])
            loss.backward()
            optim.step()
        accuracy, _ = _score(model, valid)
        history.append(accuracy)
        if accuracy > best:
            best, best_state = accuracy, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, time.perf_counter() - started, history


def _instrument(model: StandaloneSpecialist, batch: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        sample = batch[:1]
        state = model.encoder(sample)
        costs = [float(2 * sample.shape[1] * 8 * state.shape[-1])]
        for block in model.blocks:
            if model.architecture == "attention_v2":
                state, cost = block(state, sample)
            else:
                state, cost = block(state)
            costs.append(float(cost))

    for _ in range(3):
        model(batch)
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(10):
        model(batch)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "parameters": parameter_count(model),
        "estimated_forward_flops": sum(costs),
        "activation_elements": float(batch.shape[1] * state.shape[-1] * (model.depth + 1)),
        "batch_latency_ms": 1000 * elapsed / 10,
        "samples_per_second": len(batch) * 10 / elapsed,
        "peak_python_memory_bytes": float(peak),
    }


def verify_marker_neutrality(
    seeds: tuple[int, ...] = (11, 23, 37),
    train_samples: int = 360,
    test_samples: int = 240,
) -> dict[str, Any]:
    """Test whether channel 4 query marker alone leaks label or task-family information."""
    label_leakage = {}
    for family in Phase6ExpertSpecializationDataset.families:
        per_seed = []
        for seed in seeds:
            train_ds = Phase6ExpertSpecializationDataset(family, "train", train_samples, seed)
            test_ds = Phase6ExpertSpecializationDataset(family, "test", test_samples, seed)
            clf = nn.Linear(12, 2)
            opt = torch.optim.Adam(clf.parameters(), lr=0.01)
            for _ in range(40):
                opt.zero_grad()
                loss = F.cross_entropy(clf(train_ds.features[:, :, 4]), train_ds.targets)
                loss.backward()
                opt.step()
            test_acc = float((clf(test_ds.features[:, :, 4]).argmax(1) == test_ds.targets).float().mean())
            per_seed.append(test_acc)
        label_leakage[family] = {
            "mean_accuracy": statistics.mean(per_seed),
            "std_accuracy": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
            "per_seed": per_seed,
            "neutral": abs(statistics.mean(per_seed) - 0.5) < max(0.08, 2.5 * (0.25 / test_samples) ** 0.5),
        }

    # Marker-only -> task-family classifier (3 classes)
    fam_per_seed = []
    for seed in seeds:
        train_feats, train_fams = [], []
        test_feats, test_fams = [], []
        for idx, family in enumerate(Phase6ExpertSpecializationDataset.families):
            tr = Phase6ExpertSpecializationDataset(family, "train", train_samples, seed)
            te = Phase6ExpertSpecializationDataset(family, "test", test_samples, seed)
            train_feats.append(tr.features[:, :, 4])
            train_fams.append(torch.full((len(tr),), idx, dtype=torch.long))
            test_feats.append(te.features[:, :, 4])
            test_fams.append(torch.full((len(te),), idx, dtype=torch.long))
        X_tr, y_tr = torch.cat(train_feats, dim=0), torch.cat(train_fams, dim=0)
        X_te, y_te = torch.cat(test_feats, dim=0), torch.cat(test_fams, dim=0)

        clf_fam = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 3))
        opt_fam = torch.optim.Adam(clf_fam.parameters(), lr=0.01)
        for _ in range(40):
            opt_fam.zero_grad()
            loss = F.cross_entropy(clf_fam(X_tr), y_tr)
            loss.backward()
            opt_fam.step()
        fam_acc = float((clf_fam(X_te).argmax(1) == y_te).float().mean())
        fam_per_seed.append(fam_acc)

    family_leakage = {
        "mean_accuracy": statistics.mean(fam_per_seed),
        "std_accuracy": statistics.stdev(fam_per_seed) if len(fam_per_seed) > 1 else 0.0,
        "per_seed": fam_per_seed,
        "neutral": abs(statistics.mean(fam_per_seed) - (1.0 / 3.0)) < max(0.08, 2.5 * ((2.0 / 9.0) / (3 * test_samples)) ** 0.5),
    }

    return {
        "marker_to_label": label_leakage,
        "marker_to_task_family": family_leakage,
    }


def run_phase6_specialization(
    output_dir: str | Path,
    seeds: tuple[int, ...] = (11, 23, 37),
    epochs: int = 25,
    train_samples: int = 360,
    validation_samples: int = 120,
    test_samples: int = 240,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute Phase 6 common-input-contract 4x3 expert revalidation experiment."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Step 1: Verify Marker Neutrality
    neutrality = verify_marker_neutrality(seeds, train_samples, test_samples)
    for fam, res in neutrality["marker_to_label"].items():
        if not res["neutral"]:
            raise RuntimeError(f"Marker leaks label information on family {fam}: acc={res['mean_accuracy']}")
    if not neutrality["marker_to_task_family"]["neutral"]:
        raise RuntimeError(f"Marker leaks task family: acc={neutrality['marker_to_task_family']['mean_accuracy']}")

    # Step 2: Capacity Selection
    selected = select_capacity_phase6()
    architectures = ("mlp", "graph", "attention", "attention_v2")
    families = Phase6ExpertSpecializationDataset.families

    cells: dict[str, dict[str, Any]] = {}
    history: dict[str, Any] = {}
    controls: dict[str, dict[str, Any]] = {}
    marker_ablations: dict[str, dict[str, Any]] = {}
    v2_interface_controls: dict[str, dict[str, Any]] = {}
    permutation_controls: dict[str, dict[str, Any]] = {}

    # Step 3: Run 4x3 Cross-Evaluation Matrix
    for family in families:
        for arch in architectures:
            key = f"{arch}__{family}"
            entries = []
            ctrl_entries = []
            ablation_entries = []
            perm_entries = []
            v2_rand_entries = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = StandaloneSpecialist(arch, depth=selected.depths[arch])
                train = _loader(family, "train", train_samples, seed, batch_size, True)
                valid = _loader(family, "validation", validation_samples, seed, batch_size, False)
                test = _loader(family, "test", test_samples, seed, batch_size, False)

                val_acc, duration, curve = _train(model, train, valid, epochs)
                test_acc, test_loss = _score(model, test)
                instrument_data = _instrument(model, next(iter(test))["features"])

                entry = {
                    "seed": seed,
                    "validation_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "test_loss": test_loss,
                    "training_seconds": duration,
                    **instrument_data,
                }
                entries.append(entry)
                history[f"{key}__{seed}"] = curve

                # Structural control (e.g. relational node permutation, contextual key negation)
                ctrl_acc, _ = _score(
                    model,
                    test,
                    control_fn=lambda x: apply_phase6_structural_control(x, family, seed + 90_000),
                )
                ctrl_entries.append({"seed": seed, "accuracy": ctrl_acc})

                # Marker ablation control: channel 4 replaced with neutral baseline
                ablated_acc, _ = _score(
                    model,
                    test,
                    control_fn=lambda x: apply_phase6_marker_ablation(x, seed + 80_000),
                )
                ablation_entries.append({"seed": seed, "accuracy": ablated_acc})

                # Token permutation invariance
                perm_acc, _ = _score(
                    model,
                    test,
                    control_fn=lambda x: apply_phase6_token_permutation(x, seed + 70_000),
                )
                perm_entries.append({"seed": seed, "accuracy": perm_acc})

                # Critical V2 interface control: random marker position (Section 15)
                if arch == "attention_v2":
                    rand_marker_acc, _ = _score(
                        model,
                        test,
                        control_fn=lambda x: apply_phase6_random_marker(x, seed + 60_000),
                    )
                    v2_rand_entries.append({"seed": seed, "accuracy": rand_marker_acc})

            mean_test_acc = statistics.mean(x["test_accuracy"] for x in entries)
            std_test_acc = statistics.stdev(x["test_accuracy"] for x in entries) if len(entries) > 1 else 0.0

            cells[key] = {
                "architecture": arch,
                "family": family,
                "per_seed": entries,
                "mean_test_accuracy": mean_test_acc,
                "std_test_accuracy": std_test_acc,
            }

            controls[key] = {
                "original_accuracy": mean_test_acc,
                "controlled_accuracy": statistics.mean(x["accuracy"] for x in ctrl_entries),
                "per_seed": ctrl_entries,
            }

            marker_ablations[key] = {
                "original_accuracy": mean_test_acc,
                "ablated_accuracy": statistics.mean(x["accuracy"] for x in ablation_entries),
                "per_seed": ablation_entries,
            }

            permutation_controls[key] = {
                "original_accuracy": mean_test_acc,
                "permuted_accuracy": statistics.mean(x["accuracy"] for x in perm_entries),
                "per_seed": perm_entries,
            }

            if arch == "attention_v2":
                v2_interface_controls[key] = {
                    "original_accuracy": mean_test_acc,
                    "random_marker_accuracy": statistics.mean(x["accuracy"] for x in v2_rand_entries),
                    "per_seed": v2_rand_entries,
                }

    matrix = {
        arch: [cells[f"{arch}__{fam}"]["mean_test_accuracy"] for fam in families]
        for arch in architectures
    }
    spec_metrics = specialization_metrics(matrix)

    # Step 4: Phase 3 historical comparison for models present in Phase 3
    phase3_historical = {
        "mlp": [1.0, 0.6083333333333333, 0.46805555555555556],
        "graph": [1.0, 0.7138888888888889, 0.46944444444444444],
        "attention": [0.75, 0.5180555555555556, 0.4847222222222222],
    }

    result: dict[str, Any] = {
        "experiment": "Phase 6 common-input-contract expert revalidation",
        "capacity": {
            "depths": selected.depths,
            "parameter_counts": selected.parameter_counts,
            "max_relative_gap": selected.max_relative_gap,
        },
        "marker_neutrality": neutrality,
        "cells": cells,
        "matrix": matrix,
        "specialization": spec_metrics,
        "controls": {
            "structural_controls": controls,
            "marker_ablation": marker_ablations,
            "permutation_invariance": permutation_controls,
            "v2_random_marker_interface_control": v2_interface_controls,
        },
        "historical_comparison": {
            "phase3_baseline": phase3_historical,
            "phase6_matrix": matrix,
            "interpretation": (
                "Phase 3 values reflect the historical contract without query marker. "
                "Phase 6 evaluates all experts under a common neutral-marker interface supporting V2."
            ),
        },
        "routing_gate": {
            "status": "closed",
            "reason": (
                "Learned routing remains closed until expert comparison, specialization, "
                "and oracle values are formally established under the common contract."
            ),
        },
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "Phase 6 common-input-contract expert revalidation",
        "seeds": seeds,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "git_commit": None,
        "shape": [12, 8],
        "common_input_contract": {
            "sequence_length": 12,
            "input_dim": 8,
            "marker_channel": 4,
            "marker_value": 1.0,
            "marker_neutrality": "Verified at chance on label and task family",
        },
        "training": {
            "optimizer": "AdamW",
            "lr": 0.003,
            "weight_decay": 1e-4,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping": "best validation accuracy",
        },
        "capacity": result["capacity"],
    }
    result["manifest"] = manifest

    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # CSV source data
    with (destination / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        first_cell = next(iter(cells.values()))
        fieldnames = ["cell", "architecture", "family", *first_cell["per_seed"][0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k, cell in cells.items():
            for row in cell["per_seed"]:
                writer.writerow({"cell": k, "architecture": cell["architecture"], "family": cell["family"], **row})

    return result
