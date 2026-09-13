"""Figures answering the controlled-routing research questions."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_routing_diagnosis_figures(result: dict[str, Any], directory: str | Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    destination = Path(directory); destination.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    names = ["random", "static_mlp", "oracle", "learned_natural_supervised_hard", "learned_explicit_supervised_hard", "learned_natural_indirect_hard"]
    labels = [name.replace("learned_", "").replace("_", "\n") for name in names]
    values = [aggregate[name]["route_accuracy_mean"] for name in names]
    errors = [aggregate[name]["route_accuracy_std"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(labels, values, yerr=errors, color="#4f7cff"); ax.set(ylim=(0, 1.05), ylabel="oracle route accuracy", title="Routing learnability across seeds"); fig.tight_layout(); fig.savefig(destination / "routing_accuracy.png", dpi=160); plt.close(fig)
    first_seed = next(iter(result["per_seed"].values()))
    matrix = np.array(first_seed["learned_natural_supervised_hard"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax); ax.set(xticks=range(3), yticks=range(3), xticklabels=["MLP", "GNN", "Attention"], yticklabels=["MLP", "GNN", "Attention"], xlabel="learned route", ylabel="oracle route", title="Natural supervised confusion (seed 1)")
    for row in range(3):
        for column in range(3): ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(destination / "routing_confusion.png", dpi=160); plt.close(fig)
    pareto = ["static_mlp", "oracle", "learned_natural_indirect_hard", "learned_natural_supervised_hard"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in pareto: ax.scatter(aggregate[name]["estimated_compute_mean"], aggregate[name]["prediction_accuracy_mean"], label=name.replace("learned_", ""));
    ax.set(xlabel="estimated compute (proxy units)", ylabel="prediction accuracy", title="Prediction vs proxy compute"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(destination / "pareto_compute.png", dpi=160); plt.close(fig)
