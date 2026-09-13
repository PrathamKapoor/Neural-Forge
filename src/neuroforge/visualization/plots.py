"""Small, honest figures generated solely from persisted experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_summary_figure(summary: dict[str, Any], path: str | Path) -> None:
    """Save accuracy-vs-estimated-compute and utilization panels.

    This visualization does not imply a Pareto frontier; it displays all measured
    points so dominated configurations remain visible.
    """
    import matplotlib.pyplot as plt

    labels, accuracy, compute = list(summary), [], []
    for label in labels:
        accuracy.append(summary[label]["accuracy"]); compute.append(summary[label]["estimated_compute"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(compute, accuracy, s=70, color="#356ae6")
    for label, x, y in zip(labels, compute, accuracy): axes[0].annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="estimated computation (proxy units)", ylabel="test accuracy", title="Measured configurations")
    adaptive = summary["adaptive_hard"]["routing"]["utilization"]
    axes[1].bar(adaptive.keys(), adaptive.values(), color=["#5b8ff9", "#61dDAa", "#f6bd16"])
    axes[1].set(ylim=(0, 1), ylabel="mean hard-route utilization", title="Adaptive hard routing")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)
