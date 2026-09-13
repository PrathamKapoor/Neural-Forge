"""Figure generation for Phase 7 Oracle Routing and Fixed-Best Selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_phase7_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the 6 required figures for Phase 7 report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: list[str] = []

    conditions = summary["conditions"]
    palette = {
        "Fixed MLP": "#4e79a7",
        "Fixed Graph": "#59a14f",
        "Fixed Attention V1": "#e15759",
        "Fixed Attention V2": "#b07aa1",
        "Random Router": "#edc948",
        "Oracle Router": "#76b7b2",
        "Oracle Without V2": "#f28e2b",
    }

    # 1. Fixed-expert accuracy by task family
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    families = ("feature", "relational", "contextual")
    x = np.arange(len(families))
    fixed_names = ("Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2")
    width = 0.18

    for idx, name in enumerate(fixed_names):
        accs = [conditions[name]["family_accuracies"][f]["mean"] * 100 for f in families]
        stds = [conditions[name]["family_accuracies"][f]["std"] * 100 for f in families]
        ax.bar(x + idx * width - 1.5 * width, accs, width, yerr=stds, label=name, color=palette[name], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Feature", "Relational", "Contextual"], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Fixed Expert Accuracy by Task Family", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p1 = destination / "fixed_expert_accuracy_by_family.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # 2. Oracle vs fixed accuracy
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    all_conds = list(conditions.keys())
    overall_accs = [conditions[c]["overall_accuracy"]["mean"] * 100 for c in all_conds]
    overall_stds = [conditions[c]["overall_accuracy"]["std"] * 100 for c in all_conds]
    bar_colors = [palette.get(c, "#333333") for c in all_conds]

    bars = ax.bar(range(len(all_conds)), overall_accs, yerr=overall_stds, color=bar_colors, capsize=4, width=0.6)
    ax.set_xticks(range(len(all_conds)))
    ax.set_xticklabels(all_conds, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Overall Mixed Population Accuracy Across Conditions", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p2 = destination / "oracle_vs_fixed_accuracy.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # 3. Accuracy vs estimated FLOPs (Pareto Analysis)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    for name, data in conditions.items():
        flops = data["compute"]["estimated_forward_flops"]
        acc = data["overall_accuracy"]["mean"] * 100
        ax.scatter(flops, acc, s=120, color=palette.get(name, "#333333"), zorder=5, label=name)
        ax.annotate(name, (flops, acc), xytext=(6, 4), textcoords="offset points", fontsize=8.5)

    # Draw Pareto boundary connecting MLP -> Oracle
    # MLP: 8928 FLOPs, ~66.5% -> Oracle: 21112 FLOPs, ~90.3%
    mlp_f = conditions["Fixed MLP"]["compute"]["estimated_forward_flops"]
    mlp_a = conditions["Fixed MLP"]["overall_accuracy"]["mean"] * 100
    ora_f = conditions["Oracle Router"]["compute"]["estimated_forward_flops"]
    ora_a = conditions["Oracle Router"]["overall_accuracy"]["mean"] * 100
    ax.plot([mlp_f, ora_f], [mlp_a, ora_a], "k--", alpha=0.6, label="Pareto Frontier (Dominant)")

    ax.set_xlabel("Estimated Forward FLOPs per Sample", fontsize=11)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Estimated Compute (Pareto Analysis)", fontsize=12, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p3 = destination / "accuracy_vs_flops.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # 4. Accuracy vs measured latency
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    for name, data in conditions.items():
        lat = data["latency"]["batch_latency_ms"]
        acc = data["overall_accuracy"]["mean"] * 100
        ax.scatter(lat, acc, s=120, color=palette.get(name, "#333333"), zorder=5, label=name)
        ax.annotate(name, (lat, acc), xytext=(6, 4), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Batch Latency (ms)", fontsize=11)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Measured Batch Latency", fontsize=12, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p4 = destination / "accuracy_vs_latency.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # 5. Oracle routing selection matrix
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    matrix_data = summary["oracle_routing"]["selection_matrix_proportions"]
    mat = np.array([
        [matrix_data[fam][exp] * 100 for exp in ("mlp", "graph", "attention", "attention_v2")]
        for fam in ("feature", "relational", "contextual")
    ])

    cax = ax.matshow(mat, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(cax, ax=ax, label="Selection Proportion (%)")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["MLP", "Graph", "Attention V1", "Attention V2"], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Feature", "Relational", "Contextual"], fontsize=10)
    ax.set_xlabel("Selected Expert", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task Family", fontsize=11, fontweight="bold")
    ax.set_title("Oracle Routing Selection Matrix", fontsize=12, fontweight="bold", pad=20)

    for i in range(3):
        for j in range(4):
            val = mat[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    fig.tight_layout()
    p5 = destination / "oracle_routing_selection_matrix.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # 6. Compute distribution across oracle-selected experts
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    categories = ["MLP (Feature)", "Graph (Relational)", "V2 (Contextual)", "Oracle Average", "Fixed V2 (Worst Case)"]
    costs = [8928, 8112, 46296, 21112, 46296]
    colors = ["#4e79a7", "#59a14f", "#b07aa1", "#76b7b2", "#e15759"]

    b = ax.bar(categories, costs, color=colors, width=0.55)
    ax.set_ylabel("Estimated Forward FLOPs / Sample", fontsize=11)
    ax.set_title("Compute Distribution Across Oracle-Selected Experts vs Fixed V2", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for rect in b:
        h = rect.get_height()
        ax.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p6 = destination / "compute_distribution_oracle.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_paths.append(str(p6))

    return generated_paths
