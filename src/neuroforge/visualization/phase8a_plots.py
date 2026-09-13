"""Figure generation for Phase 8A Minimal Learned Sample-Level Router."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_phase8a_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the 8 required research figures for Phase 8A."""
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
        "Learned Router": "#2b5c8f",
        "Oracle Router": "#76b7b2",
    }

    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    cond_order = ["Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2", "Random Router", "Learned Router", "Oracle Router"]
    accs = [conditions[c]["overall_accuracy"]["mean"] * 100 for c in cond_order]
    stds = [conditions[c]["overall_accuracy"]["std"] * 100 for c in cond_order]
    colors = [palette[c] for c in cond_order]

    bars = ax.bar(range(len(cond_order)), accs, yerr=stds, color=colors, capsize=4, width=0.6)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Overall Accuracy Across Architectures and Routing Conditions", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p1 = destination / "accuracy_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # Figure 2: Family-specific accuracy comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    families = ("feature", "relational", "contextual")
    x = np.arange(len(families))
    width = 0.12
    display_conds = ["Fixed MLP", "Fixed Graph", "Fixed Attention V2", "Random Router", "Learned Router", "Oracle Router"]

    for idx, c in enumerate(display_conds):
        f_accs = [conditions[c]["family_accuracies"][f]["mean"] * 100 for f in families]
        ax.bar(x + idx * width - 2.5 * width, f_accs, width, label=c, color=palette[c])

    ax.set_xticks(x)
    ax.set_xticklabels(["Feature", "Relational", "Contextual"], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("Task-Family Specific Accuracy Comparison", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p2 = destination / "family_accuracy_comparison.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # Figure 3: Learned router family -> expert selection matrix
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=160)
    mat_data = summary["learned_routing"]["selection_matrix_proportions"]
    mat = np.array([
        [mat_data[fam][exp] * 100 for exp in ("mlp", "graph", "attention", "attention_v2")]
        for fam in ("feature", "relational", "contextual")
    ])

    cax = ax.matshow(mat, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(cax, ax=ax, label="Selection Proportion (%)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["MLP", "Graph", "Attention V1", "Attention V2"], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Feature", "Relational", "Contextual"], fontsize=10)
    ax.set_xlabel("Learned Selected Expert", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task Family", fontsize=11, fontweight="bold")
    ax.set_title("Learned Router Family-to-Expert Selection Matrix", fontsize=12, fontweight="bold", pad=20)

    for i in range(3):
        for j in range(4):
            val = mat[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    fig.tight_layout()
    p3 = destination / "learned_selection_matrix.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # Figure 4: Expert utilization distribution
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    experts = ("mlp", "graph", "attention", "attention_v2")
    x = np.arange(len(experts))
    w = 0.25

    rand_u = [0.25, 0.25, 0.25, 0.25]
    learn_u = [summary["learned_routing"]["module_utilization"][e] for e in experts]
    ora_u = [summary["oracle_routing"]["module_utilization"][e] for e in experts]

    ax.bar(x - w, [u * 100 for u in rand_u], w, label="Random Router", color="#edc948")
    ax.bar(x, [u * 100 for u in learn_u], w, label="Learned Router", color="#2b5c8f")
    ax.bar(x + w, [u * 100 for u in ora_u], w, label="Oracle Router", color="#76b7b2")

    ax.set_xticks(x)
    ax.set_xticklabels(["MLP", "Graph", "Attention V1", "Attention V2"], fontsize=10.5)
    ax.set_ylabel("Module Utilization (%)", fontsize=11)
    ax.set_ylim(0, 70)
    ax.set_title("Module Utilization Distribution Across Routing Modes", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p4 = destination / "expert_utilization_distribution.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # Figure 5: Accuracy vs theoretical FLOPs
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    for name in cond_order:
        data = conditions[name]
        flops = data["compute"]["estimated_forward_flops"]
        acc = data["overall_accuracy"]["mean"] * 100
        ax.scatter(flops, acc, s=130, color=palette[name], zorder=5, label=name)
        ax.annotate(name, (flops, acc), xytext=(6, 4), textcoords="offset points", fontsize=8.5)

    # Plot Pareto connection
    mlp_f = conditions["Fixed MLP"]["compute"]["estimated_forward_flops"]
    mlp_a = conditions["Fixed MLP"]["overall_accuracy"]["mean"] * 100
    learn_f = conditions["Learned Router"]["compute"]["estimated_forward_flops"]
    learn_a = conditions["Learned Router"]["overall_accuracy"]["mean"] * 100
    ax.plot([mlp_f, learn_f], [mlp_a, learn_a], "k--", alpha=0.6, label="Learned Pareto Frontier")

    ax.set_xlabel("Estimated Forward FLOPs / Sample", fontsize=11)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Estimated Compute (Pareto Analysis)", fontsize=12, fontweight="bold")
    ax.set_ylim(40, 102)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p5 = destination / "accuracy_vs_flops.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # Figure 6: Oracle vs learned routing performance
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    categories = ["Overall", "Feature", "Relational", "Contextual"]
    x = np.arange(len(categories))
    w = 0.35

    learn_vals = [
        conditions["Learned Router"]["overall_accuracy"]["mean"] * 100,
        conditions["Learned Router"]["family_accuracies"]["feature"]["mean"] * 100,
        conditions["Learned Router"]["family_accuracies"]["relational"]["mean"] * 100,
        conditions["Learned Router"]["family_accuracies"]["contextual"]["mean"] * 100,
    ]
    ora_vals = [
        conditions["Oracle Router"]["overall_accuracy"]["mean"] * 100,
        conditions["Oracle Router"]["family_accuracies"]["feature"]["mean"] * 100,
        conditions["Oracle Router"]["family_accuracies"]["relational"]["mean"] * 100,
        conditions["Oracle Router"]["family_accuracies"]["contextual"]["mean"] * 100,
    ]

    ax.bar(x - w / 2, learn_vals, w, label="Learned Router", color="#2b5c8f")
    ax.bar(x + w / 2, ora_vals, w, label="Oracle Router (Upper Bound)", color="#76b7b2")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("Learned Router vs. Oracle Upper Bound", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i in range(len(categories)):
        gap = ora_vals[i] - learn_vals[i]
        ax.annotate(f"Gap: {gap:.1f}%", xy=(x[i], max(learn_vals[i], ora_vals[i])), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p6 = destination / "oracle_vs_learned_performance.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_paths.append(str(p6))

    # Figure 7: Routing entropy/utilization across seeds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    seed_labels = [f"Seed {s}" for s in summary["seeds"]]
    seed_accs = [r["overall_accuracy"] * 100 for r in summary["seed_results"]["learned_router"]]
    seed_agrees = [r["overall_route_agreement"] * 100 for r in summary["seed_results"]["learned_router"]]

    ax1.plot(seed_labels, seed_accs, marker="o", color="#2b5c8f", label="Accuracy")
    ax1.plot(seed_labels, seed_agrees, marker="s", color="#e15759", label="Route Agreement")
    ax1.set_ylabel("Percentage (%)", fontsize=10.5)
    ax1.set_ylim(60, 102)
    ax1.set_title("Accuracy & Agreement Across Seeds", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.3)

    seed_entropies = [r["routing_entropy_bits"] for r in summary["seed_results"]["learned_router"]]
    ax2.bar(seed_labels, seed_entropies, color="#4e79a7", width=0.45)
    ax2.axhline(2.0, color="red", linestyle="--", alpha=0.5, label="Max Entropy (2.0 bits)")
    ax2.set_ylabel("Entropy (bits)", fontsize=10.5)
    ax2.set_ylim(0, 2.2)
    ax2.set_title("Routing Entropy Across Seeds", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    p7 = destination / "routing_stability_across_seeds.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_paths.append(str(p7))

    # Figure 8: End-to-end latency decomposition
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    lat_data = summary["latency_breakdown"]["learned_router"]
    components = ["Router Inference", "Dispatch / Grouping", "Expert Forward", "Reassembly"]
    times = [
        lat_data["router_inference_ms"],
        lat_data["dispatch_grouping_ms"],
        lat_data["expert_forward_ms"],
        lat_data["reassembly_ms"],
    ]
    colors_lat = ["#edc948", "#b07aa1", "#59a14f", "#4e79a7"]

    b = ax.bar(components, times, color=colors_lat, width=0.5)
    ax.set_ylabel("Time (ms) per Batch of 60", fontsize=11)
    ax.set_title(f"Learned Router Latency Breakdown (Total: {lat_data['total_end_to_end_ms']:.2f} ms)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for rect in b:
        h = rect.get_height()
        ax.annotate(f"{h:.2f} ms", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p8 = destination / "latency_decomposition.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_paths.append(str(p8))

    return generated_paths
