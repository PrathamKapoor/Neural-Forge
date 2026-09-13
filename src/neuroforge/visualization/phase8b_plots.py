"""Figure generation for Phase 8B Compute-Aware Learned Routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_phase8b_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the 10 required research figures for Phase 8B."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: list[str] = []

    lambda_results = summary["lambda_sweep"]
    lambdas = [r["lambda"] for r in lambda_results]
    lambda_labels = [f"λ={lam:.3f}" if lam > 0 else "λ=0" for lam in lambdas]
    conditions = summary["conditions"]

    # Palette
    palette = {
        "Fixed MLP": "#4e79a7",
        "Fixed Graph": "#59a14f",
        "Fixed Attention V1": "#e15759",
        "Fixed Attention V2": "#b07aa1",
        "Random Router": "#edc948",
        "Learned Router (Phase 8A)": "#2b5c8f",
        "Oracle Router": "#76b7b2",
    }
    expert_colors = {
        "mlp": "#4e79a7",
        "graph": "#59a14f",
        "attention": "#e15759",
        "attention_v2": "#b07aa1",
    }

    x_idx = np.arange(len(lambdas))

    # -------------------------------------------------------------
    # Figure 1: Accuracy vs Lambda
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    acc_means = [r["overall_accuracy"]["mean"] * 100 for r in lambda_results]
    acc_stds = [r["overall_accuracy"]["std"] * 100 for r in lambda_results]

    ax.errorbar(
        x_idx, acc_means, yerr=acc_stds, fmt="o-", color="#1f77b4",
        linewidth=2.2, markersize=7, capsize=4, label="Cost-Aware Router (Phase 8B)", zorder=5
    )

    # Reference horizontal lines
    best_fixed = conditions["Fixed Attention V2"]["overall_accuracy"]["mean"] * 100
    oracle_acc = conditions["Oracle Router"]["overall_accuracy"]["mean"] * 100
    rand_acc = conditions["Random Router"]["overall_accuracy"]["mean"] * 100
    p8a_acc = conditions["Learned Router (Phase 8A)"]["overall_accuracy"]["mean"] * 100

    ax.axhline(oracle_acc, color="#76b7b2", linestyle="--", linewidth=1.5, label=f"Oracle Router ({oracle_acc:.1f}%)")
    ax.axhline(p8a_acc, color="#2b5c8f", linestyle=":", linewidth=1.5, label=f"Phase 8A Router ({p8a_acc:.1f}%)")
    ax.axhline(best_fixed, color="#b07aa1", linestyle="-.", linewidth=1.5, label=f"Fixed Attention V2 ({best_fixed:.1f}%)")
    ax.axhline(rand_acc, color="#edc948", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Random Router ({rand_acc:.1f}%)")

    for i, (m, s) in enumerate(zip(acc_means, acc_stds)):
        ax.annotate(f"{m:.1f}%", xy=(x_idx[i], m), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(lambda_labels, fontsize=9.5)
    ax.set_xlabel("Cost Penalty Weight (λ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(45, 105)
    ax.set_title("Test Accuracy vs. Cost Penalty Weight (λ)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    p1 = destination / "accuracy_vs_lambda.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # -------------------------------------------------------------
    # Figure 2: FLOPs vs Lambda
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    flop_means = [r["compute"]["estimated_forward_flops"] for r in lambda_results]

    ax.plot(x_idx, flop_means, "s-", color="#d62728", linewidth=2.2, markersize=7, label="Cost-Aware Router (incl. 1,052 router FLOPs)", zorder=5)

    v2_flops = conditions["Fixed Attention V2"]["compute"]["estimated_forward_flops"]
    mlp_flops = conditions["Fixed MLP"]["compute"]["estimated_forward_flops"]
    graph_flops = conditions["Fixed Graph"]["compute"]["estimated_forward_flops"]
    ora_flops = conditions["Oracle Router"]["compute"]["estimated_forward_flops"]

    ax.axhline(v2_flops, color="#b07aa1", linestyle="-.", linewidth=1.5, label=f"Fixed Attention V2 ({v2_flops:,.0f})")
    ax.axhline(ora_flops, color="#76b7b2", linestyle="--", linewidth=1.5, label=f"Oracle Router ({ora_flops:,.0f})")
    ax.axhline(mlp_flops, color="#4e79a7", linestyle=":", linewidth=1.5, label=f"Fixed MLP ({mlp_flops:,.0f})")
    ax.axhline(graph_flops, color="#59a14f", linestyle=":", linewidth=1.5, label=f"Fixed Graph ({graph_flops:,.0f})")

    for i, f in enumerate(flop_means):
        ax.annotate(f"{f:,.0f}", xy=(x_idx[i], f), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(lambda_labels, fontsize=9.5)
    ax.set_xlabel("Cost Penalty Weight (λ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Forward FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_title("Total Compute (FLOPs) vs. Cost Penalty Weight (λ)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    p2 = destination / "flops_vs_lambda.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # -------------------------------------------------------------
    # Figure 3: Accuracy vs Total FLOPs Pareto Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

    # Plot baselines
    for b_name in ("Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2", "Random Router", "Learned Router (Phase 8A)", "Oracle Router"):
        data = conditions[b_name]
        fl = data["compute"]["estimated_forward_flops"]
        ac = data["overall_accuracy"]["mean"] * 100
        marker = "*" if "Oracle" in b_name else "o"
        size = 140 if "Oracle" in b_name else 100
        ax.scatter(fl, ac, s=size, color=palette[b_name], marker=marker, zorder=4, label=b_name)
        ax.annotate(b_name, (fl, ac), xytext=(6, 3), textcoords="offset points", fontsize=8)

    # Plot Cost-Aware points
    ca_flops = [r["compute"]["estimated_forward_flops"] for r in lambda_results]
    ca_accs = [r["overall_accuracy"]["mean"] * 100 for r in lambda_results]
    ax.plot(ca_flops, ca_accs, "k:", alpha=0.5, linewidth=1.2, zorder=3)
    ax.scatter(ca_flops, ca_accs, s=90, c=lambdas, cmap="coolwarm", edgecolors="black", linewidths=1.2, zorder=5, label="Cost-Aware (λ sweep)")

    for r in lambda_results:
        fl = r["compute"]["estimated_forward_flops"]
        ac = r["overall_accuracy"]["mean"] * 100
        ax.annotate(f"λ={r['lambda']}", (fl, ac), xytext=(-4, 6), textcoords="offset points", fontsize=7.5, color="#333333")

    # Connect Pareto frontier
    pareto_res = summary["pareto_analysis"]
    frontier_pts = pareto_res["frontier_sorted"]
    f_fl = [p["flops"] for p in frontier_pts]
    f_ac = [p["accuracy"] * 100 for p in frontier_pts]
    ax.plot(f_fl, f_ac, "r--", linewidth=2.0, alpha=0.85, label="Empirical Pareto Frontier", zorder=6)

    ax.set_xlabel("Total Estimated Forward FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance–Compute Pareto Frontier (Phase 8B)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    p3 = destination / "accuracy_vs_flops_pareto.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # -------------------------------------------------------------
    # Figure 4: Expert Utilization vs Lambda
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    experts = ("mlp", "graph", "attention", "attention_v2")
    expert_display = {"mlp": "MLP", "graph": "Graph", "attention": "Attention V1", "attention_v2": "AttentionBlockV2"}

    bar_width = 0.6
    bottoms = np.zeros(len(lambdas))

    for exp in experts:
        utils = [r["routing"]["module_utilization"][exp] * 100 for r in lambda_results]
        ax.bar(
            x_idx, utils, bar_width, bottom=bottoms,
            label=f"{expert_display[exp]}", color=expert_colors[exp], alpha=0.9
        )
        bottoms += np.array(utils)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(lambda_labels, fontsize=9.5)
    ax.set_xlabel("Cost Penalty Weight (λ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expert Selection Share (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Expert Utilization Shift Across Cost Penalties", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p4 = destination / "expert_utilization_vs_lambda.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # -------------------------------------------------------------
    # Figure 5: Family-to-Expert Routing Matrices across selected lambdas
    # -------------------------------------------------------------
    # Select 4 representative lambdas: 0.0, 0.03, 0.1, 1.0 (or closest)
    target_lambdas = [0.0, 0.03, 0.1, 1.0]
    selected_indices = []
    for tl in target_lambdas:
        best_idx = int(np.argmin([abs(lam - tl) for lam in lambdas]))
        if best_idx not in selected_indices:
            selected_indices.append(best_idx)
    while len(selected_indices) < 4 and len(selected_indices) < len(lambdas):
        for i in range(len(lambdas)):
            if i not in selected_indices:
                selected_indices.append(i)
                break

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=160)
    axes_flat = axes.flatten()

    for plot_i, idx in enumerate(selected_indices[:4]):
        ax_sub = axes_flat[plot_i]
        r = lambda_results[idx]
        lam_val = r["lambda"]
        sel_mat = r["routing"]["selection_matrix_proportions"]

        mat = np.array([
            [sel_mat[fam][exp] * 100 for exp in experts]
            for fam in ("feature", "relational", "contextual")
        ])

        ax_sub.matshow(mat, cmap="Blues", vmin=0, vmax=100)
        ax_sub.set_xticks(range(4))
        ax_sub.set_xticklabels(["MLP", "Graph", "Attn V1", "Attn V2"], fontsize=8.5)
        ax_sub.set_yticks(range(3))
        ax_sub.set_yticklabels(["Feature", "Relational", "Contextual"], fontsize=8.5)
        ax_sub.set_title(f"λ = {lam_val:.3f} (Acc: {r['overall_accuracy']['mean']*100:.1f}%)", fontsize=10, fontweight="bold", pad=12)

        for i in range(3):
            for j in range(4):
                val = mat[i, j]
                color = "white" if val > 50 else "black"
                ax_sub.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=8.5)

    fig.suptitle("Family-to-Expert Selection Matrices Across Cost Penalties", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    p5 = destination / "family_expert_routing_matrices.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # -------------------------------------------------------------
    # Figure 6: Routing Entropy vs Lambda
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    entropies = [r["routing"]["routing_entropy_bits"] for r in lambda_results]

    ax.plot(x_idx, entropies, "o-", color="#8c564b", linewidth=2.2, markersize=7, label="Population Routing Entropy (bits)")
    ax.axhline(2.0, color="red", linestyle="--", alpha=0.6, label="Maximum Entropy (2.0 bits = 4 uniform)")
    ax.axhline(1.585, color="orange", linestyle=":", alpha=0.6, label="3-Expert Uniform (1.585 bits)")
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.4, label="Complete Collapse (0.0 bits)")

    for i, h in enumerate(entropies):
        ax.annotate(f"{h:.2f}b", xy=(x_idx[i], h), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(lambda_labels, fontsize=9.5)
    ax.set_xlabel("Cost Penalty Weight (λ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Entropy (bits)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.1, 2.2)
    ax.set_title("Routing Decision Entropy vs. Cost Penalty Weight (λ)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    p6 = destination / "routing_entropy_vs_lambda.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_paths.append(str(p6))

    # -------------------------------------------------------------
    # Figure 7: Accuracy vs Measured Latency
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)

    # Baselines
    for b_name in ("Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2", "Learned Router (Phase 8A)"):
        data = conditions[b_name]
        lat = data["latency"]["batch_latency_ms"]
        ac = data["overall_accuracy"]["mean"] * 100
        ax.scatter(lat, ac, s=110, color=palette[b_name], zorder=4, label=b_name)
        ax.annotate(b_name, (lat, ac), xytext=(6, 3), textcoords="offset points", fontsize=8)

    # Lambda points
    ca_lats = [r["latency"]["batch_latency_ms"] for r in lambda_results]
    ax.scatter(ca_lats, acc_means, s=85, c=lambdas, cmap="viridis", edgecolors="black", linewidths=1.0, zorder=5, label="Cost-Aware (λ sweep)")
    for i, r in enumerate(lambda_results):
        ax.annotate(f"λ={r['lambda']}", (ca_lats[i], acc_means[i]), xytext=(-5, 6), textcoords="offset points", fontsize=7.5)

    ax.set_xlabel("Measured End-to-End Batch Latency (ms, batch_size=60)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Hardware Reality: Accuracy vs. Physical Batch Latency", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    p7 = destination / "accuracy_vs_latency.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_paths.append(str(p7))

    # -------------------------------------------------------------
    # Figure 8: Oracle / Phase 8A / Cost-Aware Frontier Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)

    # Fixed baselines grayed background
    for b_name in ("Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2"):
        fl = conditions[b_name]["compute"]["estimated_forward_flops"]
        ac = conditions[b_name]["overall_accuracy"]["mean"] * 100
        ax.scatter(fl, ac, s=80, color="#999999", alpha=0.7, zorder=3)
        ax.annotate(b_name, (fl, ac), xytext=(5, -5), textcoords="offset points", fontsize=7.5, color="#666666")

    # Oracle point
    ora_fl = conditions["Oracle Router"]["compute"]["estimated_forward_flops"]
    ora_ac = conditions["Oracle Router"]["overall_accuracy"]["mean"] * 100
    ax.scatter(ora_fl, ora_ac, s=180, color="#76b7b2", marker="*", zorder=6, label=f"Oracle Upper Bound ({ora_ac:.1f}%)")

    # Phase 8A point
    p8a_fl = conditions["Learned Router (Phase 8A)"]["compute"]["estimated_forward_flops"]
    p8a_ac = conditions["Learned Router (Phase 8A)"]["overall_accuracy"]["mean"] * 100
    ax.scatter(p8a_fl, p8a_ac, s=120, color="#2b5c8f", marker="D", zorder=6, label=f"Phase 8A Unconstrained ({p8a_ac:.1f}%)")

    # Phase 8B trajectory
    ax.plot(ca_flops, ca_accs, "r-o", linewidth=2.0, markersize=6, zorder=5, label="Phase 8B Cost-Aware Trajectory")

    # Shaded region between Phase 8A and Oracle
    ax.fill_between([min(ca_flops), max(ca_flops)], [p8a_ac, p8a_ac], [ora_ac, ora_ac], color="#76b7b2", alpha=0.1, label="Oracle Advantage Zone")

    ax.set_xlabel("Total Forward FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Population Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Oracle vs. Phase 8A Unconstrained vs. Phase 8B Frontier", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p8 = destination / "frontier_comparative_overview.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_paths.append(str(p8))

    # -------------------------------------------------------------
    # Figure 9: Family-Specific Compute vs Lambda
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    fam_colors = {"feature": "#4e79a7", "relational": "#59a14f", "contextual": "#b07aa1"}

    for fam in ("feature", "relational", "contextual"):
        fam_flops = [r["family_conditional_cost"][fam]["mean_flops"] for r in lambda_results]
        ax.plot(x_idx, fam_flops, "o-", color=fam_colors[fam], linewidth=2.0, markersize=6, label=f"{fam.capitalize()} Family FLOPs")
        for i, f in enumerate(fam_flops):
            ax.annotate(f"{f:,.0f}", xy=(x_idx[i], f), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7.5, color=fam_colors[fam])

    ax.set_xticks(x_idx)
    ax.set_xticklabels(lambda_labels, fontsize=9.5)
    ax.set_xlabel("Cost Penalty Weight (λ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task Family Forward FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_title("Task-Family Specific Compute Allocation Across Cost Penalties", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()
    p9 = destination / "family_specific_compute_vs_lambda.png"
    fig.savefig(p9)
    plt.close(fig)
    generated_paths.append(str(p9))

    # -------------------------------------------------------------
    # Figure 10: Latency Decomposition for Representative Operating Points
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    rep_points = summary["representative_operating_points"]
    point_names = ["Accuracy-First", "Balanced", "Compute-First"]
    def _extract_lam(pt: dict[str, Any] | None, fallback: float) -> float:
        if not pt:
            return fallback
        if "lambda" in pt:
            return float(pt["lambda"])
        import re
        m = re.search(r"λ=([0-9.]+)", pt.get("name", ""))
        if m:
            return float(m.group(1))
        return fallback

    point_lambdas = [
        _extract_lam(rep_points.get("accuracy_first"), 0.0),
        _extract_lam(rep_points.get("balanced"), 0.03),
        _extract_lam(rep_points.get("compute_first"), 1.0),
    ]

    # Find matching lambda result dictionaries
    matching_rs = []
    for pl in point_lambdas:
        idx = int(np.argmin([abs(r["lambda"] - pl) for r in lambda_results]))
        matching_rs.append(lambda_results[idx])

    cat_labels = [f"{pname}\n(λ={matching_rs[i]['lambda']})" for i, pname in enumerate(point_names)]
    x_pos = np.arange(len(cat_labels))
    b_w = 0.45

    t_router = [r["latency_breakdown"]["router_inference_ms"] for r in matching_rs]
    t_disp = [r["latency_breakdown"]["dispatch_grouping_ms"] for r in matching_rs]
    t_exp = [r["latency_breakdown"]["expert_forward_ms"] for r in matching_rs]
    t_reas = [r["latency_breakdown"]["reassembly_ms"] for r in matching_rs]

    ax.bar(x_pos, t_router, b_w, label="Router Inference", color="#edc948")
    ax.bar(x_pos, t_disp, b_w, bottom=t_router, label="Dispatch / Grouping", color="#b07aa1")
    bottom_exp = np.array(t_router) + np.array(t_disp)
    ax.bar(x_pos, t_exp, b_w, bottom=bottom_exp, label="Selected Expert Forward", color="#59a14f")
    bottom_reas = bottom_exp + np.array(t_exp)
    ax.bar(x_pos, t_reas, b_w, bottom=bottom_reas, label="Output Reassembly", color="#4e79a7")

    totals = bottom_reas + np.array(t_reas)
    for i, tot in enumerate(totals):
        ax.annotate(f"Total: {tot:.2f} ms", xy=(x_pos[i], tot), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, fontsize=9.5)
    ax.set_ylabel("Execution Time (ms) per Batch of 60", fontsize=11, fontweight="bold")
    ax.set_title("Latency Breakdown for Representative Operating Points", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p10 = destination / "representative_latency_decomposition.png"
    fig.savefig(p10)
    plt.close(fig)
    generated_paths.append(str(p10))

    return generated_paths
