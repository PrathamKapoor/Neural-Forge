"""Figure generation for Phase 10 Adaptive Multi-Expert Composition."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_phase10_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the 12 required research figures for Phase 10."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: list[str] = []

    # Shared palette
    policy_colors = {
        "k=1 Baseline (Phase 8B)": "#4e79a7",
        "Fixed k=2 (Uniform C1)": "#59a14f",
        "Fixed k=2 (Weighted C2)": "#2ca02c",
        "Fixed k=2 (Normalized C3)": "#1b7837",
        "Fixed k=3": "#e15759",
        "Adaptive k (Learned)": "#b07aa1",
        "Adaptive k (Cost-Aware λ=0.10)": "#9467bd",
        "Sequential Graph->MLP": "#d62728",
        "Sequential MLP->Graph": "#ff7f0e",
        "Sequential AttnV2->MLP": "#8c564b",
        "Sequential Graph->AttnV2": "#17becf",
        "Single-Expert Ceiling": "#7f7f7f",
    }

    # =========================================================================
    # Figure 1: Mixed-Task Accuracy Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    mix_data = summary.get("mixed_task_comparison", {})
    mixed_fams = ["FR", "RC", "FC", "FRC"]
    
    comp_policies = [
        "Single-Expert Ceiling",
        "k=1 Baseline (Phase 8B)",
        "Fixed k=2 (Uniform C1)",
        "Fixed k=3",
        "Adaptive k (Learned)",
    ]
    active_pols = [p for p in comp_policies if any(p in mix_data.get(fam, {}) for fam in mixed_fams)]
    if not active_pols:
        active_pols = comp_policies

    x = np.arange(len(mixed_fams))
    n_pols = len(active_pols)
    width = 0.8 / max(1, n_pols)

    for idx, pol in enumerate(active_pols):
        vals = [mix_data.get(fam, {}).get(pol, 0.0) * 100 for fam in mixed_fams]
        offset = (idx - (n_pols - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=pol, color=policy_colors.get(pol, "#888888"), edgecolor="black", linewidth=0.6)
        for b in bars:
            h = b.get_height()
            if h > 10:
                ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.0, fontweight="bold")

    ax.axhline(100.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(mixed_fams, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.set_title("Mixed-Structure Benchmark: Single-Expert vs. Multi-Expert Policies", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=8.0, loc="lower right")
    fig.tight_layout()
    p1 = destination / "mixed_task_accuracy_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # =========================================================================
    # Figure 2: Accuracy vs FLOPs
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    pts = summary.get("performance_compute_points", [])

    for pt in pts:
        pol = pt.get("policy", "Unknown")
        acc = pt.get("accuracy", 0.0) * 100
        flops = pt.get("flops", 1000.0)
        col = policy_colors.get(pol, "#1f77b4")
        ax.scatter(flops, acc, color=col, s=90, alpha=0.85, edgecolors="black", linewidth=0.8, label=pol)

    ax.set_xlabel("Average Theoretical FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Benchmark Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance–Theoretical Compute Frontier (Phase 10)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(45, 102)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=8.0, loc="lower right")
    fig.tight_layout()
    p2 = destination / "accuracy_vs_flops.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # =========================================================================
    # Figure 3: Accuracy vs Average Active Experts
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    k_eval = summary.get("accuracy_vs_active_experts", [])
    if k_eval:
        k_vals = [pt["active_experts"] for pt in k_eval]
        acc_pure = [pt.get("accuracy_pure", 0.0) * 100 for pt in k_eval]
        acc_mixed = [pt.get("accuracy_mixed", 0.0) * 100 for pt in k_eval]
        acc_all = [pt.get("accuracy_overall", 0.0) * 100 for pt in k_eval]

        ax.plot(k_vals, acc_mixed, "o-", color="#d62728", linewidth=2.2, markersize=8, label="Mixed Tasks (FR, RC, FC, FRC)")
        ax.plot(k_vals, acc_all, "s-", color="#1f77b4", linewidth=2.2, markersize=8, label="Overall Population")
        ax.plot(k_vals, acc_pure, "^-", color="#2ca02c", linewidth=2.2, markersize=8, label="Pure Tasks (F, R, C)")

        for i, k in enumerate(k_vals):
            ax.annotate(f"{acc_mixed[i]:.1f}%", (k, acc_mixed[i]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.0, color="#d62728", fontweight="bold")
            ax.annotate(f"{acc_all[i]:.1f}%", (k, acc_all[i]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8.0, color="#1f77b4", fontweight="bold")

        ax.set_xticks(k_vals)
        ax.set_xlabel("Active Expert Count (k)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(50, 105)
        ax.set_title("Accuracy Trajectory vs. Active Expert Count (k)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True, fontsize=9.0, loc="lower right")
    fig.tight_layout()
    p3 = destination / "accuracy_vs_active_experts.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # =========================================================================
    # Figure 4: k Distribution Across Families (Adaptive k)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    k_dist_data = summary.get("k_distribution", {})
    fams = ["F", "R", "C", "FR", "RC", "FC", "FRC"]

    p_k1 = [k_dist_data.get(f, {}).get("p_k1", 1.0) * 100 for f in fams]
    p_k2 = [k_dist_data.get(f, {}).get("p_k2", 0.0) * 100 for f in fams]
    p_k3 = [k_dist_data.get(f, {}).get("p_k3", 0.0) * 100 for f in fams]

    x_k = np.arange(len(fams))
    ax.bar(x_k, p_k1, label="k = 1 Expert", color="#4e79a7", edgecolor="black", linewidth=0.7)
    ax.bar(x_k, p_k2, bottom=p_k1, label="k = 2 Experts", color="#f28e2b", edgecolor="black", linewidth=0.7)
    bottom_k3 = [p_k1[i] + p_k2[i] for i in range(len(fams))]
    ax.bar(x_k, p_k3, bottom=bottom_k3, label="k = 3 Experts", color="#e15759", edgecolor="black", linewidth=0.7)

    for i in range(len(fams)):
        if p_k1[i] > 15:
            ax.text(x_k[i], p_k1[i] / 2, f"{p_k1[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.5)
        if p_k2[i] > 15:
            ax.text(x_k[i], p_k1[i] + p_k2[i] / 2, f"{p_k2[i]:.0f}%", ha="center", va="center", color="black", fontweight="bold", fontsize=7.5)
        if p_k3[i] > 15:
            ax.text(x_k[i], bottom_k3[i] + p_k3[i] / 2, f"{p_k3[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=7.5)

    ax.set_xticks(x_k)
    ax.set_xticklabels(fams, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Selection Frequency (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Adaptive k Allocation Across Task Families", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9.0, loc="upper right")
    fig.tight_layout()
    p4 = destination / "k_distribution.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # =========================================================================
    # Figure 5: Expert-Pair Utilization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    pair_data = summary.get("expert_pair_utilization", {})
    pairs = list(pair_data.keys())
    pair_vals = [pair_data[p] * 100 for p in pairs]

    if pair_vals:
        x_p = np.arange(len(pairs))
        bars = ax.bar(x_p, pair_vals, color="#59a14f", edgecolor="black", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            if h > 1.0:
                ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold")
        ax.set_xticks(x_p)
        ax.set_xticklabels(pairs, rotation=20, ha="right", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Pair Co-Occurrence Frequency (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, max(pair_vals) * 1.25 if pair_vals else 100)
        ax.set_title("Expert-Pair Co-Activation Distribution (Fixed k=2 & Adaptive)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    fig.tight_layout()
    p5 = destination / "expert_pair_utilization.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # =========================================================================
    # Figure 6: Family -> Composition Matrix
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=160)
    fam_comp = summary.get("family_composition_matrix", {})
    fams_list = ["F", "R", "C", "FR", "RC", "FC", "FRC"]
    
    # Collect all observed compositions
    all_comps = sorted(list(set(c for f in fams_list for c in fam_comp.get(f, {}))))
    if not all_comps:
        all_comps = ["MLP", "Graph", "V2", "Graph+MLP", "Graph+V2", "MLP+V2", "Graph+MLP+V2"]

    mat = np.zeros((len(fams_list), len(all_comps)))
    for r_idx, fam in enumerate(fams_list):
        for c_idx, comp in enumerate(all_comps):
            mat[r_idx, c_idx] = fam_comp.get(fam, {}).get(comp, 0.0) * 100

    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Selection Frequency (%)", fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(len(all_comps)))
    ax.set_xticklabels(all_comps, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(len(fams_list)))
    ax.set_yticklabels(fams_list, fontsize=10.5, fontweight="bold")

    for i in range(len(fams_list)):
        for j in range(len(all_comps)):
            v = mat[i, j]
            if v > 1.0:
                txt_col = "white" if v > 50 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color=txt_col, fontsize=7.5, fontweight="bold")

    ax.set_title("Task Family → Composition Allocation Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p6 = destination / "family_composition_matrix.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_paths.append(str(p6))

    # =========================================================================
    # Figure 7: Single-Expert Ceiling vs. Multi-Expert Performance
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    ceil_data = summary.get("ceiling_vs_multi", {})
    m_fams = ["FR", "RC", "FC", "FRC"]

    ceil_vals = [ceil_data.get(f, {}).get("ceiling", 0.0) * 100 for f in m_fams]
    k1_vals = [ceil_data.get(f, {}).get("k1", 0.0) * 100 for f in m_fams]
    k2_vals = [ceil_data.get(f, {}).get("k2", 0.0) * 100 for f in m_fams]
    k3_vals = [ceil_data.get(f, {}).get("k3", 0.0) * 100 for f in m_fams]

    x_c = np.arange(len(m_fams))
    w = 0.2
    ax.bar(x_c - 1.5 * w, ceil_vals, w, label="Single-Expert Ceiling", color="#7f7f7f", edgecolor="black")
    ax.bar(x_c - 0.5 * w, k1_vals, w, label="k=1 Router", color="#4e79a7", edgecolor="black")
    ax.bar(x_c + 0.5 * w, k2_vals, w, label="k=2 Composition", color="#2ca02c", edgecolor="black")
    ax.bar(x_c + 1.5 * w, k3_vals, w, label="k=3 Composition", color="#e15759", edgecolor="black")

    for i in range(len(m_fams)):
        ax.annotate(f"{k2_vals[i]:.1f}%", xy=(x_c[i] + 0.5 * w, k2_vals[i]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

    ax.axhline(100.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xticks(x_c)
    ax.set_xticklabels(m_fams, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.set_title("Single-Expert Ceiling vs. Multi-Expert Performance on Mixed Tasks", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p7 = destination / "single_expert_ceiling_vs_multi_expert.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_paths.append(str(p7))

    # =========================================================================
    # Figure 8: Parallel vs Sequential Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    comp_types = summary.get("parallel_vs_sequential", {})
    arch_labels = list(comp_types.keys())
    arch_accs = [comp_types[a].get("accuracy", 0.0) * 100 for a in arch_labels]
    arch_flops = [comp_types[a].get("flops", 1000.0) for a in arch_labels]

    if arch_labels:
        x_arch = np.arange(len(arch_labels))
        bars = ax.bar(x_arch, arch_accs, color=["#1f77b4" if "Parallel" in a else "#ff7f0e" for a in arch_labels], edgecolor="black", linewidth=0.7)
        for idx, b in enumerate(bars):
            h = b.get_height()
            fl = arch_flops[idx]
            ax.annotate(f"{h:.1f}%\n({fl:,.0f} FLOPs)", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x_arch)
        ax.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=9.0)
        ax.set_ylabel("Mixed-Task Accuracy (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(40, 105)
        ax.set_title("Parallel Aggregation vs. Sequential Composition Pipelines", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    fig.tight_layout()
    p8 = destination / "parallel_vs_sequential_comparison.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_paths.append(str(p8))

    # =========================================================================
    # Figure 9: Adaptive-k Behavior
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    strat_data = summary.get("adaptive_k_strategies", {})
    strats = list(strat_data.keys())
    strat_accs = [strat_data[s].get("accuracy", 0.0) * 100 for s in strats]
    strat_k = [strat_data[s].get("mean_k", 1.0) for s in strats]

    if strats:
        x_s = np.arange(len(strats))
        ax.bar(x_s, strat_accs, color="#b07aa1", edgecolor="black", linewidth=0.7)
        for i, b in enumerate(ax.patches):
            h = b.get_height()
            k_val = strat_k[i]
            ax.annotate(f"{h:.1f}%\n(avg k={k_val:.2f})", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold")
        ax.set_xticks(x_s)
        ax.set_xticklabels(strats, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(45, 105)
        ax.set_title("Comparison of Adaptive k Selection Mechanisms", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    fig.tight_layout()
    p9 = destination / "adaptive_k_behavior.png"
    fig.savefig(p9)
    plt.close(fig)
    generated_paths.append(str(p9))

    # =========================================================================
    # Figure 10: Representative Latency Decomposition
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    lat_data = summary.get("latency_decomposition", {})
    lat_pols = list(lat_data.keys())

    if lat_pols:
        x_l = np.arange(len(lat_pols))
        router_us = [lat_data[p].get("router_us", 0.0) for p in lat_pols]
        expert_us = [lat_data[p].get("expert_us", 0.0) for p in lat_pols]
        agg_us = [lat_data[p].get("agg_us", 0.0) for p in lat_pols]

        ax.bar(x_l, expert_us, 0.5, label="Expert Execution", color="#4e79a7", edgecolor="black")
        ax.bar(x_l, router_us, 0.5, bottom=expert_us, label="Router Overhead", color="#e15759", edgecolor="black")
        bottom_agg = [expert_us[i] + router_us[i] for i in range(len(lat_pols))]
        ax.bar(x_l, agg_us, 0.5, bottom=bottom_agg, label="Aggregation Overhead", color="#59a14f", edgecolor="black")

        for i in range(len(lat_pols)):
            tot = bottom_agg[i] + agg_us[i]
            ax.annotate(f"{tot:.0f} µs", xy=(x_l[i], tot), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold")

        ax.set_xticks(x_l)
        ax.set_xticklabels(lat_pols, rotation=25, ha="right", fontsize=9.0)
        ax.set_ylabel("Inference Latency (µs/sample)", fontsize=10.5, fontweight="bold")
        ax.set_title("CPU Latency Breakdown: Routing vs. Execution vs. Aggregation", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    p10 = destination / "latency_decomposition.png"
    fig.savefig(p10)
    plt.close(fig)
    generated_paths.append(str(p10))

    # =========================================================================
    # Figure 11: Compute-Accuracy Pareto Frontier
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    pareto_pts = summary.get("pareto_points", [])

    if pareto_pts:
        all_pts = summary.get("performance_compute_points", [])
        # Scatter non-Pareto in grey
        for pt in all_pts:
            if pt not in pareto_pts:
                ax.scatter(pt["flops"], pt["accuracy"] * 100, color="#cccccc", s=50, alpha=0.6, zorder=2)

        # Plot Pareto points connected
        sorted_pareto = sorted(pareto_pts, key=lambda p: p["flops"])
        px = [p["flops"] for p in sorted_pareto]
        py = [p["accuracy"] * 100 for p in sorted_pareto]
        ax.plot(px, py, "r--", linewidth=1.8, label="Empirical Pareto Frontier", zorder=3)

        for p in sorted_pareto:
            lbl = p["policy"]
            ax.scatter(p["flops"], p["accuracy"] * 100, s=90, color=policy_colors.get(lbl, "#d62728"), edgecolors="black", linewidth=0.8, zorder=4, label=lbl)
            ax.annotate(f"{lbl}\n({p['accuracy']*100:.1f}%)", xy=(p["flops"], p["accuracy"] * 100),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

        ax.set_xlabel("Estimated Theoretical FLOPs / Sample", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Empirical Performance–Theoretical Compute Pareto Frontier", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(45, 105)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    p11 = destination / "compute_accuracy_pareto_frontier.png"
    fig.savefig(p11)
    plt.close(fig)
    generated_paths.append(str(p11))

    # =========================================================================
    # Figure 12: Seed Stability (Seeds 11, 23, 37)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    seed_stability = summary.get("seed_stability", {})
    stab_pols = list(seed_stability.keys())

    if stab_pols:
        x_st = np.arange(len(stab_pols))
        means = [seed_stability[p]["mean_accuracy"] * 100 for p in stab_pols]
        stds = [seed_stability[p]["std_accuracy"] * 100 for p in stab_pols]

        bars = ax.bar(x_st, means, yerr=stds, capsize=4, color="#4e79a7", edgecolor="black", linewidth=0.7)
        for i, b in enumerate(bars):
            m = means[i]
            s = stds[i]
            ax.annotate(f"{m:.1f}±{s:.1f}%", xy=(b.get_x() + b.get_width() / 2, m),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x_st)
        ax.set_xticklabels(stab_pols, rotation=20, ha="right", fontsize=9.0)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(45, 105)
        ax.set_title("Routing Stability Across Random Seeds (11, 23, 37)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    fig.tight_layout()
    p12 = destination / "seed_stability.png"
    fig.savefig(p12)
    plt.close(fig)
    generated_paths.append(str(p12))

    return generated_paths
