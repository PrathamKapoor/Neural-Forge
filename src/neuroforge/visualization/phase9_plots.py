"""Figure generation for Phase 9 Generalization and Mixed-Structure Evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_phase9_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate the 10 required research figures for Phase 9."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: list[str] = []

    # Shared palette
    policy_colors = {
        "Oracle Router": "#76b7b2",
        "Balanced (λ=0.10)": "#1f77b4",
        "Accuracy-first (λ=0.01)": "#2ca02c",
        "Compute-first (λ=1.0)": "#d62728",
        "Unconstrained (λ=0.0)": "#9467bd",
        "Fixed Attention V2": "#b07aa1",
        "Fixed Graph": "#59a14f",
        "Fixed MLP": "#4e79a7",
        "Fixed Attention V1": "#e15759",
    }

    expert_colors = {
        "mlp": "#4e79a7",
        "graph": "#59a14f",
        "attention": "#e15759",
        "attention_v2": "#b07aa1",
    }

    # =========================================================================
    # Figure 1: Fresh-test vs Phase 8B benchmark accuracy (9A)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    data_9a = summary.get("regime_9a_fresh", {})
    bench_acc = data_9a.get("benchmark_accuracy", {})
    fresh_acc = data_9a.get("fresh_accuracy", {})

    policies = [p for p in ["Oracle Router", "Balanced (λ=0.10)", "Accuracy-first (λ=0.01)", "Compute-first (λ=1.0)", "Fixed Attention V2", "Fixed Graph", "Fixed MLP"] if p in bench_acc]
    if not policies:
        policies = list(bench_acc.keys())

    x = np.arange(len(policies))
    width = 0.35

    bench_vals = [bench_acc.get(p, 0.0) * 100 for p in policies]
    fresh_vals = [fresh_acc.get(p, 0.0) * 100 for p in policies]

    bars1 = ax.bar(x - width / 2, bench_vals, width, label="Canonical Phase 8B Test", color="#4e79a7", alpha=0.9, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, fresh_vals, width, label="Fresh In-Distribution Test (Seed +80k)", color="#59a14f", alpha=0.9, edgecolor="black", linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.set_title("9A: In-Distribution Generalization (Canonical vs Fresh Test Sets)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    fig.tight_layout()
    p1 = destination / "fresh_test_vs_phase8b_benchmark.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_paths.append(str(p1))

    # =========================================================================
    # Figure 2: Structural-transformation accuracy (9B)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=160)
    data_9b = summary.get("regime_9b_structural", {})
    t_data = data_9b.get("transformations", {})

    t_keys = ["canonical", "b1_token_perm", "b2_marker_var", "b3_seq_len_8", "b3_seq_len_16", "b4_destructive"]
    t_labels = ["Fresh Baseline", "B1: Token Perm.", "B2: Marker Var.", "B3: S=8", "B3: S=16", "B4: Destructive Rel."]

    sub_policies = ["Balanced (λ=0.10)", "Accuracy-first (λ=0.01)", "Fixed Graph", "Oracle Router"]
    active_sub_pols = [p for p in sub_policies if any(p in t_data.get(k, {}) for k in t_keys)]

    x = np.arange(len(t_keys))
    n_pols = len(active_sub_pols)
    width = 0.8 / max(1, n_pols)

    for idx, pol in enumerate(active_sub_pols):
        vals = [t_data.get(k, {}).get(pol, 0.0) * 100 for k in t_keys]
        offset = (idx - (n_pols - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=pol, color=policy_colors.get(pol, "#888888"), edgecolor="black", linewidth=0.6)
        for b in bars:
            h = b.get_height()
            if h > 10:
                ax.annotate(f"{h:.0f}%", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(35, 105)
    ax.set_title("9B: Structural Transformation Generalization (B1–B4)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    p2 = destination / "structural_transformation_accuracy.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_paths.append(str(p2))

    # =========================================================================
    # Figure 3: Distribution-shift accuracy curves (9C)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    data_9c = summary.get("regime_9c_distribution_shift", {}).get("shifts", {})
    severities = ["clean", "mild", "moderate", "strong"]
    sev_x = np.arange(len(severities))

    shift_styles = {
        "magnitude_scaling": ("s-", "#1f77b4", "Magnitude Scaling"),
        "additive_noise": ("o-", "#d62728", "Additive Gaussian Noise"),
        "variance_contrast": ("^-", "#2ca02c", "Variance Contrast"),
    }

    target_pol = "Balanced (λ=0.10)"
    for s_name, (fmt, col, label) in shift_styles.items():
        s_dict = data_9c.get(s_name, {})
        accs = [s_dict.get(sev, {}).get("accuracy", {}).get(target_pol, 0.0) * 100 for sev in severities]
        ax.plot(sev_x, accs, fmt, color=col, linewidth=2.0, markersize=7, label=label)
        for i, val in enumerate(accs):
            ax.annotate(f"{val:.1f}%", xy=(sev_x[i], val), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(sev_x)
    ax.set_xticklabels(["Clean (0)", "Mild (1)", "Moderate (2)", "Strong (3)"], fontsize=10)
    ax.set_xlabel("Perturbation Severity Level", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{target_pol} Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(45, 102)
    ax.set_title("9C: Distribution Shift Robustness (Accuracy vs. Severity)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    fig.tight_layout()
    p3 = destination / "distribution_shift_accuracy_curves.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_paths.append(str(p3))

    # =========================================================================
    # Figure 4: Distribution-shift compute curves (9C FLOPs)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for s_name, (fmt, col, label) in shift_styles.items():
        s_dict = data_9c.get(s_name, {})
        flops_list = [s_dict.get(sev, {}).get("flops", {}).get(target_pol, 0.0) for sev in severities]
        ax.plot(sev_x, flops_list, fmt, color=col, linewidth=2.0, markersize=7, label=label)
        for i, val in enumerate(flops_list):
            ax.annotate(f"{val:,.0f}", xy=(sev_x[i], val), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(sev_x)
    ax.set_xticklabels(["Clean (0)", "Mild (1)", "Moderate (2)", "Strong (3)"], fontsize=10)
    ax.set_xlabel("Perturbation Severity Level", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{target_pol} Estimated FLOPs", fontsize=11, fontweight="bold")
    ax.set_title("9C: Distribution Shift Compute Impact (FLOPs vs. Severity)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    p4 = destination / "distribution_shift_compute_curves.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_paths.append(str(p4))

    # =========================================================================
    # Figure 5: Mixed-structure expert cross-evaluation (9D)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    data_9d = summary.get("regime_9d_mixed", {})
    families_9d = data_9d.get("families", ["F", "R", "C", "FR", "RC", "FC", "FRC"])
    expert_eval = data_9d.get("expert_cross_eval", {})
    experts_ordered = ["mlp", "graph", "attention", "attention_v2"]
    expert_disp = ["Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2"]

    matrix = np.zeros((len(experts_ordered), len(families_9d)))
    for r_idx, exp in enumerate(experts_ordered):
        for c_idx, fam in enumerate(families_9d):
            matrix[r_idx, c_idx] = expert_eval.get(exp, {}).get(fam, 0.0) * 100

    im = ax.imshow(matrix, cmap="Blues", vmin=40, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(len(families_9d)))
    ax.set_xticklabels(families_9d, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(experts_ordered)))
    ax.set_yticklabels(expert_disp, fontsize=10, fontweight="bold")

    for i in range(len(experts_ordered)):
        for j in range(len(families_9d)):
            val = matrix[i, j]
            txt_color = "white" if val > 75 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")

    ax.set_title("9D: Specialist Performance Across Pure and Mixed Task Families", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p5 = destination / "mixed_structure_expert_cross_evaluation.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_paths.append(str(p5))

    # =========================================================================
    # Figure 6: Mixed-structure router selection (9D)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    router_sel = data_9d.get("router_selection", {})

    sel_matrix = np.zeros((len(families_9d), len(experts_ordered)))
    for r_idx, fam in enumerate(families_9d):
        fam_sel = router_sel.get(fam, {})
        for c_idx, exp in enumerate(experts_ordered):
            sel_matrix[r_idx, c_idx] = fam_sel.get(exp, 0.0) * 100

    im2 = ax.imshow(sel_matrix, cmap="YlGnBu", vmin=0, vmax=100)
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.03, pad=0.04)
    cbar2.set_label("Routing Selection (%)", fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(len(experts_ordered)))
    ax.set_xticklabels(["MLP", "Graph", "Attention V1", "Attention V2"], fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(families_9d)))
    ax.set_yticklabels(families_9d, fontsize=10, fontweight="bold")

    for i in range(len(families_9d)):
        for j in range(len(experts_ordered)):
            val = sel_matrix[i, j]
            txt_col = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_col, fontsize=8.5, fontweight="bold")

    ax.set_title("9D: Balanced Router (λ=0.10) Selection Distribution Across Families", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p6 = destination / "mixed_structure_router_selection.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_paths.append(str(p6))

    # =========================================================================
    # Figure 7: Generalization routing-decision preservation
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    pres_labels: list[str] = []
    pres_vals: list[float] = []

    # From 9B
    dp_9b = data_9b.get("decision_preservation", {})
    for k, name in [("b1_token_perm", "B1: Token Perm"), ("b2_marker_var", "B2: Marker Var"), ("b4_destructive", "B4: Destructive")]:
        if k in dp_9b:
            pres_labels.append(name)
            pres_vals.append(dp_9b[k] * 100)

    # From 9C
    for s_name, s_disp in [("magnitude_scaling", "Mag"), ("additive_noise", "Noise"), ("variance_contrast", "Var")]:
        s_data = data_9c.get(s_name, {})
        for sev in ["mild", "moderate", "strong"]:
            if sev in s_data and "decision_preservation" in s_data[sev]:
                pres_labels.append(f"{s_disp} ({sev[0].upper()})")
                pres_vals.append(s_data[sev]["decision_preservation"] * 100)

    if pres_vals:
        x_pres = np.arange(len(pres_vals))
        bar_colors = ["#4e79a7" if "B" in lbl else "#e15759" for lbl in pres_labels]
        bars = ax.bar(x_pres, pres_vals, color=bar_colors, edgecolor="black", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x_pres)
        ax.set_xticklabels(pres_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Decision Preservation vs Canonical (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Routing Decision Preservation Under Perturbations", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    fig.tight_layout()
    p7 = destination / "generalization_decision_preservation.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_paths.append(str(p7))

    # =========================================================================
    # Figure 8: Accuracy vs FLOPs across generalization regimes
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    pts = summary.get("accuracy_flops_points", [])

    regime_markers = {
        "Canonical Test": "o",
        "Fresh Test (9A)": "s",
        "Token Perm (9B)": "^",
        "Noise Mod (9C)": "D",
        "Mixed Task (9D)": "v",
    }

    plotted_labels = set()
    for pt in pts:
        reg = pt.get("regime", "Canonical Test")
        pol = pt.get("policy", "Balanced (λ=0.10)")
        acc = pt.get("accuracy", 0.0) * 100
        flops = pt.get("flops", 1000.0)
        marker = regime_markers.get(reg, "o")
        color = policy_colors.get(pol, "#1f77b4")

        lbl = f"{pol} ({reg})" if (pol, reg) not in plotted_labels else None
        if lbl:
            plotted_labels.add((pol, reg))
        ax.scatter(flops, acc, color=color, marker=marker, s=80, alpha=0.85, edgecolors="black", linewidth=0.6, label=lbl)

    ax.set_xlabel("Estimated Theoretical FLOPs / Sample", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Empirical Frontier Across Generalization Regimes", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(40, 105)
    # Put legend outside if large
    if len(plotted_labels) <= 10:
        ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    p8 = destination / "accuracy_vs_flops_generalization.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_paths.append(str(p8))

    # =========================================================================
    # Figure 9: Single-expert ceiling for mixed tasks (Failure D Diagnosis)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    mixed_fams = ["FR", "RC", "FC", "FRC"]
    ceilings = data_9d.get("single_expert_ceilings", {})
    r_accs = data_9d.get("router_accuracies", {})

    ceil_vals = [ceilings.get(fam, {}).get("ceiling_accuracy", 0.0) * 100 for fam in mixed_fams]
    best_exps = [ceilings.get(fam, {}).get("best_expert", "None") for fam in mixed_fams]
    rout_vals = [r_accs.get(fam, 0.0) * 100 for fam in mixed_fams]

    x_mix = np.arange(len(mixed_fams))
    width = 0.35

    bars_ceil = ax.bar(x_mix - width / 2, ceil_vals, width, label="Single-Expert Ceiling", color="#4e79a7", alpha=0.9, edgecolor="black")
    bars_rout = ax.bar(x_mix + width / 2, rout_vals, width, label="Balanced Router (λ=0.10)", color="#2ca02c", alpha=0.9, edgecolor="black")

    ax.axhline(100.0, color="#d62728", linestyle="--", linewidth=1.5, label="Composite Task Target (100%)")
    ax.axhline(75.0, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Theoretical Compositional Bound (75%)")

    for idx, b in enumerate(bars_ceil):
        h = b.get_height()
        ax.annotate(f"{h:.1f}%\n({best_exps[idx]})", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
    for b in bars_rout:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_mix)
    ax.set_xticklabels(mixed_fams, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 110)
    ax.set_title("9D: Single-Expert Ceiling & Compositional Limitation (Failure D)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p9 = destination / "single_expert_ceiling_mixed_tasks.png"
    fig.savefig(p9)
    plt.close(fig)
    generated_paths.append(str(p9))

    # =========================================================================
    # Figure 10: Representative latency decomposition
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    lat_data = summary.get("latency_decomposition", {})
    lat_pols = [p for p in ["Fixed MLP", "Fixed Graph", "Fixed Attention V2", "Balanced (λ=0.10)", "Accuracy-first (λ=0.01)"] if p in lat_data]
    if not lat_pols:
        lat_pols = list(lat_data.keys())

    x_lat = np.arange(len(lat_pols))
    overhead = [lat_data.get(p, {}).get("router_overhead_us", 0.0) for p in lat_pols]
    expert_time = [lat_data.get(p, {}).get("expert_exec_us", 0.0) for p in lat_pols]

    ax.bar(x_lat, expert_time, 0.5, label="Expert Execution", color="#4e79a7", edgecolor="black", linewidth=0.7)
    ax.bar(x_lat, overhead, 0.5, bottom=expert_time, label="Router Overhead", color="#e15759", edgecolor="black", linewidth=0.7)

    for i in range(len(lat_pols)):
        tot = overhead[i] + expert_time[i]
        pct = (overhead[i] / tot * 100) if tot > 0 else 0
        ax.annotate(f"{tot:.0f} µs\n({pct:.1f}% ovhd)", xy=(x_lat[i], tot),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold")

    ax.set_xticks(x_lat)
    ax.set_xticklabels(lat_pols, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylabel("Inference Time per Sample (µs)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Latency Decomposition: Router Overhead vs. Expert Execution", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    fig.tight_layout()
    p10 = destination / "representative_latency_decomposition.png"
    fig.savefig(p10)
    plt.close(fig)
    generated_paths.append(str(p10))

    return generated_paths
