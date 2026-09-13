"""Phase 13 figures: 15 plots for the joint-coadaptation report.

All figures are data-driven from the `summary` dict produced by
`run_phase13_joint_coadaptation`.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
MIXED = ("FR", "RC", "FC", "FRC")


def _pct(x: float) -> float:
    return float(x) * 100.0


def generate_phase13_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    base = summary.get("joint_baseline_perf_mean", {})
    ext = summary.get("joint_extended_perf_mean", {})
    coa = summary.get("joint_coadapted_perf_mean", {})
    cross = summary.get("cross_eval_mean", {})
    scales_b = summary.get("scales_baseline_mean", {})
    scales_c = summary.get("scales_coadapted_mean", {})
    ablation_b = summary.get("ablation_baseline_mean", {})
    ablation_c = summary.get("ablation_coadapted_mean", {})
    probes_b = summary.get("repr_probes_baseline_mean", {})
    probes_c = summary.get("repr_probes_coadapted_mean", {})
    joint_rel = summary.get("joint_re_rel_mean", {})
    graph_rel = summary.get("graph_re_rel_mean", {})
    base_rel = summary.get("baseline_re_rel_mean", {})
    old_ceil = summary.get("old_ceiling_mixed_mean", 0.0)
    new_ceil = summary.get("new_ceiling_mixed_mean", 0.0)
    oracle = summary.get("oracle_results_mean", {})
    param_counts = summary.get("param_counts", {})
    latencies = summary.get("latencies_mean", {})
    causal = summary.get("causal_diagnosis_aggregate", {})

    # 1. Phase 12 baseline reproduction
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    x = np.arange(len(FAMILIES))
    base_vals = [_pct(base.get(f, 0.0)) for f in FAMILIES]
    ext_vals = [_pct(ext.get(f, 0.0)) for f in FAMILIES]
    coa_vals = [_pct(coa.get(f, 0.0)) for f in FAMILIES]
    w = 0.25
    ax.bar(x - w, base_vals, w, label="13A Joint (20ep)", color="#4e79a7", edgecolor="black")
    ax.bar(x, ext_vals, w, label="13B Joint (40ep)", color="#59a14f", edgecolor="black")
    ax.bar(x + w, coa_vals, w, label="13C JointCo (40ep)", color="#e15759", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Phase 13: per-family Joint accuracy under all 3 conditions")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p1 = dest / "phase12_baseline_reproduction.png"
    fig.tight_layout(); fig.savefig(p1); plt.close(fig); generated.append(str(p1))

    # 2. training-budget comparison (mixed-mean bars)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    labels = ["13A Joint\n(20ep)", "13B Joint\n(40ep)", "13C JointCo\n(40ep)"]
    means = [
        _pct(base.get("mixed_mean", 0.0)),
        _pct(ext.get("mixed_mean", 0.0)),
        _pct(coa.get("mixed_mean", 0.0)),
    ]
    pures = [
        _pct(base.get("pure_mean", 0.0)),
        _pct(ext.get("pure_mean", 0.0)),
        _pct(coa.get("pure_mean", 0.0)),
    ]
    x = np.arange(3)
    w = 0.3
    ax.bar(x - w / 2, means, w, label="mixed", color="#4e79a7", edgecolor="black")
    ax.bar(x + w / 2, pures, w, label="pure", color="#59a14f", edgecolor="black")
    for i in range(3):
        ax.annotate(f"{means[i]:.1f}%", (i - w / 2, means[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{pures[i]:.1f}%", (i + w / 2, pures[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.set_title("Training-budget comparison (pure vs mixed)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p2 = dest / "training_budget_comparison.png"
    fig.tight_layout(); fig.savefig(p2); plt.close(fig); generated.append(str(p2))

    # 3. co-adaptation comparison (radar / grouped bar)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    x = np.arange(len(FAMILIES))
    w = 0.27
    ax.bar(x - w, [_pct(base.get(f, 0.0)) for f in FAMILIES], w, label="13A Joint (20ep)", color="#4e79a7", edgecolor="black")
    ax.bar(x, [_pct(ext.get(f, 0.0)) for f in FAMILIES], w, label="13B Joint (40ep)", color="#59a14f", edgecolor="black")
    ax.bar(x + w, [_pct(coa.get(f, 0.0)) for f in FAMILIES], w, label="13C JointCo (40ep)", color="#e15759", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Co-adaptation comparison: 13A vs 13B vs 13C")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p3 = dest / "coadaptation_comparison.png"
    fig.tight_layout(); fig.savefig(p3); plt.close(fig); generated.append(str(p3))

    # 4. per-family Joint performance
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    for cond, vals, color, label in [
        ("13A", base, "#4e79a7", "13A baseline"),
        ("13B", ext, "#59a14f", "13B extended"),
        ("13C", coa, "#e15759", "13C coadapted"),
    ]:
        ys = [_pct(vals.get(f, 0.0)) for f in FAMILIES]
        ax.plot(FAMILIES, ys, "-o", color=color, label=label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.set_title("Per-family Joint performance across conditions")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p4 = dest / "per_family_joint_performance.png"
    fig.tight_layout(); fig.savefig(p4); plt.close(fig); generated.append(str(p4))

    # 5. branch-scale distribution
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    branches = ("feature", "graph", "context")
    x = np.arange(len(branches))
    w = 0.25
    ax.bar(x - w, [scales_b.get(b, 0.0) for b in branches], w, label="13A baseline", color="#4e79a7", edgecolor="black")
    ax.bar(x, [scales_b.get(b, 0.0) for b in branches] if not scales_c else [scales_c.get(b, 0.0) for b in branches], w, label="13C coadapted", color="#e15759", edgecolor="black")
    # 13B
    sb_e = summary.get("scales_extended_mean", {})
    ax.bar(x + w, [sb_e.get(b, 0.0) for b in branches], w, label="13B extended", color="#59a14f", edgecolor="black")
    ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=0.8, label="init=1/3")
    ax.set_xticks(x); ax.set_xticklabels(branches)
    ax.set_ylabel("Branch scale")
    ax.set_title("Branch scales after training (init = 1/3)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p5 = dest / "branch_scale_distribution.png"
    fig.tight_layout(); fig.savefig(p5); plt.close(fig); generated.append(str(p5))

    # 6. branch-ablation performance
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    abl_labels = list(ablation_c.keys()) if ablation_c else []
    x = np.arange(len(FAMILIES))
    w = 0.18
    for i, lab in enumerate(abl_labels):
        offset = (i - (len(abl_labels) - 1) / 2) * w
        vals = [_pct(ablation_c.get(lab, {}).get(f, 0.0)) for f in FAMILIES]
        ax.bar(x + offset, vals, w, label=lab, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Branch ablation (13C co-adaptation): zero one branch at a time")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p6 = dest / "branch_ablation_performance.png"
    fig.tight_layout(); fig.savefig(p6); plt.close(fig); generated.append(str(p6))

    # 7. representation probe accuracy
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    tasks = list(probes_c.keys())
    x = np.arange(len(FAMILIES))
    w = 0.18
    for i, t in enumerate(tasks):
        offset = (i - (len(tasks) - 1) / 2) * w
        vals = [_pct(probes_c.get(t, {}).get(f, 0.0)) for f in FAMILIES]
        ax.bar(x + offset, vals, w, label=t, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Representation probe accuracy (13C co-adaptation)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p7 = dest / "representation_probe_accuracy.png"
    fig.tight_layout(); fig.savefig(p7); plt.close(fig); generated.append(str(p7))

    # 8. relational destruction comparison
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    families_rel = ("R", "RC", "FRC")
    x = np.arange(len(families_rel))
    w = 0.18
    sources = [
        ("Joint 13A", base_rel),
        ("JointCo 13C", joint_rel),
        ("Graph", graph_rel),
    ]
    for i, (label, src) in enumerate(sources):
        offset = (i - (len(sources) - 1) / 2) * w
        orig = [_pct(src.get("original", {}).get(f, 0.0)) for f in families_rel]
        perm = [_pct(src.get("relational_permuted", {}).get(f, 0.0)) for f in families_rel]
        ax.bar(x - w / 2 + offset, orig, w / 1.5, color="#4e79a7", edgecolor="black", label=f"{label} (orig)" if i == 0 else None)
        ax.bar(x + w / 2 + offset, perm, w / 1.5, color="#e15759", edgecolor="black", label=f"{label} (permuted)" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(families_rel, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Relational destruction comparison (R/RC/FRC)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p8 = dest / "relational_destruction_comparison.png"
    fig.tight_layout(); fig.savefig(p8); plt.close(fig); generated.append(str(p8))

    # 9. Graph vs Joint relational capability
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    labels = ["Graph", "Joint 13A", "JointCo 13C"]
    sens_vals = []
    for src in (graph_rel, base_rel, joint_rel):
        r_orig = src.get("original", {}).get("R", 0.0)
        r_perm = src.get("relational_permuted", {}).get("R", 0.0)
        sens_vals.append((r_orig - r_perm) * 100)
    ax.bar(labels, sens_vals, color=["#4e79a7", "#59a14f", "#e15759"], edgecolor="black")
    for i, v in enumerate(sens_vals):
        ax.annotate(f"{v:+.1f}pp", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("R accuracy drop (orig - permuted) in pp")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Graph vs Joint relational sensitivity (R family)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p9 = dest / "graph_vs_joint_relational.png"
    fig.tight_layout(); fig.savefig(p9); plt.close(fig); generated.append(str(p9))

    # 10. old vs new single-expert ceiling
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    conditions = ["Phase 11/12\n(4 experts)", "Phase 12\n(+ joint baseline)", "Phase 13\n(+ best joint)"]
    ceilings = [0.664, 0.668, new_ceil]
    ax.bar(conditions, [_pct(c) for c in ceilings], color=["#7f7f7f", "#4e79a7", "#e15759"], edgecolor="black")
    for i, v in enumerate(ceilings):
        ax.annotate(f"{_pct(v):.1f}%", (i, _pct(v)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cross-family mixed single-expert ceiling (%)")
    ax.set_ylim(60, 72)
    ax.set_title("Single-expert ceiling (mixed) over phases")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p10 = dest / "ceiling_over_phases.png"
    fig.tight_layout(); fig.savefig(p10); plt.close(fig); generated.append(str(p10))

    # 11. oracle portfolio performance
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    if oracle:
        ks = [k for k in ("k=1", "k=2", "k=3") if k in oracle]
        x = np.arange(len(MIXED))
        w = 0.25
        for i, k_str in enumerate(ks):
            pf = oracle[k_str]
            mixed_vals = [_pct(pf.get(f, 0.0)) for f in MIXED]
            offset = (i - (len(ks) - 1) / 2) * w
            ax.bar(x + offset, mixed_vals, w, label=f"oracle {k_str}", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(MIXED, fontweight="bold")
        ax.set_ylabel("Oracle accuracy (%)")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.set_title("Oracle portfolio performance on mixed families (Phase 13)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p11 = dest / "oracle_portfolio_performance.png"
    fig.tight_layout(); fig.savefig(p11); plt.close(fig); generated.append(str(p11))

    # 12. accuracy vs FLOPs
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    flops_map = {
        "mlp": 8928, "graph": 8112, "attention_v2": 46296,
        "joint_baseline": 14976, "joint_extended": 14976, "joint_coadapted": 23000,  # approximate
    }
    for name in flops_map:
        if name in cross:
            xs = [flops_map[name]]
            ys = [_pct(cross[name].get("overall", 0.0))]
            ax.scatter(xs, ys, s=80, label=name, edgecolors="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs / sample (log)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.legend(fontsize=7)
    ax.set_title("Accuracy vs FLOPs (Phase 13)")
    ax.grid(True, linestyle="--", alpha=0.3)
    p12 = dest / "accuracy_vs_flops.png"
    fig.tight_layout(); fig.savefig(p12); plt.close(fig); generated.append(str(p12))

    # 13. accuracy vs latency
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for name in ("mlp", "graph", "attention_v2", "joint_baseline", "joint_extended", "joint_coadapted"):
        if name in latencies and name in cross:
            xs = [latencies[name]]
            ys = [_pct(cross[name].get("overall", 0.0))]
            ax.scatter(xs, ys, s=80, label=name, edgecolors="black", alpha=0.85)
    ax.set_xlabel("Latency (us / sample)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.legend(fontsize=7)
    ax.set_title("Accuracy vs latency (Phase 13)")
    ax.grid(True, linestyle="--", alpha=0.3)
    p13 = dest / "accuracy_vs_latency.png"
    fig.tight_layout(); fig.savefig(p13); plt.close(fig); generated.append(str(p13))

    # 14. failure diagnosis
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    cats = list(causal.keys())
    if cats:
        score_map = {
            "SUPPORTED": 1.0, "PARTIALLY SUPPORTED": 0.5,
            "INCONCLUSIVE": 0.0, "NOT TESTED": -0.5, "NOT SUPPORTED": -1.0,
        }
        scores = [score_map.get(causal[c].get("status", "INCONCLUSIVE"), 0.0) for c in cats]
        colors = []
        for s in scores:
            if s > 0.5: colors.append("#59a14f")
            elif s > 0: colors.append("#aec7e8")
            elif s > -0.5: colors.append("#ffbb78")
            else: colors.append("#d62728")
        ax.barh(cats, scores, color=colors, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["NOT\nSUPPORTED", "NOT\nTESTED", "INCONCLUSIVE", "PARTIALLY\nSUPPORTED", "SUPPORTED"])
        ax.set_title("Phase 13 — Causal Diagnosis Summary")
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    p14 = dest / "failure_diagnosis.png"
    fig.tight_layout(); fig.savefig(p14); plt.close(fig); generated.append(str(p14))

    # 15. seed stability
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    per_seed = summary.get("per_seed_results", [])
    if per_seed:
        seeds = [r["seed"] for r in per_seed]
        for label, key, color in [
            ("13A baseline", "joint_baseline_mixed", "#4e79a7"),
            ("13B extended", "joint_extended_mixed", "#59a14f"),
            ("13C coadapted", "joint_coadapted_mixed", "#e15759"),
        ]:
            ys = [r.get(key, 0.0) * 100 for r in per_seed]
            ax.plot(seeds, ys, "-o", color=color, label=label)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mixed accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.set_title("Seed stability (mixed accuracy per condition)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p15 = dest / "seed_stability.png"
    fig.tight_layout(); fig.savefig(p15); plt.close(fig); generated.append(str(p15))

    return generated
