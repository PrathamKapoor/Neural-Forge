"""Phase 14 figures: 14 plots for the readout/head diagnosis report."""
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


def generate_phase14_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    base = summary.get("baseline_perf_mean", {})
    pool = summary.get("pooling_results_mean", {})
    head = summary.get("head_results_mean", {})
    br = summary.get("branch_readout_results_mean", {})
    fusion = summary.get("fusion_results_mean", {})
    bc = summary.get("branch_combination_results_mean", {})
    rel_sens = summary.get("relational_sensitivity_mean", {})
    r_probe = summary.get("r_probe_mean", 0.0)
    old_ceil = summary.get("old_ceiling_mixed_mean", 0.0)
    new_ceil = summary.get("new_ceiling_mixed_mean", 0.0)
    new_ceil_nl = summary.get("new_ceiling_nl_head_mixed_mean", 0.0)
    causal = summary.get("causal_diagnosis_aggregate", {})

    # 1. Representation pipeline (schematic)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.set_axis_off()
    boxes = [
        (0.05, "Input\n[B,S,8]"),
        (0.18, "Encoder\nLinear(8,24)"),
        (0.32, "JointCoBlock\n(3 branches + fusion)"),
        (0.46, "block_output\n[B,S,24]"),
        (0.60, "Pool (P1-P4)\n[B,24]"),
        (0.74, "Head\n(H -> 2)"),
        (0.88, "Logits\n[B,2]"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.10, 0.30, facecolor="#cce5ff", edgecolor="black"))
        ax.text(x + 0.05, 0.5, label, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0], 0.5), xytext=(boxes[i][0] + 0.10, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Phase 14: Representation Pipeline (frozen encoder + diagnostic head)")
    p1 = dest / "representation_pipeline.png"
    fig.tight_layout(); fig.savefig(p1); plt.close(fig); generated.append(str(p1))

    # 2. Pooling comparison (R/RC/FRC)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    pool_labels = list(pool.keys())
    x = np.arange(len(pool_labels))
    w = 0.25
    for i, fam in enumerate(("R", "RC", "FRC")):
        vals = [_pct(pool[k].get(fam, 0.0)) for k in pool_labels]
        ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(pool_labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("14C: Pooling comparison (R/RC/FRC) - frozen encoder + linear head")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p2 = dest / "pooling_comparison.png"
    fig.tight_layout(); fig.savefig(p2); plt.close(fig); generated.append(str(p2))

    # 3. Head comparison
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    head_labels = ("linear", "small_nonlinear")
    x = np.arange(len(head_labels))
    w = 0.25
    for i, fam in enumerate(("R", "RC", "FRC")):
        vals = [_pct(head[k].get(fam, 0.0)) for k in head_labels if isinstance(head.get(k, {}), dict)]
        ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(head_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("14D: Head comparison (R/RC/FRC) - frozen encoder + P1_query pool")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p3 = dest / "head_comparison.png"
    fig.tight_layout(); fig.savefig(p3); plt.close(fig); generated.append(str(p3))

    # 4. Relational-branch readout
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    br_labels = ("feat_only", "rel_only", "ctx_only")
    x = np.arange(len(br_labels))
    w = 0.25
    for i, fam in enumerate(("R", "RC", "FRC", "F")):
        vals = [_pct(br[k].get(fam, 0.0)) for k in br_labels]
        ax.bar(x + (i - 1.5) * w, vals, w, label=fam, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(br_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("14E: Direct branch readout (single-branch head)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p4 = dest / "relational_branch_readout.png"
    fig.tight_layout(); fig.savefig(p4); plt.close(fig); generated.append(str(p4))

    # 5. Branch-combination matrix
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    combo_labels = list(bc.keys())
    if combo_labels:
        x = np.arange(len(combo_labels))
        w = 0.10
        for i, fam in enumerate(FAMILIES):
            vals = [_pct(bc[k].get(fam, 0.0)) for k in combo_labels]
            ax.bar(x + (i - (len(FAMILIES) - 1) / 2) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(combo_labels, rotation=20, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)
        ax.set_title("14F: Branch-combination matrix (each combo = concat of branch mean-pools)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p5 = dest / "branch_combination_matrix.png"
    fig.tight_layout(); fig.savefig(p5); plt.close(fig); generated.append(str(p5))

    # 6. Fusion ablation
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    fusion_labels = list(fusion.keys())
    if fusion_labels:
        x = np.arange(len(fusion_labels))
        w = 0.10
        for i, fam in enumerate(FAMILIES):
            vals = [_pct(fusion[k].get(fam, 0.0)) for k in fusion_labels]
            ax.bar(x + (i - (len(FAMILIES) - 1) / 2) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(fusion_labels, rotation=20, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)
        ax.set_title("14F: Fusion-path diagnosis (each representation -> linear head)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p6 = dest / "fusion_ablation.png"
    fig.tight_layout(); fig.savefig(p6); plt.close(fig); generated.append(str(p6))

    # 7. Relational destruction
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    if rel_sens:
        families_rel = ("R", "RC", "FRC")
        x = np.arange(len(families_rel))
        w = 0.35
        for i, cond in enumerate(("original", "relational_permuted")):
            if cond in rel_sens:
                vals = [_pct(rel_sens[cond].get(f, 0.0)) for f in families_rel]
                ax.bar(x + (i - 0.5) * w, vals, w, label=cond, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(families_rel, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("14F: Relational destruction (small_nonlinear head)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p7 = dest / "relational_destruction.png"
    fig.tight_layout(); fig.savefig(p7); plt.close(fig); generated.append(str(p7))

    # 8. Per-family accuracy (selected conditions)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    conditions = [
        ("Phase 13 baseline", base),
        ("linear head (P1_query)", head.get("linear", {})),
        ("small_nonlinear head", head.get("small_nonlinear", {})),
        ("P2_mean + linear", pool.get("P2_mean", {})),
        ("P3_max + linear", pool.get("P3_max", {})),
        ("P4_mean+query + linear", pool.get("P4_mean_plus_query", {})),
        ("rel_only branch", br.get("rel_only", {})),
        ("branches_concat", fusion.get("branches_concat", {})),
    ]
    x = np.arange(len(FAMILIES))
    w = 0.10
    for i, (label, perf) in enumerate(conditions):
        if not perf:
            continue
        vals = [_pct(perf.get(f, 0.0)) for f in FAMILIES]
        ax.bar(x + (i - (len(conditions) - 1) / 2) * w, vals, w, label=label, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.set_title("14: Per-family accuracy across diagnostic conditions")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p8 = dest / "per_family_accuracy.png"
    fig.tight_layout(); fig.savefig(p8); plt.close(fig); generated.append(str(p8))

    # 9. Probe vs prediction gap
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    if rel_sens:
        labels = ["Phase 13 baseline", "linear head", "small_nonlinear head",
                  "rel_only branch", "branches_concat head"]
        r_vals = [
            _pct(base.get("R", 0.0)),
            _pct(head.get("linear", {}).get("R", 0.0)),
            _pct(head.get("small_nonlinear", {}).get("R", 0.0)),
            _pct(br.get("rel_only", {}).get("R", 0.0)),
            _pct(fusion.get("branches_concat", {}).get("R", 0.0)),
        ]
        x = np.arange(len(labels))
        ax.bar(x, r_vals, color="#4e79a7", edgecolor="black")
        for i, v in enumerate(r_vals):
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.axhline(_pct(r_probe), color="red", linestyle="--", linewidth=1.5,
                   label=f"R-signal probe = {_pct(r_probe):.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("14: Probe vs final R accuracy (probe-vs-prediction gap)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p9 = dest / "probe_vs_prediction_gap.png"
    fig.tight_layout(); fig.savefig(p9); plt.close(fig); generated.append(str(p9))

    # 10. Accuracy vs FLOPs
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    points = [
        ("Phase 13 baseline", base.get("overall", 0.0), 15000),
        ("linear head (P1)", head.get("linear", {}).get("overall", 0.0), 15000),
        ("small_nonlinear head", head.get("small_nonlinear", {}).get("overall", 0.0), 15000),
        ("P2_mean", pool.get("P2_mean", {}).get("overall", 0.0), 15000),
        ("P3_max", pool.get("P3_max", {}).get("overall", 0.0), 15000),
        ("rel_only branch", br.get("rel_only", {}).get("overall", 0.0), 15000),
        ("branches_concat head", fusion.get("branches_concat", {}).get("overall", 0.0), 15000),
    ]
    for label, acc, fl in points:
        ax.scatter(fl, _pct(acc), s=80, label=label, edgecolors="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Encoder FLOPs / sample (log)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Accuracy vs FLOPs (encoder FLOPs dominate; heads are negligible)")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)
    p10 = dest / "accuracy_vs_flops.png"
    fig.tight_layout(); fig.savefig(p10); plt.close(fig); generated.append(str(p10))

    # 11. Latency decomposition
    lat = summary.get("latencies_mean", {})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    if lat:
        labels = list(lat.keys())
        vals = [lat[k] for k in labels]
        ax.bar(labels, vals, color="#59a14f", edgecolor="black")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}us", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylabel("Mean latency (us / sample)")
        ax.set_title("Latency decomposition (frozen encoder + diagnostic head)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p11 = dest / "latency_decomposition.png"
    fig.tight_layout(); fig.savefig(p11); plt.close(fig); generated.append(str(p11))

    # 12. Final bottleneck diagnosis
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
        ax.set_title("Phase 14 — Causal Diagnosis Summary")
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    p12 = dest / "bottleneck_diagnosis.png"
    fig.tight_layout(); fig.savefig(p12); plt.close(fig); generated.append(str(p12))

    # 13. Seed stability
    per_seed = summary.get("per_seed_results", [])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    if per_seed:
        seeds = [r.get("seed", "?") for r in per_seed]
        for label, _key, color in [
            ("baseline R", "baseline_perf_R", "#4e79a7"),
            ("linear head R", "head_linear_R", "#59a14f"),
            ("small_nonlinear R", "head_small_nonlinear_R", "#e15759"),
            ("rel_only R", "branch_rel_only_R", "#9467bd"),
        ]:
            ys = []
            for r in per_seed:
                if label.startswith("baseline"):
                    ys.append(r["baseline_perf"].get("R", 0.0) * 100)
                elif label.startswith("linear"):
                    ys.append(r["head_results"].get("linear", {}).get("R", 0.0) * 100)
                elif label.startswith("small"):
                    ys.append(r["head_results"].get("small_nonlinear", {}).get("R", 0.0) * 100)
                else:
                    ys.append(r["branch_readout_results"].get("rel_only", {}).get("R", 0.0) * 100)
            if ys and len(ys) == len(seeds):
                ax.plot(seeds, ys, "-o", color=color, label=label)
    ax.set_ylabel("R accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Seed stability (R accuracy per condition)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p13 = dest / "seed_stability.png"
    fig.tight_layout(); fig.savefig(p13); plt.close(fig); generated.append(str(p13))

    # 14. Baseline vs best intervention
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    best_pool_label = max(pool.keys(), key=lambda k: pool[k].get("R", 0.0)) if pool else None
    best_head_label = "small_nonlinear" if head.get("small_nonlinear", {}).get("R", 0) >= head.get("linear", {}).get("R", 0) else "linear"
    conditions = [
        ("Phase 13 baseline", base.get("R", 0.0)),
        (f"best pool ({best_pool_label})", pool.get(best_pool_label, {}).get("R", 0.0)) if best_pool_label else ("-", 0),
        (f"{best_head_label} head", head.get(best_head_label, {}).get("R", 0.0)),
        ("rel_only branch", br.get("rel_only", {}).get("R", 0.0)),
    ]
    labels = [c[0] for c in conditions]
    vals = [_pct(c[1]) for c in conditions]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#7f7f7f", "#4e79a7", "#59a14f", "#e15759"], edgecolor="black")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("R accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Baseline vs best diagnostic intervention (R family)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p14 = dest / "baseline_vs_intervention.png"
    fig.tight_layout(); fig.savefig(p14); plt.close(fig); generated.append(str(p14))

    return generated
