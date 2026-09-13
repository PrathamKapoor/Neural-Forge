"""Phase 11 Figures: 12 diagnostic figures for the composition-bottleneck report.

Every figure is data-driven from the `summary` dict produced by
`run_phase11_composition_diagnosis`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
MIXED = ("FR", "RC", "FC", "FRC")
PURE = ("F", "R", "C")


def _pct(x: float) -> float:
    return float(x) * 100.0


def generate_phase11_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    oracle_ceiling = summary.get("oracle_ceiling_per_family", {})
    rep_probe = summary.get("representation_probe_per_expert", {})
    interface = summary.get("interface_per_sequence", {})
    aggregation = summary.get("aggregation_methods", {})
    order = summary.get("order_per_pair", {})
    routing = summary.get("routing_selection_summary", {})
    validation = summary.get("component_validation", {})
    minimal = summary.get("minimal_composition", {})
    repr_dec = summary.get("routing_representation_decodability_mean", {})
    causal = summary.get("causal_diagnosis_aggregate", {})
    phase10_ceiling = summary.get("phase10_ceiling_mean", 0.0)
    phase10_k1 = summary.get("phase10_k1_mixed_mean", 0.0)

    # 1. Oracle composition feasibility (per family)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    x = np.arange(len(FAMILIES))
    ax.bar(x, [_pct(oracle_ceiling.get(f, 0.0)) for f in FAMILIES], color="#59a14f", edgecolor="black")
    for i, v in enumerate([oracle_ceiling.get(f, 0.0) for f in FAMILIES]):
        ax.annotate(f"{_pct(v):.1f}%", (i, _pct(v)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Oracle-Combination Accuracy (%)")
    ax.set_title("11A — Oracle Composition Feasibility (k<=3)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p1 = dest / "oracle_composition_feasibility.png"
    fig.tight_layout(); fig.savefig(p1); plt.close(fig); generated.append(str(p1))

    # 2. Mixed-task ceiling comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = np.arange(len(MIXED))
    w = 0.27
    phase10_ceil_per_fam = [_pct(phase10_ceiling) for _ in MIXED]  # scalar, same bar
    oracle_per_fam = [_pct(oracle_ceiling.get(f, 0.0)) for f in MIXED]
    minimal_per_fam = [_pct(minimal.get(f, 0.0)) for f in MIXED]
    ax.bar(x - w, [_pct(phase10_ceiling)] * 4, w, label=f"Phase 10 single-expert ceiling ({_pct(phase10_ceiling):.1f}%)", color="#7f7f7f", edgecolor="black")
    ax.bar(x, oracle_per_fam, w, label="Oracle k<=3 (11A)", color="#59a14f", edgecolor="black")
    ax.bar(x + w, minimal_per_fam, w, label="Minimal composition (11H)", color="#4e79a7", edgecolor="black")
    for i in range(len(MIXED)):
        ax.annotate(f"{oracle_per_fam[i]:.1f}%", (i, oracle_per_fam[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MIXED, fontweight="bold")
    ax.set_ylabel("Mixed-Task Accuracy (%)")
    ax.set_title("11A vs Phase 10 Ceiling vs Minimal Composition")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p2 = dest / "mixed_task_ceiling_comparison.png"
    fig.tight_layout(); fig.savefig(p2); plt.close(fig); generated.append(str(p2))

    # 3. Representation probe (after_blocks) per expert
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    experts = ("mlp", "graph", "attention_v2")
    width = 0.25
    x = np.arange(len(FAMILIES))
    for i, exp in enumerate(experts):
        vals = []
        for f in FAMILIES:
            v = rep_probe.get(exp, {}).get("after_blocks", {}).get("final_target", 0.0)
            if isinstance(v, dict):
                v = v.get("overall", 0.0)
            vals.append(v)

        ax.bar(x + (i - 1) * width, [_pct(v) for v in vals], width, label=exp, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
    ax.set_ylabel("Probe Accuracy on Final Target (%)")
    ax.set_title("11B — Representation Decodability (after_blocks)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p3 = dest / "representation_probe_results.png"
    fig.tight_layout(); fig.savefig(p3); plt.close(fig); generated.append(str(p3))

    # 4. Interface ablation (per sequence)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    seqs = list(interface.keys())
    if seqs:
        x = np.arange(len(seqs))
        accs = [interface[s].get("overall_mean", 0.0) for s in seqs]
        flops = [interface[s].get("flops", 0.0) for s in seqs]
        ax.bar(x, [_pct(a) for a in accs], color="#4e79a7", edgecolor="black")
        for i, (a, f) in enumerate(zip(accs, flops)):
            ax.annotate(f"{_pct(a):.1f}%\n({f:,.0f} F)", (i, _pct(a)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=20, ha="right")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_title("11C — Interface Compatibility (StateChain)")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p4 = dest / "interface_ablation.png"
    fig.tight_layout(); fig.savefig(p4); plt.close(fig); generated.append(str(p4))

    # 5. Aggregation comparison
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    methods = list(aggregation.keys())
    if methods:
        x = np.arange(len(methods))
        accs = [aggregation[m].get("overall_mean", 0.0) for m in methods]
        ax.bar(x, [_pct(a) for a in accs], color="#59a14f", edgecolor="black")
        for i, a in enumerate(accs):
            ax.annotate(f"{_pct(a):.1f}%", (i, _pct(a)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_title("11D — Aggregation Comparison")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p5 = dest / "aggregation_comparison.png"
    fig.tight_layout(); fig.savefig(p5); plt.close(fig); generated.append(str(p5))

    # 6. Composition order
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    pairs = list(order.keys())
    if pairs:
        x = np.arange(len(pairs))
        w = 0.35
        fwd = [order[p].get("forward_overall_mean", 0.0) for p in pairs]
        rev = [order[p].get("reverse_overall_mean", 0.0) for p in pairs]
        ax.bar(x - w / 2, [_pct(v) for v in fwd], w, label="Forward (A->B)", color="#4e79a7", edgecolor="black")
        ax.bar(x + w / 2, [_pct(v) for v in rev], w, label="Reverse (B->A)", color="#e15759", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=15, ha="right")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_title("11E — Composition Order (Forward vs Reverse)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p6 = dest / "composition_order_comparison.png"
    fig.tight_layout(); fig.savefig(p6); plt.close(fig); generated.append(str(p6))

    # 7. Router composition recovery
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    rec = routing.get("useful_recovery_rate_per_family_mean", {})
    agr = routing.get("agreement_rate_per_family_mean", {})
    if rec:
        x = np.arange(len(FAMILIES))
        w = 0.35
        ax.bar(x - w / 2, [_pct(agr.get(f, 0.0)) for f in FAMILIES], w, label="Agreement (Phase 10 top-2 = Oracle)", color="#4e79a7", edgecolor="black")
        ax.bar(x + w / 2, [_pct(rec.get(f, 0.0)) for f in FAMILIES], w, label="Useful-composition recovery", color="#e15759", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.set_title("11F — Router Composition Recovery")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p7 = dest / "router_composition_recovery.png"
    fig.tight_layout(); fig.savefig(p7); plt.close(fig); generated.append(str(p7))

    # 8. Routing representation separability
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    pf = repr_dec.get("per_family_mean", {})
    if pf:
        x = np.arange(len(FAMILIES))
        accs = [_pct(pf.get(f, 0.0)) for f in FAMILIES]
        ax.bar(x, accs, color="#9467bd", edgecolor="black")
        for i, v in enumerate(accs):
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        ax.axhline(100.0 / 7, color="#d62728", linestyle="--", linewidth=1.2, label="Chance (1/7 = 14.3%)")
        ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontweight="bold")
        ax.set_ylabel("Probe Accuracy on Family Identity (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.set_title("11F — Routing Representation Family Decodability")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p8 = dest / "routing_representation_separability.png"
    fig.tight_layout(); fig.savefig(p8); plt.close(fig); generated.append(str(p8))

    # 9. Mixed-task component validation
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    flip = validation.get("component_flip_rates_mean", {})
    if flip:
        comps = ("F", "R", "C")
        x = np.arange(len(MIXED))
        w = 0.25
        for i, c in enumerate(comps):
            vals = [_pct(flip.get(f, {}).get(c, 0.0)) for f in MIXED]
            ax.bar(x + (i - 1) * w, vals, w, label=f"Component {c}", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(MIXED, fontweight="bold")
        ax.set_ylabel("Target-Flip Rate on Component Removal (%)")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.set_title("11G — Component Load-Bearing (analytical)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p9 = dest / "mixed_task_component_validation.png"
    fig.tight_layout(); fig.savefig(p9); plt.close(fig); generated.append(str(p9))

    # 10. Accuracy vs FLOPs
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    points = [
        ("k=1 (Phase 10)", phase10_k1, 1 * 8112.0 + 1052.0),
    ]
    for seq, vals in interface.items():
        if "+" in seq:  # multi-expert
            points.append((seq, vals.get("overall_mean", 0.0), vals.get("flops", 0.0)))
    for f in FAMILIES:
        if oracle_ceiling.get(f, 0.0) > 0:
            points.append((f"oracle-{f}", oracle_ceiling[f], 2 * 8112.0 + 1052.0))
    for label, acc, flops in points:
        ax.scatter(flops, _pct(acc), s=70, label=label, edgecolors="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical FLOPs / Sample (log)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 11 — Accuracy vs FLOPs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    p10 = dest / "accuracy_vs_flops.png"
    fig.tight_layout(); fig.savefig(p10); plt.close(fig); generated.append(str(p10))

    # 11. Latency decomposition (informational; uses Phase 10 estimates)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    policies = ["k=1", "fixed_k=2", "sequential_2", "adaptive_k_learned"]
    flops = [9164.0, 18124.0, 17040.0, 61972.0]
    colors = ["#4e79a7", "#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(policies, flops, color=colors, edgecolor="black")
    for i, f in enumerate(flops):
        ax.annotate(f"{f:,.0f} F", (i, f), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("FLOPs / sample")
    ax.set_title("Phase 11 — Latency / Compute Decomposition (inherited from Phase 10 estimates)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    p11 = dest / "latency_decomposition.png"
    fig.tight_layout(); fig.savefig(p11); plt.close(fig); generated.append(str(p11))

    # 12. Causal diagnosis summary
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    cats = list(causal.keys())
    if cats:
        # Encode status as numeric score
        score_map = {
            "SUPPORTED": 1.0,
            "PARTIALLY SUPPORTED": 0.5,
            "INCONCLUSIVE": 0.0,
            "NOT TESTED": -0.5,
            "NOT SUPPORTED": -1.0,
        }
        scores = [score_map.get(causal[c].get("status", "INCONCLUSIVE"), 0.0) for c in cats]
        colors = []
        for s in scores:
            if s > 0.5:
                colors.append("#59a14f")
            elif s > 0:
                colors.append("#aec7e8")
            elif s > -0.5:
                colors.append("#ffbb78")
            else:
                colors.append("#d62728")
        ax.barh(cats, scores, color=colors, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["NOT\nSUPPORTED", "NOT\nTESTED", "INCONCLUSIVE", "PARTIALLY\nSUPPORTED", "SUPPORTED"])
        ax.set_title("Phase 11 — Causal Diagnosis Summary")
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    p12 = dest / "causal_diagnosis_summary.png"
    fig.tight_layout(); fig.savefig(p12); plt.close(fig); generated.append(str(p12))

    return generated
