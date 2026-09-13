"""Phase 15 figures for the relational-substep diagnosis report.

All figures are pure functions of `summary` (executed-experiment data).
Figures for gated-off sections are skipped (no placeholders).
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


def _pct(x: float) -> float:
    return float(x) * 100.0


def _save(fig: plt.Figure, dest: Path, name: str, generated: list[str]) -> None:
    fig.tight_layout()
    p = dest / name
    fig.savefig(p)
    plt.close(fig)
    generated.append(str(p))


def generate_phase15_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    base = summary.get("baseline_perf_mean", {})
    depth = summary.get("depth_results_mean", {})
    aggr = summary.get("aggregation_results_mean", {})
    cap = summary.get("capacity_results_mean", {})
    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Depth ablation (R/RC/FRC grouped bars).
    labels = [c for c in ("baseline", "depth2", "depth3") if c in depth]
    if labels:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(len(labels))
        w = 0.25
        for i, fam in enumerate(("R", "RC", "FRC")):
            vals = [_pct(depth[c].get(fam, 0.0)) for c in labels]
            ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("15B: Message-passing depth ablation (R/RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "depth_ablation.png", generated)

    # 2. Per-family accuracy: baseline vs candidate.
    cand = summary.get("candidate_cond", "depth2")
    cand_perf = depth.get(cand, {}) or aggr.get(cand, {}) or cap.get(cand, {})
    if base and cand_perf:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        x = np.arange(len(FAMILIES))
        w = 0.35
        ax.bar(x - w / 2, [_pct(base.get(f, 0.0)) for f in FAMILIES], w, label="baseline", edgecolor="black")
        ax.bar(x + w / 2, [_pct(cand_perf.get(f, 0.0)) for f in FAMILIES], w, label=f"candidate ({cand})", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(list(FAMILIES))
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("Per-family accuracy: baseline vs candidate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "per_family_accuracy.png", generated)

    # 3. Aggregation comparison.
    alabels = [c for c in ("agg_sum", "agg_max") if c in aggr]
    if alabels:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(len(alabels) + 1)
        all_l = ["baseline"] + alabels
        w = 0.25
        for i, fam in enumerate(("R", "RC", "FRC")):
            vals = [_pct((depth.get(c, {}) if c == "baseline" else aggr.get(c, {})).get(fam, 0.0)) for c in all_l]
            ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(all_l, rotation=12)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("15C: Aggregation diagnosis (R/RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "aggregation_comparison.png", generated)

    # 4. Capacity comparison.
    clabels = [c for c in ("cap_mlp", "cap_control_featwide") if c in cap]
    if clabels:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(len(clabels) + 1)
        all_l = ["baseline"] + clabels
        w = 0.25
        for i, fam in enumerate(("R", "RC", "FRC")):
            vals = [_pct((depth.get(c, {}) if c == "baseline" else cap.get(c, {})).get(fam, 0.0)) for c in all_l]
            ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(all_l, rotation=12)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("15D: Update-capacity diagnosis (R/RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "capacity_comparison.png", generated)

    # 5. Relational destruction drops (baseline vs candidate).
    topo = agg.get("topology", {})
    if topo.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        cats = ["baseline", f"candidate\n({topo.get('candidate', '?')})"]
        drops = [_pct(topo.get("baseline_drop_R", 0.0)), _pct(topo.get("candidate_drop_R", 0.0))]
        ax.bar(cats, drops, color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Destruction drop on R (pp)")
        ax.set_title("15E: Relational-destruction dependence (R)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "relational_destruction.png", generated)

    # 6. Probe vs prediction gap (grouped probe/final bars).
    gap = agg.get("gap", {})
    if gap.get("tested"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        cats = ["baseline", "candidate"]
        probes = [_pct(gap.get("baseline_probe_R_mean", 0.0)), _pct(gap.get("candidate_probe_R_mean", 0.0))]
        finals = [_pct(gap.get("baseline_final_R_mean", 0.0)), _pct(gap.get("candidate_final_R_mean", 0.0))]
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, probes, w, label="R probe", color="#6baed6", edgecolor="black")
        ax.bar(x + w / 2, finals, w, label="final R", color="#fd8d3c", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("15H: Representation-to-prediction gap (R)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "probe_vs_prediction_gap.png", generated)

    # 7. Branch-combination matrix (candidate branch7, R family).
    branch_rows = []
    for r in per_seed:
        key = f"{cand}_eval" if any(f"{cand}_eval" in r for r in [r]) else "baseline_eval"
        if key in r:
            branch_rows.append(r[key].get("branch7", {}))
    if branch_rows:
        combos = list(branch_rows[0].keys())
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        x = np.arange(len(combos))
        means = [statistics.mean([br[c].get("R", 0.0) for br in branch_rows]) * 100 for c in combos]
        ax.bar(x, means, color="#9e9ac8", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=20, ha="right")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"15G: Branch-combination matrix (R, {cand})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "branch_combination_matrix.png", generated)

    # 8. Seed stability (R per seed, baseline vs candidate).
    if per_seed:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        bvals = [r.get("baseline_eval", {}).get("perf", {}).get("R", 0.0) * 100 for r in per_seed]
        cvals = [r.get(f"{cand}_eval", {}).get("perf", {}).get("R", 0.0) * 100 for r in per_seed]
        ax.bar(x - w / 2, bvals, w, label="baseline R", edgecolor="black")
        ax.bar(x + w / 2, cvals, w, label=f"candidate R ({cand})", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("Seed stability (R per seed)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    # 9. Accuracy vs params.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for cond in ("baseline", "depth2", "depth3", "agg_sum", "agg_max", "cap_mlp", "cap_control_featwide"):
            key = f"{cond}_eval"
            rr = [r[key]["perf"].get("R", 0.0) * 100 for r in per_seed if key in r]
            pp = [r[key].get("params", 0) for r in per_seed if key in r]
            if rr and pp:
                ax.scatter(pp, rr, label=cond, s=60, edgecolors="black")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("R accuracy (%)")
        ax.legend(fontsize=8)
        ax.set_title("Accuracy vs parameters (R)")
        ax.grid(True, linestyle="--", alpha=0.3)
        _save(fig, dest, "accuracy_vs_params.png", generated)

    # 10. Latency decomposition.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        conds = [c for c in ("baseline", "depth2", "depth3", "agg_sum", "agg_max", "cap_mlp") if any(f"{c}_eval" in r for r in per_seed)]
        means = [statistics.mean([r[f"{c}_eval"].get("latency_us", 0.0) for r in per_seed if f"{c}_eval" in r]) for c in conds]
        ax.bar(conds, means, color="#a1d99b", edgecolor="black")
        ax.set_ylabel("Latency (us/sample)")
        ax.set_title("Total latency per condition")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "latency_decomposition.png", generated)

    # 11. Ceiling comparison.
    ceil = agg.get("ceiling", {})
    if ceil.get("tested"):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        ax.bar(["old ceiling", "new ceiling"],
               [_pct(ceil.get("old_ceiling_mixed_mean", 0.0)), _pct(ceil.get("new_ceiling_mixed_mean", 0.0))],
               color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Mixed-mean ceiling (%)")
        ax.set_title("15K: Empirical single-expert ceiling")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "ceiling_comparison.png", generated)

    # 11b. Standalone Graph comparison (15I).
    graph_rows = [r.get("graph_eval") for r in per_seed if "graph_eval" in r]
    if graph_rows and base:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(3)
        w = 0.25
        series = {
            "baseline": [_pct(base.get(f, 0.0)) for f in ("R", "RC", "FRC")],
            f"candidate ({cand})": [_pct(cand_perf.get(f, 0.0)) for f in ("R", "RC", "FRC")] if cand_perf else [0, 0, 0],
            "graph standalone": [_pct(statistics.mean([g["perf"].get(f, 0.0) for g in graph_rows])) for f in ("R", "RC", "FRC")],
        }
        for i, (label, vals) in enumerate(series.items()):
            ax.bar(x + (i - 1) * w, vals, w, label=label, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["R", "RC", "FRC"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("15I: Standalone Graph vs JointCo relational path")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "graph_comparison.png", generated)

    # 12. Hypothesis verdict schematic.
    if hyps:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        ax.set_axis_off()
        names = list(hyps.keys())
        colors = {"SUPPORTED": "#a1d99b", "PARTIALLY SUPPORTED": "#fee08b",
                  "NOT SUPPORTED": "#fdae61", "INCONCLUSIVE": "#d9d9d9", "NOT TESTED": "#e0e0e0"}
        for i, h in enumerate(names):
            st = hyps[h].get("status", "?")
            y = 0.92 - i * 0.105
            ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.09, facecolor=colors.get(st, "#ffffff"), edgecolor="black"))
            ax.text(0.05, y, f"{h}: {st}", va="center", fontsize=9, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("H1–H8 verdicts")
        _save(fig, dest, "hypothesis_verdicts.png", generated)

    return generated
