"""Phase 20 figures (pure functions of executed-experiment `summary`)."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
EXPERTS = ("mlp", "graph", "attention", "attention_v2", "joint", "joint_co", "depth3")


def _pct(x: float) -> float:
    return float(x) * 100.0


def _save(fig: plt.Figure, dest: Path, name: str, generated: list[str]) -> None:
    fig.tight_layout()
    p = dest / name
    fig.savefig(p)
    plt.close(fig)
    generated.append(str(p))


def generate_phase20_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    cross = summary.get("cross_mean", {})
    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Cross-evaluation heatmap (expert × family, R/RC/FRC focus + full).
    if cross:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        fams = list(FAMILIES)
        mat = np.array([[cross.get(e, {}).get(f, 0.0) * 100 for f in fams] for e in EXPERTS])
        im = ax.imshow(mat, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams)
        ax.set_yticks(range(len(EXPERTS)))
        ax.set_yticklabels(EXPERTS)
        for i in range(len(EXPERTS)):
            for j in range(len(fams)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        ax.set_title("20D: Expert × family matrix (repaired)")
        _save(fig, dest, "cross_matrix.png", generated)

    # 2. Ceiling: single vs portfolio.
    ceil = agg.get("ceiling", {})
    if ceil.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["single ceiling", "portfolio best"],
               [_pct(ceil.get("single_mixed", 0.0)), _pct(ceil.get("portfolio_mixed", 0.0))],
               color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Mixed-mean (%)")
        ax.set_ylim(0, 100)
        ax.set_title("20E: Empirical ceiling (repaired)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "ceiling.png", generated)

    # 3. Oracle k=1..3 on R/RC/FRC.
    if per_seed and "oracle" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        x = np.arange(3)
        w = 0.25
        for i, fam in enumerate(("R", "RC", "FRC")):
            vals = [statistics.mean([r["oracle"][f"k={k}"].get("per_family", {}).get(fam, 0.0)
                                               for r in per_seed]) * 100 for k in (1, 2, 3)]
            ax.bar(x + (i - 1) * w, vals, w, label=fam, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["k=1", "k=2", "k=3"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("20F: Oracle top-k (R/RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "oracle_k.png", generated)

    # 4. Compute-aware sweep (accuracy vs λ + latency).
    if per_seed and "sweep" in per_seed[0]:
        lams = sorted(per_seed[0]["sweep"].keys(), key=lambda s: float(s.split("=")[1]))
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(len(lams))
        w = 0.35
        ax.bar(x - w / 2, [statistics.mean([r["sweep"][l].get("mixed_mean", 0.0) for r in per_seed]) * 100
                           for l in lams], w, label="mixed", edgecolor="black")
        ax.bar(x + w / 2, [statistics.mean([r["sweep"][l].get("RC", 0.0) for r in per_seed]) * 100
                           for l in lams], w, label="RC", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([l.split("=")[1] for l in lams])
        ax.set_xlabel("lambda")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("20J: Compute-aware sweep")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "compute_aware.png", generated)

    # 5. Routing utilization.
    if per_seed and "routing_diag" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        names = list(EXPERTS)
        counts = [sum(r["routing_diag"]["utilization"].get(n, 0) for r in per_seed) for n in names]
        ax.bar(names, counts, color="#9e9ac8", edgecolor="black")
        ax.set_ylabel("Samples routed")
        ax.set_title("20I: Router utilization (λ=0)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "routing_utilization.png", generated)

    # 6. RC/FRC primary: best single vs best composition vs router.
    if cross:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(2)
        w = 0.22
        series = {
            "best single": [max(cross[e].get("RC", 0.0) for e in EXPERTS) * 100,
                            max(cross[e].get("FRC", 0.0) for e in EXPERTS) * 100],
            "router": [statistics.mean([r["learned_routing"].get("RC", 0.0) for r in per_seed]) * 100 if per_seed else 0,
                       statistics.mean([r["learned_routing"].get("FRC", 0.0) for r in per_seed]) * 100 if per_seed else 0],
        }
        for i, (lab, vals) in enumerate(series.items()):
            ax.bar(x + (i - 0.5) * w, vals, w, label=lab, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["RC", "FRC"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("20L: RC/FRC — single vs routed")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "rc_frc_primary.png", generated)

    # 7. Agree/disagree on repaired RC.
    if per_seed and "semantics" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["semantics"]["agree_acc"] * 100 for r in per_seed], w, label="agree", edgecolor="black")
        ax.bar(x + w / 2, [r["semantics"]["disagree_acc"] * 100 for r in per_seed], w, label="disagree", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Official RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("20K: Agree vs disagree (repaired RC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "agree_disagree.png", generated)

    # 8. Destruction drops.
    if per_seed and "destruction" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        conds = ["graph", "joint_co"]
        drops = []
        for c in conds:
            vals = []
            for r in per_seed:
                d = r["destruction"].get(c, {})
                vals.append(d.get("original", {}).get("R", 0.0) - d.get("relational_permuted", {}).get("R", 0.0))
            drops.append(statistics.mean(vals) * 100)
        ax.bar(conds, drops, color="#17becf", edgecolor="black")
        ax.set_ylabel("Destruction drop on R (pp)")
        ax.set_title("20N: Causal relational sensitivity")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "destruction.png", generated)

    # 9. Hypotheses.
    if hyps:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        ax.set_axis_off()
        colors = {"SUPPORTED": "#a1d99b", "PARTIALLY SUPPORTED": "#fee08b",
                  "NOT SUPPORTED": "#fdae61", "INCONCLUSIVE": "#d9d9d9", "NOT TESTED": "#e0e0e0"}
        for i, h in enumerate(sorted(hyps.keys())):
            stt = hyps[h].get("status", "?")
            y = 0.93 - i * 0.105
            ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.09,
                                       facecolor=colors.get(stt, "#ffffff"), edgecolor="black"))
            ax.text(0.05, y, f"{h}: {stt}", va="center", fontsize=9, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("H1–H8 verdicts")
        _save(fig, dest, "hypothesis_verdicts.png", generated)

    # 10. Seed stability (mixed per expert).
    if per_seed:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.11
        for i, e in enumerate(EXPERTS):
            vals = [r["cross"][e].get("mixed_mean", 0.0) * 100 for r in per_seed]
            ax.bar(x + (i - 3) * w, vals, w, label=e, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mixed-mean (%)")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("Seed stability (mixed per expert)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    return generated
