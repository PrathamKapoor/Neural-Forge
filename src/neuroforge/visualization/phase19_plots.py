"""Phase 19 figures (pure functions of executed-experiment `summary`)."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pct(x: float) -> float:
    return float(x) * 100.0


def _save(fig: plt.Figure, dest: Path, name: str, generated: list[str]) -> None:
    fig.tight_layout()
    p = dest / name
    fig.savefig(p)
    plt.close(fig)
    generated.append(str(p))


def generate_phase19_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Bug reproduction: original R-align R vs RC/FRC.
    bug = agg.get("bug", {})
    if bug.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["R (intact)", "RC (erased)", "FRC (erased)"],
               [_pct(bug.get("orig_R", 0.0)), _pct(bug.get("orig_RC", 0.0)), _pct(bug.get("orig_FRC", 0.0))],
               color=["#2ca02c", "#d62728", "#d62728"], edgecolor="black")
        ax.set_ylabel("Autocorr-stat vs sr (%)")
        ax.set_ylim(0, 100)
        ax.set_title("19A: Historical erasure reproduced (original)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "bug_reproduction.png", generated)

    # 2. Repaired observability by family.
    obs = agg.get("observability", {})
    if obs.get("tested"):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        groups = [("R", ["r_align_R", "r_align_RC", "r_align_FRC", "r_align_FR"])]
        x = np.arange(4)
        ax.bar(["R", "RC", "FRC", "FR"], [_pct(obs.get(k, 0.0)) for k in groups[0][1]],
               color="#6baed6", edgecolor="black")
        ax.set_ylabel("R-align (%)")
        ax.set_ylim(0, 100)
        ax.set_title("19C: Repaired R observability (ch5)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "observability_r.png", generated)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(4)
        w = 0.35
        ax.bar(x - w / 2, [obs.get(f"f_mag_{f}", 0.0) for f in ("F", "FR", "FC", "FRC")], w,
               label="F offset |mean|", edgecolor="black")
        ax.bar(x + w / 2, [_pct(obs.get(f"c_ret_{f}", 0.0)) / 100 for f in ("C", "RC", "FC", "FRC")], w,
               label="C retrieval (fraction)", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["F/C", "FR/RC", "FC", "FRC"])
        ax.legend(fontsize=8)
        ax.set_title("19C: F/C observability (repaired)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "observability_fc.png", generated)

    # 3. Before/after observability.
    if per_seed and "orig_controls" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        x = np.arange(3)
        w = 0.35
        fams = ["R", "RC", "FRC"]
        orig = [statistics.mean([r["orig_controls"].get(f"r_align_{f}", 0.0) for r in per_seed]) * 100 for f in fams]
        repd = [statistics.mean([r["observability"].get(f"r_align_{f}", 0.0) for r in per_seed]) * 100 for f in fams]
        ax.bar(x - w / 2, orig, w, label="original", edgecolor="black")
        ax.bar(x + w / 2, repd, w, label="repaired", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(fams)
        ax.set_ylabel("R-align (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("19J: Observability before/after repair")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "before_after.png", generated)

    # 4. Counterfactual validity.
    cf = agg.get("counterfactual", {})
    if cf.get("tested"):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        ax.bar(["R-swap valid", "C-swap valid"],
               [cf.get("R_valid_rate", 0.0) * 100, cf.get("C_valid_rate", 0.0) * 100],
               color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Valid rate (%)")
        ax.set_ylim(0, 100)
        ax.set_title("19F: Counterfactual validity (repaired RC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "counterfactual_validity.png", generated)

    # 5. Learnability: rule + MLP bars.
    lb = agg.get("learnability", {})
    if lb.get("tested"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(["rule RC", "MLP R", "MLP RC", "MLP FRC", "stat R"],
               [_pct(lb.get("rule_RC", 0.0)), _pct(lb.get("mlp_R", 0.0)), _pct(lb.get("mlp_RC", 0.0)),
                _pct(lb.get("mlp_FRC", 0.0)), _pct(lb.get("stat_R", 0.0))],
               color="#9e9ac8", edgecolor="black")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("19I: Shallow learnability (repaired)")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "learnability.png", generated)

    # 6. Gates schematic.
    gates = agg.get("gates", {})
    if gates:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        ax.set_axis_off()
        names = sorted(gates.keys())
        for i, g in enumerate(names):
            ok = gates[g].get("passed", False)
            y = 0.93 - i * 0.105
            ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.09,
                                       facecolor="#a1d99b" if ok else "#fdae61", edgecolor="black"))
            ax.text(0.05, y, f"{g}: {'PASS' if ok else 'FAIL'}", va="center", fontsize=9, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("G1–G8 validity gates")
        _save(fig, dest, "validity_gates.png", generated)

    # 7. Hypotheses.
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

    # 8. Seed stability of key metrics.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["observability"].get("r_align_RC", 0.0) * 100 for r in per_seed],
               w, label="R-align RC", edgecolor="black")
        ax.bar(x + w / 2, [r["semantics"].get("rule_RC", 0.0) * 100 for r in per_seed],
               w, label="rule RC", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("Seed stability (key metrics)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    return generated
