"""Phase 17 figures (pure functions of executed-experiment `summary`)."""
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


def generate_phase17_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    base = summary.get("baseline_perf_mean", {})
    d3 = summary.get("depth3_perf_mean", {})
    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Stage × component probe matrix.
    pres = agg.get("preservation", {}).get("matrix_mean", {})
    if pres:
        stages = list(pres.keys())
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        x = np.arange(len(stages))
        w = 0.2
        for i, t in enumerate(("F_signal", "R_signal", "C_signal", "final_target")):
            ax.bar(x + (i - 1.5) * w, [_pct(pres[s].get(t, 0.0)) for s in stages], w, label=t, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=15, ha="right")
        ax.set_ylabel("Probe accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("17B/17C: Stage × component probe matrix")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "stage_probe_matrix.png", generated)

    # 2. Oracle vs fused head (R/RC/FRC).
    suff = agg.get("sufficiency", {})
    if suff.get("tested"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        x = np.arange(3)
        w = 0.3
        # recover fused RC/FRC from per-seed means via combo context is unavailable here;
        # plot oracle absolute + gains over fused head
        ax.bar(["oracle R", "oracle RC", "oracle FRC"],
               [_pct(suff.get("oracle_R", 0.0)), _pct(suff.get("oracle_RC", 0.0)), _pct(suff.get("oracle_FRC", 0.0))],
               color="#6baed6", edgecolor="black", label="oracle (F+R+C)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("17E: Oracle diagnostic readout (R/RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "oracle_readout.png", generated)

    # 3. Branch scales.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        branches = ["feat_delta", "rel_delta", "ctx_delta"]
        rms = {b: statistics.mean([r["scale_stats"][b]["rms"] for r in per_seed]) for b in branches}
        ax.bar(branches, [rms[b] for b in branches], color="#fd8d3c", edgecolor="black")
        ax.set_ylabel("RMS activation")
        ax.set_title("17F: Branch magnitude (RMS)")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "branch_scales.png", generated)

    # 4. Interaction gains.
    inter = agg.get("interaction", {}).get("pairs", {})
    if inter:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        keys = sorted(inter.keys())
        x = np.arange(len(keys))
        ax.bar(x, [inter[k]["gain_mean"] * 100 for k in keys],
               color=["#d62728" if inter[k]["gain_mean"] < 0 else "#2ca02c" for k in keys], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Interaction gain (pp)")
        ax.set_title("17G: Pair interaction gains (pair − best single)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "branch_interactions.png", generated)

    # 5. Composition order.
    order = agg.get("order", {})
    if order.get("tested"):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        labels = list(order["means_R"].keys())
        ax.bar(labels, [_pct(order["means_R"][l]) for l in labels], color="#9e9ac8", edgecolor="black")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("17H: Composition order (frozen chains, R)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "composition_order.png", generated)

    # 6. Joint decodability R-only vs R+C vs F+R+C on RC/FRC.
    if per_seed and "combo_evals" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        labels = ["R", "R+C", "F+R+C"]
        x = np.arange(len(labels))
        w = 0.35
        rc = [statistics.mean([r["combo_evals"][l].get("RC", 0.0) for r in per_seed]) * 100 for l in labels]
        frc = [statistics.mean([r["combo_evals"][l].get("FRC", 0.0) for r in per_seed]) * 100 for l in labels]
        ax.bar(x - w / 2, rc, w, label="RC", edgecolor="black")
        ax.bar(x + w / 2, frc, w, label="FRC", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("17J: Joint decodability (RC/FRC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "joint_decodability.png", generated)

    # 7. Frozen oracle vs production.
    if per_seed and "combo_evals" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(3)
        w = 0.25
        series = {
            "oracle": [statistics.mean([r["combo_evals"]["F+R+C"].get(f, 0.0) for r in per_seed]) * 100 for f in ("R", "RC", "FRC")],
            "fused-diag": [statistics.mean([r["fused_head_eval"].get(f, 0.0) for r in per_seed]) * 100 for f in ("R", "RC", "FRC")],
            "production": [_pct(d3.get(f, 0.0)) for f in ("R", "RC", "FRC")],
        }
        for i, (lab, vals) in enumerate(series.items()):
            ax.bar(x + (i - 1) * w, vals, w, label=lab, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["R", "RC", "FRC"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("17K: Frozen diagnostic vs production composition")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "frozen_vs_production.png", generated)

    # 8. RC/FRC primary criteria.
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    groups = [("baseline", base.get("RC", 0.0), base.get("FRC", 0.0)),
              ("depth3", d3.get("RC", 0.0), d3.get("FRC", 0.0))]
    if per_seed and "combo_evals" in per_seed[0]:
        groups.append(("oracle", statistics.mean([r["combo_evals"]["F+R+C"].get("RC", 0.0) for r in per_seed]),
                       statistics.mean([r["combo_evals"]["F+R+C"].get("FRC", 0.0) for r in per_seed])))
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w / 2, [_pct(g[1]) for g in groups], w, label="RC", edgecolor="black")
    ax.bar(x + w / 2, [_pct(g[2]) for g in groups], w, label="FRC", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("17M: RC/FRC primary criteria")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save(fig, dest, "rc_frc_primary.png", generated)

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

    # 10. Seed stability (RC).
    if per_seed:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.25
        bvals = [r["baseline_eval"]["perf"].get("RC", 0.0) * 100 for r in per_seed]
        dvals = [r["depth3_eval"]["perf"].get("RC", 0.0) * 100 for r in per_seed]
        ovals = [r["combo_evals"]["F+R+C"].get("RC", 0.0) * 100 for r in per_seed] if "combo_evals" in per_seed[0] else [0] * len(seeds)
        ax.bar(x - w, bvals, w, label="baseline RC", edgecolor="black")
        ax.bar(x, dvals, w, label="depth3 RC", edgecolor="black")
        ax.bar(x + w, ovals, w, label="oracle RC", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("Seed stability (RC per seed)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    return generated
