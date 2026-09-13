"""Phase 21 figures (pure functions of executed-experiment `summary`)."""
from __future__ import annotations

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


def generate_phase21_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})
    per_seed = summary.get("per_seed_results", [])
    cap = agg.get("capacity", {})

    # 1. RC vs depth (means + per-seed).
    depths = ["depth1", "depth2", "depth3"]
    have = [d for d in depths if d in cap.get("per_depth_RC", {})]
    if have:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        x = np.arange(len(have))
        ax.bar(x, [_pct(cap["per_depth_RC"][d]) for d in have], color="#6baed6", edgecolor="black")
        for r in per_seed:
            vals = [r.get("capacity", {}).get("depths", {}).get(d, {}).get("RC", float("nan")) * 100 for d in have]
            ax.plot(x, vals, "o-", color="black", alpha=0.5, markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(have)
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("H1: RC vs relational depth (lines = seeds)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "capacity_RC.png", generated)

    # 2. North-star chain: R/R+C decodability per location.
    rows = agg.get("locations", {}).get("rows", {})
    if rows:
        labels = list(rows.keys())
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, [_pct(rows[l].get("R_dec", 0.0)) for l in labels], w, label="R decodable", edgecolor="black")
        ax.bar(x + w / 2, [_pct(rows[l].get("RC_dec", 0.0)) for l in labels], w, label="RC decodable", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Probe accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("H2/H6: decodability along the north-star chain")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "north_star_chain.png", generated)

    # 3. Gradient shares.
    dom = agg.get("dominance", {})
    if dom.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["R share", "C share"], [_pct(dom.get("R_share", 0.0)), _pct(dom.get("C_share", 0.0))],
               color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Gradient-norm share (%)")
        ax.set_title("H3: R/C gradient contribution (RC probe batch)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "gradient_shares.png", generated)

    # 4. Counterfactual change rates.
    cf = agg.get("counterfactual", {})
    if cf.get("tested"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(["R change", "R correct", "control change"],
               [_pct(cf.get("R_change", 0.0)), _pct(cf.get("R_correct_change", 0.0)), _pct(cf.get("control_change", 0.0))],
               color=["#6baed6", "#2ca02c", "#999999"], edgecolor="black")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_title("H4/§12: R counterfactual decision sensitivity")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "counterfactual_sensitivity.png", generated)

    # 5. Frozen heads vs production.
    hd = agg.get("heads", {})
    if hd.get("tested"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(["linear", "nonlinear", "component-aware"],
               [_pct(hd.get("linear_RC", 0.0)), _pct(hd.get("nonlinear_RC", 0.0)), _pct(hd.get("component_RC", 0.0))],
               color=["#6baed6", "#fd8d3c", "#9e9ac8"], edgecolor="black")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("H5: frozen-representation heads (RC)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "frozen_heads.png", generated)

    # 6. Agree/disagree by model.
    if per_seed and per_seed[0].get("diagnostics", {}).get("agreement_table"):
        models = list(per_seed[0]["diagnostics"]["agreement_table"].keys())
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        x = np.arange(len(models))
        w = 0.35
        ag = [np.mean([r["diagnostics"]["agreement_table"][m].get("RC_agree", 0.0) for r in per_seed]) * 100 for m in models]
        dg = [np.mean([r["diagnostics"]["agreement_table"][m].get("RC_disagree", 0.0) for r in per_seed]) * 100 for m in models]
        ax.bar(x - w / 2, ag, w, label="agree", edgecolor="black")
        ax.bar(x + w / 2, dg, w, label="disagree", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("§13: RC agree vs disagree by model")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "agreement_table.png", generated)

    # 7. Order comparison.
    order = agg.get("order", {})
    if order.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["R->C chain", "C->R chain", "R->C interm.", "C->R interm."],
               [_pct(order.get("chain_RC_RtoC", 0.0)), _pct(order.get("chain_RC_CtoR", 0.0)),
                _pct(order.get("inter_RC_RtoC", 0.0)), _pct(order.get("inter_RC_CtoR", 0.0))],
               color=["#6baed6", "#fd8d3c", "#9e9ac8", "#ffbb78"], edgecolor="black")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("H7: order accuracy + controlled intermediate heads")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "order_RC.png", generated)

    # 8. Hypotheses.
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

    return generated
