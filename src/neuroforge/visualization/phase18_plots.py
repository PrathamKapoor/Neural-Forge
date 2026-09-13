"""Phase 18 figures (pure functions of executed-experiment `summary`)."""
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


def generate_phase18_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Agree vs disagree collapse.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["RC_audit"]["agree_acc"] * 100 for r in per_seed], w, label="agree", edgecolor="black")
        ax.bar(x + w / 2, [r["RC_audit"]["disagree_acc"] * 100 for r in per_seed], w, label="disagree", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Official RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("18E: RC agree vs disagree (production)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "agree_disagree.png", generated)

    # 2. Component matrix (best rep per component).
    comp = agg.get("components", {})
    if comp.get("tested") and per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        reps = sorted(per_seed[0]["comp_acc"].keys())
        x = np.arange(len(reps))
        w = 0.25
        for i, letter in enumerate(("F", "R", "C")):
            vals = [statistics.mean([r["comp_acc"][rp].get(letter, 0.0) for r in per_seed]) * 100 for rp in reps]
            ax.bar(x + (i - 1) * w, vals, w, label=f"{letter}-comp", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(reps)
        ax.set_ylabel("Component accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("18C: Component prediction matrix")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "component_matrix.png", generated)

    # 3. Joint decodability.
    joint = agg.get("joint", {})
    if joint.get("tested") and per_seed:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        reps = sorted(per_seed[0]["joint4"].keys())
        vals = [statistics.mean([r["joint4"][rp]["joint_acc"] for r in per_seed]) * 100 for rp in reps]
        ax.bar(reps, vals, color="#6baed6", edgecolor="black")
        ax.set_ylabel("Exact joint (sr,sc) accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("18D: Joint target decodability (RC)")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "joint_decodability.png", generated)

    # 4. Margins agree vs disagree.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["RC_audit"]["margin_agree"] for r in per_seed], w, label="margin agree", edgecolor="black")
        ax.bar(x + w / 2, [r["RC_audit"]["margin_disagree"] for r in per_seed], w, label="margin disagree", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Top-1 margin")
        ax.legend()
        ax.set_title("18F: Decision margins")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "margins.png", generated)

    # 5. Counterfactual flip rates.
    cf = agg.get("counterfactual", {})
    if cf.get("tested"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["R-swap", "C-swap", "same-side"],
               [cf.get("R_swap_flip_rate", 0.0) * 100, cf.get("C_swap_flip_rate", 0.0) * 100,
                cf.get("control_same_flip_rate", 0.0) * 100],
               color=["#6baed6", "#fd8d3c", "#999999"], edgecolor="black")
        ax.set_ylabel("Flip rate (%)")
        ax.set_ylim(0, 100)
        ax.set_title("18G: Counterfactual swap flips")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "counterfactuals.png", generated)

    # 6. Linear vs nonlinear on disagree.
    hd = agg.get("heads", {})
    if hd.get("tested") and per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["heads_RC"]["linear_disagree_RC"] * 100 for r in per_seed], w, label="linear", edgecolor="black")
        ax.bar(x + w / 2, [r["heads_RC"]["nonlinear_disagree_RC"] * 100 for r in per_seed], w, label="nonlinear", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("RC-disagree accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("18H: Head expressivity on disagree")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "diagnostic_heads.png", generated)

    # 7. Objective: rules vs model, train vs test.
    obj = agg.get("objective", {})
    if obj.get("tested") and per_seed:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.2
        series = [("rule-R tr", "rule_R_train_RC"), ("rule-C tr", "rule_C_train_RC"),
                  ("model tr", "model_train_RC"), ("model te", "model_test_RC")]
        for i, (lab, key) in enumerate(series):
            vals = []
            for r in per_seed:
                if key.startswith("rule"):
                    letter = "R" if "rule_R" in key else "C"
                    split = "train" if key.endswith("tr") else "test"
                    vals.append(r["objective"][f"rule_{letter}_{split}_RC"] * 100)
                else:
                    vals.append(r["objective"][f"model_{'train' if key.endswith('tr') else 'test'}_RC"] * 100)
            ax.bar(x + (i - 1.5) * w, vals, w, label=lab, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("18I: Single-component rules vs model (train/test)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "objective_rules.png", generated)

    # 8. Shuffled controls.
    if per_seed and "shuffled" in per_seed[0]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        ax.bar(["label-shuffled", "R-permuted"],
               [statistics.mean([r["shuffled"]["label_shuffled_RC"] for r in per_seed]) * 100,
                statistics.mean([r["shuffled"]["R_permuted_RC"] for r in per_seed]) * 100],
               color=["#999999", "#d62728"], edgecolor="black")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("18K: Shuffled controls (must be ~chance)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "shuffled_controls.png", generated)

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

    # 10. Seed stability (disagree).
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, [r["RC_audit"]["agree_acc"] * 100 for r in per_seed], w, label="agree", edgecolor="black")
        ax.bar(x + w / 2, [r["RC_audit"]["disagree_acc"] * 100 for r in per_seed], w, label="disagree", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("RC accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("Seed stability (agree/disagree)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    return generated
