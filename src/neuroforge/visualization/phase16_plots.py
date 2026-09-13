"""Phase 16 figures (pure functions of executed-experiment `summary`)."""
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


def generate_phase16_figures(summary: dict[str, Any], output_dir: str | Path) -> list[str]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    base = summary.get("baseline_perf_mean", {})
    graph = summary.get("graph_perf_mean", {})
    d3 = summary.get("depth3_perf_mean", {})
    bf = summary.get("rel_focused_perf_mean", {})
    br = summary.get("rel_first_perf_mean", {})
    per_seed = summary.get("per_seed_results", [])
    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})

    # 1. Baselines per family.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    x = np.arange(len(FAMILIES))
    w = 0.16
    for i, (label, perf) in enumerate(
        [("graph", graph), ("baseline", base), ("depth3", d3), ("rel_focused", bf), ("rel_first", br)]
    ):
        ax.bar(x + (i - 2) * w, [_pct(perf.get(f, 0.0)) for f in FAMILIES], w, label=label, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(list(FAMILIES))
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("16A: Exact baselines per family")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save(fig, dest, "baselines_per_family.png", generated)

    # 2. Equivalent-input computation (16C).
    comp = agg.get("computation", {})
    gdiag = agg.get("graph_diagnostic", {})
    if comp.get("tested"):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        labels = ["embedded\n+head", "graph-on-joint\n+head", "fresh graph\ndiag", "graph native\nfull"]
        vals = [_pct(comp.get("embedded_R", 0.0)), _pct(comp.get("graph_on_joint_R", 0.0)),
                _pct(gdiag.get("diagnostic_R", 0.0)),
                _pct(statistics.mean([r["graph_eval"]["perf"].get("R", 0.0) for r in per_seed])) if per_seed else 0.0]
        ax.bar(labels, vals, color=["#6baed6", "#fd8d3c", "#9e9ac8", "#31a354"], edgecolor="black")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("16C/16D: Same-input computation vs native Graph (R)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "equivalent_input_computation.png", generated)

    # 3. Stage probe trajectory (16E).
    st = agg.get("stages", {})
    if st.get("tested"):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        labels = ["pre-rel", "post-rel", "post-fusion", "native-in"]
        probes = [_pct(st.get("pre_probe_R", 0.0)), _pct(st.get("post_probe_R", 0.0)),
                  _pct(st.get("post_fusion_probe_R", 0.0)), _pct(st.get("native_input_probe_R", 0.0))]
        heads = [_pct(st.get("pre_R", 0.0)), _pct(st.get("post_R", 0.0)),
                 _pct(st.get("post_fusion_R", 0.0)), float("nan")]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, probes, w, label="probe R", color="#6baed6", edgecolor="black")
        ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in heads], w, label="head R", color="#fd8d3c", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title("16E: Relational information across computation depth (R)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "stage_probe_trajectory.png", generated)

    # 4. Co-adaptation (16F).
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    labels = ["depth3\n(joint)", "rel_focused", "rel_first"]
    vals = [_pct(d3.get("R", 0.0)), _pct(bf.get("R", 0.0)), _pct(br.get("R", 0.0))]
    ax.bar(labels, vals, color=["#6baed6", "#e377c2", "#7f7f7f"], edgecolor="black")
    ax.set_ylabel("R accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("16F: Branch-freeze co-adaptation (R)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save(fig, dest, "coadaptation.png", generated)

    # 5. Gradient shares (16G).
    grads = agg.get("gradients", {})
    if grads.get("status") == "VERIFIED":
        gm = grads.get("group_means", {})
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        labels = list(gm.keys())
        ax.bar(labels, [gm[k] for k in labels], color="#bcbd22", edgecolor="black")
        ax.set_ylabel("Mean grad-norm mass")
        ax.set_title("16G: Per-group gradient norms (probe batch, per epoch mean)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "gradient_shares.png", generated)

    # 6. Destruction drops (16J).
    if per_seed:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        conds = ["graph", "baseline", "depth3", "rel_focused", "rel_first"]
        drops = [statistics.mean([r[f"{c}_eval"].get("destruction_drop_R", 0.0) for r in per_seed]) * 100 for c in conds]
        ax.bar(conds, drops, color="#17becf", edgecolor="black")
        ax.set_ylabel("Destruction drop on R (pp)")
        ax.set_title("16J: Causal relational sensitivity per condition")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "destruction_drops.png", generated)

    # 7. RC/FRC composition (16K).
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    conds = [("baseline", base), ("depth3", d3), ("rel_focused", bf), ("rel_first", br), ("graph", graph)]
    x = np.arange(len(conds))
    w = 0.35
    ax.bar(x - w / 2, [_pct(p.get("RC", 0.0)) for _, p in conds], w, label="RC", edgecolor="black")
    ax.bar(x + w / 2, [_pct(p.get("FRC", 0.0)) for _, p in conds], w, label="FRC", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in conds], rotation=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("16K: Composition transfer (RC/FRC)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _save(fig, dest, "composition_rc_frc.png", generated)

    # 8. Branch ablation heat (16H, best condition).
    best = agg.get("best_cond", "depth3")
    rows = [r[f"{best}_eval"].get("branch7", {}) for r in per_seed if f"{best}_eval" in r]
    combos = [k for k in rows[0].keys() if isinstance(rows[0][k], dict)] if rows else []
    if combos:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        x = np.arange(len(combos))
        w = 0.35
        ax.bar(x - w / 2, [statistics.mean([brr[c].get("R", 0.0) for brr in rows]) * 100 for c in combos],
               w, label="R", edgecolor="black")
        ax.bar(x + w / 2, [statistics.mean([brr[c].get("RC", 0.0) for brr in rows]) * 100 for c in combos],
               w, label="RC", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=20, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_title(f"16H: Branch ablation ({best})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "branch_ablation.png", generated)

    # 9. Seed stability.
    if per_seed:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        seeds = [r.get("seed") for r in per_seed]
        x = np.arange(len(seeds))
        w = 0.15
        for i, cond in enumerate(["graph", "baseline", "depth3", "rel_focused", "rel_first"]):
            vals = [r[f"{cond}_eval"]["perf"].get("R", 0.0) * 100 for r in per_seed]
            ax.bar(x + (i - 2) * w, vals, w, label=cond, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("R accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title("Seed stability (R per seed)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "seed_stability.png", generated)

    # 10. Accuracy vs params.
    if per_seed:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for cond in ["graph", "baseline", "depth3", "rel_focused", "rel_first"]:
            rr = [r[f"{cond}_eval"]["perf"].get("R", 0.0) * 100 for r in per_seed]
            pp = [r[f"{cond}_eval"].get("params", 0) for r in per_seed]
            ax.scatter(pp, rr, label=cond, s=60, edgecolors="black")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("R accuracy (%)")
        ax.legend(fontsize=8)
        ax.set_title("Accuracy vs parameters (R)")
        ax.grid(True, linestyle="--", alpha=0.3)
        _save(fig, dest, "accuracy_vs_params.png", generated)

    # 11. Hypothesis verdicts.
    if hyps:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        ax.set_axis_off()
        colors = {"SUPPORTED": "#a1d99b", "PARTIALLY SUPPORTED": "#fee08b",
                  "NOT SUPPORTED": "#fdae61", "INCONCLUSIVE": "#d9d9d9", "NOT TESTED": "#e0e0e0"}
        names = sorted(hyps.keys())
        for i, h in enumerate(names):
            stt = hyps[h].get("status", "?")
            y = 0.93 - i * 0.105
            ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.09,
                                       facecolor=colors.get(stt, "#ffffff"), edgecolor="black"))
            ax.text(0.05, y, f"{h}: {stt}", va="center", fontsize=9, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("H1–H8 verdicts")
        _save(fig, dest, "hypothesis_verdicts.png", generated)

    # 12. Ceiling.
    ceil = agg.get("ceiling", {})
    if ceil.get("tested"):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        ax.bar(["old ceiling", "new ceiling"],
               [_pct(ceil.get("old_ceiling_mixed_mean", 0.0)), _pct(ceil.get("new_ceiling_mixed_mean", 0.0))],
               color=["#6baed6", "#fd8d3c"], edgecolor="black")
        ax.set_ylabel("Mixed-mean ceiling (%)")
        ax.set_title("Empirical single-expert ceiling")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _save(fig, dest, "ceiling_comparison.png", generated)

    return generated
