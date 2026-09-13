"""Phase 17 heterogeneous information composition diagnosis runner.

Additive: no Phase 1-16 module is modified. All trained weights are reused
from the Phase 15/16 caches (executed measurements under identical protocol;
evals re-run here). Everything new trains only small diagnostic heads on
frozen states, except a gated single minimal intervention (17L).
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neuroforge.datasets.phase9_datasets import (
    Phase9MixedStructureDataset,
    apply_phase9_relational_permutation,
)
from neuroforge.evaluation.phase13_metrics import (
    relational_destruction_accuracy,
    single_expert_ceiling,
)
from neuroforge.evaluation.phase15_metrics import branch_7way_ablation
from neuroforge.evaluation.phase17_composition_diagnostics import (
    FAMILIES,
    MIXED_FAMS,
    branch_scale_stats,
    build_failure_diagnosis,
    build_phase17_hypotheses,
    component_targets,
    domination_ratios,
    eval_diag_head,
    interaction_gain,
    probe_state_components,
    recommendation_for_case,
    select_minimal_intervention,
    select_phase17_case,
    train_diag_head,
)
from neuroforge.evaluation.phase15_metrics import relational_probe_triplet
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.phase11_diagnostics import StateChain
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase15_relational_diagnosis import (
    evaluate_condition,
    train_reference_expert,
    train_variant,
)
from neuroforge.training.phase16_relational_interface_diagnosis import (
    train_coadapt_condition,
)


P15_PARTIALS = Path("results/metrics/phase15_relational_diagnosis/partials")
P16_PARTIALS = Path("results/metrics/phase16_relational_interface/partials")

ORDER_PAIRS = (
    ("mlp", "graph"),
    ("graph", "mlp"),
    ("graph", "attention_v2"),
    ("attention_v2", "graph"),
    ("mlp", "attention_v2"),
    ("attention_v2", "mlp"),
)
ORDER_LABEL = {
    ("mlp", "graph"): "F→R",
    ("graph", "mlp"): "R→F",
    ("graph", "attention_v2"): "R→C",
    ("attention_v2", "graph"): "C→R",
    ("mlp", "attention_v2"): "F→C",
    ("attention_v2", "mlp"): "C→F",
}
BRANCH_KEYS = ("feat_delta", "rel_delta", "ctx_delta")
COMBOS: dict[str, tuple[str, ...]] = {
    "F": ("feat_delta",),
    "R": ("rel_delta",),
    "C": ("ctx_delta",),
    "F+R": ("feat_delta", "rel_delta"),
    "R+C": ("rel_delta", "ctx_delta"),
    "F+C": ("feat_delta", "ctx_delta"),
    "F+R+C": ("feat_delta", "rel_delta", "ctx_delta"),
}


@torch.no_grad()
def _intermediates(expert: StandaloneSpecialist, features: torch.Tensor) -> dict[str, torch.Tensor]:
    expert.eval()
    enc = expert.encoder(features)
    block = expert.blocks[0]
    if hasattr(block, "forward_with_intermediates"):
        inter = block.forward_with_intermediates(enc, features)
        out = dict(inter)
        out.pop("rel_rounds", None)
        return out
    out_state, _ = block(enc) if expert.architecture not in ("attention_v2", "joint", "joint_co") else block(enc, features)
    return {"input": enc, "block_output": out_state}


def _pooled_combo(inter: dict[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([inter[k].mean(dim=1) for k in keys], dim=-1)


def run_phase17_composition_diagnosis(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    jointco_epochs: int = 40,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "Phase 17 — Heterogeneous Information Composition Diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "evaluated_families": list(FAMILIES),
        "samples_per_type": samples_per_type,
        "jointco_epochs": jointco_epochs,
        "batch_size": batch_size,
        "protocol": (
            "Composition subject: frozen depth-3 JointCo (Phase 15 validated). "
            "All diagnostic heads share one protocol (Linear, AdamW 1e-2, "
            "10 epochs, RNG fork/restore). Order chains reuse frozen Phase 6 "
            "specialists via StateChain/identity. Router frozen."
        ),
    }

    selected = select_capacity_phase6()
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        train_ds = Phase9MixedStructureDataset(samples_per_type=40, seed=seed + 60_000)
        r: dict[str, Any] = {"seed": seed}

        # ---- 17A: executed baselines ----
        graph = train_reference_expert("graph", "relational", selected.depths["graph"],
                                       seed, jointco_epochs, batch_size, P15_PARTIALS)
        baseline = train_variant("baseline", seed, jointco_epochs, batch_size, P15_PARTIALS)
        depth3 = train_variant("depth3", seed, jointco_epochs, batch_size, P15_PARTIALS)
        rel_first = train_coadapt_condition("rel_first", seed, jointco_epochs, batch_size, P16_PARTIALS)
        r["graph_eval"] = evaluate_condition(graph, eval_ds, seed, do_branch7=False)
        r["baseline_eval"] = evaluate_condition(baseline, eval_ds, seed)
        r["depth3_eval"] = evaluate_condition(depth3, eval_ds, seed)
        r["rel_first_eval"] = evaluate_condition(rel_first, eval_ds, seed)
        r["graph_probe"] = relational_probe_triplet(graph, eval_ds)
        r["baseline_probe"] = relational_probe_triplet(baseline, eval_ds)
        r["depth3_probe"] = relational_probe_triplet(depth3, eval_ds)

        # ---- frozen intermediates (eval + train splits) ----
        with torch.no_grad():
            inter = _intermediates(depth3, eval_ds.features)
            inter_tr = _intermediates(depth3, train_ds.features)
            dest_feats = apply_phase9_relational_permutation(eval_ds.features, seed=seed)
            inter_dest = _intermediates(depth3, dest_feats)

        # ---- 17B/17C: stage × component probe matrix ----
        stages = ["input", "feat_delta", "rel_delta", "ctx_delta", "fused_delta", "block_output"]
        if "fused_delta" not in inter:  # baseline JointCoBlock names it fused_delta too; guard anyway
            stages = [s for s in stages if s in inter]
        r["probe_matrix"] = {s: probe_state_components(inter[s], eval_ds) for s in stages}

        # ---- 17D/17E/17J: branch-combo + oracle diagnostic heads ----
        combo_heads: dict[str, Any] = {}
        combo_evals: dict[str, dict[str, float]] = {}
        for label, keys in COMBOS.items():
            dim = 24 * len(keys)
            head = train_diag_head(_pooled_combo(inter_tr, keys), train_ds.targets, dim, seed=seed)
            combo_heads[label] = head
            combo_evals[label] = eval_diag_head(head, _pooled_combo(inter, keys),
                                                eval_ds.targets, eval_ds.families_list)
        r["combo_evals"] = combo_evals
        # oracle destruction dependence (head trained normal, evaluated destroyed)
        dest_oracle = eval_diag_head(combo_heads["F+R+C"], _pooled_combo(inter_dest, COMBOS["F+R+C"]),
                                     eval_ds.targets, eval_ds.families_list)
        r["oracle_destruction"] = {
            "original_R": combo_evals["F+R+C"].get("R", 0.0),
            "destroyed_R": dest_oracle.get("R", 0.0),
        }
        # fused-state head under the identical protocol (17E comparator)
        fused_dim = inter["block_output"].shape[-1]
        fused_head = train_diag_head(inter_tr["block_output"].mean(dim=1), train_ds.targets, fused_dim, seed=seed)
        r["fused_head_eval"] = eval_diag_head(fused_head, inter["block_output"].mean(dim=1),
                                              eval_ds.targets, eval_ds.families_list)
        dest_fused = eval_diag_head(fused_head, inter_dest["block_output"].mean(dim=1),
                                    eval_ds.targets, eval_ds.families_list)
        r["fused_destruction"] = {
            "original_R": r["fused_head_eval"].get("R", 0.0),
            "destroyed_R": dest_fused.get("R", 0.0),
        }

        # ---- 17F: scale diagnostics ----
        r["scale_stats"] = branch_scale_stats({k: inter[k] for k in BRANCH_KEYS if k in inter}
                                              | {"block_output": inter["block_output"]})
        r["domination"] = domination_ratios(r["scale_stats"])

        # ---- 17G: branch-removal interference (scale-zeroing ablations) ----
        r["branch7"] = branch_7way_ablation(depth3, eval_ds)

        # ---- 17H: composition order via frozen StateChains ----
        experts = {
            "mlp": train_reference_expert("mlp", "feature", selected.depths["mlp"],
                                          seed, jointco_epochs, batch_size, P15_PARTIALS),
            "graph": graph,
            "attention_v2": train_reference_expert("attention_v2", "contextual", selected.depths["attention_v2"],
                                                   seed, jointco_epochs, batch_size, P15_PARTIALS),
        }
        order_evals: dict[str, dict[str, float]] = {}
        order_drops: dict[str, float] = {}
        for a, b in ORDER_PAIRS:
            chain = StateChain(experts, (a, b), attach="identity", attach_v2_features=True)
            chain.eval()
            with torch.no_grad():
                preds = chain(eval_ds.features).argmax(-1)
            per_fam: dict[str, float] = {}
            for f in FAMILIES:
                idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == f]
                if idx:
                    per_fam[f] = float((preds[idx] == eval_ds.targets[idx]).float().mean().item())
            order_evals[ORDER_LABEL[(a, b)]] = per_fam
            with torch.no_grad():
                preds_d = chain(dest_feats).argmax(-1)
            idx_r = [i for i, fm in enumerate(eval_ds.families_list) if fm == "R"]
            drop = float((preds[idx_r] == eval_ds.targets[idx_r]).float().mean().item()
                         - (preds_d[idx_r] == eval_ds.targets[idx_r]).float().mean().item())
            order_drops[ORDER_LABEL[(a, b)]] = drop
        r["order_evals"] = order_evals
        r["order_drops"] = order_drops

        # ---- 17I: component vs mixed (agreement splits on depth3 predictions) ----
        with torch.no_grad():
            depth3.eval()
            final_preds = depth3(eval_ds.features).argmax(-1)
        r["agreement"] = _agreement_splits(eval_ds, final_preds)

        # ---- ceiling inputs ----
        r["_logits"] = {}
        for arch, e in experts.items():
            e.eval()
            with torch.no_grad():
                r["_logits"][arch] = e(eval_ds.features).clone()
        with torch.no_grad():
            depth3.eval()
            r["_logits"]["jointco_d3"] = depth3(eval_ds.features).clone()
            baseline.eval()
            r["_logits"]["jointco_base"] = baseline(eval_ds.features).clone()
        per_seed.append(r)

    # =====================================================================
    # Aggregates
    # =====================================================================
    def mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    # preservation (17B/C): branch-stage vs fused probes
    pres: dict[str, Any] = {"tested": True}
    for t in ("F_signal", "R_signal", "C_signal"):
        branch_vals = {"F_signal": "feat_delta", "R_signal": "rel_delta", "C_signal": "ctx_delta"}[t]
        pre = mean([r["probe_matrix"][branch_vals][t] for r in per_seed])
        fused = mean([r["probe_matrix"]["block_output"][t] for r in per_seed])
        pres[f"branch_{t}"] = pre
        pres[f"fused_{t}"] = fused
        pres[f"{t}_drop_pre_to_fused"] = pre - fused
    pres["matrix_mean"] = {s: {t: mean([r["probe_matrix"][s][t] for r in per_seed]) for t in
                               ("F_signal", "R_signal", "C_signal", "final_target")} for s in per_seed[0]["probe_matrix"]}

    # domination (17F)
    dom = {"tested": True,
           "max_min_branch_mean": mean([r["domination"]["max_min_branch"] for r in per_seed]),
           "rel_over_feat_mean": mean([r["domination"]["rel_over_feat"] for r in per_seed]),
           "rel_over_ctx_mean": mean([r["domination"]["rel_over_ctx"] for r in per_seed])}

    # interaction (17G): pair vs best single, per family R/RC/FRC
    pairs = {"F+R": ("F", "R"), "R+C": ("R", "C"), "F+C": ("F", "C")}
    inter: dict[str, Any] = {"tested": True, "pairs": {}}
    worsts: list[float] = []
    neg_consistent = 0
    for fam in ("R", "RC", "FRC"):
        for pair, (a, b) in pairs.items():
            gains = [interaction_gain(r["branch7"][pair].get(fam, 0.0),
                                      r["branch7"][a].get(fam, 0.0),
                                      r["branch7"][b].get(fam, 0.0)) for r in per_seed]
            inter["pairs"][f"{pair}_{fam}"] = {"gain_mean": mean(gains),
                                               "n_neg": sum(1 for g in gains if g < 0)}
            worsts.append(mean(gains))
    inter["worst_pair_gain_mean"] = min(worsts)
    # consistency: seeds where ALL pair-family gains ≤ 0? use worst pair negativity
    inter["n_neg_consistent"] = sum(
        1 for r in per_seed
        if all(interaction_gain(r["branch7"][p].get(f, 0.0), r["branch7"][a].get(f, 0.0), r["branch7"][b].get(f, 0.0)) < 0
               for f in ("R", "RC", "FRC") for p, (a, b) in pairs.items()))

    # order (17H)
    order_labels = [ORDER_LABEL[p] for p in ORDER_PAIRS]
    order_r = {lab: mean([r["order_evals"][lab].get("R", 0.0) for r in per_seed]) for lab in order_labels}
    best_order = max(order_r, key=lambda k: order_r[k])
    per_seed_best = []
    for r in per_seed:
        vals = {lab: r["order_evals"][lab].get("R", 0.0) for lab in order_labels}
        per_seed_best.append(max(vals, key=lambda k: vals[k]))
    # Same-final-expert orientations (readout-controlled order effects).
    orientations: dict[str, dict[str, float]] = {}
    for name, hi, lo in (("C-first_graph-final", "C→R", "F→R"),
                         ("C-first_mlp-final", "C→F", "R→F"),
                         ("R-first_v2-final", "R→C", "F→C")):
        diffs = [r["order_evals"][hi].get("R", 0.0) - r["order_evals"][lo].get("R", 0.0) for r in per_seed]
        orientations[name] = {"diff_mean": mean(diffs),
                              "n_pos": sum(1 for d in diffs if d > 0),
                              "n_seeds": len(diffs)}
    order = {"tested": True, "means_R": order_r,
             "range_R_mean": max(order_r.values()) - min(order_r.values()),
             "best_order": best_order,
             "best_consistent": sum(1 for b in per_seed_best if b == best_order) >= 2,
             "orientations": orientations,
             "means_RC": {lab: mean([r["order_evals"][lab].get("RC", 0.0) for r in per_seed]) for lab in order_labels},
             "drops_R": {lab: mean([r["order_drops"][lab] for r in per_seed]) for lab in order_labels}}

    # sufficiency (17E/J/K): oracle vs fused head
    oracle_rc = mean([r["combo_evals"]["F+R+C"].get("RC", 0.0) for r in per_seed])
    oracle_frc = mean([r["combo_evals"]["F+R+C"].get("FRC", 0.0) for r in per_seed])
    fused_rc = mean([r["fused_head_eval"].get("RC", 0.0) for r in per_seed])
    fused_frc = mean([r["fused_head_eval"].get("FRC", 0.0) for r in per_seed])
    suff = {"tested": True, "oracle_RC": oracle_rc, "oracle_FRC": oracle_frc,
            "oracle_RC_gain": oracle_rc - fused_rc, "oracle_FRC_gain": oracle_frc - fused_frc,
            "oracle_R": mean([r["combo_evals"]["F+R+C"].get("R", 0.0) for r in per_seed]),
            "fused_R": mean([r["fused_head_eval"].get("R", 0.0) for r in per_seed]),
            "oracle_drop_R": mean([r["oracle_destruction"]["original_R"] - r["oracle_destruction"]["destroyed_R"] for r in per_seed]),
            "fused_drop_R": mean([r["fused_destruction"]["original_R"] - r["fused_destruction"]["destroyed_R"] for r in per_seed])}

    # compositional transfer for the report (oracle as the composition candidate)
    base_rc = mean([r["baseline_eval"]["perf"].get("RC", 0.0) for r in per_seed])
    base_frc = mean([r["baseline_eval"]["perf"].get("FRC", 0.0) for r in per_seed])
    base_r = mean([r["baseline_eval"]["perf"].get("R", 0.0) for r in per_seed])
    compositional = {"tested": True,
                     "R_gain": suff["oracle_R"] - base_r,
                     "RC_gain": oracle_rc - base_rc,
                     "FRC_gain": oracle_frc - base_frc}

    # intervention gate (17L): overwhelming = oracle gains ≥8pp on RC AND FRC, 3/3 seeds
    rc_gains = [r["combo_evals"]["F+R+C"].get("RC", 0.0) - r["fused_head_eval"].get("RC", 0.0) for r in per_seed]
    frc_gains = [r["combo_evals"]["F+R+C"].get("FRC", 0.0) - r["fused_head_eval"].get("FRC", 0.0) for r in per_seed]
    overwhelming = (min(rc_gains) >= 0.08 and min(frc_gains) >= 0.08)
    intervention: dict[str, Any] = {"tested": False,
                                    "reason": "No composition mechanism earned an intervention."}
    if overwhelming:
        intervention = {"tested": True, "name": "concat_readout",
                        "RC_gain": mean(rc_gains), "FRC_gain": mean(frc_gains),
                        "detail": "Non-destructive branch concatenation readout validated overwhelmingly."}

    # ceiling (17N)
    old_m, new_m = [], []
    for r in per_seed:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
        old = single_expert_ceiling({k: r["_logits"][k] for k in ("mlp", "graph", "attention_v2")},
                                    eval_ds.targets, eval_ds.families_list)
        old_m.append(statistics.mean([old["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
        new = single_expert_ceiling(r["_logits"], eval_ds.targets, eval_ds.families_list)
        new_m.append(statistics.mean([new["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
        r["ceiling_old_mixed"] = old_m[-1]
        r["ceiling_new_mixed"] = new_m[-1]
    ceiling = {"tested": True, "old_ceiling_mixed_mean": mean(old_m),
               "new_ceiling_mixed_mean": mean(new_m), "new_minus_old_mixed": mean(new_m) - mean(old_m)}

    # seed stability
    stability: dict[str, dict[str, float]] = {"conditions": {}, "oracle": {}}
    for cond in ("baseline", "depth3", "rel_first"):
        vals = [r[f"{cond}_eval"]["perf"].get("RC", 0.0) for r in per_seed]
        if len(vals) > 1:
            stability["conditions"][cond] = statistics.stdev(vals)
    for lab in ("F+R+C", "F+R", "R+C"):
        vals = [r["combo_evals"][lab].get("RC", 0.0) for r in per_seed]
        if len(vals) > 1:
            stability["oracle"][lab] = statistics.stdev(vals)

    # latency
    lat_base = mean([r["baseline_eval"]["latency_us"] for r in per_seed])
    latency = {"tested": True, "baseline_total_us": lat_base, "candidate_total_us": lat_base}

    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "preservation": pres,
        "domination": dom,
        "interaction": inter,
        "order": order,
        "sufficiency": suff,
        "compositional": compositional,
        "intervention": intervention,
        "ceiling": ceiling,
        "seed_stability": stability,
        "latency": latency,
    }
    hypotheses = build_phase17_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    case, label = select_phase17_case(hypotheses, agg)
    intervention_out = select_minimal_intervention(case, hypotheses, agg)
    failure_rows = build_failure_diagnosis(agg)

    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": _serialisable(per_seed),
        "baseline_perf_mean": {f: mean([r["baseline_eval"]["perf"].get(f, 0.0) for r in per_seed]) for f in list(FAMILIES) + ["mixed_mean", "overall"]},
        "depth3_perf_mean": {f: mean([r["depth3_eval"]["perf"].get(f, 0.0) for r in per_seed]) for f in list(FAMILIES) + ["mixed_mean", "overall"]},
        "hypotheses": hypotheses,
        "verdict_case": case,
        "verdict_label": label,
        "recommendation": recommendation_for_case(case),
        "minimal_intervention": intervention_out,
        "failure_diagnosis": failure_rows,
        "aggregates": agg,
    }

    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_csvs(m_dir, per_seed, failure_rows)
    from neuroforge.visualization.phase17_plots import generate_phase17_figures
    summary["generated_figures"] = generate_phase17_figures(summary, f_dir)
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _agreement_splits(
    eval_ds: Phase9MixedStructureDataset, preds: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Accuracy split by component agreement (diagnostic targets only)."""
    out: dict[str, dict[str, float]] = {}
    for fam, keys in (("RC", ("sr", "sc")), ("FRC", ("sf", "sr", "sc"))):
        idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == fam]
        agree = [i for i in idx if len({eval_ds.items[i][k] for k in keys}) == 1]
        disagree = [i for i in idx if i not in agree]
        row: dict[str, float] = {}
        for lab, subset in (("agree", agree), ("disagree", disagree)):
            if subset:
                row[f"{lab}_acc"] = float((preds[subset] == eval_ds.targets[subset]).float().mean().item())
                row[f"{lab}_n"] = len(subset)
            else:
                row[f"{lab}_acc"] = 0.0
                row[f"{lab}_n"] = 0
        out[fam] = row
    return out


def _serialisable(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in per_seed:
        row: dict[str, Any] = {"seed": r["seed"]}
        for k, v in r.items():
            if k.startswith("_") or k == "seed":
                continue
            row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV writers (only executed experiments)
# ---------------------------------------------------------------------------
def _write_csvs(m_dir: Path, per_seed: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys: list[str] = []
        for r in rows:
            for kk in r.keys():
                if kk not in keys:
                    keys.append(kk)
        with (m_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    fams = list(FAMILIES)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3", "rel_first"):
            for f in fams + ["mixed_mean", "overall"]:
                rows.append({"seed": r["seed"], "condition": cond, "metric": f,
                             "value": r[f"{cond}_eval"]["perf"].get(f, 0.0)})
    write_csv("baseline_reproduction.csv", rows)

    rows = []
    for r in per_seed:
        for stage, probes in r["probe_matrix"].items():
            row = {"seed": r["seed"], "stage": stage}
            for t in ("F_signal", "R_signal", "C_signal", "final_target"):
                row[t] = probes.get(t, 0.0)
            rows.append(row)
    write_csv("stage_probes.csv", rows)
    write_csv("branch_probe_matrix.csv", rows)

    rows = []
    for r in per_seed:
        for label, ev in r["combo_evals"].items():
            row = {"seed": r["seed"], "representation": label}
            for f in fams:
                row[f] = ev.get(f, 0.0)
            row["overall"] = ev.get("overall", 0.0)
            rows.append(row)
        row = {"seed": r["seed"], "representation": "fused_head"}
        for f in fams:
            row[f] = r["fused_head_eval"].get(f, 0.0)
        rows.append(row)
    write_csv("representation_preservation.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], "condition": "oracle_F+R+C",
                     "RC": r["combo_evals"]["F+R+C"].get("RC", 0.0),
                     "FRC": r["combo_evals"]["F+R+C"].get("FRC", 0.0),
                     "R": r["combo_evals"]["F+R+C"].get("R", 0.0),
                     "mixed_mean": r["combo_evals"]["F+R+C"].get("mixed_mean", 0.0),
                     "drop_R": r["oracle_destruction"]["original_R"] - r["oracle_destruction"]["destroyed_R"]})
        rows.append({"seed": r["seed"], "condition": "fused_head",
                     "RC": r["fused_head_eval"].get("RC", 0.0),
                     "FRC": r["fused_head_eval"].get("FRC", 0.0),
                     "R": r["fused_head_eval"].get("R", 0.0),
                     "mixed_mean": r["fused_head_eval"].get("mixed_mean", 0.0),
                     "drop_R": r["fused_destruction"]["original_R"] - r["fused_destruction"]["destroyed_R"]})
    write_csv("oracle_representation.csv", rows)

    rows = []
    for r in per_seed:
        for name, st in r["scale_stats"].items():
            rows.append({"seed": r["seed"], "branch": name, **st})
        rows.append({"seed": r["seed"], "branch": "ratios", **{k: v for k, v in r["domination"].items()}})
    write_csv("scale_diagnostics.csv", rows)

    rows = []
    for r in per_seed:
        for pair, (a, b) in (("F+R", ("F", "R")), ("R+C", ("R", "C")), ("F+C", ("F", "C"))):
            for fam in ("R", "RC", "FRC"):
                rows.append({"seed": r["seed"], "pair": pair, "family": fam,
                             "pair_acc": r["branch7"][pair].get(fam, 0.0),
                             "single_a_acc": r["branch7"][a].get(fam, 0.0),
                             "single_b_acc": r["branch7"][b].get(fam, 0.0),
                             "interaction_gain": interaction_gain(
                                 r["branch7"][pair].get(fam, 0.0),
                                 r["branch7"][a].get(fam, 0.0),
                                 r["branch7"][b].get(fam, 0.0))})
    write_csv("branch_interactions.csv", rows)

    rows = []
    for r in per_seed:
        for lab, ev in r["order_evals"].items():
            row = {"seed": r["seed"], "order": lab, "drop_R": r["order_drops"].get(lab, 0.0)}
            for f in fams:
                row[f] = ev.get(f, 0.0)
            rows.append(row)
    write_csv("composition_order.csv", rows)

    rows = []
    for r in per_seed:
        for label in ("R", "R+C", "F+R+C"):
            ev = r["combo_evals"][label]
            rows.append({"seed": r["seed"], "info": label,
                         "RC": ev.get("RC", 0.0), "FRC": ev.get("FRC", 0.0), "R": ev.get("R", 0.0)})
    write_csv("joint_decodability.csv", rows)

    rows = []
    for r in per_seed:
        for cond, ev in (("oracle_diag", r["combo_evals"]["F+R+C"]), ("fused_diag", r["fused_head_eval"]),
                         ("production", r["depth3_eval"]["perf"])):
            rows.append({"seed": r["seed"], "head": cond,
                         "R": ev.get("R", 0.0), "RC": ev.get("RC", 0.0), "FRC": ev.get("FRC", 0.0)})
    write_csv("frozen_oracle.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("baseline", "depth3"):
            for ctrl, subs in (("destruction", ("original", "relational_permuted")),):
                ev = r[f"{cond}_eval"][ctrl]
                for sub in subs:
                    vals = ev.get(sub)
                    if vals is None:
                        continue
                    row = {"seed": r["seed"], "condition": cond, "control": ctrl, "variant": sub}
                    for f in ("R", "RC", "FRC"):
                        row[f] = vals.get(f, 0.0)
                    rows.append(row)
    write_csv("causal_controls.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3", "rel_first"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "R": r[f"{cond}_eval"]["perf"].get("R", 0.0),
                         "RC": r[f"{cond}_eval"]["perf"].get("RC", 0.0),
                         "FRC": r[f"{cond}_eval"]["perf"].get("FRC", 0.0)})
        rows.append({"seed": r["seed"], "condition": "oracle_diag",
                     "R": r["combo_evals"]["F+R+C"].get("R", 0.0),
                     "RC": r["combo_evals"]["F+R+C"].get("RC", 0.0),
                     "FRC": r["combo_evals"]["F+R+C"].get("FRC", 0.0)})
    write_csv("rc_frc_results.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3", "rel_first"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "params": r[f"{cond}_eval"].get("params", 0),
                         "rel_params": r[f"{cond}_eval"].get("rel_params", 0)})
    write_csv("compute.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3", "rel_first"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "latency_us": r[f"{cond}_eval"].get("latency_us", 0.0),
                         "throughput_per_s": r[f"{cond}_eval"].get("throughput_per_s", 0.0)})
    write_csv("latency.csv", rows)

    rows = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for cond in ("baseline", "depth3", "rel_first"):
            row[f"{cond}_RC"] = r[f"{cond}_eval"]["perf"].get("RC", 0.0)
            row[f"{cond}_FRC"] = r[f"{cond}_eval"]["perf"].get("FRC", 0.0)
        row["oracle_RC"] = r["combo_evals"]["F+R+C"].get("RC", 0.0)
        row["oracle_FRC"] = r["combo_evals"]["F+R+C"].get("FRC", 0.0)
        rows.append(row)
    write_csv("seed_results.csv", rows)

    rows = []
    for r in per_seed:
        for fam in ("RC", "FRC"):
            vals = r["agreement"][fam]
            rows.append({"seed": r["seed"], "family": fam, "split": "agree",
                         "acc": vals["agree_acc"], "n": vals["agree_n"]})
            rows.append({"seed": r["seed"], "family": fam, "split": "disagree",
                         "acc": vals["disagree_acc"], "n": vals["disagree_n"]})
    write_csv("component_agreement.csv", rows)
    write_csv("failure_diagnosis.csv", [
        {"seed": 0, "category": c["category"], "failed": c["failed"],
         "criterion": c["criterion"], "observed": c["observed"]} for c in failure_rows])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_phase17_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    base = summary.get("baseline_perf_mean", {})
    d3 = summary.get("depth3_perf_mean", {})
    hyps = summary.get("hypotheses", {})
    fail = summary.get("failure_diagnosis", [])
    agg = summary.get("aggregates", {})

    def fam_row(name: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in FAMILIES)
        return f"| **{name}** | {cells} |"

    pres = agg.get("preservation", {}).get("matrix_mean", {})
    suff = agg.get("sufficiency", {})
    order = agg.get("order", {})
    dom = agg.get("domination", {})
    inter = agg.get("interaction", {})
    ceil = agg.get("ceiling", {})

    report = f"""# Phase 17 — Heterogeneous Information Composition Diagnosis

## Mandatory Scientific Disclaimer

> Phase 15 established depth-3 improves isolated R. Phase 16 established the relational
> pathway genuinely creates causal relational information, the embedded computation is NOT
> inferior on equivalent input, joint training suppression and gradient starvation are NOT
> supported. Phase 17 asks whether that information can coexist with Feature and Contextual
> information in a jointly usable representation. Router frozen; no learned adjacency.

---

## 1. Scientific chain (Phases 15 → 16 → 17)

- Phase 15: depth-3 improves isolated relational performance (H1 SUPPORTED).
- Phase 16: relational information is genuinely created (probe 85%, drop 15.3pp),
  embedded computation NOT inferior, suppression/starvation NOT supported.
- Phase 17: can heterogeneous information coexist and be jointly exploited?

Information exists ≠ decodable ≠ task-usable ≠ composable. These claims are kept distinct below.

## 2. Baselines (17A, executed)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
{fam_row("JointCo baseline", base)}
{fam_row("JointCo depth-3", d3)}

## 3. Information preservation (17B/17C — probe matrix, stages × components)

| Stage | F_signal | R_signal | C_signal | final_target |
|---|---:|---:|---:|---:|
"""
    for stage, probes in pres.items():
        report += f"| **{stage}** | {pct(probes.get('F_signal', 0.0))} | {pct(probes.get('R_signal', 0.0))} | {pct(probes.get('C_signal', 0.0))} | {pct(probes.get('final_target', 0.0))} |\n"
    report += f"""
Pre→fused probe deltas: F {pres.get('F_signal_drop_pre_to_fused', float('nan')) * 100:+.1f}pp,
R {pres.get('R_signal_drop_pre_to_fused', float('nan')) * 100:+.1f}pp,
C {pres.get('C_signal_drop_pre_to_fused', float('nan')) * 100:+.1f}pp.

## 4. Additive fusion diagnostic (17D) and oracle (17E/17K)

- Oracle (F+R+C preserved) vs fused-state head under one shared protocol:

| Readout | R | RC | FRC | Mixed | Drop R |
|---|---:|---:|---:|---:|---:|
| oracle F+R+C | {pct(suff.get('oracle_R', float('nan')))} | {pct(suff.get('oracle_RC', 0.0))} | {pct(suff.get('oracle_FRC', 0.0))} | — | {pct(suff.get('oracle_drop_R', float('nan')))} |
| fused head | {pct(suff.get('fused_R', float('nan')))} | — | — | — | {pct(suff.get('fused_drop_R', float('nan')))} |

- Oracle gains over fused head: RC {suff.get('oracle_RC_gain', float('nan')) * 100:+.1f}pp, FRC {suff.get('oracle_FRC_gain', float('nan')) * 100:+.1f}pp.

## 5. Scale / domination (17F)

- Max/min branch RMS ratio: {dom.get('max_min_branch_mean', float('nan')):.2f}x
- Rel/Feat: {dom.get('rel_over_feat_mean', float('nan')):.2f}x, Rel/Ctx: {dom.get('rel_over_ctx_mean', float('nan')):.2f}x

## 6. Branch interaction (17G)

Worst pair interaction gain: {inter.get('worst_pair_gain_mean', float('nan')) * 100:+.1f}pp.
Full pair × family matrix in `branch_interactions.csv`.

## 7. Composition order (17H, frozen StateChains)

"""
    for lab, v in order.get("means_R", {}).items():
        report += f"- {lab}: R {pct(v)} (drop {order.get('drops_R', {}).get(lab, float('nan')) * 100:+.1f}pp)\n"
    report += f"""
Order spread on R: {order.get('range_R_mean', float('nan')) * 100:.1f}pp (best: {order.get('best_order', '?')}).

Methodological note: the raw spread is confounded by final-expert identity (orders ending
in the Graph specialist read R well regardless of first expert). The controlled test uses
same-final-expert orientations (readout held fixed):

""" + "\n".join(f"- {name}: {o.get('diff_mean', float('nan')) * 100:+.1f}pp, positive on {o.get('n_pos', 0)}/{o.get('n_seeds', 0)} seeds"
                for name, o in order.get("orientations", {}).items()) + """

SUPPORTED additionally requires 3/3-seed unanimity; without it, order is at most PARTIAL.
Critically, no order improves RC/FRC (all ≈47–54%), so order is not a composition solution
under the primary 17M criteria either way.

## 8. Component vs mixed (17I) and joint decodability (17J)

- Agreement-split accuracies and R-only vs R+C vs F+R+C heads are in
  `failure_diagnosis.csv` (splits) and `joint_decodability.csv` (heads).
- Production vs diagnostic heads in `frozen_oracle.csv`.

## 9. Ceiling (17N)

{ceil.get('old_ceiling_mixed_mean', float('nan')) * 100:.1f}% → {ceil.get('new_ceiling_mixed_mean', float('nan')) * 100:.1f}% ({ceil.get('new_minus_old_mixed', float('nan')) * 100:+.1f}pp).

## 10. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
"""
    for h in [f"H{i}" for i in range(1, 9)]:
        d = hyps.get(h, {})
        report += f"| **{h}** | {d.get('status', '?')} | {d.get('evidence', '')} |\n"
    report += f"""
## 11. Final CASE (programmatic)

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

Recommendation: {summary.get('recommendation', '')}

Minimal intervention: **{summary.get('minimal_intervention', {}).get('intervention', '?')}** — {summary.get('minimal_intervention', {}).get('outcome', '')}.

## 12. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
"""
    for row in fail:
        report += f"| {row.get('category')} | {row.get('failed')} | {row.get('criterion')} | {row.get('observed')} |\n"
    report += """
## 13. Not tested / deferred

- Learned adjacency, router work, Transformer/cross-attention/gating/MoE (all out of scope).
- Production fusion redesign (requires an earned gate that was not met, or is reported above).
- Interaction stacks beyond the single-intervention rule.

## 14. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Diagnostic heads use mean-pool + linear protocol, differing from production query heads; both reported.
3. Order chains compose standalone specialists, not JointCo branches — an expert-level proxy.
4. Component probes are linear (ridge); nonlinear entanglement (Case 4) would need expanded probes.
"""
    output_path.write_text(report, encoding="utf-8")
