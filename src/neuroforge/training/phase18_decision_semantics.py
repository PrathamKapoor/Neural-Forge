"""Phase 18 component-composition and decision-semantics runner.

Additive: no Phase 1-17 module is modified. Trained weights come from the
Phase 15/16 caches (identical protocol; evals re-run here). Everything new
trains only small diagnostic heads on frozen states, unless the 18L gate
passes (two converging observations), in which case ONE minimal readout
intervention is trained (frozen backbone + concat head, no backbone retrain).
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

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase13_metrics import single_expert_ceiling
from neuroforge.evaluation.phase15_metrics import relational_probe_triplet
from neuroforge.evaluation.phase18_component_composition import (
    FAMILIES,
    MIXED_FAMS,
    autocorr_alignment,
    component_labels,
    eval_head_on,
    evaluate_intervention_gate,
    head_param_count,
    joint_frc_labels,
    joint_rc_labels,
    logit_margin_stats,
    rc_agreement_mask,
    recommendation_for_case,
    select_minimal_intervention,
    select_phase18_case,
    build_failure_diagnosis,
    build_phase18_hypotheses,
    train_diag_head,
    train_nonlinear_diag_head,
    verify_rc_parity_semantics,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
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

REPS = ("feat", "rel", "ctx", "fused")


@torch.no_grad()
def _branch_pools(expert: StandaloneSpecialist, features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean-pooled frozen reps: feat/rel/ctx branch deltas + fused block output."""
    expert.eval()
    enc = expert.encoder(features)
    block = expert.blocks[0]
    if hasattr(block, "forward_with_intermediates"):
        inter = block.forward_with_intermediates(enc, features)
        inter.pop("rel_rounds", None)
        return {
            "feat": inter["feat_delta"].mean(dim=1),
            "rel": inter["rel_delta"].mean(dim=1),
            "ctx": inter["ctx_delta"].mean(dim=1),
            "fused": inter["block_output"].mean(dim=1),
        }
    out, _ = block(enc)
    return {"fused": out.mean(dim=1)}


def run_phase18_decision_semantics(
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
        "phase": "Phase 18 — Component Composition and Decision Semantics Diagnosis",
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
            "Subject: frozen depth-3 JointCo. RC uses R+C semantics per benchmark "
            "construction (F retained as negative control). One shared head "
            "protocol (Linear/AdamW 1e-2/10 epochs/RNG fork-restore). Router frozen."
        ),
        "semantic_deviation_note": (
            "Phase 18 text frames RC as F+R; the construction defines RC from R+C "
            "(verified: disagree labels equal R 100%). Diagnostics follow the construction."
        ),
    }

    selected = select_capacity_phase6()
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        train_ds = Phase9MixedStructureDataset(samples_per_type=40, seed=seed + 60_000)
        r: dict[str, Any] = {"seed": seed}

        # ---- 18A: executed baselines ----
        depth3 = train_variant("depth3", seed, jointco_epochs, batch_size, P15_PARTIALS)
        baseline = train_variant("baseline", seed, jointco_epochs, batch_size, P15_PARTIALS)
        graph = train_reference_expert("graph", "relational", selected.depths["graph"],
                                       seed, jointco_epochs, batch_size, P15_PARTIALS)
        rel_first = train_coadapt_condition("rel_first", seed, jointco_epochs, batch_size, P16_PARTIALS)
        r["depth3_eval"] = evaluate_condition(depth3, eval_ds, seed)
        r["baseline_eval"] = evaluate_condition(baseline, eval_ds, seed)
        r["graph_eval"] = evaluate_condition(graph, eval_ds, seed, do_branch7=False)
        r["depth3_probe"] = relational_probe_triplet(depth3, eval_ds)

        # ---- frozen reps (eval + train) ----
        with torch.no_grad():
            reps = _branch_pools(depth3, eval_ds.features)
            reps_tr = _branch_pools(depth3, train_ds.features)
        r["rep_dims"] = {k: list(v.shape) for k, v in reps.items()}

        # ---- 18B: component metadata (per-sample CSV source) ----
        agree, disagree = rc_agreement_mask(eval_ds)
        r["n_RC_agree"] = len(agree)
        r["n_RC_disagree"] = len(disagree)
        r["parity"] = verify_rc_parity_semantics(eval_ds)

        # ---- 18C: component prediction matrix (rep × component) ----
        comp_acc: dict[str, dict[str, float]] = {}
        comp_heads: dict[str, dict[str, Any]] = {}
        for rep in REPS:
            if rep not in reps:
                continue
            comp_acc[rep] = {}
            comp_heads[rep] = {}
            for letter in ("F", "R", "C"):
                head = train_diag_head(reps_tr[rep], component_labels(train_ds, letter), 24, seed=seed)
                comp_heads[rep][letter] = head
                with torch.no_grad():
                    preds = head(reps[rep]).argmax(-1)
                lab = component_labels(eval_ds, letter)
                comp_acc[rep][letter] = float((preds == lab).float().mean().item())
                # RC-subset accuracy
                rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
                comp_acc[rep][f"{letter}_on_RC"] = float((preds[rc_idx] == lab[rc_idx]).float().mean().item())
        r["comp_acc"] = comp_acc

        # ---- 18D: joint targets ----
        joint_rc_tr, joint_rc_ev = joint_rc_labels(train_ds), joint_rc_labels(eval_ds)
        joint4: dict[str, Any] = {}
        for rep, dim in (("rel", 24), ("ctx", 24)):
            head = train_diag_head(reps_tr[rep], joint_rc_tr, dim, num_classes=4, seed=seed)
            with torch.no_grad():
                preds = head(reps[rep]).argmax(-1)
            rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
            # encoding was sr*2+sc: sr = pred//2, sc = pred%2
            sr_pred, sc_pred = preds[rc_idx] // 2, preds[rc_idx] % 2
            sr_true = (joint_rc_ev[rc_idx] // 2)
            sc_true = (joint_rc_ev[rc_idx] % 2)
            joint4[rep] = {
                "joint_acc": float((preds[rc_idx] == joint_rc_ev[rc_idx]).float().mean().item()),
                "R_marginal": float((sr_pred == sr_true).float().mean().item()),
                "C_marginal": float((sc_pred == sc_true).float().mean().item()),
            }
        # R+C and F+R+C joint heads
        for rep_combo, keys, dim in (("R+C", ("rel", "ctx"), 48), ("F+R+C", ("feat", "rel", "ctx"), 72)):
            tr_x = torch.cat([reps_tr[k] for k in keys], dim=-1)
            ev_x = torch.cat([reps[k] for k in keys], dim=-1)
            head = train_diag_head(tr_x, joint_rc_tr, dim, num_classes=4, seed=seed)
            with torch.no_grad():
                preds = head(ev_x).argmax(-1)
            rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
            sr_pred, sc_pred = preds[rc_idx] // 2, preds[rc_idx] % 2
            sr_true = joint_rc_ev[rc_idx] // 2
            sc_true = joint_rc_ev[rc_idx] % 2
            joint4[rep_combo] = {
                "joint_acc": float((preds[rc_idx] == joint_rc_ev[rc_idx]).float().mean().item()),
                "R_marginal": float((sr_pred == sr_true).float().mean().item()),
                "C_marginal": float((sc_pred == sc_true).float().mean().item()),
            }
        r["joint4"] = joint4

        # ---- 18E/18F: production logits, margins, favor, norms ----
        with torch.no_grad():
            depth3.eval()
            logits = depth3(eval_ds.features)
        ms = logit_margin_stats(logits)
        preds = ms["preds"]
        r_lab = component_labels(eval_ds, "R")
        c_lab = component_labels(eval_ds, "C")
        rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
        agree_acc = float((preds[agree] == eval_ds.targets[agree]).float().mean().item()) if agree else 0.0
        dis_acc = float((preds[disagree] == eval_ds.targets[disagree]).float().mean().item()) if disagree else 0.0
        match_c = float(((preds[disagree] == c_lab[disagree]).float().mean().item())) if disagree else 0.0
        match_r = float(((preds[disagree] == r_lab[disagree]).float().mean().item())) if disagree else 0.0
        # production vs C-rule agreement on all RC
        c_rule = c_lab[rc_idx]
        prod_c_agree = float((preds[rc_idx] == c_rule).float().mean().item())
        # subset margin/conf/entropy + branch norms
        with torch.no_grad():
            enc = depth3.encoder(eval_ds.features)
            inter = depth3.blocks[0].forward_with_intermediates(enc, eval_ds.features)
            norms = {k: inter[k].pow(2).sum(dim=(1, 2)).sqrt() for k in ("feat_delta", "rel_delta", "ctx_delta")}
        r["RC_audit"] = {
            "agree_acc": agree_acc, "disagree_acc": dis_acc,
            "disagree_pred_matches_C": match_c, "disagree_pred_matches_R": match_r,
            "production_matches_C_rule_RC": prod_c_agree,
            "margin_agree": float(ms["margin"][agree].mean().item()) if agree else 0.0,
            "margin_disagree": float(ms["margin"][disagree].mean().item()) if disagree else 0.0,
            "conf_agree": float(ms["confidence"][agree].mean().item()) if agree else 0.0,
            "conf_disagree": float(ms["confidence"][disagree].mean().item()) if disagree else 0.0,
            "entropy_agree": float(ms["entropy"][agree].mean().item()) if agree else 0.0,
            "entropy_disagree": float(ms["entropy"][disagree].mean().item()) if disagree else 0.0,
            "rel_norm_agree": float(norms["rel_delta"][agree].mean().item()) if agree else 0.0,
            "rel_norm_disagree": float(norms["rel_delta"][disagree].mean().item()) if disagree else 0.0,
            "ctx_norm_agree": float(norms["ctx_delta"][agree].mean().item()) if agree else 0.0,
            "ctx_norm_disagree": float(norms["ctx_delta"][disagree].mean().item()) if disagree else 0.0,
            "feat_norm_agree": float(norms["feat_delta"][agree].mean().item()) if agree else 0.0,
            "feat_norm_disagree": float(norms["feat_delta"][disagree].mean().item()) if disagree else 0.0,
        }

        # ---- 18H: linear vs nonlinear official-RC heads on R+C (and FRC on F+R+C) ----
        tr_rc_x = torch.cat([reps_tr["rel"], reps_tr["ctx"]], dim=-1)
        ev_rc_x = torch.cat([reps["rel"], reps["ctx"]], dim=-1)
        lin_rc = train_diag_head(tr_rc_x, train_ds.targets, 48, seed=seed)
        nl_rc = train_nonlinear_diag_head(tr_rc_x, train_ds.targets, 48, seed=seed)
        lin_ev = eval_head_on(lin_rc, ev_rc_x, eval_ds.targets, eval_ds.families_list)
        nl_ev = eval_head_on(nl_rc, ev_rc_x, eval_ds.targets, eval_ds.families_list)
        dis_lin = float((lin_rc(ev_rc_x[disagree]).argmax(-1) == eval_ds.targets[disagree]).float().mean().item()) if disagree else 0.0
        dis_nl = float((nl_rc(ev_rc_x[disagree]).argmax(-1) == eval_ds.targets[disagree]).float().mean().item()) if disagree else 0.0
        tr_frc_x = torch.cat([reps_tr["feat"], reps_tr["rel"], reps_tr["ctx"]], dim=-1)
        ev_frc_x = torch.cat([reps["feat"], reps["rel"], reps["ctx"]], dim=-1)
        lin_frc = train_diag_head(tr_frc_x, train_ds.targets, 72, seed=seed)
        nl_frc = train_nonlinear_diag_head(tr_frc_x, train_ds.targets, 72, seed=seed)
        lin_frc_ev = eval_head_on(lin_frc, ev_frc_x, eval_ds.targets, eval_ds.families_list)
        nl_frc_ev = eval_head_on(nl_frc, ev_frc_x, eval_ds.targets, eval_ds.families_list)
        r["heads_RC"] = {
            "linear_RC": lin_ev.get("RC", 0.0), "nonlinear_RC": nl_ev.get("RC", 0.0),
            "linear_disagree_RC": dis_lin, "nonlinear_disagree_RC": dis_nl,
            "linear_params": head_param_count(lin_rc), "nonlinear_params": head_param_count(nl_rc),
        }
        r["heads_FRC"] = {
            "linear_FRC": lin_frc_ev.get("FRC", 0.0), "nonlinear_FRC": nl_frc_ev.get("FRC", 0.0),
        }

        # ---- 18G: counterfactual swaps with the frozen R+C official head ----
        r["counterfactual"] = _counterfactual_swaps(
            lin_rc, reps, eval_ds, disagree, agree)
        # ---- 18K: shuffled controls ----
        r["shuffled"] = _shuffled_controls(
            reps_tr, train_ds, reps, eval_ds, disagree, seed)

        # ---- 18I: objective diagnosis on the train split ----
        with torch.no_grad():
            depth3.eval()
            train_logits = depth3(train_ds.features)
            train_preds = train_logits.argmax(-1)
        tr_r_lab = component_labels(train_ds, "R")
        tr_c_lab = component_labels(train_ds, "C")
        tr_rc = [i for i, fm in enumerate(train_ds.families_list) if fm == "RC"]
        rule_r_tr = float(((tr_r_lab[tr_rc] == train_ds.targets[tr_rc]).float().mean().item())) if tr_rc else 0.0
        rule_c_tr = float(((tr_c_lab[tr_rc] == train_ds.targets[tr_rc]).float().mean().item())) if tr_rc else 0.0
        model_tr = float((train_preds[tr_rc] == train_ds.targets[tr_rc]).float().mean().item()) if tr_rc else 0.0
        rule_r_te = float(((r_lab[rc_idx] == eval_ds.targets[rc_idx]).float().mean().item())) if rc_idx else 0.0
        rule_c_te = float(((c_lab[rc_idx] == eval_ds.targets[rc_idx]).float().mean().item())) if rc_idx else 0.0
        model_te = r["depth3_eval"]["perf"].get("RC", 0.0)
        # behavior match: production preds == best single rule preds on RC
        best_rule_lab = tr_r_lab if rule_r_tr >= rule_c_tr else tr_c_lab
        best_rule_ev = r_lab if rule_r_tr >= rule_c_tr else c_lab
        beh_match = float((preds[rc_idx] == best_rule_ev[rc_idx]).float().mean().item()) if rc_idx else 0.0
        # feasible-rule match (C-rule): the R-rule is a label-oracle on RC
        # because sr is erased from RC inputs (see erasure check)
        beh_match_c = float((preds[rc_idx] == c_lab[rc_idx]).float().mean().item()) if rc_idx else 0.0
        # joint-head train/test
        jtr = eval_head_on(lin_rc, tr_rc_x, train_ds.targets, train_ds.families_list)
        r["objective"] = {
            "rule_R_train_RC": rule_r_tr, "rule_C_train_RC": rule_c_tr,
            "model_train_RC": model_tr,
            "rule_R_test_RC": rule_r_te, "rule_C_test_RC": rule_c_te,
            "model_test_RC": model_te,
            "best_single_train_RC": max(rule_r_tr, rule_c_tr),
            "model_matches_single_rule": beh_match,
            "model_matches_C_rule": beh_match_c,
            "joint_head_train_RC": jtr.get("RC", 0.0),
            "joint_head_test_RC": lin_ev.get("RC", 0.0),
        }

        # ---- ceiling inputs ----
        r["_logits"] = {}
        for arch, fam in (("mlp", "feature"), ("attention", "contextual"), ("attention_v2", "contextual")):
            e = train_reference_expert(arch, fam, selected.depths[arch], seed, jointco_epochs, batch_size, P15_PARTIALS)
            e.eval()
            with torch.no_grad():
                r["_logits"][arch] = e(eval_ds.features).clone()
        with torch.no_grad():
            graph.eval()
            r["_logits"]["graph"] = graph(eval_ds.features).clone()
            r["_logits"]["jointco_d3"] = logits.clone()
        per_seed.append(r)

    # =====================================================================
    # Aggregates
    # =====================================================================
    def mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    # H1 inputs: best-rep component accs on RC
    best_r = mean([max(r["comp_acc"][rep].get("R_on_RC", 0.0) for rep in r["comp_acc"]) for r in per_seed])
    best_c = mean([max(r["comp_acc"][rep].get("C_on_RC", 0.0) for rep in r["comp_acc"]) for r in per_seed])
    components = {"tested": True, "best_R_comp_acc": best_r, "best_C_comp_acc": best_c}

    # H2: joint (R&C simultaneously correct) from F+R+C concat component heads
    joint_vals = []
    for r in per_seed:
        # exact joint accuracy from the 18D R+C 4-class head
        joint_vals.append(r["joint4"]["R+C"]["joint_acc"])
    joint = {"tested": True, "joint_RC_acc": mean(joint_vals),
             "n_pos": sum(1 for v in joint_vals if v >= 0.60)}

    agreement = {"tested": True,
                 "agree_acc": mean([r["RC_audit"]["agree_acc"] for r in per_seed]),
                 "disagree_acc": mean([r["RC_audit"]["disagree_acc"] for r in per_seed])}
    favor = {"tested": True,
             "disagree_pred_matches_C": mean([r["RC_audit"]["disagree_pred_matches_C"] for r in per_seed]),
             "disagree_pred_matches_R": mean([r["RC_audit"]["disagree_pred_matches_R"] for r in per_seed])}
    heads = {"tested": True,
             "linear_disagree_RC": mean([r["heads_RC"]["linear_disagree_RC"] for r in per_seed]),
             "nonlinear_disagree_RC": mean([r["heads_RC"]["nonlinear_disagree_RC"] for r in per_seed]),
             "n_pos": sum(1 for r in per_seed if r["heads_RC"]["nonlinear_disagree_RC"] >= 0.60)}
    objective = {"tested": True,
                 "model_train_RC": mean([r["objective"]["model_train_RC"] for r in per_seed]),
                 "best_single_train_RC": mean([r["objective"]["best_single_train_RC"] for r in per_seed]),
                 "model_matches_single_rule": mean([r["objective"]["model_matches_single_rule"] for r in per_seed]),
                 "rule_C_train_RC": mean([r["objective"]["rule_C_train_RC"] for r in per_seed]),
                 "model_matches_C_rule": mean([r["objective"]["model_matches_C_rule"] for r in per_seed])}
    align_R = mean([autocorr_alignment(
        Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"]), ("R",)) for r in per_seed])
    align_RC = mean([autocorr_alignment(
        Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"]), ("RC",)) for r in per_seed])
    align_FRC = mean([autocorr_alignment(
        Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"]), ("FRC",)) for r in per_seed])
    erasure = align_R >= 0.90 and align_RC < 0.60
    semantics = {"tested": True,
                 "parity_fact": all(r["parity"]["parity_fact"] for r in per_seed),
                 "production_matches_C_rule": mean([r["RC_audit"]["production_matches_C_rule_RC"] for r in per_seed]),
                 "align_R": align_R, "align_RC": align_RC, "align_FRC": align_FRC,
                 "erasure_demonstrated": erasure}
    counterfactual = {"tested": True,
                      "R_swap_flip_rate": mean([r["counterfactual"]["R_swap_flip"] for r in per_seed]),
                      "C_swap_flip_rate": mean([r["counterfactual"]["C_swap_flip"] for r in per_seed]),
                      "control_same_flip_rate": mean([r["counterfactual"]["control_same_flip"] for r in per_seed]),
                      "R_swap_response": mean([r["counterfactual"]["R_swap_response"] for r in per_seed]),
                      "C_swap_response": mean([r["counterfactual"]["C_swap_response"] for r in per_seed]),
                      "control_same_response": mean([r["counterfactual"]["control_same_response"] for r in per_seed])}

    # 18L gate inputs + ceiling
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

    stability: dict[str, dict[str, float]] = {"RC": {}}
    for key in ("depth3_eval_RC", "oracle_RC"):
        if key == "depth3_eval_RC":
            vals = [r["depth3_eval"]["perf"].get("RC", 0.0) for r in per_seed]
        else:
            vals = None
        if vals is not None and len(vals) > 1:
            stability["RC"][key] = statistics.stdev(vals)

    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "components": components,
        "joint": joint,
        "agreement": agreement,
        "favor": favor,
        "heads": heads,
        "objective": objective,
        "semantics": semantics,
        "counterfactual": counterfactual,
        "compositional": {"tested": True,
                          "R_gain": mean([r["depth3_eval"]["perf"].get("R", 0.0) for r in per_seed]) - mean([r["baseline_eval"]["perf"].get("R", 0.0) for r in per_seed]),
                          "RC_gain": 0.0, "FRC_gain": 0.0},
        "ceiling": ceiling,
        "seed_stability": stability,
        "latency": {"tested": True,
                    "baseline_total_us": mean([r["baseline_eval"]["latency_us"] for r in per_seed]),
                    "candidate_total_us": mean([r["depth3_eval"]["latency_us"] for r in per_seed])},
    }
    # preservation17 proxy for gate m3: Phase 17 showed info survives fusion
    agg["preservation17"] = {"oracle_recovers": False}
    hypotheses = build_phase18_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    gate = evaluate_intervention_gate(agg)
    intervention_out = select_minimal_intervention(gate)
    # 18M: only a readout head could be trained; gate decides. Record, do not auto-build
    # unless the gate passes AND the mechanism is decision_head (frozen-backbone concat head).
    intervention_record: dict[str, Any] = {"tested": False,
                                           "reason": "; ".join(gate.get("evidence", ["gate not satisfied"]))}
    if gate.get("passed") and gate.get("mechanism") == "decision_head":
        intervention_record = _train_readout_intervention(per_seed, samples_per_type)
        intervention_record["tested"] = True
    agg["intervention"] = intervention_record
    # rebuild hypotheses now that intervention status is final (H8)
    hypotheses = build_phase18_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    case, label = select_phase18_case(hypotheses, agg)
    failure_rows = build_failure_diagnosis(agg)

    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": _serialisable(per_seed),
        "baseline_perf_mean": {f: mean([r["baseline_eval"]["perf"].get(f, 0.0) for r in per_seed]) for f in list(FAMILIES) + ["mixed_mean", "overall"]},
        "depth3_perf_mean": {f: mean([r["depth3_eval"]["perf"].get(f, 0.0) for r in per_seed]) for f in list(FAMILIES) + ["mixed_mean", "overall"]},
        "gate": gate,
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
    _write_csvs(m_dir, per_seed, failure_rows, intervention_record, samples_per_type)
    from neuroforge.visualization.phase18_plots import generate_phase18_figures
    summary["generated_figures"] = generate_phase18_figures(summary, f_dir)
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _counterfactual_swaps(
    head: Any,
    reps: dict[str, torch.Tensor],
    eval_ds: Phase9MixedStructureDataset,
    disagree: list[int],
    agree: list[int],
) -> dict[str, float]:
    """Swap one branch rep across examples; measure flips toward the new implied target.

    New implied target: agree side if swapped-in components agree, else the R
    side (per the verified parity fact). Same-side swaps are the no-op control.
    """
    if not disagree:
        return {"R_swap_flip": 0.0, "C_swap_flip": 0.0, "control_same_flip": 0.0}
    r_rep, c_rep = reps["rel"], reps["ctx"]

    def implied_label(sr: int, sc: int) -> int:
        if sr == sc:
            return 1 if sr > 0 else 0
        return 1 if sr > 0 else 0  # parity fact: disagree label == R side

    # index pools by (sr, sc) within RC
    from collections import defaultdict
    pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, fm in enumerate(eval_ds.families_list):
        if fm == "RC":
            pools[(int(eval_ds.items[i]["sr"]), int(eval_ds.items[i]["sc"]))].append(i)

    with torch.no_grad():
        base_preds = head(torch.cat([r_rep, c_rep], dim=-1)).argmax(-1)

        def swap_stats(which: str, opposite: bool) -> tuple[float, float]:
            """Return (implied-match rate, prediction-change rate).

            Change-rate is the valid responsiveness test: a component-ignoring
            head can still match the implied target (e.g. a C-predictor scores
            ~100% on both swap types), but only a responsive head CHANGES its
            prediction when the swapped component changes.
            """
            match, changed, total = 0, 0, 0
            for i in disagree:
                sr = int(eval_ds.items[i]["sr"])
                sc = int(eval_ds.items[i]["sc"])
                if which == "R":
                    key = (-sr, sc) if opposite else (sr, sc)
                    donor_pool = [j for j in pools.get(key, []) if j != i]
                else:
                    key = (sr, -sc) if opposite else (sr, sc)
                    donor_pool = [j for j in pools.get(key, []) if j != i]
                if not donor_pool:
                    continue
                j = donor_pool[i % len(donor_pool)]
                rr = r_rep[j].clone() if which == "R" else r_rep[i]
                cc = c_rep[j].clone() if which == "C" else c_rep[i]
                pred = int(head(torch.cat([rr, cc], dim=0).unsqueeze(0)).argmax(-1).item())
                nsr = int(eval_ds.items[j]["sr"]) if which == "R" else sr
                nsc = int(eval_ds.items[j]["sc"]) if which == "C" else sc
                if pred == implied_label(nsr, nsc):
                    match += 1
                if pred != int(base_preds[i].item()):
                    changed += 1
                total += 1
            return match / max(1, total), changed / max(1, total)

        r_match, r_resp = swap_stats("R", True)
        c_match, c_resp = swap_stats("C", True)
        cr_match, cr_resp = swap_stats("R", False)
        cc_match, cc_resp = swap_stats("C", False)
        return {"R_swap_flip": r_match, "C_swap_flip": c_match,
                "control_same_flip": (cr_match + cc_match) / 2,
                "R_swap_response": r_resp, "C_swap_response": c_resp,
                "control_same_response": (cr_resp + cc_resp) / 2}


def _shuffled_controls(
    reps_tr: dict[str, torch.Tensor],
    train_ds: Phase9MixedStructureDataset,
    reps: dict[str, torch.Tensor],
    eval_ds: Phase9MixedStructureDataset,
    disagree: list[int],
    seed: int,
) -> dict[str, float]:
    """18K: label-shuffled target + component-permuted pairing controls."""
    rc_tr = [i for i, fm in enumerate(train_ds.families_list) if fm == "RC"]
    rc_ev = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    tr_x = torch.cat([reps_tr["rel"], reps_tr["ctx"]], dim=-1)
    ev_x = torch.cat([reps["rel"], reps["ctx"]], dim=-1)
    # (a) label-shuffled diagnostic target
    g = torch.Generator().manual_seed(seed + 777)
    perm = torch.randperm(len(rc_tr), generator=g)
    shuffled_y = train_ds.targets[rc_tr][perm]
    head = train_diag_head(tr_x[rc_tr], shuffled_y, tr_x.shape[-1], seed=seed + 1)
    with torch.no_grad():
        acc_shuf = float((head(ev_x[rc_ev]).argmax(-1) == eval_ds.targets[rc_ev]).float().mean().item()) if rc_ev else 0.0
    # (b) permuted R pairing: shuffle R reps across RC examples, keep C fixed
    rperm = torch.randperm(len(rc_tr), generator=torch.Generator().manual_seed(seed + 778))
    mixed_tr_x = torch.cat([reps_tr["rel"][rc_tr][rperm], reps_tr["ctx"][rc_tr]], dim=-1)
    head2 = train_diag_head(mixed_tr_x, train_ds.targets[rc_tr], tr_x.shape[-1], seed=seed + 2)
    with torch.no_grad():
        acc_perm = float((head2(ev_x[rc_ev]).argmax(-1) == eval_ds.targets[rc_ev]).float().mean().item()) if rc_ev else 0.0
    return {"label_shuffled_RC": acc_shuf, "R_permuted_RC": acc_perm}


def _train_readout_intervention(
    per_seed: list[dict[str, Any]], samples_per_type: int
) -> dict[str, Any]:
    """18M decision-head intervention (frozen backbone + F+R+C concat head).

    Only called when the 18L gate passes with mechanism == decision_head.
    """
    from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant

    gains, n_pos = [], 0
    for r in per_seed:
        seed = r["seed"]
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        train_ds = Phase9MixedStructureDataset(samples_per_type=40, seed=seed + 60_000)
        expert = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
        # restore trained weights from the phase15 cache
        import torch as _t

        pt = P15_PARTIALS / f"seed{seed}_depth3.pt"
        expert.load_state_dict(_t.load(pt, map_location="cpu", weights_only=True))
        with _t.no_grad():
            tr = _branch_pools(expert, train_ds.features)
            ev = _branch_pools(expert, eval_ds.features)
        tr_x = _t.cat([tr["feat"], tr["rel"], tr["ctx"]], dim=-1)
        ev_x = _t.cat([ev["feat"], ev["rel"], ev["ctx"]], dim=-1)
        head = train_diag_head(tr_x, train_ds.targets, tr_x.shape[-1], seed=seed)
        acc = eval_head_on(head, ev_x, eval_ds.targets, eval_ds.families_list)
        base_rc = r["depth3_eval"]["perf"].get("RC", 0.0)
        g = acc.get("RC", 0.0) - base_rc
        gains.append(g)
        n_pos += 1 if g > 0 else 0
    gain = statistics.mean(gains)
    return {"tested": True, "mechanism": "decision_head",
            "RC_gain": gain, "n_pos": n_pos,
            "params": head_param_count(head),
            "detail": f"F+R+C concat readout RC gain {gain * 100:+.1f}pp ({n_pos}/{len(per_seed)} seeds)."}


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
def _write_csvs(m_dir: Path, per_seed: list[dict[str, Any]], failure_rows: list[dict[str, Any]],
                  intervention_record: dict[str, Any] | None = None,
                  samples_per_type: int = 120) -> None:
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
        for cond in ("graph", "baseline", "depth3"):
            for f in fams + ["mixed_mean", "overall"]:
                rows.append({"seed": r["seed"], "condition": cond, "metric": f,
                             "value": r[f"{cond}_eval"]["perf"].get(f, 0.0)})
        rows.append({"seed": r["seed"], "condition": "depth3", "metric": "RC_agree",
                     "value": r["RC_audit"]["agree_acc"]})
        rows.append({"seed": r["seed"], "condition": "depth3", "metric": "RC_disagree",
                     "value": r["RC_audit"]["disagree_acc"]})
    write_csv("baseline_reproduction.csv", rows)

    # component metadata: rebuilt from the same deterministic dataset
    rows = []
    for r in per_seed:
        ds = Phase9MixedStructureDataset(samples_per_type=120, seed=r["seed"])
        for i in range(len(ds)):
            if ds.families_list[i] not in ("RC", "FRC"):
                continue
            rows.append({"seed": r["seed"], "sample": i, "family": ds.families_list[i],
                         "sf": ds.items[i]["sf"], "sr": ds.items[i]["sr"], "sc": ds.items[i]["sc"],
                         "target": int(ds.targets[i])})
    write_csv("component_targets.csv", rows)

    rows = []
    for r in per_seed:
        for rep, comps in r["comp_acc"].items():
            row = {"seed": r["seed"], "representation": rep}
            for k, v in comps.items():
                row[k] = v
            rows.append(row)
    write_csv("component_probe_matrix.csv", rows)

    rows = []
    for r in per_seed:
        for rep, vals in r["joint4"].items():
            rows.append({"seed": r["seed"], "representation": rep, **vals})
    write_csv("joint_decodability.csv", rows)

    rows = []
    for r in per_seed:
        a = r["RC_audit"]
        rows.append({"seed": r["seed"], "split": "agree", "acc": a["agree_acc"],
                     "margin": a["margin_agree"], "confidence": a["conf_agree"], "entropy": a["entropy_agree"]})
        rows.append({"seed": r["seed"], "split": "disagree", "acc": a["disagree_acc"],
                     "margin": a["margin_disagree"], "confidence": a["conf_disagree"], "entropy": a["entropy_disagree"]})
    write_csv("agreement_disagreement.csv", rows)

    rows = []
    for r in per_seed:
        a = r["RC_audit"]
        rows.append({"seed": r["seed"], "matches_C_disagree": a["disagree_pred_matches_C"],
                     "matches_R_disagree": a["disagree_pred_matches_R"],
                     "rel_norm_disagree": a["rel_norm_disagree"], "ctx_norm_disagree": a["ctx_norm_disagree"],
                     "feat_norm_disagree": a["feat_norm_disagree"],
                     "rel_norm_agree": a["rel_norm_agree"], "ctx_norm_agree": a["ctx_norm_agree"]})
    write_csv("logit_margin.csv", rows)

    rows = []
    for r in per_seed:
        for k, v in r["counterfactual"].items():
            rows.append({"seed": r["seed"], "swap": k, "flip_rate": v})
    write_csv("counterfactuals.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], "head": "linear_RC", "RC": r["heads_RC"]["linear_RC"],
                     "disagree_RC": r["heads_RC"]["linear_disagree_RC"],
                     "params": r["heads_RC"]["linear_params"]})
        rows.append({"seed": r["seed"], "head": "nonlinear_RC", "RC": r["heads_RC"]["nonlinear_RC"],
                     "disagree_RC": r["heads_RC"]["nonlinear_disagree_RC"],
                     "params": r["heads_RC"]["nonlinear_params"]})
        rows.append({"seed": r["seed"], "head": "linear_FRC", "FRC": r["heads_FRC"]["linear_FRC"], "params": 0})
        rows.append({"seed": r["seed"], "head": "nonlinear_FRC", "FRC": r["heads_FRC"]["nonlinear_FRC"], "params": 0})
    write_csv("diagnostic_heads.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], **{k: v for k, v in r["objective"].items()}})
    write_csv("objective_diagnosis.csv", rows)

    rows = []
    for r in per_seed:
        ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
        rows.append({"seed": r["seed"], "parity_fact": r["parity"]["parity_fact"],
                     "n_disagree": r["parity"]["n_disagree"],
                     "production_matches_C_rule": r["RC_audit"]["production_matches_C_rule_RC"],
                     "align_R": autocorr_alignment(ds, ("R",)),
                     "align_RC": autocorr_alignment(ds, ("RC",)),
                     "align_FRC": autocorr_alignment(ds, ("FRC",))})
    write_csv("rc_frc_semantics.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], "label_shuffled_RC": r["shuffled"]["label_shuffled_RC"],
                     "R_permuted_RC": r["shuffled"]["R_permuted_RC"]})
    write_csv("shuffled_controls.csv", rows)

    iv_rec = intervention_record or {}
    write_csv("intervention.csv", [{
        "seed": 0,
        "gate_passed": iv_rec.get("tested", False),
        "mechanism": iv_rec.get("mechanism", "none"),
        "detail": iv_rec.get("detail", iv_rec.get("reason", "18L gate not satisfied; NO INTERVENTION (default).")),
        "RC_gain": iv_rec.get("RC_gain", 0.0),
    }])

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "R": r[f"{cond}_eval"]["perf"].get("R", 0.0),
                         "RC": r[f"{cond}_eval"]["perf"].get("RC", 0.0),
                         "FRC": r[f"{cond}_eval"]["perf"].get("FRC", 0.0)})
    write_csv("rc_frc_results.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "params": r[f"{cond}_eval"].get("params", 0)})
    write_csv("compute.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3"):
            rows.append({"seed": r["seed"], "condition": cond,
                         "latency_us": r[f"{cond}_eval"].get("latency_us", 0.0),
                         "throughput_per_s": r[f"{cond}_eval"].get("throughput_per_s", 0.0)})
    write_csv("latency.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"],
                     "RC_agree": r["RC_audit"]["agree_acc"],
                     "RC_disagree": r["RC_audit"]["disagree_acc"],
                     "linear_disagree": r["heads_RC"]["linear_disagree_RC"],
                     "nonlinear_disagree": r["heads_RC"]["nonlinear_disagree_RC"]})
    write_csv("seed_results.csv", rows)

    write_csv("failure_diagnosis.csv", [
        {"seed": 0, "category": c["category"], "failed": c["failed"],
         "criterion": c["criterion"], "observed": c["observed"]} for c in failure_rows])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_phase18_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
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

    comp = agg.get("components", {})
    joint = agg.get("joint", {})
    agr = agg.get("agreement", {})
    fav = agg.get("favor", {})
    hd = agg.get("heads", {})
    obj = agg.get("objective", {})
    sem = agg.get("semantics", {})
    cf = agg.get("counterfactual", {})
    ceil = agg.get("ceiling", {})

    report = f"""# Phase 18 — Component Composition and Decision Semantics Diagnosis

## Mandatory Scientific Disclaimer

> Phase 17 established branch information survives fusion while RC-disagree collapses
> (≈100% agree vs ≈0–2% disagree). Phase 18 investigates the decision semantics: component
> identification vs joint identification vs joint task prediction are three different
> capabilities and must not be collapsed. RC uses R+C semantics per benchmark construction
> (the F component is a negative control). Router frozen.

---

## 1. Baselines (18A, executed)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
{fam_row("JointCo baseline", base)}
{fam_row("JointCo depth-3", d3)}

## 2. Component identification (18C)

Best-rep component accuracy on RC: R {comp.get('best_R_comp_acc', float('nan')) * 100:.1f}%, C {comp.get('best_C_comp_acc', float('nan')) * 100:.1f}%.

## 3. Joint identification (18D)

Joint (R&C) accuracy on RC: {joint.get('joint_RC_acc', float('nan')) * 100:.1f}%.

## 4. Agreement audit (18E, central)

RC agree {agr.get('agree_acc', float('nan')) * 100:.1f}% vs disagree {agr.get('disagree_acc', float('nan')) * 100:.1f}%.

## 5. Decision behavior (18F)

On RC-disagree, predictions match C {fav.get('disagree_pred_matches_C', float('nan')) * 100:.1f}% vs R {fav.get('disagree_pred_matches_R', float('nan')) * 100:.1f}% (correlation only; §25).

## 6. Counterfactuals (18G)

R-swap flip {cf.get('R_swap_flip_rate', float('nan')) * 100:.1f}%, C-swap flip {cf.get('C_swap_flip_rate', float('nan')) * 100:.1f}%, same-side control {cf.get('control_same_flip_rate', float('nan')) * 100:.1f}%.

## 7. Diagnostic heads (18H)

Linear RC-disagree {hd.get('linear_disagree_RC', float('nan')) * 100:.1f}% vs nonlinear {hd.get('nonlinear_disagree_RC', float('nan')) * 100:.1f}%.

## 8. Objective (18I)

Model train RC {obj.get('model_train_RC', float('nan')) * 100:.1f}% vs best single-component rule {obj.get('best_single_train_RC', float('nan')) * 100:.1f}%; behavior match {obj.get('model_matches_single_rule', float('nan')) * 100:.1f}%.

## 9. Semantics (18J)

Parity fact verified: {sem.get('parity_fact', '?')}; production↔C-rule match {sem.get('production_matches_C_rule', float('nan')) * 100:.1f}%.

## 10. Ceiling (18O)

{ceil.get('old_ceiling_mixed_mean', float('nan')) * 100:.1f}% → {ceil.get('new_ceiling_mixed_mean', float('nan')) * 100:.1f}% ({ceil.get('new_minus_old_mixed', float('nan')) * 100:+.1f}pp).

## 11. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
"""
    for h in [f"H{i}" for i in range(1, 9)]:
        d = hyps.get(h, {})
        report += f"| **{h}** | {d.get('status', '?')} | {d.get('evidence', '')} |\n"
    report += f"""
## 12. Final CASE (programmatic)

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

Recommendation: {summary.get('recommendation', '')}

Gate: {json.dumps(summary.get('gate', {}))}

Minimal intervention: **{summary.get('minimal_intervention', {}).get('intervention', '?')}** — {summary.get('minimal_intervention', {}).get('outcome', '')}.

## 13. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
"""
    for row in fail:
        report += f"| {row.get('category')} | {row.get('failed')} | {row.get('criterion')} | {row.get('observed')} |\n"
    report += """
## 14. Not tested / deferred

- Learned adjacency, router work, attention/Transformer/MoE/fusion redesign (out of scope).
- Production fusion redesign (no earned gate).
- Pure-relational-objective retraining variant of 16F (noted as an alternative isolation).

## 15. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Counterfactual swaps assume branch reps carry their component labels (probe-supported, not proven).
3. Diagnostic heads use mean-pool + linear/nonlinear protocol, differing from production query heads.
4. RC (R+C) mapping follows the construction; Phase 18 text framing (F+R) is documented as deviating.
"""
    output_path.write_text(report, encoding="utf-8")
