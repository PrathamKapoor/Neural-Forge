"""Unit tests for Phase 18 decision-semantics diagnostic blocks."""
from __future__ import annotations

import torch

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase18_component_composition import (
    build_failure_diagnosis,
    build_phase18_hypotheses,
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
    train_diag_head,
    train_nonlinear_diag_head,
    verify_rc_parity_semantics,
)


def _tiny_ds():
    return Phase9MixedStructureDataset(samples_per_type=6, seed=11)


def test_component_labels_binary_and_known_semantics():
    ds = _tiny_ds()
    for letter in ("F", "R", "C"):
        lab = component_labels(ds, letter)
        assert set(lab.tolist()) <= {0, 1}
        assert len(lab) == len(ds)


def test_joint_label_encodings():
    ds = _tiny_ds()
    j4 = joint_rc_labels(ds)
    assert set(j4.tolist()) <= {0, 1, 2, 3}
    # encoding sr*2+sc decodes correctly
    for i in range(len(ds)):
        sr = int(ds.items[i]["sr"] == 1)
        sc = int(ds.items[i]["sc"] == 1)
        assert int(j4[i]) == sr * 2 + sc
    j8 = joint_frc_labels(ds)
    assert set(j8.tolist()) <= set(range(8))


def test_agreement_partition_covers_RC():
    ds = _tiny_ds()
    agree, disagree = rc_agreement_mask(ds)
    n_rc = sum(1 for fm in ds.families_list if fm == "RC")
    assert len(agree) + len(disagree) == n_rc
    for i in agree:
        assert ds.items[i]["sr"] == ds.items[i]["sc"]
    for i in disagree:
        assert ds.items[i]["sr"] != ds.items[i]["sc"]


def test_parity_fact_on_bigger_sample():
    ds = Phase9MixedStructureDataset(samples_per_type=20, seed=11)
    v = verify_rc_parity_semantics(ds)
    assert v["parity_fact"] is True
    assert v["disagree_y_equals_C"] == 0


def test_diag_heads_deterministic():
    torch.manual_seed(0)
    x = torch.randn(24, 16)
    y = torch.tensor([0, 1] * 12, dtype=torch.long)
    h1 = train_diag_head(x, y, 16, epochs=2, seed=5)
    h2 = train_diag_head(x, y, 16, epochs=2, seed=5)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)
    n1 = train_nonlinear_diag_head(x, y, 16, epochs=2, seed=5)
    n2 = train_nonlinear_diag_head(x, y, 16, epochs=2, seed=5)
    for p1, p2 in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(p1, p2)
    assert head_param_count(n1) > head_param_count(h1)


def test_eval_head_ranges():
    ds = _tiny_ds()
    head = train_diag_head(torch.randn(len(ds), 24), ds.targets, 24, epochs=1)
    out = eval_head_on(head, torch.randn(len(ds), 24), ds.targets, ds.families_list)
    assert 0.0 <= out["RC"] <= 1.0


def test_logit_margin_stats_sane():
    logits = torch.tensor([[2.0, 0.5], [0.1, 0.1], [-1.0, 3.0]])
    ms = logit_margin_stats(logits)
    assert ms["margin"][0] > ms["margin"][1]
    assert ms["preds"].tolist() == [0, 0, 1]
    assert all((ms["confidence"] >= 0.5) & (ms["confidence"] <= 1.0))


def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "components": {"tested": True, "best_R_comp_acc": 0.85, "best_C_comp_acc": 0.90},
        "joint": {"tested": True, "joint_RC_acc": 0.80, "n_pos": 3},
        "agreement": {"tested": True, "agree_acc": 1.0, "disagree_acc": 0.05},
        "favor": {"tested": True, "disagree_pred_matches_C": 0.95, "disagree_pred_matches_R": 0.05},
        "heads": {"tested": True, "linear_disagree_RC": 0.10, "nonlinear_disagree_RC": 0.75, "n_pos": 3},
        "objective": {"tested": True, "model_train_RC": 0.52, "best_single_train_RC": 1.0,
                      "rule_C_train_RC": 0.52, "model_matches_single_rule": 0.5,
                      "model_matches_C_rule": 0.95},
        "semantics": {"tested": True, "parity_fact": True, "production_matches_C_rule": 0.97,
                      "align_R": 0.98, "align_RC": 0.50, "align_FRC": 0.52,
                      "erasure_demonstrated": True},
        "counterfactual": {"tested": True, "R_swap_flip_rate": 0.1, "C_swap_flip_rate": 0.8,
                           "control_same_flip_rate": 0.05},
        "compositional": {"tested": True, "R_gain": 0.03, "RC_gain": 0.0, "FRC_gain": 0.0},
        "ceiling": {"tested": True, "old_ceiling_mixed_mean": 0.66, "new_ceiling_mixed_mean": 0.66,
                    "new_minus_old_mixed": 0.0},
        "seed_stability": {"RC": {}},
        "latency": {"tested": True, "baseline_total_us": 60.0, "candidate_total_us": 60.0},
        "intervention": {"tested": False, "reason": "none"},
    }
    agg.update(over)
    return agg


def test_hypotheses_cover_H1_to_H8():
    hyps = build_phase18_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    for d in hyps.values():
        assert d["status"] in ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")


def test_shortcut_mechanism_all_supported():
    hyps = build_phase18_hypotheses(_agg_fixture())
    assert hyps["H1"]["status"] == "SUPPORTED"
    assert hyps["H2"]["status"] == "SUPPORTED"
    assert hyps["H3"]["status"] == "SUPPORTED"
    assert hyps["H4"]["status"] == "SUPPORTED"
    assert hyps["H5"]["status"] == "SUPPORTED"
    assert hyps["H6"]["status"] == "SUPPORTED"
    assert hyps["H7"]["status"] == "SUPPORTED"
    assert hyps["H8"]["status"] == "NOT TESTED"


def test_case_priority_erasure_first():
    # Erasure demonstrated + shortcut demonstrated -> semantics (D) outranks
    # mechanism (C): the shortcut is the feasible optimum on erased labels.
    hyps = build_phase18_hypotheses(_agg_fixture())
    assert hyps["H7"]["status"] == "SUPPORTED"
    assert select_phase18_case(hyps, _agg_fixture())[0] == "CASE D"


def test_case_C_without_erasure():
    agg = _agg_fixture(semantics={"tested": True, "parity_fact": True,
                                  "production_matches_C_rule": 0.97,
                                  "align_R": 0.98, "align_RC": 0.90, "align_FRC": 0.52,
                                  "erasure_demonstrated": False})
    hyps = build_phase18_hypotheses(agg)
    assert hyps["H7"]["status"] == "PARTIALLY SUPPORTED"
    assert select_phase18_case(hyps, agg)[0] == "CASE C"


def test_case_A_when_nothing_available():
    agg = _agg_fixture(components={"tested": True, "best_R_comp_acc": 0.5, "best_C_comp_acc": 0.5},
                       joint={"tested": True, "joint_RC_acc": 0.4, "n_pos": 0},
                       semantics={"tested": True, "parity_fact": True,
                                  "production_matches_C_rule": 0.97,
                                  "align_R": 0.98, "align_RC": 0.90, "align_FRC": 0.52,
                                  "erasure_demonstrated": False})
    assert select_phase18_case(build_phase18_hypotheses(agg), agg)[0] == "CASE A"


def test_case_B_decision_failure():
    agg = _agg_fixture(heads={"tested": True, "linear_disagree_RC": 0.8, "nonlinear_disagree_RC": 0.85, "n_pos": 3},
                       objective={"tested": True, "model_train_RC": 0.7, "best_single_train_RC": 0.5,
                                  "model_matches_single_rule": 0.5},
                       semantics={"tested": True, "parity_fact": True, "production_matches_C_rule": 0.5})
    hyps = build_phase18_hypotheses(agg)
    assert hyps["H5"]["status"] == "NOT SUPPORTED"  # linear already succeeds
    assert select_phase18_case(hyps, agg)[0] == "CASE B"


def test_case_E_joint_unusable():
    agg = _agg_fixture(joint={"tested": True, "joint_RC_acc": 0.4, "n_pos": 0},
                       favor={"tested": True, "disagree_pred_matches_C": 0.5, "disagree_pred_matches_R": 0.5},
                       heads={"tested": True, "linear_disagree_RC": 0.1, "nonlinear_disagree_RC": 0.1, "n_pos": 0},
                       objective={"tested": True, "model_train_RC": 0.7, "best_single_train_RC": 0.5,
                                  "model_matches_single_rule": 0.5},
                       semantics={"tested": True, "parity_fact": True, "production_matches_C_rule": 0.5})
    assert select_phase18_case(build_phase18_hypotheses(agg), agg)[0] == "CASE E"


def test_case_F_empty():
    agg = _agg_fixture(components={"tested": False}, joint={"tested": False},
                       agreement={"tested": False}, favor={"tested": False},
                       heads={"tested": False}, objective={"tested": False},
                       semantics={"tested": False})
    hyps = build_phase18_hypotheses(agg)
    assert select_phase18_case(hyps, agg)[0] == "CASE F"


def test_gate_requires_two_observations():
    agg = _agg_fixture()
    agg["hypotheses"] = build_phase18_hypotheses(agg)
    gate = evaluate_intervention_gate(agg)
    assert gate["passed"] is True
    assert gate["mechanism"] in ("decision_head", "objective_shortcut", "representation_composition")
    # single observation is not enough
    agg2 = _agg_fixture(heads={"tested": True, "linear_disagree_RC": 0.1, "nonlinear_disagree_RC": 0.1, "n_pos": 0},
                        objective={"tested": True, "model_train_RC": 0.7, "best_single_train_RC": 0.5,
                                   "model_matches_single_rule": 0.5})
    agg2["hypotheses"] = build_phase18_hypotheses(agg2)
    gate2 = evaluate_intervention_gate(agg2)
    assert gate2["passed"] is False
    sel = select_minimal_intervention(gate2)
    assert sel["intervention"] == "none"


def test_recommendations_and_failures():
    for case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F"):
        assert isinstance(recommendation_for_case(case), str) and recommendation_for_case(case)
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    for expected in ("component_unavailable", "joint_unavailable", "disagree_collapse",
                     "systematic_favoring", "head_inexpressive", "shortcut_incentive",
                     "semantic_asymmetry", "seed_instability", "counterfactual_unresponsive"):
        assert expected in cats
