"""Unit tests for Phase 16 embedded-vs-dedicated diagnostic building blocks."""
from __future__ import annotations

import torch

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase16_metrics import (
    apply_graph_blocks,
    build_failure_diagnosis,
    build_phase16_hypotheses,
    compare_input_stats,
    embedded_relational_output,
    eval_graph_diagnostic,
    eval_states_head,
    grad_group_of,
    graph_native_input,
    input_equivalence_stats,
    joint_pre_relational,
    probe_r_signal,
    recommendation_for_case,
    record_grad_norms,
    select_minimal_intervention,
    select_phase16_case,
    train_graph_diagnostic_head,
    train_linear_head_on_states,
)
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant
from neuroforge.training.phase16_relational_interface_diagnosis import (
    _freeze_to_relational_only,
    _unfreeze_to_full,
)


def _tiny_ds():
    return Phase9MixedStructureDataset(samples_per_type=4, seed=11)


def test_input_equivalence_stats_and_comparison():
    ds = _tiny_ds()
    ex = build_expert_for_variant(None)
    g = StandaloneSpecialist("graph", depth=1)
    with torch.no_grad():
        a = joint_pre_relational(ex, ds.features)
        b = graph_native_input(g, ds.features)
    sa = input_equivalence_stats(a, ds.features, "joint")
    sb = input_equivalence_stats(b, ds.features, "native")
    assert sa["shape"] == [len(ds), 12, 24]
    assert sb["shape"] == [len(ds), 12, 24]
    comp = compare_input_stats(sb, sa)
    assert comp["shape_equal"] is True
    assert comp["marker_magnitude_equal"] is True
    assert comp["global_std_ratio_b_over_a"] > 0


def test_graph_on_joint_input_shapes():
    ds = _tiny_ds()
    g = StandaloneSpecialist("graph", depth=1)
    ex = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
    with torch.no_grad():
        pre = joint_pre_relational(ex, ds.features)
        out = apply_graph_blocks(g, pre)
        rel = embedded_relational_output(ex, ds.features)
    assert out.shape == pre.shape
    assert rel.shape == pre.shape


def test_head_protocol_deterministic():
    torch.manual_seed(0)
    x = torch.randn(24, 24)
    y = torch.tensor([0, 1] * 12, dtype=torch.long)
    h1 = train_linear_head_on_states(x, y, 24, epochs=2, seed=5)
    h2 = train_linear_head_on_states(x, y, 24, epochs=2, seed=5)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)


def test_eval_states_head_ranges():
    ds = _tiny_ds()
    head = train_linear_head_on_states(torch.randn(len(ds), 24), ds.targets, 24, epochs=1)
    out = eval_states_head(head, torch.randn(len(ds), 24), ds.targets, ds.families_list)
    assert 0.0 <= out["R"] <= 1.0


def test_graph_diagnostic_trains_and_evals():
    ds = _tiny_ds()
    gb, head = train_graph_diagnostic_head(torch.randn(len(ds), 12, 24), ds.targets, epochs=1)
    out = eval_graph_diagnostic(gb, head, torch.randn(len(ds), 12, 24), ds.targets, ds.families_list)
    assert 0.0 <= out["R"] <= 1.0


def test_probe_r_signal_range():
    ds = _tiny_ds()
    assert 0.0 <= probe_r_signal(torch.randn(len(ds), 12, 24), ds) <= 1.0


def test_freeze_unfreeze_helpers():
    ex = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
    _freeze_to_relational_only(ex)
    assert ex.blocks[0].feature_scale.item() == 0.0
    assert ex.blocks[0].context_scale.item() == 0.0
    assert ex.blocks[0].feature_scale.requires_grad is False
    assert ex.blocks[0].graph_scale.requires_grad is True
    _unfreeze_to_full(ex.blocks[0])
    assert ex.blocks[0].feature_scale.requires_grad is True
    assert ex.blocks[0].feature_scale.item() > 0.0


def test_grad_group_mapping():
    assert grad_group_of("blocks.0.graph_message.0.weight") == "relational"
    assert grad_group_of("blocks.0.message.weight") == "relational"
    assert grad_group_of("blocks.0.feature_net.0.weight") == "feature"
    assert grad_group_of("blocks.0.context_value.weight") == "contextual"
    assert grad_group_of("blocks.0.fusion.weight") == "fusion"
    assert grad_group_of("encoder.weight") == "encoder"
    assert grad_group_of("head.1.weight") == "head"


def test_record_grad_norms_no_step():
    ex = build_expert_for_variant(None)
    ds = _tiny_ds()
    before = [p.detach().clone() for p in ex.parameters()]
    norms = record_grad_norms(ex, ds.features[:8], ds.targets[:8])
    for name, p in ex.named_parameters():
        idx = list(ex.named_parameters()).index((name, p))
        assert torch.equal(p, before[idx])
    assert set(norms) == {"relational", "feature", "contextual", "fusion", "encoder", "head"}
    assert all(v >= 0.0 for v in norms.values())
    # all grads must be discarded afterwards
    assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in ex.parameters())


def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "computation": {"tested": True, "graph_on_joint_minus_embedded_R": 0.0, "n_pos": 0,
                        "native_minus_graph_on_joint_R": 0.30, "shapes_equal": True},
        "graph_diagnostic": {"tested": True, "native_minus_diagnostic_R": 0.25},
        "coadaptation": {"tested": True, "relfocused_minus_joint_R": 0.0, "relfirst_minus_joint_R": 0.0, "n_pos": 0},
        "stages": {"tested": True, "pre_probe_R": 0.80, "post_probe_R": 0.82,
                   "post_fusion_probe_R": 0.81, "native_input_probe_R": 0.85, "final_R": 0.60},
        "compositional": {"tested": True, "R_gain": 0.0, "RC_gain": 0.0, "FRC_gain": 0.0},
        "sensitivity": {"tested": True, "candidate_drop_minus_baseline_drop_R": 0.0},
        "ceiling": {"tested": True, "new_minus_old_mixed": 0.0,
                    "old_ceiling_mixed_mean": 0.66, "new_ceiling_mixed_mean": 0.66},
        "gradients": {"status": "VERIFIED", "rel_share_mean": 0.30},
        "seed_stability": {"trained": {}},
        "latency": {"tested": True, "baseline_total_us": 60.0, "candidate_total_us": 90.0},
    }
    agg.update(over)
    return agg


def test_hypotheses_cover_H1_to_H8():
    hyps = build_phase16_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    for d in hyps.values():
        assert d["status"] in ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")


def test_H1_supported_when_input_suspect():
    hyps = build_phase16_hypotheses(_agg_fixture())
    assert hyps["H1"]["status"] == "SUPPORTED"
    assert hyps["H2"]["status"] == "NOT SUPPORTED"


def test_H2_supported_when_computation_differs():
    agg = _agg_fixture(computation={"tested": True, "graph_on_joint_minus_embedded_R": 0.05, "n_pos": 3,
                                    "native_minus_graph_on_joint_R": 0.02, "shapes_equal": True})
    hyps = build_phase16_hypotheses(agg)
    assert hyps["H2"]["status"] == "SUPPORTED"


def test_H3_supported_when_relfocused_wins():
    agg = _agg_fixture(coadaptation={"tested": True, "relfocused_minus_joint_R": 0.04,
                                     "relfirst_minus_joint_R": 0.01, "n_pos": 3})
    hyps = build_phase16_hypotheses(agg)
    assert hyps["H3"]["status"] == "SUPPORTED"


def test_case_selection_all_branches():
    base = _agg_fixture()
    assert select_phase16_case(build_phase16_hypotheses(_agg_fixture(
        coadaptation={"tested": True, "relfocused_minus_joint_R": 0.05, "relfirst_minus_joint_R": 0.0, "n_pos": 3})), base["stages"])[0] == "CASE B"
    assert select_phase16_case(build_phase16_hypotheses(_agg_fixture(
        computation={"tested": True, "graph_on_joint_minus_embedded_R": 0.05, "n_pos": 3,
                     "native_minus_graph_on_joint_R": 0.02, "shapes_equal": True})), base["stages"])[0] == "CASE C"
    assert select_phase16_case(build_phase16_hypotheses(base), base["stages"])[0] == "CASE A"
    agg_d = _agg_fixture(computation={"tested": True, "graph_on_joint_minus_embedded_R": 0.0, "n_pos": 0,
                                      "native_minus_graph_on_joint_R": 0.05, "shapes_equal": True},
                         graph_diagnostic={"tested": True, "native_minus_diagnostic_R": 0.02},
                         stages={"tested": True, "pre_probe_R": 0.50, "post_probe_R": 0.55,
                                 "post_fusion_probe_R": 0.54, "native_input_probe_R": 0.85, "final_R": 0.50})
    assert select_phase16_case(build_phase16_hypotheses(agg_d), agg_d["stages"])[0] == "CASE D"
    agg_e = _agg_fixture(computation={"tested": True, "graph_on_joint_minus_embedded_R": 0.0, "n_pos": 0,
                                      "native_minus_graph_on_joint_R": 0.05, "shapes_equal": True},
                         graph_diagnostic={"tested": True, "native_minus_diagnostic_R": 0.02})
    assert select_phase16_case(build_phase16_hypotheses(agg_e), agg_e["stages"])[0] == "CASE E"
    agg_f = _agg_fixture(computation={"tested": False}, graph_diagnostic={"tested": False},
                         coadaptation={"tested": False},
                         stages={"tested": True, "pre_probe_R": 0.50, "post_probe_R": 0.52,
                                 "post_fusion_probe_R": 0.52, "native_input_probe_R": 0.55, "final_R": 0.50},
                         compositional={"tested": False}, sensitivity={"tested": False}, ceiling={"tested": False})
    assert select_phase16_case(build_phase16_hypotheses(agg_f), agg_f["stages"])[0] == "CASE F"


def test_intervention_and_recommendation():
    hyps = build_phase16_hypotheses(_agg_fixture())
    sel = select_minimal_intervention("CASE A", hyps, _agg_fixture())
    assert sel["intervention"] == "document_adapter"
    sel = select_minimal_intervention("CASE E", hyps, _agg_fixture())
    assert sel["intervention"] == "none"
    for case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F"):
        assert isinstance(recommendation_for_case(case), str) and recommendation_for_case(case)


def test_failure_diagnosis_categories():
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    for expected in ("input_mismatch", "computation_gap", "coadaptation_suppression",
                     "pre_computation_loss", "post_computation_loss",
                     "compositional_transfer_failure", "seed_instability",
                     "gradient_starvation", "latency_regression"):
        assert expected in cats
