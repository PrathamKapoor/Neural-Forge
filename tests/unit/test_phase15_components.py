"""Unit tests for Phase 15 relational-substep diagnostic building blocks."""
from __future__ import annotations

import torch

from neuroforge.blocks.joint_co import JointCoBlock
from neuroforge.blocks.joint_co_relational import (
    JointCoRelationalBlock,
    ring_neighbourhood_max,
)
from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase15_metrics import (
    branch_7way_ablation,
    build_failure_diagnosis,
    build_phase15_hypotheses,
    latency_microseconds,
    marker_ablation_accuracy,
    per_family_accuracy,
    relational_probe_triplet,
    select_minimal_intervention,
    select_phase15_case,
    token_permutation_accuracy,
)
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant


# ---------------------------------------------------------------------------
# Relational block variants
# ---------------------------------------------------------------------------
def test_depth1_replica_matches_jointco_math():
    torch.manual_seed(0)
    base = JointCoBlock()
    var = JointCoRelationalBlock(rel_depth=1, aggregation="mean", rel_update="linear")
    var.graph_message[0].load_state_dict(base.graph_message.state_dict())
    var.graph_update[0].load_state_dict(base.graph_update.state_dict())
    s = torch.randn(2, 12, 24)
    g = base.graph_adapter(s)
    m = torch.bmm(g.adjacency, base.graph_message(g.node_features))
    d_base = torch.tanh(base.graph_update(torch.cat((s, m), dim=-1)))
    d_var, _ = var._relational_delta(s)
    assert torch.allclose(d_base, d_var, atol=1e-6)


def test_variant_shapes_and_determinism():
    for kw in (
        {"rel_depth": 1, "aggregation": "mean", "rel_update": "linear"},
        {"rel_depth": 2, "aggregation": "mean", "rel_update": "linear"},
        {"rel_depth": 3, "aggregation": "sum", "rel_update": "linear"},
        {"rel_depth": 1, "aggregation": "max", "rel_update": "linear"},
        {"rel_depth": 1, "aggregation": "mean", "rel_update": "mlp"},
    ):
        ex = build_expert_for_variant(kw)
        x = torch.randn(3, 12, 8)
        assert ex(x).shape == (3, 2)
        assert torch.equal(ex(x).argmax(-1), ex(x).argmax(-1))


def test_invalid_variant_args_raise():
    import pytest

    with pytest.raises(ValueError):
        JointCoRelationalBlock(rel_depth=0)
    with pytest.raises(ValueError):
        JointCoRelationalBlock(aggregation="attention")
    with pytest.raises(ValueError):
        JointCoRelationalBlock(rel_update="transformer")


def test_ring_max_aggregation_semantics():
    m = torch.zeros(1, 4, 2)
    m[0, 1, 0] = 5.0
    out = ring_neighbourhood_max(m)
    # node 1 (self), node 0 (right neighbour of 3? ring: neighbours of 0 are 3,0,1) get 5
    assert out[0, 1, 0].item() == 5.0
    assert out[0, 0, 0].item() == 5.0
    assert out[0, 2, 0].item() == 5.0
    assert out[0, 3, 0].item() == 0.0


def test_relational_param_accounting():
    assert JointCoRelationalBlock(rel_depth=1, rel_update="linear").relational_param_count() == 1776
    assert JointCoRelationalBlock(rel_depth=2, rel_update="linear").relational_param_count() == 2 * 1776
    mlp = JointCoRelationalBlock(rel_depth=1, rel_update="mlp").relational_param_count()
    assert mlp - 1776 == 600  # documented +600 capacity delta


def test_describe_relational_computation_audit():
    audit = JointCoRelationalBlock(rel_depth=2, aggregation="sum").describe_relational_computation()
    for key in ("input_shape", "topology", "message_passing_rounds", "neighbourhood_aggregation",
                "update_transformation", "output_shape", "relational_param_count", "when_applied"):
        assert key in audit
    assert audit["message_passing_rounds"] == 2
    assert audit["input_shape"] == "[B, S, H]"


def test_branch_scales_present_for_ablation():
    ex = build_expert_for_variant({"rel_depth": 2, "aggregation": "mean", "rel_update": "linear"})
    for name in ("feature_scale", "graph_scale", "context_scale"):
        assert hasattr(ex.blocks[0], name)


# ---------------------------------------------------------------------------
# Metric evaluators
# ---------------------------------------------------------------------------
def _tiny_ds():
    return Phase9MixedStructureDataset(samples_per_type=4, seed=11)


def test_per_family_accuracy_ranges():
    ex = StandaloneSpecialist("joint_co", depth=1)
    perf = per_family_accuracy(ex, _tiny_ds())
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC", "mixed_mean", "overall"):
        assert 0.0 <= perf[f] <= 1.0


def test_probe_triplet_structure():
    ex = StandaloneSpecialist("joint_co", depth=1)
    t = relational_probe_triplet(ex, _tiny_ds())
    assert set(t) == {"probe_R", "final_R", "gap"}
    assert abs(t["gap"] - (t["probe_R"] - t["final_R"])) < 1e-9


def test_marker_and_token_controls_structure():
    ex = StandaloneSpecialist("joint_co", depth=1)
    ds = _tiny_ds()
    m = marker_ablation_accuracy(ex, ds)
    assert set(m) == {"original", "marker_neutralised"}
    t = token_permutation_accuracy(ex, ds, seed=11)
    assert set(t) == {"original", "token_permuted"}


def test_branch_7way_labels():
    ex = build_expert_for_variant({"rel_depth": 1, "aggregation": "mean", "rel_update": "linear"})
    out = branch_7way_ablation(ex, _tiny_ds())
    assert set(out) == {"F", "R", "C", "F+R", "R+C", "F+C", "F+R+C"}


def test_latency_positive():
    ex = StandaloneSpecialist("joint_co", depth=1)
    ds = _tiny_ds()
    assert latency_microseconds(ex, ds.features[:8], n_runs=2) > 0.0


# ---------------------------------------------------------------------------
# Hypotheses / case / intervention / failure diagnosis
# ---------------------------------------------------------------------------
def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "depth": {"tested": True, "best_label": "depth2", "best_R_gain_mean": 0.04,
                  "best_R_gain_n_positive": 3,
                  "variants": {"depth2": {"R_gain_mean": 0.04, "R_gain_n_positive": 3, "status": "SUPPORTED"}}},
        "aggregation": {"tested": False, "gate_reason": "gated"},
        "capacity": {"tested": False, "gate_reason": "gated"},
        "topology": {"tested": True, "candidate_R_gain_mean": 0.04,
                     "candidate_drop_minus_baseline_drop_R": 0.01, "candidate_drop_R": 0.05},
        "compositional": {"tested": True, "R_gain_mean": 0.04, "RC_gain_mean": 0.03, "FRC_gain_mean": 0.03},
        "specificity": {"tested": True, "rel_gain_mean": 0.035, "nonrel_gain_mean": 0.0},
        "gap": {"tested": True, "gap_close_mean": 0.04, "baseline_gap_mean": 0.15, "candidate_gap_mean": 0.11},
        "ceiling": {"tested": True, "new_minus_old_mixed": 0.02,
                    "old_ceiling_mixed_mean": 0.66, "new_ceiling_mixed_mean": 0.68},
        "seed_stability": {"depth": {}, "aggregation": {}, "capacity": {}},
        "branch_interference": {"present": False, "observed": "not observed"},
        "compute": {"baseline_params": 6464, "candidate_params": 8240},
        "latency": {"tested": True, "baseline_total_us": 100.0, "candidate_total_us": 120.0},
    }
    agg.update(over)
    return agg


def test_hypotheses_cover_H1_to_H8():
    hyps = build_phase15_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    for h, d in hyps.items():
        assert d["status"] in ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
        assert isinstance(d["evidence"], str) and d["evidence"]


def test_gated_off_sections_report_not_tested():
    hyps = build_phase15_hypotheses(_agg_fixture())
    assert hyps["H2"]["status"] == "NOT TESTED"
    assert hyps["H3"]["status"] == "NOT TESTED"


def test_select_case_valid():
    hyps = build_phase15_hypotheses(_agg_fixture())
    case, label = select_phase15_case(hyps, baseline_gap=0.15, candidate_r_gain=0.04)
    assert case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")
    assert isinstance(label, str) and label


def test_select_case_A_when_no_gap():
    hyps = build_phase15_hypotheses(_agg_fixture())
    case, _ = select_phase15_case(hyps, baseline_gap=0.02, candidate_r_gain=0.0)
    assert case == "CASE A"


def test_select_minimal_intervention_prefers_smallest_depth():
    agg = _agg_fixture()
    agg["depth"]["variants"]["depth3"] = {"R_gain_mean": 0.06, "R_gain_n_positive": 3, "status": "SUPPORTED"}
    sel = select_minimal_intervention(agg)
    assert sel["intervention"] == "depth2"


def test_select_minimal_intervention_none_when_nothing_helps():
    agg = _agg_fixture(depth={"tested": True, "best_label": "depth2", "best_R_gain_mean": -0.01,
                              "best_R_gain_n_positive": 0,
                              "variants": {"depth2": {"R_gain_mean": -0.01, "R_gain_n_positive": 0, "status": "NOT SUPPORTED"}}},
                       topology={"tested": True, "candidate_R_gain_mean": -0.01,
                                 "candidate_drop_minus_baseline_drop_R": 0.0, "candidate_drop_R": 0.0},
                       compositional={"tested": True, "R_gain_mean": -0.01, "RC_gain_mean": 0.0, "FRC_gain_mean": 0.0},
                       specificity={"tested": True, "rel_gain_mean": 0.0, "nonrel_gain_mean": 0.0},
                       gap={"tested": True, "gap_close_mean": 0.0, "baseline_gap_mean": 0.15, "candidate_gap_mean": 0.15},
                       ceiling={"tested": True, "new_minus_old_mixed": 0.0,
                                "old_ceiling_mixed_mean": 0.66, "new_ceiling_mixed_mean": 0.66})
    sel = select_minimal_intervention(agg)
    assert sel["intervention"] == "none"


def test_failure_diagnosis_explicit_criteria():
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    for expected in ("depth_failure", "aggregation_failure", "capacity_failure", "topology_failure",
                     "representation_prediction_gap", "compositional_transfer_failure", "seed_instability",
                     "compute_regression", "latency_regression", "branch_interference"):
        assert expected in cats
    for r in rows:
        assert isinstance(r["failed"], bool)
        assert r["criterion"] and r["observed"] is not None
