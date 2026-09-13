"""Unit tests for Phase 17 heterogeneous composition diagnostic blocks."""
from __future__ import annotations

import torch

from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock
from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase17_composition_diagnostics import (
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
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant


def _tiny_ds():
    return Phase9MixedStructureDataset(samples_per_type=4, seed=11)


def test_intermediates_available_on_relational_block():
    ex = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
    block = ex.blocks[0]
    assert isinstance(block, JointCoRelationalBlock)
    with torch.no_grad():
        inter = block.forward_with_intermediates(ex.encoder(torch.randn(2, 12, 8)), torch.randn(2, 12, 8))
    for key in ("input", "feat_delta", "rel_delta", "rel_rounds", "ctx_delta",
                "scaled_sum", "fused_delta", "block_output"):
        assert key in inter
    assert len(inter["rel_rounds"]) == 3


def test_component_targets_binary():
    ds = _tiny_ds()
    tgts = component_targets(ds)
    assert set(tgts) == {"F_signal", "R_signal", "C_signal", "final_target"}
    for v in tgts.values():
        assert set(v.tolist()) <= {0, 1}


def test_probe_state_components_ranges():
    ds = _tiny_ds()
    out = probe_state_components(torch.randn(len(ds), 12, 24), ds)
    assert set(out) == {"F_signal", "R_signal", "C_signal", "final_target"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_diag_head_deterministic():
    torch.manual_seed(0)
    x = torch.randn(24, 48)
    y = torch.tensor([0, 1] * 12, dtype=torch.long)
    h1 = train_diag_head(x, y, 48, epochs=2, seed=5)
    h2 = train_diag_head(x, y, 48, epochs=2, seed=5)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)


def test_eval_diag_head_ranges():
    ds = _tiny_ds()
    head = train_diag_head(torch.randn(len(ds), 24), ds.targets, 24, epochs=1)
    out = eval_diag_head(head, torch.randn(len(ds), 24), ds.targets, ds.families_list)
    assert 0.0 <= out["RC"] <= 1.0
    assert 0.0 <= out["mixed_mean"] <= 1.0


def test_scale_stats_and_domination():
    states = {"feat_delta": torch.randn(4, 12, 24), "rel_delta": torch.randn(4, 12, 24),
              "ctx_delta": torch.randn(4, 12, 24), "block_output": torch.randn(4, 12, 24)}
    stats = branch_scale_stats(states)
    assert set(stats) == set(states)
    assert all(v["rms"] > 0 for v in stats.values())
    ratios = domination_ratios(stats)
    assert ratios["max_min_branch"] >= 1.0
    assert ratios["rel_over_feat"] > 0


def test_interaction_gain_sign():
    import pytest

    assert interaction_gain(0.5, 0.6, 0.4) == pytest.approx(-0.1)
    assert interaction_gain(0.7, 0.6, 0.4) == pytest.approx(0.1)


def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "preservation": {"tested": True, "branch_F_signal": 0.9, "branch_R_signal": 0.85,
                         "branch_C_signal": 0.9, "F_signal_drop_pre_to_fused": 0.01,
                         "R_signal_drop_pre_to_fused": 0.01, "C_signal_drop_pre_to_fused": 0.01},
        "domination": {"tested": True, "max_min_branch_mean": 1.5},
        "interaction": {"tested": True, "worst_pair_gain_mean": 0.02, "n_neg_consistent": 0},
        "order": {"tested": True, "range_R_mean": 0.01, "best_order": "F→R", "best_consistent": False},
        "sufficiency": {"tested": True, "oracle_RC": 0.55, "oracle_FRC": 0.70,
                        "oracle_RC_gain": 0.01, "oracle_FRC_gain": 0.01},
        "compositional": {"tested": True, "R_gain": 0.03, "RC_gain": 0.0, "FRC_gain": 0.0},
        "intervention": {"tested": False, "reason": "none"},
        "ceiling": {"tested": False},
        "seed_stability": {"conditions": {}},
        "latency": {"tested": True, "baseline_total_us": 60.0, "candidate_total_us": 60.0},
    }
    agg.update(over)
    return agg


def test_hypotheses_cover_H1_to_H8():
    hyps = build_phase17_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    for d in hyps.values():
        assert d["status"] in ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")


def test_H1_supported_on_probe_loss():
    agg = _agg_fixture(preservation={"tested": True, "branch_F_signal": 0.9, "branch_R_signal": 0.85,
                                     "branch_C_signal": 0.9, "F_signal_drop_pre_to_fused": 0.08,
                                     "R_signal_drop_pre_to_fused": 0.07, "C_signal_drop_pre_to_fused": 0.0})
    assert build_phase17_hypotheses(agg)["H1"]["status"] == "SUPPORTED"


def test_H3_supported_on_negative_interaction():
    agg = _agg_fixture(interaction={"tested": True, "worst_pair_gain_mean": -0.05, "n_neg_consistent": 3})
    assert build_phase17_hypotheses(agg)["H3"]["status"] == "SUPPORTED"


def test_H5_H6_supported_on_oracle_gains():
    agg = _agg_fixture(sufficiency={"tested": True, "oracle_RC": 0.65, "oracle_FRC": 0.78,
                                    "oracle_RC_gain": 0.08, "oracle_FRC_gain": 0.06})
    hyps = build_phase17_hypotheses(agg)
    assert hyps["H5"]["status"] == "SUPPORTED"
    assert hyps["H6"]["status"] == "SUPPORTED"


def test_case_branches():
    base = _agg_fixture()
    assert select_phase17_case(build_phase17_hypotheses(_agg_fixture(
        preservation={**base["preservation"], "F_signal_drop_pre_to_fused": 0.08,
                      "R_signal_drop_pre_to_fused": 0.07})), base)[0] == "CASE A"
    assert select_phase17_case(build_phase17_hypotheses(_agg_fixture(
        domination={"tested": True, "max_min_branch_mean": 3.5})), base)[0] == "CASE B"
    assert select_phase17_case(build_phase17_hypotheses(_agg_fixture(
        interaction={"tested": True, "worst_pair_gain_mean": -0.05, "n_neg_consistent": 3})), base)[0] == "CASE C"
    assert select_phase17_case(build_phase17_hypotheses(_agg_fixture(
        order={"tested": True, "range_R_mean": 0.08, "best_order": "R→F", "best_consistent": True,
               "orientations": {"C-first_graph-final": {"diff_mean": 0.08, "n_pos": 3, "n_seeds": 3}}})),
        base)[0] == "CASE D"
    assert select_phase17_case(build_phase17_hypotheses(base), base)[0] == "CASE E"
    agg_f = _agg_fixture(preservation={"tested": True, "branch_F_signal": 0.5, "branch_R_signal": 0.5,
                                       "branch_C_signal": 0.5, "F_signal_drop_pre_to_fused": 0.0,
                                       "R_signal_drop_pre_to_fused": 0.0, "C_signal_drop_pre_to_fused": 0.0},
                         sufficiency={"tested": True, "oracle_RC": 0.5, "oracle_FRC": 0.5,
                                      "oracle_RC_gain": 0.0, "oracle_FRC_gain": 0.0})
    assert select_phase17_case(build_phase17_hypotheses(agg_f), agg_f)[0] == "CASE F"


def test_intervention_none_without_gate():
    sel = select_minimal_intervention("CASE E", build_phase17_hypotheses(_agg_fixture()), _agg_fixture())
    assert sel["intervention"] == "none"
    for case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F"):
        assert isinstance(recommendation_for_case(case), str) and recommendation_for_case(case)


def test_failure_categories():
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    for expected in ("fusion_information_loss", "scale_domination", "destructive_interaction",
                     "insufficient_joint_information", "compositional_transfer_failure",
                     "seed_instability", "order_sensitivity"):
        assert expected in cats
