"""Unit tests for Phase 20 repaired-portfolio diagnostic blocks."""
from __future__ import annotations

import torch

from neuroforge.datasets.phase9_mixed_repaired import (
    CONSTRUCTION_VERSION,
    Phase9MixedRepairedDataset,
)
from neuroforge.evaluation.phase20_repaired_portfolio import (
    analytical_flops,
    assert_repaired,
    build_failure_diagnosis,
    build_phase20_hypotheses,
    check_zero_overlap,
    dataset_fingerprint,
    per_family_accuracy,
    recommendation_for_case,
    select_phase20_case,
)
from neuroforge.models.specialists import StandaloneSpecialist


def _rep(n=12, seed=11):
    return Phase9MixedRepairedDataset(samples_per_type=n, seed=seed)


def test_assert_repaired_rejects_original():
    import pytest

    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = _rep()
    assert_repaired(ds)  # must not raise
    with pytest.raises(ValueError):
        assert_repaired(Phase9MixedStructureDataset(samples_per_type=4, seed=11))


def test_fingerprints_distinguish_versions_and_splits():
    a = Phase9MixedRepairedDataset(samples_per_type=12, seed=11)
    b = Phase9MixedRepairedDataset(samples_per_type=12, seed=11)
    c = Phase9MixedRepairedDataset(samples_per_type=12, seed=23)
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(c)
    assert check_zero_overlap(a, c) == 0


def test_per_family_accuracy_structure():
    ds = _rep()
    ex = StandaloneSpecialist("joint_co", depth=1)
    out = per_family_accuracy(ex, ds.features, ds.targets, ds.families_list)
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC", "mixed_mean", "overall"):
        assert 0.0 <= out[f] <= 1.0


def test_analytical_flops_scales_with_depth():
    e1 = StandaloneSpecialist("joint_co", depth=1)
    e2 = StandaloneSpecialist("graph", depth=2)
    f1 = analytical_flops(e1)
    f2 = analytical_flops(e2)
    assert f1 > 0 and f2 > 0 and f1 != f2


def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "pure": {"tested": True, "best_F": 0.95, "best_R": 0.9, "best_C": 0.95},
        "mixed": {"tested": True, "best_mixed_mean": 0.75},
        "requires_multi": {"tested": True, "rc_margin": 0.3, "frc_margin": 0.2},
        "composition": {"tested": True, "RC_gain": 0.05, "FRC_gain": 0.05, "n_pos": 3},
        "routing": {"tested": True, "learned_minus_best_single_mixed": 0.04, "n_pos": 3},
        "behavior": {"tested": True, "agree_acc": 1.0, "disagree_acc": 0.9},
        "ceiling": {"tested": True, "portfolio_minus_single_mixed": 0.05,
                    "portfolio_RC": 0.8, "portfolio_FRC": 0.85,
                    "single_RC": 0.7, "single_FRC": 0.75,
                    "single_mixed": 0.7, "portfolio_mixed": 0.75},
        "smoke": {"tested": True, "passed": True},
        "seed_stability": {"RC": {}},
        "latency": {"tested": True, "baseline_total_us": 60.0, "candidate_total_us": 60.0},
    }
    agg.update(over)
    return agg


def test_hypotheses_all_supported_on_strong():
    hyps = build_phase20_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    for i in (1, 2, 3, 4, 5, 6, 8):
        assert hyps[f"H{i}"]["status"] == "SUPPORTED", (i, hyps[f"H{i}"])
    assert hyps["H7"]["status"] == "NOT SUPPORTED"  # collapse gone


def test_case_E_on_genuine_gain():
    assert select_phase20_case(build_phase20_hypotheses(_agg_fixture()), _agg_fixture())[0] == "CASE E"


def test_case_A_on_resolved_failure():
    agg = _agg_fixture(composition={"tested": True, "RC_gain": 0.0, "FRC_gain": 0.0, "n_pos": 0},
                       ceiling={"tested": True, "portfolio_minus_single_mixed": 0.0,
                                "portfolio_RC": 0.7, "portfolio_FRC": 0.75,
                                "single_RC": 0.7, "single_FRC": 0.75,
                                "single_mixed": 0.7, "portfolio_mixed": 0.7})
    hyps = build_phase20_hypotheses(agg)
    assert hyps["H5"]["status"] == "NOT SUPPORTED"
    assert hyps["H8"]["status"] == "NOT SUPPORTED"
    assert select_phase20_case(hyps, agg)[0] == "CASE A"


def test_case_C_routing_fails():
    agg = _agg_fixture(routing={"tested": True, "learned_minus_best_single_mixed": -0.01, "n_pos": 0},
                       ceiling={"tested": True, "portfolio_minus_single_mixed": 0.0,
                                "portfolio_RC": 0.7, "portfolio_FRC": 0.75,
                                "single_RC": 0.7, "single_FRC": 0.75,
                                "single_mixed": 0.7, "portfolio_mixed": 0.7})
    hyps = build_phase20_hypotheses(agg)
    assert hyps["H6"]["status"] == "NOT SUPPORTED"
    assert select_phase20_case(hyps, agg)[0] == "CASE C"


def test_case_D_composition_without_ceiling():
    agg = _agg_fixture(routing={"tested": True, "learned_minus_best_single_mixed": 0.04, "n_pos": 3},
                       ceiling={"tested": True, "portfolio_minus_single_mixed": 0.0,
                                "portfolio_RC": 0.7, "portfolio_FRC": 0.75,
                                "single_RC": 0.7, "single_FRC": 0.75,
                                "single_mixed": 0.7, "portfolio_mixed": 0.7})
    hyps = build_phase20_hypotheses(agg)
    # H6 SUPPORTED blocks C; H8 NOT SUPPORTED blocks E; H5 SUPPORTED -> D
    assert select_phase20_case(hyps, agg)[0] == "CASE D"


def test_case_B_failure_persists():
    agg = _agg_fixture(mixed={"tested": True, "best_mixed_mean": 0.5},
                       composition={"tested": True, "RC_gain": 0.0, "FRC_gain": 0.0, "n_pos": 0},
                       routing={"tested": True, "learned_minus_best_single_mixed": 0.0, "n_pos": 0},
                       ceiling={"tested": True, "portfolio_minus_single_mixed": 0.0,
                                "portfolio_RC": 0.5, "portfolio_FRC": 0.5,
                                "single_RC": 0.5, "single_FRC": 0.5,
                                "single_mixed": 0.5, "portfolio_mixed": 0.5})
    hyps = build_phase20_hypotheses(agg)
    assert hyps["H2"]["status"] == "NOT SUPPORTED"
    assert select_phase20_case(hyps, agg)[0] == "CASE B"


def test_case_B_rc_specific_failure():
    # Mixed capable overall, but RC collapse persists with weak RC -> CASE B.
    agg = _agg_fixture(mixed={"tested": True, "best_mixed_mean": 0.74, "best_RC": 0.55,
                              "best_FRC": 0.76},
                       behavior={"tested": True, "agree_acc": 0.98, "disagree_acc": 0.01},
                       composition={"tested": True, "RC_gain": 0.0, "FRC_gain": 0.0, "n_pos": 0},
                       routing={"tested": True, "learned_minus_best_single_mixed": 0.025, "n_pos": 3},
                       ceiling={"tested": True, "portfolio_minus_single_mixed": -0.013,
                                "portfolio_RC": 0.558, "portfolio_FRC": 0.761,
                                "single_RC": 0.572, "single_FRC": 0.811,
                                "single_mixed": 0.759, "portfolio_mixed": 0.746})
    hyps = build_phase20_hypotheses(agg)
    assert hyps["H2"]["status"] == "SUPPORTED"
    assert hyps["H7"]["status"] == "SUPPORTED"
    assert hyps["H8"]["status"] == "NOT SUPPORTED"
    assert select_phase20_case(hyps, agg)[0] == "CASE B"


def test_case_F_empty():
    agg = _agg_fixture(pure={"tested": False}, mixed={"tested": False},
                       requires_multi={"tested": False}, composition={"tested": False},
                       routing={"tested": False}, behavior={"tested": False},
                       ceiling={"tested": False})
    assert select_phase20_case(build_phase20_hypotheses(agg), agg)[0] == "CASE F"


def test_recommendations_and_failures():
    for case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F"):
        assert isinstance(recommendation_for_case(case), str) and recommendation_for_case(case)
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    for expected in ("pure_capability_failure", "mixed_capability_failure", "composition_no_gain",
                     "routing_no_gain", "single_component_collapse", "ceiling_unmoved",
                     "seed_instability", "smoke_failed"):
        assert expected in cats
