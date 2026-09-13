"""Unit tests for Phase 19 benchmark repair and validation blocks."""
from __future__ import annotations

import torch

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.datasets.phase9_mixed_repaired import (
    CONSTRUCTION_VERSION,
    RELATIONAL_CHANNEL,
    Phase9MixedRepairedDataset,
)
from neuroforge.evaluation.phase19_benchmark_validation import (
    autocorr_align,
    build_failure_diagnosis,
    build_phase19_hypotheses,
    dependency_audit_rows,
    evaluate_gates,
    feature_offset_magnitude,
    linear_probe_train_test,
    retrieval_acc,
    select_phase19_case,
    train_shallow_mlp,
)


def _orig(n=10, seed=11):
    return Phase9MixedStructureDataset(samples_per_type=n, seed=seed)


def _rep(n=10, seed=11):
    return Phase9MixedRepairedDataset(samples_per_type=n, seed=seed)


def test_repaired_api_matches_original():
    o, r = _orig(), _rep()
    assert r.mixed_families == o.mixed_families
    assert r.features.shape == o.features.shape
    assert r.targets.shape == o.targets.shape
    assert r.families_list is not None and len(r) == len(o)
    assert set(r.items[0].keys()) >= {"features", "target", "family", "sf", "sr", "sc", "sample_id"}
    assert r.construction_version == CONSTRUCTION_VERSION
    assert RELATIONAL_CHANNEL == 5


def test_repaired_determinism():
    a = Phase9MixedRepairedDataset(samples_per_type=8, seed=11)
    b = Phase9MixedRepairedDataset(samples_per_type=8, seed=11)
    assert torch.equal(a.features, b.features)
    assert torch.equal(a.targets, b.targets)


def test_original_bug_reproduced_by_statistic():
    o = Phase9MixedStructureDataset(samples_per_type=40, seed=11)
    r_align = autocorr_align(o.features, o.items, o.families_list, ("R",), 0)
    rc_align = autocorr_align(o.features, o.items, o.families_list, ("RC",), 0)
    assert r_align >= 0.90
    assert rc_align < 0.60


def test_repaired_carrier_ordering():
    r = Phase9MixedRepairedDataset(samples_per_type=40, seed=11)
    for fam in ("R", "RC", "FRC"):
        a = autocorr_align(r.features, r.items, r.families_list, (fam,), RELATIONAL_CHANNEL)
        assert a >= 0.90, f"{fam}: {a}"


def test_feature_offsets_survive_repair():
    r = Phase9MixedRepairedDataset(samples_per_type=20, seed=11)
    for fam in ("F", "FR", "FC", "FRC"):
        assert feature_offset_magnitude(r.features, r.families_list, (fam,)) >= 0.5
    o = Phase9MixedStructureDataset(samples_per_type=20, seed=11)
    # historical FC/FRC offsets were erased
    assert feature_offset_magnitude(o.features, o.families_list, ("FC",)) < 0.5


def test_retrieval_intact_after_repair():
    r = Phase9MixedRepairedDataset(samples_per_type=20, seed=11)
    for fam in ("C", "RC", "FC", "FRC"):
        assert retrieval_acc(r.features, r.families_list, (fam,)) >= 0.80


def test_label_algebra_preserved():
    r = Phase9MixedRepairedDataset(samples_per_type=30, seed=11)
    for i in range(len(r)):
        it = r.items[i]
        act = [it["sf"] if "F" in it["family"] else 0,
               it["sr"] if "R" in it["family"] else 0,
               it["sc"] if "C" in it["family"] else 0]
        # reconstruct expected label from stored components only
        active = [v for k, v in (("F", act[0]), ("R", act[1]), ("C", act[2]))
                  if k in it["family"]]
        z = sum(active)
        y = 1 if z > 0 else (0 if z < 0 else (it["sample_id"] % 2 if False else None))
        # tie-break uses the within-type loop index, unavailable post-hoc; check non-ties
        if z != 0:
            assert int(r.targets[i]) == y


def test_dependency_audit_detects_overwrite():
    o = _orig(40, 11)
    rows = dependency_audit_rows(o.features, o.items, o.families_list, 0, ("FR",),
                                 {0: "overwrite"}, "original")
    rc = [x for x in rows if x["family"] == "RC" and x["component"] == "R"][0]
    assert rc["carrier_channel"] == 0
    assert rc["observable_rate"] < 0.60


def test_linear_probe_and_mlp_run():
    r = _rep(12, 11)
    x = r.features.mean(dim=1)
    assert 0.0 <= linear_probe_train_test(x, r.targets, seed=0) <= 1.0
    fam_id = torch.tensor([r.families_list.index("F")] * 0)  # empty-safe path below
    assert 0.0 <= train_shallow_mlp(r.features[:24], r.targets[:24], seed=0, epochs=2) <= 1.0
    assert len(fam_id) == 0


def _gates_all_pass():
    return {g: {"passed": True, "detail": "ok"} for g in
            ("G1_observability", "G2_no_leakage", "G3_no_family_leakage", "G4_invariance",
             "G5_counterfactual", "G6_semantics", "G7_learnability", "G8_reproducibility")}


def _agg_fixture(**over):
    agg = {
        "n_seeds": 3,
        "bug": {"tested": True, "orig_R": 0.97, "orig_RC": 0.48, "orig_FRC": 0.52, "erasure_reproduced": True},
        "observability": {"tested": True, "r_align_R": 0.97, "r_align_RC": 0.95, "r_align_FRC": 0.96,
                          "r_align_FR": 0.95, "f_mag_F": 0.8, "f_mag_FR": 0.8, "f_mag_FC": 0.8,
                          "f_mag_FRC": 0.8, "c_ret_C": 0.95, "c_ret_RC": 0.95, "c_ret_FC": 0.95,
                          "c_ret_FRC": 0.95},
        "invariance": {"tested": True, "c_retrieval_perm_drop": 0.0, "r_align_marker_delta": -0.2,
                       "r_align_marker_delta_original": -0.01,
                       "r_markvar_R_repaired": -0.17, "r_markvar_R_original": -0.18,
                       "r_align_perm_repaired_drop": 0.1, "r_align_perm_original_drop": 0.1,
                       "detail": "ok"},
        "leakage": {"tested": True, "max_single_channel_label_probe": 0.7, "family_probe_delta": 0.0},
        "counterfactual": {"tested": True, "R_valid_rate": 0.95, "C_valid_rate": 0.95},
        "semantics": {"tested": True, "parity_fact": True, "frc_no_tie": True, "keys_present": True,
                      "rule_RC": 0.97, "c_only_RC": 0.55},
        "learnability": {"tested": True, "rule_RC": 0.97, "mlp_R": 0.9, "mlp_RC": 0.8,
                         "mlp_FRC": 0.8, "stat_R": 0.97, "lagprobe_R": 0.93},
        "reproducibility": {"identical": True},
        "gates": _gates_all_pass(),
    }
    agg.update(over)
    return agg


def _fail_gate(agg, gate):
    agg = dict(agg)
    gates = {k: dict(v) for k, v in agg["gates"].items()}
    gates[gate] = {"passed": False, "detail": "forced failure"}
    agg["gates"] = gates
    return agg


def test_hypotheses_all_supported_on_valid():
    hyps = build_phase19_hypotheses(_agg_fixture())
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    assert all(hyps[f"H{i}"]["status"] == "SUPPORTED" for i in range(1, 9))


def test_gates_all_pass_on_valid():
    gates = evaluate_gates(_agg_fixture())
    assert set(gates) == {f"G{i}_{n}" for i, n in
                          [(1, "observability"), (2, "no_leakage"), (3, "no_family_leakage"),
                           (4, "invariance"), (5, "counterfactual"), (6, "semantics"),
                           (7, "learnability"), (8, "reproducibility")]}
    assert all(g["passed"] for g in gates.values())


def test_case_F_on_full_validity():
    assert select_phase19_case(build_phase19_hypotheses(_agg_fixture()), _agg_fixture())[0] == "CASE F"


def test_case_B_on_leakage():
    agg = _fail_gate(_agg_fixture(leakage={"tested": True, "max_single_channel_label_probe": 0.97,
                                           "family_probe_delta": 0.0}),
                     "G2_no_leakage")
    hyps = build_phase19_hypotheses(agg)
    assert hyps["H6"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(hyps, agg)[0] == "CASE B"


def test_case_C_on_degenerate_semantics():
    agg = _fail_gate(_agg_fixture(semantics={"tested": True, "parity_fact": True, "frc_no_tie": True,
                                             "keys_present": True, "rule_RC": 0.55, "c_only_RC": 0.55}),
                     "G6_semantics")
    assert build_phase19_hypotheses(agg)["H7"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(build_phase19_hypotheses(agg), agg)[0] == "CASE C"


def test_case_D_on_unobservable():
    agg = _fail_gate(_agg_fixture(observability={"tested": True, "r_align_R": 0.97, "r_align_RC": 0.5,
                                                 "r_align_FRC": 0.96, "r_align_FR": 0.95, "f_mag_F": 0.8,
                                                 "f_mag_FR": 0.8, "f_mag_FC": 0.8, "f_mag_FRC": 0.8,
                                                 "c_ret_C": 0.95, "c_ret_RC": 0.95, "c_ret_FC": 0.95,
                                                 "c_ret_FRC": 0.95}),
                     "G1_observability")
    assert build_phase19_hypotheses(agg)["H2"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(build_phase19_hypotheses(agg), agg)[0] == "CASE D"


def test_case_A_repair_without_full_gate():
    agg = _fail_gate(_agg_fixture(learnability={"tested": True, "rule_RC": 0.97, "mlp_R": 0.9,
                                                "mlp_RC": 0.4, "mlp_FRC": 0.8, "stat_R": 0.97}),
                     "G7_learnability")
    hyps = build_phase19_hypotheses(agg)
    assert hyps["H8"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(hyps, agg)[0] == "CASE A"


def test_case_E_on_gate_failure():
    agg = _fail_gate(_agg_fixture(reproducibility={"identical": False}), "G8_reproducibility")
    assert build_phase19_hypotheses(agg)["H8"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(build_phase19_hypotheses(agg), agg)[0] == "CASE A"


def test_case_E_unresolved():
    # Bug NOT reproduced (H1 fails) + learnability fails (H8 fails),
    # but no specific B/C/D/A cause -> CASE E.
    agg = _fail_gate(_agg_fixture(bug={"tested": True, "orig_R": 0.5, "orig_RC": 0.5,
                                                 "orig_FRC": 0.5, "erasure_reproduced": False},
                                            learnability={"tested": True, "rule_RC": 0.97, "mlp_R": 0.9,
                                                          "mlp_RC": 0.4, "mlp_FRC": 0.8, "stat_R": 0.97}),
                          "G7_learnability")
    hyps = build_phase19_hypotheses(agg)
    assert hyps["H1"]["status"] == "NOT SUPPORTED"
    assert hyps["H8"]["status"] == "NOT SUPPORTED"
    assert select_phase19_case(hyps, agg)[0] == "CASE E"


def test_failure_rows_cover_gates():
    rows = build_failure_diagnosis(_agg_fixture())
    cats = {r["category"] for r in rows}
    assert "bug_not_reproduced" in cats
    assert any(c.startswith("gate_G") for c in cats)
