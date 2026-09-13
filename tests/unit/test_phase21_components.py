"""Unit tests for Phase 21 RC diagnosis components (no training)."""
from __future__ import annotations

import torch

from neuroforge.datasets.phase9_mixed_repaired import Phase9MixedRepairedDataset
from neuroforge.evaluation.phase21_rc_diagnosis import (
    analytical_rc_baselines,
    branch_rms,
    build_failure_diagnosis,
    build_phase21_hypotheses,
    counterfactual_swap_rates,
    grad_norms_unfrozen_clone,
    h8_gate,
    implied_label,
    mean,
    recommendation_for_case,
    select_phase21_case,
    split_metrics,
    stat_sr_repaired,
)
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase21_rc_diagnosis import (
    build_joint_variant,
    final_rep_query,
    probe_joint_on_state,
    stage_states,
)


def _tiny_ds(n: int = 16, seed: int = 11) -> Phase9MixedRepairedDataset:
    return Phase9MixedRepairedDataset(samples_per_type=n, seed=seed)


def test_implied_label_algebra():
    assert implied_label(1, 1) == 1
    assert implied_label(-1, -1) == 0
    assert implied_label(1, -1) == 1  # disagree -> R side
    assert implied_label(-1, 1) == 0


def test_stage_states_shapes_and_determinism():
    ds = _tiny_ds()
    expert = build_joint_variant(2)
    expert.eval()
    s1 = stage_states(expert, ds.features)
    s2 = stage_states(expert, ds.features)
    assert set(s1) == {"input", "relational_output", "fusion"}
    for k in s1:
        assert s1[k].shape == (len(ds), 12, 24)
        assert torch.equal(s1[k], s2[k])


def test_final_rep_query_shape():
    ds = _tiny_ds()
    expert = build_joint_variant(3)
    expert.eval()
    q = final_rep_query(expert, ds.features)
    assert q.shape == (len(ds), 1, 24)


def test_split_metrics_agree_disagree():
    ds = _tiny_ds()
    expert = StandaloneSpecialist("graph", input_dim=8, hidden_dim=24, depth=1)
    expert.eval()
    with torch.no_grad():
        preds = expert(ds.features).argmax(-1)
    m = split_metrics(preds, ds)
    assert m["n_RC_agree"] + m["n_RC_disagree"] == sum(1 for f in ds.families_list if f == "RC")
    assert 0.0 <= m["RC_disagree"] <= 1.0


def test_stat_sr_repaired_binary():
    ds = _tiny_ds()
    s = stat_sr_repaired(ds.features)
    assert set(s.tolist()) <= {0, 1}
    assert len(s) == len(ds)


def test_analytical_baselines_ranges():
    ds = _tiny_ds()
    out = analytical_rc_baselines(ds)
    assert set(out) == {"R_only", "C_only", "oracle_combo"}
    for v in out.values():
        assert 0.0 <= v["RC"] <= 1.0


def test_branch_rms_keys():
    ds = _tiny_ds()
    expert = build_joint_variant(3)
    expert.eval()
    rms = branch_rms(expert, ds.features)
    assert "rel_delta" in rms and "ctx_delta" in rms
    assert all(v >= 0.0 for v in rms.values())


def test_grad_norms_leave_original_frozen():
    ds = _tiny_ds()
    expert = build_joint_variant(2)
    for p in expert.parameters():
        p.requires_grad_(False)
    before = [p.detach().clone() for p in expert.parameters()]
    norms = grad_norms_unfrozen_clone(expert, ds.features[:8], ds.targets[:8])
    assert set(norms) >= {"relational", "contextual", "feature", "fusion", "encoder", "head"}
    for p, b in zip(expert.parameters(), before):
        assert not p.requires_grad
        assert torch.equal(p.detach(), b)


def test_counterfactual_structure():
    ds = _tiny_ds(8)
    expert = StandaloneSpecialist("graph", input_dim=8, hidden_dim=24, depth=1)
    expert.eval()
    out = counterfactual_swap_rates(expert, ds, seed=0)
    for k in ("R_change", "R_correct_change", "R_incorrect_change", "C_change",
              "C_correct_change", "C_incorrect_change", "control_change",
              "control_correct_change", "control_incorrect_change",
              "R_n", "C_n", "control_n"):
        assert k in out
    assert 0.0 <= out["R_change"] <= 1.0
    assert out["R_correct_change"] + out["R_incorrect_change"] <= out["R_change"] + 1e-9


def _hyps_all(status: str) -> dict[str, dict[str, str]]:
    return {f"H{i}": {"status": status, "evidence": "t"} for i in range(1, 9)}


def test_h8_gate_logic():
    agg = {"hypotheses": _hyps_all("INCONCLUSIVE"),
           "dominance": {"R_share": 0.06}}
    agg["hypotheses"]["H1"] = {"status": "SUPPORTED", "evidence": "t"}
    assert h8_gate(agg)["passed"] is True
    agg["dominance"] = {"R_share": 0.01}
    assert h8_gate(agg)["passed"] is False
    agg["hypotheses"]["H1"] = {"status": "NOT SUPPORTED", "evidence": "t"}
    agg["dominance"] = {"R_share": 0.50}
    assert h8_gate(agg)["passed"] is False


def test_case_selection_programmatic():
    base = {"agreement": {"agree_acc": 0.5, "disagree_acc": 0.5},
            "locations": {"rows": {"relational_output": {"RC_dec": 0.9}}},
            "production_RC": 0.5,
            "heads": {"component_RC": 0.9}}
    hy = _hyps_all("NOT SUPPORTED")
    hy["H8"] = {"status": "SUPPORTED", "evidence": "t"}
    assert select_phase21_case(hy, base)[0] == "CASE D"
    hy = _hyps_all("NOT SUPPORTED")
    hy["H5"] = {"status": "SUPPORTED", "evidence": "t"}
    agg = dict(base, heads={"component_RC": 0.85}, production_RC=0.5)
    assert select_phase21_case(hy, agg)[0] == "CASE C"
    hy = _hyps_all("NOT SUPPORTED")
    hy["H2"] = {"status": "SUPPORTED", "evidence": "t"}
    hy["H5"] = {"status": "NOT SUPPORTED", "evidence": "t"}
    assert select_phase21_case(hy, base)[0] == "CASE B"
    agg2 = dict(base, agreement={"agree_acc": 0.95, "disagree_acc": 0.05})
    hy = _hyps_all("NOT SUPPORTED")
    assert select_phase21_case(hy, agg2)[0] == "CASE E"
    hy = _hyps_all("NOT SUPPORTED")
    assert select_phase21_case(hy, base)[0] == "CASE F"
    # CASE A needs H2 unsettled + low post-path RC dec
    agg3 = dict(base, locations={"rows": {"relational_output": {"RC_dec": 0.4}}})
    hy = _hyps_all("NOT SUPPORTED")
    assert select_phase21_case(hy, agg3)[0] == "CASE A"
    for c in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F"):
        assert isinstance(recommendation_for_case(c), str) and recommendation_for_case(c)


def test_hypotheses_cover_H1_to_H8():
    agg = {"n_seeds": 3,
           "capacity": {"tested": True, "RC_gains": [0.03, 0.04, 0.05], "R_kept": True},
           "alignment": {"tested": True, "R_dec": 0.85, "RC_dec": 0.5},
           "dominance": {"tested": True, "C_over_R_grad": 4.0, "R_share": 0.04},
           "counterfactual": {"tested": True, "R_correct_change": 0.7,
                              "R_change": 0.8, "control_change": 0.1},
           "heads": {"tested": True, "linear_RC": 0.5, "nonlinear_RC": 0.55, "component_RC": 0.85},
           "locations": {"tested": True, "rows": {
               "relational_output": {"R_dec": 0.85, "RC_dec": 0.7},
               "final_representation": {"R_dec": 0.8, "RC_dec": 0.65}}},
           "order": {"tested": True, "spread_RC": 0.01, "separable": True},
           "intervention": {"tested": False}}
    hyps = build_phase21_hypotheses(agg)
    assert set(hyps) == {f"H{i}" for i in range(1, 9)}
    assert hyps["H1"]["status"] == "SUPPORTED"
    assert hyps["H2"]["status"] == "SUPPORTED"
    assert hyps["H3"]["status"] == "SUPPORTED"
    assert hyps["H4"]["status"] == "SUPPORTED"
    assert hyps["H5"]["status"] == "SUPPORTED"
    assert hyps["H6"]["status"] == "SUPPORTED"
    assert hyps["H8"]["status"] == "NOT TESTED"
    assert all(hyps[f"H{i}"]["status"] in
               ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
               for i in range(1, 9))


def test_failure_rows_have_criteria():
    agg = {"reproduction": {"tested": True, "passed": True, "detail": "ok"},
           "capacity": {"tested": True, "RC_gains": [0.0]},
           "alignment": {"tested": True, "R_dec": 0.9, "RC_dec": 0.5},
           "dominance": {"tested": True, "R_share": 0.2},
           "counterfactual": {"tested": True, "R_change": 0.5},
           "heads": {"tested": True, "linear_RC": 0.4, "nonlinear_RC": 0.4, "component_RC": 0.4},
           "agreement": {"tested": True, "agree_acc": 0.9, "disagree_acc": 0.1}}
    rows = build_failure_diagnosis(agg)
    assert len(rows) == 7
    for row in rows:
        assert isinstance(row["failed"], bool)
        assert row["criterion"] and row["observed"]


def test_probe_joint_on_state_runs():
    from neuroforge.training.phase21_rc_diagnosis import probe_joint_on_state
    ds = _tiny_ds()
    expert = build_joint_variant(2)
    expert.eval()
    st = stage_states(expert, ds.features)["fusion"]
    assert 0.0 <= probe_joint_on_state(st, ds) <= 1.0


def test_mean_helper():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == 1.5
