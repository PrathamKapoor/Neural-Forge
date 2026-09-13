"""Unit tests for Phase 9 Evaluation Metrics and Diagnostics."""
import torch
import pytest

from neuroforge.evaluation.phase9_metrics import (
    calculate_generalization_gap,
    calculate_decision_preservation,
    calculate_single_expert_ceiling,
    diagnose_mixed_structure_failure,
    calculate_mixed_counterfactual_analysis,
)


def test_calculate_generalization_gap():
    """Verify calculation of generalization gap metrics."""
    res = calculate_generalization_gap(
        baseline_acc=0.90,
        eval_acc=0.88,
        baseline_flops=25000.0,
        eval_flops=26000.0,
    )
    assert pytest.approx(res["accuracy_gap"], rel=1e-4) == 0.02
    assert pytest.approx(res["relative_accuracy_drop"], rel=1e-4) == 0.02 / 0.90
    assert pytest.approx(res["flops_shift"], rel=1e-4) == 1000.0
    assert pytest.approx(res["relative_flops_shift"], rel=1e-4) == 1000.0 / 25000.0


def test_calculate_decision_preservation():
    """Verify routing decision preservation rate calculation."""
    clean = ["mlp", "graph", "attention_v2", "mlp"]
    exact = ["mlp", "graph", "attention_v2", "mlp"]
    assert calculate_decision_preservation(clean, exact) == 1.0

    different = ["attention", "attention", "graph", "graph"]
    assert calculate_decision_preservation(clean, different) == 0.0

    partial = ["mlp", "graph", "graph", "attention"]
    assert calculate_decision_preservation(clean, partial) == 0.5


def test_calculate_single_expert_ceiling():
    """Verify determination of single expert performance ceiling."""
    accs = {"mlp": 0.70, "graph": 0.52, "attention_v2": 0.74}
    res = calculate_single_expert_ceiling(accs)
    assert res["best_expert"] == "attention_v2"
    assert pytest.approx(res["ceiling_accuracy"]) == 0.74

    empty_res = calculate_single_expert_ceiling({})
    assert empty_res["best_expert"] is None
    assert empty_res["ceiling_accuracy"] == 0.0


def test_diagnose_mixed_structure_failure():
    """Verify categorization across all defined failure modes and success."""
    expert_accs = {"mlp": 0.72, "graph": 0.50, "attention_v2": 0.65}

    # 1. Success
    succ = diagnose_mixed_structure_failure(0.92, 0.95, expert_accs, desired_threshold=0.85)
    assert "Success" in succ["failure_mode"]

    # 2. Failure B: Router selection failure (ceiling >= 0.85, router < ceiling - 0.08)
    exp_b = {"mlp": 0.92, "graph": 0.50}
    fail_b = diagnose_mixed_structure_failure(0.70, 0.92, exp_b, desired_threshold=0.85)
    assert "Failure B" in fail_b["failure_mode"]

    # 3. Failure A: Expert incapability (ceiling < 0.60)
    exp_a = {"mlp": 0.52, "graph": 0.55}
    fail_a = diagnose_mixed_structure_failure(0.51, 0.55, exp_a, desired_threshold=0.85)
    assert "Failure A" in fail_a["failure_mode"]

    # 4. Failure D: Single-expert compositional limitation (ceiling < desired_threshold)
    fail_d = diagnose_mixed_structure_failure(0.70, 0.72, expert_accs, desired_threshold=0.85)
    assert "Failure D" in fail_d["failure_mode"]


def test_calculate_mixed_counterfactual_analysis():
    """Verify offline counterfactual analysis on candidate expert predictions."""
    targets = torch.tensor([1, 0, 1, 0])
    expert_preds = {
        "mlp": torch.tensor([1, 0, 0, 0]),          # acc: 3/4 = 0.75
        "graph": torch.tensor([0, 1, 1, 0]),        # acc: 2/4 = 0.50
        "attention_v2": torch.tensor([1, 0, 1, 1]), # acc: 3/4 = 0.75
    }
    selected_choices = ["mlp", "mlp", "attention_v2", "graph"]
    selected_preds = torch.tensor([1, 0, 1, 0])      # acc: 4/4 = 1.0

    flops_map = {"mlp": 8928.0, "graph": 8112.0, "attention_v2": 46296.0}

    res = calculate_mixed_counterfactual_analysis(
        selected_choices=selected_choices,
        selected_preds=selected_preds,
        expert_preds=expert_preds,
        targets=targets,
        expert_flops=flops_map,
        router_flops=1052.0,
    )

    assert pytest.approx(res["router_accuracy"]) == 1.0
    assert pytest.approx(res["ceiling_accuracy"]) == 0.75
    assert res["best_single_expert"] in ("mlp", "attention_v2")
    assert "mean_selected_flops" in res
    assert res["mean_selected_flops"] > 0
